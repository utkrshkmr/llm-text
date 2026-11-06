"""Lab 21 (reference solution): CUDA kernels from first principles --- the Python side.

The CUDA kernels of this lab are the chapter's sources in ``code/csrc`` (``cuda/softmax.cu``,
``cuda/layernorm.cu``, ``cuda/rmsnorm_bwd.cu`` + ``cpu/rmsnorm_bwd.cpp``, ``cuda/gemm_ladder.cu``);
the starter versions with TODO markers live in ``starter_csrc/``. This module holds everything that
is not a kernel: the occupancy calculator, the coalescing and bank-conflict models, the Python
simulations of the tiled GEMM and of the online softmax, the byte counts and roofline placement
used by ``bench.py``, the Nsight Compute parser, and thin launch wrappers around
``torch.ops.llmsys``.

Every function here runs on a CPU-only machine; only the launch wrappers need a GPU and a build.
"""
from __future__ import annotations

import csv
import io
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
KERNEL_DIR = os.path.join(CODE, "csrc", "cuda")        # the reference kernels

# ------------------------------------------------------------------------------- hardware facts
# A100-40GB PCIe (compute capability 8.0). The SM limits are those of the CUDA occupancy
# calculator; the peaks are NVIDIA's published numbers (dense, no sparsity).
A100 = dict(
    name="A100-40GB PCIe", cc=(8, 0), sms=108, warp_size=32,
    max_threads_per_sm=2048, max_warps_per_sm=64, max_blocks_per_sm=32,
    max_threads_per_block=1024,
    regs_per_sm=65536, max_regs_per_thread=255, reg_alloc_unit=256, warp_alloc_granularity=4,
    smem_per_sm=164 * 1024, max_smem_per_block=163 * 1024, smem_reserved_per_block=1024,
    smem_alloc_granularity=128,
    fp32_lanes_per_sm=64, l2_bytes=40 * 2**20,
    peak=dict(fp32=19.5e12, tf32=156e12, bf16=312e12, fp16=312e12),
    hbm_bytes_per_s=1.555e12,
)


# =============================================================================== Task 1
def occupancy(threads_per_block: int, regs_per_thread: int, smem_per_block: int,
              hw: dict = A100) -> Dict[str, object]:
    """Resident blocks, warps and occupancy per SM for one kernel configuration.

    Three resources are rationed per SM and the tightest one wins:
      * warps:     ``max_warps_per_sm`` warps and ``max_blocks_per_sm`` blocks;
      * registers: a warp is allocated ``ceil(regs * 32 / 256) * 256`` registers; each of the four
        scheduler partitions owns a quarter of the register file, so register-limited warps come in
        multiples of ``warp_alloc_granularity``;
      * shared memory: every block pays ``smem_reserved_per_block`` on top of its own, rounded up to
        the allocation granularity, out of the ``smem_per_sm`` carve-out.
    Returns blocks_per_sm, warps_per_sm, occupancy (fraction of 64 warps) and the limiter.
    """
    ws = hw["warp_size"]
    if not 1 <= threads_per_block <= hw["max_threads_per_block"]:
        raise ValueError(f"threads_per_block must be in [1, {hw['max_threads_per_block']}]")
    if regs_per_thread > hw["max_regs_per_thread"]:
        raise ValueError(f"regs_per_thread exceeds {hw['max_regs_per_thread']}")
    if smem_per_block > hw["max_smem_per_block"]:
        raise ValueError(f"smem_per_block exceeds {hw['max_smem_per_block']} bytes")
    warps_per_block = math.ceil(threads_per_block / ws)
    limits = {}
    limits["warps"] = min(hw["max_blocks_per_sm"], hw["max_warps_per_sm"] // warps_per_block)
    if regs_per_thread > 0:
        regs_per_warp = math.ceil(regs_per_thread * ws / hw["reg_alloc_unit"]) * hw["reg_alloc_unit"]
        partitions = hw["warp_alloc_granularity"]
        warps_by_regs = (hw["regs_per_sm"] // partitions // regs_per_warp) * partitions
        limits["registers"] = warps_by_regs // warps_per_block
    gran = hw["smem_alloc_granularity"]
    smem_block = math.ceil((smem_per_block + hw["smem_reserved_per_block"]) / gran) * gran
    limits["shared memory"] = hw["smem_per_sm"] // smem_block
    limiter = min(limits, key=limits.get)
    blocks = limits[limiter]
    warps = blocks * warps_per_block
    return dict(blocks_per_sm=blocks, warps_per_sm=warps, threads_per_sm=warps * ws,
                occupancy=warps / hw["max_warps_per_sm"], limiter=limiter, limits=limits)


# =============================================================================== Task 2
def sectors_touched(byte_addresses: Iterable[int], sector_bytes: int = 32) -> int:
    """Number of distinct 32-byte DRAM/L2 sectors a set of byte addresses touches."""
    return len({a // sector_bytes for a in byte_addresses})


def warp_access(stride_bytes: int, elem_bytes: int = 4, lanes: int = 32) -> Dict[str, float]:
    """Sectors and efficiency when lane ``l`` of a warp reads ``elem_bytes`` at ``l * stride_bytes``.

    Efficiency = bytes the warp asked for / bytes the memory system had to move. Stride 4 with 4-byte
    elements: 128 B in 4 sectors, 100 %. Stride 32 or more: one sector per lane, 12.5 % or less.
    """
    addrs = [l * stride_bytes for l in range(lanes)]
    sectors = sectors_touched(a + o for a in addrs for o in range(0, elem_bytes, 4))
    useful = lanes * elem_bytes
    return dict(sectors=sectors, bytes_moved=sectors * 32, efficiency=useful / (sectors * 32))


def bank_conflict_degree(word_addresses: Sequence[int], banks: int = 32) -> int:
    """How many wavefronts a warp's 4-byte shared-memory accesses need.

    Word ``w`` lives in bank ``w % banks``. Lanes reading the *same* word are a broadcast and cost
    nothing extra; lanes reading *different* words of the same bank serialise. The degree is the
    largest number of distinct words any bank must serve (1 = conflict-free).
    """
    per_bank: Dict[int, set] = {}
    for w in word_addresses:
        per_bank.setdefault(w % banks, set()).add(w)
    return max((len(s) for s in per_bank.values()), default=1)


def smem_stride_conflicts(stride_words: int, lanes: int = 32, banks: int = 32) -> int:
    """Conflict degree of the pattern ``shared[lane * stride]`` (the kernel in microbench.cu)."""
    return bank_conflict_degree([l * stride_words for l in range(lanes)], banks)


# =============================================================================== Task 3
def tiled_gemm_sim(a: torch.Tensor, b: torch.Tensor, bm: int = 32, bn: int = 32,
                   bk: int = 32) -> torch.Tensor:
    """C = A @ B with the loop structure of ``gemm_tiled_kernel`` (rung 1 of the ladder).

    One iteration of the two outer loops is one thread block; the inner loop walks K in slabs of
    ``bk``. Ragged edges are handled as the kernel does: a slab that runs past M, N or K is padded
    with zeros, and only the in-range part of the tile is stored. fp32 accumulation, output in the
    input dtype.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "inner dimensions differ"
    af, bf = a.float(), b.float()
    c = torch.zeros(M, N, dtype=torch.float32)
    for row0 in range(0, M, bm):                       # blockIdx.y
        for col0 in range(0, N, bn):                   # blockIdx.x
            acc = torch.zeros(bm, bn)                  # one accumulator per thread of the block
            for k0 in range(0, K, bk):                 # the K loop, one shared-memory slab each
                a_s = torch.zeros(bm, bk)              # As[ty][tx], zero outside the matrix
                b_s = torch.zeros(bk, bn)              # Bs[ty][tx]
                am, ak = min(bm, M - row0), min(bk, K - k0)
                bn_, bk_ = min(bn, N - col0), ak
                a_s[:am, :ak] = af[row0:row0 + am, k0:k0 + ak]
                b_s[:bk_, :bn_] = bf[k0:k0 + bk_, col0:col0 + bn_]
                acc += a_s @ b_s                       # the TILE-long FMA loop of every thread
            rm, rn = min(bm, M - row0), min(bn, N - col0)
            c[row0:row0 + rm, col0:col0 + rn] = acc[:rm, :rn]
    return c.to(a.dtype)


def tile_traffic(M: int, N: int, K: int, bm: int, bn: int, bk: int,
                 elem_bytes: int = 2) -> Dict[str, float]:
    """Global-memory traffic and arithmetic intensity of a (bm x bn x bk)-tiled GEMM.

    Every output tile streams its full A row panel (bm x K) and B column panel (K x bn) once, so
    ``bytes = tiles * (bm + bn) * K * b + M * N * b``; the shared-memory footprint per block is one
    slab of each. Intensity against those loads is ``2 bm bn / ((bm + bn) b)`` FLOP per byte: a
    1x1 tile (rung 0) gives 1/b, a 32x32 fp32 tile 8, a 128x128 bf16 tile 64.
    """
    tiles = math.ceil(M / bm) * math.ceil(N / bn)
    loads = tiles * (bm + bn) * K * elem_bytes
    store = M * N * elem_bytes
    flops = 2.0 * M * N * K
    return dict(tiles=tiles, global_bytes=loads + store, flops=flops,
                intensity=flops / (loads + store),
                tile_intensity=2.0 * bm * bn / ((bm + bn) * elem_bytes),
                smem_bytes=(bm * bk + bk * bn) * elem_bytes)


# =============================================================================== Task 4
def merge_maxsum(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Combine two online-softmax states (running max m, running sum l of exp(x - m))."""
    (ma, la), (mb, lb) = a, b
    m = max(ma, mb)
    la = 0.0 if ma == -math.inf else la * math.exp(ma - m)
    lb = 0.0 if mb == -math.inf else lb * math.exp(mb - m)
    return m, la + lb


def softmax_online(x: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """Row softmax computed the way ``softmax_fwd_cached`` does: per-chunk (m, l) states merged
    with ``merge_maxsum`` instead of a max pass followed by a sum pass. Matches ``torch.softmax``."""
    xf = x.float()
    out = torch.empty_like(xf)
    for r in range(xf.shape[0]):
        row = xf[r].tolist()
        state = (-math.inf, 0.0)                       # the empty state: nothing seen yet
        for c0 in range(0, len(row), chunk):           # one "thread's" elements, then the block tree
            part = row[c0:c0 + chunk]
            m = max(part)
            l = sum(math.exp(v - m) for v in part)
            state = merge_maxsum(state, (m, l))
        m, l = state
        out[r] = torch.tensor([math.exp(v - m) / l for v in row])
    return out.to(x.dtype)


# =============================================================================== Task 5
def norm_bytes(kind: str, rows: int, d: int, elem_bytes: int = 2) -> Dict[str, int]:
    """HBM bytes of the fused normalisation kernels of the chapter, ideal and as implemented.

    ``ideal`` reads every input once and writes every output once. ``ours`` counts what the kernels
    in csrc/cuda actually do: the backward passes read the row inputs twice because dx (one block
    per row) and dweight (a column reduction) are separate kernels.
    """
    rd = rows * d * elem_bytes
    table = {
        "softmax_fwd": (2 * rd, 2 * rd),
        "softmax_bwd": (3 * rd, 3 * rd),
        "rmsnorm_fwd": (2 * rd + 4 * d, 2 * rd + 4 * d),
        "rmsnorm_bwd": (3 * rd + 8 * d, 5 * rd + 8 * d + 4 * rows),
        "layernorm_fwd": (2 * rd + 8 * d + 8 * rows, 2 * rd + 8 * d + 8 * rows),
        "layernorm_bwd": (3 * rd + 12 * d + 8 * rows, 5 * rd + 12 * d + 16 * rows),
    }
    if kind not in table:
        raise KeyError(f"unknown kind {kind!r}; one of {sorted(table)}")
    ideal, ours = table[kind]
    return dict(ideal=ideal, ours=ours)


def roofline_time(flops: float, bytes_moved: float, dtype: str = "bf16", hw: dict = A100) -> float:
    """Lower bound on the kernel time: max(compute time, memory time)."""
    return max(flops / hw["peak"][dtype], bytes_moved / hw["hbm_bytes_per_s"])


def place_on_roofline(flops: float, bytes_moved: float, seconds: float, dtype: str = "bf16",
                      hw: dict = A100) -> Dict[str, float]:
    """Achieved rates of a measured kernel as fractions of the two roofs.

    ``bound`` names the roof the kernel's intensity sits under (compute when I > ridge). The
    fraction of *that* roof is what to quote; the other one says how far the kernel is from
    saturating the resource it does not need.
    """
    ridge = hw["peak"][dtype] / hw["hbm_bytes_per_s"]
    intensity = flops / bytes_moved
    tflops = flops / seconds
    gbps = bytes_moved / seconds
    return dict(intensity=intensity, ridge=ridge, bound="compute" if intensity > ridge else "memory",
                tflops=tflops / 1e12, gbps=gbps / 1e9,
                frac_compute=tflops / hw["peak"][dtype], frac_bandwidth=gbps / hw["hbm_bytes_per_s"],
                floor_s=roofline_time(flops, bytes_moved, dtype, hw))


# =============================================================================== Task 6
# The counters we ask Nsight Compute for, and what each one answers.
NCU_METRICS = {
    "gpu__time_duration.sum": "kernel time (ns)",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved occupancy (% of 64 warps)",
    "dram__bytes.sum": "bytes that actually crossed HBM",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "% of HBM bandwidth",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "% of cycles tensor cores busy",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": "% of cycles FMA pipe busy",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": "shared-memory bank conflicts (wavefronts)",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "global load sectors (32 B)",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "global load requests (warp instructions)",
    "launch__registers_per_thread": "registers per thread",
    "launch__occupancy_limit_registers": "blocks/SM allowed by registers",
    "launch__occupancy_limit_shared_mem": "blocks/SM allowed by shared memory",
}


def ncu_command(script: str, metrics: Sequence[str] = tuple(NCU_METRICS), kernel_regex: str = "gemm",
                launch_count: int = 1) -> str:
    """The ``ncu`` command line that produces the CSV ``parse_ncu_csv`` reads."""
    return (f"ncu --csv --metrics {','.join(metrics)} -k regex:{kernel_regex} "
            f"--launch-skip 3 --launch-count {launch_count} python {script}")


def _to_number(s: str) -> Optional[float]:
    s = s.strip().replace(",", "")
    if s in ("", "n/a", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_ncu_csv(text: str) -> List[Dict[str, object]]:
    """Parse ``ncu --csv`` output (one row per kernel and metric) into one dict per kernel launch.

    ``ncu`` prints banner lines (``==PROF==``, ``==WARNING==``) before the header; the header names
    the columns ``"ID", ..., "Kernel Name", ..., "Metric Name", "Metric Unit", "Metric Value"``.
    Numbers may carry thousands separators. Result: ``[{"id": 0, "kernel": name, "block": ...,
    "grid": ..., "metrics": {metric: value}, "units": {metric: unit}}, ...]`` in launch order.
    """
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("==")]
    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    launches: Dict[str, Dict[str, object]] = {}
    for row in reader:
        if "Metric Name" not in row or row.get("Metric Name") is None:
            continue
        key = row.get("ID", "0")
        entry = launches.setdefault(key, dict(id=int(key) if key.isdigit() else key,
                                              kernel=row.get("Kernel Name", ""),
                                              block=row.get("Block Size", ""),
                                              grid=row.get("Grid Size", ""),
                                              metrics={}, units={}))
        name = row["Metric Name"].strip()
        value = _to_number(row.get("Metric Value", ""))
        entry["metrics"][name] = value if value is not None else row.get("Metric Value", "")
        entry["units"][name] = (row.get("Metric Unit") or "").strip()
    return list(launches.values())


def ncu_verdict(metrics: Dict[str, float], flops: Optional[float] = None,
                dtype: str = "bf16", hw: dict = A100) -> Dict[str, object]:
    """Turn a launch's counters into the three questions of a kernel review.

    1. Is the machine full?    achieved occupancy (warps active / 64).
    2. Which roof is it under? bytes from HBM and time give achieved GB/s; FLOPs (if the caller
                                knows them) give TFLOP/s; the tensor-pipe counter says whether the
                                tensor cores, not the FMA units, did the work.
    3. Is shared memory clean? bank conflicts per shared-memory request.
    """
    t = metrics.get("gpu__time_duration.sum")
    t_s = (t or 0.0) * 1e-9
    out: Dict[str, object] = dict(time_us=(t or 0.0) * 1e-3)
    occ = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active")
    if occ is not None:
        out["achieved_occupancy"] = occ / 100.0
    dram = metrics.get("dram__bytes.sum")
    if dram is not None and t_s > 0:
        out["dram_gbps"] = dram / t_s / 1e9
        out["frac_bandwidth"] = dram / t_s / hw["hbm_bytes_per_s"]
    if flops is not None and t_s > 0:
        out["tflops"] = flops / t_s / 1e12
        out["frac_compute"] = flops / t_s / hw["peak"][dtype]
    for k in ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
              "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"):
        if metrics.get(k) is not None:
            out[k.split("__")[1].split(".")[0]] = metrics[k] / 100.0
    conflicts = metrics.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum")
    if conflicts is not None:
        out["bank_conflicts"] = conflicts
    fb, fc = out.get("frac_bandwidth", 0.0), out.get("frac_compute", 0.0)
    if fb > 0.6 and fb >= fc:
        out["verdict"] = "memory-bound"
    elif fc > 0.6:
        out["verdict"] = "compute-bound"
    elif out.get("achieved_occupancy", 1.0) < 0.25:
        out["verdict"] = "latency-bound: too few warps"
    else:
        out["verdict"] = "latency-bound: stalls (check shared memory and dependencies)"
    return out


# =============================================================================== launch wrappers
def build(verbose: bool = False) -> bool:
    """Make every kernel of this lab available in ``torch.ops.llmsys`` (JIT when LLMSYS_JIT=1).

    The reference solution builds the chapter's sources through the operator modules; the starter
    builds ``starter_csrc/`` instead (see ``starter.py``). Returns False on a machine without a GPU.
    """
    if not torch.cuda.is_available():
        return False
    from llmsys.ops import gemm, layernorm, rmsnorm, softmax
    from llmsys.ops import has_op
    ok = all([softmax.build(verbose), layernorm.build(verbose), gemm.build(verbose),
              rmsnorm.build(verbose)])
    return ok and has_op("rmsnorm_backward")


def gemm(a: torch.Tensor, b: torch.Tensor, variant: int) -> torch.Tensor:
    """One rung of the ladder: ``torch.ops.llmsys.gemm`` (fp32 or bf16, fp32 accumulation)."""
    return torch.ops.llmsys.gemm(a, b, variant)


def gemm_variants() -> Tuple[str, ...]:
    from llmsys.ops import gemm as G
    return G.gemm_variants()


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Fused row softmax with autograd (``softmax_fwd`` / ``softmax_bwd``)."""
    from llmsys.ops import softmax as S
    return S.softmax(x, backend="cuda")


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Fused LayerNorm with autograd (``layernorm_fwd`` / ``layernorm_bwd``)."""
    from llmsys.ops import layernorm as L
    return L.layernorm(x, weight, bias, eps, backend="cuda")


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm whose backward runs through the compiled ``rmsnorm_backward`` on x's device."""
    from llmsys.ops import rmsnorm as R
    return R.rmsnorm(x, weight, eps, backend="cuda" if x.is_cuda else "cpp")
