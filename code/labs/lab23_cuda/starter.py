"""Lab 23 (starter): CUDA kernels from first principles --- the Python side.

Your CUDA kernels live in ``starter_csrc/`` (``softmax.cu``, ``layernorm.cu``, ``rmsnorm_bwd.cu``,
``gemm_ladder.cu``; the host wrappers are complete, every kernel body has a TODO) and are compiled
by :func:`build`; the finished versions are the chapter's sources in ``code/csrc/cuda``. This
module holds everything that is not a kernel: the occupancy calculator, the coalescing and bank-conflict models, the Python
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
KERNEL_DIR = os.path.join(HERE, "starter_csrc")          # your kernels

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
    raise NotImplementedError("TODO Task 1: occupancy")


# =============================================================================== Task 2
def sectors_touched(byte_addresses: Iterable[int], sector_bytes: int = 32) -> int:
    """Number of distinct 32-byte DRAM/L2 sectors a set of byte addresses touches."""
    raise NotImplementedError("TODO Task 2: sectors_touched")


def warp_access(stride_bytes: int, elem_bytes: int = 4, lanes: int = 32) -> Dict[str, float]:
    """Sectors and efficiency when lane ``l`` of a warp reads ``elem_bytes`` at ``l * stride_bytes``.

    Efficiency = bytes the warp asked for / bytes the memory system had to move. Stride 4 with 4-byte
    elements: 128 B in 4 sectors, 100 %. Stride 32 or more: one sector per lane, 12.5 % or less.
    """
    raise NotImplementedError("TODO Task 2: warp_access")


def bank_conflict_degree(word_addresses: Sequence[int], banks: int = 32) -> int:
    """How many wavefronts a warp's 4-byte shared-memory accesses need.

    Word ``w`` lives in bank ``w % banks``. Lanes reading the *same* word are a broadcast and cost
    nothing extra; lanes reading *different* words of the same bank serialise. The degree is the
    largest number of distinct words any bank must serve (1 = conflict-free).
    """
    raise NotImplementedError("TODO Task 2: bank_conflict_degree")


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
    raise NotImplementedError("TODO Task 3: tiled_gemm_sim")


def tile_traffic(M: int, N: int, K: int, bm: int, bn: int, bk: int,
                 elem_bytes: int = 2) -> Dict[str, float]:
    """Global-memory traffic and arithmetic intensity of a (bm x bn x bk)-tiled GEMM.

    Every output tile streams its full A row panel (bm x K) and B column panel (K x bn) once, so
    ``bytes = tiles * (bm + bn) * K * b + M * N * b``; the shared-memory footprint per block is one
    slab of each. Intensity against those loads is ``2 bm bn / ((bm + bn) b)`` FLOP per byte: a
    1x1 tile (rung 0) gives 1/b, a 32x32 fp32 tile 8, a 128x128 bf16 tile 64.
    """
    raise NotImplementedError("TODO Task 3: tile_traffic")


# =============================================================================== Task 4
def merge_maxsum(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Combine two online-softmax states (running max m, running sum l of exp(x - m))."""
    raise NotImplementedError("TODO Task 4: merge_maxsum")


def softmax_online(x: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """Row softmax computed the way ``softmax_fwd_cached`` does: per-chunk (m, l) states merged
    with ``merge_maxsum`` instead of a max pass followed by a sum pass. Matches ``torch.softmax``."""
    raise NotImplementedError("TODO Task 4: softmax_online")


# =============================================================================== Task 5
def norm_bytes(kind: str, rows: int, d: int, elem_bytes: int = 2) -> Dict[str, int]:
    """HBM bytes of the fused normalisation kernels of the chapter, ideal and as implemented.

    ``ideal`` reads every input once and writes every output once. ``ours`` counts what the kernels
    in csrc/cuda actually do: the backward passes read the row inputs twice because dx (one block
    per row) and dweight (a column reduction) are separate kernels.
    """
    raise NotImplementedError("TODO Task 5: norm_bytes")


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
    raise NotImplementedError("TODO Task 5: place_on_roofline")


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
    raise NotImplementedError("TODO Task 6: parse_ncu_csv")


def ncu_verdict(metrics: Dict[str, float], flops: Optional[float] = None,
                dtype: str = "bf16", hw: dict = A100) -> Dict[str, object]:
    """Turn a launch's counters into the three questions of a kernel review.

    1. Is the machine full?    achieved occupancy (warps active / 64).
    2. Which roof is it under? bytes from HBM and time give achieved GB/s; FLOPs (if the caller
                                knows them) give TFLOP/s; the tensor-pipe counter says whether the
                                tensor cores, not the FMA units, did the work.
    3. Is shared memory clean? bank conflicts per shared-memory request.
    """
    raise NotImplementedError("TODO Task 6: ncu_verdict")


# =============================================================================== launch wrappers
def build(verbose: bool = False) -> bool:
    """Compile the kernels in ``starter_csrc/`` into ``torch.ops.llmsys`` (JIT when LLMSYS_JIT=1).

    The schema of ``rmsnorm_backward`` and the RMSNorm forward come from the book's ``csrc`` (they
    are not part of this lab); everything else is yours. Returns False on a machine without a GPU.
    """
    if not torch.cuda.is_available():
        return False
    from llmsys.ops import ensure, has_op
    from llmsys.ops import rmsnorm
    mine = lambda f: os.path.join(KERNEL_DIR, f)  # noqa: E731
    ok = ensure("rmsnorm_backward", ["cpu/rmsnorm_bwd.cpp", mine("rmsnorm_bwd.cu")], verbose)
    ok &= ensure("softmax_fwd", [mine("softmax.cu")], verbose)
    ok &= ensure("layernorm_fwd", [mine("layernorm.cu")], verbose)
    ok &= ensure("gemm", [mine("gemm_ladder.cu")], verbose)
    ok &= rmsnorm.build(verbose)
    return bool(ok) and has_op("rmsnorm_backward")


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
