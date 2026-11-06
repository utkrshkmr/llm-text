"""The anatomy of a CPU and on-device runtime in Python (Chapter 47).

Four things every GGML-style engine has, in the smallest form that still runs:

* a weight store: ``PackedLinear`` / ``PackedEmbedding`` keep weights in block formats and multiply
  through the kernels of ``llmsys.ops.gemv_q4``; ``quantize_for_device`` applies a mixed-precision
  ``Layout`` to ``llmsys.qwen3`` or ``llmsys.gpt`` models;
* a file: ``save_packed`` / ``load_packed`` / ``load_quantized`` write an aligned, GGUF-shaped file
  and map it back with no copies, so pages reach RAM only when a kernel touches them;
* a policy: the dequantise-early (prefill) versus dequantise-late (decode) switch, by token count;
* cost models: ``MemorySystem``, ``tokens_per_second``, ``qwen3_param_split``, ``latency_ms`` and
  ``measure_thread_scaling``, which predict and measure what the hardware allows.
"""
from __future__ import annotations

import ctypes
import json
import mmap
import os
import statistics
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from llmsys.ops import gemv_q4 as G

BLOCK_FORMATS = ("q4_0", "q4_g", "q8_0")
RAW_FORMATS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
FORMAT_BITS = {  # storage bits per weight, scales included
    "q4_0": 4 + 16 / 32,                # 16-byte block + fp16 scale per 32 weights
    "q4_g": 4 + (16 + 8) / 128,         # fp16 scale + uint8 zero point per 128 weights
    "q4_k": 144 * 8 / 256,              # 128 B codes, 12 B of 6-bit scales/mins, two fp16, per 256
    "q8_0": 8 + 16 / 32,
    "fp16": 16.0,
    "bf16": 16.0,
    "fp32": 32.0,
}


# ----------------------------------------------------------------------------- the switch
def choose_mode(tokens: int, switch_tokens: Optional[int]) -> str:
    """"early" dequantises the matrix once and runs a dense GEMM; "late" stays in the GEMV kernel."""
    return "early" if switch_tokens is not None and tokens >= switch_tokens else "late"


def crossover_tokens(t_dequant: float, t_dense_per_token: float, t_late_per_token: float) -> float:
    """Tokens T* above which ``t_dequant + T * t_dense`` beats ``T * t_late``.

    Infinite when the kernel that unpacks in place is at least as fast per token as the dense one,
    which is what happens once the dense matrix no longer fits in cache.
    """
    gain = t_late_per_token - t_dense_per_token
    return float("inf") if gain <= 0 else t_dequant / gain


# ----------------------------------------------------------------------------- weight store
class PackedLinear(nn.Module):
    """A Linear whose weight lives in one of the storage formats; y = x W^T + b.

    ``switch_tokens`` is the prefill threshold: a call with at least that many tokens dequantises
    the matrix and runs a dense GEMM, a shorter one calls the in-place kernel. Inference only.
    """

    def __init__(self, in_features: int, out_features: int, fmt: str = "q4_0", group: Optional[int] = None,
                 bias: Optional[torch.Tensor] = None, switch_tokens: Optional[int] = None,
                 backend: Optional[str] = None):
        super().__init__()
        if fmt not in BLOCK_FORMATS and fmt not in RAW_FORMATS:
            raise ValueError(f"unknown format {fmt!r}")
        self.in_features, self.out_features = in_features, out_features
        self.fmt = fmt
        self.group = group or (128 if fmt == "q4_g" else G.BLOCK)
        self.switch_tokens, self.backend = switch_tokens, backend
        self.calls = {"late": 0, "early": 0}
        for name in ("packed", "scales", "zeros"):
            self.register_buffer(name, None)
        self.bias = None if bias is None else nn.Parameter(bias.clone(), requires_grad=False)

    @classmethod
    @torch.no_grad()
    def from_weight(cls, w: torch.Tensor, fmt: str = "q4_0", group: Optional[int] = None,
                    bias: Optional[torch.Tensor] = None, **kw) -> "PackedLinear":
        out_features, in_features = w.shape
        m = cls(in_features, out_features, fmt, group, bias, **kw)
        if fmt == "q4_0":
            m.packed, m.scales = G.quantize_q4_0(w)
        elif fmt == "q4_g":
            m.packed, m.scales, m.zeros = G.quantize_q4_g(w, m.group)
        elif fmt == "q8_0":
            m.packed, m.scales = G.quantize_q8_0(w, m.group)
        else:
            m.packed = w.detach().to(RAW_FORMATS[fmt]).clone()
        return m

    def dequantize(self) -> torch.Tensor:
        """The whole (N, K) weight as fp32: this is what "early" materialises."""
        if self.fmt == "q4_0":
            w = G.dequantize_q4_0(self.packed, self.scales, self.group, backend=self.backend)
        elif self.fmt == "q4_g":
            w = G.dequantize_q4_g(self.packed, self.scales, self.zeros, self.group, backend=self.backend)
        elif self.fmt == "q8_0":
            w = G.dequantize_q8_0_ref(self.packed, self.scales, self.group)
        else:
            w = self.packed.float()
        return w[:, : self.in_features]

    def dequantize_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Selected rows as fp32, touching only those rows' pages (the embedding lookup)."""
        p = self.packed[ids]
        if self.fmt == "q4_0":
            w = G.dequantize_q4_0_ref(p, self.scales[ids], self.group)
        elif self.fmt == "q4_g":
            w = G.dequantize_q4_g_ref(p, self.scales[ids], self.zeros[ids], self.group)
        elif self.fmt == "q8_0":
            w = G.dequantize_q8_0_ref(p, self.scales[ids], self.group)
        else:
            w = p.float()
        return w[:, : self.in_features]

    def _late(self, x: torch.Tensor) -> torch.Tensor:
        if self.fmt == "q4_0":
            return G.gemv_q4_0(x, self.packed, self.scales, self.group, backend=self.backend)
        if self.fmt == "q4_g":
            return G.gemv_q4_g(x, self.packed, self.scales, self.zeros, self.group, backend=self.backend)
        if self.fmt == "q8_0":
            return G.gemv_q8_0(x, self.packed, self.scales, self.group, backend=self.backend)
        return (x.float() @ self.packed.float().T).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.numel() // max(self.in_features, 1)
        mode = "late" if self.fmt in RAW_FORMATS else choose_mode(tokens, self.switch_tokens)
        self.calls[mode] += 1
        y = (x.float() @ self.dequantize().T).to(x.dtype) if mode == "early" else self._late(x)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def storage_bytes(self) -> int:
        tensors = [t for t in (self.packed, self.scales, self.zeros, self.bias) if t is not None]
        return sum(t.numel() * t.element_size() for t in tensors)

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, fmt={self.fmt}, group={self.group}"


class PackedEmbedding(nn.Module):
    """Token embedding that reads rows of a ``PackedLinear``.

    For a tied model the table is the LM head itself (one copy on disk and in the page cache). An
    embedding lookup touches B*T rows of the table; the head touches all of them.
    """

    def __init__(self, table: PackedLinear, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.table = table
        self.dtype = dtype

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        w = self.table.dequantize_rows(ids.reshape(-1))
        return w.reshape(*ids.shape, -1).to(self.dtype)


# ----------------------------------------------------------------------------- mixed layouts
@dataclass
class Layout:
    """Which format each tensor class gets. The head and embedding default to 8 bits: an error in
    the head is an error in every logit, and it is the tensor with the fewest bytes to save per
    unit of quality lost when the vocabulary is small and the most when it is large."""
    blocks: str = "q4_0"
    attention: Optional[str] = None     # None: same as ``blocks``
    head: str = "q8_0"
    group: Optional[int] = None         # group for q4_g; q4_0 and q8_0 always use 32
    switch_tokens: Optional[int] = None
    backend: Optional[str] = None
    overrides: Dict[str, str] = field(default_factory=dict)   # module-name substring -> format

    def format_for(self, name: str) -> str:
        for key, fmt in self.overrides.items():
            if key in name:
                return fmt
        if "out_head" in name or "tok_emb" in name:
            return self.head
        if ".att." in name and self.attention:
            return self.attention
        return self.blocks


def _module_dtype(model: nn.Module) -> torch.dtype:
    return next(p.dtype for p in model.parameters() if p.is_floating_point())


@torch.no_grad()
def quantize_for_device(model: nn.Module, layout: Optional[Layout] = None) -> nn.Module:
    """Swap Linear and token-embedding modules for packed ones, in place; returns the model.

    A model built under ``torch.device("meta")`` yields a skeleton whose buffers are filled by
    ``load_quantized``: nothing is quantised and no weight memory is allocated.
    """
    layout = layout or Layout()
    dtype = _module_dtype(model)
    head, emb = getattr(model, "out_head", None), getattr(model, "tok_emb", None)
    tied = head is not None and emb is not None and head.weight is emb.weight
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear):
                setattr(module, child_name, _pack_linear(child, layout.format_for(full), layout))
    if emb is not None:
        table = model.out_head if tied else _pack_weight(emb.weight, layout.head, layout)
        model.tok_emb = PackedEmbedding(table, dtype)
    return model.eval()


def _pack_weight(w: torch.Tensor, fmt: str, layout: Layout, bias=None) -> PackedLinear:
    kw = dict(group=layout.group if fmt == "q4_g" else None, bias=bias,
              switch_tokens=layout.switch_tokens, backend=layout.backend)
    if w.is_meta:
        m = PackedLinear(w.shape[1], w.shape[0], fmt, kw["group"], None, kw["switch_tokens"], kw["backend"])
        if bias is not None:
            m.bias = nn.Parameter(torch.empty_like(bias), requires_grad=False)
        return m
    return PackedLinear.from_weight(w.float(), fmt, **kw)


def _pack_linear(linear: nn.Linear, fmt: str, layout: Layout) -> PackedLinear:
    bias = None if linear.bias is None else linear.bias.data
    return _pack_weight(linear.weight, fmt, layout, bias)


def set_switch_tokens(model: nn.Module, switch_tokens: Optional[int]) -> None:
    for m in model.modules():
        if isinstance(m, PackedLinear):
            m.switch_tokens = switch_tokens


def model_bytes(model: nn.Module) -> Dict[str, int]:
    """Bytes per tensor class, with shared tables counted once (a tied head is one tensor)."""
    seen, out = set(), {"head": 0, "attention": 0, "blocks": 0, "other": 0}
    for name, m in model.named_modules():
        if isinstance(m, PackedLinear) and id(m) not in seen:
            seen.add(id(m))
            key = "head" if "out_head" in name or "tok_emb" in name else (
                "attention" if ".att." in name else "blocks")
            out[key] += m.storage_bytes()
    for name, p in model.named_parameters():
        if ".bias" not in name:
            out["other"] += p.numel() * p.element_size()
    return out


# ----------------------------------------------------------------------------- k-quant nesting
def quantize_q4_k(w: torch.Tensor, super_block: int = 256, sub_block: int = 32
                  ) -> Tuple[torch.Tensor, int]:
    """Q4_K-style nested scales; returns (dequantised weight, storage bytes).

    Each sub-block has its own minimum and scale, stored as 6-bit codes relative to two fp16 values
    per super-block: 128 + 12 + 4 = 144 bytes per 256 weights, the same 4.5 bits per weight as Q4_0
    but with a finer grid (per-32 scales *and* minimums) paid for by coarser scale precision.
    """
    n, k = w.shape
    wp = G.pad_to_group(w.float(), super_block)
    blocks = wp.reshape(n, -1, super_block // sub_block, sub_block)
    lo, hi = blocks.amin(-1).clamp(max=0.0), blocks.amax(-1).clamp(min=0.0)
    scale, mins = (hi - lo) / 15.0, -lo
    d = scale.amax(-1, keepdim=True).clamp(min=1e-12) / 63.0
    dmin = mins.amax(-1, keepdim=True).clamp(min=1e-12) / 63.0
    s6 = torch.ceil(scale / d).clamp(0, 63) * d           # round up: the top code must stay in range
    m6 = torch.round(mins / dmin).clamp(0, 63) * dmin
    safe = torch.where(s6 == 0, torch.ones_like(s6), s6)
    q = torch.clamp(torch.round((blocks + m6[..., None]) / safe[..., None]), 0, 15)
    w_hat = (q * s6[..., None] - m6[..., None]).reshape(n, -1)[:, :k]
    n_super = wp.numel() // super_block
    return w_hat, n_super * 144


# ----------------------------------------------------------------------------- the file
MAGIC = b"LLMQ"
_ALIGN = 32
_DTYPES = {"uint8": torch.uint8, "int8": torch.int8, "float16": torch.float16,
           "bfloat16": torch.bfloat16, "float32": torch.float32}


def save_packed(model: nn.Module, path: str, metadata: Optional[dict] = None) -> int:
    """Write ``magic, u32 version, u64 header_len, JSON header, aligned tensor data``.

    Same shape as GGUF: self-describing header, every tensor at a 32-byte aligned offset so a mapped
    file can be read by SIMD kernels without copying. Tensors with the same storage (a tied head)
    share one offset. Returns the file size.
    """
    tensors: List[Tuple[str, torch.Tensor]] = list(model.state_dict().items())
    known = {n for n, _ in tensors}
    tensors += [(n, b) for n, b in model.named_buffers() if n not in known and b is not None]
    entries: Dict[str, dict] = {}
    placed: Dict[Tuple[int, tuple, str], int] = {}
    payload: List[Tuple[int, torch.Tensor]] = []
    offset = 0
    for name, t in tensors:
        dt = str(t.dtype).replace("torch.", "")
        key = (t.data_ptr(), tuple(t.shape), dt)
        if key not in placed:
            offset += -offset % _ALIGN
            placed[key] = offset
            payload.append((offset, t))
            offset += t.numel() * t.element_size()
        entries[name] = dict(dtype=dt, shape=list(t.shape), offset=placed[key])
    header = json.dumps(dict(tensors=entries, metadata=metadata or {}, data_bytes=offset)).encode()
    start = len(MAGIC) + 4 + 8 + len(header)
    start += -start % _ALIGN
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<IQ", 1, len(header)) + header)
        f.write(b"\0" * (start - f.tell()))
        for off, t in payload:
            f.seek(start + off)
            f.write(_raw_bytes(t))
    return os.path.getsize(path)


def _raw_bytes(t: torch.Tensor) -> bytes:
    # Straight from the tensor's memory, because numpy has no bf16.
    c = t.detach().cpu().contiguous()
    return ctypes.string_at(c.data_ptr(), c.numel() * c.element_size())


def load_packed(path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Map the file and return zero-copy tensors plus metadata; no weight page is touched here.

    ``mode="c"`` is MAP_PRIVATE: read-only use shares the page cache with every other process that
    maps the file, and an accidental write copies one page instead of corrupting the file.
    """
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path} is not a packed model file")
        _version, hlen = struct.unpack("<IQ", f.read(12))
        header = json.loads(f.read(hlen))
    start = 4 + 12 + hlen
    start += -start % _ALIGN
    mm = np.memmap(path, dtype=np.uint8, mode="c")
    out = {}
    for name, e in header["tensors"].items():
        dt = _DTYPES[e["dtype"]]
        count = int(np.prod(e["shape"], dtype=np.int64))
        t = torch.frombuffer(mm, dtype=dt, count=count, offset=start + e["offset"])
        out[name] = t.view(e["shape"])
    return out, header["metadata"]


def load_quantized(path: str, make_model: Callable[[], nn.Module], layout: Layout) -> nn.Module:
    """Cold-start path: skeleton on the meta device, then every tensor bound to the mapped file."""
    with torch.device("meta"):
        skeleton = make_model()
    quantize_for_device(skeleton, layout)
    tensors, _ = load_packed(path)
    for name, t in tensors.items():
        *parents, leaf = name.split(".")
        owner = skeleton
        for p in parents:
            owner = getattr(owner, p)
        if leaf in owner._parameters:
            owner._parameters[leaf] = nn.Parameter(t, requires_grad=False)
        else:
            owner._buffers[leaf] = t
    return skeleton.eval()


def drop_page_cache(path: str) -> None:
    """Ask the kernel to forget a clean file's pages, so the next read is a real cold start."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def page_cache_fraction(path: str) -> float:
    """Fraction of a file's pages in the page cache right now (Linux ``mincore``)."""
    size = os.path.getsize(path)
    if size == 0:
        return 0.0
    page = mmap.PAGESIZE
    n_pages = (size + page - 1) // page
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            vec = (ctypes.c_ubyte * n_pages)()
            if libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return sum(v & 1 for v in vec) / n_pages
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def resident_bytes() -> int:
    """This process's resident set, from /proc (file-backed pages count once they are faulted in)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * mmap.PAGESIZE


# ----------------------------------------------------------------------------- rooflines
@dataclass(frozen=True)
class MemorySystem:
    """Peak bandwidth from the interface arithmetic: channels * bits/8 * MT/s."""
    name: str
    channels: int
    channel_bits: int
    mts: int
    source: str = "vendor specification"

    @property
    def peak_gbs(self) -> float:
        return self.channels * self.channel_bits / 8 * self.mts / 1000


# Peak numbers: JEDEC data rate times bus width as listed in each vendor's specification. Sustained
# read rates are lower, typically 60 to 85 percent of these on CPUs (measure with ``bench.py``).
MEMORY_SYSTEMS: Tuple[MemorySystem, ...] = (
    MemorySystem("Raspberry Pi 5 (LPDDR4X-4267, 32 bit)", 2, 16, 4267),
    MemorySystem("Apple A17 Pro (LPDDR5-6400, 64 bit)", 4, 16, 6400),
    MemorySystem("Snapdragon 8 Gen 2 (LPDDR5X-8400, 64 bit)", 4, 16, 8400),
    MemorySystem("Snapdragon 8 Gen 3 (LPDDR5X-9600, 64 bit)", 4, 16, 9600),
    MemorySystem("Desktop (DDR5-5600, 2 channels)", 2, 64, 5600),
    MemorySystem("Apple M4 (LPDDR5X-7500, 128 bit)", 8, 16, 7500),
    MemorySystem("Apple M4 Max (LPDDR5X-8533, 512 bit)", 32, 16, 8533),
    MemorySystem("Xeon Gold 6330, one socket (DDR4-2933, 8 ch)", 8, 64, 2933),
    MemorySystem("Xeon Gold 6330, two sockets (DDR4-2933, 16 ch)", 16, 64, 2933),
)
HBM_SYSTEMS = {"A100-40GB (HBM2)": 1555.0, "H100-SXM (HBM3)": 3350.0}


def tokens_per_second(params: float, bits: float, bw_gbs: float) -> float:
    """Decode upper bound: every weight byte crosses the memory bus once per token."""
    return bw_gbs * 1e9 / (params * bits / 8)


def token_rate_table(params: Sequence[float] = (1e9, 3e9, 7e9), bits: Sequence[float] = (4, 8, 16),
                     bandwidths: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    """tokens/s bounds for each memory system and (params, bits) pair."""
    if bandwidths is None:
        bandwidths = {m.name: m.peak_gbs for m in MEMORY_SYSTEMS} | HBM_SYSTEMS
    return {name: {f"{p / 1e9:g}B@{b:g}": tokens_per_second(p, b, bw) for p in params for b in bits}
            for name, bw in bandwidths.items()}


@dataclass(frozen=True)
class CoreModel:
    """Compute side of the roofline: each FMA lane consumes one weight."""
    simd_bits: int = 512
    fma_ports: int = 2
    ghz: float = 2.6
    decode_cycles_per_block: float = 0.0     # extra cycles to unpack one 32-weight block

    def weights_per_second(self) -> float:
        lanes = self.simd_bits // 32 * self.fma_ports
        cycles_per_weight = 1 / lanes + self.decode_cycles_per_block / 32
        return self.ghz * 1e9 / cycles_per_weight


def roofline_gbs(threads: int, per_thread_gbs: float, memory_gbs: float) -> float:
    """Achievable weight bytes per second: the smaller of what the cores can consume and DRAM."""
    return min(threads * per_thread_gbs, memory_gbs)


def saturation_threads(per_thread_gbs: float, memory_gbs: float) -> float:
    """Threads beyond which the memory bus, not the core count, sets the token rate."""
    return memory_gbs / per_thread_gbs


# ----------------------------------------------------------------------------- latency model
def qwen3_param_split(cfg: dict) -> Dict[str, int]:
    """Parameters by what they cost per token: embedding row, head, one block."""
    d, h, hd = cfg["emb_dim"], cfg["hidden_dim"], cfg["head_dim"]
    q_out = cfg["n_heads"] * hd
    kv_out = cfg["n_kv_groups"] * hd
    attention = d * q_out + 2 * d * kv_out + q_out * d
    mlp = 2 * d * h + h * d
    norms = 2 * d + (2 * hd if cfg.get("qk_norm") else 0)
    v_d = cfg["vocab_size"] * d
    return dict(embedding=v_d, head=0 if cfg.get("tie_embeddings") else v_d, embedding_row=d,
                head_read=v_d, attention=attention * cfg["n_layers"], mlp=mlp * cfg["n_layers"],
                norms=norms * cfg["n_layers"] + d)


def bytes_per_token(cfg: dict, bits_blocks: float = 4.5, bits_head: float = 8.5, kv_tokens: int = 0,
                    bits_kv: float = 16.0) -> Dict[str, float]:
    """Bytes read from DRAM to produce one token, by source.

    The head is read in full every token whatever the tie; the embedding costs one row. Each KV
    token is read once per group and layer for K (scores) and once for V (context).
    """
    s = qwen3_param_split(cfg)
    kv_row = cfg["n_kv_groups"] * cfg["head_dim"] * 2 * bits_kv / 8
    return dict(embedding=s["embedding_row"] * bits_head / 8, head=s["head_read"] * bits_head / 8,
                attention=s["attention"] * bits_blocks / 8, mlp=s["mlp"] * bits_blocks / 8,
                kv=kv_tokens * kv_row * cfg["n_layers"])


def latency_ms(cfg: dict, bw_gbs: float, **kw) -> Dict[str, float]:
    """Milliseconds per token at ``bw_gbs``, by source, with the total under ``total``."""
    b = bytes_per_token(cfg, **kw)
    ms = {k: v / (bw_gbs * 1e9) * 1e3 for k, v in b.items()}
    ms["total"] = sum(ms.values())
    return ms


def lm_head_ms(vocab: int, d: int, bits: float, bw_gbs: float) -> float:
    return vocab * d * bits / 8 / (bw_gbs * 1e9) * 1e3


# ----------------------------------------------------------------------------- measurement
def time_samples(fn: Callable[[], object], reps: int = 20, warmup: int = 2) -> Dict[str, float]:
    """Median, p10 and p90 of ``fn`` in seconds after a warm-up."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return dict(median=statistics.median(samples), p10=samples[int(0.1 * (reps - 1))],
                p90=samples[int(0.9 * (reps - 1))], reps=reps)


def measure_thread_scaling(fn: Callable[[], object], threads: Iterable[int], reps: int = 20,
                           warmup: int = 2) -> List[Dict[str, float]]:
    """Time ``fn`` at each thread count; speedup and efficiency are relative to the first count."""
    saved = torch.get_num_threads()
    rows: List[Dict[str, float]] = []
    try:
        for n in threads:
            torch.set_num_threads(n)
            r = time_samples(fn, reps, warmup)
            base = rows[0] if rows else None
            t_base = base["median"] * base["threads"] if base else r["median"] * n
            speedup = t_base / r["median"]
            rows.append(dict(r, threads=n, speedup=speedup, efficiency=speedup / n))
    finally:
        torch.set_num_threads(saved)
    return rows
