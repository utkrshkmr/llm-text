"""Lab 43 reference solution: three decoders, KV-cache sizing, decode roofline, metrics."""
from __future__ import annotations

import time
from typing import Callable, Dict, Iterator, Optional

import torch

from llmsys.flops import HARDWARE
from llmsys.kvcache import KVCache, StaticKVCache


# --------------------------------------------------------------------------- Task 1
@torch.inference_mode()
def generate_nocache(model, token_ids: torch.Tensor, max_new_tokens: int,
                     eos_id: Optional[int] = None) -> Iterator[torch.Tensor]:
    """Greedy decoding that re-runs the full sequence every step. Yields (B, 1) ids."""
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(token_ids)[:, -1]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and bool((nxt == eos_id).all()):
            return
        yield nxt
        token_ids = torch.cat([token_ids, nxt], dim=1)


# --------------------------------------------------------------------------- Task 2
@torch.inference_mode()
def generate_kvcache(model, token_ids: torch.Tensor, max_new_tokens: int,
                     eos_id: Optional[int] = None) -> Iterator[torch.Tensor]:
    """Prefill once, then feed only the newest token (dynamic, growing cache)."""
    model.eval()
    cache = KVCache(model.n_layers)
    logits = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and bool((nxt == eos_id).all()):
            return
        yield nxt
        logits = model(nxt, cache=cache)[:, -1]


# --------------------------------------------------------------------------- Task 3
def _kv_geometry(model):
    att = model.trf_blocks[0].att
    n_kv = getattr(att, "num_kv_groups", None) or att.num_heads
    return n_kv, att.head_dim


@torch.inference_mode()
def generate_static(model, token_ids: torch.Tensor, max_new_tokens: int,
                    eos_id: Optional[int] = None) -> Iterator[torch.Tensor]:
    """Same as Task 2 but with a preallocated StaticKVCache (fixed shapes, no realloc)."""
    model.eval()
    B, T = token_ids.shape
    n_kv, hd = _kv_geometry(model)
    p = next(model.parameters())
    cache = StaticKVCache(model.n_layers, B, n_kv, T + max_new_tokens, hd, p.dtype, p.device)
    logits = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and bool((nxt == eos_id).all()):
            return
        yield nxt
        logits = model(nxt, cache=cache)[:, -1]


# --------------------------------------------------------------------------- Task 4
def kv_bytes_per_token(cfg: dict, bytes_per_elem: float = 2) -> float:
    """K and V, every layer, every KV head: 2 * L * H_kv * d_head * bytes."""
    if "n_kv_groups" in cfg:                      # Qwen3-style config
        n_kv, hd = cfg["n_kv_groups"], cfg["head_dim"]
    else:                                         # GPT-2-style config (MHA)
        n_kv, hd = cfg["n_heads"], cfg["emb_dim"] // cfg["n_heads"]
    return 2 * cfg["n_layers"] * n_kv * hd * bytes_per_elem


def max_tokens_in_budget(cfg: dict, budget_bytes: float, bytes_per_elem: float = 2) -> int:
    """How many cached tokens (summed over all sequences) fit in a KV budget."""
    return int(budget_bytes // kv_bytes_per_token(cfg, bytes_per_elem))


# --------------------------------------------------------------------------- Task 5
def decode_roofline(n_params: float, kv_bytes_tok: float, batch: int, context: int,
                    hw: str = "H100-SXM", bytes_per_param: float = 2) -> Dict[str, float]:
    """Lower bound on one decode step for ``batch`` sequences of length ``context``.

    FLOPs  ~ 2 * N * B                (weight GEMMs; attention ignored here)
    bytes  ~ N * b + B * context * kv (weights read once, every sequence reads its KV)
    """
    h = HARDWARE[hw]
    flops = 2.0 * n_params * batch
    nbytes = n_params * bytes_per_param + batch * context * kv_bytes_tok
    t_c, t_m = flops / h["flops"], nbytes / h["bw"]
    return dict(flops=flops, bytes=nbytes, intensity=flops / nbytes, t_compute=t_c,
                t_memory=t_m, t_step=max(t_c, t_m), tokens_per_s=batch / max(t_c, t_m),
                bound="compute" if t_c > t_m else "memory")


def critical_batch(n_params: float, hw: str = "H100-SXM", bytes_per_param: float = 2) -> float:
    """Batch size where weight-GEMM intensity (= 2B/b FLOP/byte) meets the ridge point."""
    h = HARDWARE[hw]
    return (h["flops"] / h["bw"]) * bytes_per_param / 2.0


# --------------------------------------------------------------------------- Task 6
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_generation(gen_fn: Callable, model, token_ids: torch.Tensor,
                       max_new_tokens: int, clock=time.perf_counter) -> Dict[str, float]:
    """Run a streaming generator and report TTFT, mean TPOT, and decode tokens/s."""
    _sync()
    t0 = clock()
    stamps = []
    for _ in gen_fn(model, token_ids, max_new_tokens):
        _sync()
        stamps.append(clock())
    n = len(stamps)
    ttft = stamps[0] - t0 if n else float("nan")
    tpot = (stamps[-1] - stamps[0]) / (n - 1) if n > 1 else float("nan")
    total = stamps[-1] - t0 if n else float("nan")
    return dict(n_tokens=n, ttft=ttft, tpot=tpot, total=total,
                tokens_per_s=n / total if n else 0.0,
                decode_tokens_per_s=1.0 / tpot if n > 1 and tpot > 0 else float("nan"))
