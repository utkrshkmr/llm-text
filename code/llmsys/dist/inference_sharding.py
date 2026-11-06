"""Sharding for inference: cost models and a sharded decode step (Chapter 24).

Training shards to fit and to use more FLOP/s; inference shards for two different reasons that
pull in different directions. *Prefill* is a compute-bound pass over thousands of tokens, and
tensor parallelism amortises its collectives over all of them. *Decode* produces one token per
sequence per step: the matmuls are bandwidth-bound reads of the weights, and the two all-reduces
per layer carry only ``batch x d_model`` elements, so their alpha term dominates. The functions
here put numbers on that:

    decode_step_time      weight-read, KV-read and all-reduce time of one decode step at TP = p
    prefill_time          compute and communication time of a prefill at TP = p
    mlp_layout_bytes      bytes per device per layer of the 1-D (Megatron) and 2-D
                          weight-stationary layouts of Pope et al. (2023), derived with the
                          matmul rules of ``llmsys.dist.sharding``
    kv_shard_plan         KV cache sharded by head (GQA permitting) or by batch
    kv_transfer_time      moving a prompt's KV cache from a prefill to a decode pool

``TPDecodeLayer`` is a tiny tensor-parallel transformer layer with a head-sharded KV cache that
runs a decode step on gloo or NCCL and matches the single-process layer bit for bit (up to
floating-point summation order), the smallest complete example of inference sharding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .sharding import (Link, Mesh, ShardSpec, collective_time, bytes_per_rank, mlp_program,
                       PCIE_GEN4)


# =========================================================================== model shapes
@dataclass(frozen=True)
class ModelShape:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int
    vocab: int
    dtype_bytes: int = 2
    gated_mlp: bool = True          # SwiGLU: three matrices in the MLP

    @property
    def attn_params(self) -> int:
        q_o = 2 * self.d_model * self.n_heads * self.d_head
        kv = 2 * self.d_model * self.n_kv_heads * self.d_head
        return q_o + kv

    @property
    def mlp_params(self) -> int:
        return (3 if self.gated_mlp else 2) * self.d_model * self.d_ff

    @property
    def block_params(self) -> int:
        return self.attn_params + self.mlp_params

    @property
    def params(self) -> int:
        """Non-embedding parameters (the ones every decode step reads)."""
        return self.n_layers * self.block_params

    @property
    def embedding_params(self) -> int:
        return 2 * self.vocab * self.d_model

    def weight_bytes(self, with_head: bool = True) -> int:
        return (self.params + (self.vocab * self.d_model if with_head else 0)) * self.dtype_bytes

    def kv_bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.d_head * self.dtype_bytes


LLAMA_7B = ModelShape(n_layers=32, d_model=4096, n_heads=32, n_kv_heads=32, d_head=128, d_ff=11008,
                      vocab=32000)
LLAMA_70B = ModelShape(n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8, d_head=128, d_ff=28672,
                       vocab=32000)
QWEN3_8B = ModelShape(n_layers=36, d_model=4096, n_heads=32, n_kv_heads=8, d_head=128, d_ff=12288,
                      vocab=151936)

A100_HBM = 1.555e12          # bytes/s
A100_BF16 = 312e12           # FLOP/s dense
H100_HBM = 3.35e12
H100_BF16 = 989e12


# =========================================================================== decode and prefill
def decode_step_time(shape: ModelShape, batch: int, tp: int, link: Link = PCIE_GEN4,
                     hbm_bw: float = A100_HBM, flops: float = A100_BF16, context: int = 0,
                     sequence_parallel: bool = False) -> Dict[str, float]:
    """Seconds for one decode step with Megatron TP over ``tp`` devices.

    weights:  each device reads its 1/tp of the weights once per step (bandwidth-bound)
    kv:       each device reads its 1/tp of the KV cache of ``batch`` sequences at ``context``
    compute:  2 * params * batch FLOPs spread over tp devices (tiny at small batch)
    comm:     two all-reduces per layer of the (batch, d_model) activation, each
              2(tp-1) alpha + 2 (tp-1)/tp * bytes / beta on a ring; with sequence parallelism
              the same bytes travel as reduce-scatter + all-gather (same time in this model)
    The step time is the max of the memory and compute terms plus the (exposed) communication.
    """
    w = shape.weight_bytes() / tp / hbm_bw
    kv = batch * context * shape.kv_bytes_per_token() / tp / hbm_bw
    comp = 2 * shape.params * batch / tp / flops
    act_bytes = batch * shape.d_model * shape.dtype_bytes
    comm = 2 * shape.n_layers * collective_time("all_reduce", act_bytes, tp, link)
    if sequence_parallel:
        comm = 2 * shape.n_layers * (collective_time("reduce_scatter", act_bytes, tp, link)
                                     + collective_time("all_gather", act_bytes, tp, link))
    return {"weights": w, "kv": kv, "compute": comp, "comm": comm,
            "total": max(w + kv, comp) + comm, "comm_bytes_per_device":
            2 * shape.n_layers * bytes_per_rank("all_reduce", act_bytes, tp)}


def prefill_time(shape: ModelShape, tokens: int, tp: int, link: Link = PCIE_GEN4,
                 flops: float = A100_BF16, mfu: float = 0.6) -> Dict[str, float]:
    """Seconds for a prefill of ``tokens`` tokens of one sequence with Megatron TP.

    compute: 2 * params * tokens (projections and MLPs) plus the causal attention
             4 * n_layers * tokens^2 * n_heads * d_head / 2 FLOPs, over tp devices at ``mfu``
    comm:    two all-reduces per layer of the (tokens, d_model) activation
    """
    matmul = 2 * shape.params * tokens
    attn = 2 * shape.n_layers * tokens * tokens * shape.n_heads * shape.d_head   # QK^T and PV, causal
    comp = (matmul + attn) / tp / (flops * mfu)
    act_bytes = tokens * shape.d_model * shape.dtype_bytes
    comm = 2 * shape.n_layers * collective_time("all_reduce", act_bytes, tp, link)
    return {"compute": comp, "comm": comm, "total": comp + comm,
            "comm_bytes_per_device": 2 * shape.n_layers * bytes_per_rank("all_reduce", act_bytes, tp)}


def tp_break_even_batch(shape: ModelShape, tp: int, link: Link = PCIE_GEN4,
                        hbm_bw: float = A100_HBM) -> float:
    """Batch at which the per-step all-reduce time equals the weight-read time saved by TP.

    Saved: W/hbm * (1 - 1/tp). Spent: 2L * (2(tp-1) alpha + 2(tp-1)/tp * B d 2 / beta).
    Solving for B gives the batch above which TP stops paying for itself (latency-wise).
    """
    saved = shape.weight_bytes() / hbm_bw * (1 - 1 / tp)
    per_layer_alpha = 2 * (tp - 1) * link.alpha
    per_byte = 2 * (tp - 1) / tp * shape.d_model * shape.dtype_bytes / link.bandwidth
    b = (saved / (2 * shape.n_layers) - per_layer_alpha) / per_byte
    return max(b, 0.0)


# =========================================================================== 1-D vs 2-D layouts
def mlp_layout_bytes(shape: ModelShape, batch: int, layout: Tuple[int, ...]) -> Dict[str, object]:
    """Bytes per device per layer that the MLP's activations move under a weight-stationary layout.

    ``layout=(p,)``       1-D: W1 split on d_ff, W2 split on d_ff (Megatron column/row); X is
                          replicated and the output partial sum is all-reduced.
    ``layout=(px, py)``   2-D: W1 is [x, y] (d_model over x, d_ff over y), W2 is [y, x]; X enters
                          sharded on d_model over x. The first product is a partial sum over x
                          that is reduce-scattered along the batch, the activation is local, the
                          second product needs the batch gathered back (a case-IV conflict on x)
                          and ends in an all-reduce over y. Each collective now moves a tensor that
                          the other axis already shrank.
    Both are derived by ``mlp_program``; the counts are bytes each device sends.
    """
    S = ShardSpec.parse
    if len(layout) == 1:
        mesh = Mesh(("t",), tuple(layout))
        steps = mlp_program(S("-,-"), S("-,t"), S("t,-"), mesh, batch, shape.d_model, shape.d_ff,
                            shape.dtype_bytes)
    elif len(layout) == 2:
        mesh = Mesh(("x", "y"), tuple(layout))
        steps = mlp_program(S("-,x"), S("x,y"), S("y,x"), mesh, batch, shape.d_model, shape.d_ff,
                            shape.dtype_bytes, finish1="reduce_scatter:0", finish2="all_reduce")
    else:
        raise ValueError("layout is (p,) or (px, py)")
    cols = [c for s in steps for c in s.collectives() if c.kind != "slice"]
    return {"collectives": [f"{c.kind}@{c.axis}" for c in cols],
            "bytes_per_device": sum(c.bytes_per_rank(mesh) for c in cols),
            "messages": sum(2 * (mesh.size(c.axis) - 1) if c.kind == "all_reduce"
                            else mesh.size(c.axis) - 1 for c in cols)}


def best_mlp_layout(shape: ModelShape, batch: int, n: int) -> Tuple[Tuple[int, ...], float]:
    """The layout (1-D or any 2-D factorisation of ``n``) that moves the fewest bytes."""
    best = ((n,), mlp_layout_bytes(shape, batch, (n,))["bytes_per_device"])
    for px in range(2, n):
        if n % px == 0:
            b = mlp_layout_bytes(shape, batch, (px, n // px))["bytes_per_device"]
            if b < best[1]:
                best = ((px, n // px), b)
    return best


# =========================================================================== KV cache sharding
def kv_shard_plan(shape: ModelShape, tp: int, batch: int, context: int) -> Dict[str, Dict[str, object]]:
    """KV cache per device and the extra traffic of sharding the cache by head or by batch.

    by head   each device keeps n_kv_heads / tp KV heads for every sequence. Attention is local;
              the only collectives are the layer's two all-reduces. Needs tp | n_kv_heads; with
              tp > n_kv_heads the KV heads are replicated tp / n_kv_heads times (wasted memory).
    by batch  each device keeps all heads for batch / tp sequences. Q, K, V come out of the
              column-parallel projections sharded by head, so an all-to-all (head -> batch) is
              needed before attention and another (batch -> head) after it, per layer.
    """
    per_token = shape.kv_bytes_per_token()
    total = batch * context * per_token
    out: Dict[str, Dict[str, object]] = {}
    heads_per = math.ceil(shape.n_kv_heads / tp)
    replication = heads_per * tp / shape.n_kv_heads
    out["head"] = {"feasible": shape.n_kv_heads % tp == 0 or tp % shape.n_kv_heads == 0,
                   "kv_heads_per_device": heads_per, "replication": replication,
                   "kv_bytes_per_device": total / tp * replication,
                   "extra_bytes_per_layer": 0.0}
    # all-to-all of the q/k/v for one token step: (batch, n_heads + 2 n_kv_heads, d_head) in and
    # the attention output (batch, n_heads, d_head) back; each device sends (tp-1)/tp of its block
    qkv = batch * (shape.n_heads + 2 * shape.n_kv_heads) * shape.d_head * shape.dtype_bytes / tp
    o = batch * shape.n_heads * shape.d_head * shape.dtype_bytes / tp
    out["batch"] = {"feasible": batch % tp == 0, "kv_heads_per_device": shape.n_kv_heads,
                    "replication": 1.0, "kv_bytes_per_device": total / tp,
                    "extra_bytes_per_layer": bytes_per_rank("all_to_all", qkv, tp)
                    + bytes_per_rank("all_to_all", o, tp)}
    return out


# =========================================================================== disaggregation
def kv_transfer_time(shape: ModelShape, prompt_tokens: int, link: Link = PCIE_GEN4,
                     layerwise: bool = True) -> Dict[str, float]:
    """Moving a prompt's KV cache from the prefill pool to the decode pool.

    ``layerwise``: the cache of layer l is sent while layers l+1.. are still computing, so only the
    last layer's slice is exposed; otherwise the whole cache moves after the prefill.
    """
    nbytes = prompt_tokens * shape.kv_bytes_per_token()
    full = link.alpha + nbytes / link.bandwidth
    per_layer = link.alpha + nbytes / shape.n_layers / link.bandwidth
    return {"bytes": nbytes, "time": full, "exposed": per_layer if layerwise else full}


def disaggregation_check(shape: ModelShape, prompt_tokens: int, link: Link = PCIE_GEN4,
                         tp: int = 1, flops: float = A100_BF16) -> Dict[str, float]:
    """Is the KV transfer hidden behind the prefill that produces it?"""
    pf = prefill_time(shape, prompt_tokens, tp, link, flops)
    tr = kv_transfer_time(shape, prompt_tokens, link)
    return {"prefill": pf["total"], "transfer": tr["time"], "exposed": tr["exposed"],
            "ratio": tr["time"] / pf["total"]}


# =========================================================================== a sharded decode step
class TPDecodeLayer:
    """One pre-norm-free transformer layer (attention + GELU MLP), tensor parallel by heads.

    Weights are given in full on every rank and sliced here: Wq/Wk/Wv column-parallel by head,
    Wo row-parallel, W1 column-parallel, W2 row-parallel. The KV cache is sharded by head, so a
    decode step issues exactly two all-reduces of the (batch, d_model) activation, like Megatron.
    GQA is handled by giving each rank whole groups of query heads with their KV head.
    """

    def __init__(self, shape: ModelShape, weights: Dict[str, torch.Tensor], group=None):
        self.s, self.group = shape, group
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        if shape.n_kv_heads % world or shape.n_heads % world or shape.d_ff % world:
            raise ValueError("heads, KV heads and d_ff must divide the TP degree")
        self.world = world
        hq, hkv, d = shape.n_heads // world, shape.n_kv_heads // world, shape.d_head
        self.hq, self.hkv = hq, hkv
        q_cols = slice(rank * hq * d, (rank + 1) * hq * d)
        kv_cols = slice(rank * hkv * d, (rank + 1) * hkv * d)
        f_cols = slice(rank * shape.d_ff // world, (rank + 1) * shape.d_ff // world)
        self.wq, self.wk, self.wv = weights["wq"][:, q_cols], weights["wk"][:, kv_cols], weights["wv"][:, kv_cols]
        self.wo = weights["wo"][q_cols, :]
        self.w1, self.w2 = weights["w1"][:, f_cols], weights["w2"][f_cols, :]
        self.k_cache: List[torch.Tensor] = []            # per step: (batch, hkv, d)
        self.v_cache: List[torch.Tensor] = []

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (batch, d_model), replicated; returns the layer output, replicated."""
        b, d = x.shape[0], self.s.d_head
        q = (x @ self.wq).view(b, self.hq, d)
        self.k_cache.append((x @ self.wk).view(b, self.hkv, d))
        self.v_cache.append((x @ self.wv).view(b, self.hkv, d))
        k = torch.stack(self.k_cache, dim=1)              # (b, T, hkv, d)
        v = torch.stack(self.v_cache, dim=1)
        rep = self.hq // self.hkv                         # GQA: query heads per KV head
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        att = torch.einsum("bhd,bthd->bht", q, k) / math.sqrt(d)
        att = att.softmax(-1)
        o = torch.einsum("bht,bthd->bhd", att, v).reshape(b, self.hq * d)
        y = o @ self.wo                                    # partial sum over heads
        dist.all_reduce(y, group=self.group)              # all-reduce 1
        x = x + y
        h = F.gelu(x @ self.w1, approximate="tanh")       # local d_ff block
        m = h @ self.w2                                    # partial sum over d_ff
        dist.all_reduce(m, group=self.group)              # all-reduce 2
        return x + m


def decode_layer_reference(shape: ModelShape, weights: Dict[str, torch.Tensor],
                           xs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Single-process decode of the same layer over the token steps ``xs``."""
    d, hq, hkv = shape.d_head, shape.n_heads, shape.n_kv_heads
    ks, vs, outs = [], [], []
    for x in xs:
        b = x.shape[0]
        q = (x @ weights["wq"]).view(b, hq, d)
        ks.append((x @ weights["wk"]).view(b, hkv, d))
        vs.append((x @ weights["wv"]).view(b, hkv, d))
        k = torch.stack(ks, 1).repeat_interleave(hq // hkv, dim=2)
        v = torch.stack(vs, 1).repeat_interleave(hq // hkv, dim=2)
        att = (torch.einsum("bhd,bthd->bht", q, k) / math.sqrt(d)).softmax(-1)
        o = torch.einsum("bht,bthd->bhd", att, v).reshape(b, hq * d)
        x = x + o @ weights["wo"]
        outs.append(x + F.gelu(x @ weights["w1"], approximate="tanh") @ weights["w2"])
    return outs


def random_layer_weights(shape: ModelShape, seed: int = 0, dtype=torch.float32) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    d, hd = shape.d_model, shape.d_head

    def w(*sz):
        return torch.randn(*sz, generator=g, dtype=dtype) / math.sqrt(sz[0])
    return {"wq": w(d, shape.n_heads * hd), "wk": w(d, shape.n_kv_heads * hd),
            "wv": w(d, shape.n_kv_heads * hd), "wo": w(shape.n_heads * hd, d),
            "w1": w(d, shape.d_ff), "w2": w(shape.d_ff, d)}
