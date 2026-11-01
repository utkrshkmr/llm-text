"""Napkin-math calculators used throughout the book's Systems Lens boxes and labs.

All formulas are for dense decoder-only transformers and ignore small terms
(norms, biases, softmax) unless stated.
"""
from __future__ import annotations

from dataclasses import dataclass

BYTES = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}

# Published peak numbers for common accelerators (dense, no sparsity).
# flops in FLOP/s for bf16 tensor-core matmul, bw in bytes/s of HBM.
HARDWARE = {
    "A100-80GB-SXM": dict(flops=312e12, bw=2.039e12, mem=80e9),
    "H100-SXM": dict(flops=989e12, bw=3.35e12, mem=80e9),
    "H200-SXM": dict(flops=989e12, bw=4.8e12, mem=141e9),
    "RTX-4090": dict(flops=165e12, bw=1.008e12, mem=24e9),
    "L4": dict(flops=121e12, bw=0.3e12, mem=24e9),
}


@dataclass
class ModelShape:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    gated_mlp: bool = True       # SwiGLU has 3 matrices, GELU MLP has 2
    tied_embeddings: bool = True

    @classmethod
    def from_qwen3(cls, cfg):
        return cls(cfg["n_layers"], cfg["emb_dim"], cfg["n_heads"], cfg["n_kv_groups"],
                   cfg["head_dim"], cfg["hidden_dim"], cfg["vocab_size"], True,
                   cfg.get("tie_embeddings", False))

    @classmethod
    def from_gpt(cls, cfg):
        d = cfg["emb_dim"]
        return cls(cfg["n_layers"], d, cfg["n_heads"], cfg["n_heads"], d // cfg["n_heads"],
                   4 * d, cfg["vocab_size"], False, True)


def params(m: ModelShape) -> dict:
    attn = m.d_model * m.n_heads * m.head_dim * 2 + m.d_model * m.n_kv_heads * m.head_dim * 2
    mlp = (3 if m.gated_mlp else 2) * m.d_model * m.d_ff
    per_layer = attn + mlp
    emb = m.vocab * m.d_model * (1 if m.tied_embeddings else 2)
    return dict(per_layer=per_layer, layers=per_layer * m.n_layers, embeddings=emb,
                total=per_layer * m.n_layers + emb,
                non_embedding=per_layer * m.n_layers)


def flops_per_token_forward(m: ModelShape, context: int = 0) -> float:
    """~2 FLOPs per parameter touched, plus attention score/value matmuls over context."""
    p = params(m)
    matmul = 2 * (p["layers"] + m.vocab * m.d_model)
    attn = 2 * 2 * m.n_layers * m.n_heads * m.head_dim * context
    return matmul + attn


def training_flops(n_params: float, n_tokens: float) -> float:
    """The classic C ~= 6 N D estimate (forward 2ND + backward 4ND)."""
    return 6.0 * n_params * n_tokens


def kv_cache_bytes(m: ModelShape, seq_len: int, batch: int = 1, dtype: str = "bf16") -> float:
    return 2 * m.n_layers * m.n_kv_heads * m.head_dim * seq_len * batch * BYTES[dtype]


def weight_bytes(m: ModelShape, dtype: str = "bf16") -> float:
    return params(m)["total"] * BYTES[dtype]


def training_memory_bytes(n_params: float, mixed_precision: bool = True) -> float:
    """Weights + grads + AdamW states (excluding activations).

    Mixed precision with fp32 master weights: 2 (bf16 w) + 2 (bf16 g) + 4 (fp32 master)
    + 4 + 4 (Adam m, v) = 16 bytes/param. Pure fp32: 4+4+4+4 = 16 as well, but no copy.
    """
    return 16.0 * n_params


def decode_step_time_lower_bound(m: ModelShape, batch: int, context: int, hw: str = "H100-SXM",
                                  dtype: str = "bf16") -> dict:
    """Roofline lower bound for one decode step: max(compute time, memory time)."""
    h = HARDWARE[hw]
    flops = batch * flops_per_token_forward(m, context)
    bytes_moved = weight_bytes(m, dtype) + kv_cache_bytes(m, context, batch, dtype)
    t_compute, t_mem = flops / h["flops"], bytes_moved / h["bw"]
    return dict(t_compute=t_compute, t_memory=t_mem, t=max(t_compute, t_mem),
                bound="memory" if t_mem > t_compute else "compute",
                arithmetic_intensity=flops / bytes_moved)


def mfu(tokens_per_sec: float, n_params: float, hw: str = "H100-SXM", n_gpus: int = 1) -> float:
    """Model FLOPs utilisation for training: achieved 6N FLOPs/token vs peak."""
    return tokens_per_sec * 6 * n_params / (HARDWARE[hw]["flops"] * n_gpus)
