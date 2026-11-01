"""Lab 1 (reference solution): a napkin-math calculator for decoder-only LLMs.

Every function works on the *config dicts* used by llmsys (``GPT_CONFIG_124M``-style for
GPT-2, ``QWEN3_CONFIGS[...]``-style for Qwen3), so results can be checked against the
real ``nn.Module`` built from the same dict.
"""
from __future__ import annotations

import math

BYTES_PER = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}

HARDWARE = {  # dense bf16 tensor-core FLOP/s, HBM bytes/s, HBM capacity in bytes
    "A100-80GB-SXM": dict(flops=312e12, bw=2.039e12, mem=80e9),
    "H100-SXM": dict(flops=989e12, bw=3.35e12, mem=80e9),
    "RTX-4090": dict(flops=165e12, bw=1.008e12, mem=24e9),
}


def is_qwen3(cfg: dict) -> bool:
    return "n_kv_groups" in cfg


# --------------------------------------------------------------------------- Task 1
def gpt_param_count(cfg: dict) -> int:
    """Exact parameter count of ``llmsys.gpt.GPTModel(cfg)`` (output head NOT tied)."""
    V, C, d, L = cfg["vocab_size"], cfg["context_length"], cfg["emb_dim"], cfg["n_layers"]
    qkv_bias = d if cfg.get("qkv_bias", False) else 0
    attn = 3 * (d * d + qkv_bias) + (d * d + d)          # W_q, W_k, W_v, out_proj(+bias)
    mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)           # two Linear layers with bias
    norms = 2 * (2 * d)                                   # two LayerNorms: scale + shift
    per_layer = attn + mlp + norms
    embeddings = V * d + C * d                            # token + learned positions
    head = V * d                                          # separate out_head (no bias)
    final_norm = 2 * d
    return embeddings + L * per_layer + final_norm + head


def qwen3_param_count(cfg: dict) -> int:
    """Exact parameter count of ``llmsys.qwen3.Qwen3Model(cfg)`` (unique tensors)."""
    V, d, L = cfg["vocab_size"], cfg["emb_dim"], cfg["n_layers"]
    H, G, hd, ff = cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], cfg["hidden_dim"]
    attn = d * H * hd + 2 * d * G * hd + H * hd * d       # q, k, v, o (no biases)
    qk_norm = 2 * hd if cfg.get("qk_norm", False) else 0
    mlp = 3 * d * ff                                      # gate, up, down
    norms = 2 * d                                         # two RMSNorms (scale only)
    per_layer = attn + qk_norm + mlp + norms
    head = 0 if cfg.get("tie_embeddings", False) else V * d
    return V * d + L * per_layer + d + head


def param_count(cfg: dict) -> int:
    return qwen3_param_count(cfg) if is_qwen3(cfg) else gpt_param_count(cfg)


# --------------------------------------------------------------------------- Task 2
def forward_flops_per_token(cfg: dict, context: int) -> float:
    """Matmul FLOPs to process ONE token whose query attends to ``context`` keys.

    2 FLOPs per weight in every Linear layer (including the LM head; embeddings are
    lookups, not matmuls) plus QK^T and AV: 2 * 2 * n_heads * head_dim * context per layer.
    """
    d, L, V = cfg["emb_dim"], cfg["n_layers"], cfg["vocab_size"]
    if is_qwen3(cfg):
        H, G, hd, ff = cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"], cfg["hidden_dim"]
        linear = d * H * hd + 2 * d * G * hd + H * hd * d + 3 * d * ff
    else:
        H, hd = cfg["n_heads"], d // cfg["n_heads"]
        linear = 4 * d * d + 8 * d * d
    attn = 2 * 2 * H * hd * context
    return L * (2 * linear + attn) + 2 * d * V


def training_flops(n_params: float, n_tokens: float) -> float:
    """C ~= 6 N D: 2ND forward + 4ND backward."""
    return 6.0 * n_params * n_tokens


def training_days(n_params: float, n_tokens: float, n_gpus: int, hw: str = "H100-SXM",
                  mfu: float = 0.4) -> float:
    secs = training_flops(n_params, n_tokens) / (n_gpus * HARDWARE[hw]["flops"] * mfu)
    return secs / 86400.0


# --------------------------------------------------------------------------- Task 3
def kv_cache_bytes(cfg: dict, seq_len: int, batch: int = 1, dtype: str = "bf16") -> float:
    """2 (K and V) x layers x kv_heads x head_dim x tokens x batch x bytes."""
    if is_qwen3(cfg):
        kv_heads, hd = cfg["n_kv_groups"], cfg["head_dim"]
    else:
        kv_heads, hd = cfg["n_heads"], cfg["emb_dim"] // cfg["n_heads"]
    return 2 * cfg["n_layers"] * kv_heads * hd * seq_len * batch * BYTES_PER[dtype]


def training_state_bytes(n_params: float) -> float:
    """bf16 weights (2) + bf16 grads (2) + fp32 master (4) + Adam m, v (4 + 4) = 16 B/param."""
    return 16.0 * n_params


def max_batch_for_kv(cfg: dict, hw: str, seq_len: int, dtype: str = "bf16",
                     reserve_frac: float = 0.1) -> int:
    """Largest batch whose weights + KV cache fit in HBM, keeping ``reserve_frac`` free."""
    budget = HARDWARE[hw]["mem"] * (1 - reserve_frac) - param_count(cfg) * BYTES_PER[dtype]
    per_seq = kv_cache_bytes(cfg, seq_len, 1, dtype)
    return max(0, int(budget // per_seq))


# --------------------------------------------------------------------------- Task 4
def decode_step_roofline(cfg: dict, batch: int, context: int, hw: str = "H100-SXM",
                         dtype: str = "bf16") -> dict:
    """Lower bound on one decode step (each of ``batch`` sequences emits one token)."""
    h = HARDWARE[hw]
    flops = batch * forward_flops_per_token(cfg, context)
    bytes_moved = param_count(cfg) * BYTES_PER[dtype] + kv_cache_bytes(cfg, context, batch, dtype)
    t_c, t_m = flops / h["flops"], bytes_moved / h["bw"]
    return dict(flops=flops, bytes=bytes_moved, t_compute=t_c, t_memory=t_m,
                seconds=max(t_c, t_m), bound="memory" if t_m >= t_c else "compute",
                tokens_per_s=batch / max(t_c, t_m))


def critical_batch(cfg: dict, hw: str = "H100-SXM", dtype: str = "bf16", context: int = 0) -> int:
    """Smallest batch at which a decode step becomes compute-bound (KV traffic ignored
    when context=0). Solves batch * F / peak_flops >= W / peak_bw."""
    for b in range(1, 1 << 16):
        if decode_step_roofline(cfg, b, context, hw, dtype)["bound"] == "compute":
            return b
    return -1
