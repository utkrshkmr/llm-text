"""Lab 1 (starter -- fill in the TODOs): a napkin-math calculator for decoder-only LLMs.

Every function works on the *config dicts* used by llmsys (``GPT_CONFIG_124M``-style for
GPT-2, ``QWEN3_CONFIGS[...]``-style for Qwen3), so results can be checked against the
real ``nn.Module`` built from the same dict.
"""
from __future__ import annotations

import math

BYTES_PER = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}

HARDWARE = {  # dense bf16 tensor-core FLOP/s, HBM bytes/s, HBM capacity in bytes
    "A100-40GB-PCIe": dict(flops=312e12, bw=1.555e12, mem=40e9),   # the book's reference box
    "A100-80GB-SXM": dict(flops=312e12, bw=2.039e12, mem=80e9),
    "H100-SXM": dict(flops=989e12, bw=3.35e12, mem=80e9),
    "RTX-4090": dict(flops=165e12, bw=1.008e12, mem=24e9),
}


def is_qwen3(cfg: dict) -> bool:
    return "n_kv_groups" in cfg


# --------------------------------------------------------------------------- Task 1
def gpt_param_count(cfg: dict) -> int:
    """Exact parameter count of ``llmsys.gpt.GPTModel(cfg)`` (output head NOT tied)."""
    raise NotImplementedError('TODO: Sum embeddings (token + position), per-layer attention/MLP/LayerNorm params, the final norm and the untied output head.')


def qwen3_param_count(cfg: dict) -> int:
    """Exact parameter count of ``llmsys.qwen3.Qwen3Model(cfg)`` (unique tensors)."""
    raise NotImplementedError('TODO: q/k/v/o without biases (k, v use n_kv_groups heads), optional q/k RMSNorm (head_dim each), 3 SwiGLU matrices, 2 RMSNorms per layer, final norm; respect tie_embeddings.')


def param_count(cfg: dict) -> int:
    return qwen3_param_count(cfg) if is_qwen3(cfg) else gpt_param_count(cfg)


# --------------------------------------------------------------------------- Task 2
def forward_flops_per_token(cfg: dict, context: int) -> float:
    """Matmul FLOPs to process ONE token whose query attends to ``context`` keys.

    2 FLOPs per weight in every Linear layer (including the LM head; embeddings are
    lookups, not matmuls) plus QK^T and AV: 2 * 2 * n_heads * head_dim * context per layer.
    """
    raise NotImplementedError('TODO: 2 FLOPs per Linear weight + 4*n_heads*head_dim*context for QK^T and AV, per layer, plus 2*d*V for the LM head.')


def training_flops(n_params: float, n_tokens: float) -> float:
    """C ~= 6 N D: 2ND forward + 4ND backward."""
    raise NotImplementedError('TODO: Use C = 6 N D.')


def training_days(n_params: float, n_tokens: float, n_gpus: int, hw: str = "H100-SXM",
                  mfu: float = 0.4) -> float:
    raise NotImplementedError('TODO: FLOPs / (n_gpus * peak FLOP/s * mfu), converted to days.')


# --------------------------------------------------------------------------- Task 3
def kv_cache_bytes(cfg: dict, seq_len: int, batch: int = 1, dtype: str = "bf16") -> float:
    """2 (K and V) x layers x kv_heads x head_dim x tokens x batch x bytes."""
    raise NotImplementedError('TODO: 2 tensors (K and V) per layer, each (batch, kv_heads, seq_len, head_dim) elements.')


def training_state_bytes(n_params: float) -> float:
    """bf16 weights (2) + bf16 grads (2) + fp32 master (4) + Adam m, v (4 + 4) = 16 B/param."""
    raise NotImplementedError('TODO: Mixed-precision AdamW: 16 bytes per parameter.')


def max_batch_for_kv(cfg: dict, hw: str, seq_len: int, dtype: str = "bf16",
                     reserve_frac: float = 0.1) -> int:
    """Largest batch whose weights + KV cache fit in HBM, keeping ``reserve_frac`` free."""
    raise NotImplementedError('TODO: (HBM*(1-reserve) - weight bytes) // KV bytes of one sequence, floored at 0.')


# --------------------------------------------------------------------------- Task 4
def decode_step_roofline(cfg: dict, batch: int, context: int, hw: str = "H100-SXM",
                         dtype: str = "bf16") -> dict:
    """Lower bound on one decode step (each of ``batch`` sequences emits one token)."""
    raise NotImplementedError('TODO: t_compute = FLOPs/peak, t_memory = (weights + KV)/bandwidth; step time is the max. Return the dict keys used in the tests.')


def critical_batch(cfg: dict, hw: str = "H100-SXM", dtype: str = "bf16", context: int = 0) -> int:
    """Smallest batch at which a decode step becomes compute-bound (KV traffic ignored
    when context=0). Solves batch * F / peak_flops >= W / peak_bw."""
    raise NotImplementedError("TODO: Smallest batch for which decode_step_roofline(...)['bound'] == 'compute'.")


# --------------------------------------------------------------------------- Task 7
def kaplan_forward_flops(cfg: dict, n_ctx: int) -> float:
    """Forward FLOPs per token in the convention of Kaplan et al. (2020), Table 1:
    C_forward = 2 N + 2 n_layer n_ctx d_attn. N counts the weight matrices of the blocks
    (12 n_layer d_model^2 for GPT-2; biases, norms, embeddings and the LM head are left out) and
    the context term is the causal average: a token attends to n_ctx / 2 keys on average.
    """
    raise NotImplementedError(
        "TODO Task 7: 2 * (block weight-matrix params) + 2 * n_layers * n_ctx * n_heads * head_dim.")
