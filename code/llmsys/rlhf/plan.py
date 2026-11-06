"""Memory planning for the four models of PPO-RLHF.

Byte counts per parameter (mixed precision, AdamW, no sharding):
    trained model : 2 (bf16 weights) + 2 (bf16 grads) + 4 (fp32 master) + 8 (Adam m, v) = 16
    frozen model  : 2 (bf16 weights)
These are the numbers behind the placement tables in Chapter 61.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

TRAINED_BYTES = 16
FROZEN_BYTES = 2
GB = 1e9


@dataclass
class RLHFModels:
    policy: float          # parameters
    value: float
    reward: float
    reference: float


def state_bytes(m: RLHFModels, offload_frozen: bool = False, zero_shards: int = 1) -> Dict[str, float]:
    """Static memory (weights, grads, optimizer) per model. ``zero_shards`` > 1 divides the
    trained state across data-parallel ranks (ZeRO-3/FSDP); frozen models may be offloaded."""
    frozen = 0.0 if offload_frozen else FROZEN_BYTES
    out = {
        "policy": m.policy * TRAINED_BYTES / zero_shards,
        "value": m.value * TRAINED_BYTES / zero_shards,
        "reward": m.reward * frozen / zero_shards,
        "reference": m.reference * frozen / zero_shards,
    }
    out["total"] = sum(out.values())
    return out


def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, tokens: int, dtype_bytes: int = 2) -> float:
    return 2 * n_layers * n_kv_heads * head_dim * tokens * dtype_bytes


def min_gpus(m: RLHFModels, gpu_mem_gb: float = 80.0, headroom: float = 0.7, **kw) -> int:
    """Smallest number of GPUs (fully sharded) whose usable memory holds the static state."""
    total = state_bytes(m, **kw)["total"]
    usable = gpu_mem_gb * GB * headroom
    n = 1
    while total / n > usable and n < 4096:
        n += 1
    return n


def step_flops(policy_params: float, value_params: float, reward_params: float, ref_params: float,
               batch: int, prompt_len: int, resp_len: int, ppo_epochs: int) -> Dict[str, float]:
    """Forward = 2N FLOPs/token, backward = 4N (attention terms ignored).

    Generation is counted as forward FLOPs on response tokens (prefill on prompt tokens),
    but on real hardware it is memory-bound and takes far longer than its FLOPs suggest.
    """
    T = prompt_len + resp_len
    gen = 2 * policy_params * batch * T
    score = 2 * (policy_params + ref_params + value_params + reward_params) * batch * T
    learn = ppo_epochs * 6 * (policy_params + value_params) * batch * T
    return {"generate": gen, "score": score, "learn": learn, "total": gen + score + learn}
