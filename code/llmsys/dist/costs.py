"""Analytical cost models for Chapter 14.

alpha-beta model: sending a message of n bytes costs  alpha + n * beta  seconds,
alpha = latency per message, beta = 1 / link bandwidth (s/byte).
"""
from __future__ import annotations

import math


# --------------------------------------------------------------------------- collectives
def allreduce_time(n_bytes: float, p: int, alpha: float, beta: float, algo: str = "ring") -> float:
    """Time for an all-reduce of n bytes over p ranks (one link per rank)."""
    if p == 1:
        return 0.0
    if algo == "ring":            # 2(p-1) steps, each moves n/p bytes
        return 2 * (p - 1) * alpha + 2 * (p - 1) / p * n_bytes * beta
    if algo == "naive":           # root receives p-1 full messages then sends p-1
        return 2 * (p - 1) * (alpha + n_bytes * beta)
    if algo == "tree":            # reduce + broadcast along a binary tree (latency-optimal)
        return 2 * math.ceil(math.log2(p)) * (alpha + n_bytes * beta)
    if algo == "rec_halving":     # Rabenseifner: reduce-scatter by halving + all-gather by doubling
        return 2 * math.ceil(math.log2(p)) * alpha + 2 * (p - 1) / p * n_bytes * beta
    raise ValueError(algo)


def reduce_scatter_time(n_bytes, p, alpha, beta):
    return (p - 1) * alpha + (p - 1) / p * n_bytes * beta


def all_gather_time(n_bytes, p, alpha, beta):
    """n_bytes is the size of the *full* gathered tensor."""
    return (p - 1) * alpha + (p - 1) / p * n_bytes * beta


def bus_bandwidth(n_bytes: float, seconds: float, p: int) -> float:
    """NCCL-tests convention: algorithm bandwidth scaled by 2(p-1)/p for all-reduce."""
    return n_bytes / seconds * 2 * (p - 1) / p


# --------------------------------------------------------------------------- memory
def zero_memory_per_gpu(n_params: float, n_gpus: int, stage: int = 0,
                        weight_bytes: int = 2, grad_bytes: int = 2, optim_bytes: int = 12) -> float:
    """Model-state bytes per GPU for mixed-precision Adam (ZeRO paper notation K=12).

    stage 0: plain DDP   (2 + 2 + 12) * N
    stage 1: shard optimizer states          (2 + 2 + 12/Nd) * N
    stage 2: + shard gradients               (2 + (2 + 12)/Nd) * N
    stage 3: + shard parameters (== FSDP)    (2 + 2 + 12) * N / Nd
    """
    w, g, o = weight_bytes, grad_bytes, optim_bytes
    d = n_gpus
    if stage == 0:
        per = w + g + o
    elif stage == 1:
        per = w + g + o / d
    elif stage == 2:
        per = w + (g + o) / d
    elif stage == 3:
        per = (w + g + o) / d
    else:
        raise ValueError(stage)
    return per * n_params


def dp_comm_bytes_per_step(n_params: float, stage: int = 0, grad_bytes: int = 2) -> float:
    """Bytes each rank sends per step (ring algorithms, large p limit).

    DDP / ZeRO-1 / ZeRO-2: one all-reduce (= reduce-scatter + all-gather) of the grads -> 2*Psi.
    ZeRO-3: all-gather params in forward + again in backward + reduce-scatter grads -> 3*Psi.
    """
    psi = n_params * grad_bytes
    return 3 * psi if stage == 3 else 2 * psi


# --------------------------------------------------------------------------- tensor / pipeline
def tp_comm_bytes_per_layer(batch: int, seq: int, d_model: int, tp: int, act_bytes: int = 2,
                            training: bool = True) -> float:
    """Megatron TP: 2 all-reduces of the (b, s, h) activation per layer in forward
    (after attention and after MLP), 2 more in backward. Returns bytes sent per rank."""
    n = batch * seq * d_model * act_bytes
    per_allreduce = 2 * (tp - 1) / tp * n
    return (4 if training else 2) * per_allreduce


def pipeline_bubble_fraction(stages: int, microbatches: int, interleave: int = 1) -> float:
    """Idle fraction of a GPipe/1F1B schedule: (p-1) / (m + p - 1); interleaving with v
    virtual stages per device shrinks the bubble by v."""
    p, m, v = stages, microbatches, interleave
    bubble = (p - 1) / v
    return bubble / (m + bubble)


def activation_bytes_per_layer(batch: int, seq: int, d_model: int, n_heads: int,
                               act_bytes: int = 2, flash: bool = True) -> float:
    """Korthikanti et al. (2022) estimate: s*b*h*(34 + 5*a*s/h) bytes per layer in fp16/bf16
    (without recomputation). With FlashAttention the 5as/h term (the materialised score
    matrices) disappears."""
    s, b, h, a = seq, batch, d_model, n_heads
    per = 34 + (0 if flash else 5 * a * s / h)
    return s * b * h * per * act_bytes / 2
