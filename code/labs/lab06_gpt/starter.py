"""Lab 6 (starter): assemble GPT-2 and account for every parameter, FLOP and byte.

The classes below deliberately use the same attribute names as ``llmsys.gpt`` so that a
state_dict can be moved between the two implementations and outputs compared exactly.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from llmsys.layers import MultiHeadAttention

# --------------------------------------------------------------------------- Task 1
class LayerNorm(nn.Module):
    """y = scale * (x - mean) / sqrt(var + eps) + shift, statistics over the last dim.

    Uses the *biased* variance (divide by n), matching GPT-2's TensorFlow implementation.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Hint: statistics over dim=-1 with keepdim=True; use unbiased=False for the variance.
        raise NotImplementedError("TODO Task 1: LayerNorm.forward")


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """tanh approximation of GELU."""
    raise NotImplementedError("TODO Task 1: gelu_tanh (0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))))")


class GELU(nn.Module):
    def forward(self, x):
        return gelu_tanh(x)


class FeedForward(nn.Module):
    """Expand d -> 4d, GELU, contract 4d -> d. Attribute ``layers`` = Sequential(lin, act, lin)."""

    def __init__(self, emb_dim: int):
        super().__init__()
        # Keep the attribute name ``layers`` so state_dicts match llmsys.gpt.
        raise NotImplementedError("TODO Task 1: FeedForward.__init__")

    def forward(self, x):
        return self.layers(x)


# --------------------------------------------------------------------------- Task 2
class TransformerBlock(nn.Module):
    """Pre-LN block with two residual branches and dropout on each branch output."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"],
                                      cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"])
        self.ff = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre-LN: normalise, transform, dropout, then add the shortcut. Twice.
        raise NotImplementedError("TODO Task 2: TransformerBlock.forward")


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        # token emb + position emb -> dropout -> blocks -> final norm -> out_head
        raise NotImplementedError("TODO Task 2: GPTModel.forward")


# --------------------------------------------------------------------------- Task 3
@torch.no_grad()
def generate_greedy(model, idx: torch.Tensor, max_new_tokens: int, context_size: int):
    """Append argmax tokens one at a time, cropping the context to ``context_size``."""
    raise NotImplementedError("TODO Task 3: generate_greedy")


# --------------------------------------------------------------------------- Task 4
def count_params_analytic(cfg, tie_embeddings: bool = False) -> dict:
    """Closed-form parameter count of GPTModel (must match the module exactly)."""
    raise NotImplementedError("TODO Task 4: count_params_analytic")


def forward_flops(cfg, batch: int, seq_len: int) -> int:
    """Matmul FLOPs of one forward pass (2*m*n*k per GEMM), as counted by FlopCounterMode.

    Per token and layer: 2*(4d^2) for Q,K,V,O + 2*(8d^2) for the MLP; plus the two
    attention matmuls QK^T and AV, each 2*T*d per token (full T x T, no causal savings
    because the explicit implementation computes the masked half too); plus 2*d*V for
    the output head.
    """
    raise NotImplementedError("TODO Task 4: forward_flops")


def activation_bytes_estimate(cfg, batch: int, seq_len: int, bytes_per_el: int = 4) -> int:
    """Rough count of tensors saved for backward in one block, times L (no recomputation).

    Saved per block (elements): LN1 in (sbd), Q,K,V (3sbd), softmax probs (b*h*s*s),
    dropout masks ignored (drop_rate=0), context (sbd), out_proj in (sbd), LN2 in (sbd),
    MLP in (sbd), MLP hidden pre-GELU (4sbd) and post-GELU (4sbd), plus residual adds are
    free. Total ~ 16*s*b*d + b*h*s^2 elements per layer.
    """
    raise NotImplementedError("TODO Task 5: activation_bytes_estimate")


def measure_saved_activation_bytes(model, idx: torch.Tensor) -> int:
    """Run forward with autograd enabled and sum the bytes of tensors saved for backward.

    Uses ``torch.autograd.graph.saved_tensors_hooks``; storages are de-duplicated so a tensor
    saved by two ops is counted once. Parameters are excluded (they live anyway).
    """
    raise NotImplementedError("TODO Task 5: measure_saved_activation_bytes (use torch.autograd.graph.saved_tensors_hooks)")


def model_memory_bytes(n_params: int, bytes_per_param: int = 4) -> int:
    return n_params * bytes_per_param
