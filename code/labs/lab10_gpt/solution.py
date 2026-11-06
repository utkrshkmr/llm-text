"""Lab 10 (reference solution): assemble GPT-2 and account for every parameter, FLOP and byte.

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
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """tanh approximation of GELU."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class GELU(nn.Module):
    def forward(self, x):
        return gelu_tanh(x)


class FeedForward(nn.Module):
    """Expand d -> 4d, GELU, contract 4d -> d. Attribute ``layers`` = Sequential(lin, act, lin)."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(emb_dim, 4 * emb_dim), GELU(),
                                    nn.Linear(4 * emb_dim, emb_dim))

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
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x


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
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop_emb(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.trf_blocks:
            x = blk(x)
        return self.out_head(self.final_norm(x))


# --------------------------------------------------------------------------- Task 3
@torch.no_grad()
def generate_greedy(model, idx: torch.Tensor, max_new_tokens: int, context_size: int):
    """Append argmax tokens one at a time, cropping the context to ``context_size``."""
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


# --------------------------------------------------------------------------- Task 4
def count_params_analytic(cfg, tie_embeddings: bool = False) -> dict:
    """Closed-form parameter count of GPTModel (must match the module exactly)."""
    d, V, T, L = cfg["emb_dim"], cfg["vocab_size"], cfg["context_length"], cfg["n_layers"]
    attn = 3 * d * d + (3 * d if cfg["qkv_bias"] else 0) + d * d + d      # QKV (+bias) + out_proj
    mlp = d * 4 * d + 4 * d + 4 * d * d + d                                # two linears w/ bias
    norms = 2 * 2 * d                                                      # 2 LayerNorms
    per_layer = attn + mlp + norms
    tok, pos, final_norm = V * d, T * d, 2 * d
    head = 0 if tie_embeddings else V * d
    total = tok + pos + L * per_layer + final_norm + head
    return dict(per_layer=per_layer, attention=attn, mlp=mlp, layer_norms=norms,
                token_embedding=tok, position_embedding=pos, out_head=head, total=total)


def forward_flops(cfg, batch: int, seq_len: int) -> int:
    """Matmul FLOPs of one forward pass (2*m*n*k per GEMM), as counted by FlopCounterMode.

    Per token and layer: 2*(4d^2) for Q,K,V,O + 2*(8d^2) for the MLP; plus the two
    attention matmuls QK^T and AV, each 2*T*d per token (full T x T, no causal savings
    because the explicit implementation computes the masked half too); plus 2*d*V for
    the output head.
    """
    d, V, L = cfg["emb_dim"], cfg["vocab_size"], cfg["n_layers"]
    tokens = batch * seq_len
    linear = L * (2 * 4 * d * d + 2 * 8 * d * d) + 2 * d * V
    attention = L * 2 * (2 * seq_len * d)
    return tokens * (linear + attention)


def activation_bytes_estimate(cfg, batch: int, seq_len: int, bytes_per_el: int = 4) -> int:
    """Rough count of tensors saved for backward in one block, times L (no recomputation).

    Saved per block (elements): LN1 in (sbd), Q,K,V (3sbd), softmax probs (b*h*s*s),
    dropout masks ignored (drop_rate=0), context (sbd), out_proj in (sbd), LN2 in (sbd),
    MLP in (sbd), MLP hidden pre-GELU (4sbd) and post-GELU (4sbd), plus residual adds are
    free. Total ~ 16*s*b*d + b*h*s^2 elements per layer.
    """
    d, h, L = cfg["emb_dim"], cfg["n_heads"], cfg["n_layers"]
    s, b = seq_len, batch
    per_layer = 16 * s * b * d + b * h * s * s
    return L * per_layer * bytes_per_el


def measure_saved_activation_bytes(model, idx: torch.Tensor) -> int:
    """Run forward with autograd enabled and sum the bytes of tensors saved for backward.

    Uses ``torch.autograd.graph.saved_tensors_hooks``; storages are de-duplicated so a tensor
    saved by two ops is counted once. Parameters are excluded (they live anyway).
    """
    param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen = {}

    def pack(t):
        try:
            ptr = t.untyped_storage().data_ptr()
            if ptr not in param_ptrs and ptr not in seen:
                seen[ptr] = t.untyped_storage().nbytes()
        except Exception:
            pass
        return t

    model.train()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(idx)
    del out
    return sum(seen.values())


def model_memory_bytes(n_params: int, bytes_per_param: int = 4) -> int:
    return n_params * bytes_per_param
