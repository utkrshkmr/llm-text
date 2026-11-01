"""GPT-2 architecture from scratch (Part II of the book)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import FeedForward, LayerNorm, MultiHeadAttention

GPT_CONFIG_124M = dict(vocab_size=50257, context_length=1024, emb_dim=768, n_heads=12,
                       n_layers=12, drop_rate=0.1, qkv_bias=False)

GPT2_SIZES = {
    "gpt2-small (124M)": dict(emb_dim=768, n_layers=12, n_heads=12),
    "gpt2-medium (355M)": dict(emb_dim=1024, n_layers=24, n_heads=16),
    "gpt2-large (774M)": dict(emb_dim=1280, n_layers=36, n_heads=20),
    "gpt2-xl (1558M)": dict(emb_dim=1600, n_layers=48, n_heads=25),
}

# A tiny configuration that trains in seconds on a laptop CPU (used by tests/labs)
GPT_CONFIG_TINY = dict(vocab_size=512, context_length=64, emb_dim=64, n_heads=4,
                       n_layers=2, drop_rate=0.0, qkv_bias=False)


def gpt2_config(name: str = "gpt2-small (124M)", **overrides) -> dict:
    cfg = dict(GPT_CONFIG_124M)
    cfg.update(GPT2_SIZES[name])
    cfg.update(qkv_bias=True)  # the released OpenAI weights use biases on Q, K, V
    cfg.update(overrides)
    return cfg


class TransformerBlock(nn.Module):
    """Pre-LN block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"],
            use_sdpa=cfg.get("use_sdpa", False))
        self.ff = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, cache=None, pad_mask=None):
        x = x + self.drop_shortcut(self.att(self.norm1(x), cache=cache, pad_mask=pad_mask))
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
        for i, blk in enumerate(self.trf_blocks):
            blk.att.layer_idx = i
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    @property
    def n_layers(self):
        return len(self.trf_blocks)

    def forward(self, in_idx, cache=None, pad_mask=None):
        B, T = in_idx.shape
        start = cache.seq_len(0) if cache is not None else 0
        if start + T > self.cfg["context_length"]:
            raise ValueError("sequence exceeds context_length; truncate the input")
        pos = torch.arange(start, start + T, device=in_idx.device)
        x = self.drop_emb(self.tok_emb(in_idx) + self.pos_emb(pos)[None])
        for blk in self.trf_blocks:
            x = blk(x, cache=cache, pad_mask=pad_mask)
        return self.out_head(self.final_norm(x))


# --------------------------------------------------------------------------- weights
def load_gpt2_hf_state_dict(model: GPTModel, sd: dict) -> GPTModel:
    """Load weights from the Hugging Face ``gpt2`` safetensors layout.

    HF stores GPT-2 linear layers as Conv1D with weight shape (in, out), so every
    projection is transposed. The fused c_attn (in, 3*out) is split into Q, K, V.
    Usage::

        from safetensors.torch import load_file
        sd = load_file("gpt2/model.safetensors")
        model = GPTModel(gpt2_config("gpt2-small (124M)"))
        load_gpt2_hf_state_dict(model, sd)
    """
    def get(name):
        for key in (name, "transformer." + name):
            if key in sd:
                return sd[key]
        raise KeyError(name)

    def assign(param, tensor):
        if param.shape != tensor.shape:
            raise ValueError(f"shape mismatch {tuple(param.shape)} vs {tuple(tensor.shape)}")
        with torch.no_grad():
            param.copy_(tensor.to(param.dtype))

    assign(model.tok_emb.weight, get("wte.weight"))
    assign(model.pos_emb.weight, get("wpe.weight"))
    for i, blk in enumerate(model.trf_blocks):
        p = f"h.{i}."
        qw, kw, vw = torch.chunk(get(p + "attn.c_attn.weight"), 3, dim=-1)
        qb, kb, vb = torch.chunk(get(p + "attn.c_attn.bias"), 3, dim=-1)
        assign(blk.att.W_query.weight, qw.T); assign(blk.att.W_query.bias, qb)
        assign(blk.att.W_key.weight, kw.T); assign(blk.att.W_key.bias, kb)
        assign(blk.att.W_value.weight, vw.T); assign(blk.att.W_value.bias, vb)
        assign(blk.att.out_proj.weight, get(p + "attn.c_proj.weight").T)
        assign(blk.att.out_proj.bias, get(p + "attn.c_proj.bias"))
        assign(blk.ff.layers[0].weight, get(p + "mlp.c_fc.weight").T)
        assign(blk.ff.layers[0].bias, get(p + "mlp.c_fc.bias"))
        assign(blk.ff.layers[2].weight, get(p + "mlp.c_proj.weight").T)
        assign(blk.ff.layers[2].bias, get(p + "mlp.c_proj.bias"))
        assign(blk.norm1.scale, get(p + "ln_1.weight")); assign(blk.norm1.shift, get(p + "ln_1.bias"))
        assign(blk.norm2.scale, get(p + "ln_2.weight")); assign(blk.norm2.shift, get(p + "ln_2.bias"))
    assign(model.final_norm.scale, get("ln_f.weight"))
    assign(model.final_norm.shift, get("ln_f.bias"))
    assign(model.out_head.weight, get("wte.weight"))  # GPT-2 ties input/output embeddings
    return model
