"""Qwen3 (dense) architecture from scratch: RMSNorm, SwiGLU, RoPE, GQA, QK-norm.

Used from Part IV onwards as the base model for inference, evaluation, RL and
distillation. ``load_qwen3_hf_state_dict`` maps the official Hugging Face safetensors
layout onto this module so the real Qwen3-0.6B checkpoint can be loaded on a machine
with internet access.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn

from .layers import GroupedQueryAttention, RMSNorm, SwiGLUFeedForward, compute_rope_params

QWEN3_CONFIGS = {
    "0.6B": dict(vocab_size=151_936, context_length=40_960, emb_dim=1024, n_heads=16,
                 n_layers=28, hidden_dim=3072, head_dim=128, qk_norm=True, n_kv_groups=8,
                 rope_base=1_000_000.0, tie_embeddings=True),
    "1.7B": dict(vocab_size=151_936, context_length=40_960, emb_dim=2048, n_heads=16,
                 n_layers=28, hidden_dim=6144, head_dim=128, qk_norm=True, n_kv_groups=8,
                 rope_base=1_000_000.0, tie_embeddings=True),
    "4B": dict(vocab_size=151_936, context_length=40_960, emb_dim=2560, n_heads=32,
               n_layers=36, hidden_dim=9728, head_dim=128, qk_norm=True, n_kv_groups=8,
               rope_base=1_000_000.0, tie_embeddings=True),
    "8B": dict(vocab_size=151_936, context_length=40_960, emb_dim=4096, n_heads=32,
               n_layers=36, hidden_dim=12288, head_dim=128, qk_norm=True, n_kv_groups=8,
               rope_base=1_000_000.0, tie_embeddings=False),
    "14B": dict(vocab_size=151_936, context_length=40_960, emb_dim=5120, n_heads=40,
                n_layers=40, hidden_dim=17408, head_dim=128, qk_norm=True, n_kv_groups=8,
                rope_base=1_000_000.0, tie_embeddings=False),
    "32B": dict(vocab_size=151_936, context_length=40_960, emb_dim=5120, n_heads=64,
                n_layers=64, hidden_dim=25600, head_dim=128, qk_norm=True, n_kv_groups=8,
                rope_base=1_000_000.0, tie_embeddings=False),
}

# A tiny Qwen3-shaped model for CPU tests and labs
QWEN3_CONFIG_TINY = dict(vocab_size=512, context_length=256, emb_dim=64, n_heads=4,
                         n_layers=2, hidden_dim=128, head_dim=16, qk_norm=True,
                         n_kv_groups=2, rope_base=10_000.0, tie_embeddings=True)


class Qwen3Block(nn.Module):
    def __init__(self, cfg, dtype=None):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"], num_heads=cfg["n_heads"], num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"], qk_norm=cfg["qk_norm"], dtype=dtype,
            use_sdpa=cfg.get("use_sdpa", False))
        self.ff = SwiGLUFeedForward(cfg["emb_dim"], cfg["hidden_dim"], dtype=dtype)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        x = x + self.att(self.norm1(x), cos, sin, cache=cache, pad_mask=pad_mask)
        x = x + self.ff(self.norm2(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg, dtype=None, rope_len: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=dtype)
        self.trf_blocks = nn.ModuleList([Qwen3Block(cfg, dtype) for _ in range(cfg["n_layers"])])
        for i, blk in enumerate(self.trf_blocks):
            blk.att.layer_idx = i
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=dtype)
        if cfg.get("tie_embeddings", False):
            self.out_head.weight = self.tok_emb.weight
        # RoPE tables are buffers (moved with .to(device)) but not saved in state_dict
        cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"],
                                       rope_len or cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def n_layers(self):
        return len(self.trf_blocks)

    def forward(self, in_idx, cache=None, pad_mask=None):
        x = self.tok_emb(in_idx)
        for blk in self.trf_blocks:
            x = blk(x, self.cos, self.sin, cache=cache, pad_mask=pad_mask)
        return self.out_head(self.final_norm(x))


def load_qwen3_hf_state_dict(model: Qwen3Model, sd: dict) -> Qwen3Model:
    """Copy tensors from the HF ``Qwen/Qwen3-*`` safetensors layout into ``model``."""
    def assign(param, name):
        t = sd[name]
        if param.shape != t.shape:
            raise ValueError(f"{name}: {tuple(param.shape)} vs {tuple(t.shape)}")
        with torch.no_grad():
            param.copy_(t.to(param.dtype))

    assign(model.tok_emb.weight, "model.embed_tokens.weight")
    for i, blk in enumerate(model.trf_blocks):
        p = f"model.layers.{i}."
        assign(blk.att.W_query.weight, p + "self_attn.q_proj.weight")
        assign(blk.att.W_key.weight, p + "self_attn.k_proj.weight")
        assign(blk.att.W_value.weight, p + "self_attn.v_proj.weight")
        assign(blk.att.out_proj.weight, p + "self_attn.o_proj.weight")
        if blk.att.q_norm is not None:
            assign(blk.att.q_norm.scale, p + "self_attn.q_norm.weight")
            assign(blk.att.k_norm.scale, p + "self_attn.k_norm.weight")
        assign(blk.norm1.scale, p + "input_layernorm.weight")
        assign(blk.norm2.scale, p + "post_attention_layernorm.weight")
        assign(blk.ff.fc1.weight, p + "mlp.gate_proj.weight")
        assign(blk.ff.fc2.weight, p + "mlp.up_proj.weight")
        assign(blk.ff.fc3.weight, p + "mlp.down_proj.weight")
    assign(model.final_norm.scale, "model.norm.weight")
    if "lm_head.weight" in sd and not model.cfg.get("tie_embeddings", False):
        assign(model.out_head.weight, "lm_head.weight")
    return model


def load_qwen3_from_dir(path: str, size: str = "0.6B", dtype=torch.bfloat16,
                        device="cpu") -> Qwen3Model:
    """Load a downloaded HF snapshot directory (``huggingface-cli download Qwen/Qwen3-0.6B``)."""
    from safetensors.torch import load_file

    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    sd = {}
    for f in files:
        sd.update(load_file(os.path.join(path, f)))
    model = Qwen3Model(QWEN3_CONFIGS[size], dtype=dtype)
    load_qwen3_hf_state_dict(model, sd)
    return model.to(device).eval()


class Qwen3Tokenizer:
    """Thin wrapper over the HF ``tokenizers`` fast tokenizer plus Qwen's chat format.

    ``apply_chat_template`` reproduces the ChatML layout Qwen3 was trained with::

        <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    """

    IM_START, IM_END, EOT = "<|im_start|>", "<|im_end|>", "<|endoftext|>"

    def __init__(self, tokenizer_json: str):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(tokenizer_json)
        self.eos_token_id = self.tok.token_to_id(self.IM_END)
        self.pad_token_id = self.tok.token_to_id(self.EOT)

    def encode(self, text: str, chat: bool = False, enable_thinking: bool = False):
        if chat:
            text = self.apply_chat_template(text, enable_thinking)
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def apply_chat_template(self, prompt_or_messages, enable_thinking: bool = False) -> str:
        msgs = prompt_or_messages
        if isinstance(msgs, str):
            msgs = [{"role": "user", "content": msgs}]
        s = "".join(f"{self.IM_START}{m['role']}\n{m['content']}{self.IM_END}\n" for m in msgs)
        s += f"{self.IM_START}assistant\n"
        if not enable_thinking:
            s += "<think>\n\n</think>\n\n"
        return s
