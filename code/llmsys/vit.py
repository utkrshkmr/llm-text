"""Vision transformers as encoders: patchify, patch embedding, a ViT encoder (Chapter 9).

An image is turned into a sequence of (H/P)(W/P) patch tokens and then processed by exactly the
bidirectional ``EncoderBlock`` of ``llmsys.encoder``. Nothing in the transformer knows it is
looking at pixels; the only vision-specific parts are the patchify step, a 2-D position table
and the [CLS] token used for pooling.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncoderBlock, encoder_flops_per_sequence
from .layers import LayerNorm

VIT_CONFIG_TINY = dict(img_size=16, patch=4, in_ch=3, emb_dim=64, n_heads=4, n_layers=2,
                       drop_rate=0.0, qkv_bias=True, pooling="cls")


# --------------------------------------------------------------------------- patches
def n_tokens(H: int, W: int, P: int) -> int:
    """Number of patch tokens for an H x W image with P x P patches."""
    return (H // P) * (W // P)


def patchify(img: torch.Tensor, P: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, N, C*P*P) with N = (H/P)(W/P), patches in row-major order and each
    patch flattened channel-major (c, then row, then column), the same layout as F.unfold."""
    B, C, H, W = img.shape
    if H % P or W % P:
        raise ValueError(f"image {H}x{W} is not divisible by patch size {P}")
    x = img.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5)                    # (B, H/P, W/P, C, P, P)
    return x.reshape(B, (H // P) * (W // P), C * P * P)


def unpatchify(patches: torch.Tensor, P: int, C: int, H: int, W: int) -> torch.Tensor:
    """Inverse of ``patchify``."""
    B = patches.shape[0]
    x = patches.reshape(B, H // P, W // P, C, P, P).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C, H, W)


class PatchEmbed(nn.Module):
    """Flatten each patch and project it linearly: one GEMM of shape (B*N, C*P*P) x (C*P*P, d).
    A Conv2d with kernel = stride = P computes exactly the same thing."""

    def __init__(self, in_ch: int, patch: int, emb_dim: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(in_ch * patch * patch, emb_dim)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.proj(patchify(img, self.patch))


# --------------------------------------------------------------------------- the encoder
class ViTEncoder(nn.Module):
    """Patch embedding + [CLS] + learned 2-D position table + L encoder blocks + final norm.

    ``forward`` returns all token states (B, 1+N, d), what a VLM's projector consumes;
    ``embed`` returns one vector per image, by [CLS] or by mean pooling over the patches."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d, P = cfg["emb_dim"], cfg["patch"]
        self.grid = cfg["img_size"] // P
        self.patch_embed = PatchEmbed(cfg["in_ch"], P, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.grid ** 2, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg["n_layers"]))
        self.final_norm = LayerNorm(d)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(img)                                       # (B, N, d)
        B, N, _ = x.shape
        pos = self.pos_emb
        if N != self.grid ** 2:                                         # other resolution
            g = int(math.isqrt(N))
            pos = interpolate_pos_embed(self.pos_emb, self.grid, g)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + pos
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)                                                  # no mask: all tokens real
        return self.final_norm(x)

    def embed(self, img: torch.Tensor) -> torch.Tensor:
        tokens = self.forward(img)
        if self.cfg.get("pooling", "cls") == "cls":
            return tokens[:, 0]
        return tokens[:, 1:].mean(1)


def interpolate_pos_embed(pos_emb: torch.Tensor, old_grid: int, new_grid: int) -> torch.Tensor:
    """Resize a (1, 1+g^2, d) position table to a new g' x g' grid by bicubic interpolation of the
    patch part; the [CLS] slot is kept. This is how a ViT trained at 224 px is run at 384 px."""
    cls, grid = pos_emb[:, :1], pos_emb[:, 1:]
    d = grid.shape[-1]
    grid = grid.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)          # (1, d, g, g)
    grid = F.interpolate(grid, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, d)
    return torch.cat([cls, grid], dim=1)


# --------------------------------------------------------------------------- cost model
def vit_params_nonembed(cfg: dict) -> int:
    """Parameters of the transformer blocks (attention 4d^2, MLP 8d^2 per layer, plus norms)."""
    d = cfg["emb_dim"]
    return cfg["n_layers"] * (12 * d * d + 13 * d)


def vit_flops(cfg: dict, H: Optional[int] = None, W: Optional[int] = None) -> float:
    """Forward FLOPs for one image: the patch-embedding GEMM plus the encoder over 1+N tokens."""
    H = H or cfg["img_size"]
    W = W or cfg["img_size"]
    P, d = cfg["patch"], cfg["emb_dim"]
    N = n_tokens(H, W, P)
    embed = 2.0 * N * cfg["in_ch"] * P * P * d
    return embed + encoder_flops_per_sequence(vit_params_nonembed(cfg), 1 + N, cfg["n_layers"], d)
