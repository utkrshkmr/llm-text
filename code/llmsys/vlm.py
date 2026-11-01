"""A minimal vision-language model in the LLaVA mould (Chapter 31).

image -> patches -> small ViT -> MLP projector -> spliced into the token-embedding
sequence at <image> placeholder positions -> a core ``Qwen3Model`` decoder.

Also contains the contrastive objectives used to pretrain image encoders (CLIP's
softmax/InfoNCE loss and SigLIP's sigmoid loss), vision-token accounting (AnyRes tiling,
2x2 token merging, video frames), an image-embedding cache, and multimodal-aware prefix
hashing for serving.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================== encoder
class PatchEmbed(nn.Module):
    """Split (B, C, H, W) into non-overlapping p x p patches and project each to d dims.

    A Conv2d with kernel = stride = p is exactly 'flatten each patch, multiply by W'.
    """

    def __init__(self, in_ch: int, patch: int, dim: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.proj(img)                       # (B, d, H/p, W/p)
        return x.flatten(2).transpose(1, 2)      # (B, N, d) with N = (H/p)(W/p)


class TinyViT(nn.Module):
    """Patch embedding + learned 2-D position embedding + a few pre-LN encoder blocks."""

    def __init__(self, img_size=16, patch=4, in_ch=3, dim=64, depth=1, heads=4):
        super().__init__()
        self.embed = PatchEmbed(in_ch, patch, dim)
        self.n_patches = (img_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):                     # (B, C, H, W) -> (B, N, dim)
        return self.norm(self.blocks(self.embed(img) + self.pos))


class MLPProjector(nn.Module):
    """LLaVA-1.5-style 2-layer MLP from vision width to LM width."""

    def __init__(self, d_vis: int, d_lm: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_vis, d_lm), nn.GELU(), nn.Linear(d_lm, d_lm))

    def forward(self, x):
        return self.net(x)


# =========================================================================== splicing + LM
def splice_image_embeddings(tok_embeds: torch.Tensor, input_ids: torch.Tensor,
                            image_feats: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Replace the embeddings at <image> placeholder positions by projected image features.

    tok_embeds: (B, T, d); input_ids: (B, T); image_feats: (M, N, d) for the M images in the
    batch, in row-major order of appearance. Every image must own exactly N placeholders.
    """
    mask = input_ids == image_token_id
    n_slots = int(mask.sum())
    if n_slots != image_feats.shape[0] * image_feats.shape[1]:
        raise ValueError(f"{n_slots} <image> slots but {image_feats.shape[0]} images x "
                         f"{image_feats.shape[1]} tokens")
    out = tok_embeds.clone()
    out[mask] = image_feats.reshape(-1, image_feats.shape[-1]).to(out.dtype)
    return out


def qwen3_forward_embeds(model, x: torch.Tensor, cache=None, pad_mask=None) -> torch.Tensor:
    """Run a core ``Qwen3Model`` starting from embeddings instead of token ids
    (the core forward always embeds ``in_idx`` itself)."""
    for blk in model.trf_blocks:
        x = blk(x, model.cos, model.sin, cache=cache, pad_mask=pad_mask)
    return model.out_head(model.final_norm(x))


class TinyVLM(nn.Module):
    """Vision encoder + projector + decoder LM. ``image_token_id`` must be in the LM vocab."""

    def __init__(self, lm: nn.Module, vision: nn.Module, d_vis: int, image_token_id: int):
        super().__init__()
        self.lm, self.vision = lm, vision
        self.proj = MLPProjector(d_vis, lm.cfg["emb_dim"])
        self.image_token_id = image_token_id

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.vision(images))           # (M, N, d_lm)

    def forward(self, input_ids, images=None, image_feats=None, cache=None):
        x = self.lm.tok_emb(input_ids)
        if image_feats is None and images is not None:
            image_feats = self.encode_images(images)
        if image_feats is not None:
            x = splice_image_embeddings(x, input_ids, image_feats, self.image_token_id)
        return qwen3_forward_embeds(self.lm, x, cache=cache)

    def freeze_for_stage(self, stage: int):
        """Stage 1 (alignment): train projector only. Stage 2: projector + LM.
        The vision tower stays frozen in both (the LLaVA recipe)."""
        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.lm.parameters():
            p.requires_grad = stage >= 2
        for p in self.proj.parameters():
            p.requires_grad = True


def caption_loss(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor):
    """Next-token loss only where ``loss_mask`` marks *target* tokens (the caption)."""
    tgt = input_ids[:, 1:]
    m = loss_mask[:, 1:]
    lp = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), tgt.flatten(), reduction="none")
    return (lp * m.flatten()).sum() / m.sum().clamp_min(1)


# =========================================================================== contrastive losses
def clip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, logit_scale: torch.Tensor):
    """Symmetric InfoNCE: each image should pick its own caption among the batch, and vice
    versa. Needs the *full* B x B similarity matrix (an all-gather across data-parallel ranks)."""
    i = F.normalize(img_emb, dim=-1)
    t = F.normalize(txt_emb, dim=-1)
    logits = logit_scale.exp() * i @ t.T
    target = torch.arange(len(i), device=i.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def siglip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, logit_scale: torch.Tensor,
                logit_bias: torch.Tensor):
    """Pairwise sigmoid loss: every (image, text) cell is an independent binary problem
    (diagonal = match). No softmax normaliser, so cells can be computed block by block."""
    i = F.normalize(img_emb, dim=-1)
    t = F.normalize(txt_emb, dim=-1)
    logits = logit_scale.exp() * i @ t.T + logit_bias
    labels = 2 * torch.eye(len(i), device=i.device) - 1          # +1 diag, -1 elsewhere
    return -F.logsigmoid(labels * logits).sum() / len(i)


def siglip_loss_chunked(img_emb, txt_emb, logit_scale, logit_bias, n_chunks: int):
    """Same value as ``siglip_loss`` but computed one text chunk at a time, mimicking how
    SigLIP passes text shards around a ring of devices instead of all-gathering."""
    i = F.normalize(img_emb, dim=-1)
    t = F.normalize(txt_emb, dim=-1)
    B = len(i)
    total = i.new_zeros(())
    for c, tc in enumerate(t.chunk(n_chunks)):
        start = sum(len(x) for x in t.chunk(n_chunks)[:c])
        logits = logit_scale.exp() * i @ tc.T + logit_bias
        labels = -torch.ones_like(logits)
        rows = torch.arange(start, start + len(tc))
        labels[rows, torch.arange(len(tc))] = 1
        total = total - F.logsigmoid(labels * logits).sum()
    return total / B


# =========================================================================== token accounting
def vit_tokens(height: int, width: int, patch: int = 14, merge: int = 1) -> int:
    """Patch tokens for one image; ``merge=2`` models 2x2 token merging (Qwen2-VL-style)."""
    return math.ceil(height / patch / merge) * math.ceil(width / patch / merge)


def anyres_tokens(height: int, width: int, tile: int = 336, patch: int = 14,
                  max_tiles: int = 6, include_base: bool = True) -> Dict[str, int]:
    """AnyRes-style accounting: cut the image into tile x tile crops (grid capped at
    ``max_tiles``), encode each crop plus one downscaled overview image."""
    gh, gw = math.ceil(height / tile), math.ceil(width / tile)
    while gh * gw > max_tiles:          # shrink the grid along the longer side
        if gh >= gw:
            gh -= 1
        else:
            gw -= 1
    per_tile = (tile // patch) ** 2
    tiles = gh * gw + (1 if include_base else 0)
    return {"grid": (gh, gw), "tiles": tiles, "tokens": tiles * per_tile}


def video_tokens(seconds: float, fps: float, height: int, width: int, patch: int = 14,
                 merge: int = 2, temporal_merge: int = 2) -> int:
    """Tokens for a clip: frames sampled at ``fps``, pairs of frames merged temporally."""
    frames = max(1, int(seconds * fps))
    return math.ceil(frames / temporal_merge) * vit_tokens(height, width, patch, merge)


def vlm_prefill_flops(n_image_tokens: int, n_text_tokens: int, vit_params: float,
                      lm_params: float, n_patches_encoded: Optional[int] = None) -> dict:
    """~2 FLOPs per parameter per token for each tower (attention terms ignored).

    The encoder runs over *patches* (before any merging), the LM over the merged image
    tokens plus the text tokens.
    """
    patches = n_patches_encoded if n_patches_encoded is not None else n_image_tokens
    enc = 2 * vit_params * patches
    lm = 2 * lm_params * (n_image_tokens + n_text_tokens)
    return {"encoder": enc, "lm": lm, "total": enc + lm, "encoder_share": enc / (enc + lm)}


# =========================================================================== serving helpers
def image_hash(img: torch.Tensor) -> str:
    """Content hash of a preprocessed image tensor (the cache key)."""
    t = img.detach().contiguous().cpu()
    return hashlib.sha256(t.numpy().tobytes() + str(tuple(t.shape)).encode()).hexdigest()[:16]


class ImageEmbeddingCache:
    """LRU cache of projected image features keyed by content hash.

    Multi-turn chats resend the same image every turn; caching skips the vision tower.
    ``capacity_bytes`` bounds memory (features are (N, d) per image).
    """

    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.used = 0
        self.store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.hits = self.misses = 0

    def get_or_compute(self, img: torch.Tensor, encode_fn) -> torch.Tensor:
        key = image_hash(img)
        if key in self.store:
            self.hits += 1
            self.store.move_to_end(key)
            return self.store[key]
        self.misses += 1
        feats = encode_fn(img[None])[0].detach()
        nbytes = feats.numel() * feats.element_size()
        while self.store and self.used + nbytes > self.capacity:
            _, old = self.store.popitem(last=False)
            self.used -= old.numel() * old.element_size()
        if nbytes <= self.capacity:
            self.store[key] = feats
            self.used += nbytes
        return feats


def multimodal_block_hashes(token_ids: Sequence[int], block_size: int, image_token_id: int,
                            image_hashes: Sequence[str]) -> List[str]:
    """Chained prefix-cache block hashes that include image *content*.

    All images share the same placeholder id, so hashing token ids alone would let two
    requests with different images hit each other's KV blocks. Each placeholder run is
    tagged with the hash of the image it stands for (images consumed in order).
    """
    hashes, prev = [], ""
    img_iter = iter(image_hashes)
    current_img = None
    in_run = False
    feed: List[str] = []
    for tok in token_ids:
        if tok == image_token_id:
            if not in_run:
                current_img = next(img_iter)
                in_run = True
            feed.append(f"img:{current_img}")
        else:
            in_run = False
            feed.append(str(tok))
    for s in range(0, len(feed) - len(feed) % block_size, block_size):
        blob = prev + "|" + ",".join(feed[s:s + block_size])
        prev = hashlib.sha256(blob.encode()).hexdigest()[:16]
        hashes.append(prev)
    return hashes
