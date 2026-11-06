"""Lab 67: generalist post-training plumbing, evaluation at scale, and a tiny VLM.

Starter code. Fill in the TODOs. Every function here has a twin in ``llmsys.alignment`` or
``llmsys.vlm``; the tests check that your version agrees with the library and
with the invariants stated in the chapter.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from llmsys.eval.harness import wilson_interval
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model
from llmsys.vlm import TinyViT, TinyVLM, caption_loss, image_hash

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- Task 1
def mixture_weights(sizes: Dict[str, int], alpha: float = 1.0) -> Dict[str, float]:
    """Sampling probability per source, proportional to n_i ** alpha."""
    raise NotImplementedError("TODO: Task 1: p_i proportional to n_i ** alpha (skip empty sources).")


# --------------------------------------------------------------------------- Task 2
def pack_stream(docs: Iterable[dict], encode: Callable[[str], List[int]], seq_len: int,
                eot_id: int, template: str = "{prompt}\n{response}",
                mask_prompt: bool = True) -> Iterator[dict]:
    """Concat-and-chunk SFT packing with optional prompt masking.

    Encode the prompt part and the response part separately, append ``eot_id`` after
    every document, cut the stream into rows of ``seq_len + 1`` tokens that overlap by
    one token, and emit input_ids / labels / doc_ids. Drop the final partial row.
    """
    raise NotImplementedError("TODO: Task 2: encode prompt and response separately, add EOT, emit overlapping rows of seq_len+1 tokens, mask prompt targets.")


# --------------------------------------------------------------------------- Task 3
def debiased_preference(judge_fn, instruction: str, out_model: str, out_ref: str) -> float:
    """Average P(model wins) over both presentation orders."""
    raise NotImplementedError("TODO: Task 3: average judge_fn over both presentation orders.")


# --------------------------------------------------------------------------- Task 4
def safety_report(records: Sequence[dict]) -> dict:
    """Unsafe-compliance rate on harmful prompts and over-refusal rate on benign ones."""
    raise NotImplementedError("TODO: Task 4: unsafe rate on harmful prompts, over-refusal rate on benign prompts, Wilson intervals.")


# --------------------------------------------------------------------------- Task 5
def splice_image_embeddings(tok_embeds, input_ids, image_feats, image_token_id):
    """Write image features (M, N, d) into the <image> slots of tok_embeds (B, T, d)."""
    raise NotImplementedError("TODO: Task 5: write image features into the <image> slots (check the slot count).")


# --------------------------------------------------------------------------- Task 6
def clip_loss(img_emb, txt_emb, logit_scale):
    raise NotImplementedError("TODO: Task 6a: symmetric InfoNCE over the normalized similarity matrix.")


def siglip_loss(img_emb, txt_emb, logit_scale, logit_bias):
    raise NotImplementedError("TODO: Task 6b: per-cell sigmoid loss, +1 on the diagonal and -1 elsewhere, divided by batch size.")


# --------------------------------------------------------------------------- Task 7
def anyres_tokens(height: int, width: int, tile: int = 336, patch: int = 14,
                  max_tiles: int = 6, include_base: bool = True) -> int:
    """Total vision tokens for AnyRes tiling (grid capped at max_tiles, + overview)."""
    raise NotImplementedError("TODO: Task 7: tile grid capped at max_tiles (shrink the longer side), plus an overview tile.")


# --------------------------------------------------------------------------- Task 8
def multimodal_block_hashes(token_ids, block_size, image_token_id, image_hashes):
    """Chained block hashes where each <image> placeholder carries its image's hash."""
    raise NotImplementedError("TODO: Task 8: tag each <image> run with its image hash, then chain block hashes.")


# --------------------------------------------------------------------------- Task 9
class ImageEmbeddingCache:
    """LRU cache of image features keyed by content hash, bounded in bytes."""

    def __init__(self, capacity_bytes: int):
        self.capacity, self.used = capacity_bytes, 0
        self.store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.hits = self.misses = 0

    def get_or_compute(self, img, encode_fn):
        raise NotImplementedError("TODO: Task 9: LRU lookup by image_hash; on a miss encode, evict oldest until it fits, insert.")


# --------------------------------------------------------------------------- synthetic VLM task
COLORS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "yellow": (1, 1, 0)}
SHAPES = ["square", "hbar", "vbar", "dot"]
PAD, BOS, EOS, IMG = 0, 1, 2, 3
WORD = {w: 4 + i for i, w in enumerate(list(COLORS) + SHAPES)}


def draw(color: str, shape: str, g: torch.Generator, size: int = 8) -> torch.Tensor:
    img = torch.zeros(3, size, size)
    r, c = torch.randint(1, size - 3, (2,), generator=g).tolist()
    sl = {"square": (slice(r, r + 3), slice(c, c + 3)),
          "hbar": (slice(r, r + 1), slice(0, size)),
          "vbar": (slice(0, size), slice(c, c + 1)),
          "dot": (slice(r, r + 1), slice(c, c + 1))}[shape]
    for ch, v in enumerate(COLORS[color]):
        img[ch][sl] = v
    return img + 0.05 * torch.randn(3, size, size, generator=g)


def make_batch(n: int, g: torch.Generator, size: int = 8, patch: int = 4):
    """Images + token rows [BOS, <image> x N, color, shape, EOS] + caption loss mask."""
    n_img = (size // patch) ** 2
    cs = torch.randint(0, 4, (n,), generator=g).tolist()
    ss = torch.randint(0, 4, (n,), generator=g).tolist()
    names = list(COLORS)
    imgs = torch.stack([draw(names[c], SHAPES[s], g, size) for c, s in zip(cs, ss)])
    ids = torch.tensor([[BOS] + [IMG] * n_img + [WORD[names[c]], WORD[SHAPES[s]], EOS]
                        for c, s in zip(cs, ss)])
    mask = torch.zeros(ids.shape)
    mask[:, n_img + 1:] = 1.0
    return imgs, ids, mask


def build_tiny_vlm(size: int = 8, patch: int = 4, dim: int = 32, seed: int = 0) -> TinyVLM:
    torch.manual_seed(seed)
    cfg = dict(QWEN3_CONFIG_TINY, vocab_size=16, context_length=32, emb_dim=dim, n_heads=2,
               n_kv_groups=1, head_dim=16, hidden_dim=2 * dim, n_layers=1,
               tie_embeddings=False)
    vit = TinyViT(img_size=size, patch=patch, in_ch=3, dim=dim, depth=1, heads=2)
    return TinyVLM(Qwen3Model(cfg), vit, dim, IMG)


def train_vlm(vlm: TinyVLM, steps: int, stage: int = 2, batch: int = 32, lr: float = 3e-3,
              seed: int = 1, size: int = 8, patch: int = 4) -> List[float]:
    """Caption-loss training; stage 1 trains the projector only, stage 2 also the LM."""
    vlm.freeze_for_stage(stage)
    opt = torch.optim.AdamW([p for p in vlm.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        imgs, ids, mask = make_batch(batch, g, size, patch)
        loss = caption_loss(vlm(ids, images=imgs), ids, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def caption_accuracy(vlm: TinyVLM, n: int = 128, seed: int = 99, size: int = 8,
                     patch: int = 4) -> float:
    imgs, ids, _ = make_batch(n, torch.Generator().manual_seed(seed), size, patch)
    n_img = (size // patch) ** 2
    pred = vlm(ids, images=imgs)[:, n_img:n_img + 2].argmax(-1)
    return float((pred == ids[:, n_img + 1:n_img + 3]).all(-1).float().mean())
