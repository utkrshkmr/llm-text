"""Lab 65: generalist post-training plumbing, evaluation at scale, and a tiny VLM.

Reference solution. Every function here has a twin in ``llmsys.alignment`` or
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
    raw = {k: float(n) ** alpha for k, n in sizes.items() if n > 0}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


# --------------------------------------------------------------------------- Task 2
def pack_stream(docs: Iterable[dict], encode: Callable[[str], List[int]], seq_len: int,
                eot_id: int, template: str = "{prompt}\n{response}",
                mask_prompt: bool = True) -> Iterator[dict]:
    """Concat-and-chunk SFT packing with optional prompt masking.

    Encode the prompt part and the response part separately, append ``eot_id`` after
    every document, cut the stream into rows of ``seq_len + 1`` tokens that overlap by
    one token, and emit input_ids / labels / doc_ids. Drop the final partial row.
    """
    head, tail = template.split("{response}")
    buf: List[Tuple[int, bool, int]] = []
    for d, ex in enumerate(docs):
        items = [(t, True, d) for t in encode(head.format(prompt=ex["prompt"]))]
        items += [(t, False, d) for t in encode(ex["response"] + tail)]
        items.append((eot_id, False, d))
        for it in items:
            buf.append(it)
            if len(buf) == seq_len + 1:
                toks = torch.tensor([t for t, _, _ in buf])
                is_p = torch.tensor([p for _, p, _ in buf])
                labels = toks[1:].clone()
                if mask_prompt:
                    labels[is_p[1:]] = IGNORE_INDEX
                yield {"input_ids": toks[:-1], "labels": labels,
                       "doc_ids": torch.tensor([x for _, _, x in buf[:-1]])}
                buf = buf[-1:]


# --------------------------------------------------------------------------- Task 3
def debiased_preference(judge_fn, instruction: str, out_model: str, out_ref: str) -> float:
    """Average P(model wins) over both presentation orders."""
    return 0.5 * (judge_fn(instruction, out_model, out_ref)
                  + 1.0 - judge_fn(instruction, out_ref, out_model))


# --------------------------------------------------------------------------- Task 4
def safety_report(records: Sequence[dict]) -> dict:
    """Unsafe-compliance rate on harmful prompts and over-refusal rate on benign ones."""
    harm = [r for r in records if r["kind"] == "harmful"]
    ben = [r for r in records if r["kind"] == "benign"]
    ku = sum(bool(r["unsafe"]) for r in harm)
    ko = sum(bool(r["refused"]) for r in ben)
    return {"unsafe_rate": ku / max(1, len(harm)), "unsafe_ci95": wilson_interval(ku, len(harm)),
            "over_refusal_rate": ko / max(1, len(ben)),
            "over_refusal_ci95": wilson_interval(ko, len(ben))}


# --------------------------------------------------------------------------- Task 5
def splice_image_embeddings(tok_embeds, input_ids, image_feats, image_token_id):
    """Write image features (M, N, d) into the <image> slots of tok_embeds (B, T, d)."""
    mask = input_ids == image_token_id
    if int(mask.sum()) != image_feats.shape[0] * image_feats.shape[1]:
        raise ValueError("number of <image> slots does not match the image features")
    out = tok_embeds.clone()
    out[mask] = image_feats.reshape(-1, image_feats.shape[-1]).to(out.dtype)
    return out


# --------------------------------------------------------------------------- Task 6
def clip_loss(img_emb, txt_emb, logit_scale):
    i, t = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
    logits = logit_scale.exp() * i @ t.T
    y = torch.arange(len(i))
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def siglip_loss(img_emb, txt_emb, logit_scale, logit_bias):
    i, t = F.normalize(img_emb, dim=-1), F.normalize(txt_emb, dim=-1)
    logits = logit_scale.exp() * i @ t.T + logit_bias
    labels = 2 * torch.eye(len(i)) - 1
    return -F.logsigmoid(labels * logits).sum() / len(i)


# --------------------------------------------------------------------------- Task 7
def anyres_tokens(height: int, width: int, tile: int = 336, patch: int = 14,
                  max_tiles: int = 6, include_base: bool = True) -> int:
    """Total vision tokens for AnyRes tiling (grid capped at max_tiles, + overview)."""
    gh, gw = math.ceil(height / tile), math.ceil(width / tile)
    while gh * gw > max_tiles:
        if gh >= gw:
            gh -= 1
        else:
            gw -= 1
    return (gh * gw + (1 if include_base else 0)) * (tile // patch) ** 2


# --------------------------------------------------------------------------- Task 8
def multimodal_block_hashes(token_ids, block_size, image_token_id, image_hashes):
    """Chained block hashes where each <image> placeholder carries its image's hash."""
    feed, imgs, prev_was_img, cur = [], iter(image_hashes), False, None
    for t in token_ids:
        if t == image_token_id:
            if not prev_was_img:
                cur = next(imgs)
            feed.append(f"img:{cur}")
            prev_was_img = True
        else:
            feed.append(str(t))
            prev_was_img = False
    out, prev = [], ""
    for s in range(0, len(feed) - len(feed) % block_size, block_size):
        prev = hashlib.sha256((prev + "|" + ",".join(feed[s:s + block_size])).encode()
                              ).hexdigest()[:16]
        out.append(prev)
    return out


# --------------------------------------------------------------------------- Task 9
class ImageEmbeddingCache:
    """LRU cache of image features keyed by content hash, bounded in bytes."""

    def __init__(self, capacity_bytes: int):
        self.capacity, self.used = capacity_bytes, 0
        self.store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.hits = self.misses = 0

    def get_or_compute(self, img, encode_fn):
        k = image_hash(img)
        if k in self.store:
            self.hits += 1
            self.store.move_to_end(k)
            return self.store[k]
        self.misses += 1
        f = encode_fn(img[None])[0].detach()
        nb = f.numel() * f.element_size()
        while self.store and self.used + nb > self.capacity:
            _, old = self.store.popitem(last=False)
            self.used -= old.numel() * old.element_size()
        if nb <= self.capacity:
            self.store[k] = f
            self.used += nb
        return f


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
