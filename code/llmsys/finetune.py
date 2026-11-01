"""Fine-tuning utilities (Chapters 9 and 10).

Classification fine-tuning
    balance_by_undersampling, split_indices, pad_batch, ClassificationDataset,
    classification_collate, add_classification_head, select_class_logits,
    classification_loss, accuracy_loader, length_bucketed_batches, padding_stats
Instruction fine-tuning
    format_alpaca, format_phi3, InstructionDataset, instruction_collate,
    pack_sequences, gpt_forward_packed, packed_loss, extract_response,
    build_judge_prompt, parse_score, query_ollama
"""
from __future__ import annotations

import json
import math
import random
import re
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .layers import _attend, causal_mask

IGNORE_INDEX = -100


# =========================================================================== data prep
def balance_by_undersampling(labels: Sequence[int], seed: int = 123) -> List[int]:
    """Return indices that keep every example of the rarest class and an equal-sized
    random subset of each other class (the strategy used for the SMS spam data)."""
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    n_min = min(len(v) for v in by_class.values())
    rng = random.Random(seed)
    keep: List[int] = []
    for y in sorted(by_class):
        idx = by_class[y]
        keep.extend(idx if len(idx) == n_min else rng.sample(idx, n_min))
    rng.shuffle(keep)
    return keep


def split_indices(n: int, train_frac: float = 0.7, val_frac: float = 0.1,
                  seed: int = 123) -> Tuple[List[int], List[int], List[int]]:
    """Shuffle 0..n-1 and cut into train / validation / test (test = remainder)."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    a = int(n * train_frac)
    b = a + int(n * val_frac)
    return idx[:a], idx[a:b], idx[b:]


def pad_batch(seqs: Sequence[Sequence[int]], pad_id: int, side: str = "right",
              max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad (and optionally truncate) to a rectangle. Returns (ids, lengths)."""
    if max_length is not None:
        seqs = [list(s)[:max_length] for s in seqs]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        t = torch.as_tensor(list(s), dtype=torch.long)
        if side == "right":
            ids[i, : len(s)] = t
        else:
            ids[i, T - len(s):] = t
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    return ids, lengths


class ClassificationDataset(Dataset):
    """Holds token-id lists + integer labels. Padding happens in the collate function,
    so each batch is only as wide as its longest member (dynamic padding)."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer,
                 max_length: Optional[int] = None):
        self.encoded = [tokenizer.encode(t)[:max_length] if max_length else tokenizer.encode(t)
                        for t in texts]
        self.labels = [int(y) for y in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.encoded[i], self.labels[i]


def classification_collate(batch, pad_id: int, side: str = "right"):
    seqs, labels = zip(*batch)
    ids, lengths = pad_batch(seqs, pad_id, side)
    return ids, lengths, torch.tensor(labels, dtype=torch.long)


# =========================================================================== heads
def add_classification_head(model: nn.Module, num_classes: int, trainable_blocks: int = 1,
                            train_final_norm: bool = True) -> nn.Module:
    """Swap the vocabulary projection for a ``num_classes`` projection and freeze the
    rest of the network except the last ``trainable_blocks`` transformer blocks."""
    for p in model.parameters():
        p.requires_grad = False
    emb_dim = model.out_head.in_features
    model.out_head = nn.Linear(emb_dim, num_classes)  # new params are trainable
    if trainable_blocks > 0:
        for blk in model.trf_blocks[-trainable_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    if train_final_norm:
        for p in model.final_norm.parameters():
            p.requires_grad = True
    return model


def select_class_logits(logits: torch.Tensor, lengths: Optional[torch.Tensor] = None,
                        side: str = "right") -> torch.Tensor:
    """Pick the logits of the token that has 'seen' the whole message.

    logits: (B, T, C). With right padding and ``lengths`` given we gather the last *real*
    token; without lengths (or with left padding) we take position -1.
    """
    if lengths is None or side == "left":
        return logits[:, -1, :]
    idx = (lengths - 1).clamp(min=0).to(logits.device)
    return logits[torch.arange(logits.shape[0], device=logits.device), idx]


def classification_loss(model, ids, lengths, labels, device="cpu", use_lengths=True):
    ids, labels = ids.to(device), labels.to(device)
    logits = select_class_logits(model(ids), lengths if use_lengths else None)
    return F.cross_entropy(logits.float(), labels)


@torch.no_grad()
def accuracy_loader(model, loader, device="cpu", num_batches: Optional[int] = None,
                    use_lengths: bool = True) -> float:
    model.eval()
    correct = total = 0
    for i, (ids, lengths, labels) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        logits = select_class_logits(model(ids.to(device)), lengths if use_lengths else None)
        correct += (logits.argmax(-1).cpu() == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


# =========================================================================== batching
def length_bucketed_batches(lengths: Sequence[int], batch_size: int, seed: int = 0,
                            bucket_mult: int = 50) -> List[List[int]]:
    """Sort-within-mega-batch bucketing: shuffle, cut into chunks of batch_size*bucket_mult,
    sort each chunk by length, slice into batches, shuffle batch order.

    Keeps randomness (needed for SGD) while making each batch's lengths similar."""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    mega = batch_size * bucket_mult
    batches: List[List[int]] = []
    for s in range(0, len(idx), mega):
        chunk = sorted(idx[s: s + mega], key=lambda i: lengths[i])
        batches.extend(chunk[j: j + batch_size] for j in range(0, len(chunk), batch_size))
    rng.shuffle(batches)
    return batches


def padding_stats(lengths: Sequence[int], batches: Iterable[Sequence[int]]) -> dict:
    """Fraction of computed token slots that are padding when each batch is padded to its max."""
    real = slots = 0
    for b in batches:
        L = [lengths[i] for i in b]
        real += sum(L)
        slots += max(L) * len(L)
    return dict(real_tokens=real, padded_slots=slots, waste=1 - real / max(slots, 1))


# =========================================================================== instruction data
def format_alpaca(entry: dict, with_response: bool = False) -> str:
    """Alpaca-style prompt (instruction, optional input, response header)."""
    text = ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}")
    if entry.get("input"):
        text += f"\n\n### Input:\n{entry['input']}"
    text += "\n\n### Response:\n"
    if with_response:
        text += entry["output"]
    return text


def format_phi3(entry: dict, with_response: bool = False) -> str:
    """Phi-3 style chat markup; shorter prompts -> fewer tokens per example."""
    user = entry["instruction"] + (f"\n{entry['input']}" if entry.get("input") else "")
    text = f"<|user|>\n{user}\n<|assistant|>\n"
    if with_response:
        text += entry["output"]
    return text


class InstructionDataset(Dataset):
    """Pre-tokenises prompt+response once. Items are (ids, prompt_len)."""

    def __init__(self, data: Sequence[dict], tokenizer, fmt=format_alpaca):
        self.items = []
        for e in data:
            prompt = tokenizer.encode(fmt(e))
            full = tokenizer.encode(fmt(e, with_response=True))
            self.items.append((full, len(prompt)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def instruction_collate(batch, pad_id: int, ignore_index: int = IGNORE_INDEX,
                        allowed_max_length: Optional[int] = None, mask_prompt: bool = False,
                        device="cpu"):
    """Turn a list of token lists (or (ids, prompt_len) pairs) into (inputs, targets).

    1. append one end-of-text token (pad_id doubles as EOT) so the model learns to stop
    2. pad every row to the batch max (+1 so we can shift)
    3. inputs = row[:-1], targets = row[1:]
    4. in targets, keep the first EOT but replace all later padding with ignore_index
    5. optionally replace prompt tokens in targets with ignore_index (train on responses only)
    6. optionally truncate to allowed_max_length
    """
    items = [(b, 0) if not isinstance(b, tuple) else b for b in batch]
    max_len = max(len(ids) + 1 for ids, _ in items)
    inputs, targets = [], []
    for ids, prompt_len in items:
        row = list(ids) + [pad_id]
        row = row + [pad_id] * (max_len - len(row))
        x = torch.tensor(row[:-1], dtype=torch.long)
        y = torch.tensor(row[1:], dtype=torch.long)
        n_real = len(ids)                     # targets[n_real-1] is the appended EOT
        y[n_real:] = ignore_index             # everything after the first EOT is padding
        if mask_prompt and prompt_len > 0:
            y[: prompt_len - 1] = ignore_index  # target j predicts token j+1
        if allowed_max_length is not None:
            x, y = x[:allowed_max_length], y[:allowed_max_length]
        inputs.append(x)
        targets.append(y)
    return torch.stack(inputs).to(device), torch.stack(targets).to(device)


# =========================================================================== packing
def pack_sequences(seqs: Sequence[Sequence[int]], max_len: int, pad_id: int,
                   ignore_index: int = IGNORE_INDEX):
    """Greedy first-fit packing of (already EOT-terminated) sequences into rows of max_len.

    Returns dict of (R, max_len) tensors: input_ids, targets, segment_ids, position_ids.
    segment_ids is 0 for padding and 1, 2, ... for each packed document; position_ids
    restart at 0 for each document; targets never cross a document boundary.
    Every sequence must have length <= max_len + 1 (inputs and targets are shifted).
    """
    rows: List[List[List[int]]] = []
    space: List[int] = []
    for s in sorted(seqs, key=len, reverse=True):   # first-fit decreasing
        need = len(s) - 1
        if need > max_len:
            raise ValueError("sequence longer than max_len + 1")
        for r, free in enumerate(space):
            if free >= need:
                rows[r].append(list(s)); space[r] -= need
                break
        else:
            rows.append([list(s)]); space.append(max_len - need)
    R = len(rows)
    out = dict(input_ids=torch.full((R, max_len), pad_id, dtype=torch.long),
               targets=torch.full((R, max_len), ignore_index, dtype=torch.long),
               segment_ids=torch.zeros((R, max_len), dtype=torch.long),
               position_ids=torch.zeros((R, max_len), dtype=torch.long))
    for r, docs in enumerate(rows):
        c = 0
        for d, s in enumerate(docs, start=1):
            n = len(s) - 1
            out["input_ids"][r, c:c + n] = torch.tensor(s[:-1])
            out["targets"][r, c:c + n] = torch.tensor(s[1:])
            out["segment_ids"][r, c:c + n] = d
            out["position_ids"][r, c:c + n] = torch.arange(n)
            c += n
    return out


def segment_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """(B,1,T,T) bool mask, True = forbidden: causal AND same-document only."""
    T = segment_ids.shape[1]
    causal = causal_mask(T, T, segment_ids.device)[None, None]
    diff = segment_ids[:, None, :, None] != segment_ids[:, None, None, :]
    return causal | diff


def gpt_forward_packed(model, input_ids, position_ids, segment_ids):
    """Forward a core ``GPTModel`` over packed rows with block-diagonal causal attention
    and per-document positions (what FlashAttention's varlen kernels do natively)."""
    x = model.tok_emb(input_ids) + model.pos_emb(position_ids)
    x = model.drop_emb(x)
    mask = segment_mask(segment_ids)
    for blk in model.trf_blocks:
        att = blk.att
        h = blk.norm1(x)
        B, T, _ = h.shape
        def heads(lin):
            return lin(h).view(B, T, att.num_heads, att.head_dim).transpose(1, 2)
        ctx = _attend(heads(att.W_query), heads(att.W_key), heads(att.W_value), mask,
                      att.dropout, att.training, False)
        ctx = ctx.transpose(1, 2).reshape(B, T, att.d_out)
        x = x + blk.drop_shortcut(att.out_proj(ctx))
        x = x + blk.drop_shortcut(blk.ff(blk.norm2(x)))
    return model.out_head(model.final_norm(x))


def packed_loss(model, packed: dict, ignore_index: int = IGNORE_INDEX):
    logits = gpt_forward_packed(model, packed["input_ids"], packed["position_ids"],
                                packed["segment_ids"])
    return F.cross_entropy(logits.flatten(0, 1).float(), packed["targets"].flatten(),
                           ignore_index=ignore_index)


# =========================================================================== evaluation
def extract_response(generated_text: str, marker: str = "### Response:") -> str:
    """Return the text after the last response marker, trimmed and cut at the next header."""
    tail = generated_text.split(marker)[-1]
    tail = tail.split("### Instruction:")[0]
    return tail.replace("<|endoftext|>", "").strip()


def build_judge_prompt(entry: dict, model_response: str, scale: int = 100) -> str:
    return (f"You are grading a model's answer.\n\n"
            f"Instruction:\n{entry['instruction']}\n"
            + (f"Input:\n{entry['input']}\n" if entry.get("input") else "")
            + f"Reference answer:\n{entry['output']}\n\n"
            f"Model answer:\n{model_response}\n\n"
            f"Score the model answer from 0 to {scale}, where {scale} is fully correct and "
            f"as good as the reference. Respond with the integer only.")


def parse_score(text: str, lo: int = 0, hi: int = 100) -> Optional[int]:
    """Pull the first integer out of a judge reply and clamp it; None if absent."""
    m = re.search(r"-?\d+", text)
    if not m:
        return None
    return max(lo, min(hi, int(m.group())))


def query_ollama(prompt: str, model: str = "llama3", url: str = "http://localhost:11434/api/chat",
                 seed: int = 123, temperature: float = 0.0, num_ctx: int = 2048) -> str:
    """Minimal client for a locally running Ollama server (streamed JSON lines)."""
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "options": {"seed": seed, "temperature": temperature, "num_ctx": num_ctx}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    chunks = []
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if not line.strip():
                continue
            msg = json.loads(line)
            chunks.append(msg.get("message", {}).get("content", ""))
            if msg.get("done"):
                break
    return "".join(chunks)
