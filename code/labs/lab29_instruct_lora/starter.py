"""Lab 29 starter: instruction data pipeline, packing, LoRA, QLoRA-style NF4.

Implement every function/method that raises NotImplementedError, then run
    LAB_IMPL=starter python -m pytest -q labs/lab29_instruct_lora
"""
from __future__ import annotations

import math
import random
import re
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmsys.layers import _attend, causal_mask

IGNORE_INDEX = -100


# ----------------------------------------------------------------- Task 1: prompt formats
def format_alpaca(entry: dict, with_response: bool = False) -> str:
    """Return:
    'Below is an instruction that describes a task. Write a response that appropriately
    completes the request.' + '\n\n### Instruction:\n{instruction}'
    + ('\n\n### Input:\n{input}' only if input is non-empty) + '\n\n### Response:\n'
    + (output if with_response)."""
    raise NotImplementedError("TODO Task 1")


def format_phi3(entry: dict, with_response: bool = False) -> str:
    user = entry["instruction"] + (f"\n{entry['input']}" if entry.get("input") else "")
    text = f"<|user|>\n{user}\n<|assistant|>\n"
    return text + entry["output"] if with_response else text


# ----------------------------------------------------------------- Task 2: collate
def instruction_collate(batch, pad_id: int, ignore_index: int = IGNORE_INDEX,
                        allowed_max_length: Optional[int] = None, mask_prompt: bool = False,
                        device="cpu"):
    """batch: list of token lists, or of (token list, prompt_len) tuples.
    Steps: append one EOT (=pad_id); pad to batch max (+1); inputs=row[:-1], targets=row[1:];
    in targets keep the FIRST EOT, set later padding to ignore_index; if mask_prompt, set
    targets that predict prompt tokens to ignore_index; truncate to allowed_max_length.
    Return (inputs, targets) LongTensors on `device`."""
    raise NotImplementedError("TODO Task 2")


# ----------------------------------------------------------------- Task 3: packing
def pack_sequences(seqs: Sequence[Sequence[int]], max_len: int, pad_id: int,
                   ignore_index: int = IGNORE_INDEX):
    """First-fit-decreasing packing of EOT-terminated sequences. A sequence s occupies
    len(s)-1 slots (inputs s[:-1], targets s[1:]). Return a dict of (R, max_len) tensors:
    input_ids (pad_id fill), targets (ignore_index fill), segment_ids (0 = pad, 1.. = doc),
    position_ids (restart at 0 per document)."""
    raise NotImplementedError("TODO Task 3")


def segment_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """(B, T) -> (B, 1, T, T) bool, True = forbidden. Forbid future keys (causal) and keys
    from a different segment."""
    raise NotImplementedError("TODO Task 3")


def gpt_forward_packed(model, input_ids, position_ids, segment_ids):
    """Given: GPTModel forward with per-document positions and block-diagonal attention."""
    x = model.drop_emb(model.tok_emb(input_ids) + model.pos_emb(position_ids))
    mask = segment_mask(segment_ids)
    for blk in model.trf_blocks:
        att = blk.att
        h = blk.norm1(x)
        B, T, _ = h.shape
        def heads(lin):
            return lin(h).view(B, T, att.num_heads, att.head_dim).transpose(1, 2)
        ctx = _attend(heads(att.W_query), heads(att.W_key), heads(att.W_value), mask,
                      att.dropout, att.training, False)
        x = x + blk.drop_shortcut(att.out_proj(ctx.transpose(1, 2).reshape(B, T, att.d_out)))
        x = x + blk.drop_shortcut(blk.ff(blk.norm2(x)))
    return model.out_head(model.final_norm(x))


def packed_loss(model, packed: dict, ignore_index: int = IGNORE_INDEX):
    logits = gpt_forward_packed(model, packed["input_ids"], packed["position_ids"],
                                packed["segment_ids"])
    return F.cross_entropy(logits.flatten(0, 1).float(), packed["targets"].flatten(),
                           ignore_index=ignore_index)


# ----------------------------------------------------------------- Task 4: LoRA
class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * x A^T B^T with A (r, d_in) kaiming-init, B (d_out, r) zeros."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0, scale_by_rank: bool = True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank if scale_by_rank else alpha
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        raise NotImplementedError("TODO Task 4: scaling * B @ A, shape (d_out, d_in)")

    def forward(self, x):
        raise NotImplementedError("TODO Task 4: base path + low-rank path (skip if merged)")

    @torch.no_grad()
    def merge(self):
        raise NotImplementedError("TODO Task 4: add delta into base.weight once")

    @torch.no_grad()
    def unmerge(self):
        raise NotImplementedError("TODO Task 4: subtract delta again")


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0,
               target: Optional[Callable[[str, nn.Linear], bool]] = None,
               scale_by_rank: bool = True, freeze_rest: bool = True) -> nn.Module:
    """Freeze the model (if freeze_rest), then recursively replace each nn.Linear (matching
    target(name, module) if given) by LoRALinear. Return the model."""
    raise NotImplementedError("TODO Task 4")


def count_trainable(model: nn.Module):
    """Return (trainable, total) parameter counts, de-duplicating shared tensors."""
    raise NotImplementedError("TODO Task 4")


class MultiLoRALinear(nn.Module):
    """One frozen base, N adapters stacked in A (N, r, d_in) and B (N, d_out, r)."""

    def __init__(self, base: nn.Linear, n_adapters: int, rank: int, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.zeros(n_adapters, rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(n_adapters, base.out_features, rank))

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """x (B, T, d_in); adapter_ids (B,), -1 = base model only. Gather each row's A/B and
        use two batched matmuls (torch.bmm). Do not loop over rows."""
        raise NotImplementedError("TODO Task 5")


# ----------------------------------------------------------------- Task 6: NF4
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


def nf4_levels(offset: float = 0.9677083) -> torch.Tensor:
    normal = torch.distributions.Normal(0.0, 1.0)
    pos = normal.icdf(torch.linspace(offset, 0.5, 9)[:-1])
    neg = -normal.icdf(torch.linspace(offset, 0.5, 8)[:-1])
    v = torch.cat([pos, torch.zeros(1), neg]).sort().values
    return v / v.abs().max()


def quantize_nf4(w: torch.Tensor, block: int = 64):
    """Flatten, zero-pad to a multiple of `block`, per-block absmax scale, nearest NF4 level.
    Return (codes uint8, absmax float (n_blocks,), original shape)."""
    raise NotImplementedError("TODO Task 6")


def dequantize_nf4(codes, absmax, shape, dtype=torch.float32):
    raise NotImplementedError("TODO Task 6")


# ----------------------------------------------------------------- Task 7: evaluation helpers
def extract_response(generated_text: str, marker: str = "### Response:") -> str:
    """Text after the last marker, cut at any following '### Instruction:', with
    '<|endoftext|>' removed and whitespace stripped."""
    raise NotImplementedError("TODO Task 7")


def parse_score(text: str, lo: int = 0, hi: int = 100):
    """First integer in text clamped to [lo, hi]; None if there is none."""
    raise NotImplementedError("TODO Task 7")


# ----------------------------------------------------------------- scaffolding (given)
class ByteTokenizer:
    """UTF-8 bytes as token ids 0..255, EOT = 256. Enough for CPU-sized experiments."""
    eot_id = 256

    def encode(self, text: str):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def make_toy_instructions(n: int = 200, seed: int = 0):
    """Tiny instruction-following tasks with a checkable answer."""
    rng = random.Random(seed)
    words = ["cat", "dog", "sun", "map", "red", "box", "pen", "sky", "owl", "fig"]
    data = []
    for _ in range(n):
        w = rng.choice(words)
        kind = rng.randrange(3)
        if kind == 0:
            data.append(dict(instruction="Reverse the word.", input=w, output=w[::-1]))
        elif kind == 1:
            data.append(dict(instruction="Uppercase the word.", input=w, output=w.upper()))
        else:
            data.append(dict(instruction=f"Repeat {w} twice.", input="", output=f"{w} {w}"))
    return data


# ----------------------------------------------------------------- Task 8: fine-tuning loop
def finetune(model, batches, steps: int = 50, lr: float = 3e-3, ignore_index: int = IGNORE_INDEX):
    """AdamW (weight_decay=0) over trainable params, cycling over (inputs, targets) batches,
    cross-entropy with ignore_index. Return the list of per-step losses."""
    raise NotImplementedError("TODO Task 8")
