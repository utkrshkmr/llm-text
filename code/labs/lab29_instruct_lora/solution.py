"""Lab 29 reference solution: instruction data pipeline, packing, LoRA, QLoRA-style NF4.

Core implementations live in ``llmsys/finetune.py`` and ``llmsys/lora.py`` (re-exported
here); this file adds a byte tokenizer, a toy instruction task and the fine-tuning loop.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from llmsys.finetune import (IGNORE_INDEX, build_judge_prompt, extract_response,  # noqa: F401
                             format_alpaca, format_phi3, gpt_forward_packed,
                             instruction_collate, pack_sequences, packed_loss, parse_score,
                             segment_mask)
from llmsys.lora import (LoRALinear, MultiLoRALinear, NF4_LEVELS, NF4LoRALinear,  # noqa: F401
                         apply_lora, count_trainable, dequantize_nf4, lora_param_count,
                         lora_state_dict, merge_lora, nf4_levels, quantize_nf4)


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


# ----------------------------------------------------------------- fine-tuning loop
def finetune(model, batches, steps: int = 50, lr: float = 3e-3, ignore_index: int = IGNORE_INDEX):
    """Plain AdamW loop over a list of (inputs, targets) batches, cycling as needed.
    Only parameters with requires_grad=True are optimised. Returns the loss curve."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    losses = []
    for step in range(steps):
        x, y = batches[step % len(batches)]
        logits = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), ignore_index=ignore_index)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
