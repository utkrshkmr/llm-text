"""Shared plumbing for the Chainlit chat apps (Chapter 48, section on chat interfaces).

Configuration is read from environment variables because ``chainlit run app.py`` owns
the command line:

    QWEN3_DIR        directory of an HF snapshot (default: ./qwen3-0.6b)
    QWEN3_SIZE       config name in llmsys.qwen3.QWEN3_CONFIGS (default: 0.6B)
    CHECKPOINT_PATH  optional .pth state_dict from your own fine-tuning/RL/distillation run
    MAX_NEW_TOKENS   generation budget per reply (default: 2048)
    THINKING         "1" to let the model emit <think> ... </think> (default: 1)
    QUANT_BITS       "8" or "4" to serve weight-only quantized linears (default: off)
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import AsyncIterator, Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from llmsys.generate import generate_stream  # noqa: E402
from llmsys.kvcache import KVCache  # noqa: E402
from llmsys.qwen3 import QWEN3_CONFIGS, Qwen3Model, Qwen3Tokenizer, load_qwen3_from_dir  # noqa: E402
from llmsys.utils import get_device  # noqa: E402

QWEN3_DIR = os.getenv("QWEN3_DIR", "qwen3-0.6b")
QWEN3_SIZE = os.getenv("QWEN3_SIZE", "0.6B")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048"))
THINKING = os.getenv("THINKING", "1") == "1"
QUANT_BITS = os.getenv("QUANT_BITS")
DEVICE = get_device()
DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


def load_model_and_tokenizer():
    """Official weights by default; a custom state_dict if CHECKPOINT_PATH is set."""
    tokenizer = Qwen3Tokenizer(os.path.join(QWEN3_DIR, "tokenizer.json"))
    if CHECKPOINT_PATH:
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(CHECKPOINT_PATH)
        model = Qwen3Model(QWEN3_CONFIGS[QWEN3_SIZE], dtype=DTYPE)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        model = model.to(DEVICE).eval()
    else:
        model = load_qwen3_from_dir(QWEN3_DIR, QWEN3_SIZE, dtype=DTYPE, device=DEVICE)
    if QUANT_BITS:
        from solution import quantize_linear_layers
        model = quantize_linear_layers(model, int(QUANT_BITS), 128)
    return model, tokenizer


class IncrementalDetokenizer:
    """Turn a stream of token ids into a stream of *complete* text pieces.

    Decoding one BPE token at a time can split a multi-byte UTF-8 character (emoji,
    CJK, accented letters) across tokens. We decode a sliding window and only emit
    text once it no longer ends in the replacement character U+FFFD.
    """

    def __init__(self, tokenizer, window: int = 8):
        self.tok, self.ids, self.emitted, self.window = tokenizer, [], 0, window

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        start = max(0, self.emitted - self.window)
        prefix = self.tok.decode(self.ids[start:self.emitted])
        full = self.tok.decode(self.ids[start:])
        if full.endswith("�"):
            return ""                       # wait for the rest of the character
        self.emitted = len(self.ids)
        return full[len(prefix):]


async def stream_tokens(model, input_ids: torch.Tensor, stop_ids, cache=None
                        ) -> AsyncIterator[int]:
    """Run blocking generation in a worker thread and hand tokens to the event loop.

    Calling the model directly inside an ``async def`` handler would block Chainlit's
    event loop for the whole reply (no UI updates, no other sessions served).
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def worker():
        try:
            for tok in generate_stream(model, input_ids, MAX_NEW_TOKENS, cache=cache):
                t = int(tok[0, 0])
                loop.call_soon_threadsafe(queue.put_nowait, t)
                if t in stop_ids:
                    break
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is done:
            return
        yield item


def stop_token_ids(tokenizer) -> set:
    return {tokenizer.tok.token_to_id("<|im_end|>"), tokenizer.tok.token_to_id("<|endoftext|>")}


def trim_history(history: List[Dict[str, str]], tokenizer, budget: int) -> List[Dict[str, str]]:
    """Drop the oldest *whole* user/assistant turns (never the system prompt) until the
    serialized prompt fits in ``budget`` tokens. Dropping whole turns keeps the chat
    template well-formed; cutting raw tokens off the left can leave a dangling
    half-message that the model was never trained on."""
    system = [m for m in history[:1] if m["role"] == "system"]
    rest = history[len(system):]
    while rest:
        n = len(tokenizer.encode(tokenizer.apply_chat_template(system + rest, THINKING)))
        if n <= budget:
            break
        rest = rest[2:] if len(rest) > 2 else rest[-1:]
        if len(rest) == 1:
            break
    return system + rest
