"""Multi-turn chat UI with per-session history *and* KV-cache reuse across turns.

    QWEN3_DIR=qwen3-0.6b chainlit run chat_multi_turn.py

Each turn the history is re-serialised with the ChatML template. Because the new prompt
usually *extends* the previous one token-for-token, we keep the session's KV cache and
prefill only the new suffix (a one-session version of prefix caching, Chapter 12).
"""
import chainlit as cl
import torch

from chat_common import (DEVICE, MAX_NEW_TOKENS, THINKING, IncrementalDetokenizer, KVCache,
                         load_model_and_tokenizer, stop_token_ids, stream_tokens, trim_history)
from solution import common_prefix_len
from llmsys.specdec import truncate_cache

MODEL, TOKENIZER = load_model_and_tokenizer()
STOP = stop_token_ids(TOKENIZER)
CONTEXT = MODEL.cfg["context_length"]
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


@cl.on_chat_start
async def on_start():
    cl.user_session.set("history", [SYSTEM])
    cl.user_session.set("cache", KVCache(MODEL.n_layers))
    cl.user_session.set("cached_ids", [])        # tokens whose K/V are in the cache


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    history = trim_history(history, TOKENIZER, CONTEXT - MAX_NEW_TOKENS)
    prompt_ids = TOKENIZER.encode(TOKENIZER.apply_chat_template(history, THINKING))

    # ---- reuse the KV cache for the longest common prefix with what we cached last turn
    cache, cached = cl.user_session.get("cache"), cl.user_session.get("cached_ids")
    reuse = min(common_prefix_len(cached, prompt_ids), len(prompt_ids) - 1)
    truncate_cache(cache, reuse)
    new_ids = torch.tensor([prompt_ids[reuse:]], device=DEVICE)

    reply = cl.Message(content="")
    await reply.send()
    detok = IncrementalDetokenizer(TOKENIZER)
    generated = []
    async for tok in stream_tokens(MODEL, new_ids, STOP, cache=cache):
        generated.append(tok)
        if tok in STOP:
            break
        piece = detok.push(tok)
        if piece:
            await reply.stream_token(piece)
    await reply.update()

    # The cache now holds prompt + all generated tokens except the last one fed back.
    cl.user_session.set("cached_ids", (prompt_ids + generated)[: cache.seq_len(0)])
    history.append({"role": "assistant", "content": reply.content})
    cl.user_session.set("history", history)
    await cl.Message(content=f"_(reused {reuse} of {len(prompt_ids)} prompt tokens "
                             f"from the KV cache)_").send()
