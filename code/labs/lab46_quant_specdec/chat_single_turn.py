"""Single-turn chat UI: every message is answered independently.

    pip install chainlit
    QWEN3_DIR=qwen3-0.6b chainlit run chat_single_turn.py      # opens http://localhost:8000

The browser shows the whole conversation, but the model only ever sees the latest
message -- ask "what did I just ask?" to convince yourself.
"""
import chainlit as cl
import torch

from chat_common import (DEVICE, THINKING, IncrementalDetokenizer, load_model_and_tokenizer,
                         stop_token_ids, stream_tokens)

MODEL, TOKENIZER = load_model_and_tokenizer()   # loaded once per server process
STOP = stop_token_ids(TOKENIZER)


@cl.on_chat_start
async def on_start():
    await cl.Message(content="Single-turn mode: each question is answered on its own.").send()


@cl.on_message
async def on_message(message: cl.Message):
    prompt = TOKENIZER.apply_chat_template(message.content, enable_thinking=THINKING)
    ids = torch.tensor([TOKENIZER.encode(prompt)], device=DEVICE)
    reply = cl.Message(content="")
    await reply.send()                            # an empty bubble we stream into
    detok = IncrementalDetokenizer(TOKENIZER)
    async for tok in stream_tokens(MODEL, ids, STOP):
        if tok in STOP:
            break
        piece = detok.push(tok)
        if piece:
            await reply.stream_token(piece)
    await reply.update()
