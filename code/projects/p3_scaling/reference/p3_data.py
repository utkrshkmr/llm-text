"""Project 3 (reference solution): token data for the sweep (Chapter 29).

Two sources behind one interface (a ``uint16`` token array plus a held-out tail):

* ``--synthetic``: a random sparse Markov chain over a small vocabulary. It has real
  structure (a model can lower its loss) and a known entropy, needs no download, and lets the
  whole pipeline run on a CPU in minutes.
* real text: a streamed sample of a public web corpus through ``datasets``, tokenised with
  the GPT-2 BPE (``tiktoken``) and written as a memory-mapped ``.bin`` file. ``n_tokens`` is
  the only knob: the smoke configuration takes ~1e8 tokens, the full one ~2e9.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional, Tuple

import numpy as np


def synthetic_tokens(n_tokens: int, vocab: int = 1024, order_sparsity: int = 8,
                     seed: int = 0) -> Tuple[np.ndarray, float]:
    """First-order Markov chain: each token has ``order_sparsity`` likely successors with
    Dirichlet weights. Returns the tokens and the chain's entropy rate in nats (the loss floor a
    perfect model reaches)."""
    rng = np.random.default_rng(seed)
    probs = np.zeros((vocab, vocab), dtype=np.float64)
    for v in range(vocab):
        succ = rng.choice(vocab, order_sparsity, replace=False)
        probs[v, succ] = rng.dirichlet(np.full(order_sparsity, 0.5))
    cdf = np.cumsum(probs, axis=1)
    toks = np.empty(n_tokens, dtype=np.uint16)
    cur = int(rng.integers(vocab))
    u = rng.random(n_tokens)
    for i in range(n_tokens):
        toks[i] = cur
        cur = int(np.searchsorted(cdf[cur], u[i]))
        if cur >= vocab:
            cur = vocab - 1
    # stationary distribution by power iteration, then entropy rate
    pi = np.full(vocab, 1.0 / vocab)
    for _ in range(200):
        pi = pi @ probs
    with np.errstate(divide="ignore", invalid="ignore"):
        h_rows = -np.nansum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    return toks, float(pi @ h_rows)


def write_bin(tokens: np.ndarray, path: str, meta: Optional[dict] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.asarray(tokens, dtype=np.uint16).tofile(path)
    with open(path + ".json", "w") as f:
        json.dump(dict(meta or {}, n_tokens=int(len(tokens)), dtype="uint16"), f, indent=1)
    return path


def load_bin(path: str, val_frac: float = 0.02) -> Tuple[np.ndarray, np.ndarray, dict]:
    toks = np.memmap(path, dtype=np.uint16, mode="r")
    meta = json.load(open(path + ".json")) if os.path.exists(path + ".json") else {}
    split = len(toks) - max(1024, int(val_frac * len(toks)))
    return toks[:split], toks[split:], meta


def prepare_real(path: str, n_tokens: int, dataset: str = "HuggingFaceFW/fineweb-edu",
                 name: Optional[str] = "sample-10BT", split: str = "train",
                 text_field: str = "text", encoding: str = "gpt2", seed: int = 0) -> str:
    """Stream documents, tokenise with tiktoken, append an end-of-text token per document,
    stop at ``n_tokens``. Everything is written through a memmap, so the corpus never has to
    fit in RAM."""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding(encoding)
    ds = load_dataset(dataset, name=name, split=split, streaming=True).shuffle(seed=seed, buffer_size=10_000)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = np.memmap(path, dtype=np.uint16, mode="w+", shape=(n_tokens,))
    pos, docs = 0, 0
    for ex in ds:
        ids = enc.encode_ordinary(ex[text_field]) + [enc.eot_token]
        take = min(len(ids), n_tokens - pos)
        out[pos:pos + take] = np.asarray(ids[:take], dtype=np.uint16)
        pos += take
        docs += 1
        if pos >= n_tokens:
            break
    out.flush()
    with open(path + ".json", "w") as f:
        json.dump(dict(dataset=dataset, name=name, split=split, encoding=encoding, vocab=enc.n_vocab,
                       n_tokens=int(pos), documents=docs, seed=seed, dtype="uint16"), f, indent=1)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="path of the .bin token file")
    ap.add_argument("--tokens", type=float, required=True, help="number of tokens to write")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--vocab", type=int, default=1024, help="synthetic vocabulary")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    n = int(a.tokens)
    if a.synthetic:
        toks, h = synthetic_tokens(n, a.vocab, seed=a.seed)
        print(write_bin(toks, a.out, dict(source="synthetic-markov", vocab=a.vocab, entropy_nats=h)))
        print(f"entropy rate {h:.4f} nats (ln V = {math.log(a.vocab):.4f})")
    else:
        print(prepare_real(a.out, n, a.dataset, a.name, seed=a.seed))


if __name__ == "__main__":
    main()
