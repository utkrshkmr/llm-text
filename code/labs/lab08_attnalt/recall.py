"""Associative-recall probe (MQAR) for different sequence-mixer patterns.

    python recall.py --patterns AA LL MM LA --pairs 8 16 --steps 2500           # CPU, minutes per run
    python recall.py --device cuda --d-model 256 --pairs 64 128 256 --steps 3000

Trains a small HybridLM per (pattern, n_pairs) on next-token prediction where only answer
positions are scored, then reports recall accuracy on fresh sequences.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmsys.linear_attn import HybridLM, make_mqar_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", nargs="+", default=["AA", "LL", "MM", "LA"])
    ap.add_argument("--pairs", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="mqar.json")
    a = ap.parse_args()
    res = []
    for n in a.pairs:
        for pat in a.patterns:
            torch.manual_seed(0)
            m = HybridLM(vocab_size=a.vocab, d_model=a.d_model, n_heads=4, pattern=pat, context_length=4 * n + 8).to(a.device)
            opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.0)
            for step in range(a.steps):
                x, y, mk = make_mqar_batch(a.batch, n, n // 2, vocab=a.vocab, seed=step + 1000)
                x, y, mk = x.to(a.device), y.to(a.device), mk.to(a.device)
                loss = F.cross_entropy(m(x)[mk], y[mk])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                x, y, mk = make_mqar_batch(256, n, n // 2, vocab=a.vocab, seed=123)
                x, y, mk = x.to(a.device), y.to(a.device), mk.to(a.device)
                acc = (m(x).argmax(-1)[mk] == y[mk]).float().mean().item()
            res.append(dict(pattern=pat, n_pairs=n, acc=acc))
            print(res[-1], flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
