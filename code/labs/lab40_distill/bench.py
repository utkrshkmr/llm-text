"""Lab 40 measurement: does distilling the *trace* help a tiny student?

Trains three students on the same teacher data (multi-digit addition):
  think   : <think>column-by-column trace</think> \\boxed{sum}
  plain   : same trace without the think tags
  direct  : only \\boxed{sum}  (no reasoning at all)
and reports held-out exact-match accuracy and train tokens per example.

    python labs/lab40_distill/bench.py --digits 2 --n_train 1500 --epochs 6
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import time

import torch

from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model
from llmsys.utils import set_seed
from solution import (CharTokenizer, accuracy, build_examples, length_stats, make_records,
                      render_short_prompt, train_distillation)


def run(variant, args, tok, train_recs, test_recs):
    set_seed(args.seed)
    recs = [dict(r) for r in train_recs]
    if variant == "direct":
        for r in recs:
            r["reasoning"] = ""
    ex, _ = build_examples(recs, tok.encode_chat_prompt, tok.encode, tok.eos_id,
                           use_think=(variant == "think"), render=render_short_prompt)
    cfg = dict(QWEN3_CONFIG_TINY, vocab_size=tok.vocab_size, emb_dim=128, n_heads=4, head_dim=32,
               n_kv_groups=2, hidden_dim=384, n_layers=4)
    m = Qwen3Model(cfg)
    t0 = time.perf_counter()
    train_distillation(m, ex, ex[:16], epochs=args.epochs, lr=args.lr, batch_size=args.bs,
                       pad_id=tok.pad_id, log_every=0, log=lambda *_: None)
    dt = time.perf_counter() - t0
    acc = accuracy(m, tok, test_recs)
    st = length_stats(ex, answer_only=True)
    return dict(variant=variant, acc=acc, answer_tokens=round(st["mean"], 1), train_s=round(dt, 1))


def main():
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--n_train", type=int, default=1500)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="think,plain,direct")
    args = ap.parse_args()
    tok = CharTokenizer()
    train_recs = make_records(args.n_train, args.digits, seed=1)
    seen = {r["problem"] for r in train_recs}
    test_recs = [r for r in make_records(4 * args.n_test, args.digits, seed=2)
                 if r["problem"] not in seen][: args.n_test]
    for v in args.variants.split(","):
        print(run(v, args, tok, train_recs, test_recs), flush=True)


if __name__ == "__main__":
    main()
