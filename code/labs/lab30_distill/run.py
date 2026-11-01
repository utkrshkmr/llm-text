"""Real-scale distillation: Qwen3-0.6B-Base student on teacher reasoning traces.

Requires network access (or local copies) and ideally a GPU. Example:

    huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir qwen3-base
    huggingface-cli download Qwen/Qwen3-0.6B tokenizer.json --local-dir qwen3-reasoning
    huggingface-cli download --repo-type dataset rasbt/math_distill \
        data/deepseek-r1-math-train.json --local-dir math_distill
    python run.py --model_dir qwen3-base --tokenizer qwen3-reasoning/tokenizer.json \
        --data math_distill/data/deepseek-r1-math-train.json --max_len 2048 \
        --epochs 3 --lr 1e-5 --val_size 25 --batch_size 1

The tokenizer of the *reasoning* checkpoint is used because its vocabulary already
contains <think> and </think> as single tokens.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import argparse
import json

import torch

from llmsys.distill import (build_examples, filter_by_max_len, length_stats, normalize_record,
                            split_train_val, train_distillation)
from llmsys.qwen3 import Qwen3Tokenizer, load_qwen3_from_dir
from llmsys.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--val_size", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--no_think", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv", default="distill_metrics.csv")
    ap.add_argument("--ckpt_dir", default="checkpoints/distill")
    args = ap.parse_args()

    set_seed(123)
    device = get_device()
    tok = Qwen3Tokenizer(args.tokenizer)
    with open(args.data) as f:
        raw = json.load(f)
    records = [normalize_record(r) for r in raw]
    if args.limit:
        records = records[: args.limit]

    def enc_prompt(text):  # chat-wrapped, generation prompt open, thinking enabled
        return tok.encode(text, chat=True, enable_thinking=True)

    def enc_answer(text):  # raw continuation: no chat header
        return tok.encode(text)

    ex, skipped = build_examples(records, enc_prompt, enc_answer, tok.eos_token_id,
                                 use_think=not args.no_think)
    print("examples", len(ex), "skipped", skipped, length_stats(ex))
    ex = filter_by_max_len(ex, args.max_len)
    print("after filter", length_stats(ex))
    train, val = split_train_val(ex, args.val_size)

    model = load_qwen3_from_dir(args.model_dir, "0.6B", dtype=torch.bfloat16, device=device)
    train_distillation(model, train, val, device=device, epochs=args.epochs, lr=args.lr,
                       grad_clip=args.grad_clip, batch_size=args.batch_size,
                       pad_id=tok.pad_token_id, log_every=200, checkpoint_dir=args.ckpt_dir,
                       csv_path=args.csv)


if __name__ == "__main__":
    main()
