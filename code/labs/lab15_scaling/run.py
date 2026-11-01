"""Self-consistency on MATH-style problems with naive vs shared-prefix sampling.

    python run.py --ckpt ckpt/qwen3-0.6b-base --n 8 --limit 20 [--cot]
Prints accuracy for greedy, CoT, and self-consistency, and wall-clock for naive vs
shared-prefix sampling (same number of samples).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llmsys.eval.harness import load_jsonl  # noqa: E402
from llmsys.eval.math_verifier import extract_final_candidate, grade_answer, normalize_text, render_math_prompt  # noqa: E402
from llmsys.generate import generate  # noqa: E402
from llmsys.qwen3 import Qwen3Tokenizer, load_qwen3_from_dir  # noqa: E402
from llmsys.tts import add_cot, majority_vote, sample_n_shared_prefix  # noqa: E402
from llmsys.utils import get_device, timer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "../../data/math_sample.jsonl"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--cot", action="store_true")
    args = ap.parse_args()
    dev = get_device()
    model = load_qwen3_from_dir(args.ckpt, device=dev)
    tok = Qwen3Tokenizer(os.path.join(args.ckpt, "tokenizer.json"))
    eos = tok.pad_token_id
    data = load_jsonl(args.data)[: args.limit]
    stats, correct = {}, {"greedy": 0, "sc": 0}
    g = torch.Generator(device=dev).manual_seed(123)
    for ex in data:
        prompt = render_math_prompt(add_cot(ex["problem"]) if args.cot else ex["problem"])
        ids = tok.encode(prompt)
        out = generate(model, torch.tensor([ids], device=dev), args.max_new, eos_id=eos)[0, len(ids):]
        correct["greedy"] += grade_answer(extract_final_candidate(tok.decode(out.tolist())), ex["answer"])
        with timer("naive", stats):
            for _ in range(args.n):
                generate(model, torch.tensor([ids], device=dev), args.max_new, temperature=args.temperature,
                         top_p=args.top_p, eos_id=eos, generator=g)
        with timer("shared", stats):
            r = sample_n_shared_prefix(model, ids, args.n, args.max_new, args.temperature,
                                       top_p=args.top_p, eos_id=eos, generator=g)
        answers = [extract_final_candidate(tok.decode(t)) for t in r["tokens"]]
        win = majority_vote(answers, key=normalize_text)["answer"]
        correct["sc"] += bool(win) and grade_answer(win, ex["answer"])
    n = len(data)
    print(f"greedy acc {correct['greedy']/n:.3f} | self-consistency(n={args.n}) acc {correct['sc']/n:.3f}")
    print(f"sampling time naive {stats['naive']:.1f}s vs shared-prefix batch {stats['shared']:.1f}s")


if __name__ == "__main__":
    main()
