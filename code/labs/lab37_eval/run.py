"""Evaluate Qwen3-0.6B (base or reasoning) on MATH-500 with the Lab 37 verifier.

Requires a downloaded HF snapshot, e.g.
    huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir ckpt/qwen3-0.6b-base
and the MATH-500 test split as JSONL ({"problem","answer",...} per line), e.g. exported from
the HuggingFaceH4/MATH-500 dataset.

    python run.py --ckpt ckpt/qwen3-0.6b-base --data math500_test.jsonl --limit 50 --batch 8
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llmsys.eval.harness import evaluate_math, load_jsonl  # noqa: E402
from llmsys.eval.math_verifier import render_math_prompt  # noqa: E402
from llmsys.generate import generate_batch  # noqa: E402
from llmsys.qwen3 import Qwen3Tokenizer, load_qwen3_from_dir  # noqa: E402
from llmsys.utils import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "../../data/math_sample.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--chat", action="store_true", help="wrap prompts in the chat template")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--cache", default="math_eval_results.jsonl")
    args = ap.parse_args()

    dev = get_device()
    model = load_qwen3_from_dir(args.ckpt, dtype=torch.bfloat16, device=dev)
    tok = Qwen3Tokenizer(os.path.join(args.ckpt, "tokenizer.json"))
    eos = tok.eos_token_id if args.chat else tok.pad_token_id

    def gen(prompts):
        ids = [tok.encode(p, chat=args.chat, enable_thinking=args.thinking) for p in prompts]
        outs = generate_batch(model, ids, args.max_new, pad_id=tok.pad_token_id, eos_id=eos)
        return [tok.decode([t for t in o if t != eos]) for o in outs]

    data = load_jsonl(args.data)[: args.limit]
    res = evaluate_math(data, gen, batch_size=args.batch, cache_path=args.cache, verbose=True,
                        config=dict(ckpt=os.path.basename(args.ckpt), chat=args.chat,
                                    thinking=args.thinking, max_new=args.max_new),
                        prompt_fn=render_math_prompt)
    lo, hi = res["ci95"]
    print(f"accuracy {res['accuracy']:.3f}  ({res['num_correct']}/{res['num_total']}), "
          f"95% CI [{lo:.3f}, {hi:.3f}]  in {res['seconds']:.1f}s")
    for s, a in sorted(res["by_subject"].items()):
        print(f"  {s:28s} {a:.3f}")


if __name__ == "__main__":
    main()
