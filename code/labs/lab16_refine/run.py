"""Self-refinement with heuristic vs average-logprob gating on MATH-style problems.

    python run.py --ckpt ckpt/qwen3-0.6b-base --limit 20 --iterations 2 --scorer logprob
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llmsys.eval.harness import load_jsonl  # noqa: E402
from llmsys.eval.math_verifier import extract_final_candidate, grade_answer, render_math_prompt  # noqa: E402
from llmsys.generate import generate  # noqa: E402
from llmsys.qwen3 import Qwen3Tokenizer, load_qwen3_from_dir  # noqa: E402
from llmsys.tts import avg_logprob_score, heuristic_score, self_refine  # noqa: E402
from llmsys.utils import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "../../data/math_sample.jsonl"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--scorer", choices=["none", "heuristic", "logprob"], default="logprob")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    dev = get_device()
    model = load_qwen3_from_dir(args.ckpt, device=dev)
    tok = Qwen3Tokenizer(os.path.join(args.ckpt, "tokenizer.json"))
    eos = tok.pad_token_id
    gen_tokens = [0]

    def generate_fn(prompt, max_new):
        ids = tok.encode(prompt)
        out = generate(model, torch.tensor([ids], device=dev), max_new, temperature=args.temperature,
                       top_p=0.9, eos_id=eos)[0, len(ids):].tolist()
        gen_tokens[0] += len(out)
        return tok.decode([t for t in out if t != eos])

    def score(question, answer):
        if args.scorer == "heuristic":
            return heuristic_score(answer)
        return avg_logprob_score(model, tok.encode(render_math_prompt(question)), tok.encode(answer))

    data = load_jsonl(args.data)[: args.limit]
    first_ok = final_ok = 0
    t0 = time.perf_counter()
    for ex in data:
        r = self_refine(ex["problem"], generate_fn, args.iterations,
                        None if args.scorer == "none" else score, answer_prompt_fn=render_math_prompt,
                        max_answer_tokens=512, max_critique_tokens=192)
        first = r.steps[0].draft if r.steps else r.final
        first_ok += grade_answer(extract_final_candidate(first), ex["answer"])
        final_ok += grade_answer(extract_final_candidate(r.final), ex["answer"])
    n = len(data)
    print(f"draft acc {first_ok/n:.3f} -> refined acc {final_ok/n:.3f} | "
          f"{gen_tokens[0]} generated tokens, {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
