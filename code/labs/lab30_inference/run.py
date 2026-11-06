"""Lab 30 on the real model: Qwen3-0.6B base, tokenizer round trip, three decoders, compile.

Download once (needs internet):
    pip install huggingface_hub
    huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir qwen3-0.6b-base
Then:
    python3 labs/lab30_inference/run.py --model-dir qwen3-0.6b-base --device cuda
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from llmsys.qwen3 import Qwen3Tokenizer, load_qwen3_from_dir  # noqa: E402
from llmsys.utils import get_device  # noqa: E402


def load_impl(name):
    spec = importlib.util.spec_from_file_location("lab15_" + name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--prompt", default="Explain large language models in a single sentence.")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    lab = load_impl(args.impl)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # TF32 tensor cores for fp32 matmuls
    tok = Qwen3Tokenizer(os.path.join(args.model_dir, "tokenizer.json"))

    # 1) tokenizer round trip
    ids = tok.encode(args.prompt)
    print("token ids:", ids)
    for i in ids:
        print(f"  {i:>7d} -> {tok.decode([i])!r}")
    assert tok.decode(ids) == args.prompt

    # 2) load weights (bf16 on GPU, fp32 on CPU for speed on most CPUs)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = load_qwen3_from_dir(args.model_dir, "0.6B", dtype=dtype, device=device)
    if args.compile:
        model = torch.compile(model)
    x = torch.tensor([ids], device=device)

    # 3) compare decoders: identical text, very different speed
    for name in ["generate_nocache", "generate_kvcache", "generate_static"]:
        gen = getattr(lab, name)
        for rep in range(2 if args.compile else 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            m = lab.measure_generation(lambda mdl, t, n: gen(mdl, t, n, tok.pad_token_id),
                                       model, x, args.max_new)
            out = torch.cat(list(gen(model, x, args.max_new, tok.pad_token_id)), dim=1)
            mem = (f", peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"
                   if device.type == "cuda" else "")
            print(f"\n[{name} rep {rep}] TTFT {m['ttft']*1e3:.1f} ms, TPOT {m['tpot']*1e3:.1f} ms, "
                  f"{m['tokens_per_s']:.1f} tok/s{mem}")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
