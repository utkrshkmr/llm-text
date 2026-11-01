"""Lab 13 measurements.

Toy mode (CPU, no downloads) -- acceptance statistics vs gamma and draft quality:
    python bench.py toy

Real mode (needs downloaded HF snapshots, ideally a GPU):
    huggingface-cli download Qwen/Qwen3-0.6B --local-dir qwen3-0.6b
    huggingface-cli download Qwen/Qwen3-1.7B --local-dir qwen3-1.7b
    python bench.py spec  --target qwen3-1.7b --target-size 1.7B --draft qwen3-0.6b
    python bench.py quant --model qwen3-0.6b --size 0.6B --bits 8 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from llmsys.generate import generate  # noqa: E402
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model  # noqa: E402
from llmsys.specdec import expected_tokens_per_round, speculative_generate  # noqa: E402
from llmsys.utils import get_device, synchronize  # noqa: E402

import solution as lab  # noqa: E402


def toy(args):
    torch.manual_seed(0)
    # untied embeddings: a tied random model just copies its last input token
    cfg = dict(QWEN3_CONFIG_TINY, n_layers=4, tie_embeddings=False, context_length=1024)
    target = Qwen3Model(cfg).eval()
    prompt = torch.randint(0, cfg["vocab_size"], (1, 8))
    print("noise  gamma  kept/prop  alpha_hat  tok/call  predicted(alpha_hat)")
    for noise in (0.1, 0.3, 1.0):
        draft = lab.noisy_copy(target, noise, seed=1)
        for gamma in (1, 2, 4, 8):
            g = torch.Generator().manual_seed(0)
            _, st = speculative_generate(target, draft, prompt, 160, gamma=gamma,
                                         temperature=0.25, generator=g)
            a = st.alpha_hat
            print(f"{noise:5.2f} {gamma:6d} {st.acceptance_rate:10.3f} {a:10.3f} "
                  f"{st.tokens_per_target_call:9.2f} "
                  f"{expected_tokens_per_round(a, gamma):10.2f}")


def _load(path, size, device, dtype):
    from llmsys.qwen3 import load_qwen3_from_dir
    return load_qwen3_from_dir(path, size, dtype=dtype, device=device)


def spec(args):
    from llmsys.qwen3 import Qwen3Tokenizer
    dev = get_device()
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    target = _load(args.target, args.target_size, dev, dtype)
    draft = _load(args.draft, "0.6B", dev, dtype)
    tok = Qwen3Tokenizer(os.path.join(args.target, "tokenizer.json"))
    ids = torch.tensor([tok.encode(args.prompt, chat=True)], device=dev)
    for name, fn in [("target only", lambda: (generate(target, ids, args.n), None)),
                     ("speculative", lambda: speculative_generate(target, draft, ids, args.n,
                                                                  gamma=args.gamma))]:
        synchronize(); t0 = time.perf_counter()
        out, st = fn()
        synchronize(); dt = time.perf_counter() - t0
        extra = "" if st is None else f" alpha={st.acceptance_rate:.2f} calls={st.target_calls}"
        print(f"{name:12s}: {args.n / dt:7.1f} tok/s{extra}")
    print(tok.decode(out[0, ids.shape[1]:].tolist()))


def quant(args):
    dev = get_device()
    model = _load(args.model, args.size, dev, torch.float32 if dev.type == "cpu" else torch.bfloat16)
    ids = torch.randint(0, 150_000, (1, 256), device=dev)
    with torch.no_grad():
        ref = model(ids).float()
        for b in args.bits:
            mq = lab.quantize_linear_layers(model, b, 128)
            out = mq(ids).float()
            kl = torch.nn.functional.kl_div(out.log_softmax(-1), ref.log_softmax(-1),
                                            log_target=True, reduction="batchmean")
            print(f"int{b}: linear bytes {lab.linear_weight_bytes(mq)/2**20:8.1f} MiB  "
                  f"(fp: {lab.linear_weight_bytes(model)/2**20:8.1f})  KL(ref||q)={kl:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("toy")
    s = sub.add_parser("spec")
    s.add_argument("--target", required=True); s.add_argument("--target-size", default="1.7B")
    s.add_argument("--draft", required=True); s.add_argument("--gamma", type=int, default=4)
    s.add_argument("--n", type=int, default=256)
    s.add_argument("--prompt", default="Explain why the sky is blue in three sentences.")
    q = sub.add_parser("quant")
    q.add_argument("--model", required=True); q.add_argument("--size", default="0.6B")
    q.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    a = ap.parse_args()
    {"toy": toy, "spec": spec, "quant": quant}[a.cmd](a)
