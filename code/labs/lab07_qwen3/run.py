"""Lab 7: run *your* Qwen3 implementation with the official weights, and plan memory.

    # one-time download (needs internet):
    #   huggingface-cli download Qwen/Qwen3-0.6B --local-dir qwen3-0.6b
    python run.py --impl starter --weights-dir qwen3-0.6b --prompt "Explain RoPE in one sentence."
    python run.py --plan --gpu-gb 24 --seq 8192          # memory planning table only
"""
import argparse
import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from llmsys.generate import generate_stream  # noqa: E402
from llmsys.qwen3 import QWEN3_CONFIGS, Qwen3Tokenizer, load_qwen3_hf_state_dict  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan(lab, gpu_gb, seq):
    print(f"{'model':>6} {'params':>8} {'weights bf16':>13} {'KV/token':>10} {'KV@seq':>10} {'max seqs':>9}")
    for name, cfg in QWEN3_CONFIGS.items():
        n = lab.count_params(cfg)
        kv_tok = lab.kv_cache_bytes(cfg, 1)
        print(f"{name:>6} {n/1e9:7.2f}B {2*n/1e9:11.1f} GB {kv_tok/1024:8.0f} KiB "
              f"{lab.kv_cache_bytes(cfg, seq)/1e9:8.2f} GB "
              f"{lab.max_concurrent_sequences(cfg, gpu_gb*1e9, seq):9d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="starter")
    ap.add_argument("--size", default="0.6B")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--prompt", default="Give me a short introduction to large language models.")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--thinking", action="store_true", help="enable <think> mode")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--gpu-gb", type=float, default=24)
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    lab = load(args.impl)
    if args.plan or not args.weights_dir:
        plan(lab, args.gpu_gb, args.seq)
        if not args.weights_dir:
            return
    from safetensors.torch import load_file
    cfg = QWEN3_CONFIGS[args.size]
    with torch.device("meta"):
        model = lab.Qwen3Model(cfg)
    model = model.to_empty(device="cpu")
    cos, sin = lab.compute_rope(cfg["head_dim"], cfg["rope_base"], cfg["context_length"])
    model.cos, model.sin = cos, sin
    sd = {}
    for f in sorted(os.listdir(args.weights_dir)):
        if f.endswith(".safetensors"):
            sd.update(load_file(os.path.join(args.weights_dir, f)))
    if cfg.get("tie_embeddings"):
        model.out_head.weight = model.tok_emb.weight
    load_qwen3_hf_state_dict(model, sd)
    for p in model.parameters():          # weights in bf16, RoPE tables stay fp32
        p.data = p.data.to(torch.bfloat16)
    model = model.to(args.device).eval()
    tok = Qwen3Tokenizer(os.path.join(args.weights_dir, "tokenizer.json"))
    ids = torch.tensor([tok.encode(args.prompt, chat=True, enable_thinking=args.thinking)],
                       device=args.device)
    t0, n = time.perf_counter(), 0
    for t in generate_stream(model, ids, args.max_new, eos_id=tok.eos_token_id):
        print(tok.decode(t[0].tolist()), end="", flush=True)
        n += 1
    print(f"\n[{n} tokens, {n/(time.perf_counter()-t0):.1f} tok/s on {args.device}]")


if __name__ == "__main__":
    main()
