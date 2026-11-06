"""Lab 10 measurements: where does the time go in a GPT-2 forward pass?

    python bench.py --impl solution --size "gpt2-small (124M)" --seq 256 --batch 4
    python bench.py --weights path/to/gpt2/model.safetensors --prompt "Hello, I am"

Prints (1) the analytic vs. module parameter count, (2) analytic forward FLOPs,
(3) a torch.profiler table grouped by operator, (4) achieved FLOP/s.
Requires no network unless you pass --weights (download the HF ``gpt2`` repo yourself).
"""
import argparse
import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from llmsys.gpt import gpt2_config, load_gpt2_hf_state_dict  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="starter")
    ap.add_argument("--size", default="gpt2-small (124M)")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--weights", default=None, help="HF gpt2 model.safetensors")
    ap.add_argument("--prompt", default="Every effort moves you")
    args = ap.parse_args()
    lab = load(args.impl)
    cfg = gpt2_config(args.size, drop_rate=0.0)
    model = lab.GPTModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"module params: {n:,}  analytic: {lab.count_params_analytic(cfg)['total']:,}")
    if args.weights:
        from safetensors.torch import load_file
        load_gpt2_hf_state_dict(model, load_file(args.weights))
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        idx = torch.tensor([enc.encode(args.prompt)])
        out = lab.generate_greedy(model.eval(), idx, 20, cfg["context_length"])
        print(enc.decode(out[0].tolist()))
    dtype = getattr(torch, args.dtype)
    model = model.to(args.device, dtype).eval()
    idx = torch.randint(0, cfg["vocab_size"], (args.batch, args.seq), device=args.device)
    flops = lab.forward_flops(cfg, args.batch, args.seq)
    with torch.no_grad():
        for _ in range(2):
            model(idx)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            model(idx)
        if args.device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
    print(f"forward: {dt*1e3:.1f} ms, {flops/1e9:.1f} GFLOP, {flops/dt/1e12:.2f} TFLOP/s achieved")
    acts = [torch.profiler.ProfilerActivity.CPU]
    if args.device == "cuda":
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad(), torch.profiler.profile(activities=acts) as prof:
        model(idx)
    key = "cuda_time_total" if args.device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=15))


if __name__ == "__main__":
    main()
