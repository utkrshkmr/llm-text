"""Print everything you should record before benchmarking (Chapter 43, "Setup").

    python projects/p2_systems/env_check.py            # human-readable report
    python projects/p2_systems/env_check.py --json     # machine-readable, attach to every results file

Reports Python/PyTorch/CUDA/cuDNN/NCCL/Triton versions, GPU models with memory and SM counts,
interconnect topology (``nvidia-smi topo -m``), whether Nsight Systems/Compute are on PATH, the
allocator configuration, and a 1-second matmul sanity check against the published peak.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

import torch

# dense bf16 tensor-core peaks (FLOP/s) used for the sanity check; extend for your hardware
PEAKS = {"H100": 989e12, "H200": 989e12, "A100": 312e12, "B200": 2250e12, "L4": 121e12,
         "4090": 165e12, "A6000": 155e12}


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def collect(run_matmul: bool = True) -> dict:
    info = dict(python=sys.version.split()[0], platform=platform.platform(),
                torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                cuda_available=torch.cuda.is_available(),
                allocator_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                nsys=shutil.which("nsys"), ncu=shutil.which("ncu"))
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = None
    if torch.cuda.is_available():
        info["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        info["gpus"] = [dict(name=torch.cuda.get_device_name(i),
                             mem_GiB=round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
                             sms=torch.cuda.get_device_properties(i).multi_processor_count,
                             cc=".".join(map(str, torch.cuda.get_device_capability(i))))
                        for i in range(torch.cuda.device_count())]
        info["topology"] = _cmd(["nvidia-smi", "topo", "-m"])
        if run_matmul:
            info["matmul_check"] = matmul_check()
    return info


def matmul_check(n: int = 8192) -> dict:
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        a @ b
    end.record()
    torch.cuda.synchronize()
    tflops = 10 * 2 * n ** 3 / (start.elapsed_time(end) / 1e3) / 1e12
    name = torch.cuda.get_device_name(0)
    peak = next((v for k, v in PEAKS.items() if k in name), None)
    return dict(bf16_TFLOPs=round(tflops, 1),
                fraction_of_peak=round(tflops * 1e12 / peak, 2) if peak else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-matmul", action="store_true")
    a = p.parse_args()
    info = collect(not a.no_matmul)
    if a.json:
        print(json.dumps(info, indent=2))
        return
    for k, v in info.items():
        if k == "topology" and v:
            print("topology:\n" + v)
        elif k == "gpus":
            for i, g in enumerate(v):
                print(f"gpu[{i}]: {g}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
