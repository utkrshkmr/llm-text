"""Pre-flight check for an RL-on-LLMs project (e.g. the CS336 A5 setup): run it on the
machine that will do the work *before* burning GPU hours.

    python code/projects/a5/env_check.py --need-gpus 2 --min-gpu-gb 40

It only inspects the environment; it never downloads models or touches your repo.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys


def probe_module(name: str):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "installed")
    except Exception as e:  # noqa: BLE001 - we want to report any import failure
        return None if isinstance(e, ModuleNotFoundError) else f"import error: {e.__class__.__name__}"


def gpu_inventory():
    try:
        import torch
    except ModuleNotFoundError:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append({"index": i, "name": p.name, "mem_gb": round(p.total_memory / 1e9, 1),
                    "cc": f"{p.major}.{p.minor}", "bf16": p.major >= 8})
    return out


def check(need_gpus: int = 0, min_gpu_gb: float = 0.0, python=(3, 12), min_disk_gb: float = 20.0):
    report = {"python": platform.python_version(), "platform": platform.platform(), "problems": []}
    if sys.version_info[:2] != tuple(python):
        report["problems"].append(f"python {python[0]}.{python[1]} expected (found {report['python']})")
    for mod in ["torch", "transformers", "vllm", "flash_attn", "wandb", "math_verify"]:
        report[mod] = probe_module(mod)
    gpus = gpu_inventory()
    report["gpus"] = gpus
    if len(gpus) < need_gpus:
        report["problems"].append(f"need {need_gpus} GPUs, found {len(gpus)}")
    small = [g for g in gpus if g["mem_gb"] < min_gpu_gb]
    if small:
        report["problems"].append(f"GPUs below {min_gpu_gb} GB: {[g['index'] for g in small]}")
    if any(not g["bf16"] for g in gpus):
        report["problems"].append("some GPUs lack native bf16 (compute capability < 8.0)")
    free_gb = shutil.disk_usage(os.getcwd()).free / 1e9
    report["disk_free_gb"] = round(free_gb, 1)
    if free_gb < min_disk_gb:
        report["problems"].append(f"only {free_gb:.0f} GB free disk (checkpoints are large)")
    report["wandb_key_set"] = bool(os.environ.get("WANDB_API_KEY"))
    report["hf_home"] = os.environ.get("HF_HOME", "~/.cache/huggingface (default)")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--need-gpus", type=int, default=0)
    ap.add_argument("--min-gpu-gb", type=float, default=0.0)
    a = ap.parse_args()
    r = check(a.need_gpus, a.min_gpu_gb)
    print(json.dumps(r, indent=2))
    sys.exit(1 if r["problems"] else 0)
