"""Print the facts about this machine that decide how you should run the experiments."""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys


def _mem_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return float("nan")


def report() -> dict:
    out = dict(python=sys.version.split()[0], platform=platform.platform(),
               cpus=os.cpu_count(), ram_gb=round(_mem_gb(), 1),
               disk_free_gb=round(shutil.disk_usage(".").free / 1e9, 1))
    for pkg in ["torch", "numpy", "regex", "einops", "jaxtyping", "wandb", "tiktoken"]:
        try:
            m = importlib.import_module(pkg)
            out[pkg] = getattr(m, "__version__", "installed")
        except ImportError:
            out[pkg] = None
    try:
        import torch
        out["cuda"] = torch.cuda.is_available()
        if out["cuda"]:
            p = torch.cuda.get_device_properties(0)
            out["gpu"] = p.name
            out["gpu_mem_gb"] = round(p.total_memory / 1e9, 1)
            out["compute_capability"] = f"{p.major}.{p.minor}"
            out["bf16"] = torch.cuda.is_bf16_supported()
            out["tf32_capable"] = p.major >= 8
        out["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        out["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return out


def advice(r: dict) -> list:
    tips = []
    if r.get("regex") is None:
        tips.append("install the `regex` package: the GPT-2 pre-tokenizer needs \\p{L}/\\p{N} classes")
    if r.get("cuda"):
        if r.get("tf32_capable"):
            tips.append("CUDA: consider torch.set_float32_matmul_precision('high') (TF32) and bf16 autocast")
    elif r.get("mps"):
        tips.append("MPS: use the low-resource settings; avoid TF32 on MPS; compile with backend='aot_eager'")
    else:
        tips.append("CPU only: use the low-resource token budget and torch.compile(model)")
    if r.get("disk_free_gb", 1e9) < 30:
        tips.append("less than 30 GB free: OpenWebText text + uint16 token files need room")
    return tips


if __name__ == "__main__":
    r = report()
    w = max(len(k) for k in r)
    for k, v in r.items():
        print(f"{k:<{w}}  {v}")
    for t in advice(r):
        print("tip:", t)
