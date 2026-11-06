"""Project 11, milestone 3: the run matrix (Chapter 7).

A config names the models (with their task and training settings), the recipes to run on each,
the seeds, and optional sweeps of one training setting. ``expand`` turns it into a flat list of
run configs; ``run`` executes them in order, writes the results document after every run (so an
interrupted study keeps its finished runs) and appends the summary, the half-ulp reproduction and
the failure/fix table.

Config schema (see configs/smoke.json):
    {"name": ..., "baseline": "fp32", "verdict": {...thresholds...},
     "models": {"gpt": {model, task and training keys}, "mlp": {...}},
     "runs":   [{"model": "gpt", "recipes": [...], "seeds": [0], "set": {key: value}}],
     "sweeps": [{"name": "lr", "model": "mlp", "recipes": [...], "seeds": [0],
                 "key": "lr", "values": [...]}]}
"""
from __future__ import annotations

import json
import os
import platform
import time
from typing import Dict, List, Optional

import torch

from . import analysis
from .train import train


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def expand(cfg: dict) -> List[dict]:
    """Flat list of run configs; each carries ``group`` (its name without the seed) and
    ``base_group`` (the baseline run it is compared with)."""
    base = cfg.get("baseline", "fp32")
    out = []
    for spec in cfg.get("runs", []):
        m = spec["model"]
        for recipe in spec["recipes"]:
            for seed in spec.get("seeds", [0]):
                rc = {**cfg["models"][m], **spec.get("set", {}), "recipe": recipe, "seed": seed,
                      "name": recipe}
                rc.update(group=f"{m}/{recipe}", base_group=f"{m}/{base}")
                out.append(rc)
    for sw in cfg.get("sweeps", []):
        m, key = sw["model"], sw["key"]
        for val in sw["values"]:
            for recipe in sw["recipes"]:
                for seed in sw.get("seeds", [0]):
                    tag = f"@{key}={val}"
                    rc = {**cfg["models"][m], **sw.get("set", {}), key: val, "recipe": recipe,
                          "seed": seed, "name": recipe + tag, "sweep": sw["name"]}
                    rc.update(group=f"{m}/{recipe}{tag}", base_group=f"{m}/{base}{tag}")
                    out.append(rc)
    return out


def versions(device: str) -> Dict[str, str]:
    v = {"torch": torch.__version__, "cuda": str(torch.version.cuda),
         "python": platform.python_version(), "device": device,
         "allow_tf32": str(torch.backends.cuda.matmul.allow_tf32)}
    if device.startswith("cuda") and torch.cuda.is_available():
        name = torch.cuda.get_device_name(torch.device(device))
        v["gpu"] = name
        v["hardware"] = ("2x A100-40GB (PCIe)" if "A100-PCIE-40GB" in name
                         and torch.cuda.device_count() >= 1 else name)
    else:
        v["hardware"] = platform.processor() or platform.machine()
    return v


def finish(doc: dict) -> dict:
    """Add the summary, the half-ulp reproduction and the failure/fix table to ``doc``."""
    th = doc["config"].get("verdict")
    base = doc["config"].get("baseline", "fp32")
    doc["summary"] = analysis.summarize(doc["results"], th, base)
    doc["halfulp"] = analysis.halfulp_rows(doc["results"])
    doc["fixes"] = analysis.fix_rows(doc["summary"])
    return doc


def write(doc: dict, out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, default=float)
    os.replace(tmp, out)
    with open(os.path.splitext(out)[0] + ".md", "w") as f:
        f.write(analysis.render_markdown(doc))


def run(cfg: dict, device: str = "cpu", out: Optional[str] = None,
        only: Optional[List[str]] = None, verbose: bool = False) -> dict:
    """Run every expanded config whose group contains one of the ``only`` substrings."""
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False    # an "fp32" GEMM must be fp32
        torch.backends.cudnn.allow_tf32 = False
    runs = expand(cfg)
    if only:
        wanted = [r for r in runs if any(o in r["group"] for o in only)]
        bases = {r["base_group"] for r in wanted}
        runs = [r for r in runs if r in wanted or r["group"] in bases]
    doc = {"config": cfg, "versions": versions(device), "results": [], "seconds": 0.0}
    t0 = time.perf_counter()
    for i, rc in enumerate(runs):
        if verbose:
            print(f"[{i + 1}/{len(runs)}] {rc['group']} seed {rc['seed']}", flush=True)
        res = train(rc, device)
        res.update(group=rc["group"], base_group=rc["base_group"], sweep=rc.get("sweep"))
        doc["results"].append(res)
        doc["seconds"] = time.perf_counter() - t0
        if out:
            write(finish(doc), out)
    return finish(doc)
