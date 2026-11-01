"""Check that a machine is ready for Project 4 (packages, CPUs, disk, data paths).

    python projects/p4_data/env_check.py --data-root data/ --scratch /tmp

Prints a table and exits non-zero if something required is missing. It only inspects the
environment; it does not download anything.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys

PACKAGES = [  # (import name, required?, what it is for)
    ("fastwarc", True, "fast WARC/WET record iteration"),
    ("resiliparse", True, "encoding detection + HTML-to-text"),
    ("fasttext", True, "language-ID / toxicity / quality classifiers"),
    ("mmh3", False, "MurmurHash3 for MinHash"),
    ("nltk", False, "word tokenisation for rules"),
    ("tldextract", False, "domain extraction from URLs"),
    ("numpy", True, "arrays, token files"),
    ("transformers", False, "GPT-2 tokenizer for the final .bin"),
    ("modal", False, "serverless cloud execution (optional)"),
    ("wandb", False, "experiment tracking"),
]


def check(data_root: str, scratch: str, expect=("cc", "models")) -> bool:
    ok = True
    print(f"python {sys.version.split()[0]}")
    for name, req, why in PACKAGES:
        try:
            m = importlib.import_module(name)
            ver = getattr(m, "__version__", "?")
            print(f"  [ok]   {name:<13} {ver:<10} {why}")
        except Exception:
            print(f"  [{'MISS' if req else 'opt '}] {name:<13} {'':<10} {why}")
            ok &= not req
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"CPUs available to this process: {cpus}")
    for path in (scratch, data_root):
        if path and os.path.exists(path):
            free = shutil.disk_usage(path).free / 1e9
            print(f"free disk at {path}: {free:.1f} GB")
    if data_root:
        for sub in expect:
            p = os.path.join(data_root, sub)
            print(f"  {'[ok]  ' if os.path.exists(p) else '[MISS]'} {p}")
            ok &= os.path.exists(p)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="")
    ap.add_argument("--scratch", default="/tmp")
    a = ap.parse_args()
    sys.exit(0 if check(a.data_root, a.scratch) else 1)
