"""Project 11: run the low-precision training study (Chapter 7).

    python run.py --config configs/smoke.json --out results/smoke.json      # the smoke matrix
    python run.py --config configs/full.json  --out results/full.json       # the full study
    python run.py --config configs/cpu.json --device cpu --out /tmp/p11.json  # dry run, no GPU
    python run.py --config configs/smoke.json --only bf16_pure fp8          # a subset (+ baselines)
    P11_IMPL=~/my_p11 python run.py --config configs/smoke.json               # your package

The tasks are synthetic and generated from a seed, so ``--synthetic`` (accepted for uniformity
with the other kits) changes nothing. The results document (config, versions, every run's logged
records, summary, half-ulp table, failure/fix table) is written after each run, with a Markdown
report next to it.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))                    # code/ for llmsys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "smoke.json"))
    ap.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if present)")
    ap.add_argument("--out", default=None, help="results JSON path (Markdown written next to it)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only groups containing one of these substrings (and their baselines)")
    ap.add_argument("--impl", default=None, help="package directory (default: P11_IMPL or reference/)")
    ap.add_argument("--synthetic", action="store_true", help="no-op: every task here is synthetic")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    import torch
    import impl
    p11 = impl.load(a.impl)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = p11.study.load_config(a.config)
    doc = p11.study.run(cfg, device=device, out=a.out, only=a.only, verbose=not a.quiet)
    print(p11.analysis.render_markdown(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
