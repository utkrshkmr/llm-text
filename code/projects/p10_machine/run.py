"""Project 10: run the microbenchmark suite and write the spec sheet (Chapter 6).

    python run.py --config configs/smoke.json --out results/smoke.json        # ~2 min, 1 GPU (+P2P)
    python run.py --config configs/full.json  --out results/full.json         # the full study
    python run.py --config configs/cpu.json --device cpu --out /tmp/p10.json  # dry run, no GPU
    python run.py --config configs/smoke.json --milestones 1 3                # a subset
    P10_IMPL=~/my_p10 python run.py --config configs/smoke.json               # your package

The results file follows the repository schema {config, versions, metrics, timings, artifacts};
a Markdown spec sheet is written next to it.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # code/ for llmsys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "smoke.json"))
    ap.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if present)")
    ap.add_argument("--out", default=None, help="results JSON path (Markdown written next to it)")
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument(
        "--impl", default=None, help="package directory (default: P10_IMPL or reference/)"
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="accepted for uniformity with other kits: this project needs no dataset",
    )
    a = ap.parse_args(argv)

    import torch
    import impl

    p10 = impl.load(a.impl)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = p10.suite.load_config(a.config)
    doc = p10.suite.run(cfg, device=device, out=a.out, milestones=a.milestones)
    print(p10.report.render_markdown(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
