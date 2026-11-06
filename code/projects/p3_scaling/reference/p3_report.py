"""Project 3 (reference solution): report tables from the sweep records (Chapter 44).

    python p3_report.py --runs out/full/runs.jsonl [--heldout out/full/heldout.jsonl] --out out/full/report.md

Everything is regenerated from ``runs.jsonl``; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

import p3_fit as F  # noqa: E402
from llmsys import scaling as S  # noqa: E402


def md_table(header: Sequence[str], rows: Sequence[Sequence]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}" if (abs(v) < 1e-2 or abs(v) >= 1e5) and v != 0 else f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v)
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def lr_table(records: Sequence[dict]) -> Optional[str]:
    lr_recs = [r for r in records if r.get("kind") == "lr"]
    if not lr_recs:
        return None
    rows, basins = [], []
    for size in sorted({r["size_id"] for r in lr_recs}, key=lambda s: float(s[1:])):
        rs = sorted((r for r in lr_recs if r["size_id"] == size), key=lambda r: r["lr"])
        lrs = [r["lr"] for r in rs]
        losses = [r["loss"] if not r.get("diverged") else float("nan") for r in rs]
        best = S.fit_lr_optimum(lrs, losses) if sum(math.isfinite(l) for l in losses) >= 3 else float("nan")
        basins.append((size, rs[0]["d"], best))
        rows += [(size, r["d"], r["lr"], "diverged" if r.get("diverged") else r["loss"]) for r in rs]
    text = "### Learning-rate basins (muP proxy widths)\n\n" + md_table(["size", "d", "peak lr", "val loss"], rows)
    text += "\n\n" + md_table(["size", "d", "lr* (parabola vertex)"], basins)
    if len(basins) >= 2 and all(math.isfinite(b[2]) for b in basins):
        ratio = max(b[2] for b in basins) / min(b[2] for b in basins)
        text += f"\n\nSpread of lr* across widths: x{ratio:.2f} (muP transfer: close to 1)."
    return text


def build_report(records: Sequence[dict], heldout: Optional[dict] = None, n_boot: int = 200,
                 seed: int = 0, pred_nd: Optional[tuple] = None) -> Dict[str, object]:
    runs = F.runs_from_records(records)
    parts: List[str] = ["# Project 3 report (generated)\n"]
    result: Dict[str, object] = {"n_runs": len(runs)}
    lr = lr_table(records)
    if lr:
        parts.append(lr)
    branch = [r for r in records if r.get("kind") == "branch"]
    if branch:
        rows = [(r["size_id"], r["d"], r["L"], r["n_params"], f"{r['budget']:.1e}", r["tokens"],
                 round(r["tokens"] / r["n_params"], 1), r["loss"], round(r.get("decay_seconds", 0.0), 1))
                for r in sorted(branch, key=lambda r: (r["n_params"], r["budget"]))]
        parts.append("### Branched runs\n\n" + md_table(
            ["size", "d", "L", "N_ne", "budget", "tokens", "D/N", "loss", "decay s"], rows))
        sav = F.wsd_saving(records)
        if sav:
            result["wsd_saving"] = sav
            parts.append(f"WSD branching used {sav['wsd_tokens']:.3g} training tokens where separate cosine "
                         f"runs would have needed {sav['cosine_tokens']:.3g}: a saving of {100 * sav['saving']:.1f}%.")
    if len(runs) >= 6 and len(S.group_by_budget(runs, 0.15)) >= 2:
        iso = F.fit_isoflop(runs)
        chin = F.fit_parametric(runs)
        result["isoflop"] = dict(optima=iso.optima, n_law=iso.n_law, d_law=iso.d_law, l_law=iso.l_law)
        result["chinchilla"] = chin.as_dict()
        parts.append("### IsoFLOP optima\n\n" + md_table(
            ["C", "N_opt", "D_opt", "L_opt", "method"],
            [(f"{o['compute']:.2e}", f"{o['n_opt']:.3g}", f"{o['d_opt']:.3g}", o["loss"], o["method"]) for o in iso.optima]))
        parts.append(f"Allocation: N_opt = {iso.n_law['k']:.3g} C^{iso.n_law['a']:.3f}, "
                     f"D_opt = {iso.d_law['k']:.3g} C^{iso.d_law['a']:.3f} (exponents sum {iso.n_law['a'] + iso.d_law['a']:.3f}).")
        parts.append(f"Parametric: E={chin.E:.4f} A={chin.A:.3g} B={chin.B:.3g} alpha={chin.alpha:.3f} "
                     f"beta={chin.beta:.3f}; implied a = beta/(alpha+beta) = {chin.beta / (chin.alpha + chin.beta):.3f}.")
        if pred_nd is None and heldout is not None:
            pred_nd = (heldout["n_params"], heldout["tokens"])
        if pred_nd is not None:
            ci = F.bootstrap_statistics(runs, F.standard_statistics(*pred_nd), n_boot=n_boot, seed=seed)
            result["intervals"] = {k: v.as_dict() for k, v in ci.items()}
            parts.append("### Bootstrap intervals (90%)\n\n" + md_table(
                ["statistic", "point", "lo", "hi", "draws"],
                [(k, v.point, v.lo, v.hi, v.n_boot) for k, v in ci.items()]))
    if heldout is not None:
        sc = heldout.get("score", {})
        result["heldout"] = sc
        parts.append("### Held-out run\n\n" + md_table(
            ["N_ne", "tokens", "predicted", "interval", "observed", "error", "inside"],
            [(heldout["n_params"], heldout["tokens"], sc.get("predicted"), f"[{sc.get('lo'):.4f}, {sc.get('hi'):.4f}]",
              sc.get("observed"), sc.get("error"), sc.get("inside"))]))
    result["markdown"] = "\n\n".join(parts) + "\n"
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--heldout", help="heldout.jsonl written by p3_sweep.py heldout")
    ap.add_argument("--out", help="write the Markdown here (and .json next to it)")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args(argv)
    with open(a.runs) as f:
        records = [json.loads(l) for l in f if l.strip()]
    heldout = None
    if a.heldout and os.path.exists(a.heldout):
        with open(a.heldout) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        heldout = rows[-1] if rows else None
    rep = build_report(records, heldout, n_boot=a.n_boot)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            f.write(rep["markdown"])
        with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
            json.dump({k: v for k, v in rep.items() if k != "markdown"}, f, indent=1, default=float)
        print(a.out)
    else:
        print(rep["markdown"])


if __name__ == "__main__":
    main()
