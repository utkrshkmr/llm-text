"""Project 11: draw the chapter's figures as SVG (house palette, light theme) (Chapter 7).

    python figures.py                                     # from results/smoke_a100_pcie.json
    python figures.py --json results/full.json --out-dir /tmp/f

Diagrams are hand-written SVG; plots use matplotlib and read a results document written by
run.py (measured on 2x A100-40GB (PCIe) for the file shipped in results/).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from llmsys import numerics as N  # noqa: E402

BLUE, AMBER, PURPLE, GREEN, RED, GREY = ("#1f4e79", "#8a5a00", "#6a4c93", "#2e6b30", "#9b2c2c",
                                         "#555555")
TINT = {BLUE: "#dce6f1", AMBER: "#f3e9d2", PURPLE: "#ebe4f2", GREEN: "#deeedc", RED: "#f2dcdc",
        GREY: "#eeeeee"}
BOOK_FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "book", "figures")
STEM = "ch07_proj_lowprec"
MEASURED = "measured on 2x A100-40GB (PCIe)"
INK = "#111111"   # near-black, written explicitly into the SVG so the dark theme can convert it
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "axes.titlecolor": INK, "svg.hashsalt": "ch07",
                     "font.family": "DejaVu Sans", "figure.facecolor": "#ffffff",
                     "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff",
                     "legend.frameon": False})
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"


# --------------------------------------------------------------------------- SVG helpers
class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill="#ffffff", stroke="#000000", sw=1.0, rx=0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                          f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x, y, s, size=12, anchor="start", weight="normal", fill="#000000",
             style="normal"):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
                          f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" '
                          f'fill="{fill}" font-family="{FONT}">{s}</text>')

    def line(self, x1, y1, x2, y2, stroke="#000000", sw=1.0, arrow=False, dash=None):
        extra = ' marker-end="url(#arr)"' if arrow else ""
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        self.items.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{extra}/>')

    def path(self, d, stroke="#000000", sw=1.0, arrow=False, dash=None):
        extra = ' marker-end="url(#arr)"' if arrow else ""
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        self.items.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"'
                          f'{extra}/>')

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">\n<defs><marker id="arr" viewBox="0 0 10 10" '
                'refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                '<path d="M 0 0 L 10 5 L 0 10 z" fill="#000000"/></marker></defs>\n'
                f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>\n')
        with open(path, "w") as f:
            f.write(head + "\n".join(self.items) + "\n</svg>\n")


def box(s: Svg, x, y, w, h, title, sub=None, colour=BLUE, size=12, sub2=None):
    s.rect(x, y, w, h, fill=TINT[colour], stroke=colour, sw=1.4, rx=6)
    if sub2:
        s.text(x + w / 2, y + h / 2 - 10, title, size, "middle", "bold")
        s.text(x + w / 2, y + h / 2 + 6, sub, size - 2, "middle")
        s.text(x + w / 2, y + h / 2 + 20, sub2, size - 2, "middle")
    elif sub:
        s.text(x + w / 2, y + h / 2 - 3, title, size, "middle", "bold")
        s.text(x + w / 2, y + h / 2 + 13, sub, size - 2, "middle")
    else:
        s.text(x + w / 2, y + h / 2 + 4, title, size, "middle", "bold")


# --------------------------------------------------------------------------- 1 roadmap
def roadmap(path):
    s = Svg(800, 290)
    s.text(10, 18, "Research questions: where does training break, and which fix recovers it?",
           12, weight="bold")
    rqs = [("FP16", "no loss scaling", "fix: loss scaling"),
           ("BF16", "no master weights", "fix: master / SR"),
           ("FP8", "E4M3/E5M2 scaling", "fix: finer scales"),
           ("FP4", "MXFP4, NVFP4", "fix: SR, keep bwd/head"),
           ("Accumulate", "and sensitive ops", "fix: FP32 / promote")]
    w, gap = 148, 10
    for i, (t, a, b) in enumerate(rqs):
        box(s, 10 + i * (w + gap), 28, w, 62, t, a, RED, sub2=b)
    ms = [("M1 Emulation", "casts, quantisers,", BLUE, "3 GEMMs, accumulator"),
          ("M2 Instruments", "under/overflow,", GREEN, "update/weight, lost"),
          ("M3 Run matrix", "27 recipes x 2 models,", PURPLE, "sweeps, seeds"),
          ("M4 Analysis", "gap to fp32, verdicts,", AMBER, "half-ulp reproduction"),
          ("M5 Write-up", "paper, rubric,", GREY, "every number from JSON")]
    for i, (t, a, c, b) in enumerate(ms):
        x = 10 + i * (w + gap)
        box(s, x, 140, w, 62, t, a, c, sub2=b)
        if i:
            s.line(x - gap + 1, 171, x - 1, 171, arrow=True)
    s.line(400, 92, 400, 138, arrow=True, stroke=GREY, dash="4 3")
    s.rect(10, 222, 780, 56, fill="#ffffff", stroke=GREY, sw=1, rx=6, dash="5 3")
    s.text(400, 244, "Built only on Part I: llmsys.numerics (formats, rounding, scaling) and "
           "PyTorch autograd.", 11, "middle")
    s.text(400, 262, "The kit supplies the models (an MLP, a tiny GPT) and synthetic tasks "
           "whose optimal loss is known exactly.", 11, "middle")
    s.save(path)


# --------------------------------------------------------------------------- 2 three GEMMs
def three_gemms(path):
    s = Svg(820, 360)
    s.text(10, 18, "One linear layer, y = x W^T: three GEMMs, five quantised operands", 12,
           weight="bold")
    rows = [("forward", 40, [("X", "tokens x d_in"), ("W", "d_out x d_in")],
             ("Q_x, Q_w", "blocks along d_in"), "Y = Q(X) Q(W)^T", "store(Y)", "bf16 / fp16"),
            ("input grad", 140, [("dY", "tokens x d_out"), ("W", "d_out x d_in")],
             ("Q_g, Q_wb", "blocks along d_out"), "dX = Q(dY) Q(W)", "store(dX)", "to layer below"),
            ("weight grad", 240, [("dY", "tokens x d_out"), ("X", "tokens x d_in")],
             ("Q_g, Q_xb", "blocks along tokens"), "dW = Q(dY)^T Q(X)", "store(dW)",
             "then the optimiser")]
    for name, y, ops, (qn, qsub), gemm, out, outsub in rows:
        s.text(10, y + 34, name, 12, weight="bold")
        for j, (t, sub) in enumerate(ops):
            box(s, 100, y + j * 40, 120, 34, t, sub, GREEN, 11)
            s.line(222, y + 17 + j * 40, 268, y + 37, arrow=True)
        box(s, 270, y + 14, 150, 46, qn, qsub, RED, 11)
        s.line(422, y + 37, 458, y + 37, arrow=True)
        box(s, 460, y + 14, 170, 46, gemm, "accumulator: fp32 or emulated", BLUE, 11)
        s.line(632, y + 37, 668, y + 37, arrow=True)
        box(s, 670, y + 14, 140, 46, out, outsub, AMBER, 11)
    s.text(410, 340, "Per-tensor scales: Q(W) and Q(X) of the forward pass are reused. "
           "Block scales: W and X are re-quantised", 10, "middle", fill=GREY)
    s.text(410, 354, "along the new reduction dimension, so the backward pass sees "
           "different numbers than the forward pass.", 10, "middle", fill=GREY)
    s.save(path)


# --------------------------------------------------------------------------- plots
def _save(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _by_group(doc):
    out = {}
    for r in doc["results"]:
        out.setdefault(r["group"], r)
    return out


def _curve(r, key="val"):
    pts = [(rec["step"], rec[key]) for rec in r["records"] if key in rec]
    return [p[0] for p in pts], [p[1] for p in pts]


def update_ratios(doc, path):
    """Histogram of |delta|/|w| for the fp32 runs against the bf16 and fp16 half-ulp bands."""
    g = _by_group(doc)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    edges = 2.0 ** np.arange(-40, 1)
    for name, colour, label in (("gpt/fp32", BLUE, "tiny GPT, fp32"),
                                ("mlp/fp32", GREEN, "MLP, fp32")):
        if name not in g:
            continue
        recs = [(rec["step"], rec["update"]) for rec in g[name]["records"]
                if "ratio_hist" in rec.get("update", {})]
        h = np.array(recs[-1][1]["ratio_hist"], dtype=float)
        h = h / h.sum()
        ax.step(edges[:-1], h, where="post", color=colour, lw=1.6, label=label + " (last step)")
        tenth = min(recs, key=lambda sr: abs(sr[0] - 0.1 * g[name]["steps"]))[1]
        h0 = np.array(tenth["ratio_hist"], dtype=float)
        ax.step(edges[:-1], h0 / h0.sum(), where="post", color=colour, lw=1.0, ls="--",
                label=label + " (10% of training)")
    for fmt, colour, m in (("bf16", RED, 7), ("fp16", AMBER, 10)):
        lo, hi = 2.0 ** -(m + 2), 2.0 ** -(m + 1)
        ax.axvspan(lo, hi, color=colour, alpha=0.18)
        ax.text(math.sqrt(lo * hi), 1.01, f"{fmt} band", color=colour, ha="center",
                va="bottom", fontsize=9, transform=ax.get_xaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set_xlim(2.0 ** -26, 2.0 ** -2)
    ax.set_xlabel("|update| / |weight| per element")
    ax.set_ylabel("fraction of weights")
    ax.legend(fontsize=8, loc="center left")
    ax.set_title("Update-to-weight ratios against the half-ulp bands (" + MEASURED + ")",
                 fontsize=10, pad=18)
    _save(fig, path)


def bf16_master(doc, path):
    g = _by_group(doc)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    styles = {"fp32": (GREY, "-"), "bf16_pure": (RED, "-"), "bf16_pure_sr": (BLUE, "--")}
    for tag, lw in (("@lr=0.003", 0.9), ("", 1.6), ("@lr=0.0003", 0.9)):
        for rec, (c, ls) in styles.items():
            k = f"mlp/{rec}{tag}"
            if k in g:
                x, y = _curve(g[k])
                lr = {"": "1e-3", "@lr=0.003": "3e-3", "@lr=0.0003": "3e-4"}[tag]
                a1.plot(x, y, color=c, ls=ls, lw=lw, label=f"{rec}, lr {lr}" if tag == "" else None)
                if tag and rec == "bf16_pure":
                    a1.annotate(f"lr {lr}", (x[-1], y[-1]), fontsize=8, color=GREY,
                                xytext=(3, 0), textcoords="offset points")
    a1.set_yscale("log")
    a1.set_xlabel("step")
    a1.set_ylabel("validation MSE (floor 0)")
    a1.legend(fontsize=8)
    a1.set_title("MLP: BF16 weights without a master copy", fontsize=10)
    for k, c, label in (("mlp/bf16_pure", RED, "MLP, lr 1e-3"),
                        ("mlp/bf16_pure@lr=0.0003", AMBER, "MLP, lr 3e-4"),
                        ("gpt/bf16_pure", BLUE, "tiny GPT, lr 1e-2")):
        if k not in g:
            continue
        recs = [(rec["step"], rec["update"]) for rec in g[k]["records"] if "update" in rec]
        x = [s for s, _ in recs]
        a2.plot(x, [u["lost"] for _, u in recs], color=c, lw=2.2, alpha=0.5, label=label + ", measured")
        a2.plot(x, [u["pred_lost_bf16"] for _, u in recs], color=c, lw=1, ls=":",
                label=label + ", half-ulp rule")
    a2.set_xlabel("step")
    a2.set_ylabel("fraction of updates lost")
    a2.set_ylim(0, 1)
    a2.legend(fontsize=8)
    a2.set_title("Lost updates: measured vs predicted", fontsize=10)
    fig.suptitle(MEASURED, fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def grad_histogram(doc, path):
    g = _by_group(doc).get("gpt/fp32")
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    if g and g.get("hist"):
        for (step, h), c in zip(sorted(g["hist"].items(), key=lambda t: int(t[0])),
                                (GREEN, BLUE, PURPLE)):
            counts = np.array(h["counts"], dtype=float)
            tot = counts.sum() + h["zeros"]
            e = np.arange(h["lo"], h["hi"])
            ax.step(e, counts / tot, where="post", color=c, lw=1.5, label=f"step {int(step) + 1}")
    for fmt, c, y in ((N.FP16, AMBER, 0.93), (N.E5M2, RED, 0.83), (N.E4M3, GREY, 0.73)):
        lo, hi = math.log2(fmt.min_subnormal), math.log2(fmt.max_finite)
        ax.annotate("", xy=(lo, y), xytext=(hi, y), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="<->", color=c, lw=1.4))
        ax.axvline(math.log2(fmt.min_normal), color=c, lw=0.8, ls=":")
        ax.text(hi + 0.5, y, {"fp16": "FP16", "fp8_e5m2": "E5M2",
                               "fp8_e4m3": "E4M3"}[fmt.name] + " range", color=c, fontsize=9,
                va="center", transform=ax.get_xaxis_transform())
    ax.set_xlim(-50, 28)
    ax.set_xlabel("log2 |activation gradient| (unscaled, all linear-layer outputs)")
    ax.set_ylabel("fraction of values")
    ax.legend(fontsize=8, loc="center left")
    ax.set_title("Gradients reaching the GEMMs of the tiny GPT in fp32 (" + MEASURED + ")",
                 fontsize=10)
    _save(fig, path)


def fp16_sweep(doc, path):
    rows = {r["group"]: r for r in doc["summary"]}
    g = _by_group(doc)
    ws = [1.0] + sorted({float(k.split("=")[1]) for k in rows if "@loss_weight=" in k},
                        reverse=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for rec, c in (("fp16_noscale", RED), ("fp16_static", AMBER), ("fp16_dynamic", BLUE)):
        xs, gaps, unders = [], [], []
        for w in ws:
            k = f"mlp/{rec}" if w == 1.0 else f"mlp/{rec}@loss_weight={w:g}"
            if k not in rows:
                continue
            xs.append(w)
            gaps.append(rows[k]["gap"])
            recs = [rr["casts"]["g"] for rr in g[k]["records"] if "g" in rr.get("casts", {})]
            unders.append((max(np.nanmean([q["underflow"] for q in recs]), 1e-9),
                           max(np.nanmean([q.get("underflow_mass", 0.0) for q in recs]), 1e-9)))
        if xs:
            a1.plot(xs, [u[0] for u in unders], "o-", color=c, label=rec + ", count")
            a1.plot(xs, [u[1] for u in unders], "s--", color=c, ms=4, label=rec + ", mass")
            a2.plot(xs, gaps, "o-", color=c, label=rec)
    for a in (a1, a2):
        a.set_xscale("log")
        a.invert_xaxis()
        a.set_xlabel("loss weight w (gradients scaled by w)")
    a1.set_yscale("log")
    a1.set_ylabel("share flushed to zero (mean over run)")
    a1.set_title("Underflow of the GEMM gradient inputs", fontsize=10)
    a1.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    a2.axhline(0.10, color=GREY, lw=0.8, ls=":")
    a2.axhline(0.69, color=GREY, lw=0.8, ls="--")
    a2.set_ylabel("gap = ln(excess / excess of fp32)")
    a2.set_title("Effect on the final loss (MLP)", fontsize=10)
    a2.legend(fontsize=8)
    fig.suptitle(MEASURED + "; fp32 is identical at every w by construction", fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def low_bit_curves(doc, path):
    g = _by_group(doc)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    fp8 = [("fp32", GREY, "-"), ("fp8_noscale", RED, "-"), ("fp8_tensor", BLUE, "-"),
           ("fp8_delayed", PURPLE, "--"), ("fp8_tensor_hp", GREEN, "-"),
           ("fp8_tensor_sr", GREEN, "--"), ("fp8_e4m3_tensor", AMBER, "-."),
           ("fp8_block", AMBER, ":")]
    fp4 = [("fp32", GREY, "-"), ("mxfp4", RED, "-"), ("mxfp4_sr", RED, "--"),
           ("mxfp4_sr_ceil", AMBER, "-"), ("mxfp4_fwd", GREEN, "-."), ("nvfp4", PURPLE, "-"),
           ("nvfp4_sr", BLUE, "--"), ("nvfp4_sr_hp", BLUE, ":")]
    floor = None
    for ax, series, title in ((a1, fp8, "FP8 recipes"), (a2, fp4, "FP4 recipes")):
        for rec, c, ls in series:
            k = f"gpt/{rec}"
            if k in g:
                x, y = _curve(g[k])
                ax.plot(x, y, color=c, ls=ls, lw=1.4, label=rec)
                floor = g[k]["floor"]
        if floor is not None:
            ax.axhline(floor, color=INK, lw=0.8, ls=":")
            ax.text(1.0, floor + 0.03, "entropy floor", fontsize=8, ha="right",
                    transform=ax.get_yaxis_transform())
        ax.set_xlabel("step")
        ax.set_title("tiny GPT: " + title, fontsize=10)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.set_ylim(1.7, 4.9)
    a1.set_ylabel("validation loss (nats/token)")
    fig.suptitle(MEASURED + " (emulated formats: the A100 has no FP8/FP4 tensor cores)",
                 fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


VERDICT_COLOUR = {"ok": GREEN, "degraded": AMBER, "broken": RED, "broken, stalled": RED,
                  "diverged": PURPLE, "baseline": GREY}


def gap_bars(doc, path):
    rows = [r for r in doc["summary"] if "@" not in r["group"] and r["recipe"] != "fp32"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2))
    for ax, model in zip(axes, ("gpt", "mlp")):
        rr = [r for r in rows if r["model"] == model]
        names = [r["recipe"] for r in rr]
        vals = [r["gap"] if math.isfinite(r["gap"]) else 3.0 for r in rr]
        cols = [VERDICT_COLOUR.get(r["verdict"].split("/")[0], GREY) for r in rr]
        y = np.arange(len(rr))
        ax.barh(y, np.clip(vals, -0.3, 3.0), color=cols)
        for yi, r, v in zip(y, rr, vals):
            lab = "diverged" if not math.isfinite(r["gap"]) else f"{r['gap']:+.2f}"
            ax.text(max(min(v, 3.0), 0) + 0.04, yi, lab, va="center", fontsize=8, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0.10, color=GREY, lw=0.8, ls=":")
        ax.axvline(0.69, color=GREY, lw=0.8, ls="--")
        ax.set_xlim(-0.3, 3.6)
        ax.set_xlabel("gap to fp32 = ln(excess loss / excess loss of fp32)")
        ax.set_title({"gpt": "tiny GPT", "mlp": "MLP"}[model], fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, AMBER, RED, PURPLE)]
    fig.legend(handles, ["ok", "degraded", "broken", "diverged"], loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Every recipe of the smoke matrix, one seed (" + MEASURED + ")", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    _save(fig, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=os.path.join(HERE, "results", "smoke_a100_pcie.json"))
    ap.add_argument("--out-dir", default=BOOK_FIGS)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    f = lambda k: os.path.join(a.out_dir, f"{STEM}-{k}.svg")         # noqa: E731
    roadmap(f(1))
    three_gemms(f(2))
    if os.path.exists(a.json):
        with open(a.json) as fh:
            doc = json.load(fh)
        update_ratios(doc, f(3))
        bf16_master(doc, f(4))
        grad_histogram(doc, f(5))
        fp16_sweep(doc, f(6))
        low_bit_curves(doc, f(7))
        gap_bars(doc, f(8))
    return 0


if __name__ == "__main__":
    sys.exit(main())
