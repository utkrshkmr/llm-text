"""Lab 2: draw the chapter's roofline figures as SVG (house palette, light theme).

    python labs/lab02_pytorch_roofline/figures.py                       # reference JSON
    python labs/lab02_pytorch_roofline/figures.py --json results/roofline.json --out-dir /tmp

Writes ch02_pytorch_hardware-11.svg (the measured empirical roofline with the kernel
catalogue) and ch02_pytorch_hardware-12.svg (analytic operational intensity of GEMM and
attention) into --out-dir, by default the book's figures directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import solution as lab  # noqa: E402

BLUE, AMBER, PURPLE = "#1f4e79", "#8a5a00", "#6a4c93"
GREEN, RED, GREY = "#2e6b30", "#9b2c2c", "#555555"
BOOK_FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "book", "figures")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": "#000000",
                     "xtick.color": "#000000", "ytick.color": "#000000", "svg.hashsalt": "ch02",
                     "font.family": "DejaVu Sans"})

LABELS = {"add": "add", "rowsum": "row sum", "softmax": "softmax", "gemv": "GEMV (M=1)",
          "gemm_m16": "GEMM M=16", "gemm_m256": "GEMM M=256", "gemm_square": "GEMM 8192³",
          "attn_prefill": "attention prefill", "attn_decode": "attention decode"}
COLOUR = {"add": AMBER, "rowsum": AMBER, "softmax": AMBER, "gemv": BLUE, "gemm_m16": BLUE,
          "gemm_m256": BLUE, "gemm_square": BLUE, "attn_prefill": PURPLE, "attn_decode": PURPLE}
OFFSET = {"add": (7, -12), "rowsum": (7, -15), "softmax": (9, -6), "gemv": (8, 2),
          "gemm_m16": (8, -12), "gemm_m256": (5, -17), "gemm_square": (8, -14),
          "attn_prefill": (8, -30), "attn_decode": (8, -12)}


def measured(res: dict, path: str) -> None:
    pf, pb = res["roof"]["peak_flops"] / 1e12, res["roof"]["peak_bw"] / 1e12
    ds_f, ds_b = res["datasheet"]["flops"] / 1e12, res["datasheet"]["bw"] / 1e12
    xs = np.logspace(-1, 5, 400)
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.loglog(xs, np.minimum(ds_f, xs * ds_b), "--", color=GREY, lw=1.2,
              label=f"datasheet: {ds_f:.0f} TFLOP/s, {ds_b*1e3:,.0f} GB/s")
    ax.loglog(xs, np.minimum(pf, xs * pb), "-", color=RED, lw=1.8,
              label=f"measured: {pf:.0f} TFLOP/s (GEMM), {pb*1e3:,.0f} GB/s (copy)")
    ridge = pf / pb
    ax.axvline(ridge, color=RED, lw=0.8, ls=":")
    ax.text(ridge / 1.1, 0.55, f"ridge {ridge:.0f} FLOP/B", color=RED, fontsize=9, ha="right")
    for p in res["kernels"]:
        x, y = p["intensity"], p["flops_per_s"] / 1e12
        ax.plot([x], [y], "o", color=COLOUR[p["name"]], ms=6)
        ax.annotate(f"{LABELS[p['name']]} ({100*p['fraction_of_roof']:.0f}%)", (x, y),
                    textcoords="offset points", xytext=OFFSET[p["name"]], fontsize=9,
                    color=COLOUR[p["name"]])
    ax.set_xlim(0.1, 1e5)
    ax.set_ylim(0.1, 500)
    ax.set_xlabel("operational intensity at HBM (FLOP/byte)")
    ax.set_ylabel("achieved TFLOP/s (bf16)")
    ax.grid(True, which="major", color="#dddddd", lw=0.6)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    m = res["meta"]
    ax.set_title(f"measured on {m['hardware']}; torch {m['torch'].split('+')[0]}, "
                 f"CUDA {m['cuda']}", fontsize=9, color=GREY)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor="#ffffff")
    plt.close(fig)


def analytic(path: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5))
    ms = np.unique(np.logspace(0, 4.2, 120).astype(int))
    for nk, ls in ((8192, "-"), (2048, "--")):
        a1.loglog(ms, [lab.matmul_intensity(m, nk, nk, 2) for m in ms], ls, color=BLUE, lw=1.6,
                  label=f"N = K = {nk}")
    for name, ridge, c, dy in (("A100 PCIe ridge (datasheet)", 312 / 1.555, RED, 0.62),
                               ("H100 SXM ridge", 989 / 3.35, AMBER, 1.18)):
        a1.axhline(ridge, color=c, lw=1.0, ls=":")
        a1.text(1.3, ridge * dy, f"{name} {ridge:.0f}", color=c, fontsize=9)
    a1.set_xlabel("M (tokens in the batch)")
    a1.set_ylabel("FLOP/byte (bf16)")
    a1.set_title("(a) GEMM: M x K times K x N", fontsize=10)
    a1.legend(loc="lower right", fontsize=9, frameon=False)
    a1.grid(True, color="#dddddd", lw=0.6)

    ts = np.unique(np.logspace(2, 5, 80).astype(int))
    a2.loglog(ts, [lab.attention_intensity(t, t, 32, 32, 128, 2, causal=True) for t in ts],
              "-", color=PURPLE, lw=1.6, label="prefill, causal")
    for g, ls, lbl in ((32, "--", "decode, MHA (G = H)"), (8, "-.", "decode, GQA H/G = 4"),
                       (4, ":", "decode, GQA H/G = 8")):
        a2.loglog(ts, [lab.attention_intensity(1, t, 32, g, 128, 2) for t in ts], ls,
                  color=GREEN, lw=1.6, label=lbl)
    a2.axhline(312 / 1.555, color=RED, lw=1.0, ls=":")
    a2.text(1.2e4, 312 / 1.555 * 0.62, "A100 ridge", color=RED, fontsize=9)
    a2.set_xlabel("context length T (tokens)")
    a2.set_ylim(0.5, 1e5)
    a2.set_title("(b) fused attention, 32 heads, $d_h$ = 128", fontsize=10)
    a2.legend(loc="upper left", fontsize=9, frameon=False)
    a2.grid(True, color="#dddddd", lw=0.6)
    fig.suptitle("analytic, minimum HBM traffic", fontsize=9, color=GREY, y=0.99)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor="#ffffff")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=os.path.join(HERE, "reference_a100", "roofline.json"))
    ap.add_argument("--out-dir", default=BOOK_FIGS)
    a = ap.parse_args()
    with open(a.json) as f:
        res = json.load(f)
    measured(res, os.path.join(a.out_dir, "ch02_pytorch_hardware-11.svg"))
    analytic(os.path.join(a.out_dir, "ch02_pytorch_hardware-12.svg"))
    print("wrote figures to", a.out_dir)


if __name__ == "__main__":
    main()
