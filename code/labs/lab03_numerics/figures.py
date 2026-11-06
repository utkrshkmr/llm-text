"""Lab 3: draw the chapter's figures as SVG (house palette, light theme).

    python labs/lab03_numerics/figures.py                       # into book/figures
    python labs/lab03_numerics/figures.py --json results/bench_results.json --out-dir /tmp/f

Diagrams are written as hand-made SVG; plots use matplotlib. Analytic plots come from
llmsys.numerics; the measured ones read bench.py's JSON (measured on 2x A100-40GB (PCIe)).
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
import torch  # noqa: E402

from llmsys import numerics as N  # noqa: E402

BLUE, AMBER, PURPLE, GREEN, RED, GREY = ("#1f4e79", "#8a5a00", "#4b2c83", "#2e6b30", "#9b2c2c",
                                         "#555555")
TINT = {BLUE: "#dce6f1", AMBER: "#f3e9d2", PURPLE: "#e6e0f0", GREEN: "#deeedc", RED: "#f2dcdc",
        GREY: "#eeeeee"}
BOOK_FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "book", "figures")
STEM = "ch03_numerics"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": "#000000",
                     "xtick.color": "#000000", "ytick.color": "#000000", "svg.hashsalt": "ch03",
                     "font.family": "DejaVu Sans", "figure.facecolor": "#ffffff",
                     "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff"})
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"


# --------------------------------------------------------------------------- SVG helpers
class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill="#ffffff", stroke="#000000", sw=1.0, rx=0):
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                          f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

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

    def save(self, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">\n<defs><marker id="arr" viewBox="0 0 10 10" '
                'refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                '<path d="M 0 0 L 10 5 L 0 10 z" fill="#000000"/></marker></defs>\n'
                f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>\n')
        with open(path, "w") as f:
            f.write(head + "\n".join(self.items) + "\n</svg>\n")


def box(s: Svg, x, y, w, h, title, sub=None, colour=BLUE, size=12):
    s.rect(x, y, w, h, fill=TINT[colour], stroke=colour, sw=1.4, rx=6)
    if sub:
        s.text(x + w / 2, y + h / 2 - 3, title, size, "middle", "bold")
        s.text(x + w / 2, y + h / 2 + 13, sub, size - 2, "middle")
    else:
        s.text(x + w / 2, y + h / 2 + 4, title, size, "middle", "bold")


# --------------------------------------------------------------------------- 1 roadmap
def roadmap(path):
    s = Svg(760, 250)
    row1 = [("1 Bytes are the budget", "memory, bandwidth, FLOP/s", BLUE),
            ("2 Integers, fixed point", "two's complement, saturation", BLUE),
            ("3 IEEE 754", "FP32, TF32, FP16, BF16", BLUE),
            ("4 FP8, FP6, FP4, MX", "block scaling, NVFP4", BLUE)]
    row2 = [("5 Rounding", "RNE, stochastic, Kahan", GREEN),
            ("6 Accumulation", "error bounds, promotion", GREEN),
            ("7 Quantisation maths", "scales, grids, Hessian", GREEN),
            ("8 Mixed precision", "recipes, DeepSeek-V3 FP8", PURPLE)]
    w, h, gap = 170, 56, 20
    for i, (t, sub, c) in enumerate(row1):
        box(s, 10 + i * (w + gap), 20, w, h, t, sub, c)
        if i:
            s.line(10 + i * (w + gap) - gap + 2, 48, 10 + i * (w + gap) - 2, 48, arrow=True)
    for i, (t, sub, c) in enumerate(row2):
        box(s, 10 + i * (w + gap), 110, w, h, t, sub, c)
        if i:
            s.line(10 + i * (w + gap) - gap + 2, 138, 10 + i * (w + gap) - 2, 138, arrow=True)
    s.line(10 + 3 * (w + gap) + w / 2, 76, 10 + w / 2, 110, arrow=True, stroke=GREY, dash="4 3")
    box(s, 10, 190, 360, 46, "9 Stable online algorithms",
        "online softmax (FlashAttention's root), Welford, cross-entropy", AMBER)
    box(s, 390, 190, 360, 46, "Lab: a bit-exact format simulator",
        "tested against torch casts; bench on the A100", RED)
    s.line(10 + 3 * (w + gap) + w / 2, 166, 190, 190, arrow=True, stroke=GREY, dash="4 3")
    s.line(370, 213, 388, 213, arrow=True)
    s.save(path)


# --------------------------------------------------------------------------- 2 bit layouts
def bit_layouts(path):
    rows = [N.FP32, N.TF32, N.BF16, N.FP16, N.E4M3, N.E5M2, N.E3M2, N.E2M3, N.E2M1]
    cw, x0, y0, rh = 13, 140, 30, 30
    s = Svg(x0 + 32 * cw + 210, y0 + rh * (len(rows) + 1) + 30)
    s.text(x0, 18, "sign", 11, fill=RED)
    s.text(x0 + 40, 18, "exponent", 11, fill=BLUE)
    s.text(x0 + 110, 18, "fraction (mantissa)", 11, fill=GREEN)
    s.text(x0 + 32 * cw + 12, 18, "max | eps | smallest normal", 11, weight="bold")
    for r, f in enumerate(rows + ["e8m0"]):
        y = y0 + r * rh
        if f == "e8m0":
            name, fields = "E8M0 (MX scale)", [(BLUE, 8)]
            info = "2^-127 ... 2^127, no sign, no zero"
        else:
            name = {"fp32": "FP32", "tf32": "TF32 (compute)", "bf16": "BF16", "fp16": "FP16",
                    "fp8_e4m3": "FP8 E4M3", "fp8_e5m2": "FP8 E5M2", "fp6_e3m2": "FP6 E3M2",
                    "fp6_e2m3": "FP6 E2M3", "fp4_e2m1": "FP4 E2M1"}[f.name]
            fields = [(RED, 1), (BLUE, f.ebits), (GREEN, f.mbits)]
            mx = f"{f.max_finite:.3g}" if f.max_finite < 1e5 else f"{f.max_finite:.1e}"
            info = f"{mx} | 2^-{f.mbits} | 2^{f.emin}"
        s.text(x0 - 8, y + 18, name, 12, "end", "bold")
        x = x0
        for colour, n in fields:
            for _ in range(n):
                s.rect(x, y + 4, cw, 20, fill=TINT[colour], stroke=colour, sw=0.8)
                x += cw
            if n >= 2:
                s.text(x - n * cw / 2, y + 18, str(n), 10, "middle", fill=colour)
        s.text(x0 + 32 * cw + 12, y + 18, info, 11)
    s.save(path)


# --------------------------------------------------------------------------- 3 number line
def number_line(path):
    s = Svg(760, 250)
    x0, x1 = 40, 730

    def axis(y, vmax, vals, label, colour, ticks):
        X = lambda v: x0 + (x1 - x0) * v / vmax                            # noqa: E731
        s.line(x0, y, x1, y, stroke=GREY)
        for v in ticks:
            s.line(X(v), y + 2, X(v), y + 8, stroke=GREY)
            s.text(X(v), y + 21, f"{v:g}", 10, "middle", fill=GREY)
        for v, sub in vals:
            s.line(X(v), y - 12, X(v), y, stroke=RED if sub else colour, sw=1.6)
        s.text(x0, y - 58, label, 12, weight="bold")
        return X

    v = [float(t) for t in N.all_values(N.E3M2) if 0 <= t <= 2.0]
    vals = [(t, 0 < t < N.E3M2.min_normal) for t in sorted(set(v))]
    X = axis(100, 2.0, vals, "FP6 E3M2, every value in [0, 2]; red ticks are subnormals", BLUE,
             [0, 0.25, 0.5, 0.75, 1, 1.5, 2])
    for lo, hi, step, c in [(0, 0.25, "0.0625", RED), (0.25, 0.5, "0.0625", BLUE),
                            (0.5, 1, "0.125", BLUE), (1, 2, "0.25", BLUE)]:
        s.line(X(lo) + 2, 66, X(hi) - 2, 66, stroke=c, sw=0.8)
        s.line(X(lo) + 2, 62, X(lo) + 2, 70, stroke=c, sw=0.8)
        s.line(X(hi) - 2, 62, X(hi) - 2, 70, stroke=c, sw=0.8)
        s.text((X(lo) + X(hi)) / 2, 60, f"step {step}", 10, "middle", fill=c)
    v2 = [float(t) for t in N.all_values(N.E2M1) if t >= 0]
    vals2 = [(t, 0 < t < N.E2M1.min_normal) for t in sorted(set(v2))]
    axis(215, 6.0, vals2, "FP4 E2M1, all non-negative values: 0, 0.5, 1, 1.5, 2, 3, 4, 6", GREEN,
         [0, 0.5, 1, 1.5, 2, 3, 4, 6])
    s.save(path)


# --------------------------------------------------------------------------- 4 relative spacing
def rel_spacing(path):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for f, c, lab in [(N.FP32, GREY, "FP32"), (N.BF16, BLUE, "BF16"), (N.FP16, GREEN, "FP16"),
                      (N.E5M2, AMBER, "FP8 E5M2"), (N.E4M3, RED, "FP8 E4M3"),
                      (N.E2M1, PURPLE, "FP4 E2M1")]:
        lo, hi = math.log10(f.min_subnormal), math.log10(f.max_finite)
        lo, hi = max(lo, -9), min(hi, 6)
        x = torch.logspace(lo, hi, 6000, dtype=torch.float64)
        y = N.ulp(x, f).double() / x
        ax.loglog(x.numpy(), y.numpy(), color=c, lw=1.3, label=lab)
    ax.set_xlabel("magnitude |x|")
    ax.set_ylabel("relative spacing ulp(x)/x")
    ax.set_xlim(1e-9, 1e6)
    ax.set_ylim(3e-8, 2)
    ax.legend(fontsize=9, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              handlelength=1.2, columnspacing=0.8)
    ax.text(2e-9, 1.5e-7, "FP32 and BF16 continue flat to 1e38 (subnormals below 1e-38)",
            fontsize=9, color=GREY)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


# --------------------------------------------------------------------------- 5 MX / NVFP4
def mx_layout(path):
    s = Svg(760, 265)
    cw = 13
    s.text(10, 22, "MXFP4 (OCP): one E8M0 scale per 32 elements", 13, weight="bold")
    s.rect(10, 34, 46, 26, fill=TINT[BLUE], stroke=BLUE, sw=1.2)
    s.text(33, 51, "E8M0", 11, "middle", "bold", BLUE)
    for i in range(32):
        s.rect(64 + i * cw, 34, cw, 26, fill=TINT[GREEN], stroke=GREEN, sw=0.7)
    s.text(64 + 16 * cw, 76, "32 x E2M1 (4 bits): x_i = 2^X * p_i", 11, "middle")
    s.text(64 + 32 * cw + 10, 46, "8 + 32 x 4 = 136 bits", 11)
    s.text(64 + 32 * cw + 10, 61, "= 4.25 bits per element", 11)
    s.text(10, 112, "NVFP4: one E4M3 scale per 16 elements, and one FP32 scale per tensor", 13,
           weight="bold")
    s.rect(10, 124, 46, 26, fill=TINT[AMBER], stroke=AMBER, sw=1.2)
    s.text(33, 141, "E4M3", 11, "middle", "bold", AMBER)
    for i in range(16):
        s.rect(64 + i * cw, 124, cw, 26, fill=TINT[GREEN], stroke=GREEN, sw=0.7)
    s.rect(64 + 16 * cw + 12, 124, 46, 26, fill=TINT[AMBER], stroke=AMBER, sw=1.2)
    s.text(64 + 16 * cw + 35, 141, "E4M3", 11, "middle", "bold", AMBER)
    for i in range(16):
        s.rect(64 + 16 * cw + 66 + i * cw, 124, cw, 26, fill=TINT[GREEN], stroke=GREEN, sw=0.7)
    s.text(64 + 32 * cw + 80, 136, "(8 + 16 x 4) / 16", 11)
    s.text(64 + 32 * cw + 80, 151, "= 4.5 bits per element", 11)
    s.rect(10, 170, 150, 26, fill=TINT[PURPLE], stroke=PURPLE, sw=1.2)
    s.text(85, 187, "FP32 tensor scale g", 11, "middle", "bold", PURPLE)
    s.text(172, 187, "x_i = g * s_b * p_i,  s_b in E4M3 holds amax_b / (6 g),"
           "  g = amax / (6 * 448)", 11)
    s.text(10, 225, "A power-of-two scale (E8M0) can leave the block's largest element up to one "
           "binade below the top of the grid;", 11, fill=GREY)
    s.text(10, 242, "an E4M3 scale has 3 fraction bits, so it maps amax_b to within about 6% of "
           "6 for every block.", 11, fill=GREY)
    s.save(path)


# --------------------------------------------------------------------------- 6 grids
def grids(path):
    fig, axes = plt.subplots(4, 1, figsize=(6.6, 5.0), sharex=True)
    xs = np.linspace(-1.1, 1.1, 400)
    pdf = np.exp(-0.5 * (xs / 0.33) ** 2)
    int4 = np.arange(-7, 8) / 7
    torch.manual_seed(0)
    skew = (torch.rand(10000) * 1.6 - 0.6).double()        # values in [-0.6, 1.0]
    q = N.quantize(skew.reshape(1, -1), N.UINT4, block="tensor", symmetric=False)
    s0, z0 = float(q.scale.flatten()[0]), float(q.zero.flatten()[0])
    asym = (np.arange(16) - z0) * s0
    rows = [("INT4 symmetric (15 levels)", int4, BLUE),
            ("UINT4 + zero point on [-0.6, 1]", asym, GREEN),
            ("NF4 (normal quantiles)", np.array(N.NF4.values), PURPLE),
            ("FP4 E2M1 / 6", np.array([float(v) for v in N.all_values(N.E2M1)]) / 6, RED)]
    for ax, (name, g, c) in zip(axes, rows):
        ax.fill_between(xs, 0, pdf, color="#eeeeee")
        ax.vlines(g, 0, 1, color=c, lw=1.4)
        ax.set_yticks([])
        ax.set_ylim(0, 1.05)
        ax.set_title(name, loc="left", fontsize=9, color=c, pad=2)
    axes[-1].set_xlabel("normalised value x / scale (shaded: a Gaussian weight distribution)")
    fig.tight_layout(h_pad=0.3)
    fig.savefig(path, format="svg")
    plt.close(fig)


# --------------------------------------------------------------------------- 7 clipping MSE
def clipping(path):
    torch.manual_seed(0)
    g = torch.randn(1 << 19, dtype=torch.float64)
    lap = torch.distributions.Laplace(0.0, 1 / math.sqrt(2)).sample((1 << 19,)).double()
    ratios = np.linspace(0.08, 1.0, 70)
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    for x, nm, c in [(g, "Gaussian", BLUE), (lap, "Laplace", RED)]:
        amax = float(x.abs().max())
        for bits, ls in [(4, "-"), (6, "--")]:
            fmt = N.IntFormat("i", bits, narrow=True)
            sw = N.clip_sweep(x.reshape(1, -1), fmt, ratios, block="tensor")
            cl = np.array([r * amax for r, _ in sw])
            mse = np.array([m for _, m in sw])
            ax.semilogy(cl, mse, ls, color=c, lw=1.4, label=f"{nm}, INT{bits}")
            k = int(np.argmin(mse))
            ax.plot(cl[k], mse[k], "o", color=c, ms=5)
            ax.annotate(f"{cl[k]:.2f}σ", (cl[k], mse[k]), textcoords="offset points",
                        xytext=(-4, -13), fontsize=9, color=c, ha="center")
    ax.set_xlabel("clipping threshold (multiples of the standard deviation σ)")
    ax.set_ylabel("mean squared error")
    ax.set_title("Computed on 2^19 samples: clipping error vs rounding error", fontsize=10)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


# --------------------------------------------------------------------------- 8 SR + updates
def sr_updates(path, res):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1, 1.5]})
    a.plot([0, 1], [0, 0], color=GREY)
    a.plot([0, 1], [0, 0], "o", color=BLUE, ms=8)
    a.plot([0.3], [0], "o", color=RED, ms=7)
    a.annotate("", xy=(0.02, 0.06), xytext=(0.28, 0.06),
               arrowprops=dict(arrowstyle="->", color=BLUE))
    a.annotate("", xy=(0.98, 0.06), xytext=(0.32, 0.06),
               arrowprops=dict(arrowstyle="->", color=BLUE))
    a.text(0.15, 0.1, "p = 0.7", ha="center", fontsize=9, color=BLUE)
    a.text(0.65, 0.1, "p = 0.3", ha="center", fontsize=9, color=BLUE)
    a.text(0.3, -0.08, "x", ha="center", fontsize=10, color=RED)
    a.text(0, -0.08, "lo", ha="center", fontsize=9)
    a.text(1, -0.08, "hi = lo + ulp", ha="center", fontsize=9)
    a.text(0.5, -0.2, "E[round(x)] = 0.7 lo + 0.3 hi = x\nRNE always returns lo",
           ha="center", fontsize=9)
    a.set_xlim(-0.15, 1.2)
    a.set_ylim(-0.3, 0.25)
    a.axis("off")
    a.set_title("stochastic rounding", fontsize=10)
    if res and "updates" in res:
        c = res["updates"]["curve"]
        for k, col, lab in [("bf16_rne", RED, "BF16 weights, RNE"),
                            ("bf16_sr", BLUE, "BF16 weights, stochastic"),
                            ("fp32_master", GREEN, "FP32 master, BF16 copy")]:
            b.semilogy(c["step"], c[k], color=col, lw=1.5)
            b.text(9800, c[k][-1] * 1.35, lab, color=col, fontsize=9, ha="right")
        b.set_xlabel("step")
        b.set_ylabel("rel. error of total update")
        b.set_ylim(1e-3, 3)
        u = res["updates"]
        b.set_title(f"measured on 2x A100-40GB (PCIe): {u['n']:,} weights,\n"
                    f"std 0.02, updates ~1e-5", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


# --------------------------------------------------------------------------- 9 accumulation
def accumulation(path, res):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 4.0))
    rows = [r for r in res["accum"] if r["dist"] == "uniform"]
    K = [r["K"] for r in rows]
    for key, col, lab, ls in [("sim_bf16_acc", RED, "BF16 accumulator (sim)", "-"),
                              ("sim_fp16_acc", AMBER, "FP16 accumulator (sim)", "-"),
                              ("sim_fp16_acc_splitk8", PURPLE, "FP16 acc., split-K 8", "-"),
                              ("cublas_fp16_out_fp32", BLUE, "cuBLAS FP16, FP32 out", "--"),
                              ("cublas_bf16_out_fp32", GREY, "cuBLAS BF16, FP32 out", "--"),
                              ("sim_fp32_acc", GREEN, "FP32 accumulator (sim)", "-")]:
        a.loglog(K, [r[key] for r in rows], ls, marker="o", ms=3, color=col, lw=1.3, label=lab)
    a.set_xlabel("K (inner dimension)")
    a.set_ylabel("relative error of A B")
    a.set_title("FP16 inputs in U[0,1)", fontsize=10)
    fig.legend(*a.get_legend_handles_labels(), fontsize=8.5, frameon=False, ncol=3,
               loc="lower center")
    d = res["deepseek"]["vs_interval"]
    b.loglog([r["interval"] for r in d], [r["accum"] for r in d], "o-", color=BLUE, lw=1.4,
             ms=4)
    b.axvline(128, color=GREY, ls=":", lw=1)
    b.text(135, d[0]["accum"], "128 (DeepSeek-V3)", fontsize=9, color=GREY)
    b.set_xlabel("promotion interval (elements of K)")
    b.set_ylabel("accumulation error")
    b.set_title(f"FP8 E4M3, 14-bit accumulator, K={d[0]['K']}", fontsize=10)
    fig.suptitle("Simulated on 2x A100-40GB (PCIe); cuBLAS curves measured", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, format="svg")
    plt.close(fig)


# --------------------------------------------------------------------------- 10 granularity
def granularity(path, res):
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    styles = {"fp8_e4m3": (BLUE, "o"), "int8": (GREEN, "s"), "fp4_e2m1": (RED, "o"),
              "int4": (AMBER, "s")}
    for fmt, (c, m) in styles.items():
        rows = [r for r in res["blocks"] if r["format"] == fmt]
        ax.semilogy([r["bits_per_elem"] for r in rows], [r["rel_err"] for r in rows], m + "-",
                    color=c, lw=1.2, ms=4, label=fmt.replace("_", " ").upper())
        for r in rows:
            if fmt in ("fp4_e2m1", "int8"):
                off = (4, -11) if r["block"] == "row" else (3, 4)
                ax.annotate(r["block"], (r["bits_per_elem"], r["rel_err"]), fontsize=8,
                            textcoords="offset points", xytext=off, color=c)
    for r in res["blocks"]:
        if r["format"] == "mxfp4_ceil":
            continue                                    # sits on top of MXFP4
        if r["scale"] != "fp32":
            ax.plot(r["bits_per_elem"], r["rel_err"], "*", color=PURPLE, ms=9)
            ax.annotate(r["format"].upper().replace("_E4M3", "").replace("_CEIL", " ceil"),
                        (r["bits_per_elem"], r["rel_err"]), fontsize=8,
                        textcoords="offset points", xytext=(5, -10), color=PURPLE)
    ax.set_xlabel("bits per element including scales")
    ax.set_ylabel("relative error of X W^T")
    ax.set_title("Emulated on 2x A100-40GB (PCIe): activations with 8 outlier channels (x30)",
                 fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


# --------------------------------------------------------------------------- 11 online softmax
def online_softmax(path):
    s = Svg(760, 240)
    s.text(10, 20, "One pass over blocks of scores s; state (m, l, o) lives in registers", 13,
           weight="bold")
    bw, x0 = 168, 12
    for j in range(4):
        x = x0 + j * (bw + 18)
        s.rect(x, 36, bw, 30, fill=TINT[BLUE], stroke=BLUE, sw=1.2, rx=4)
        s.text(x + bw / 2, 56, f"scores s_{j + 1}, values V_{j + 1}", 11, "middle")
        s.rect(x, 100, bw, 76, fill=TINT[GREEN], stroke=GREEN, sw=1.2, rx=4)
        s.text(x + 8, 118, f"m_{j + 1} = max(m_{j}, max s_{j + 1})", 11)
        s.text(x + 8, 136, f"a = exp(m_{j} - m_{j + 1})", 11)
        s.text(x + 8, 154, f"l_{j + 1} = a l_{j} + sum e^(s - m_{j + 1})", 10)
        s.text(x + 8, 170, f"o_{j + 1} = a o_{j} + e^(s - m) V", 10)
        s.line(x + bw / 2, 66, x + bw / 2, 98, arrow=True)
        if j:
            s.line(x - 17, 138, x - 2, 138, arrow=True, stroke=GREEN)
    s.text(10, 205, "Start: m_0 = -inf, l_0 = 0, o_0 = 0.  End: softmax(s) V = o / l, "
           "logsumexp(s) = m + log l.", 11)
    s.text(10, 225, "Rescaling identity: e^(s - m_old) = e^(s - m_new) e^(m_new - m_old), so "
           "old partial sums are corrected by one multiply.", 11, fill=GREY)
    s.save(path)


# --------------------------------------------------------------------------- 12 DeepSeek FP8
def deepseek(path):
    s = Svg(760, 300)
    box(s, 10, 20, 200, 50, "Activations (BF16)", "1x128 tiles, scale per tile", BLUE, 12)
    box(s, 10, 100, 200, 50, "Weights (FP32 master)", "128x128 blocks, scale per block",
        BLUE, 12)
    box(s, 250, 20, 140, 50, "online amax", "-> E4M3 + FP32 scale", AMBER, 12)
    box(s, 250, 100, 140, 50, "online amax", "-> E4M3 + FP32 scale", AMBER, 12)
    s.line(210, 45, 248, 45, arrow=True)
    s.line(210, 125, 248, 125, arrow=True)
    s.rect(430, 15, 320, 145, fill="#ffffff", stroke=PURPLE, sw=1.4, rx=6)
    s.text(590, 34, "one output tile, K loop", 12, "middle", "bold", PURPLE)
    box(s, 445, 45, 140, 50, "Tensor core", "4 WGMMAs (K = 4 x 32)", PURPLE, 12)
    box(s, 600, 45, 135, 50, "~14-bit acc.", "partial sum, no scales", RED, 12)
    box(s, 575, 108, 165, 44, "CUDA cores: FP32", "acc += partial * s_a * s_b", GREEN, 12)
    s.line(390, 45, 443, 60, arrow=True)
    s.line(390, 125, 443, 80, arrow=True)
    s.line(585, 70, 598, 70, arrow=True)
    s.line(668, 95, 668, 106, arrow=True)
    s.text(445, 126, "every 128", 11, fill=GREEN)
    s.text(445, 141, "elements of K", 11, fill=GREEN)
    s.text(10, 190, "Kept in BF16/FP32: embedding, output head, MoE gating, normalisation, "
           "attention; FP32 master weights and", 11)
    s.text(10, 207, "weight gradients; BF16 AdamW moments. FP8 for Fprop, Dgrad and Wgrad GEMMs; "
           "activations cached and dispatched in FP8", 11)
    s.text(10, 224, "(attention-output Linear inputs in a custom E5M6, power-of-two scales for "
           "the cached activations of certain layers).", 11)
    s.text(10, 252, "Reported: relative loss gap to BF16 below 0.25% (DeepSeek-AI, 2024). "
           "Our 14-bit accumulator model: max relative accumulation", 11, fill=GREY)
    s.text(10, 269, "error 1.1% at K = 4096 without promotion, 0.05% with promotion every 128 "
           "(the paper reports nearly 2% without).", 11, fill=GREY)
    s.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=os.path.join(HERE, "results", "bench_results.json"))
    ap.add_argument("--out-dir", default=BOOK_FIGS)
    args = ap.parse_args()
    res = None
    if os.path.exists(args.json):
        with open(args.json) as f:
            res = json.load(f)
    out = lambda k: os.path.join(args.out_dir, f"{STEM}-{k}.svg")      # noqa: E731
    roadmap(out(1))
    bit_layouts(out(2))
    number_line(out(3))
    rel_spacing(out(4))
    mx_layout(out(5))
    grids(out(6))
    clipping(out(7))
    sr_updates(out(8), res)
    if res:
        accumulation(out(9), res)
        granularity(out(10), res)
    else:
        print("no bench JSON: skipping the measured figures 9 and 10")
    online_softmax(out(11))
    deepseek(out(12))
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
