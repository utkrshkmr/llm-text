"""Lab 5: draw the quantisation chapter's figures as SVG (house palette, light theme).

    python labs/lab05_quant/figures.py                       # into book/figures
    python labs/lab05_quant/figures.py --json results/bench_results.json --out-dir /tmp/f

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
STEM = "ch05_quant"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": "#000000",
                     "xtick.color": "#000000", "ytick.color": "#000000", "svg.hashsalt": "ch05",
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
    s = Svg(760, 330)
    box(s, 10, 20, 170, 56, "Formats, rounding,", "accumulators (last chapter)", GREY)
    row1 = [("1 Block scaling", "MX, NVFP4, metadata", BLUE),
            ("2 Hadamard rotations", "spreading outliers", BLUE),
            ("3 Backward-pass scales", "two copies, 2-D, pow2", BLUE)]
    w, h, gap = 170, 56, 20
    for i, (t, sub, c) in enumerate(row1):
        x = 10 + (i + 1) * (w + gap)
        box(s, x, 20, w, h, t, sub, c)
        s.line(x - gap + 2, 48, x - 2, 48, arrow=True, stroke=GREY if i == 0 else "#000000")
    row2 = [("4 Quantisation maths", "affine, 6 dB, clipping, grids", GREEN),
            ("5 Hessian-weighted error", "which errors matter; OBS, GPTQ", GREEN)]
    w2 = 360
    for i, (t, sub, c) in enumerate(row2):
        box(s, 10 + i * (w2 + 20), 110, w2, h, t, sub, c)
        if i:
            s.line(10 + w2 + 2, 138, 10 + w2 + 18, 138, arrow=True)
    row3 = [("6 Mixed precision", "master weights, loss scaling, FP8 scaling, inference", PURPLE),
            ("7 DeepSeek-V3 FP8", "tiles, blocks, promotion, FP8 storage", PURPLE)]
    for i, (t, sub, c) in enumerate(row3):
        box(s, 10 + i * (w2 + 20), 200, w2, h, t, sub, c)
        if i:
            s.line(10 + w2 + 2, 228, 10 + w2 + 18, 228, arrow=True)
    s.line(10 + 3 * (w + gap) + w / 2, 76, 10 + w2 / 2, 108, arrow=True, stroke=GREY, dash="4 3")
    s.line(10 + w2 + 20 + w2 / 2, 166, 10 + w2 / 2, 198, arrow=True, stroke=GREY, dash="4 3")
    box(s, 200, 276, 360, 46, "Lab: scales, quantisers, block-scaled FP8 GEMM",
        "emulated on the A100; feeds the low-precision capstone", RED)
    s.line(10 + w2 + 20 + w2 / 2, 256, 380, 274, arrow=True, stroke=GREY, dash="4 3")
    s.save(path)


# --------------------------------------------------------------------------- 2 MX / NVFP4
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


# --------------------------------------------------------------------------- 3 granularity
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


# --------------------------------------------------------------------------- 4 Hadamard
def hadamard_plot(path, res):
    rows = res["hadamard"]
    lab = ["none" if r["hadamard_block"] == 0 else str(r["hadamard_block"]) for r in rows]
    x = np.arange(len(rows))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})
    for key, col, name, m in [("mxfp4", RED, "MXFP4 (E8M0 per 32)", "o"),
                              ("nvfp4", BLUE, "NVFP4 (E4M3 per 16)", "s"),
                              ("int4_g32", AMBER, "INT4, groups of 32", "^")]:
        a.plot(x, [100 * r[key] for r in rows], m + "-", color=col, lw=1.4, ms=5, label=name)
    a.set_xticks(x, lab)
    a.set_xlabel("Hadamard transform size along K")
    a.set_ylabel("relative error of X W^T (%)")
    a.legend(fontsize=8.5, frameon=False)
    b.bar(x, [r["act_amax_over_rms"] for r in rows], color=TINT[GREEN], edgecolor=GREEN)
    for i, r in enumerate(rows):
        b.text(i, r["act_amax_over_rms"] + 1, f"{r['act_amax_over_rms']:.1f}", ha="center",
               fontsize=8.5)
    b.set_xticks(x, lab)
    b.set_xlabel("Hadamard transform size")
    b.set_ylabel("max |x| / RMS of activations")
    fig.suptitle("Emulated on 2x A100-40GB (PCIe): activations with 8 outlier channels (x30)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


# --------------------------------------------------------------------------- 5 clipping MSE
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


# --------------------------------------------------------------------------- 7 FP8 scaling
def fp8_scaling(path, res):
    sc = res["scaling"]
    rows = sc["rows"]
    t = [r["step"] for r in rows]
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True,
                               gridspec_kw={"height_ratios": [1.3, 1]})
    for key, col, name, ls in [("delayed_err", RED, "delayed (amax history)", "-"),
                               ("current_err", BLUE, "current (per tensor)", "--"),
                               ("block_err", GREEN, "block (1x128)", ":")]:
        a.semilogy(t, [100 * r[key] for r in rows], ls, color=col, lw=1.5, label=name)
    a.axvline(sc["spike_at"], color=GREY, lw=0.8, ls=":")
    a.text(sc["spike_at"] + 0.5, 50, "one token x100", fontsize=8.5, color=GREY)
    a.set_ylabel("relative error (%)")
    a.legend(fontsize=8.5, frameon=False, loc="center left")
    for key, col, name in [("delayed_flushed", RED, "delayed"),
                           ("current_flushed", BLUE, "current")]:
        b.plot(t, [r[key] for r in rows], "o-", color=col, lw=1.2, ms=2.5, label=name)
    b.axvspan(sc["spike_at"] + 0.5, sc["spike_at"] + sc["history"] + 0.5, color="#eeeeee")
    b.set_ylabel("values flushed to 0")
    b.set_xlabel("step")
    b.legend(fontsize=8.5, frameon=False, loc="upper left")
    a.set_title(f"Emulated on 2x A100-40GB (PCIe): E4M3, {sc['history']}-step amax history",
                fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


# --------------------------------------------------------------------------- 8 DeepSeek FP8
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
    mx_layout(out(2))
    clipping(out(5))
    grids(out(6))
    deepseek(out(8))
    if res:
        granularity(out(3), res)
        hadamard_plot(out(4), res)
        fp8_scaling(out(7), res)
    else:
        print("no bench JSON: skipping the measured figures 3, 4 and 7")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
