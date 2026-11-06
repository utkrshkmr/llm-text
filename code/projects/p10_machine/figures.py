"""Project 10: draw the chapter's figures as SVG (house palette, light theme) (Chapter 6).

    python figures.py                                          # from results/smoke_a100_pcie.json
    python figures.py --json results/full.json --out-dir /tmp/figs

Diagrams are hand-made SVG; plots use matplotlib and read the suite's results JSON, so every
measured figure is regenerated from data (measured on 2x A100-40GB (PCIe)).
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK_FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "book", "figures")
STEM = "ch06_proj_machine"
BLUE, AMBER, PURPLE, GREEN, RED, GREY = (
    "#1f4e79",
    "#8a5a00",
    "#4b2c83",
    "#2e6b30",
    "#9b2c2c",
    "#555555",
)
TINT = {
    BLUE: "#dce6f1",
    AMBER: "#f3e9d2",
    PURPLE: "#e6e0f0",
    GREEN: "#deeedc",
    RED: "#f2dcdc",
    GREY: "#eeeeee",
}
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": GREY,
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "svg.hashsalt": "ch06",
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#bbbbbb",
    }
)
MEASURED = "measured on 2x A100-40GB (PCIe)"


# --------------------------------------------------------------------------- SVG helpers
class Svg:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.items = w, h, []

    def rect(self, x, y, w, h, fill="#ffffff", stroke="#000000", sw=1.0, rx=0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def text(
        self, x, y, s, size=12, anchor="start", weight="normal", fill="#000000", style="normal"
    ):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-style="{style}" '
            f'fill="{fill}" font-family="{FONT}">{s}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke="#000000", sw=1.0, arrow=False, dash=None, both=False):
        extra = ' marker-end="url(#arr)"' if arrow else ""
        if both:
            extra += ' marker-start="url(#arr)"'
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        self.items.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def save(self, path):
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">\n<defs><marker id="arr" viewBox="0 0 10 10" '
            'refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#000000"/></marker></defs>\n'
            f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#ffffff"/>\n'
        )
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


def _gbs(v):
    return f"{v / 1e9:,.1f} GB/s"


# --------------------------------------------------------------------------- 1 roadmap
def roadmap(path):
    s = Svg(780, 260)
    top = [
        ("M1 Bandwidth", "HBM, L2, PCIe, P2P", "size sweeps, alpha-beta fits", AMBER),
        ("M2 Compute", "matmul peak per dtype", "FMA chain across the ridge", BLUE),
        ("M3 Overheads", "launch, sync, small copies", "CUDA graph replay", GREY),
    ]
    w, h, gap = 230, 70, 25
    for i, (t, a, b, c) in enumerate(top):
        box(s, 10 + i * (w + gap), 15, w, h, t, a, c, sub2=b)
    box(
        s,
        140,
        115,
        500,
        50,
        "The machine model: measured ceilings + fixed costs",
        "t = t0 + (n-1) gap + max(F / pi_dtype, Q / beta)",
        PURPLE,
    )
    for i in range(3):
        s.line(10 + i * (w + gap) + w / 2, 85, 390 + (i - 1) * 150, 115, arrow=True)
    box(
        s,
        10,
        195,
        370,
        55,
        "M4 Validate on 20 kernels",
        "elementwise, reductions, GEMMs, attention",
        GREEN,
    )
    box(
        s,
        400,
        195,
        370,
        55,
        "M5 Spec sheet and error analysis",
        "datasheet gaps explained, a paper-style write-up",
        RED,
    )
    s.line(300, 165, 195, 195, arrow=True)
    s.line(380, 222, 398, 222, arrow=True)
    s.save(path)


# --------------------------------------------------------------------------- 2 the box
def machine(path, m):
    s = Svg(800, 470)
    for i, x0 in enumerate((15, 445)):
        s.rect(x0, 15, 340, 440, fill="#ffffff", stroke=GREY, sw=1.2, rx=8, dash="5 4")
        s.text(x0 + 10, 33, f"NUMA node {i}", 12, weight="bold", fill=GREY)
        box(s, x0 + 15, 45, 145, 55, f"CPU socket {i}", "Xeon Gold 6330", GREY)
        box(s, x0 + 180, 45, 145, 55, "DRAM", "pageable + pinned", AMBER)
        s.line(x0 + 160, 72, x0 + 180, 72, both=True, arrow=True)
        box(s, x0 + 15, 140, 310, 40, "PCIe 4.0 root complex + IOMMU", None, PURPLE, size=11)
        s.line(x0 + 87, 100, x0 + 87, 140, both=True, arrow=True)
        s.rect(x0 + 15, 250, 310, 190, fill="#ffffff", stroke=BLUE, sw=1.6, rx=8)
        s.text(x0 + 170, 270, f"GPU{i}: A100-PCIE-40GB, 250 W", 12, "middle", "bold", fill=BLUE)
        box(s, x0 + 30, 285, 120, 50, "108 SMs", "tensor cores", BLUE, size=11)
        box(s, x0 + 190, 285, 120, 50, "L2, 40 MiB", "2 partitions", BLUE, size=11)
        box(s, x0 + 30, 375, 280, 50, "HBM2, 40 GB", "datasheet 1,555 GB/s", AMBER, size=11)
        s.line(x0 + 60, 180, x0 + 60, 250, both=True, arrow=True, stroke=PURPLE, sw=1.6)
        s.line(x0 + 150, 310, x0 + 190, 310, both=True, arrow=True)
        s.line(x0 + 250, 335, x0 + 250, 375, both=True, arrow=True)
    # measured numbers, GPU0 side
    s.text(85, 205, f"H2D pinned {_gbs(m['h2d_pinned_bw'])}", 11, fill=PURPLE)
    s.text(85, 220, f"H2D pageable {_gbs(m['h2d_pageable_bw'])}", 11, fill=PURPLE)
    s.text(85, 235, f"D2H pinned {_gbs(m['d2h_pinned_bw'])}", 11, fill=PURPLE)
    s.text(185, 350, f"{m['l2_bw'] / 1e12:.1f} TB/s", 10, "middle", fill=BLUE)
    s.text(185, 362, "L2 copy, warm", 10, "middle", fill=BLUE)
    s.text(272, 355, f"{m['hbm_copy_bw'] / 1e9:,.0f} GB/s", 10, fill=AMBER)
    s.text(272, 367, "HBM copy", 10, fill=AMBER)
    # the inter-socket paths
    s.line(340, 72, 460, 72, both=True, arrow=True, stroke=GREY, sw=1.6)
    s.text(400, 64, "UPI", 11, "middle", weight="bold", fill=GREY)
    if "staged_bw" in m:
        s.text(400, 92, "staged via host", 10, "middle", fill=GREEN)
        s.text(400, 105, _gbs(m["staged_bw"]), 10, "middle", "bold", fill=GREEN)
    s.line(340, 160, 460, 160, both=True, arrow=True, stroke=RED, sw=1.4, dash="6 3")
    s.text(400, 198, "P2P", 11, "middle", "bold", fill=RED)
    s.text(400, 212, _gbs(m["p2p_bw"]), 10, "middle", fill=RED)
    if m.get("p2p_correct", 1.0) != 1.0:
        s.text(400, 228, "data", 10, "middle", "bold", fill=RED)
        s.text(400, 241, "wrong!", 10, "middle", "bold", fill=RED)
    s.save(path)


# --------------------------------------------------------------------------- 3 timers
def timers(path, m):
    s = Svg(840, 330)
    rows = [
        ("(a) events, idle queue", 20),
        ("(b) events, queue kept ahead", 125),
        ("(c) graph replay of n kernels", 230),
    ]
    for title, y in rows:
        s.text(20, y + 12, title, 12, weight="bold")
        s.text(20, y + 38, "host", 11, fill=GREY)
        s.text(20, y + 73, "GPU", 11, fill=GREY)
        s.line(70, y + 34, 780, y + 34, stroke="#bbbbbb")
        s.line(70, y + 69, 780, y + 69, stroke="#bbbbbb")
    # (a)
    y = 20
    box(s, 80, y + 22, 60, 22, "rec s", None, GREY, size=10)
    box(s, 150, y + 22, 170, 22, "launch kernel (Python, driver)", None, GREY, size=10)
    box(s, 330, y + 22, 60, 22, "rec e", None, GREY, size=10)
    s.rect(90, y + 58, 6, 22, fill=RED, stroke=RED)
    s.rect(330, y + 58, 40, 22, fill=TINT[BLUE], stroke=BLUE)
    s.rect(395, y + 58, 6, 22, fill=RED, stroke=RED)
    s.line(93, y + 92, 398, y + 92, both=True, arrow=True, stroke=RED)
    s.text(
        430,
        y + 96,
        f"elapsed {m['launch_host_s'] * 1e6:.1f} us: mostly the host's launch",
        11,
        fill=RED,
    )
    # (b)
    y = 125
    box(s, 80, y + 22, 70, 22, "spin", None, GREY, size=10)
    box(s, 155, y + 22, 50, 22, "rec s", None, GREY, size=10)
    box(s, 210, y + 22, 90, 22, "launch", None, GREY, size=10)
    box(s, 305, y + 22, 50, 22, "rec e", None, GREY, size=10)
    s.rect(110, y + 58, 300, 22, fill=TINT[GREY], stroke=GREY)
    s.text(260, y + 73, "spin kernel (~100 us) hides the host", 10, "middle")
    s.rect(412, y + 58, 6, 22, fill=RED, stroke=RED)
    s.rect(420, y + 58, 30, 22, fill=TINT[BLUE], stroke=BLUE)
    s.rect(452, y + 58, 6, 22, fill=RED, stroke=RED)
    s.line(415, y + 92, 455, y + 92, both=True, arrow=True, stroke=RED)
    s.text(
        470,
        y + 96,
        f"elapsed {m['launch_s'] * 1e6:.1f} us: the device's own cost "
        f"(timer floor {m['empty_event_s'] * 1e6:.1f} us)",
        11,
        fill=RED,
    )
    # (c)
    y = 230
    box(s, 80, y + 22, 120, 22, "replay(graph)", None, GREY, size=10)
    x = 210
    s.rect(x - 8, y + 58, 6, 22, fill=RED, stroke=RED)
    for i in range(14):
        s.rect(x + i * 30, y + 58, 22, 22, fill=TINT[BLUE], stroke=BLUE)
    s.rect(x + 14 * 30, y + 58, 6, 22, fill=RED, stroke=RED)
    s.text(x + 14 * 30 + 20, y + 73, f"{m['graph_kernel_s'] * 1e6:.2f} us per kernel", 11, fill=RED)
    s.text(
        x + 14 * 30 + 20,
        y + 88,
        f"(eager: {m['eager_kernel_s'] * 1e6:.1f} us, host-bound)",
        11,
        fill=GREY,
    )
    s.save(path)


# --------------------------------------------------------------------------- plots
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def bandwidth_curve(path, d):
    a, m = d["artifacts"]["m1"], d["metrics"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    styles = {
        "copy": (AMBER, "o"),
        "read": (BLUE, "s"),
        "reduce": (GREY, "v"),
        "write": (GREEN, "^"),
        "triad": (PURPLE, "D"),
    }
    for kind, rows in a["stream"].items():
        c, mk = styles[kind]
        ax.plot(
            [r["working_set"] / 2**20 for r in rows],
            [r["bw"] / 1e9 for r in rows],
            marker=mk,
            color=c,
            ms=4,
            lw=1.2,
            label=f"{kind}, cold (L2 flushed)",
        )
    warm = a["warm"]["copy"]
    ax.plot(
        [r["working_set"] / 2**20 for r in warm],
        [r["bw"] / 1e9 for r in warm],
        color=RED,
        lw=1.6,
        marker=".",
        label="copy, warm (graph burst)",
    )
    ax.axhline(1555, color=GREY, ls="--", lw=1)
    ax.text(2**11.5, 1600, "datasheet HBM 1,555 GB/s", color=GREY, fontsize=9, ha="right")
    ax.axvline(40, color=BLUE, ls=":", lw=1)
    ax.text(41, 300, "L2 = 40 MiB", color=BLUE, fontsize=9)
    if "l2_knee_bytes" in m:
        k = m["l2_knee_bytes"] / 2**20
        ax.axvline(k, color=RED, ls=":", lw=1)
        ax.text(k * 0.55, 3250, f"knee {k:.0f} MiB", color=RED, fontsize=9, ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("working set (MiB)")
    ax.set_ylabel("bandwidth (GB/s)")
    ax.set_ylim(0, 3500)
    ax.set_title(f"Streaming bandwidth vs working set ({MEASURED})", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, path)


def transfers(path, d):
    a, m = d["artifacts"]["m1"], d["metrics"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    styles = {
        "h2d_pinned": (BLUE, "-", "o"),
        "d2h_pinned": (GREEN, "-", "s"),
        "h2d_pageable": (BLUE, "--", "o"),
        "d2h_pageable": (GREEN, "--", "s"),
    }
    for tag, rows in a["host"].items():
        c, ls, mk = styles[tag]
        ax.plot(
            [r["nbytes"] for r in rows],
            [r["bw"] / 1e9 for r in rows],
            color=c,
            ls=ls,
            marker=mk,
            ms=3.5,
            lw=1.2,
            label=tag.replace("_", " ").replace("h2d", "H2D").replace("d2h", "D2H"),
        )
    p2p = a["p2p"].get("uni", [])
    if p2p:
        bad = "" if m.get("p2p_correct", 1.0) == 1.0 else " (data wrong!)"
        ax.plot(
            [r["nbytes"] for r in p2p],
            [r["bw"] / 1e9 for r in p2p],
            color=RED,
            ls=":",
            marker="x",
            ms=4,
            lw=1.2,
            label="GPU0 to GPU1, P2P" + bad,
        )
        n = np.logspace(12, 28, 60, base=2)
        al, be = m["p2p_alpha"], m["p2p_beta"]
        ax.plot(n, n / (al + n / be) / 1e9, color=RED, lw=0.8, alpha=0.6)
    for r in a["p2p"].get("staged", []):
        ax.plot(
            [r["nbytes"]],
            [r["bw"] / 1e9],
            marker="*",
            ms=12,
            color=PURPLE,
            ls="none",
            label=f"GPU0 to GPU1 via pinned host ({r['chunk_bytes'] // 2**20} MiB chunks)",
        )
    n = np.logspace(12, 28, 60, base=2)
    al, be = m["h2d_pinned_alpha"], m["h2d_pinned_beta"]
    ax.plot(
        n,
        n / (al + n / be) / 1e9,
        color=BLUE,
        lw=0.8,
        alpha=0.6,
        label=f"fit t = {al * 1e6:.1f} us + n / {be / 1e9:.1f} GB/s",
    )
    ax.axhline(31.5, color=GREY, ls="--", lw=1)
    ax.text(
        2**28, 32.3, "PCIe 4.0 x16 raw: 31.5 GB/s per direction", color=GREY, fontsize=9, ha="right"
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("transfer size (bytes)")
    ax.set_ylabel("bandwidth (GB/s)")
    ax.set_ylim(0, 46)
    ax.set_title(f"Host and peer transfers ({MEASURED})", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    _save(fig, path)


def matmul_plot(path, d):
    a = d["artifacts"]["m2"]["matmul"]
    ds = {"fp32": 19.5, "tf32": 156, "fp16": 312, "bf16": 312, "int8": 624}
    cols = {"fp32": GREY, "tf32": GREEN, "fp16": PURPLE, "bf16": BLUE, "int8": AMBER}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for dt, rows in a.items():
        ax.plot(
            [r["n"] for r in rows],
            [r["flops_per_s"] / 1e12 for r in rows],
            marker="o",
            ms=4,
            color=cols[dt],
            label=f"{dt} (datasheet {ds[dt]})",
        )
        ax.axhline(ds[dt], color=cols[dt], ls="--", lw=0.8)
        last = rows[-1]
        if last.get("clock_mhz") and last["clock_mhz"] < 1400 and dt in ("bf16", "int8", "tf32"):
            ax.annotate(
                f"{last['clock_mhz']:.0f} MHz",
                (last["n"], last["flops_per_s"] / 1e12),
                textcoords="offset points",
                xytext=(7, -3),
                fontsize=8,
                color=cols[dt],
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([r["n"] for r in next(iter(a.values()))])
    ax.set_xticklabels([str(r["n"]) for r in next(iter(a.values()))])
    ax.set_xlim(850, 14000)
    ax.set_ylim(10, 2500)
    ax.set_xlabel("square GEMM size n (calibration sizes, disjoint from the zoo)")
    ax.set_ylabel("TFLOP/s (int8: TOP/s)")
    ax.set_title(f"Matmul throughput per dtype; dashed = datasheet ({MEASURED})", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    _save(fig, path)


def roofline_plot(path, d):
    m, art = d["metrics"], d["artifacts"]
    bw = m["hbm_copy_bw"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.logspace(-2, 4, 300)
    for key, label, c in (
        ("matmul_bf16", "bf16 tensor cores", BLUE),
        ("matmul_tf32", "tf32", GREEN),
        ("simt_fp32", "fp32 CUDA cores", GREY),
    ):
        ax.plot(
            x,
            np.minimum(m[key], x * bw) / 1e12,
            color=c,
            lw=1.4,
            label=f"{label}: {m[key] / 1e12:.0f} TFLOP/s",
        )
    fma = art["m2"]["fma"]
    ax.plot(
        [r["intensity"] for r in fma],
        [r["flops_per_s"] / 1e12 for r in fma],
        "s",
        color=RED,
        ms=6,
        label="FMA chain (synthetic)",
    )
    marks = {"elementwise": "o", "reduction": "D", "gemm": "^", "attention": "*"}
    seen = set()
    for k in art["m4"]["kernels"]:
        if k["flops"] <= 0:
            continue
        lab = None if k["category"] in seen else f"zoo: {k['category']}"
        seen.add(k["category"])
        ax.plot(
            k["intensity"],
            k["flops"] / k["measured_s"] / 1e12,
            marks[k["category"]],
            color=AMBER,
            ms=6 if k["category"] != "attention" else 9,
            mfc="none",
            mew=1.4,
            label=lab,
        )
    ax.axvline(m["matmul_bf16"] / bw, color=BLUE, ls=":", lw=0.8)
    ax.text(
        m["matmul_bf16"] / bw * 1.1,
        0.02,
        f"bf16 ridge {m['matmul_bf16'] / bw:.0f}",
        color=BLUE,
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 600)
    ax.set_xlabel("arithmetic intensity (FLOP/byte of HBM traffic)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"The measured roofline with the 20-kernel zoo ({MEASURED})", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    _save(fig, path)


def latency_ladder(path, d):
    m = d["metrics"]
    items = [
        ("kernel inside a CUDA graph", m["graph_kernel_s"]),
        ("empty CUDA event pair (timer floor)", m["empty_event_s"]),
        ("tiny kernel, device side", m["launch_s"]),
        ("host enqueue of one eager op", m["enqueue_s"]),
        (".item() round trip", m["item_s"]),
        ("torch.cuda.synchronize(), idle", m["sync_s"]),
        ("tiny kernel timed with an idle queue", m["launch_host_s"]),
        ("4-byte H2D copy + sync, pinned", m["copy4_pinned_s"]),
        ("4-byte H2D copy + sync, pageable", m["copy4_pageable_s"]),
        ("P2P alpha (fit, wall clock incl. 2 syncs)", m.get("p2p_alpha", float("nan"))),
    ]
    items = [it for it in items if not math.isnan(it[1])]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ys = np.arange(len(items))
    cols = [
        BLUE
        if "graph" in n or "device" in n or "timer" in n
        else (PURPLE if "copy" in n or "P2P" in n else GREY)
        for n, _ in items
    ]
    ax.barh(ys, [v * 1e6 for _, v in items], color=cols)
    for y, (_, v) in zip(ys, items):
        ax.text(v * 1e6 * 1.05, y, f"{v * 1e6:.1f} us", va="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _ in items], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 200)
    ax.invert_yaxis()
    ax.set_xlabel("microseconds (median)")
    ax.set_title(f"Fixed costs ({MEASURED})", fontsize=10)
    _save(fig, path)


def prediction_plot(path, d):
    art = d["artifacts"]["m4"]
    meas = np.array([k["measured_s"] for k in art["kernels"]]) * 1e6
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    lim = (2, 2e4)
    xs = np.array(lim)
    ax.fill_between(xs, xs / 1.2, xs * 1.2, color="#eeeeee", label="within 20%")
    ax.plot(xs, xs, color=GREY, lw=0.8)
    styles = {
        "datasheet": (GREY, "x"),
        "roofline": (AMBER, "o"),
        "roofline+launch": (BLUE, "s"),
        "fitted_loo": (GREEN, "^"),
    }
    errs = art["errors"]
    for name, (c, mk) in styles.items():
        if name not in art["predictions"]:
            continue
        p = np.array(art["predictions"][name]) * 1e6
        p = np.maximum(p, lim[0] * 0.6)
        ax.plot(
            meas,
            p,
            mk,
            color=c,
            ms=5,
            mfc="none" if mk != "x" else c,
            mew=1.2,
            label=f"{name}: {100 * errs[name]['within']:.0f}% within band, "
            f"MAPE {100 * errs[name]['mape']:.0f}%",
        )
    offsets = {
        "softmax_bf16": (-78, -4),
        "sdpa_flash_bf16": (8, -2),
        "gemm_bf16_1024": (6, -14),
        "attn_naive_bf16": (8, -24),
    }
    for k, t in zip(art["kernels"], meas):
        if k["name"] in offsets:
            ax.annotate(
                k["name"],
                (t, t),
                textcoords="offset points",
                xytext=offsets[k["name"]],
                fontsize=8,
                color=RED,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(lim[0] * 0.5, lim[1])
    ax.set_xlabel("measured time (us)")
    ax.set_ylabel("predicted time (us)")
    ax.set_title(f"Predicted vs measured, 20 kernels ({MEASURED})", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    _save(fig, path)


def sustained_plot(path, d):
    s = d["artifacts"]["m2"].get("sustained")
    if not s:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(
        [r["t"] for r in s["rate"]],
        [r["flops_per_s"] / 1e12 for r in s["rate"]],
        color=BLUE,
        lw=1.4,
        label="bf16 GEMM 8192^3, TFLOP/s (left)",
    )
    scaled = [
        (r["t"], r["flops_per_s"] * 1410 / r["clock_mhz"] / 1e12)
        for r in s["rate"]
        if r.get("clock_mhz")
    ]
    ax.plot(
        [t for t, _ in scaled],
        [v for _, v in scaled],
        color=GREEN,
        lw=1.0,
        label="same x 1,410 / SM clock (left)",
    )
    ax.axhline(312, color=BLUE, ls="--", lw=0.8)
    ax.text(2.0, 316, "datasheet 312 TFLOP/s at 1,410 MHz", color=BLUE, fontsize=8.5, va="bottom")
    ax.set_ylim(0, 370)
    ax.set_xlabel("seconds of back-to-back GEMMs")
    ax.set_ylabel("TFLOP/s")
    ax2 = ax.twinx()
    clocks = [r for r in s["clocks"] if -0.2 <= r["t"] <= s["rate"][-1]["t"]]
    ax2.plot(
        [r["t"] for r in clocks],
        [r["sm_mhz"] for r in clocks],
        color=RED,
        lw=1.2,
        label="SM clock, MHz (right)",
    )
    ax2.plot(
        [r["t"] for r in clocks],
        [4 * r["power_w"] for r in clocks],
        color=AMBER,
        lw=1.0,
        ls=":",
        label="board power x4, W (right)",
    )
    ax2.set_ylim(0, 1600 * 370 / 340)
    ax2.set_ylabel("MHz  (power: W x 4)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax.set_title(f"Burst vs sustained: the 250 W limit sets the clock ({MEASURED})", fontsize=10)
    _save(fig, path)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", default=os.path.join(HERE, "results", "smoke_a100_pcie.json"))
    ap.add_argument(
        "--peer-json",
        default=os.path.join(HERE, "results", "smoke_a100_pcie_2gpu.json"),
        help="a two-GPU run to take the peer-path rows from (if --json has none)",
    )
    ap.add_argument("--out-dir", default=BOOK_FIGS)
    a = ap.parse_args(argv)
    with open(a.json) as f:
        d = json.load(f)
    if not d["artifacts"]["m1"].get("p2p") and os.path.exists(a.peer_json):
        with open(a.peer_json) as f:
            peer = json.load(f)
        d["artifacts"]["m1"]["p2p"] = peer["artifacts"]["m1"]["p2p"]
        for k, v in peer["metrics"].items():
            if k.startswith(("p2p_", "staged_")):
                d["metrics"].setdefault(k, v)
    os.makedirs(a.out_dir, exist_ok=True)
    out = lambda k: os.path.join(a.out_dir, f"{STEM}-{k}.svg")  # noqa: E731
    roadmap(out(1))
    machine(out(2), d["metrics"])
    timers(out(3), d["metrics"])
    bandwidth_curve(out(4), d)
    transfers(out(5), d)
    matmul_plot(out(6), d)
    roofline_plot(out(7), d)
    latency_ladder(out(8), d)
    prediction_plot(out(9), d)
    n = 10 if sustained_plot(out(10), d) else 9
    print("wrote", ", ".join(f"{STEM}-{k}.svg" for k in range(1, n + 1)), "to", a.out_dir)


if __name__ == "__main__":
    main()
