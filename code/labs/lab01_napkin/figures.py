"""Lab 1: draw the book map of the landscape chapter as a hand-written SVG.

    python labs/lab01_napkin/figures.py                 # writes book/figures/ch01_landscape-12.svg
    python labs/lab01_napkin/figures.py --out /tmp/map.svg

One band per part, chapters as boxes coloured by theme, capstone projects as dashed boxes at the
end of their part. Boxes carry names, never numbers: numbering belongs to the site build.
"""
from __future__ import annotations

import argparse
import os
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "book", "figures",
                   "ch01_landscape-12.svg")

# theme -> (stroke, pale fill); house palette, pale tints so the dark-theme conversion stays legible
THEME = {"model": ("#1f4e79", "#e3ecf5"), "systems": ("#8a5a00", "#f6ecd9"),
         "data": ("#2e6b30", "#e2efe2"), "reasoning": ("#6a4c93", "#ece6f4"),
         "appendix": ("#555555", "#eeeeee"), "project": ("#9b2c2c", "#ffffff")}
THEME_NAME = {"model": "models, training", "systems": "systems, serving",
              "data": "data, evaluation", "reasoning": "reasoning, post-training",
              "appendix": "appendix", "project": "capstone project"}

M, S, D, R, P, A = "model", "systems", "data", "reasoning", "project", "appendix"
PARTS = [
    ("I", "Foundations", [
        ("Landscape", M), ("PyTorch|essentials", M), ("The machine|underneath", S),
        ("Numbers on|the machine", S), ("Scaling and|quantisation", S),
        ("P10 Know|your machine", P), ("P11 Low-|precision training", P)]),
    ("II", "Building an LLM", [
        ("Text and|tokens", D), ("Attention", M), ("GPT-2", M), ("Qwen3 and|MoE intro", M),
        ("Design|space", M), ("Beyond|softmax", M), ("Encoders", M), ("MoE end|to end", M),
        ("Anatomy of|open models", M), ("P12 Bring up|a new model", P),
        ("P13 Sequence|mixers", P)]),
    ("III", "Pretraining at scale", [
        ("Optimizers", M), ("Pretraining", M), ("Pretraining|data", D), ("Scaling|laws", D),
        ("One-GPU|performance", S), ("Distributed|training", S), ("FSDP", S),
        ("Sharded|computation", S), ("P1 LM from|scratch", P), ("P2 Training|systems", P),
        ("P3 Scaling|study", P), ("P4 Web|data", P)]),
    ("IV", "Python to the metal", [
        ("C++ for|PyTorch", S), ("GPU|architecture", S), ("CUDA|kernels", S),
        ("Attention and|MoE kernels", S), ("Profiling", S), ("Compilers", S),
        ("Memory and|interconnects", S), ("P8 MoE|systems", P), ("P14 Tensor|compiler", P),
        ("P15 Kernel|library", P)]),
    ("V", "Fine-tuning", [
        ("Classification", M), ("Instruction|tuning, LoRA", M), ("P16 PEFT|study", P),
        ("P17 Domain|specialist", P)]),
    ("VI", "Inference and serving", [
        ("Inference", S), ("Serving|engines", S), ("KV-cache|management", S),
        ("Cheaper|inference", S), ("On-device", S), ("Retrieval", D),
        ("Data-movement|study", S), ("P6 Serving|engine", P), ("P7 CPU|runtime", P),
        ("P9 Profiler", P), ("P18 Long-|context serving", P)]),
    ("VII", "Reasoning and post-training", [
        ("Evaluation", D), ("Inference-|time scaling", R), ("Self-|refinement", R),
        ("The RL|setup", R), ("Policy|gradients", R), ("RLHF and|PPO", R), ("DPO", R),
        ("GRPO", R), ("Improving|GRPO", R), ("RL|systems", S), ("Distillation", R),
        ("Alignment,|multimodal", R), ("P5 Post-|training", P), ("P19 Agentic|RL", P),
        ("P20 Reasoning|distillation", P)]),
    ("", "Appendices", [
        ("A Triton", A), ("B Accelerators", A), ("C JAX|and XLA", A), ("D Lab|setup", A),
        ("E Containers", A), ("F References", A), ("G Doing|research", A),
        ("H Solutions", A)]),
]

W, LABEL_W, BOX_W, BOX_H, GAP, ROW_GAP, PER_ROW = 1000, 150, 99, 42, 6, 6, 8
FONT = 'font-family="Helvetica, Arial, sans-serif"'
INK = "#1a1a1a"


def box(x: float, y: float, text: str, theme: str) -> str:
    stroke, fill = THEME[theme]
    dash = ' stroke-dasharray="5,3"' if theme == P else ""
    width = 2.0 if theme == P else 1.2
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{BOX_W}" height="{BOX_H}" rx="5" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash}/>']
    lines = text.split("|")
    y0 = y + BOX_H / 2 - (len(lines) - 1) * 7.5 + 4.5
    for i, line in enumerate(lines):
        out.append(f'<text x="{x + BOX_W / 2:.1f}" y="{y0 + 15 * i:.1f}" {FONT} font-size="12" '
                   f'text-anchor="middle" fill="{INK}">{escape(line)}</text>')
    return "\n".join(out)


def draw() -> str:
    body, y = [], 58
    # legend
    lx = 10
    for theme in (M, S, D, R, P, A):
        stroke, fill = THEME[theme]
        dash = ' stroke-dasharray="5,3"' if theme == P else ""
        body.append(f'<rect x="{lx}" y="14" width="22" height="14" rx="3" fill="{fill}" '
                    f'stroke="{stroke}" stroke-width="1.5"{dash}/>')
        body.append(f'<text x="{lx + 28}" y="26" {FONT} font-size="12" fill="{INK}">'
                    f'{THEME_NAME[theme]}</text>')
        lx += 28 + 7.0 * len(THEME_NAME[theme]) + 22
    for roman, title, chapters in PARTS:
        rows = (len(chapters) + PER_ROW - 1) // PER_ROW
        band_h = rows * BOX_H + (rows - 1) * ROW_GAP + 12
        shade = "#f7f7f7" if roman else "#ffffff"
        body.append(f'<rect x="4" y="{y - 6}" width="{W - 8}" height="{band_h}" rx="6" '
                    f'fill="{shade}" stroke="#bbbbbb" stroke-width="0.8"/>')
        ty = y + band_h / 2 - 6
        if roman:
            body.append(f'<text x="14" y="{ty - 4:.1f}" {FONT} font-size="13" font-weight="bold" '
                        f'fill="{INK}">Part {roman}</text>')
            words = title.split()
            half = (len(words) + 1) // 2 if len(title) > 18 else len(words)
            for j, part in enumerate((" ".join(words[:half]), " ".join(words[half:]))):
                if part:
                    body.append(f'<text x="14" y="{ty + 12 + 14 * j:.1f}" {FONT} font-size="12" '
                                f'fill="{INK}">{escape(part)}</text>')
        else:
            body.append(f'<text x="14" y="{ty + 6:.1f}" {FONT} font-size="13" '
                        f'font-weight="bold" fill="{INK}">{title}</text>')
        for i, (text, theme) in enumerate(chapters):
            r, c = divmod(i, PER_ROW)
            body.append(box(LABEL_W + c * (BOX_W + GAP), y + r * (BOX_H + ROW_GAP), text, theme))
        y += band_h + 8
    h = y + 2
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" '
            f'viewBox="0 0 {W} {h}">\n<rect x="0" y="0" width="{W}" height="{h}" fill="#ffffff"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        f.write(draw())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
