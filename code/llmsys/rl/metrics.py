"""Logging and plotting helpers for RL runs."""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence


class CSVLogger:
    """Append one row of metrics per step; the header is fixed by the first row."""

    def __init__(self, path: str):
        self.path = path
        self.fields: List[str] = []
        if os.path.exists(path):
            os.remove(path)

    def log(self, row: Dict):
        if not self.fields:
            self.fields = list(row)
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)


def read_csv(path: str) -> Dict[str, List[float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out: Dict[str, List[float]] = {k: [] for k in rows[0]} if rows else {}
    for r in rows:
        for k, v in r.items():
            out[k].append(float(v) if v not in ("", None) else float("nan"))
    return out


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Trailing moving average (window shrinks at the start)."""
    out, s = [], 0.0
    for i, v in enumerate(values):
        s += v
        if i >= window:
            s -= values[i - window]
        out.append(s / min(i + 1, window))
    return out


def plot_metrics(csv_path: str, columns: Sequence[str], save_as: str = None, window_frac=0.1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_csv(csv_path)
    n = len(columns)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), sharex=True, squeeze=False)
    for ax, name in zip(axes.flat, columns):
        y = data[name]
        x = data["step"]
        w = max(1, int(window_frac * len(y)))
        ax.plot(x, y, alpha=0.35)
        ax.plot(x, moving_average(y, w))
        ax.set_ylabel(name)
    for ax in axes[-1]:
        ax.set_xlabel("step")
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as, dpi=150)
    return fig
