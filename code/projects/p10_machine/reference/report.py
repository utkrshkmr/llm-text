"""Project 10 (reference solution): the spec sheet and the results files, milestone 5 (Chapter 4).

``results_document`` builds the repository-wide results schema
``{config, versions, metrics, timings: {median, p10, p90}, artifacts}``; ``spec_sheet_rows``
turns the measured ceilings into the machine's "spec sheet" next to the datasheet; ``write``
saves JSON with a Markdown rendering next to it.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # code/ for llmsys

from llmsys.perf import to_markdown  # noqa: E402

GB = 1e9
TF = 1e12

# Vendor numbers, dense (no sparsity). A100: NVIDIA A100 datasheet and architecture whitepaper
# (2020). H100 SXM: NVIDIA H100 datasheet. MI300X: AMD Instinct MI300X datasheet. PCIe x16 per
# direction: Gen4 16 GT/s x 16 x 128/130 / 8 = 31.5 GB/s; Gen5 twice that. ``llc_bytes`` is the
# largest on-chip cache, which the flush buffer must exceed twice over. ``simt`` is the non-matrix
# fp32 rate (CUDA cores / vector ALUs).
DATASHEETS: Dict[str, Dict[str, float]] = {
    "A100-PCIE-40GB": dict(
        fp32=19.5 * TF,
        tf32=156 * TF,
        bf16=312 * TF,
        fp16=312 * TF,
        int8=624 * TF,
        simt=19.5 * TF,
        hbm_bw=1555 * GB,
        l2_bytes=40 * 2**20,
        llc_bytes=40 * 2**20,
        pcie_bw=31.5 * GB,
        p2p_bw=31.5 * GB,
        sms=108,
        boost_clock_hz=1.41e9,
        tdp_w=250,
    ),
    "H100-SXM5-80GB": dict(
        fp32=66.9 * TF,
        tf32=494.7 * TF,
        bf16=989.4 * TF,
        fp16=989.4 * TF,
        int8=1978.9 * TF,
        simt=66.9 * TF,
        hbm_bw=3350 * GB,
        l2_bytes=50e6,
        llc_bytes=50e6,
        pcie_bw=63.0 * GB,
        p2p_bw=450 * GB,
        sms=132,
        tdp_w=700,
    ),
    "MI300X": dict(
        fp32=163.4 * TF,
        tf32=653.7 * TF,
        bf16=1307.4 * TF,
        fp16=1307.4 * TF,
        int8=2614.9 * TF,
        simt=163.4 * TF,
        hbm_bw=5300 * GB,
        llc_bytes=256 * 2**20,
        pcie_bw=63.0 * GB,
        tdp_w=750,
    ),
}

REQUIRED_KEYS = ("config", "versions", "metrics", "timings", "artifacts")


def results_document(
    config: Dict,
    versions: Dict,
    metrics: Dict,
    timings: Dict[str, Dict],
    artifacts: Optional[Dict] = None,
) -> Dict[str, object]:
    """Assemble a results file; every timing entry must carry median, p10 and p90."""
    doc = dict(
        config=config,
        versions=versions,
        metrics=metrics,
        timings=timings,
        artifacts=artifacts or {},
    )
    validate(doc)
    return doc


def validate(doc: Dict) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in doc]
    if missing:
        raise ValueError(f"results document lacks {missing}")
    for name, t in doc["timings"].items():
        for q in ("median", "p10", "p90"):
            if q not in t:
                raise ValueError(f"timing {name!r} lacks {q}")
        if not t["p10"] <= t["median"] <= t["p90"]:
            raise ValueError(f"timing {name!r} is not ordered p10 <= median <= p90")


def _row(
    quantity: str,
    measured: Optional[float],
    unit: str,
    scale: float,
    datasheet: Optional[float],
    method: str,
    spread: Optional[tuple] = None,
) -> Dict[str, object]:
    m = None if measured is None else measured / scale
    d = None if datasheet is None else datasheet / scale
    frac = None if (measured is None or not datasheet) else measured / datasheet
    sp = "-"
    if spread and None not in spread:
        lo, hi = sorted(v / scale for v in spread)
        sp = f"{lo:,.4g}-{hi:,.4g}"
    return {
        "quantity": quantity,
        "measured": "n/a" if m is None else f"{m:,.4g} {unit}",
        "p10-p90": sp,
        "datasheet": "-" if d is None else f"{d:,.4g} {unit}",
        "fraction": "-" if frac is None else f"{100 * frac:.0f}%",
        "how": method,
    }


def spec_sheet_rows(
    metrics: Dict[str, float], datasheet: Optional[Dict[str, float]] = None
) -> List[Dict[str, object]]:
    """The measured spec sheet, one row per ceiling with its p10-p90 spread where the metrics
    carry one (``<key>_p10``/``<key>_p90``). Missing metrics render as n/a; a path whose data
    check failed is marked DATA WRONG."""
    ds = datasheet or {}
    g = metrics.get
    sp = lambda k: (g(k + "_p10"), g(k + "_p90")) if g(k + "_p10") is not None else None  # noqa
    wrong = lambda k: ": DATA WRONG" if g(k) == 0.0 else ""  # noqa: E731
    rows = [
        _row(
            "HBM bandwidth, copy",
            g("hbm_copy_bw"),
            "GB/s",
            GB,
            ds.get("hbm_bw"),
            "cold copy, plateau",
            sp("hbm_copy_bw"),
        ),
        _row(
            "HBM bandwidth, read",
            g("hbm_read_bw"),
            "GB/s",
            GB,
            ds.get("hbm_bw"),
            "cold sum of 64 rows, plateau",
            sp("hbm_read_bw"),
        ),
        _row(
            "HBM bandwidth, full reduction",
            g("hbm_reduce_bw"),
            "GB/s",
            GB,
            ds.get("hbm_bw"),
            "cold sum(), plateau",
            sp("hbm_reduce_bw"),
        ),
        _row(
            "HBM bandwidth, write",
            g("hbm_write_bw"),
            "GB/s",
            GB,
            ds.get("hbm_bw"),
            "cold fill_(), plateau",
            sp("hbm_write_bw"),
        ),
        _row(
            "L2 bandwidth (copy, warm), lower bound",
            g("l2_bw"),
            "GB/s",
            GB,
            None,
            "graph burst; includes per-kernel gap",
        ),
        _row(
            "L2 bandwidth, gap-corrected",
            g("l2_bw_gap_corrected"),
            "GB/s",
            GB,
            None,
            "graph burst minus measured gap",
        ),
        _row(
            "L2 capacity (knee)",
            g("l2_knee_bytes"),
            "MiB",
            2**20,
            ds.get("l2_bytes"),
            "warm copy sweep knee",
        ),
    ]
    for tag, label in (
        ("h2d_pinned", "H2D pinned"),
        ("h2d_pageable", "H2D pageable"),
        ("d2h_pinned", "D2H pinned"),
        ("d2h_pageable", "D2H pageable"),
    ):
        rows.append(
            _row(
                label,
                g(f"{tag}_bw"),
                "GB/s",
                GB,
                ds.get("pcie_bw"),
                "best size; plateau "
                + (f"{g(tag + '_bw_plateau') / GB:.1f}" if g(tag + "_bw_plateau") else "n/a"),
            )
        )
    rows += [
        _row(
            "GPU0 -> GPU1 (P2P)",
            g("p2p_bw"),
            "GB/s",
            GB,
            ds.get("p2p_bw"),
            "copy_, best size" + wrong("p2p_correct"),
        ),
        _row(
            "GPU0 <-> GPU1 (both ways)",
            g("p2p_bidir_bw"),
            "GB/s",
            GB,
            2 * ds["p2p_bw"] if "p2p_bw" in ds else None,
            "two streams, bytes of both" + wrong("p2p_correct"),
        ),
        _row(
            "GPU0 -> host -> GPU1 (staged)",
            g("staged_bw"),
            "GB/s",
            GB,
            ds.get("p2p_bw"),
            "pipelined through pinned host buffers" + wrong("staged_correct"),
        ),
    ]
    for dt in ("fp32", "tf32", "fp16", "bf16", "int8"):
        rows.append(
            _row(
                f"matmul {dt}",
                g(f"matmul_{dt}"),
                "TFLOP/s" if dt != "int8" else "TOP/s",
                TF,
                ds.get(dt),
                "best square GEMM (burst)",
                sp(f"matmul_{dt}"),
            )
        )
    rows.append(
        _row(
            "matmul bf16, sustained",
            g("matmul_bf16_sustained"),
            "TFLOP/s",
            TF,
            ds.get("bf16"),
            "8192^3 back to back, second half of the run",
        )
    )
    rows.append(
        _row(
            "  same, scaled to the boost clock",
            g("matmul_bf16_sustained_at_boost"),
            "TFLOP/s",
            TF,
            ds.get("bf16"),
            "per-sample FLOP/s x boost / clock",
        )
    )
    rows.append(
        _row(
            "SIMT fp32 (FMA chain)",
            g("simt_fp32"),
            "TFLOP/s",
            TF,
            ds.get("simt"),
            "compiled FMA chain",
        )
    )
    for key, label, how in (
        ("launch_s", "tiny kernel, device side", "events, queue kept ahead"),
        ("empty_event_s", "empty event pair", "timer floor"),
        ("enqueue_s", "host enqueue per kernel", "no sync in timed loop"),
        ("graph_kernel_s", "kernel in a CUDA graph", "graph replay / n"),
        ("sync_s", "synchronize(), idle", "wall clock"),
        ("item_s", ".item() round trip", "wall clock"),
        ("copy4_pinned_s", "4-byte H2D, pinned, + sync", "wall clock"),
    ):
        rows.append(_row(label, g(key), "us", 1e-6, None, how, sp(key)))
    return rows


def prediction_rows(
    names: Sequence[str],
    measured: Sequence[float],
    predictions: Dict[str, Sequence[float]],
    bounds: Sequence[str],
) -> List[Dict]:
    """One row per kernel: measured time and each model's prediction with its signed error."""
    rows = []
    for i, name in enumerate(names):
        r: Dict[str, object] = {
            "kernel": name,
            "bound": bounds[i],
            "measured_us": f"{measured[i] * 1e6:,.1f}",
        }
        for model, pred in predictions.items():
            err = (pred[i] - measured[i]) / measured[i]
            r[f"{model}_us"] = f"{pred[i] * 1e6:,.1f}"
            r[f"{model}_err"] = f"{100 * err:+.0f}%"
        rows.append(r)
    return rows


def write(doc: Dict, path: str, markdown: Optional[str] = None) -> str:
    """Write ``doc`` as JSON to ``path`` and its Markdown rendering to ``path`` with .md."""
    validate(doc)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    md_path = os.path.splitext(path)[0] + ".md"
    with open(md_path, "w") as f:
        f.write(markdown if markdown is not None else render_markdown(doc))
    return md_path


def render_markdown(doc: Dict) -> str:
    v = doc["versions"]
    lines = [
        f"# Spec sheet: {v.get('hardware', '?')}",
        "",
        f"driver {v.get('driver', 'n/a')}, CUDA {v.get('cuda')}, torch {v.get('torch')}, "
        f"config `{doc['config'].get('name', '?')}`",
        "",
    ]
    ds = DATASHEETS.get(doc["config"].get("datasheet", ""), None)
    lines += [to_markdown(spec_sheet_rows(doc["metrics"], ds)), ""]
    pred = doc["artifacts"].get("prediction_rows")
    if pred:
        lines += ["## Predicted vs measured kernel times", "", to_markdown(pred), ""]
    summ = doc["artifacts"].get("model_errors")
    if summ:
        lines += ["## Model error summary", "", to_markdown(summ), ""]
    return "\n".join(lines)
