"""Project 11, milestones 4-5: verdicts, the half-ulp reproduction and the report (Chapter 7).

Every run is compared with the fp32 baseline of the same model and seed through its *excess*
validation loss, the final loss minus the task's known floor. The gap

    gap = ln(excess_run / excess_fp32)

is 0 for a run as good as fp32 and ln 2 = 0.69 for one that leaves twice as much reducible loss.
A run is "ok" below ``ok``, "degraded" up to ``broken``, "broken" above it, and "diverged" if it
stopped on a non-finite or exploding loss; a broken run whose excess loss improved by less than
``stall_tol`` (relative) over the last ``stall_window`` of training is also marked "stalled".
The thresholds live in the config ("verdict") and are calibrated on the full runs.
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional

from .recipes import FIXES

DEFAULT_VERDICT = {"ok": 0.10, "broken": 0.69, "stall_window": 0.25, "stall_tol": 0.05,
                   "tail": 0.1}


def val_curve(result: dict):
    return [(r["step"], r["val"]) for r in result["records"] if "val" in r]


def final_val(result: dict, tail: float = 0.1) -> float:
    """Mean validation loss over the last ``tail`` fraction of the logged points."""
    if result.get("diverged_step") is not None:
        return math.nan
    v = [y for _, y in val_curve(result)]
    if not v:
        return math.nan
    k = max(1, int(round(tail * len(v))))
    return sum(v[-k:]) / k


def excess(result: dict, tail: float = 0.1) -> float:
    return final_val(result, tail) - result["floor"]


def gap(result: dict, base: dict, tail: float = 0.1) -> float:
    e, b = excess(result, tail), excess(base, tail)
    if not (math.isfinite(e) and math.isfinite(b)) or b <= 0:
        return math.nan if not math.isfinite(e) else math.inf
    return math.log(max(e, 1e-30) / b)


def stalled(result: dict, window: float = 0.25, tol: float = 0.05) -> bool:
    """True if the excess loss fell by less than ``tol`` (relative) over the last ``window``."""
    c = val_curve(result)
    if len(c) < 4:
        return False
    k = max(2, int(round(window * len(c))))
    e0, e1 = c[-k][1] - result["floor"], c[-1][1] - result["floor"]
    return e0 > 0 and (e0 - e1) / e0 < tol


def verdict(result: dict, base: Optional[dict], th: Optional[dict] = None) -> str:
    th = {**DEFAULT_VERDICT, **(th or {})}
    if result.get("diverged_step") is not None or not math.isfinite(final_val(result, th["tail"])):
        return "diverged"
    if base is None or base is result:
        return "baseline"
    g = gap(result, base, th["tail"])
    if g <= th["ok"]:
        return "ok"
    if g <= th["broken"]:
        return "degraded"
    return "broken, stalled" if stalled(result, th["stall_window"], th["stall_tol"]) else "broken"


def _mean_cast(result: dict, role: str, key: str) -> float:
    """Mean over the logged intervals of one cast fraction (NaN intervals, from skipped steps,
    are ignored); this is the number the chapter's tables and figures report."""
    vals = [r["casts"][role][key] for r in result["records"]
            if role in r.get("casts", {}) and key in r["casts"][role]]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else 0.0


def _max_cast(result: dict, role: str, key: str) -> float:
    vals = [r["casts"][role][key] for r in result["records"]
            if role in r.get("casts", {}) and key in r["casts"][role]]
    vals = [v for v in vals if v == v]                     # a skipped step can log NaN mass
    return max(vals) if vals else 0.0


def _last_update(result: dict, key: str) -> float:
    vals = [r["update"][key] for r in result["records"] if key in r.get("update", {})]
    return vals[-1] if vals else math.nan


def _mean_update(result: dict, key: str, last_frac: float = 0.5) -> float:
    vals = [r["update"][key] for r in result["records"] if key in r.get("update", {})]
    if not vals:
        return math.nan
    vals = vals[int(len(vals) * (1 - last_frac)):] or vals
    return sum(vals) / len(vals)


def group_key(result: dict) -> str:
    return result.get("group", f"{result['model']}/{result['name']}")


def baseline_for(result: dict, results: List[dict], baseline: str = "fp32") -> Optional[dict]:
    """The baseline run of the same model, seed and sweep point."""
    want = result.get("base_group")
    for r in results:
        if r["seed"] == result["seed"] and group_key(r) == want:
            return r
    for r in results:
        if r["model"] == result["model"] and r["seed"] == result["seed"] \
                and r["recipe"] == baseline and not r.get("sweep"):
            return r
    return None


def summarize(results: List[dict], th: Optional[dict] = None, baseline: str = "fp32"
              ) -> List[Dict[str, object]]:
    """One row per run group (seeds pooled): losses, gap, verdicts and instrumentation."""
    th = {**DEFAULT_VERDICT, **(th or {})}
    groups: Dict[str, List[dict]] = {}
    for r in results:
        groups.setdefault(group_key(r), []).append(r)
    rows = []
    for key, runs in groups.items():
        finals = [final_val(r, th["tail"]) for r in runs]
        gaps, verdicts = [], []
        for r in runs:
            b = baseline_for(r, results, baseline)
            verdicts.append(verdict(r, None if b is r else b, th))
            gaps.append(gap(r, b, th["tail"]) if b is not None and b is not r else 0.0)
        fin = [f for f in finals if math.isfinite(f)]
        rows.append({
            "group": key, "model": runs[0]["model"], "recipe": runs[0]["recipe"],
            "describe": runs[0]["describe"], "seeds": len(runs),
            "final_val": statistics.mean(fin) if fin else math.nan,
            "final_std": statistics.stdev(fin) if len(fin) > 1 else 0.0,
            "floor": runs[0]["floor"],
            "gap": statistics.mean(gaps) if all(math.isfinite(g) for g in gaps) else math.nan,
            "verdict": "/".join(sorted(set(verdicts))),
            "diverged_at": [r["diverged_step"] for r in runs if r.get("diverged_step") is not None],
            "skipped": max(r["skipped"] for r in runs),
            # means over the logged intervals (then over seeds); x_overflow_max is the worst
            # logged interval, the number that matters for delayed scaling
            "g_underflow": statistics.mean(_mean_cast(r, "g", "underflow") for r in runs),
            "g_underflow_mass": statistics.mean(_mean_cast(r, "g", "underflow_mass")
                                                for r in runs),
            "x_overflow_max": max(_max_cast(r, "x", "overflow") for r in runs),
            "g_overflow": statistics.mean(_mean_cast(r, "g", "overflow") for r in runs),
            "lost": _mean_update(runs[0], "lost"),
            "pred_lost_bf16": _mean_update(runs[0], "pred_lost_bf16"),
            "ratio_p50": _mean_update(runs[0], "ratio_p50"),
            "ms_per_step": 1000 * statistics.mean(
                r["seconds"] / max(r["diverged_step"] or r["steps"], 1) for r in runs),
        })
    return rows


def halfulp_rows(results: List[dict]) -> List[Dict[str, object]]:
    """Milestone 4's reproduction: measured lost-update fraction against the half-ulp rule's
    prediction, for every run that stores its weights without a master copy."""
    rows = []
    for r in results:
        fmt = None
        for f in ("bf16", "fp16"):
            if f"params={f}/rne" in r["describe"]:
                fmt = f
        if fmt is None:
            continue
        pairs = [(rec["update"]["lost"], rec["update"][f"pred_lost_{fmt}"])
                 for rec in r["records"] if "update" in rec]
        if not pairs:
            continue
        rows.append({"group": group_key(r), "seed": r["seed"], "fmt": fmt,
                     "lost_mean": sum(p[0] for p in pairs) / len(pairs),
                     "pred_mean": sum(p[1] for p in pairs) / len(pairs),
                     "max_abs_diff": max(abs(a - b) for a, b in pairs),
                     "lost_final": pairs[-1][0], "pred_final": pairs[-1][1]})
    return rows


def fix_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """For each failure-probing recipe, its gap next to the gaps of the recipes meant to fix it."""
    by = {(r["model"], r["recipe"]): r for r in rows if "@" not in str(r["group"])}
    out = []
    for (model, recipe), r in by.items():
        if recipe not in FIXES:
            continue
        fixes = [(f, by[(model, f)]) for f in FIXES[recipe] if (model, f) in by]
        out.append({"model": model, "failure": recipe, "gap": r["gap"], "verdict": r["verdict"],
                    "fixes": [(f, fr["gap"], fr["verdict"]) for f, fr in fixes]})
    return out


def _f(x, fmt="{:.4f}") -> str:
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else "inf"
    return fmt.format(x)


def render_markdown(doc: dict) -> str:
    v = doc.get("versions", {})
    lines = [f"# Project 11 results: {doc['config'].get('name', '')}", "",
             f"Hardware: {v.get('hardware', '?')} ({v.get('gpu', v.get('device', '?'))}); "
             f"torch {v.get('torch', '?')}, CUDA {v.get('cuda', '?')}. "
             f"Wall time {doc.get('seconds', 0):.0f} s.", "",
             "Cast columns: share of the gradient elements entering GEMMs that were flushed to zero, by "
             "count and by magnitude, as the mean over logged intervals; activation overflow as the "
             "worst logged interval. Update columns: mean over the second half of training.", "",
             "| run | recipe | final val | gap | verdict | g flushed, count (mean) | g flushed, "
             "mass (mean) | x overflow (max) | lost updates | pred. lost (bf16) | ms/step |",
             "|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|"]
    for r in doc["summary"]:
        lines.append(f"| {r['group']} | {r['describe']} | {_f(r['final_val'])} | "
                     f"{_f(r['gap'], '{:+.3f}')} | {r['verdict']} | "
                     f"{_f(r['g_underflow'], '{:.2e}')} | "
                     f"{_f(r.get('g_underflow_mass', 0.0), '{:.2e}')} | "
                     f"{_f(r['x_overflow_max'], '{:.2e}')} | "
                     f"{_f(r['lost'], '{:.3f}')} | {_f(r['pred_lost_bf16'], '{:.3f}')} | "
                     f"{_f(r['ms_per_step'], '{:.1f}')} |")
    hu = doc.get("halfulp", [])
    if hu:
        lines += ["", "## Half-ulp rule: measured vs predicted lost updates", "",
                  "| run | seed | fmt | measured (mean) | predicted (mean) | max abs diff |",
                  "|:--|--:|:--|--:|--:|--:|"]
        for r in hu:
            lines.append(f"| {r['group']} | {r['seed']} | {r['fmt']} | {r['lost_mean']:.4f} | "
                         f"{r['pred_mean']:.4f} | {r['max_abs_diff']:.2e} |")
    fx = doc.get("fixes", [])
    if fx:
        lines += ["", "## Failures and fixes (gap to fp32; ln 2 = 0.69 is twice the excess loss)",
                  "", "| model | failure | gap | fixes |", "|:--|:--|--:|:--|"]
        for r in fx:
            fixes = "; ".join(f"{f} {_f(g, '{:+.3f}')} ({v})" for f, g, v in r["fixes"])
            lines.append(f"| {r['model']} | {r['failure']} | {_f(r['gap'], '{:+.3f}')} "
                         f"({r['verdict']}) | {fixes} |")
    return "\n".join(lines) + "\n"
