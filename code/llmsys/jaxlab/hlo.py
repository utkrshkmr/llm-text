"""Lowering and inspecting XLA programs (Appendix C): StableHLO, optimised HLO, profiles.

    jax.jit(f).lower(*args).as_text()            StableHLO: what JAX handed to XLA
    jax.jit(f).lower(*args).compile().as_text()  HLO after optimisation: fusions, layouts,
                                                 collectives, the program that actually runs

The helpers here count collectives and fusions in HLO text, dump both texts to files, and
record a profile with ``jax.profiler`` whose ``trace.json.gz`` opens in Perfetto.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from typing import Any, Callable, Iterable

import jax

COLLECTIVES: tuple[str, ...] = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all",
                                "collective-permute")

# An HLO instruction `%name = type op(operands), attrs`; also the StableHLO spelling with `_`.
# Async pairs (`all-reduce-start` / `all-reduce-done`) count once.
_COLLECTIVE_RE = re.compile(
    r"[\s\"](all[-_]reduce|all[-_]gather|reduce[-_]scatter|all[-_]to[-_]all|collective[-_]permute)"
    r"(-start|-done)?[\s(\"]")
_FUSION_RE = re.compile(r"=\s*\S+\s+fusion\((?:[^)]*)\),\s*kind=k(\w+)")


def count_collectives(hlo_text: str) -> dict[str, int]:
    """Number of collective instructions of each kind in HLO (or StableHLO) text."""
    counts = {c: 0 for c in COLLECTIVES}
    for m in _COLLECTIVE_RE.finditer(hlo_text):
        if m.group(2) == "-done":
            continue
        counts[m.group(1).replace("_", "-")] += 1
    return counts


def count_fusions(hlo_text: str) -> dict[str, int]:
    """Fusion instructions by kind (Loop, Input, Output, Custom) in optimised HLO text."""
    return dict(Counter(m.group(1) for m in _FUSION_RE.finditer(hlo_text)))


def _jit(fn: Callable):
    return fn if hasattr(fn, "lower") else jax.jit(fn)


def stablehlo_text(fn: Callable, *args) -> str:
    """The StableHLO module JAX produces for ``fn(*args)`` before XLA optimises it."""
    return _jit(fn).lower(*args).as_text()


def compiled_text(fn: Callable, *args) -> str:
    """The optimised, scheduled HLO that XLA will run (after SPMD partitioning and fusion)."""
    return _jit(fn).lower(*args).compile().as_text()


def cost_summary(fn: Callable, *args) -> dict[str, float]:
    """XLA's own estimate of FLOPs and bytes for the compiled program (per device)."""
    compiled = _jit(fn).lower(*args).compile()
    cost = compiled.cost_analysis() or {}
    if isinstance(cost, list):  # older jaxlib returned a one-element list
        cost = cost[0] if cost else {}
    mem = compiled.memory_analysis()
    out = {"flops": float(cost.get("flops", 0.0)),
           "bytes_accessed": float(cost.get("bytes accessed", 0.0))}
    if mem is not None:
        out["temp_bytes"] = float(getattr(mem, "temp_size_in_bytes", 0))
        out["argument_bytes"] = float(getattr(mem, "argument_size_in_bytes", 0))
    return out


def hlo_report(fn: Callable, *args) -> dict[str, Any]:
    """Everything the lab's run.py prints for one function: both texts and their counts."""
    lowered = _jit(fn).lower(*args)
    before, after = lowered.as_text(), lowered.compile().as_text()
    return {"stablehlo": before, "hlo": after, "collectives": count_collectives(after),
            "fusions": count_fusions(after), "stablehlo_lines": before.count("\n"),
            "hlo_lines": after.count("\n")}


def dump_hlo(fn: Callable, args: Iterable, stem: str) -> tuple[str, str]:
    """Write ``<stem>.stablehlo.mlir`` and ``<stem>.hlo.txt``; returns the two paths."""
    report = hlo_report(fn, *args)
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    paths = (stem + ".stablehlo.mlir", stem + ".hlo.txt")
    for path, text in zip(paths, (report["stablehlo"], report["hlo"])):
        with open(path, "w") as fh:
            fh.write(text)
    return paths


def collective_instructions(hlo_text: str) -> list[str]:
    """The HLO lines that define a collective, trimmed: what to read first in a big module."""
    lines = []
    for line in hlo_text.splitlines():
        s = line.strip()
        if "=" in s and _COLLECTIVE_RE.search(" " + s.split("=", 1)[1][:80] + " "):
            lines.append(s.split(", metadata=")[0])
    return lines


# --------------------------------------------------------------------------- profiles
def profile_trace(fn: Callable, args: Iterable, logdir: str, repeats: int = 3) -> list[str]:
    """Run ``fn`` under ``jax.profiler.trace`` and return the files it wrote.

    The ``*.trace.json.gz`` file is a Chrome/Perfetto trace: open it at ui.perfetto.dev. The
    ``*.xplane.pb`` is the raw XProf dump for TensorBoard's profile plugin.
    """
    args = tuple(args)
    jax.block_until_ready(fn(*args))          # compile outside the trace
    with jax.profiler.trace(logdir):
        for _ in range(repeats):
            out = fn(*args)
        jax.block_until_ready(out)
    return sorted(os.path.join(root, name) for root, _, names in os.walk(logdir) for name in names)


def summarize_trace(path: str, top: int = 10) -> list[tuple[str, float, int]]:
    """Top ``top`` trace events by total duration (name, total microseconds, count).

    Reads the Chrome-trace JSON that ``profile_trace`` produced. On a GPU/TPU the ``X`` events
    on device rows are kernels and collectives; on CPU they are XLA's thunks.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        events = json.load(fh).get("traceEvents", [])
    total, count = Counter(), Counter()
    for ev in events:
        if ev.get("ph") == "X" and "dur" in ev:
            total[ev["name"]] += float(ev["dur"])
            count[ev["name"]] += 1
    return [(name, total[name], count[name]) for name, _ in total.most_common(top)]
