"""Pipeline-parallel schedules: generate per-stage op orders, simulate them, measure bubbles.

An op is (kind, microbatch) with kind in {"F", "B"}. Dependencies:
  F(s, j) needs F(s-1, j) ;  B(s, j) needs B(s+1, j) and F(s, j).
Each stage executes its op list strictly in order. Times: forward = tf, backward = tb.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

Op = Tuple[str, int]


def gpipe_order(p: int, m: int) -> List[List[Op]]:
    """All forwards, then all backwards (reverse microbatch order)."""
    return [[("F", j) for j in range(m)] + [("B", j) for j in reversed(range(m))]
            for _ in range(p)]


def one_f_one_b_order(p: int, m: int) -> List[List[Op]]:
    """PipeDream-flush / Megatron 1F1B: warm up with (p-s-1) forwards, then alternate."""
    orders = []
    for s in range(p):
        warm = min(p - s - 1, m)
        ops: List[Op] = [("F", j) for j in range(warm)]
        f, b = warm, 0
        while f < m:
            ops.append(("F", f)); f += 1
            ops.append(("B", b)); b += 1
        while b < m:
            ops.append(("B", b)); b += 1
        orders.append(ops)
    return orders


@dataclass
class Event:
    stage: int
    kind: str
    mb: int
    start: float
    end: float


def simulate(orders: List[List[Op]], tf: float = 1.0, tb: float = 2.0) -> List[Event]:
    p = len(orders)
    done: Dict[Tuple[int, str, int], float] = {}
    ptr = [0] * p
    free = [0.0] * p
    events: List[Event] = []
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for s in range(p):
            if ptr[s] >= len(orders[s]):
                continue
            kind, j = orders[s][ptr[s]]
            deps = []
            if kind == "F" and s > 0:
                deps.append((s - 1, "F", j))
            if kind == "B":
                deps.append((s, "F", j))
                if s < p - 1:
                    deps.append((s + 1, "B", j))
            if all(d in done for d in deps):
                start = max([free[s]] + [done[d] for d in deps])
                end = start + (tf if kind == "F" else tb)
                done[(s, kind, j)] = end
                free[s] = end
                events.append(Event(s, kind, j, start, end))
                ptr[s] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("deadlock: invalid schedule")
    return events


def bubble_fraction(events: List[Event], p: int) -> float:
    makespan = max(e.end for e in events)
    busy = sum(e.end - e.start for e in events)
    return 1.0 - busy / (p * makespan)


def peak_inflight(orders: List[List[Op]]) -> List[int]:
    """Max number of microbatches whose activations a stage holds at once."""
    peaks = []
    for ops in orders:
        live = peak = 0
        for kind, _ in ops:
            live += 1 if kind == "F" else -1
            peak = max(peak, live)
        peaks.append(peak)
    return peaks


def to_tikz(events: List[Event], p: int, unit: float = 0.32, height: float = 0.5) -> str:
    """Render a schedule as a TikZ snippet (used for the book's pipeline figures)."""
    lines = []
    for e in events:
        y = -(e.stage) * height
        color = "fcompute!35" if e.kind == "F" else "fcomm!35"
        lines.append(
            f"\\draw[fill={color}] ({e.start*unit:.3f},{y:.3f}) rectangle "
            f"({e.end*unit:.3f},{y-height*0.85:.3f}) node[midway,font=\\tiny] {{{e.mb + 1}}};")
    for s in range(p):
        lines.append(f"\\node[anchor=east,font=\\scriptsize] at (-0.1,{-(s)*height-height*0.42:.3f}) {{GPU {s}}};")
    return "\n".join(lines)
