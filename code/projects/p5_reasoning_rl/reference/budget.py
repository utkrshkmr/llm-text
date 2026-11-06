"""GPU-hour accounting for the matched-compute protocol (Chapter 53).

The comparison in Project 5 is fair only if every method receives the same GPU-hours *after*
the shared SFT stage. ``matched_compute_plan`` splits a total budget into the shared stages
(SFT, evaluation) and one equal slice per method, and converts each slice into a step count
from a per-step cost. ``BudgetLedger`` records what a run actually spent so the plan can be
re-calibrated from measurements rather than guesses.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

GPU_HOUR = 3600.0


def gpu_hours(seconds: float, n_gpus: int) -> float:
    """Wall-clock seconds on ``n_gpus`` devices -> GPU-hours (every device is charged)."""
    return seconds * n_gpus / GPU_HOUR


@dataclass
class MethodAllocation:
    method: str
    gpu_hours: float
    step_gpu_seconds: float     # measured or estimated cost of one optimizer step, GPU-seconds
    steps: int                  # floor(gpu_hours * 3600 / step_gpu_seconds)

    @property
    def used_gpu_hours(self) -> float:
        return self.steps * self.step_gpu_seconds / GPU_HOUR


@dataclass
class Plan:
    total_gpu_hours: float
    shared: Dict[str, float]                    # stage -> GPU-hours (sft, eval, ...)
    methods: List[MethodAllocation]

    @property
    def allocated(self) -> float:
        return sum(self.shared.values()) + sum(m.gpu_hours for m in self.methods)

    def as_dict(self) -> Dict:
        return dict(total_gpu_hours=self.total_gpu_hours, shared=dict(self.shared),
                    methods=[asdict(m) for m in self.methods], allocated=self.allocated)


def matched_compute_plan(total_gpu_hours: float, shared: Dict[str, float],
                         step_gpu_seconds: Dict[str, float]) -> Plan:
    """Equal GPU-hours per method after the shared stages; the slices sum to the budget exactly.

    ``shared`` holds the stages every method depends on (SFT warm start, final evaluation);
    ``step_gpu_seconds`` maps each compared method to the GPU-seconds of one of its optimizer
    steps (rollout + scoring + update, on every device it occupies).
    """
    if not step_gpu_seconds:
        raise ValueError("no methods to compare")
    rest = total_gpu_hours - sum(shared.values())
    if rest <= 0:
        raise ValueError(f"shared stages ({sum(shared.values()):.2f} GPU-h) exhaust the budget")
    per_method = rest / len(step_gpu_seconds)
    methods = []
    for name, cost in step_gpu_seconds.items():
        if cost <= 0:
            raise ValueError(f"{name}: step cost must be positive")
        methods.append(MethodAllocation(name, per_method, cost,
                                        int(math.floor(per_method * GPU_HOUR / cost))))
    return Plan(total_gpu_hours, dict(shared), methods)


def rollouts_per_gpu_hour(step_gpu_seconds: float, rollouts_per_step: int) -> float:
    """Second currency of the comparison: samples the policy drew per GPU-hour."""
    return rollouts_per_step * GPU_HOUR / step_gpu_seconds


@dataclass
class BudgetLedger:
    """Charges wall-clock seconds x devices to named phases; serialises next to the results."""
    n_gpus: int = 1
    charges: Dict[str, float] = field(default_factory=dict)     # phase -> GPU-seconds
    _open: Dict[str, float] = field(default_factory=dict, repr=False)

    def start(self, phase: str) -> None:
        self._open[phase] = time.perf_counter()

    def stop(self, phase: str) -> float:
        t = time.perf_counter() - self._open.pop(phase)
        self.charge(phase, t)
        return t

    def charge(self, phase: str, seconds: float, n_gpus: Optional[int] = None) -> None:
        self.charges[phase] = self.charges.get(phase, 0.0) + seconds * (n_gpus or self.n_gpus)

    @property
    def total_gpu_seconds(self) -> float:
        return sum(self.charges.values())

    @property
    def total_gpu_hours(self) -> float:
        return self.total_gpu_seconds / GPU_HOUR

    def remaining_gpu_hours(self, budget_gpu_hours: float) -> float:
        return budget_gpu_hours - self.total_gpu_hours

    def as_dict(self) -> Dict:
        return dict(n_gpus=self.n_gpus, gpu_seconds=dict(self.charges),
                    total_gpu_hours=self.total_gpu_hours)

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2)
