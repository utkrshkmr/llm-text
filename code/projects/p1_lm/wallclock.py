"""A wall-clock budget for time-capped training runs (e.g. the speed-run challenge in Chapter 27).

    budget = WallClockBudget(total_s=45 * 60, reserve_s=120)   # keep 2 min for final eval
    for step in range(max_steps):
        if budget.should_stop(step_time_estimate=budget.ema_step):
            break
        ... train ...
        budget.tick()
    final_eval()

``should_stop`` stops *before* a step that would eat into the reserve, using an exponential
moving average of measured step times. Pass ``clock`` to make it testable.
"""
from __future__ import annotations

import time
from typing import Callable, Optional


class WallClockBudget:
    def __init__(self, total_s: float, reserve_s: float = 0.0, ema: float = 0.9,
                 clock: Callable[[], float] = time.perf_counter):
        self.total_s, self.reserve_s, self.ema_beta = total_s, reserve_s, ema
        self.clock = clock
        self.t0 = clock()
        self.last = self.t0
        self.ema_step: Optional[float] = None
        self.steps = 0

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def remaining(self) -> float:
        return self.total_s - self.elapsed()

    def tick(self):
        now = self.clock()
        dt = now - self.last
        self.last = now
        self.steps += 1
        self.ema_step = dt if self.ema_step is None else self.ema_beta * self.ema_step + (1 - self.ema_beta) * dt
        return dt

    def should_stop(self, step_time_estimate: Optional[float] = None) -> bool:
        est = step_time_estimate if step_time_estimate is not None else (self.ema_step or 0.0)
        return self.remaining() - est < self.reserve_s
