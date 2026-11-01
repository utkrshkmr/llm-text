import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget  # noqa: E402
import check_env  # noqa: E402
from wallclock import WallClockBudget  # noqa: E402


def test_budget_roundtrip():
    p = budget.plan(tokens=40_960_000, batch=32, context=256)
    assert p["steps"] == 5000
    assert budget.tokens_for(5000, 32, 256) == 40_960_000
    assert budget.plan(steps=10, batch=4, context=8, tok_per_s=32)["hours"] == pytest.approx(320 / 32 / 3600)
    assert budget.steps_for(1001, 10, 10) == 11          # round up: never under-train


def test_wallclock_stops_with_reserve():
    t = [0.0]
    b = WallClockBudget(total_s=10.0, reserve_s=2.0, ema=0.0, clock=lambda: t[0])
    steps = 0
    while not b.should_stop():
        t[0] += 1.0          # each step takes one second
        b.tick()
        steps += 1
    assert steps == 8         # after 8 one-second steps exactly the 2 s reserve is left
    assert b.remaining() >= b.reserve_s


def test_env_report_runs():
    r = check_env.report()
    assert "python" in r and "torch" in r
    assert isinstance(check_env.advice(r), list)
