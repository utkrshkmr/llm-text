"""Project 10 reference solution: know your machine (Chapter 4).

Modules: stats (robust statistics), timing (wall/event/burst timers), bandwidth (milestone 1),
compute (milestone 2), latency (milestone 3), kernels + model (milestone 4), report + env
(milestone 5), suite (all milestones in order).
"""

from . import bandwidth, compute, env, kernels, latency, model, report, stats, timing  # noqa: F401
from . import collectives, suite  # noqa: F401
