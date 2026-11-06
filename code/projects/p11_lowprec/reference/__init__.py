"""Project 11 reference solution: a low-precision training study (Chapter 7).

Modules: emulate (milestone 1, the emulation layer), recipes, models, data, optim and train
(milestone 2, instrumented runs), study (milestone 3, the run matrix) and analysis (milestones 4
and 5: verdicts, the half-ulp reproduction, the report).
"""
from . import emulate, recipes, models, data, optim, train  # noqa: F401
from . import analysis, study, diagnose  # noqa: F401
