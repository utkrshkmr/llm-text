# Lab 12 — A budgeted scaling-law study (Chapter 12)

You get a *synthetic training API* (`llmsys.scaling.SyntheticTrainingAPI`): submit a model
config and a token count, it "trains" for a simulated wall-clock time, charges your budget and
returns validation losses. A hidden loss surface sits behind it, with learning-rate, batch-size
and aspect-ratio effects plus seed noise. Your job: with 12 simulated accelerator-hours, find
the config that gives the lowest loss for a 48-hour run, and predict that loss.

Tasks (all in `starter.py`):

| # | Function | What it does |
|---|---|---|
| 1 | `count_params_and_flops` | params and training FLOPs of a config |
| 2 | `fit_power_law` | log-log least squares |
| 3 | `isoflop_optima` | min-of-runs and quadratic-vertex IsoFLOP minima |
| 4 | `fit_allocation` | N_opt(C), D_opt(C), L_opt(C) |
| 5 | `fit_chinchilla` | Huber fit of E + A/N^α + B/D^β |
| 6 | `chinchilla_optimal` | closed-form split of C into N and D |
| 7 | `fit_lr_optimum`, `fit_lr_scaling` | LR basin and LR transfer law |
| 8 | `bootstrap_ci` | uncertainty of any fitted quantity |
| 9 | `choose_final_config` | best config that *finishes* in T seconds (MFU depends on shape) |
| 10 | `run_study` | the whole pipeline, within budget |

The fitting functions in `llmsys.scaling` are the reference implementation; don't call them
from your starter.

```
cd code
LAB_IMPL=starter  python -m pytest labs/lab12_scaling_laws     # your work
LAB_IMPL=solution python -m pytest labs/lab12_scaling_laws     # reference
python labs/lab12_scaling_laws/run.py study --impl starter --plot study.png
python labs/lab12_scaling_laws/run.py replicate --seeds 5      # how stable is it?
python labs/lab12_scaling_laws/run.py cpu-sweep                # a real, tiny IsoFLOP sweep
```

`cpu-sweep` trains small Qwen3-shaped byte-level models on the book's own sources at fixed FLOP
budgets and fits the allocation law to what it measures. Expect noisy exponents at these tiny
scales; use `--device cuda --budgets 1e14 3e14 1e15` on a GPU for a cleaner curve.
