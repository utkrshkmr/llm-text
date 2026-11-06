# Project 3: A Budgeted Scaling Study {#ch:proj3}

You lead the pretraining team for a new model. The final run will get 48 device-hours on one
accelerator; nobody will give you a second chance. Before it starts, you have a separate budget of
12 device-hours to run experiments, which you buy from a shared training service that reports
validation losses and charges wall-clock time. Your job is to choose the architecture, batch size,
learning rate and token count of the final run so that it reaches the lowest possible loss, and to
predict that loss.

This project puts [@ch:scalinglaws] into practice end to end, with nothing simulated on your
side: you write the client, the bookkeeping, the study plan, the fits and the write-up, against a
service that behaves like a real shared cluster (queues, reservations, timeouts, divergences,
noise). Unlike Lab 15, the service runs as a separate process with a *randomised* hidden loss
surface, so you cannot read the answer from the source, and it scores your final choice only after
you commit to it.

::: keyidea
[What this guide is, and is not]{.box-title}

This chapter specifies the project, explains how to set it up on this repository, maps each part to
the material that teaches it, and gives milestones, sanity checks and debugging advice. It
deliberately stops short of solutions. The decisions that make a study good (which runs to buy, how
to fit, what to trust) are the project. Lab 15 builds the individual tools on a toy service; here
you design the study yourself.
:::

![The project at a glance. Part A trains your fitting code on a fixed dataset; Part B is the
budgeted study. The dashed arrow is the most important one: a study that cannot revise its grid
after the first fits wastes its budget.](figures/ch44_proj_scaling-1.svg){#fig:proj3:overview}

## Learning goals {#sec:proj3:goals}

By the end of the project you should be able to:

- fit and compare the IsoFLOP and parametric scaling-law methods, and quantify their uncertainty
  ([@sec:scalinglaws:methods; @sec:scalinglaws:uncertainty]);

- model scale-dependent hyperparameters instead of fixing them
  ([@sec:scalinglaws:hparams]);

- design and execute a staged experiment under a wall-clock budget with reservations and failures
  ([@sec:scalinglaws:design]);

- build the infrastructure such a study needs: an idempotent client, a crash-safe ledger, and
  analysis that is regenerated from that ledger ([@sec:scalinglaws:running]);

- turn a fit into a concrete, finishable configuration and an honest prediction.

::: {#tab:proj3:cheat .table-float}
| **Need** | **Key relation** | **Where** |
|:---|:---|:---|
| Parameters | $N_{\text{ne}} = L(4d^2+3dd_{\text{ff}})$ | [@eq:scalinglaws:12ld2] |
| Compute | $C/D \approx 6(N_{\text{ne}}+Vd)+6LTd$ | [@eq:scalinglaws:fpt] |
| Wall-clock | $t = t_0 + C/(P\cdot\mathrm{MFU})$ | [@sec:scalinglaws:accounting] |
| IsoFLOP optimum | bracketed min or parabola vertex in $\log N$ | [@sec:scalinglaws:method2] |
| Surface | $E + A N^{-\alpha} + B D^{-\beta}$, Huber on $\log L$ | [@eq:scalinglaws:chin] |
| Allocation | $N_{\text{opt}} = G (C/6)^{\beta/(\alpha+\beta)}$ | [@eq:scalinglaws:closed] |
| LR transfer | basin vertex per size; $\text{lr}^*(N) = kN^{a}$ | [@sec:scalinglaws:lr] |
| Batch | $(S/S_{\min}-1)(E/E_{\min}-1)=1$ | [@eq:scalinglaws:critbatch] |
| Uncertainty | bootstrap within budgets; error $\times10^{k\,\delta a}$ | [@sec:scalinglaws:uncertainty] |

: The relations this project relies on, with pointers to where they are derived.
:::

## Setup {#sec:proj3:setup}

Everything lives in `code/projects/p3_scaling/` and uses the `llmsys` package.

``` bash
cd code && pip install -r requirements.txt
cd projects/p3_scaling
python -m pytest -q .                    # infrastructure self-test (no network)
python make_warmup_data.py --seed 41     # Part A data -> data/isoflop_runs_seed41.json
python training_service.py --port 8765 --budget-hours 12 \
       --time-scale 60 --truth-seed 41 --api-key 12345678 &   # Part B service
export TRAIN_API_URL=http://127.0.0.1:8765 TRAIN_API_KEY=12345678
python -c "from ledger import HTTPClient; print(HTTPClient().budget())"
```

Choose your own seed and do not look at the hidden constants; `--reveal` on the data generator and
`GET /score` on the service exist for *after* you commit. `--time-scale 60` makes one simulated
minute pass per real second, so a 12-hour budget takes about 12 real minutes if you keep the workers
busy; use a larger value while you debug.

![The moving parts. Only the left column is yours to write; the ledger and client in the repository
are a starting point you are free to replace.](figures/ch44_proj_scaling-2.svg){#fig:proj3:arch}

### The service contract {#sec:proj3:contract}

Every request carries an `X-API-Key` header. A run is described by a JSON body:

    {"architecture_config": {"hidden_size": 512, "num_attention_heads": 8, "head_dim": 64,
                             "num_key_value_heads": 8, "num_hidden_layers": 8,
                             "intermediate_size": 1408, "vocab_size": 32000,
                             "tie_word_embeddings": false},
     "optimizer_config":    {"lr_scheduler": {"peak_value": 3e-3}},
     "train_batch_size": 128,          # sequences of 512 tokens
     "total_train_tokens": 1048576000, # must be a multiple of 512 * batch
     "n_evals": 8,                     # validation losses reported along the run
     "max_runtime_seconds": 1800,      # the reservation
     "model_seed": 0}

The rules are those of [@sec:scalinglaws:service]: a submission reserves its
`max_runtime_seconds` until it finishes; completed and failed runs are charged their actual runtime
clipped to $[1,\text{reservation}]$; a run that exceeds its reservation times out, is charged in
full and returns only the losses evaluated before the cut-off; an exact duplicate returns HTTP 409;
a reservation larger than the remaining budget returns 400; malformed configs return 422. The model
is a dense pre-norm transformer with SwiGLU MLPs and a warmup--cosine schedule; losses are mean
next-token cross-entropy on held-out data. You are told nothing else about the service's internals,
in particular not its runtime model, which you must estimate.

![The life of a run and what each outcome costs. Your client and ledger must handle every arrow,
including the rejections, which should never be retried
blindly.](figures/ch44_proj_scaling-3.svg){#fig:proj3:states}

### A rehearsal harness {#sec:proj3:rehearsal}

Because the service is yours, you can rehearse. Start it in-process on a free port with a different
truth seed, run your whole pipeline against it, read the score, shut it down, and repeat. The
repository's own tests do exactly this, and the same pattern works for your study code:

``` {#lst:proj3:rehearse .python caption="Starting a throw-away service for a rehearsal (pattern from `test_p3_infra.py`)."}
import threading
from training_service import serve
from ledger import HTTPClient

def rehearse(pipeline, truth_seed, budget_hours=12.0):
    srv, svc = serve(0, budget_hours=budget_hours, time_scale=1e6,
                     api_key="00000000", truth_seed=truth_seed)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        pipeline(HTTPClient(url, "00000000"))          # your study, unchanged
        return HTTPClient(url, "00000000")._call("GET", "/score?hours=48")
    finally:
        srv.shutdown()
```

Rehearse on five or ten seeds that are not your real one, and look at the distribution of regret and
prediction error. A pipeline that is excellent on one seed and poor on another is telling you which
stage is fragile.

## Part A: warm-up fits on a fixed dataset {#sec:proj3:partA}

`make_warmup_data.py` writes 45 finished runs (five IsoFLOP budgets from $10^{17}$ to $10^{19}$
FLOPs, nine sizes each) drawn from a hidden surface. Each record has `n_params`, `tokens`, `flops`
and `loss`.

![The raw warm-up data for the example seed 0 shipped in `data/` (one line per FLOP budget, in
FLOPs). This is all you get: no fits, no truth. Generate your own seed for the project; seed 0 is
only for checking that your loading and plotting code
works.](figures/ch44_proj_scaling-4.svg){#fig:proj3:raw}

1.  **IsoFLOP allocation.** Find each budget's optimum and fit power laws $N_{\text{opt}}(C)$ and
    $D_{\text{opt}}(C)$. Plot the minima and the fits extrapolated to $10^{24}$, and report the
    predicted $N_{\text{opt}}$ and $D_{\text{opt}}$ at $10^{23}$ and $10^{24}$ FLOPs. Teaching
    material: [@sec:scalinglaws:method2], [@lst:scalinglaws:isoflop].

2.  **Parametric fit.** Fit [@eq:scalinglaws:chin] to all 45 runs, derive the allocation in
    closed form ([@eq:scalinglaws:closed]), and compare with A1. Material:
    [@sec:scalinglaws:method3].

3.  **Uncertainty.** Bootstrap both estimates at $10^{24}$ FLOPs and report intervals. Which method
    is more stable, and why? Material: [@sec:scalinglaws:uncertainty].

4.  **Reveal.** Only after writing down A1--A3, run `--reveal` and discuss what you got right and
    wrong.

::: pitfall
[Part A sanity checks]{.box-title}

Your two allocation exponents from A1 should add to almost exactly one, because the dataset was
built with $C=6ND$; if they do not, you are mixing up $N$ and $D$ or grouping runs wrongly. Parabola
vertices must fall inside each budget's range of sizes. If the parametric fit's $E$ is above the
smallest observed loss, the optimiser has found a nonsensical solution; try more starts
([@lst:scalinglaws:chin]).
:::

## Part B: the budgeted study {#sec:proj3:partB}

### B1. Client, ledger and a dry run {#b1.-client-ledger-and-a-dry-run .unnumbered}

Build (or adopt) a client with retries for transient errors only, and a ledger that makes
re-submission free and recovers state after a crash (`ledger.py` is a working starting point).
Regenerate every table and plot from the ledger. Then buy one small run and check: the losses fall;
the first reported loss is plausible for a $V=32{,}000$ vocabulary (below $\ln V\approx10.4$); the
charged runtime is close to your reservation logic. Material: [@sec:scalinglaws:running].

### B2. Calibrate: runtime and learning rate {#b2.-calibrate-runtime-and-learning-rate .unnumbered}

- **Runtime model.** From the first few completed runs, estimate the service's achieved FLOP rate as
  a function of model shape and batch: compute each run's training FLOPs with
  [@eq:scalinglaws:fpt] and divide by its `used_runtime_seconds` (after subtracting a fixed
  overhead you also estimate). You need this model to size reservations and, later, to pick a final
  run that finishes in 48 hours ([@sec:scalinglaws:accounting]).

- **Learning rate.** Sweep the peak LR at a few small sizes, fit each basin and a transfer law
  across sizes ([@sec:scalinglaws:lr]). Diverged runs are information: they mark the right
  edge of the basin.

- Optionally, test whether batch size matters at your scale ([@sec:scalinglaws:batch]).

##### Estimating a runtime model.

Treat the service as a black box that turns FLOPs into seconds. For each completed run you know its
training FLOPs $C_i$ (from your accounting), its shape and batch, and its charged time $t_i$. A
useful first model is $t_i = t_0 + C_i/R_i$, where $t_0$ is a fixed per-run overhead and $R_i$ is
the achieved FLOP rate. Estimate $t_0$ from runs of very different sizes, then study how $R_i$
depends on width and on tokens per batch, for example by regressing $\log R_i$ on $\log d$ and
$\log(\text{batch tokens})$. Expect saturation: rates rise quickly for narrow models and then
flatten. Validate the model on the next few runs before trusting it with a 48-hour decision, and
keep its residuals: they set your reservation margin.

### B3. The IsoFLOP grid {#b3.-the-isoflop-grid .unnumbered}

Choose budgets, sizes and reservations, and write the plan into a versioned file before you submit
it. Use the napkin math of [@sec:scalinglaws:stages]: the largest budget dominates the
cost, so decide it first, and keep part of the 12 hours in reserve for B5.

::: {#tab:proj3:worksheet .table-float}
| **Stage**                | **Runs** | **Planned s** | **Purpose / gate**             | **Spent s** |
|:-------------------------|:--------:|:-------------:|:-------------------------------|:-----------:|
| B1 dry run               |          |               | pipeline works, losses sane    |             |
| B2 runtime calibration   |          |               | runtime model within margin    |             |
| B2 LR sweeps             |          |               | basins bracketed, transfer law |             |
| B3 grid: smaller budgets |          |               | minima interior                |             |
| B3 grid: top budget      |          |               | minimum interior               |             |
| B4 refinement            |          |               | intervals acceptable           |             |
| B5 verification          |          |               | residual inside interval       |             |
| reserve                  |          |               | failures, re-runs              |             |
| total                    |          |    43,200     |                                |             |

: A planning worksheet. Fill in the planned columns before submitting anything and the spent column
as you go; the gap between them is the most honest summary of a study.
:::

### B4. Fit and diagnose {#b4.-fit-and-diagnose .unnumbered}

Fit both methods, look at residuals, bootstrap, and check bracketing. If the largest budget's
minimum is at the edge of its grid, buy more runs there before going on.

### B5. Verification {#b5.-verification .unnumbered}

Spend part of the remaining budget on one run at the largest compute you can afford, at the
predicted optimum, and compare its loss with your prediction ([@sec:scalinglaws:log]).

### B6. Final choice and prediction {#b6.-final-choice-and-prediction .unnumbered}

Search for the configuration that minimises predicted loss *and finishes* within 48 device-hours
under your runtime model ([@sec:scalinglaws:wallchoice]). Submit it with
`POST /final_submission` together with your predicted final loss, then read your score from
`GET /score?hours=48`.

::: napkinmath
[Before you buy anything]{.box-title}

How many FLOPs is 48 device-hours? You do not know the service's MFU yet, but you can bound it. If
the device peaks at $P$ FLOP/s and a well-shaped model reaches 30--45% MFU, 48 hours buys
$48\cdot3600\cdot P\cdot(0.30\ \text{to}\ 0.45)$ FLOPs. With $P=10^{15}$ that is
$5$--$8\times10^{19}$. Your fitting runs should therefore reach at least $10^{17}$--$10^{18}$ FLOPs,
or the final extrapolation spans more than two decades. Replace the guess with your measured rate
after B2, and redo this estimate.
:::

## Milestones, invariants and debugging {#sec:proj3:debug}

::: {#tab:proj3:gates .table-float}
| **Milestone** | **Check before moving on** |
|:---|:---|
| Client + ledger | Killing your script mid-study and rerunning it resubmits nothing; the ledger replays to the same table. |
| Dry run | Losses decrease; the charged runtime matches `used_runtime_seconds`; your predicted runtime is within $\sim$`<!-- -->`{=html}20%. |
| Runtime model | Predictions for the next five runs land within your safety margin; no timeouts. |
| LR transfer | Each basin is bracketed (losses rise on both sides of the vertex). |
| IsoFLOP grid | Every budget's minimum is interior; measured compute within 5% of nominal. |
| Fits | Methods agree within their bootstrap intervals, or you can explain why not. |
| Verification | $|$observed $-$ predicted$|$ is inside your interval. |

: Gates between stages. Each gate is cheap compared with the budget it protects.
:::

::: pitfall
[Common failures]{.box-title}

- **Budget leaks.** Retrying a 400 ("out of budget") in a loop, or reserving the maximum for every
  run, serialises the study; retrying after a timeout without raising the reservation pays twice for
  nothing.

- **Silent mis-accounting.** Using nominal rather than actual compute, or counting embeddings in $N$
  for some runs and not others.

- **Mid-run losses as data.** The intermediate `val_losses` of a cosine run are not the losses of
  shorter runs ([@sec:scalinglaws:wsd]).

- **Fixed hyperparameters.** One LR for all sizes bends the allocation law.

- **Unfinishable final run.** A final config that would take 50 hours scores as if it never
  finished.
:::

## The write-up {#sec:proj3:writeup}

Write the report so that someone else could reproduce your study from it and the ledger alone. It
should contain:

1.  Part A: plots of the minima and both fits, the predictions at $10^{23}$ and $10^{24}$ with
    intervals, and a discussion after the reveal.

2.  The plan as executed: a table of every run bought (stage, config, compute, reservation, charged
    time, outcome) and a budget timeline in the style of [@fig:scalinglaws:gantt].

3.  The runtime and learning-rate models, with fits and residuals.

4.  IsoFLOP profiles, the allocation fits, the parametric fit, and bootstrap intervals.

5.  The verification run and how it changed your prediction.

6.  The final configuration, its predicted loss with an interval, and why you believe it finishes in
    time.

7.  A paragraph on what you would do differently with twice the budget, and with half.

::: {#tab:proj3:rubric .table-float}
| **Component**  | **What earns full marks**                                       | **Weight** |
|:---------------|:----------------------------------------------------------------|-----------:|
| Part A         | Correct fits, intervals, and an honest discussion of the reveal |        15% |
| Infrastructure | Idempotent, crash-safe, analysis regenerated from the ledger    |        15% |
| Study design   | Staged plan, gates respected, budget used deliberately          |        25% |
| Result         | Low regret against the oracle (`/score`)                        |        25% |
| Prediction     | Small prediction error, and the truth inside your interval      |        10% |
| Write-up       | Reproducible, with decisions justified                          |        10% |

: A self-assessment rubric. Regret depends on the hidden surface, so judge it against the spread you
get from re-running your pipeline on a few other truth seeds.
:::

## Questions that come up {#sec:proj3:faq}

##### The service restarted and my budget is back to 12 hours. Can I keep going?

The service keeps its state in memory, so a restart resets it. That is a property of the stand-in,
not of the task: treat the budget recorded in *your* ledger as the binding one, and report any
restart in the write-up. (A production service persists its job table; making `training_service.py`
do so with SQLite is a good extension.)

##### What time scale should I use?

The time scale changes only how long you wait, never what anything costs. Debug at
`--time-scale 1e6`; run the real attempt at a pace that lets you watch the queue (`60`--`600`).

##### Can I use grouped-query attention or tied embeddings?

The service accepts both fields. It prices attention as full multi-head attention, so GQA neither
helps nor hurts here; tied embeddings change the parameter count your accounting must use.

##### Should the final run reuse a configuration I already trained?

Only if it is the best predicted choice. The service scores the final configuration from its hidden
surface, not by looking up an earlier run, and duplicates of fitting runs are rejected anyway.

##### How noisy are the losses?

Measure it: repeat one small configuration with different `model_seed` values. The noise level sets
how much a single run can be trusted and how many sizes per budget you need.

## Strategy discussion {#sec:proj3:strategy}

There is no single right plan, but several principles separate strong studies from weak ones.

- **Buy information where it matters.** The final choice depends most on the allocation law near the
  top of your fitted range and on $E$, so the largest IsoFLOP budget and the verification run
  deserve most of the money. Tiny runs are cheap in FLOPs but not in device time
  ([@sec:scalinglaws:accounting]).

- **Stage, then revise.** Run a coarse grid, fit, and spend the rest where the fit is least certain.
  A plan fixed in advance cannot do this.

- **Exploit flat valleys.** Near the optimum the loss changes slowly with $N$. Being 20% off in size
  costs little, whereas an unfinishable or diverging final run costs everything. When in doubt,
  choose robustness: a slightly smaller model, a conservative LR, a safety margin on runtime.

- **Keep calendar time in mind.** Reservations bound how many runs are in flight. A study that keeps
  every worker busy finishes sooner and leaves time to react.

- **Validate the pipeline on other seeds.** Because the service is self-hosted, you can run your
  entire pipeline against several other truth seeds before your real attempt and measure its regret
  distribution. That is the closest thing to a rehearsal a real team has.

## A suggested schedule {#sec:proj3:schedule}

::: description
Read [@ch:scalinglaws]; complete Lab 15 so the fitting tools are tested. Set up the project
directory, the service and the infrastructure tests.

Part A end to end, including the reveal discussion.

Client, ledger and rehearsal harness; rehearse a deliberately naive pipeline on two seeds to learn
how the service behaves.

Real attempt, stages B1--B3, with the worksheet filled in before each stage.

B4--B6 and the score; rehearse the final pipeline on other seeds for the robustness section.

Write-up.
:::

## Extensions {#sec:proj3:extensions}

- **WSD branching.** Add an endpoint to the service that continues a finished run's stable phase and
  charges only a short decay ([@sec:scalinglaws:wsd]), then redesign the study around
  method 1 and compare its regret per hour of budget.

- **Batch-size law.** Add batch sweeps to B2 and fit $B_{\text{opt}}$ against compute; does it
  change your final choice?

- **Noise model.** Spend a little budget on repeated runs with different `model_seed`; use the
  measured noise to set the Huber $\delta$ and to weight the fit.

- **A different objective.** Re-plan for an inference-heavy deployment
  ([@sec:scalinglaws:beyond]): minimise training plus serving cost for a fixed number of
  served tokens, and report how far your choice moves from the training-optimal one.

## Taking it to real hardware {#sec:proj3:real}

The service is a stand-in for real training. Two ways to replace it:

1.  **Real runs, same study.** Lab 15's `run.py cpu-sweep` trains small Qwen3-shaped models and fits
    what it measures. On a GPU, raise the budgets ($10^{15}$--$10^{17}$ FLOPs with `--device cuda`)
    and use the pretraining loop of [@ch:pretrain] with a corpus from [@ch:data].

2.  **Self-host a real training service.** Keep the HTTP contract of [@sec:proj3:contract]
    and put a real backend behind it: a small web server that validates requests and writes them to
    a job table (SQLite is enough), one worker process per GPU that claims jobs, runs a training
    script with a hard timeout, streams validation losses back to the table, and records the
    measured runtime. The interesting systems questions are the ones you have just been on the other
    side of: how to price reservations, how to handle preemption and partial results, and how to
    keep runs deterministic given a seed. Because `training_service.py` already implements the
    accounting rules, you can swap only the part that produces losses.

::: systemslens
[What the service teaches about real clusters]{.box-title}

Every rule of the service exists on real infrastructure under a different name: reservations are
Slurm time limits, timeouts are jobs killed at the limit with partial checkpoints, 409s are
deduplicated submissions, and the hidden runtime model is the MFU you only learn by profiling. Teams
that run scaling studies at scale build exactly the client, ledger and gates of this project, plus
WSD branching ([@sec:scalinglaws:wsd]) to make each run yield several data points.
:::
