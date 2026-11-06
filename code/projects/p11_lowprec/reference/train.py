"""Project 11, milestone 2: one instrumented training run under a precision recipe (Chapter 5).

``train(cfg)`` builds the task, the model and the recipe, trains with AdamW (warm-up + cosine),
and logs every ``log_every`` steps: training loss, the validation loss of the current weights
evaluated *exactly* (fp32, no emulation), the learning rate, the loss scale and skipped steps,
the overflow/underflow/subnormal fractions of every GEMM operand role, and the update statistics
of ``optim.update_stats``. A run that produces a non-finite loss, or a loss above
``diverge_factor`` times its initial loss, is stopped and marked diverged.

Determinism: on a CPU two runs with the same config produce bit-identical logs (the acceptance
tests check this); on a GPU, atomics in embedding backward make runs agree only statistically.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import torch

from llmsys import numerics as N

from .data import build_task
from .emulate import Log2Histogram
from .models import Precision, build_model
from .optim import LowPrecisionAdamW
from .recipes import make_recipe

DEFAULTS = dict(recipe="fp32", recipe_overrides={}, steps=200, lr=3e-3, min_lr_frac=0.1,
                warmup=20, weight_decay=0.0, betas=(0.9, 0.95), clip=1.0, seed=0, data_seed=0,
                log_every=20, diverge_factor=3.0, hist_steps=[], eval=True, eps=1e-8,
                loss_weight=1.0)


def lr_at(step: int, cfg: dict) -> float:
    """Linear warm-up, then cosine decay to ``min_lr_frac`` of the peak."""
    peak, w, n = cfg["lr"], cfg["warmup"], cfg["steps"]
    if step < w:
        return peak * (step + 1) / w
    t = (step - w) / max(n - w, 1)
    return peak * (cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * t)))


class LossScaler:
    """none: S = 1, never skip (a non-finite gradient reaches the optimiser, as in plain fp16);
    static: fixed S, skip non-finite steps; dynamic: ``numerics.DynamicLossScaler``."""

    def __init__(self, mode: str, init: float, growth_interval: int):
        self.mode = mode
        self.dyn = N.DynamicLossScaler(init, growth_interval=growth_interval) \
            if mode == "dynamic" else None
        self.static = init if mode == "static" else 1.0
        self.skipped = 0

    @property
    def scale(self) -> float:
        return self.dyn.scale if self.dyn else self.static

    def unscale_and_check(self, params) -> bool:
        """Divide the gradients by S (in fp32); return True if the step should be taken."""
        grads = [p.grad for p in params if p.grad is not None]
        s = self.scale
        if s != 1.0:
            for g in grads:
                g.div_(s)
        if self.mode == "none":
            return True
        bad = N.DynamicLossScaler.found_nonfinite(grads)
        take = self.dyn.update(bad) if self.dyn else not bad
        self.skipped += int(not take)
        return take


@torch.no_grad()
def evaluate(model, task, prec: Precision) -> float:
    with prec.exact():
        losses = []
        for xb, yb in task.val_batches():
            if hasattr(task, "start"):
                losses.append(float(model.loss(xb, yb, task.start)))
            else:
                losses.append(float(model.loss(xb, yb)))
    return sum(losses) / len(losses)


def train(cfg: dict, device="cpu", verbose: bool = False, setup=None) -> Dict[str, object]:
    """One run. ``setup(prec)``, if given, is called once the model exists (diagnostics use it to
    attach captures or histograms to particular quantisers)."""
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator(device=device).manual_seed(10_007 + cfg["seed"])
    recipe = make_recipe(cfg["recipe"], **cfg["recipe_overrides"])
    prec = Precision(recipe, gen)
    task = build_task(cfg, device)
    model = build_model(cfg, prec).to(device)
    if setup is not None:
        setup(prec)
    # loss_weight w multiplies the loss, Adam's eps and the clipping threshold: in exact
    # arithmetic training is unchanged (Adam is invariant to the gradient's scale), so any effect
    # of w is an effect of the number format on gradients w times smaller or larger
    w_loss = cfg["loss_weight"]
    opt = LowPrecisionAdamW(model.named_parameters(), cfg["lr"], tuple(cfg["betas"]),
                            eps=cfg["eps"] * w_loss,
                            weight_decay=cfg["weight_decay"], params_fmt=recipe.params,
                            rounding=recipe.update_rounding, generator=gen)
    scaler = LossScaler(recipe.loss_scale, recipe.loss_scale_init, recipe.growth_interval)
    records: List[Dict[str, object]] = []
    hists: Dict[str, object] = {}
    init_loss: Optional[float] = None
    diverged: Optional[int] = None
    run_loss, run_n = 0.0, 0
    t0 = time.perf_counter()
    for step in range(cfg["steps"]):
        hist = Log2Histogram(device=device) if step in cfg["hist_steps"] else None
        for q in prec.quantizers("g"):
            q.hist = hist
        xb, yb = task.batch_at(step)
        loss = model.loss(xb, yb, task.start) if hasattr(task, "start") else model.loss(xb, yb)
        lval = float(loss.detach())
        if init_loss is None:
            init_loss = lval
        if not math.isfinite(lval) or lval > cfg["diverge_factor"] * init_loss:
            diverged = step
            records.append({"step": step, "loss": lval, "diverged": True})
            break
        opt.zero_grad()
        (loss * (w_loss * scaler.scale)).backward()
        if hist is not None:
            hists[str(step)] = hist.as_dict()
        take = scaler.unscale_and_check(opt.params)
        log_now = (step + 1) % cfg["log_every"] == 0 or step == cfg["steps"] - 1
        stats = None
        if take:
            if cfg["clip"]:
                torch.nn.utils.clip_grad_norm_(opt.params, cfg["clip"] * w_loss)
            stats = opt.step(lr_at(step, cfg), measure=log_now)
        run_loss, run_n = run_loss + lval, run_n + 1
        if log_now:
            rec: Dict[str, object] = {"step": step + 1, "loss": run_loss / run_n,
                                      "lr": lr_at(step, cfg), "scale": scaler.scale,
                                      "skipped": scaler.skipped,
                                      "time": time.perf_counter() - t0}
            if cfg["eval"]:
                rec["val"] = evaluate(model, task, prec)
            rec["casts"] = prec.fractions()
            if stats:
                rec["update"] = stats
            prec.reset_counters()
            records.append(rec)
            run_loss, run_n = 0.0, 0
            if verbose:
                print(f"  step {step + 1:5d} loss {rec['loss']:.4f} val {rec.get('val', 0):.4f}",
                      flush=True)
    seconds = time.perf_counter() - t0
    return {"name": cfg.get("name", cfg["recipe"]), "recipe": recipe.name,
            "describe": recipe.describe(), "model": cfg["model"], "seed": cfg["seed"],
            "steps": cfg["steps"], "floor": task.floor, "init_loss": init_loss,
            "diverged_step": diverged, "skipped": scaler.skipped, "seconds": seconds,
            "records": records, "hist": hists,
            "params": sum(p.numel() for p in model.parameters())}
