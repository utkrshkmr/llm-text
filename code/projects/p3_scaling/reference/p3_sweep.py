"""Project 3 (reference solution): the sweep driver (Chapter 27).

    python p3_sweep.py lrsweep --config ../configs/smoke.json [--synthetic] --out ../out/smoke
    python p3_sweep.py sweep   --config ../configs/smoke.json [--synthetic] --out ../out/smoke
    python p3_sweep.py heldout --config ../configs/smoke.json --out ../out/smoke \
                               --prediction ../predictions/smoke.json

One process, one GPU. Every finished unit of work is appended to ``<out>/runs.jsonl`` as one
JSON record, keyed so that re-running the command skips what is already there; checkpoints of
the stable phases live under ``<out>/ckpt/``. The analysis (``p3_report.py``) and the
pre-registration (``p3_prereg.py``) read only ``runs.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # code/ for llmsys

import p3_data as D  # noqa: E402
import p3_wsd as W  # noqa: E402
from p3_mup import Shape, shape_for_params  # noqa: E402


# ----------------------------------------------------------------------------- config
def load_config(path: str, synthetic: bool = False) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    if synthetic:
        for k, v in cfg.get("synthetic", {}).items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = dict(cfg[k], **v)
            else:
                cfg[k] = v
        cfg["is_synthetic"] = True
    cfg["config_path"] = path
    return cfg


def shape_from_cfg(cfg: dict, n_params: float) -> Shape:
    m = cfg["model"]
    return shape_for_params(n_params, head_dim=m["head_dim"], aspect=m["aspect"],
                            vocab=cfg["data"]["vocab"], context=m["context"], min_layers=m.get("min_layers", 2))


def batch_size(cfg: dict) -> int:
    return max(1, cfg["train"]["batch_tokens"] // cfg["model"]["context"])


def tokens_for_flops(shape: Shape, flops: float) -> float:
    return flops / shape.flops_per_token()


def heldout_config(cfg: dict) -> dict:
    """The exact held-out run: shape, tokens and steps derived from the config's budget."""
    h = cfg["heldout"]
    shape = shape_from_cfg(cfg, h["n_params"])
    bs = batch_size(cfg)
    bt = bs * shape.context
    steps = max(1, int(round(tokens_for_flops(shape, h["flops"]) / bt)))
    return dict(size_id="heldout", d=shape.d, L=shape.L, head_dim=shape.head_dim, vocab=shape.vocab,
                context=shape.context, n_params=shape.non_embedding_params, batch_size=bs,
                steps=steps, tokens=steps * bt, flops=steps * bt * shape.flops_per_token(),
                lr=cfg["train"]["lr"], base_width=cfg["model"]["base_width"],
                decay_frac=cfg["train"]["decay_frac"], warmup_frac=cfg["train"]["warmup_frac"],
                data_seed=cfg["train"].get("data_seed", 0), model_seed=cfg["train"].get("model_seed", 0))


def plan_sweep(cfg: dict) -> List[dict]:
    """For every size, the token budgets (one per IsoFLOP budget inside the allowed
    tokens-per-parameter window) and the corresponding branch and end steps."""
    sw, bs = cfg["sweep"], batch_size(cfg)
    plan = []
    for n_target in sw["sizes"]:
        shape = shape_from_cfg(cfg, n_target)
        bt = bs * shape.context
        n = shape.non_embedding_params
        ends = []
        for c in sw["budgets"]:
            d_end = tokens_for_flops(shape, c)
            if not sw["min_tokens_per_param"] <= d_end / n <= sw["max_tokens_per_param"]:
                continue
            end_step = max(cfg["train"].get("min_steps", 20), int(round(d_end / bt)))
            branch_step = max(1, int(round(end_step * (1 - cfg["train"]["decay_frac"]))))
            ends.append(dict(budget=c, end_step=end_step, branch_step=branch_step))
        if ends:
            plan.append(dict(size_id=f"N{n_target:.3g}", n_target=n_target, shape=shape, n_params=n,
                             batch_size=bs, branches=ends))
    return plan


# ----------------------------------------------------------------------------- records
class Records:
    """Append-only JSONL with a set of keys for idempotent re-runs."""

    def __init__(self, path: str):
        self.path = path
        self.rows: List[dict] = []
        if os.path.exists(path):
            with open(path) as f:
                self.rows = [json.loads(l) for l in f if l.strip()]
        self.keys = {r["key"] for r in self.rows if "key" in r}

    def has(self, key: str) -> bool:
        return key in self.keys

    def append(self, rec: dict) -> dict:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.rows.append(rec)
        self.keys.add(rec["key"])
        return rec


def _device_info(device: str) -> dict:
    info = dict(torch=torch.__version__, device=device)
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


def _amp_dtype(cfg: dict, device: str) -> Optional[torch.dtype]:
    if device.startswith("cuda") and cfg["train"].get("amp") == "bf16":
        return torch.bfloat16
    return None


def _data(cfg: dict, out: str):
    """Load the token file; build the synthetic one on demand."""
    path = cfg["data"]["path"]
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(cfg["config_path"])), path)
    if not os.path.exists(path):
        if cfg.get("is_synthetic"):
            toks, h = D.synthetic_tokens(int(cfg["data"]["tokens"]), cfg["data"]["vocab"],
                                         seed=cfg["train"].get("data_seed", 0))
            D.write_bin(toks, path, dict(source="synthetic-markov", entropy_nats=h, vocab=cfg["data"]["vocab"]))
        else:
            raise FileNotFoundError(f"{path}: run p3_data.py first (or pass --synthetic)")
    return D.load_bin(path)


# ----------------------------------------------------------------------------- commands
def cmd_sweep(cfg: dict, out: str, device: str, max_seconds: float) -> List[dict]:
    t_start = time.time()
    train_toks, val_toks, _ = _data(cfg, out)
    recs = Records(os.path.join(out, "runs.jsonl"))
    tr, amp = cfg["train"], _amp_dtype(cfg, device)
    new = []
    for item in plan_sweep(cfg):
        shape, bs = item["shape"], item["batch_size"]
        todo = [b for b in item["branches"] if not recs.has(f"{item['size_id']}:branch:{b['budget']:.3g}")]
        if not todo:
            continue
        if time.time() - t_start > max_seconds:
            print(f"[sweep] wall budget reached before {item['size_id']}", flush=True)
            break
        bt = bs * shape.context
        warmup = max(1, int(tr["warmup_frac"] * max(b["end_step"] for b in item["branches"])))

        def factory(shape=shape, bs=bs):
            return W.make_state(shape, train_toks, cfg["model"]["base_width"], tr["lr"], bs,
                                tr["weight_decay"], tr.get("data_seed", 0), tr.get("model_seed", 0), device)

        print(f"[sweep] {item['size_id']}: d={shape.d} L={shape.L} N_ne={shape.non_embedding_params:,} "
              f"branches at {[b['branch_step'] for b in item['branches']]}", flush=True)
        state = factory()
        ck_dir = os.path.join(out, "ckpt", item["size_id"])
        t0 = time.time()
        paths = W.stable_with_branches(state, tr["lr"], warmup, [b["branch_step"] for b in item["branches"]],
                                       ck_dir, tr.get("grad_clip", 1.0), amp_dtype=amp)
        stable_seconds = time.time() - t0
        for b in item["branches"]:
            key = f"{item['size_id']}:branch:{b['budget']:.3g}"
            if recs.has(key):
                continue
            res = W.branch_decay(paths[b["branch_step"]], factory, tr["lr"], warmup,
                                 b["end_step"] - b["branch_step"], val_toks, tr.get("floor", 0.0),
                                 tr.get("decay_shape", "linear"), tr.get("grad_clip", 1.0), amp_dtype=amp)
            rec = dict(key=key, kind="branch", size_id=item["size_id"], budget=b["budget"], d=shape.d,
                       L=shape.L, n_params=shape.non_embedding_params, batch_size=bs,
                       branch_step=b["branch_step"], end_step=res.end_step, branch_tokens=b["branch_step"] * bt,
                       tokens=res.tokens, flops=res.tokens * shape.flops_per_token(), loss=res.loss,
                       decay_seconds=res.seconds, stable_seconds=stable_seconds, lr=tr["lr"],
                       fingerprint=res.fingerprint, **_device_info(device))
            new.append(recs.append(rec))
            print(json.dumps({k: rec[k] for k in ("size_id", "budget", "tokens", "loss", "decay_seconds")}), flush=True)
    return new


def cmd_lrsweep(cfg: dict, out: str, device: str, max_seconds: float) -> List[dict]:
    """Peak-LR basins at two or more proxy widths: muP says the minima line up."""
    t_start = time.time()
    train_toks, val_toks, _ = _data(cfg, out)
    recs = Records(os.path.join(out, "runs.jsonl"))
    ls, tr, amp = cfg["lr_sweep"], cfg["train"], _amp_dtype(cfg, device)
    bs = batch_size(cfg)
    new = []
    for n_target in ls["sizes"]:
        shape = shape_from_cfg(cfg, n_target)
        bt = bs * shape.context
        steps = max(cfg["train"].get("min_steps", 20), int(round(ls["tokens_per_param"] * shape.non_embedding_params / bt)))
        warmup = max(1, int(tr["warmup_frac"] * steps))
        for lr in ls["lrs"]:
            key = f"lr:N{n_target:.3g}:{lr:.3g}"
            if recs.has(key):
                continue
            if time.time() - t_start > max_seconds:
                return new
            st = W.make_state(shape, train_toks, cfg["model"]["base_width"], lr, bs, tr["weight_decay"],
                              tr.get("data_seed", 0), tr.get("model_seed", 0), device)
            sched = W.Schedule(lr, warmup, decay_start=int(steps * (1 - tr["decay_frac"])), decay_end=steps,
                               floor=tr.get("floor", 0.0))
            t0 = time.time()
            W.train_steps(st, sched, steps, tr.get("grad_clip", 1.0), amp_dtype=amp)
            diverged = not all(math.isfinite(l) for l in st.losses[-10:])
            loss = float("nan") if diverged else W.evaluate(st.model, val_toks, shape.context, device=device)
            rec = dict(key=key, kind="lr", size_id=f"N{n_target:.3g}", d=shape.d, L=shape.L,
                       n_params=shape.non_embedding_params, lr=lr, base_width=cfg["model"]["base_width"],
                       steps=steps, tokens=st.tokens, loss=loss, diverged=diverged, seconds=time.time() - t0,
                       **_device_info(device))
            new.append(recs.append(rec))
            print(json.dumps({k: rec[k] for k in ("size_id", "d", "lr", "loss", "seconds")}), flush=True)
    return new


def cmd_heldout(cfg: dict, out: str, device: str, max_seconds: float, prediction: str,
                ckpt_every: int = 500) -> dict:
    """Verify the pre-registration, then train the held-out run it describes."""
    import p3_prereg as P
    runs_path = os.path.join(out, "runs.jsonl")
    pred = P.verify_prediction(prediction, runs_path)
    hd = heldout_config(cfg)
    for k in ("d", "L", "n_params", "steps", "batch_size", "lr"):
        if pred["heldout"][k] != hd[k]:
            raise P.PreregistrationError(f"held-out {k} changed since pre-registration")
    train_toks, val_toks, _ = _data(cfg, out)
    tr, amp = cfg["train"], _amp_dtype(cfg, device)
    shape = Shape(hd["d"], hd["L"], hd["head_dim"], hd["vocab"], hd["context"])
    st = W.make_state(shape, train_toks, hd["base_width"], hd["lr"], hd["batch_size"], tr["weight_decay"],
                      hd["data_seed"], hd["model_seed"], device)
    ck = os.path.join(out, "ckpt", "heldout", "latest.pt")
    if os.path.exists(ck):
        st.load(ck)
        print(f"[heldout] resumed at step {st.step}", flush=True)
    steps = hd["steps"]
    sched = W.Schedule(hd["lr"], max(1, int(hd["warmup_frac"] * steps)),
                       decay_start=int(steps * (1 - hd["decay_frac"])), decay_end=steps, floor=tr.get("floor", 0.0))
    t0 = time.time()
    while st.step < steps:
        W.train_steps(st, sched, min(steps, st.step + ckpt_every), tr.get("grad_clip", 1.0), amp_dtype=amp)
        st.save(ck)
        print(f"[heldout] step {st.step}/{steps} loss {np.mean(st.losses[-50:]):.4f}", flush=True)
        if time.time() - t0 > max_seconds:
            print("[heldout] wall budget reached; rerun to resume", flush=True)
            return {}
    loss = W.evaluate(st.model, val_toks, shape.context, max_batches=64, device=device)
    score = P.score_prediction(pred, loss)
    rec = dict(key="heldout", kind="heldout", **{k: v for k, v in hd.items() if k != "size_id"}, size_id="heldout",
               loss=loss, seconds=time.time() - t0, prediction=pred["prediction"], score=score,
               fingerprint=st.fingerprint(), **_device_info(device))
    Records(os.path.join(out, "heldout.jsonl")).append(rec)
    print(json.dumps(score, indent=1))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "lrsweep", "sweep", "heldout"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true", help="use the config's synthetic override")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-seconds", type=float, default=float("inf"))
    ap.add_argument("--prediction", help="pre-registration file (heldout)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.synthetic)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "plan":
        for item in plan_sweep(cfg):
            sh = item["shape"]
            print(f"{item['size_id']:>8} d={sh.d:4d} L={sh.L:2d} N_ne={item['n_params']:>12,} "
                  f"budgets={[f'{b['budget']:.0e}' for b in item['branches']]} "
                  f"end_steps={[b['end_step'] for b in item['branches']]}")
        hd = heldout_config(cfg)
        print(f" heldout d={hd['d']} L={hd['L']} N_ne={hd['n_params']:,} tokens={hd['tokens']:.3g} steps={hd['steps']}")
    elif a.cmd == "lrsweep":
        cmd_lrsweep(cfg, a.out, a.device, a.max_seconds)
    elif a.cmd == "sweep":
        cmd_sweep(cfg, a.out, a.device, a.max_seconds)
    else:
        if not a.prediction:
            ap.error("heldout needs --prediction")
        cmd_heldout(cfg, a.out, a.device, a.max_seconds, a.prediction)


if __name__ == "__main__":
    main()
