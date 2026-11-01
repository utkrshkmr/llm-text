"""Experiment management: configs, run directories, logging, sweeps, plots (Chapter 9).

Design goals: no mandatory dependencies, append-only JSON-lines logs that survive crashes
and can be re-plotted later, wall-clock *and* step on every record, and an optional
Weights & Biases mirror when ``wandb`` is installed and requested.

    cfg = config_from_cli(TrainConfig, sys.argv[1:])     # --lr=3e-4 --n_layers=4 ...
    run = RunLogger.create("runs", "ts-lr3e-4", cfg)       # runs/ts-lr3e-4-<hash>/
    run.log(step, split="train", loss=2.31, lr=3e-4, tokens=step * tok_per_step)
    run.close()
    records = load_jsonl(run.metrics_path)
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence


# --------------------------------------------------------------------------- configs
def _coerce(value: str, typ):
    if typ is bool or isinstance(typ, bool):
        return value.lower() in ("1", "true", "yes", "y", "on")
    if typ in (int, float, str):
        return typ(value)
    try:                                   # Optional[...] / other annotations
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def config_from_cli(cls, argv: Sequence[str], base: Optional[dict] = None):
    """Build a dataclass from defaults, then an optional JSON file, then ``--key=value``.

    ``--config=path.json`` loads a file; later flags override earlier ones. Unknown keys
    raise immediately (typos in sweep scripts are a classic way to waste GPU hours).
    """
    fields = {f.name: f for f in dataclasses.fields(cls)}
    vals: Dict[str, Any] = dict(base or {})
    for arg in argv:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"expected --key=value, got {arg!r}")
        k, v = arg[2:].split("=", 1)
        if k == "config":
            with open(v) as f:
                vals.update(json.load(f))
            continue
        if k not in fields:
            raise KeyError(f"unknown config key {k!r}; valid: {sorted(fields)}")
        typ = fields[k].type
        typ = {"int": int, "float": float, "str": str, "bool": bool}.get(typ, typ)
        default = fields[k].default
        if typ not in (int, float, str, bool) and default is not dataclasses.MISSING:
            typ = type(default)
        vals[k] = _coerce(v, typ)
    for k in list(vals):
        if k not in fields:
            raise KeyError(f"unknown config key {k!r}")
    return cls(**vals)


def config_hash(cfg) -> str:
    d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _git_rev() -> Optional[str]:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------- logging
class RunLogger:
    """Append-only JSONL metrics + a ``config.json`` + ``meta.json`` per run directory.

    Every record gets ``step``, ``wall`` (seconds since the logger was created) and
    ``time`` (unix). Lines are flushed immediately so ``tail -f`` and crash-recovery work.
    """

    def __init__(self, run_dir: str, config=None, stdout: bool = True,
                 wandb_project: Optional[str] = None, flush_every: int = 1):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self._f = open(self.metrics_path, "a", buffering=1)
        self.t0 = time.perf_counter()
        self.stdout = stdout
        self.flush_every = flush_every
        self._n = 0
        cfg = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else (config or {})
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        meta = dict(argv=sys.argv, git=_git_rev(), start=time.time(), python=sys.version.split()[0])
        try:
            import torch
            meta["torch"] = torch.__version__
            meta["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:
            pass
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        self._wandb = None
        if wandb_project:
            try:
                import wandb
                self._wandb = wandb.init(project=wandb_project, name=os.path.basename(run_dir),
                                         config=cfg, dir=run_dir)
            except ImportError:
                print("[explog] wandb not installed; logging to JSONL only", file=sys.stderr)

    @classmethod
    def create(cls, root: str, name: str, config=None, **kw) -> "RunLogger":
        h = config_hash(config) if config is not None else "nocfg"
        return cls(os.path.join(root, f"{name}-{h}"), config, **kw)

    def log(self, step: int, **metrics):
        rec = dict(step=int(step), wall=round(time.perf_counter() - self.t0, 4), time=time.time())
        for k, v in metrics.items():
            rec[k] = float(v) if hasattr(v, "item") else v
        self._f.write(json.dumps(rec) + "\n")
        self._n += 1
        if self._n % self.flush_every == 0:
            self._f.flush()
        if self.stdout:
            shown = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in rec.items() if k not in ("time",))
            print(shown, flush=True)
        if self._wandb is not None:
            self._wandb.log({k: v for k, v in rec.items() if k not in ("step", "time")},
                            step=rec["step"])
        return rec

    def summary(self, **kv):
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(kv, f, indent=2, default=str)
        if self._wandb is not None:
            self._wandb.summary.update(kv)

    def close(self):
        self._f.close()
        if self._wandb is not None:
            self._wandb.finish()


def load_jsonl(path: str) -> List[dict]:
    """Read a metrics file, skipping a possibly truncated last line (crash mid-write)."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def series(records: Iterable[dict], y: str, x: str = "step", where: Optional[dict] = None):
    xs, ys = [], []
    for r in records:
        if y not in r:
            continue
        if where and any(r.get(k) != v for k, v in where.items()):
            continue
        xs.append(r[x]); ys.append(r[y])
    return xs, ys


def plot_runs(run_dirs: Sequence[str], y: str = "loss", x: str = "step",
              where: Optional[dict] = None, out: str = "runs.png", logy: bool = False):
    """Overlay one metric from several runs (x can be 'step', 'wall' or 'tokens')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in run_dirs:
        xs, ys = series(load_jsonl(os.path.join(d, "metrics.jsonl")), y, x, where)
        ax.plot(xs, ys, label=os.path.basename(d.rstrip("/")))
    ax.set_xlabel(x); ax.set_ylabel(y)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out, dpi=150)
    return out


# --------------------------------------------------------------------------- sweeps
def grid(**axes) -> List[dict]:
    """Cartesian product of hyperparameter axes: grid(lr=[1e-3, 3e-3], bs=[32, 64])."""
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(axes[k] for k in keys))]


def sweep_commands(script: str, points: Iterable[dict], fixed: Optional[dict] = None,
                   name_keys: Optional[Sequence[str]] = None) -> List[str]:
    """Turn sweep points into shell commands ``python script --k=v ... --run_name=...``."""
    cmds = []
    for p in points:
        args = dict(fixed or {}); args.update(p)
        keys = name_keys or list(p)
        args["run_name"] = "-".join(f"{k}{p[k]}" for k in keys if k in p)
        cmds.append(" ".join([f"python {script}"] + [f"--{k}={v}" for k, v in args.items()]))
    return cmds


def log_spaced(lo: float, hi: float, n: int) -> List[float]:
    """n values log-uniformly between lo and hi (inclusive), e.g. for LR sweeps."""
    import math
    if n == 1:
        return [lo]
    a, b = math.log10(lo), math.log10(hi)
    return [10 ** (a + (b - a) * i / (n - 1)) for i in range(n)]
