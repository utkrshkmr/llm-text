"""Project 3 (reference solution): pre-registered predictions (Chapter 51).

A prediction is only a prediction if it is written down before the run it predicts, and if
nobody can quietly swap the data it was fitted on. The prediction file therefore records a
hash of the runs file and of the fitting code at the time of writing, plus its own hash; the
held-out driver refuses to start unless both still verify.

    python p3_prereg.py write --runs out/runs.jsonl --config configs/full.json --out predictions/heldout.json
    python p3_prereg.py verify --prediction predictions/heldout.json --runs out/runs.jsonl
    python p3_prereg.py score  --prediction predictions/heldout.json --observed 3.21
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from typing import Dict, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))     # code/ for llmsys

REQUIRED = ("schema", "created_utc", "runs_sha256", "code_sha256", "heldout", "fit",
            "prediction", "self_sha256")
SCHEMA = "p3-prediction-v1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_files(paths: Sequence[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def self_hash(pred: dict) -> str:
    body = {k: v for k, v in pred.items() if k != "self_sha256"}
    return hashlib.sha256(canonical(body)).hexdigest()


def code_files() -> list:
    return [os.path.join(HERE, f) for f in ("p3_fit.py", "p3_prereg.py")]


def make_prediction(runs_path: str, heldout: dict, fit: dict, prediction: dict,
                    notes: str = "", code_paths: Optional[Sequence[str]] = None) -> dict:
    """Assemble a prediction record. ``heldout`` is the exact training config of the run to
    come, ``fit`` the fitted parameters, ``prediction`` the point and interval."""
    for key in ("loss", "lo", "hi"):
        if key not in prediction:
            raise ValueError(f"prediction needs '{key}'")
    pred = dict(schema=SCHEMA,
                created_utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                runs_sha256=sha256_file(runs_path), runs_path=os.path.basename(runs_path),
                code_sha256=sha256_files(code_paths or code_files()),
                heldout=heldout, fit=fit, prediction=prediction, notes=notes)
    pred["self_sha256"] = self_hash(pred)
    return pred


def write_prediction(path: str, pred: dict) -> str:
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: a pre-registration is written once")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(pred, indent=2, sort_keys=True) + "\n")
    return path


class PreregistrationError(RuntimeError):
    pass


def verify_prediction(path: str, runs_path: Optional[str] = None,
                      code_paths: Optional[Sequence[str]] = None, check_code: bool = True) -> dict:
    """Load and verify a prediction file: schema, every required field, its own hash, the
    hash of the runs it was fitted on and (optionally) of the fitting code."""
    with open(path) as f:
        pred = json.load(f)
    missing = [k for k in REQUIRED if k not in pred]
    if missing:
        raise PreregistrationError(f"missing fields: {missing}")
    if pred["schema"] != SCHEMA:
        raise PreregistrationError(f"unknown schema {pred['schema']!r}")
    if self_hash(pred) != pred["self_sha256"]:
        raise PreregistrationError("prediction file was edited after it was written")
    if runs_path is not None and sha256_file(runs_path) != pred["runs_sha256"]:
        raise PreregistrationError("runs file differs from the one the prediction was fitted on")
    if check_code and sha256_files(code_paths or code_files()) != pred["code_sha256"]:
        raise PreregistrationError("fitting code changed since the prediction was written")
    p = pred["prediction"]
    if not (p["lo"] <= p["loss"] <= p["hi"]):
        raise PreregistrationError("point prediction outside its own interval")
    return pred


def score_prediction(pred: dict, observed: float) -> Dict[str, float]:
    p = pred["prediction"]
    return dict(observed=observed, predicted=p["loss"], error=observed - p["loss"],
                rel_error=(observed - p["loss"]) / observed, lo=p["lo"], hi=p["hi"],
                inside=bool(p["lo"] <= observed <= p["hi"]),
                half_width=0.5 * (p["hi"] - p["lo"]))


# ----------------------------------------------------------------------------- CLI
def _cmd_write(a):
    import p3_fit as F
    from p3_sweep import heldout_config, load_config
    cfg = load_config(a.config)
    hd = heldout_config(cfg)
    runs = F.load_runs(a.runs)
    stats = F.standard_statistics(hd["n_params"], hd["tokens"])
    ci = F.bootstrap_statistics(runs, stats, n_boot=a.n_boot, seed=a.seed)
    chin = F.fit_parametric(runs)
    iso = F.fit_isoflop(runs)
    fit = dict(chinchilla=chin.as_dict(), isoflop=dict(n_law=iso.n_law, d_law=iso.d_law, l_law=iso.l_law),
               intervals={k: v.as_dict() for k, v in ci.items()}, n_runs=len(runs))
    pl = ci["pred_loss"]
    pred = make_prediction(a.runs, hd, fit, dict(loss=pl.point, lo=pl.lo, hi=pl.hi, level=0.9,
                                                 method="chinchilla+bootstrap"), notes=a.notes)
    print(write_prediction(a.out, pred))
    print(json.dumps(pred["prediction"], indent=1))


def _cmd_verify(a):
    pred = verify_prediction(a.prediction, a.runs)
    print("ok", pred["self_sha256"][:16], json.dumps(pred["prediction"]))


def _cmd_score(a):
    pred = verify_prediction(a.prediction, None, check_code=False)
    print(json.dumps(score_prediction(pred, a.observed), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--runs", required=True)
    w.add_argument("--config", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--n-boot", type=int, default=400)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--notes", default="")
    w.set_defaults(fn=_cmd_write)
    v = sub.add_parser("verify")
    v.add_argument("--prediction", required=True)
    v.add_argument("--runs")
    v.set_defaults(fn=_cmd_verify)
    s = sub.add_parser("score")
    s.add_argument("--prediction", required=True)
    s.add_argument("--observed", type=float, required=True)
    s.set_defaults(fn=_cmd_score)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
