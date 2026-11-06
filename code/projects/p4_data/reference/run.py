"""Command-line driver of the Project 4 reference solution.

    python -m reference.run all      --config configs/smoke.json --out out/smoke --synthetic
    python -m reference.run pipeline --config configs/full.json  --out out/full [--variant baseline]
    python -m reference.run dedup    --config configs/smoke.json --out out/smoke --synthetic
    python -m reference.run mixture  --config configs/smoke.json --out out/smoke --synthetic
    python -m reference.run train    --config configs/smoke.json --out out/smoke --synthetic

Run from ``code/projects/p4_data``. ``--synthetic`` generates the planted synthetic crawl under
``<out>/synthetic`` when the configured WET files are absent and applies the config's
``synthetic.overrides`` (tiny recipe, byte tokenizer), so a smoke run completes in minutes on a CPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
for p in (KIT, os.path.dirname(os.path.dirname(KIT))):
    if p not in sys.path:
        sys.path.insert(0, p)

from reference import config as C  # noqa: E402
from reference import dedup as D  # noqa: E402
from reference import mixture as X  # noqa: E402
from reference import stages as S  # noqa: E402
from reference import synthetic as Y  # noqa: E402
from reference import train_variants as T  # noqa: E402


def resolve_inputs(cfg: dict, out: str, synthetic: bool, log=print):
    """WET shards + eval paths; generate the synthetic crawl when asked or when nothing is there."""
    paths = sorted(glob.glob(cfg["data"]["wet_glob"]))
    if synthetic or not paths:
        syn = cfg["synthetic"]
        d = os.path.join(out, "synthetic")
        if not os.path.exists(os.path.join(d, "manifest.json")):
            log(f"generating synthetic crawl in {d}")
            Y.generate(d, syn["n_shards"], syn["docs_per_shard"], syn["seed"])
        with open(os.path.join(d, "manifest.json")) as f:
            man = json.load(f)
        cfg = C.deep_update(cfg, syn.get("overrides", {}))
        cfg["data"] = {**cfg["data"], "target_jsonl": man["target"], "benchmark_jsonl": [man["benchmark"]],
                       "truth": man["truth"]}
        paths = man["paths"]
    if cfg["data"].get("max_shards"):
        paths = paths[:cfg["data"]["max_shards"]]
    return cfg, paths


def val_bin_for(cfg: dict, out: str) -> str:
    path = os.path.join(out, f"val-{cfg['tokenizer']}.bin")
    if not os.path.exists(path):
        S.tokenize_target(cfg["data"]["target_jsonl"], cfg["tokenizer"], path)
    return path


def cmd_pipeline(cfg, paths, out, a, log=print):
    results = {}
    variants = [a.variant] if a.variant else [None] + sorted(cfg.get("variants", {}))
    for v in variants:
        vcfg = C.apply_variant(cfg, v)
        log(f"== pipeline variant {vcfg['name']} over {len(paths)} shards")
        res = S.run_pipeline(vcfg, paths, os.path.join(out, "pipeline"), log=log)
        log(S.funnel_table(res["funnel"]))
        log(f"tokens {res['tokens']:,}  corpus {res['corpus_hash'][:12]}  {res['seconds']:.1f}s")
        results[vcfg["name"]] = res
    return results


def cmd_dedup(cfg, paths, out, a, log=print):
    """Planted-duplicate evaluation and the (bands, threshold) sensitivity sweep."""
    truth_path = cfg["data"].get("truth")
    if not truth_path:
        log("dedup evaluation needs planted duplicates: run with --synthetic")
        return {}
    texts = Y.read_texts(paths)
    truth = Y.truth_pairs(Y.load_truth(truth_path))
    nd = cfg["near_dedup"]
    rows = D.threshold_sweep(texts, truth, num_perm=nd["num_perm"], ngram=nd["ngram"],
                             verify=nd["verify"], backend=nd.get("backend"))
    log(f"{'bands':>6}{'rows':>6}{'thr':>6}{'cands':>8}{'confirmed':>10}{'recall':>8}{'prec':>8}")
    for r in rows:
        log(f"{r['bands']:>6}{r['rows']:>6}{r['threshold']:>6.2f}{r['candidates']:>8}{r['confirmed']:>10}"
            f"{r['recall']:>8.3f}{r['precision']:>8.3f}")
    with open(os.path.join(out, "dedup_sweep.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def cmd_mixture(cfg, paths, out, a, log=print):
    """Mixture search with real proxy runs over the sources of the default pipeline's output."""
    res = S.run_pipeline(cfg, paths, os.path.join(out, "pipeline"), log=lambda s: None)
    docs = [json.loads(l) for p in res["final_docs"] for l in open(p, encoding="utf-8")]
    mx = cfg["mixture"]
    sources = X.bucket_docs(docs, by=mx["sources"])
    log(f"sources: { {k: len(v) for k, v in sources.items()} }")
    val_bin = val_bin_for(cfg, out)
    evaluate = X.make_proxy_evaluator(docs, sources, cfg["tokenizer"], val_bin, os.path.join(out, "mixture"),
                                      cfg["train"]["recipe"], mx["proxy_max_steps"], n_docs=len(docs),
                                      seed=mx["seed"], log=log)
    result = X.search_mixture(evaluate, len(sources), mx["n_evals"], mx["n_init"], mx["seed"], log=log)
    result["sources"] = sorted(sources)
    with open(os.path.join(out, "mixture", "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    log(f"best mixture {dict(zip(sorted(sources), np.round(result['best_w'], 3).tolist()))} "
        f"loss {result['best_loss']:.4f}")
    return result


def cmd_train(cfg, paths, out, a, log=print):
    """Fixed-recipe runs on every variant, several seeds, with the results table."""
    val_bin = val_bin_for(cfg, out)
    rows = []
    for v in [None] + sorted(cfg.get("variants", {})):
        vcfg = C.apply_variant(cfg, v)
        res = S.run_pipeline(vcfg, paths, os.path.join(out, "pipeline"), log=lambda s: None)
        tr = cfg["train"]
        rows.append(T.train_variant(vcfg["name"], res["train_bin"], val_bin, os.path.join(out, "runs"),
                                    tr["recipe"], tr["seeds"], tr["max_steps"], tr["eval_every"], log=log))
    table = T.results_table(rows)
    log(table)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(table + "\n")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pipeline", "dedup", "mixture", "train", "all"])
    ap.add_argument("--config", default=os.path.join(KIT, "configs", "smoke.json"))
    ap.add_argument("--out", default="out/smoke")
    ap.add_argument("--synthetic", action="store_true", help="use the planted synthetic crawl")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    if a.workers is not None:
        cfg["workers"] = a.workers
    os.makedirs(a.out, exist_ok=True)
    cfg, paths = resolve_inputs(cfg, a.out, a.synthetic)
    if not paths:
        sys.exit(f"no input shards match {cfg['data']['wet_glob']}; pass --synthetic or download WET files")
    with open(os.path.join(a.out, "config.resolved.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    t0 = time.perf_counter()
    if a.cmd in ("pipeline", "all"):
        cmd_pipeline(cfg, paths, a.out, a)
    if a.cmd in ("dedup", "all"):
        cmd_dedup(cfg, paths, a.out, a)
    if a.cmd in ("mixture", "all"):
        cmd_mixture(cfg, paths, a.out, a)
    if a.cmd in ("train", "all"):
        cmd_train(cfg, paths, a.out, a)
    print(f"done in {time.perf_counter() - t0:.1f}s; outputs under {a.out}")


if __name__ == "__main__":
    main()
