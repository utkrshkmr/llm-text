"""Configuration for the Project 4 reference pipeline: defaults, loading, content hashing.

Every threshold, model path and seed lives in one dictionary. A stage's output directory is named
after a hash of (stage name, the configuration it depends on, the input list), so two pipelines can
never overwrite each other and a rerun with the same configuration is a no-op (Chapter 45).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict, Optional, Sequence

DEFAULTS: Dict[str, Any] = {
    "name": "default",
    "workers": 4,
    "data": {
        "wet_glob": "data/cc/wet/*.warc.wet.gz",
        "target_jsonl": "data/val/target.jsonl",
        "benchmark_jsonl": ["data/val/benchmark.jsonl"],
        "max_shards": 0,
    },
    "filters": {
        "language": "en",
        "lang_threshold": 0.65,
        "lang_model": None,          # path to fastText lid.176.bin; None = Chapter 14 Naive Bayes
        "pii": True,
        "rules": True,
        "min_words": 50,
        "max_words": 100000,
        "quality": True,
        "quality_threshold": 0.5,
        "quality_model": None,       # path to a fastText quality model; None = hashed n-gram LR
    },
    "exact_dedup": {"enabled": True, "line_policy": "unique_only", "documents": True},
    "near_dedup": {
        "enabled": True, "num_perm": 128, "bands": 16, "ngram": 5, "threshold": 0.8,
        "verify": "exact",           # exact | estimate | none
        "max_bucket": 500,
        "backend": None,             # llmsys.ops backend: None = auto (cpp when built)
    },
    "decontam": {"enabled": True, "ngram": 13, "min_hits": 1},
    "tokenizer": "bytes",            # bytes (vocab 512, for the tiny recipe) | gpt2 (tiktoken)
    "train": {"recipe": "tiny", "max_steps": 60, "eval_every": 30, "seeds": [0, 1]},
    "variants": {},
    "mixture": {"n_evals": 16, "n_init": 8, "seed": 0, "proxy_max_steps": 40, "sources": "domain"},
    "acceptance": {
        "dedup_min_recall": 0.9, "dedup_min_precision": 0.9,
        "decontam_min_recall": 1.0,
        "mixture_max_l1": 0.08,
        "improvement_min_sigmas": 0.0,  # calibrate in the GPU stage
    },
    "synthetic": {
        "n_shards": 8, "docs_per_shard": 60, "seed": 0,
        "overrides": {"tokenizer": "bytes", "train": {"recipe": "tiny", "max_steps": 60, "eval_every": 30}},
    },
}


def deep_update(base: Dict, upd: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Dict] = None) -> Dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                user = yaml.safe_load(f)
            else:
                user = json.load(f)
        cfg = deep_update(cfg, user or {})
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def apply_variant(cfg: Dict, variant: Optional[str]) -> Dict:
    """Variants are named partial configs (e.g. ``{"filters": {"quality": false}}``)."""
    if not variant:
        return cfg
    if variant not in cfg.get("variants", {}):
        raise KeyError(f"unknown variant {variant!r}; known: {sorted(cfg.get('variants', {}))}")
    out = deep_update(cfg, cfg["variants"][variant])
    out["name"] = f"{cfg['name']}-{variant}"
    return out


def content_hash(*parts: Any, n: int = 10) -> str:
    """Short hash of JSON-serialisable parts (sorted keys, so dict order does not matter)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:n]


def stage_dir(root: str, stage: str, depends_on: Dict, inputs: Sequence[str]) -> str:
    """``<root>/<stage>-<hash>``: content-addressed by configuration and input basenames."""
    return os.path.join(root, f"{stage}-{content_hash(stage, depends_on, [os.path.basename(p) for p in inputs])}")
