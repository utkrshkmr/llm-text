"""Project 4 reference solution: a web-scale data pipeline (Chapter 45).

Attempt the project before reading this package. The acceptance tests (``test_p4_acceptance.py``)
run against it by default and against your own implementation through ``P4_IMPL=/path/to/dir``,
a directory whose ``__init__.py`` exposes the same names:

    generate_synthetic(out_dir, n_shards, docs_per_shard, seed) -> manifest dict
    run_pipeline(cfg, inputs, root, workers=None, log=print, eval_paths=None) -> result dict
    near_dedup(texts, num_perm, bands, ngram, threshold, seed, verify, ...) -> (keep, clusters, stats)
    evaluate_pairs(clusters, truth_pairs) -> {"recall", "precision", ...}
    search_mixture(evaluate, n_sources, n_evals, n_init, seed) -> {"best_w", ...}
    load_config(path=None, overrides=None) -> cfg
    minhash: a module with signatures(texts, num_perm, ngram, seed, backend) and
             band_keys(sigs, bands, backend)
"""
from llmsys.ops import minhash  # noqa: F401  (the compiled operators; python backend always exists)

from .config import DEFAULTS, apply_variant, deep_update, load_config  # noqa: F401
from .dedup import evaluate_pairs, near_dedup, threshold_sweep  # noqa: F401
from .mixture import quadratic_surface, search_mixture  # noqa: F401
from .stages import corpus_hash, funnel_table, run_pipeline, tokenize_target  # noqa: F401
from .synthetic import generate as generate_synthetic  # noqa: F401
from .synthetic import load_truth, truth_pairs  # noqa: F401
from .train_variants import train_variant  # noqa: F401
