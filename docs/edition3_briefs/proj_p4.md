# Brief: Project 4, a web-scale data pipeline (chapter 45, kit p4_data)
Read projects_common.md first, then docs/PLAN.md §4 "P4" and §2 (csrc/cpu/minhash.cpp is yours), the existing chapter
and kit (keep shard_runner, train_on_bin, inspect_samples and their tests), Chapter 21 (book + code,
llmsys.datapipe.*), BRIEF_conventions.md §6 for the C++ op. No GPU needed except a short train_on_bin check
(CUDA_VISIBLE_DEVICES=1).
Scope: Common Crawl text -> filtered corpus: WET/WARC reading (fastwarc optional; a pure-Python fallback), language
ID, PII masking, heuristic and model-based quality filters, C++ MinHash/LSH near-duplicate removal
(csrc/cpu/minhash.cpp: shingle hashing, min-hash signatures, LSH banding; op `llmsys::minhash_signatures` +
Python LSH in llmsys.datapipe.dedup reuse), exact dedup, benchmark decontamination, proxy-model data-mixture
search, training small models per variant. reference/: the pipeline stages as resumable deterministic shard jobs,
dedup with planted-duplicate evaluation, mixture search, configs smoke/full (~20 GPU-h + CPU).
Acceptance (CPU, tiny synthetic corpus): stage determinism (same output hash twice) and resumability (kill/restart
via manifest); dedup recall/precision on planted near-duplicates above thresholds, C++ signatures equal the Python
reference; decontamination catches planted benchmark strings; mixture search returns the planted optimum on a
synthetic proxy surface.
