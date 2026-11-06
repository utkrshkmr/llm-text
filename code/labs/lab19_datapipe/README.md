# Lab 19: A pretraining-data pipeline from raw crawl to clean text

Build every stage of a web-data pipeline yourself, on a synthetic labelled crawl:
charset detection and HTML-to-text extraction, a character-n-gram language identifier,
PII masking, Gopher-style rules, a hashed-n-gram quality classifier, exact line dedup,
MinHash + LSH near-dedup with union-find clustering, a Bloom filter, UniMax mixture weights,
and a multiprocess shard driver with per-stage accounting. See Chapter 19, "Lab 19".

```bash
cd code
pytest -q labs/lab19_datapipe                        # grades starter.py (your work)
LAB_IMPL=solution pytest -q labs/lab19_datapipe      # reference solution
python labs/lab19_datapipe/bench.py stages --docs 2000
python labs/lab19_datapipe/bench.py scaling --docs 4000 --shards 16 --workers 1 2 4 8
python labs/lab19_datapipe/bench.py lsh
python labs/lab19_datapipe/run.py --help             # real Common Crawl files (optional)
```

Tests use only synthetic data and pure Python/NumPy (< 15 s). `run.py` uses FastWARC,
Resiliparse and fastText if installed and falls back to the `llmsys.datapipe` versions otherwise.
On macOS (spawn start method) run the end-to-end task from a script guarded by
`if __name__ == "__main__":`.
