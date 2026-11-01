# Project A4 infrastructure (CS336 Assignment 4: Data)

Generic tooling for the data assignment; it contains **no filter implementations**
(see Chapter 35 for the guide and Chapter 11 for the concepts).

| script | purpose |
|---|---|
| `env_check.py` | verify packages (FastWARC, Resiliparse, fastText, ...), CPUs, disk and data paths |
| `inspect_samples.py` | reservoir-sample WARC/WET/JSONL records, print them, write an annotation CSV |
| `shard_runner.py` | resumable parallel map of *your* `fn(in_path, out_path) -> counters` over files, with a manifest, summed counters, and an ETA / full-crawl extrapolation |

```bash
python projects/a4/env_check.py --data-root /shared-data
python projects/a4/inspect_samples.py --inputs example.warc.wet.gz --n 25 --csv notes.csv
python projects/a4/shard_runner.py --fn my_filters:process_file --inputs wet/*.gz \
    --out-dir filtered/ --workers 32 --total-files 90000
pytest -q projects/a4
```
