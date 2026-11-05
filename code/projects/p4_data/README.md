# Project 4 infrastructure: filtering web data for language modeling

Generic tooling for Project 4; it contains **no filter implementations**
(see Chapter 45 for the guide and Chapter 14 for the concepts).

| script | purpose |
|---|---|
| `env_check.py` | verify packages (FastWARC, Resiliparse, fastText, ...), CPUs, disk and data paths |
| `inspect_samples.py` | reservoir-sample WARC/WET/JSONL records, print them, write an annotation CSV |
| `train_on_bin.py` | fixed-recipe GPT training on a `uint16` token file (recipes `tiny`/`small`/`medium`, single GPU or `torchrun`), logging validation loss and epochs over the data |
| `shard_runner.py` | resumable parallel map of *your* `fn(in_path, out_path) -> counters` over files, with a manifest, summed counters, and an ETA / full-crawl extrapolation |

```bash
python projects/p4_data/env_check.py --data-root data/   # expects data/cc and data/models
python projects/p4_data/inspect_samples.py --inputs example.warc.wet.gz --n 25 --csv notes.csv
python projects/p4_data/shard_runner.py --fn my_filters:process_file --inputs wet/*.gz \
    --out-dir filtered/ --workers 32 --total-files 90000
python projects/p4_data/train_on_bin.py --recipe small --train-bin train.bin --val-bin val.bin --out runs/v1
pytest -q projects/p4_data
```
