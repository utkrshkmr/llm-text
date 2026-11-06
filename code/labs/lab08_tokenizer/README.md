# Lab 8 — Tokenizers and the input pipeline

Implement a word-level tokenizer, byte-level BPE (naive and incremental training),
sliding-window sampling, document packing, memory-mapped token shards, and the
token+position embedding layer. See Chapter 8, "Lab 8".

```bash
cd code
pytest -q labs/lab08_tokenizer                      # grades starter.py (your work)
LAB_IMPL=solution pytest -q labs/lab08_tokenizer    # reference solution
python labs/lab08_tokenizer/bench.py --vocab 600    # measurements
```
