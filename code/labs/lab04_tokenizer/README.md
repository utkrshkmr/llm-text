# Lab 4 — Tokenizers and the input pipeline

Implement a word-level tokenizer, byte-level BPE (naive and incremental training),
sliding-window sampling, document packing, memory-mapped token shards, and the
token+position embedding layer. See Chapter 4, "Lab 4".

```bash
cd code
pytest -q labs/lab04_tokenizer                      # grades starter.py (your work)
LAB_IMPL=solution pytest -q labs/lab04_tokenizer    # reference solution
python labs/lab04_tokenizer/bench.py --vocab 600    # measurements
```
