# Lab 3 — Tokenizers and the input pipeline

Implement a word-level tokenizer, byte-level BPE (naive and incremental training),
sliding-window sampling, document packing, memory-mapped token shards, and the
token+position embedding layer. See Chapter 3, "Lab 3".

```bash
cd code
pytest -q labs/lab03_tokenizer                      # grades starter.py (your work)
LAB_IMPL=solution pytest -q labs/lab03_tokenizer    # reference solution
python labs/lab03_tokenizer/bench.py --vocab 600    # measurements
```
