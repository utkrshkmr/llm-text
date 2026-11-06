# Lab 9: Beyond softmax attention

Chapter 9 (`ch:attnalt`). Implement linear attention in its parallel, recurrent and
chunkwise forms (with optional per-head or data-dependent decay), the delta-rule update, a
parallel associative scan, one step of a selective SSM, sliding-window masks, and a
decode-memory calculator for full / windowed / linear / hybrid stacks. The reference layers
live in `llmsys/linear_attn.py` and `llmsys/ssm.py`.

```
LAB_IMPL=starter  python -m pytest -q labs/lab09_attnalt
LAB_IMPL=solution python -m pytest -q labs/lab09_attnalt
python labs/lab09_attnalt/bench.py --device cpu          # decode cost vs context length
python labs/lab09_attnalt/bench.py --device cuda --d 1024 --heads 16 --lens 1024 4096 16384 65536
python labs/lab09_attnalt/recall.py --patterns AA LL MM LA --pairs 8 32   # associative recall
```
Tests use float64 so the three linear-attention forms can be compared to 1e-10.
