# Lab 8: Beyond softmax attention

Chapter 8 (`ch:attnalt`). Implement linear attention in its parallel, recurrent and
chunkwise forms (with optional per-head or data-dependent decay), the delta-rule update, a
parallel associative scan, one step of a selective SSM, sliding-window masks, and a
decode-memory calculator for full / windowed / linear / hybrid stacks. The reference layers
live in `llmsys/linear_attn.py` and `llmsys/ssm.py`.

```
LAB_IMPL=starter  python -m pytest -q labs/lab08_attnalt
LAB_IMPL=solution python -m pytest -q labs/lab08_attnalt
python labs/lab08_attnalt/bench.py --device cpu          # decode cost vs context length
python labs/lab08_attnalt/bench.py --device cuda --d 1024 --heads 16 --lens 1024 4096 16384 65536
```
Tests use float64 so the three linear-attention forms can be compared to 1e-10.
