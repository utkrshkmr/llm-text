# Lab 4 — Attention, from dot products to IO-aware tiling

Implement non-trainable attention, scaled dot-product self-attention, causal masking
(two equivalent ways) with dropout, multi-head attention (stacked heads and weight
split), the online softmax, and a FlashAttention-style tiled forward pass. See
Chapter 4, "Lab 4". The full forward+backward reference lives in `llmsys/flash.py`.

```bash
cd code
pytest -q labs/lab04_attention                      # grades starter.py
LAB_IMPL=solution pytest -q labs/lab04_attention    # reference solution
python labs/lab04_attention/bench.py                # CPU timings
python labs/lab04_attention/bench.py --device cuda  # GPU timings + peak memory
```
