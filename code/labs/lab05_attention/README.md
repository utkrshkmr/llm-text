# Lab 5 — Attention, from dot products to IO-aware tiling

Implement non-trainable attention, scaled dot-product self-attention, causal masking
(two equivalent ways) with dropout, multi-head attention (stacked heads and weight
split), the online softmax, and a FlashAttention-style tiled forward pass. See
Chapter 5, "Lab 5". The full forward+backward reference lives in `llmsys/flash.py`.

```bash
cd code
pytest -q labs/lab05_attention                      # grades starter.py
LAB_IMPL=solution pytest -q labs/lab05_attention    # reference solution
python labs/lab05_attention/bench.py                # CPU timings
python labs/lab05_attention/bench.py --device cuda  # GPU timings + peak memory
```
