# Lab 22 — Sampling, self-consistency, and shared-prefix parallel sampling

**Goal.** Implement temperature scaling, nucleus (top-p) filtering and inverse-CDF sampling;
majority voting with tie-breaking and early stopping; and a sampler that prefills a prompt once
and decodes `n` samples as one batch over a broadcast KV cache. Book: Chapter 22.

**Tasks:** `softmax_with_temperature`, `top_p_filter_probs`, `sample_from_probs`,
`majority_vote`, `self_consistency`, `sample_n_shared_prefix`, `forward_token_counts`.

```bash
cd code
LAB_IMPL=starter python -m pytest -q labs/lab22_tts
python labs/lab22_tts/run.py --ckpt ckpt/qwen3-0.6b-base --n 8 --limit 20 --cot
```
