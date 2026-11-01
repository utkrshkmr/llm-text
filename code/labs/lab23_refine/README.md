# Lab 23 — Scoring responses and score-gated self-refinement

**Goal.** Implement a rule-based scorer, one-pass answer log-probabilities and confidence
summaries, the critique→refine loop with score-gated acceptance, and a token cost model that
compares refinement against parallel sampling. Book: Chapter 23.

**Tasks:** `heuristic_score`, `answer_token_logprobs`, `confidence_scores`, `self_refine`,
`refine_cost_tokens`.

```bash
cd code
LAB_IMPL=starter python -m pytest -q labs/lab23_refine
python labs/lab23_refine/run.py --ckpt ckpt/qwen3-0.6b-base --limit 20 --scorer logprob
```
