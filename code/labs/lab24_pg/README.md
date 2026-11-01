# Lab 24 — Policy-gradient foundations

Build the pieces every LLM RL trainer is made of, on toy problems where you can see what
they do: score-function estimators and baselines on a 3-armed softmax bandit, the RL batch
layout with a response mask, chunked log-probs and entropy, group advantages, loss
aggregation that survives gradient accumulation, importance-weighted and clipped losses,
rejection fine-tuning, and the variance of sequence-level importance ratios.

```
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab24_pg     # your implementation
LAB_IMPL=solution python -m pytest -q labs/lab24_pg     # reference
python labs/lab24_pg/run.py bandit                      # variance vs baseline
python labs/lab24_pg/run.py compare --steps 80 --seeds 3
python labs/lab24_pg/run.py plot pg_runs.json
python labs/lab24_pg/run.py isvar
```
Book: Chapter 24. Tasks 1–8 correspond to the functions in `starter.py`.
