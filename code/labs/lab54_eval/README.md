# Lab 54 — Building an evaluation harness for reasoning models

**Goal.** Implement the answer-extraction and grading half of a math verifier, a batched and
resumable evaluation loop, multiple-choice scoring (generated letter and logit-based), and two
preference-ranking methods (Elo and Bradley–Terry). Book: Chapter 54.

**Tasks** (in `starter.py`): `last_boxed`, `extract_final_candidate`, `grade_answer`, `evaluate`,
`wilson_interval`, `first_choice_letter`, `predict_by_logits`, `elo_ratings`, `bradley_terry`.

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab54_eval    # grade your work
LAB_IMPL=solution python -m pytest -q labs/lab54_eval    # reference
# real model (needs downloaded Qwen3 checkpoint + MATH-500 JSONL):
python labs/lab54_eval/run.py --ckpt ckpt/qwen3-0.6b-base --data math500_test.jsonl --limit 100
```
