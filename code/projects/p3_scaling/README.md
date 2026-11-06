# Project 3 infrastructure (Chapter 27): a budgeted scaling study

Tooling for a study in which you train nothing yourself: you buy runs from a training service
with a wall-clock budget. Nothing here fits a scaling law or chooses runs for you; that is the
project.

| File | What it is |
|---|---|
| `training_service.py` | The self-hosted training service (HTTP: /budget, /submit, /experiments, /experiment/{id}, /final_submission, /score). Losses come from a hidden, randomisable loss surface in `llmsys.scaling`. |
| `ledger.py` | `HTTPClient` (stdlib only; retries 5xx and network errors, never retries 4xx) and `Ledger`, an append-only JSONL record that makes submission idempotent and survives crashes. |
| `make_warmup_data.py` | Generates the warm-up dataset (finished IsoFLOP runs from a hidden surface; `--reveal` shows the truth afterwards). |
| `test_p3_infra.py` | Tests: reservation and refund accounting, duplicates, timeouts, ledger replay, scoring. |

```
cd code/projects/p3_scaling
python training_service.py --port 8765 --budget-hours 12 --time-scale 3600 --truth-seed 7 &
export TRAIN_API_URL=http://127.0.0.1:8765
python -c "from ledger import HTTPClient; print(HTTPClient().budget())"
python -m pytest -q .
```

Pick a `--truth-seed` and don't look at the hidden constants. When you have submitted your
final config, `GET /score?hours=48` reports the true loss of your choice, the oracle optimum
and your regret.
