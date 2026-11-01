# Project 3 infrastructure (Chapter 34): budgeted scaling studies against a remote API

Generic tooling for the CS336 Assignment 3 setting, where you train nothing yourself and instead
buy runs from a shared training service with a wall-clock budget. Nothing here fits a scaling
law or chooses runs for you; that is the assignment.

| File | What it is |
|---|---|
| `mock_api_server.py` | Offline stand-in service with the same request/response *shape* (/budget, /submit, /experiments, /experiment/{id}, /final_submission). Losses come from the synthetic ground truth in `llmsys.scaling`, not from the course's data. |
| `ledger.py` | `HTTPClient` (stdlib only, retries 5xx/network errors, never retries 4xx) and `Ledger`, an append-only JSONL record that makes submission idempotent and survives crashes. |
| `test_a3_infra.py` | Tests: reservation and refund accounting, duplicates, timeouts, and ledger replay. |

```
cd code/projects/a3
python mock_api_server.py --port 8765 --budget-hours 12 --time-scale 3600 &
export A3_BASE_URL=http://127.0.0.1:8765
python -c "from ledger import HTTPClient; print(HTTPClient().budget())"
python -m pytest -q .
```

Develop your whole pipeline against the mock first: submission, polling with backoff, ledger
replay after a crash, and the plots. Then point `A3_BASE_URL` at the real service.
