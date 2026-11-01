# Lab 9 — Classification fine-tuning and padding-aware batching

Book: Chapter 9, Section "Lab 9".

* `starter.py` — your implementation (functions raise `NotImplementedError`).
* `solution.py` — reference (re-exports `llmsys/finetune.py` + training loop).
* `run.py` — the real thing: fine-tune GPT-2 small on the SMS Spam Collection (needs
  internet + `safetensors` GPT-2 weights; GPU recommended).

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab09_classify   # grade your work
LAB_IMPL=solution python -m pytest -q labs/lab09_classify   # reference
python labs/lab09_classify/run.py --gpt2-dir /path/to/gpt2 --data /path/to/SMSSpamCollection
```
