# Lab 24 — Classification fine-tuning and padding-aware batching

Book: Chapter 24, Section "Lab 24".

* `starter.py` — your implementation (functions raise `NotImplementedError`).
* `solution.py` — reference (re-exports `llmsys/finetune.py` + training loop).
* `run.py` — the real thing: fine-tune GPT-2 small on the SMS Spam Collection (needs
  internet + `safetensors` GPT-2 weights; GPU recommended).

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab24_classify   # grade your work
LAB_IMPL=solution python -m pytest -q labs/lab24_classify   # reference
python labs/lab24_classify/run.py --gpt2-dir /path/to/gpt2 --data /path/to/SMSSpamCollection
```
