# Lab 42 — Instruction fine-tuning pipeline, packing, LoRA and QLoRA-style NF4

Book: Chapter 42, Section "Lab 42".

* `starter.py` — your implementation.
* `solution.py` — reference (re-exports `llmsys/finetune.py`, `llmsys/lora.py`).
* `run.py` — real run: instruction-tune GPT-2 medium (full or LoRA) on an Alpaca-style JSON
  file, write responses, score them with a local Ollama judge.

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab42_instruct_lora
LAB_IMPL=solution python -m pytest -q labs/lab42_instruct_lora
python labs/lab42_instruct_lora/run.py --data instruction-data.json --gpt2-dir gpt2-medium --lora-rank 16
python labs/lab42_instruct_lora/run.py --judge-only responses.json --judge-model llama3
```
