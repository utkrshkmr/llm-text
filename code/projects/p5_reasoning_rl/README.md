# Project 5: Reasoning RL and Alignment (Chapter 36)

Infrastructure for the capstone project. Your own implementation goes in `p5/`, which you
create next to these files; your tests go in `tests/`.

```
code/projects/p5_reasoning_rl/
  env_check.py      pre-flight check: Python, torch/transformers/vLLM/flash-attn/wandb/math-verify, GPUs, disk
  run_matrix.py     expand {"fixed":..., "vary":...} into one command line per run
  seeds_ci.py       aggregate per-seed JSONL logs into mean / 95% interval / min / max per step
  test_p5_infra.py  tests for the three tools (CPU, no network)
```

Quick start:
```
pip install -r code/requirements.txt
pip install "transformers>=4.50" vllm wandb math-verify      # pin exact versions
python code/projects/p5_reasoning_rl/env_check.py --need-gpus 2 --min-gpu-gb 40
huggingface-cli download allenai/OLMo-2-0425-1B --local-dir ckpt/olmo2-1b
python -m pytest -q code/projects/p5_reasoning_rl
```
Example grid for `run_matrix.py`:
```json
{"script": "code/projects/p5_reasoning_rl/p5/train_rl.py",
 "fixed": {"lr": 1e-5, "group_size": 8, "steps": 200},
 "vary": {"normalizer": ["std", "none", "mean"], "seed": [0, 1, 2, 3]}}
```
The tools only expand configurations, check the environment and aggregate logs. The project's
algorithms are yours to implement; Chapter 24 and Lab 24 teach the building blocks.
