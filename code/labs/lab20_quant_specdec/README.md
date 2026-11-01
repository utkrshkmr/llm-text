# Lab 20 — Cheaper inference: quantization, speculative decoding, and a chat UI

Goal: implement weight-only int8/int4 quantization and lossless speculative decoding,
measure what they buy you, and serve a model through a streaming chat interface.

| File | What |
|---|---|
| `starter.py` | your work: fill in every `NotImplementedError` |
| `solution.py` | reference implementation |
| `test_lab20.py` | CPU tests (tiny random models, no downloads) |
| `bench.py` | `toy` (CPU acceptance statistics) / `spec` / `quant` (real Qwen3 checkpoints) |
| `chat_common.py`, `chat_single_turn.py`, `chat_multi_turn.py` | Chainlit apps |

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab20_quant_specdec   # grade your work
LAB_IMPL=solution python -m pytest -q labs/lab20_quant_specdec   # reference
python labs/lab20_quant_specdec/bench.py toy
pip install chainlit
QWEN3_DIR=/path/to/qwen3-0.6b chainlit run labs/lab20_quant_specdec/chat_multi_turn.py
```
The chat apps import `solution.py`; point them at `starter.py` once your version passes.
