# Lab 36: Direct preference optimisation and its family (Chapter 36)

You build offline preference optimisation from the data path up:

| Task | Function(s) in `starter.py` |
|---|---|
| 1 | `build_pair_batch` (chosen rows then rejected rows, response mask) |
| 2 | `sequence_logps` (shifted, masked sum of token log-probs + counts) |
| 3 | `dpo_loss` (with conservative label smoothing) |
| 4 | `ipo_loss`, `kto_loss`, `simpo_loss`, `orpo_loss` |
| 5 | `precompute_reference_logps`, `preference_metrics` |
| 6 | `dpo_train_loop` (cached reference, accumulation, clipping, metrics) |

The toy task (`llmsys.prefopt.ToyPrefTask`) asks a tiny model to reverse digits and to start
with a style token. The reference policy says `sure` only about 25% of the time; preference pairs
prefer `sure` over `nope` (style pairs) and correct over corrupted digits (quality pairs). Because
everything is tiny, the preferred behaviour, P(first token = `sure`), can be measured exactly.

```
cd code
LAB_IMPL=starter  python -m pytest labs/lab36_dpo    # your implementation
LAB_IMPL=solution python -m pytest labs/lab36_dpo    # reference
python labs/lab36_dpo/run.py --out results           # SFT -> DPO/IPO/KTO/SimPO/ORPO/CPO/LoRA-DPO (CPU, minutes)
```

`run.py` writes `summary.json` and `curves.json` and prints pgfplots coordinates. On a GPU with a
real model, swap `ToyPrefTask` for `llmsys.prefopt.load_preference_jsonl` + `tokenize_pair` with your
tokenizer and chat template, and keep everything else the same.
