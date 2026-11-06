# Brief: additions to Chapter 20 "Scaling Laws in Practice"

Read first: BRIEF_conventions.md (all, esp. §7), the existing chapter book/chapters/ch20_scaling_laws.md (keep all
content), code/llmsys/scaling.py, code/labs/lab20_scaling_laws/*.

## DeepSeek contributions (owner request, 2026-11-05)
- DeepSeek LLM (2024) scaling study: measuring model scale as non-embedding FLOPs per token (M) rather than
  parameters, and why that changes the compute-optimal allocation; power-law fits for the optimal batch size and
  learning rate as functions of compute; how data quality shifts the optimal model/data split. Fit these laws in
  the lab on our small sweeps and compare the exponents.
- Research frontier section and Reading-the-research box (§7) if missing.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Scaling-law variants: inference-aware scaling (training beyond Chinchilla when inference volume is large;
  overtraining), data-constrained scaling (repeated epochs and their diminishing value), precision-aware scaling laws
  (training/inference precision as a variable), MoE scaling laws (granularity, active vs total parameters); fit at
  least the data-constrained law on the lab's sweeps.
