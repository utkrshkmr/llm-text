# Lab 38 — Making GRPO robust (Chapter 38)

Small, testable building blocks for diagnosing and stabilising GRPO, plus an ablation
driver that runs every variant discussed in the chapter on the toy addition task.

* diagnostics: `token_entropy`, `sequence_logprob_and_entropy`, `chunked_logprob_entropy`, `advantage_stats`
* clipping: `policy_ratios`, `clipped_policy_loss`
* KL: `kl_k1`, `kl_k2`, `kl_k3`, `kl_sequence_surrogate`, `kl_importance_weighted`
* rewards: `reward_format`, `total_reward`
* descendants: `dr_grpo_advantages`, `aggregate_token_loss`, `zero_variance_keep`, `expected_wasted_fraction`

```bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab38_grpo_plus
LAB_IMPL=solution python -m pytest -q labs/lab38_grpo_plus

python labs/lab38_grpo_plus/run.py --list
python labs/lab38_grpo_plus/run.py --variants baseline,noclip_reuse4,clip_reuse4 \
       --seeds 0,1,2 --steps 150 --sft-cache sft.pt --out out18
```
Each run writes `out18/<variant>_s<seed>.csv` (one row per step) and appends to
`out18/summary.json`. Plot with `llmsys.rl.metrics.plot_metrics`.
