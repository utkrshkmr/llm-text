# Lab 25 — PPO-RLHF from scratch

Chapter 25 (RLHF: reward models, value functions and PPO).

You implement the mathematical core of RLHF in `starter.py`:

1. `masked_mean` and `whiten` (statistics over real tokens only)
2. `bt_loss` and `rm_accuracy` (the Bradley–Terry reward-model loss)
3. `ranking_loss` (the grouped K-way loss)
4. `kl_shaped_rewards` (per-token KL tolls plus the score on the last token)
5. `gae` (generalised advantage estimation as a masked backward scan)
6. `ppo_policy_loss` (the clipped surrogate, with clipfrac and approx_kl)
7. `value_loss` (the clipped critic loss)
8. `explained_variance` and `AdaptiveKLController.update`

The provided four-model trainer (`llmsys.rlhf.ppo.PPOTrainer`) receives your module as
`fns`, so `train_pipeline()` runs SFT, preference sampling and labelling, reward-model
training and PPO **on your code**.

```bash
cd code
LAB_IMPL=starter  python -m pytest labs/lab25_rlhf_ppo -q    # grade your work
LAB_IMPL=solution python -m pytest labs/lab25_rlhf_ppo -q    # reference
python labs/lab25_rlhf_ppo/run.py all --iters 70 --betas 0.2,0.05,0          # beta sweep + plots
python labs/lab25_rlhf_ppo/run.py lam --lams 0,0.95,1 --iters 40             # GAE ablation
python labs/lab25_rlhf_ppo/run.py all --impl starter                         # your implementation
```

The task is a synthetic preference world (`llmsys/rlhf/task.py`) with a hidden "human"
scoring function. The reward model never sees responses that repeat words, so with a weak
KL penalty PPO eventually learns to repeat positive words: the proxy reward keeps rising
while the true score falls. That is reward over-optimisation, reproduced on a laptop.

Everything runs on CPU. The test suite takes about 10 s of CPU time; a 70-iteration sweep of
three β values takes a few minutes.
