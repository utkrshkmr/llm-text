"""RLHF with PPO (Chapter 25): reward models, value models, GAE, the PPO-RLHF trainer.

    task    -- a synthetic preference world with hidden "human" judgements
    models  -- RewardModel (sequence scalar), ValueModel (per-token scalar)
    losses  -- Bradley-Terry, K-way ranking, KL-shaped rewards, GAE, PPO policy/value losses,
               KL controllers
    ppo     -- sampling, reward-model training, PPOConfig, PPOTrainer
    plan    -- memory and FLOP planning for the four-model system
"""
from . import losses, models, plan, ppo, task  # noqa: F401
