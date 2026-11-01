"""Reinforcement learning with verifiable rewards (Chapters 17-18).

    rewards     -- boxed-answer extraction, correctness/format rewards, GDPO combine
    advantages  -- group-relative advantages, zero-variance detection
    logprobs    -- token/sequence log-probs and entropies of completions
    losses      -- REINFORCE, PPO-clip (seq/token), KL estimators, GSPO, CISPO, TIS
    rollout     -- sampling single and batched rollouts
    trainer     -- GRPOConfig / GRPOTrainer / train_grpo / evaluate_accuracy
    toy         -- a single-digit-addition task + SFT warm start that learns on CPU
    metrics     -- CSV logging, moving averages, plotting
"""
from .advantages import advantage_stats, group_advantages, prob_zero_variance, zero_variance_mask
from .losses import (aggregate, cispo_loss, gspo_loss, kl_estimate, offpolicy_sequence_mask,
                     reinforce_seq_loss, sequence_clipped_loss, token_clipped_loss,
                     truncated_is_weights)
from .rewards import (extract_boxed, normalize_answer, reward_correct, reward_partial_format,
                      reward_think_format, think_format_ok)
from .trainer import GRPOConfig, GRPOTrainer, evaluate_accuracy, train_grpo
