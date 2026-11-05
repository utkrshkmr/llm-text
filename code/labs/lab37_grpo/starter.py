"""Lab 37 -- GRPO from scratch (starter -- fill in the TODOs).

Build every stage of RL with verifiable rewards by hand:
    1. sample_response      -- one rollout with a KV cache, temperature + top-p
    2. reward_rlvr          -- binary correctness reward (answer must be boxed)
    3. reward_partial       -- graded format-aware reward (Exercise 37.1)
    4. compute_advantages   -- group-relative advantages
    5. sequence_logprob     -- log pi(y|x) for one rollout, differentiable
    6. policy_gradient_loss -- -(A * log pi).mean()
    7. compute_grpo_loss    -- stages 1-5 for one prompt
    8. train_rlvr_grpo      -- the outer training loop with logging + checkpoints
    9. time_breakdown       -- where does a GRPO step spend its time?
"""
from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional

import torch

from llmsys.generate import sample_next_token
from llmsys.kvcache import KVCache
from llmsys.rl.rewards import extract_boxed, extract_last_number, normalize_answer


# ----------------------------------------------------------------------------- 1
@torch.no_grad()
def sample_response(model, tokenizer, prompt: str, max_new_tokens: int = 32,
                    temperature: float = 1.0, top_p: float = 1.0,
                    generator: Optional[torch.Generator] = None):
    """Return (full_ids 1-D LongTensor, prompt_len, completion_text).

    Prefill the prompt once into a KV cache, then decode one token at a time,
    sampling with temperature/top-p. Stop after EOS (EOS is kept in the ids).
    """
    raise NotImplementedError("TODO: prefill the prompt into a KVCache, then sample one token per step with sample_next_token; stop after EOS")


# ----------------------------------------------------------------------------- 2, 3
def reward_rlvr(text: str, truth: str) -> float:
    """1.0 iff the last \\boxed{...} in ``text`` matches ``truth`` (after normalisation)."""
    raise NotImplementedError("TODO: extract the last boxed answer and compare normalised strings")


def reward_partial(text: str, truth: str) -> float:
    """1.0 boxed & correct; 0.5 correct but not boxed (last number); 0.0 otherwise."""
    raise NotImplementedError("TODO: 1.0 boxed+correct, 0.5 correct last number without box, else 0.0")


# ----------------------------------------------------------------------------- 4
def compute_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """(r - mean) / (std + eps) over one group; std is the unbiased sample std."""
    raise NotImplementedError("TODO: (r - mean) / (std + eps); return zeros for groups of size 1")


# ----------------------------------------------------------------------------- 5
def sequence_logprob(model, token_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Sum of log p(y_t | y_<t, x) over the completion tokens. Keeps the graph."""
    raise NotImplementedError("TODO: log_softmax over logits, gather next-token logprobs, sum from prompt_len-1")


# ----------------------------------------------------------------------------- 6
def policy_gradient_loss(advantages: torch.Tensor, logps: torch.Tensor) -> torch.Tensor:
    """Negative advantage-weighted sequence log-likelihood, averaged over rollouts."""
    raise NotImplementedError("TODO: negate the mean of detached advantages times logps")


# ----------------------------------------------------------------------------- 7
def compute_grpo_loss(model, tokenizer, example: Dict, render_prompt, num_rollouts: int = 4,
                      max_new_tokens: int = 16, temperature: float = 1.0, top_p: float = 1.0,
                      generator: Optional[torch.Generator] = None) -> Dict:
    raise NotImplementedError("TODO: sample num_rollouts responses, reward them, compute advantages, logprobs and the loss")


# ----------------------------------------------------------------------------- 8
def train_rlvr_grpo(model, tokenizer, data: List[Dict], render_prompt, steps: int = 50,
                    num_rollouts: int = 4, max_new_tokens: int = 16, temperature: float = 1.0,
                    top_p: float = 1.0, lr: float = 3e-4, grad_clip: float = 1.0,
                    checkpoint_every: int = 0, checkpoint_dir: str = "checkpoints",
                    csv_log_path: Optional[str] = None, seed: int = 0,
                    log_every: int = 10) -> List[Dict]:
    """Standard loop: zero_grad -> GRPO loss -> backward -> clip -> step -> log -> ckpt."""
    raise NotImplementedError("TODO: loop over steps - zero_grad, compute_grpo_loss, backward, clip, step, log, checkpoint")


# ----------------------------------------------------------------------------- 9
def time_breakdown(model, tokenizer, example: Dict, render_prompt, num_rollouts: int = 4,
                   max_new_tokens: int = 16, generator=None) -> Dict[str, float]:
    """Seconds spent in (a) sampling, (b) differentiable log-prob pass, (c) backward."""
    raise NotImplementedError("TODO: time sampling, the differentiable logprob pass and backward separately")
