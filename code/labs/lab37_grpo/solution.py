"""Lab 37 -- GRPO from scratch (reference solution).

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
    device = next(model.parameters()).device
    prompt_ids = torch.tensor(tokenizer.encode(prompt), device=device)
    cache = KVCache(model.n_layers)
    logits = model(prompt_ids[None], cache=cache)[:, -1]
    generated: List[int] = []
    for _ in range(max_new_tokens):
        nxt = sample_next_token(logits, temperature, None, top_p, generator)
        tid = int(nxt)
        generated.append(tid)
        if tid == tokenizer.eos_token_id:
            break
        logits = model(nxt, cache=cache)[:, -1]
    full = torch.cat([prompt_ids, torch.tensor(generated, dtype=torch.long, device=device)])
    return full, prompt_ids.numel(), tokenizer.decode(generated)


# ----------------------------------------------------------------------------- 2, 3
def reward_rlvr(text: str, truth: str) -> float:
    """1.0 iff the last \\boxed{...} in ``text`` matches ``truth`` (after normalisation)."""
    pred = extract_boxed(text)
    if pred is None:
        return 0.0
    return float(normalize_answer(pred) == normalize_answer(truth))


def reward_partial(text: str, truth: str) -> float:
    """1.0 boxed & correct; 0.5 correct but not boxed (last number); 0.0 otherwise."""
    if reward_rlvr(text, truth) == 1.0:
        return 1.0
    num = extract_last_number(text)
    return 0.5 if num is not None and normalize_answer(num) == normalize_answer(truth) else 0.0


# ----------------------------------------------------------------------------- 4
def compute_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """(r - mean) / (std + eps) over one group; std is the unbiased sample std."""
    rewards = rewards.float()
    if rewards.numel() < 2:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std() + eps)


# ----------------------------------------------------------------------------- 5
def sequence_logprob(model, token_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Sum of log p(y_t | y_<t, x) over the completion tokens. Keeps the graph."""
    logits = model(token_ids[None])[0].float()           # (T, V)
    logprobs = torch.log_softmax(logits, dim=-1)
    picked = logprobs[:-1].gather(1, token_ids[1:, None]).squeeze(-1)   # (T-1,)
    return picked[prompt_len - 1:].sum()


# ----------------------------------------------------------------------------- 6
def policy_gradient_loss(advantages: torch.Tensor, logps: torch.Tensor) -> torch.Tensor:
    """Negative advantage-weighted sequence log-likelihood, averaged over rollouts."""
    return -(advantages.detach() * logps).mean()


# ----------------------------------------------------------------------------- 7
def compute_grpo_loss(model, tokenizer, example: Dict, render_prompt, num_rollouts: int = 4,
                      max_new_tokens: int = 16, temperature: float = 1.0, top_p: float = 1.0,
                      generator: Optional[torch.Generator] = None) -> Dict:
    assert num_rollouts >= 2, "GRPO needs a group to compare against"
    prompt = render_prompt(example)
    was_training = model.training
    model.eval()
    logps, rewards, samples = [], [], []
    for _ in range(num_rollouts):
        ids, plen, text = sample_response(model, tokenizer, prompt, max_new_tokens,
                                          temperature, top_p, generator)
        r = reward_rlvr(text, example["answer"])
        logps.append(sequence_logprob(model, ids, plen))   # differentiable
        rewards.append(r)
        samples.append({"text": text, "reward": r, "gen_len": ids.numel() - plen})
    if was_training:
        model.train()
    rew = torch.tensor(rewards)
    adv = compute_advantages(rew)
    loss = policy_gradient_loss(adv, torch.stack(logps))
    return {"loss": loss.item(), "loss_tensor": loss, "rewards": rewards,
            "advantages": adv.tolist(), "samples": samples}


# ----------------------------------------------------------------------------- 8
def train_rlvr_grpo(model, tokenizer, data: List[Dict], render_prompt, steps: int = 50,
                    num_rollouts: int = 4, max_new_tokens: int = 16, temperature: float = 1.0,
                    top_p: float = 1.0, lr: float = 3e-4, grad_clip: float = 1.0,
                    checkpoint_every: int = 0, checkpoint_dir: str = "checkpoints",
                    csv_log_path: Optional[str] = None, seed: int = 0,
                    log_every: int = 10) -> List[Dict]:
    """Standard loop: zero_grad -> GRPO loss -> backward -> clip -> step -> log -> ckpt."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = []
    if csv_log_path and os.path.exists(csv_log_path):
        os.remove(csv_log_path)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        ex = data[step % len(data)]
        stats = compute_grpo_loss(model, tokenizer, ex, render_prompt, num_rollouts,
                                  max_new_tokens, temperature, top_p, gen)
        stats["loss_tensor"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        row = {"step": step + 1, "loss": stats["loss"],
               "reward_avg": sum(stats["rewards"]) / len(stats["rewards"]),
               "avg_response_len": sum(s["gen_len"] for s in stats["samples"]) / len(stats["samples"])}
        history.append(row)
        if csv_log_path:
            new = not os.path.exists(csv_log_path)
            with open(csv_log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"grpo-step{step+1:05d}.pt"))
        if log_every and (step + 1) % log_every == 0:
            print(f"[step {step+1}/{steps}] loss={row['loss']:.4f} "
                  f"reward_avg={row['reward_avg']:.3f} len={row['avg_response_len']:.1f}")
    return history


# ----------------------------------------------------------------------------- 9
def time_breakdown(model, tokenizer, example: Dict, render_prompt, num_rollouts: int = 4,
                   max_new_tokens: int = 16, generator=None) -> Dict[str, float]:
    """Seconds spent in (a) sampling, (b) differentiable log-prob pass, (c) backward."""
    prompt = render_prompt(example)
    t0 = time.perf_counter()
    rolls = [sample_response(model, tokenizer, prompt, max_new_tokens, 1.0, 1.0, generator)
             for _ in range(num_rollouts)]
    t1 = time.perf_counter()
    logps = torch.stack([sequence_logprob(model, ids, pl) for ids, pl, _ in rolls])
    t2 = time.perf_counter()
    (-logps.mean()).backward()
    t3 = time.perf_counter()
    model.zero_grad(set_to_none=True)
    return {"sample": t1 - t0, "logprob": t2 - t1, "backward": t3 - t2}
