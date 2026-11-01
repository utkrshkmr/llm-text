"""A complete, compact PPO-RLHF trainer with four models.

    policy  pi_theta   trained   generates responses; the model we keep
    ref     pi_ref     frozen    the SFT model; anchors the policy via a per-token KL penalty
    reward  r_phi      frozen    scores a finished (prompt, response) with one scalar
    value   V_psi      trained   per-token baseline used by GAE

One call to ``PPOTrainer.step()`` performs the classic three-phase iteration:

    1. rollout  -- sample B responses from the current policy (KV-cached decoding)
    2. score    -- one forward pass each through policy (old log-probs), ref, value, reward;
                   build KL-shaped per-token rewards; run GAE; whiten advantages
    3. learn    -- ``ppo_epochs`` passes over shuffled minibatches, clipped policy loss for
                   the actor and clipped value loss for the critic

The loss functions are injected via ``fns`` (defaults to ``llmsys.rlhf.losses``) so a lab
can plug in its own implementations and watch the whole system train.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass, field
from types import ModuleType
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from ..kvcache import KVCache
from . import losses as default_fns
from .models import RewardModel, ValueModel
from .task import PreferencePair, pad_pairs


# --------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample_responses(policy, prompt_ids: torch.Tensor, max_new_tokens: int, eos_id: int,
                     pad_id: int, temperature: float = 1.0,
                     generator: Optional[torch.Generator] = None):
    """Batched ancestral sampling for equal-length prompts.

    Returns (responses (B, T) right-padded with ``pad_id``, mask (B, T) float, has_eos (B,)).
    The mask covers the generated tokens up to and including the first <eos>.
    """
    policy.eval()
    B = prompt_ids.shape[0]
    cache = KVCache(policy.n_layers)
    logits = policy(prompt_ids, cache=cache)[:, -1]
    done = torch.zeros(B, dtype=torch.bool)
    toks, masks = [], []
    for _ in range(max_new_tokens):
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        toks.append(nxt)
        masks.append((~done).float())
        done = done | (nxt == eos_id)
        if bool(done.all()):
            break
        logits = policy(nxt[:, None], cache=cache)[:, -1]
    resp = torch.stack(toks, 1)
    return resp, torch.stack(masks, 1), done


def response_logprobs(policy, full_ids: torch.Tensor, prompt_len: int, with_entropy: bool = False):
    """log pi(y_t | x, y_<t) for every response position of ``full_ids`` = [prompt | response]."""
    logits = policy(full_ids)[:, prompt_len - 1: -1].float()
    targets = full_ids[:, prompt_len:]
    logp = torch.log_softmax(logits, -1).gather(-1, targets[..., None]).squeeze(-1)
    if with_entropy:
        return logp, default_fns.entropy_from_logits(logits)
    return logp


# --------------------------------------------------------------------------- reward model training
def train_reward_model(rm: RewardModel, pairs: Sequence[PreferencePair], pad_id: int,
                       epochs: int = 3, batch_size: int = 32, lr: float = 1e-3,
                       val_pairs: Optional[Sequence[PreferencePair]] = None, seed: int = 0,
                       loss_fn: Optional[Callable] = None, log: Callable = lambda *_: None):
    """Fit r_phi to preference pairs with the Bradley-Terry loss. Returns a history dict."""
    loss_fn = loss_fn or default_fns.bt_loss
    rng = random.Random(seed)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.0)
    hist = {"loss": [], "train_acc": [], "val_acc": []}
    pairs = list(pairs)
    for ep in range(epochs):
        rm.train()
        rng.shuffle(pairs)
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i: i + batch_size]
            ids, lens = pad_pairs(batch, pad_id)
            r = rm(ids, lens)
            n = len(batch)
            loss = loss_fn(r[:n], r[n:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            opt.step()
            hist["loss"].append(loss.item())
        hist["train_acc"].append(evaluate_reward_model(rm, pairs[:512], pad_id))
        if val_pairs is not None:
            hist["val_acc"].append(evaluate_reward_model(rm, val_pairs, pad_id))
        log(f"rm epoch {ep+1}: loss {hist['loss'][-1]:.3f} train acc {hist['train_acc'][-1]:.3f}"
            + (f" val acc {hist['val_acc'][-1]:.3f}" if val_pairs is not None else ""))
    rm.eval()
    return hist


@torch.no_grad()
def evaluate_reward_model(rm: RewardModel, pairs: Sequence[PreferencePair], pad_id: int,
                          batch_size: int = 256) -> float:
    rm.eval()
    correct, total = 0, 0
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i: i + batch_size]
        ids, lens = pad_pairs(batch, pad_id)
        r = rm(ids, lens)
        n = len(batch)
        correct += int((r[:n] > r[n:]).sum())
        total += n
    return correct / max(total, 1)


# --------------------------------------------------------------------------- PPO
@dataclass
class PPOConfig:
    batch_size: int = 64              # prompts (= responses) per rollout
    max_new_tokens: int = 11
    temperature: float = 1.0
    ppo_epochs: int = 4
    minibatch_size: int = 16
    lr_policy: float = 1e-3
    lr_value: float = 3e-3
    clip: float = 0.2
    vf_clip: Optional[float] = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    init_beta: float = 0.05
    target_kl: Optional[float] = None  # set -> adaptive KL controller
    kl_horizon: int = 2000
    kl_kind: str = "k1"
    whiten_adv: bool = True
    ent_coef: float = 0.0
    max_grad_norm: float = 1.0
    missing_eos_score: Optional[float] = -2.0   # replace the RM score if no <eos>
    ptx_coef: float = 0.0                       # PPO-ptx: weight of a pretraining/SFT loss
    seed: int = 0


@dataclass
class Rollout:
    prompts: torch.Tensor        # (B, P)
    responses: torch.Tensor      # (B, T)
    mask: torch.Tensor           # (B, T)
    has_eos: torch.Tensor        # (B,)
    logp_old: torch.Tensor = None
    logp_ref: torch.Tensor = None
    values_old: torch.Tensor = None
    score: torch.Tensor = None   # RM score per response (after the missing-eos rule)
    rewards: torch.Tensor = None
    kl: torch.Tensor = None
    advantages: torch.Tensor = None
    returns: torch.Tensor = None

    @property
    def full(self):
        return torch.cat([self.prompts, self.responses], 1)


class PPOTrainer:
    def __init__(self, policy, ref, reward_model: RewardModel, value_model: ValueModel,
                 prompt_fn: Callable[[random.Random], List[int]], eos_id: int, pad_id: int,
                 cfg: PPOConfig = PPOConfig(), fns: ModuleType = default_fns,
                 true_score_fn: Optional[Callable] = None,
                 ptx_batch_fn: Optional[Callable] = None):
        self.policy, self.ref, self.rm, self.value = policy, ref, reward_model, value_model
        for m in (self.ref, self.rm):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        self.prompt_fn, self.eos_id, self.pad_id = prompt_fn, eos_id, pad_id
        self.cfg, self.fns = cfg, fns
        self.true_score_fn, self.ptx_batch_fn = true_score_fn, ptx_batch_fn
        self.rng = random.Random(cfg.seed)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.opt_pi = torch.optim.AdamW(policy.parameters(), lr=cfg.lr_policy, weight_decay=0.0)
        self.opt_v = torch.optim.AdamW(value_model.parameters(), lr=cfg.lr_value, weight_decay=0.0)
        self.kl_ctl = (default_fns.AdaptiveKLController(cfg.init_beta, cfg.target_kl, cfg.kl_horizon)
                       if cfg.target_kl else default_fns.FixedKLController(cfg.init_beta))
        self.iteration = 0

    # ---------------------------------------------------------------- phase 1
    def rollout(self) -> Rollout:
        prompts = torch.tensor([self.prompt_fn(self.rng) for _ in range(self.cfg.batch_size)])
        resp, mask, has_eos = sample_responses(self.policy, prompts, self.cfg.max_new_tokens,
                                               self.eos_id, self.pad_id, self.cfg.temperature,
                                               self.gen)
        return Rollout(prompts, resp, mask, has_eos)

    # ---------------------------------------------------------------- phase 2
    @torch.no_grad()
    def score(self, ro: Rollout) -> Rollout:
        P = ro.prompts.shape[1]
        full = ro.full
        self.policy.eval(); self.value.eval()
        ro.logp_old = response_logprobs(self.policy, full, P) * ro.mask
        ro.logp_ref = response_logprobs(self.ref, full, P) * ro.mask
        ro.values_old = self.value(full)[:, P - 1: -1] * ro.mask
        lengths = P + ro.mask.sum(1).long()
        score = self.rm(full, lengths)
        if self.cfg.missing_eos_score is not None:
            score = torch.where(ro.has_eos, score, torch.full_like(score, self.cfg.missing_eos_score))
        ro.score = score
        ro.rewards, ro.kl = self.fns.kl_shaped_rewards(score, ro.logp_old, ro.logp_ref, ro.mask,
                                                       self.kl_ctl.beta, self.cfg.kl_kind)
        adv, ret = self.fns.gae(ro.rewards, ro.values_old, ro.mask, self.cfg.gamma, self.cfg.lam)
        ro.returns = ret
        ro.advantages = self.fns.whiten(adv, ro.mask) if self.cfg.whiten_adv else adv
        return ro

    # ---------------------------------------------------------------- phase 3
    def learn(self, ro: Rollout) -> Dict[str, float]:
        cfg, P = self.cfg, ro.prompts.shape[1]
        full = ro.full
        B = full.shape[0]
        stats = {"policy_loss": [], "value_loss": [], "clipfrac": [], "approx_kl": [],
                 "entropy": [], "vf_clipfrac": []}
        self.policy.train(); self.value.train()
        for _ in range(cfg.ppo_epochs):
            idx = torch.randperm(B, generator=self.gen)
            for s in range(0, B, cfg.minibatch_size):
                mb = idx[s: s + cfg.minibatch_size]
                m = ro.mask[mb]
                logp, ent = response_logprobs(self.policy, full[mb], P, with_entropy=True)
                pl, st = self.fns.ppo_policy_loss(logp * m, ro.logp_old[mb], ro.advantages[mb], m, cfg.clip)
                ent_mean = default_fns.masked_mean(ent, m)
                loss_pi = pl - cfg.ent_coef * ent_mean
                if cfg.ptx_coef > 0 and self.ptx_batch_fn is not None:
                    ids, tgt = self.ptx_batch_fn(self.rng)
                    ptx = F.cross_entropy(self.policy(ids).flatten(0, 1), tgt.flatten(), ignore_index=-100)
                    loss_pi = loss_pi + cfg.ptx_coef * ptx
                self.opt_pi.zero_grad(set_to_none=True)
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt_pi.step()

                v = self.value(full[mb])[:, P - 1: -1] * m
                vl, vfrac = self.fns.value_loss(v, ro.values_old[mb], ro.returns[mb], m, cfg.vf_clip)
                self.opt_v.zero_grad(set_to_none=True)
                vl.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), cfg.max_grad_norm)
                self.opt_v.step()

                stats["policy_loss"].append(pl.item()); stats["value_loss"].append(vl.item())
                stats["clipfrac"].append(st["clipfrac"]); stats["approx_kl"].append(st["approx_kl"])
                stats["entropy"].append(ent_mean.item()); stats["vf_clipfrac"].append(vfrac)
        return {k: sum(v) / max(len(v), 1) for k, v in stats.items()}

    # ---------------------------------------------------------------- one iteration
    def step(self) -> Dict[str, float]:
        t0 = time.perf_counter()
        ro = self.rollout()
        t1 = time.perf_counter()
        ro = self.score(ro)
        t2 = time.perf_counter()
        learn_stats = self.learn(ro)
        t3 = time.perf_counter()
        kl_seq = float((ro.kl * ro.mask).sum(1).mean())
        beta = self.kl_ctl.beta
        self.kl_ctl.update(kl_seq, self.cfg.batch_size)
        self.iteration += 1
        out = {
            "iter": self.iteration,
            "rm_score": float(ro.score.mean()),
            "kl": kl_seq,
            "beta": beta,
            "total_reward": float(ro.rewards.sum(1).mean()),
            "resp_len": float(ro.mask.sum(1).mean()),
            "eos_rate": float(ro.has_eos.float().mean()),
            "explained_var": default_fns.explained_variance(ro.values_old, ro.returns, ro.mask),
            **learn_stats,
            "t_rollout": t1 - t0, "t_score": t2 - t1, "t_learn": t3 - t2,
        }
        if self.true_score_fn is not None:
            out["true_score"] = sum(self.true_score_fn(p, r) for p, r in
                                    zip(ro.prompts.tolist(), ro.responses.tolist())) / ro.prompts.shape[0]
        self.last_rollout = ro
        return out


def make_reference(policy):
    """Frozen copy of the SFT policy (pi_ref)."""
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref
