"""Offline preference optimisation: DPO and its family (Chapter 43).

Contents
--------
* data      : ``PreferencePair``, JSONL loading, tokenisation with a response mask,
              ``build_pair_batch`` (chosen rows first, then rejected rows).
* log-probs : ``sequence_logps`` (sum and token count over response tokens only).
* reference : ``precompute_reference_logps`` (cache once, train without a reference
              model on the device) and ``disable_lora`` (LoRA base weights *are* the
              reference model).
* losses    : DPO (+ conservative/label-smoothed cDPO), IPO, KTO, SimPO, ORPO, CPO.
* metrics   : implicit-reward accuracy, margins, chosen/rejected log-prob drift.
* trainer   : ``PrefOptConfig`` + ``train_preference`` (gradient accumulation, clipping,
              logging), usable with any model that maps ids (B, T) -> logits (B, T, V).
* toy task  : ``ToyPrefTask``: a tiny "reverse the digits, politely" task whose preferred
              behaviour can be measured exactly, for CPU demos and tests.
"""
from __future__ import annotations

import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# =========================================================================== data
@dataclass
class PreferencePair:
    """One preference judgement: the same prompt, a preferred and a dispreferred response.

    All fields are token-id lists. ``chosen``/``rejected`` should already end with EOS.
    """
    prompt: List[int]
    chosen: List[int]
    rejected: List[int]
    meta: dict = field(default_factory=dict)


def load_preference_jsonl(path: str) -> List[dict]:
    """Read ``{"prompt": str, "chosen": str, "rejected": str, ...}`` records (one per line)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if not all(k in rec for k in ("prompt", "chosen", "rejected")):
                    raise ValueError(f"record is missing keys: {sorted(rec)}")
                out.append(rec)
    return out


def tokenize_pair(rec: dict, encode: Callable[[str], List[int]], eos_id: int,
                  template: Callable[[str], str] = lambda p: p, max_prompt: int = 512,
                  max_response: int = 512) -> Optional[PreferencePair]:
    """Text record -> ``PreferencePair``. Prompts are truncated from the *left* (keep the
    end of the conversation), responses from the right. Returns None for degenerate pairs
    (identical responses carry no preference signal)."""
    if rec["chosen"] == rec["rejected"]:
        return None
    p = encode(template(rec["prompt"]))[-max_prompt:]
    c = encode(rec["chosen"])[: max_response - 1] + [eos_id]
    r = encode(rec["rejected"])[: max_response - 1] + [eos_id]
    return PreferencePair(p, c, r, {k: v for k, v in rec.items()
                                    if k not in ("prompt", "chosen", "rejected")})


def _pad_rows(rows: Sequence[Sequence[int]], masks: Sequence[Sequence[bool]], pad_id: int):
    T = max(len(r) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.bool)
    for i, (r, mk) in enumerate(zip(rows, masks)):
        ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        m[i, : len(r)] = torch.tensor(mk, dtype=torch.bool)
    return ids, m


def build_pair_batch(pairs: Sequence[PreferencePair], pad_id: int) -> Dict[str, torch.Tensor]:
    """Concatenate prompt+response for both sides into one right-padded (2B, T) batch.

    Rows 0..B-1 are prompt+chosen, rows B..2B-1 are prompt+rejected (same order), so one
    forward pass serves both halves. ``loss_mask[i, t]`` is True when token t of row i is a
    response token (these are the tokens whose log-probabilities we sum). Right padding is
    safe without an attention mask: under a causal mask, real tokens never see later pads.
    """
    rows, masks = [], []
    for side in ("chosen", "rejected"):
        for pr in pairs:
            resp = getattr(pr, side)
            rows.append(list(pr.prompt) + list(resp))
            masks.append([False] * len(pr.prompt) + [True] * len(resp))
    ids, loss_mask = _pad_rows(rows, masks, pad_id)
    return {"input_ids": ids, "loss_mask": loss_mask, "n_pairs": torch.tensor(len(pairs))}


# =========================================================================== log-probs
def sequence_logps(model, input_ids: torch.Tensor, loss_mask: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum of log p(token | prefix) over masked (response) tokens, and the token count.

    The logit at position t predicts token t+1, so the mask is shifted by one. Uses
    ``gather - logsumexp`` instead of materialising a full log-softmax tensor.
    """
    logits = model(input_ids)[:, :-1].float()
    targets = input_ids[:, 1:]
    mask = loss_mask[:, 1:].to(logits.dtype)
    tok = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    return (tok * mask).sum(-1), mask.sum(-1)


def split_pair_logps(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(2B,) -> (chosen (B,), rejected (B,))."""
    B = x.shape[0] // 2
    return x[:B], x[B:]


# =========================================================================== reference
@torch.no_grad()
def precompute_reference_logps(ref_model, pairs: Sequence[PreferencePair], pad_id: int,
                               batch_size: int = 16, with_kl: bool = False,
                               device="cpu") -> torch.Tensor:
    """Run the frozen reference once over the dataset and cache its log-probs.

    Returns a float32 tensor (N, 2) = [log pi_ref(y_w|x), log pi_ref(y_l|x)], or (N, 3)
    with a third column log pi_ref(y_{(i+1) mod N}^w | x_i) for KTO's mismatched-pair KL
    estimate. During training the reference model then never needs to be on the device.
    """
    ref_model.eval()
    cols = []
    N = len(pairs)
    for s in range(0, N, batch_size):
        chunk = list(pairs[s: s + batch_size])
        b = build_pair_batch(chunk, pad_id)
        lp, _ = sequence_logps(ref_model, b["input_ids"].to(device), b["loss_mask"].to(device))
        c, r = split_pair_logps(lp.float().cpu())
        block = [c, r]
        if with_kl:
            mism = [mismatched_pair(pairs, s + j) for j in range(len(chunk))]
            mb = build_pair_batch(mism, pad_id)
            mlp, _ = sequence_logps(ref_model, mb["input_ids"].to(device), mb["loss_mask"].to(device))
            block.append(split_pair_logps(mlp.float().cpu())[0])
        cols.append(torch.stack(block, dim=1))
    return torch.cat(cols, dim=0)


def mismatched_pair(pairs: Sequence[PreferencePair], i: int) -> PreferencePair:
    """Prompt i with the chosen response of example i+1: an (approximately) unrelated
    completion used by KTO to estimate KL(pi_theta || pi_ref) without sampling."""
    j = (i + 1) % len(pairs)
    return PreferencePair(pairs[i].prompt, pairs[j].chosen, pairs[j].chosen)


@contextmanager
def disable_lora(model):
    """Temporarily zero every LoRA adapter so ``model`` computes the *base* (reference)
    policy. With LoRA-DPO this replaces a second model copy entirely."""
    from .lora import LoRALinear
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    if any(m.merged for m in mods):
        raise RuntimeError("unmerge adapters before using them as a reference")
    saved = [m.scaling for m in mods]
    for m in mods:
        m.scaling = 0.0
    try:
        yield model
    finally:
        for m, s in zip(mods, saved):
            m.scaling = s


# =========================================================================== losses
def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float = 0.1, label_smoothing: float = 0.0):
    """Direct Preference Optimisation (Rafailov et al., 2023).

    h = beta * [(log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l))]
    loss = -(1-eps) log sigma(h) - eps log sigma(-h)    (eps>0: conservative DPO / cDPO)
    Returns per-pair losses and the implicit rewards beta*log(pi/ref) of both sides.
    """
    r_c = beta * (pol_c - ref_c)
    r_r = beta * (pol_r - ref_r)
    h = r_c - r_r
    loss = -(1 - label_smoothing) * F.logsigmoid(h) - label_smoothing * F.logsigmoid(-h)
    return loss, r_c.detach(), r_r.detach()


def ipo_loss(pol_c, pol_r, ref_c, ref_r, tau: float = 0.1):
    """IPO (Azar et al., 2023): regress the log-ratio margin onto 1/(2 tau).

    Unlike DPO the target is finite, so deterministic preferences cannot push the margin
    to infinity. (Many implementations feed *length-averaged* log-probs here.)
    """
    h = (pol_c - ref_c) - (pol_r - ref_r)
    return (h - 1.0 / (2.0 * tau)) ** 2, (pol_c - ref_c).detach(), (pol_r - ref_r).detach()


def kto_loss(pol, ref, desirable: torch.Tensor, kl_ref_point: torch.Tensor,
             beta: float = 0.1, lambda_d: float = 1.0, lambda_u: float = 1.0):
    """Kahneman-Tversky Optimisation (Ethayarajh et al., 2024) on *unpaired* examples.

    r = log pi(y|x) - log ref(y|x); z0 = detached, non-negative KL estimate (reference
    point). Desirable: v = lambda_d * sigma(beta (r - z0)); undesirable:
    v = lambda_u * sigma(beta (z0 - r)). loss = lambda_y - v.
    """
    r = pol - ref
    z0 = kl_ref_point.detach().clamp(min=0)
    v_d = lambda_d * torch.sigmoid(beta * (r - z0))
    v_u = lambda_u * torch.sigmoid(beta * (z0 - r))
    lam = torch.where(desirable, torch.full_like(r, lambda_d), torch.full_like(r, lambda_u))
    v = torch.where(desirable, v_d, v_u)
    return lam - v, (beta * r).detach()


def kto_kl_estimate(pol_mismatch, ref_mismatch) -> torch.Tensor:
    """Batch estimate of KL(pi || ref) from mismatched (prompt, completion) pairs."""
    return (pol_mismatch - ref_mismatch).mean().clamp(min=0).detach()


def simpo_loss(pol_c_sum, pol_r_sum, len_c, len_r, beta: float = 2.0, gamma: float = 0.5):
    """SimPO (Meng et al., 2024): reference-free, length-normalised reward + target margin.

    reward(y) = beta * mean_t log pi(y_t | ...);  loss = -log sigma(r_w - r_l - gamma).
    """
    r_c = beta * pol_c_sum / len_c.clamp(min=1)
    r_r = beta * pol_r_sum / len_r.clamp(min=1)
    return -F.logsigmoid(r_c - r_r - gamma), r_c.detach(), r_r.detach()


def _log1mexp(a: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(a)) for a < 0, numerically stable."""
    return torch.where(a > -math.log(2), torch.log(-torch.expm1(a)), torch.log1p(-torch.exp(a)))


def orpo_loss(pol_c_sum, pol_r_sum, len_c, len_r, lam: float = 0.1):
    """ORPO (Hong et al., 2024): SFT NLL on the chosen response plus an odds-ratio term.

    With p = exp(mean log-prob), odds(y) = p / (1 - p);
    loss = NLL_w - lam * log sigma(log odds(y_w) - log odds(y_l)). No reference model.
    """
    a_c = (pol_c_sum / len_c.clamp(min=1)).clamp(max=-1e-6)
    a_r = (pol_r_sum / len_r.clamp(min=1)).clamp(max=-1e-6)
    log_odds = (a_c - _log1mexp(a_c)) - (a_r - _log1mexp(a_r))
    nll = -a_c
    return nll - lam * F.logsigmoid(log_odds), a_c.detach(), a_r.detach()


def cpo_loss(pol_c_sum, pol_r_sum, len_c, beta: float = 0.1):
    """CPO (Xu et al., 2024): reference-free DPO term plus NLL on the preferred response."""
    pref = -F.logsigmoid(beta * (pol_c_sum - pol_r_sum))
    nll = -pol_c_sum / len_c.clamp(min=1)
    return pref + nll, (beta * pol_c_sum).detach(), (beta * pol_r_sum).detach()


# =========================================================================== metrics
def preference_metrics(r_c: torch.Tensor, r_r: torch.Tensor) -> Dict[str, float]:
    """Implicit-reward diagnostics; ties count as half-correct."""
    acc = (r_c > r_r).float().mean() + 0.5 * (r_c == r_r).float().mean()
    return {"reward_acc": float(acc), "margin": float((r_c - r_r).mean()),
            "reward_chosen": float(r_c.mean()), "reward_rejected": float(r_r.mean())}


# =========================================================================== trainer
@dataclass
class PrefOptConfig:
    loss: str = "dpo"            # dpo | ipo | kto | simpo | orpo | cpo
    beta: float = 0.1
    label_smoothing: float = 0.0
    ipo_tau: float = 0.1
    simpo_gamma: float = 0.5
    orpo_lambda: float = 0.1
    kto_lambda_d: float = 1.0
    kto_lambda_u: float = 1.0
    lr: float = 1e-3
    batch_size: int = 16         # pairs per micro-batch
    grad_accum: int = 1
    steps: int = 100             # optimizer steps
    max_grad_norm: float = 1.0
    seed: int = 0
    log_every: int = 10

    @property
    def needs_reference(self) -> bool:
        return self.loss in ("dpo", "ipo", "kto")


def preference_step_loss(model, pairs: Sequence[PreferencePair], ref: Optional[torch.Tensor],
                         cfg: PrefOptConfig, pad_id: int, device="cpu",
                         all_pairs: Optional[Sequence[PreferencePair]] = None,
                         idx: Optional[Sequence[int]] = None):
    """Loss and metrics for one micro-batch. ``ref`` is the cached (b, 2|3) slice."""
    b = build_pair_batch(pairs, pad_id)
    lp, n = sequence_logps(model, b["input_ids"].to(device), b["loss_mask"].to(device))
    pc, pr = split_pair_logps(lp)
    nc, nr = split_pair_logps(n)
    if cfg.needs_reference:
        ref = ref.to(device)
        rc, rr = ref[:, 0], ref[:, 1]
    if cfg.loss == "dpo":
        loss, a, c = dpo_loss(pc, pr, rc, rr, cfg.beta, cfg.label_smoothing)
    elif cfg.loss == "ipo":
        loss, a, c = ipo_loss(pc, pr, rc, rr, cfg.ipo_tau)
    elif cfg.loss == "simpo":
        loss, a, c = simpo_loss(pc, pr, nc, nr, cfg.beta, cfg.simpo_gamma)
    elif cfg.loss == "orpo":
        loss, a, c = orpo_loss(pc, pr, nc, nr, cfg.orpo_lambda)
    elif cfg.loss == "cpo":
        loss, a, c = cpo_loss(pc, pr, nc, cfg.beta)
    elif cfg.loss == "kto":
        assert ref.shape[1] == 3 and all_pairs is not None and idx is not None
        mism = [mismatched_pair(all_pairs, i) for i in idx]
        mb = build_pair_batch(mism, pad_id)
        mlp, _ = sequence_logps(model, mb["input_ids"].to(device), mb["loss_mask"].to(device))
        z0 = kto_kl_estimate(split_pair_logps(mlp)[0], ref[:, 2])
        pol = torch.cat([pc, pr]); rl = torch.cat([rc, rr])
        des = torch.cat([torch.ones_like(pc, dtype=torch.bool), torch.zeros_like(pr, dtype=torch.bool)])
        l, r = kto_loss(pol, rl, des, z0, cfg.beta, cfg.kto_lambda_d, cfg.kto_lambda_u)
        B = pc.shape[0]
        loss, a, c = l[:B] + l[B:], r[:B], r[B:]
    else:
        raise ValueError(cfg.loss)
    m = preference_metrics(a, c)
    m.update(logp_chosen=float(pc.detach().mean()), logp_rejected=float(pr.detach().mean()))
    return loss.mean(), m


def train_preference(model, pairs: Sequence[PreferencePair], cfg: PrefOptConfig, pad_id: int,
                     ref_logps: Optional[torch.Tensor] = None, optimizer=None, device="cpu",
                     log: Optional[Callable[[dict], None]] = None) -> List[dict]:
    """Offline preference optimisation over a fixed dataset with cached reference log-probs."""
    if cfg.needs_reference and ref_logps is None:
        raise ValueError(f"{cfg.loss} needs reference log-probs; run precompute_reference_logps")
    rng = random.Random(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optimizer or torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    hist, order, ptr = [], [], 0
    model.train()
    for step in range(cfg.steps):
        agg: Dict[str, float] = {}
        for _ in range(cfg.grad_accum):
            if ptr + cfg.batch_size > len(order):
                order = list(range(len(pairs))); rng.shuffle(order); ptr = 0
            idx = order[ptr: ptr + cfg.batch_size]; ptr += cfg.batch_size
            ref = ref_logps[idx] if ref_logps is not None else None
            loss, m = preference_step_loss(model, [pairs[i] for i in idx], ref, cfg, pad_id,
                                           device, all_pairs=pairs, idx=idx)
            (loss / cfg.grad_accum).backward()
            m["loss"] = float(loss.detach())
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v / cfg.grad_accum
        agg["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        opt.step(); opt.zero_grad(set_to_none=True)
        agg["step"] = step + 1
        hist.append(agg)
        if log is not None and (step + 1) % cfg.log_every == 0:
            log(agg)
    model.eval()
    return hist


# =========================================================================== toy task
class ToyPrefTask:
    """"Reverse the digits" with a style marker, built to make preferences measurable.

    Vocabulary: <pad> <eos> Q A > sure nope 0..9. A prompt is ``Q d1 .. dk >``; a response
    is ``STYLE dk .. d1 <eos>`` where STYLE is ``sure`` (preferred) or ``nope``. Pairs come
    in two kinds: *style* pairs (same correct digits, sure > nope) and *quality* pairs (both
    polite, correct reversal > corrupted digits). Everything we care about is exactly
    computable: P(first response token == sure) and greedy exact-match accuracy.
    """

    TOKENS = ["<pad>", "<eos>", "Q", "A", ">", "sure", "nope"] + [str(d) for d in range(10)]

    def __init__(self, min_len: int = 3, max_len: int = 5, seed: int = 0):
        self.stoi = {t: i for i, t in enumerate(self.TOKENS)}
        self.pad_id, self.eos_id = self.stoi["<pad>"], self.stoi["<eos>"]
        self.sure_id, self.nope_id = self.stoi["sure"], self.stoi["nope"]
        self.min_len, self.max_len = min_len, max_len
        self.rng = random.Random(seed)

    @property
    def vocab_size(self) -> int:
        return len(self.TOKENS)

    def decode(self, ids) -> str:
        return " ".join(self.TOKENS[int(i)] for i in ids)

    def _digits(self):
        k = self.rng.randint(self.min_len, self.max_len)
        return [self.rng.randrange(10) for _ in range(k)]

    def prompt(self, digits) -> List[int]:
        return [self.stoi["Q"]] + [self.stoi[str(d)] for d in digits] + [self.stoi[">"]]

    def response(self, digits, polite: bool = True, corrupt: bool = False) -> List[int]:
        ds = list(reversed(digits))
        if corrupt:
            j = self.rng.randrange(len(ds))
            ds[j] = (ds[j] + self.rng.randint(1, 9)) % 10
        style = self.sure_id if polite else self.nope_id
        return [style] + [self.stoi[str(d)] for d in ds] + [self.eos_id]

    def sft_sequences(self, n: int, p_polite: float = 0.25) -> List[Tuple[List[int], List[int]]]:
        out = []
        for _ in range(n):
            d = self._digits()
            out.append((self.prompt(d), self.response(d, polite=self.rng.random() < p_polite)))
        return out

    def pairs(self, n: int, frac_style: float = 0.7) -> List[PreferencePair]:
        out = []
        for _ in range(n):
            d = self._digits()
            if self.rng.random() < frac_style:
                out.append(PreferencePair(self.prompt(d), self.response(d, True),
                                          self.response(d, False), {"kind": "style"}))
            else:
                out.append(PreferencePair(self.prompt(d), self.response(d, True),
                                          self.response(d, True, corrupt=True), {"kind": "quality"}))
        return out

    def model_config(self, emb_dim: int = 64, n_layers: int = 2) -> dict:
        return dict(vocab_size=self.vocab_size, context_length=32, emb_dim=emb_dim,
                    n_heads=4, n_layers=n_layers, hidden_dim=2 * emb_dim, head_dim=16,
                    qk_norm=True, n_kv_groups=2, rope_base=10_000.0, tie_embeddings=False)

    @torch.no_grad()
    def p_polite(self, model, n: int = 200, seed: int = 123) -> float:
        """Average exact probability that the first response token is ``sure``."""
        rng, saved = random.Random(seed), self.rng
        self.rng = rng
        prompts = [self.prompt(self._digits()) for _ in range(n)]
        self.rng = saved
        tot = 0.0
        for p in prompts:
            logits = model(torch.tensor([p]))[0, -1].float()
            tot += float(torch.softmax(logits, -1)[self.sure_id])
        return tot / n

    @torch.no_grad()
    def greedy_accuracy(self, model, n: int = 100, seed: int = 321) -> float:
        """Fraction of prompts whose greedy response has the correctly reversed digits
        (style ignored)."""
        from .generate import generate
        rng, saved = random.Random(seed), self.rng
        self.rng = rng
        ok = 0
        for _ in range(n):
            d = self._digits()
            p = torch.tensor([self.prompt(d)])
            out = generate(model, p, len(d) + 2, eos_id=self.eos_id)[0, p.shape[1]:].tolist()
            want = [self.stoi[str(x)] for x in reversed(d)]
            ok += int(out[1: 1 + len(want)] == want)
        self.rng = saved
        return ok / n


def sft_train(model, seqs: Sequence[Tuple[List[int], List[int]]], pad_id: int, steps: int = 300,
              batch_size: int = 64, lr: float = 3e-3, seed: int = 0) -> List[float]:
    """Plain response-only SFT used to build the toy reference policy."""
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    model.train()
    for _ in range(steps):
        batch = rng.sample(list(seqs), batch_size)
        rows = [p + r for p, r in batch]
        masks = [[False] * len(p) + [True] * len(r) for p, r in batch]
        ids, m = _pad_rows(rows, masks, pad_id)
        lp, n = sequence_logps(model, ids, m)
        loss = -(lp.sum() / n.sum())
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    model.eval()
    return losses


# =========================================================================== on-policy data
@torch.no_grad()
def build_onpolicy_pairs(model, prompts: Sequence[List[int]], score_fn: Callable[[List[int], List[int]], float],
                         pad_id: int, eos_id: int, k: int = 4, max_new_tokens: int = 64,
                         temperature: float = 1.0, min_gap: float = 0.0,
                         generator: Optional[torch.Generator] = None) -> List[PreferencePair]:
    """Sample k responses per prompt from ``model`` (batched, left-padded, KV-cached), score
    them, and keep (best, worst) when the score gap exceeds ``min_gap``.

    ``score_fn(prompt_ids, response_ids)`` can be a verifier, a reward model, or a judge.
    This is one round of *online / iterative DPO* data collection.
    """
    from .generate import generate_batch
    flat = [list(p) for p in prompts for _ in range(k)]
    outs = generate_batch(model, flat, max_new_tokens, pad_id=pad_id, eos_id=eos_id,
                          temperature=temperature, generator=generator)
    pairs = []
    for i, p in enumerate(prompts):
        cands = []
        for j in range(k):
            resp = outs[i * k + j]
            if eos_id in resp:
                resp = resp[: resp.index(eos_id) + 1]
            cands.append((score_fn(list(p), resp), resp))
        cands.sort(key=lambda t: t[0])
        (s_lo, worst), (s_hi, best) = cands[0], cands[-1]
        if s_hi - s_lo > min_gap and best != worst:
            pairs.append(PreferencePair(list(p), best, worst, {"gap": s_hi - s_lo}))
    return pairs


@torch.no_grad()
def token_logratios(policy, ref, prompt: Sequence[int], response: Sequence[int]) -> torch.Tensor:
    """Per-token log pi(y_t|...) - log ref(y_t|...) over the response: the token-level
    decomposition of DPO's implicit reward (multiply by beta to get reward units)."""
    ids = torch.tensor([list(prompt) + list(response)])
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, len(prompt):] = True

    def per_token(m):
        logits = m(ids)[:, :-1].float()
        tgt = ids[:, 1:]
        return (logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1))[0]

    d = per_token(policy) - per_token(ref)
    return d[len(prompt) - 1:]
