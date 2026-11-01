"""Lab 7 reference solution: a complete, instrumented pretraining loop.

Tasks
  1  cross_entropy_from_scratch / perplexity          (the training objective)
  2  warmup_cosine_lr                                 (LR schedule)
  3  global_grad_norm / clip_grad_norm_               (gradient clipping)
  4  generate_text                                    (temperature + top-k + EOS)
  5  save_checkpoint / load_checkpoint                (exact resume, incl. RNG state)
  6  enable_activation_checkpointing                  (trade compute for memory)
  7  train_model                                      (accumulation, AMP, clipping, ckpts)
  8  training_memory_estimate / chinchilla_optimal / mfu   (napkin math)
  9  load_openai_gpt2_params                          (weights in OpenAI's original layout)
"""
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =========================================================================== Task 1
def cross_entropy_from_scratch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood of the targets.

    logits: (B, T, V), targets: (B, T). Uses log-softmax (log-sum-exp trick) rather than
    log(softmax(.)) so that very negative logits cannot underflow to log(0) = -inf.
    """
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)               # (B, T, 1)
    log_probs = logits - lse                                           # log softmax
    target_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T)
    return -target_lp.mean()


def perplexity(loss: float) -> float:
    """exp(mean NLL): the effective number of equally likely next tokens."""
    return math.exp(loss)


def train_val_split(text: str, train_ratio: float = 0.9):
    split = int(train_ratio * len(text))
    return text[:split], text[split:]


# =========================================================================== Task 2
def warmup_cosine_lr(step: int, total_steps: int, peak_lr: float, warmup_steps: int,
                     initial_lr: float = 0.0, min_lr: float = 0.0) -> float:
    """Linear ramp initial_lr -> peak_lr over warmup_steps, then half-cosine to min_lr."""
    if step < warmup_steps:
        return initial_lr + (peak_lr - initial_lr) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# =========================================================================== Task 3
def global_grad_norm(parameters) -> float:
    """L2 norm of all gradients viewed as one long vector."""
    sq = 0.0
    for p in parameters:
        if p.grad is not None:
            sq += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(sq)


@torch.no_grad()
def clip_grad_norm_(parameters, max_norm: float, eps: float = 1e-6) -> float:
    """Rescale all gradients by min(1, max_norm / ||g||). Returns the pre-clip norm."""
    parameters = [p for p in parameters if p.grad is not None]
    total = global_grad_norm(parameters)
    scale = max_norm / (total + eps)
    if scale < 1.0:
        for p in parameters:
            p.grad.mul_(scale)
    return total


# =========================================================================== Task 4
@torch.no_grad()
def generate_text(model, idx: torch.Tensor, max_new_tokens: int, context_size: int,
                  temperature: float = 0.0, top_k: Optional[int] = None,
                  eos_id: Optional[int] = None, generator: Optional[torch.Generator] = None):
    """Autoregressive sampling with temperature scaling and top-k filtering.

    temperature == 0 -> greedy (argmax). Stops early if every row emits eos_id.
    """
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :].float()
        if top_k is not None:
            kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
        idx = torch.cat([idx, nxt], dim=1)
    return idx


# =========================================================================== Task 5
def save_checkpoint(path: str, model, optimizer, step: int, extra: Optional[dict] = None):
    """Everything needed for a *bit-exact* resume: weights, optimizer moments, step, RNG."""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str, model, optimizer=None) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ck["optimizer"])
    torch.set_rng_state(ck["rng_cpu"])
    if ck.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["rng_cuda"])
    return ck["step"]


# =========================================================================== Task 6
def enable_activation_checkpointing(model) -> int:
    """Wrap every transformer block so its internals are recomputed during backward.

    Only each block's *input* is kept alive between forward and backward; the attention
    scores, MLP hidden states, etc. are rebuilt on the fly. Returns #blocks wrapped.
    """
    n = 0
    for blk in model.trf_blocks:
        inner = blk.forward

        def wrapped(x, cache=None, pad_mask=None, _inner=inner, _blk=blk):
            if _blk.training and cache is None and torch.is_grad_enabled():
                return checkpoint(_inner, x, None, pad_mask, use_reentrant=False)
            return _inner(x, cache=cache, pad_mask=pad_mask)

        blk.forward = wrapped
        n += 1
    return n


# =========================================================================== Task 7
@dataclass
class TrainConfig:
    epochs: int = 1
    peak_lr: float = 4e-4
    initial_lr: float = 1e-5
    min_lr: float = 1e-5
    warmup_frac: float = 0.1
    grad_clip: Optional[float] = 1.0
    grad_accum: int = 1
    amp_dtype: Optional[torch.dtype] = None      # torch.bfloat16 for mixed precision
    eval_freq: int = 20
    eval_iter: int = 5
    ckpt_path: Optional[str] = None
    ckpt_every: int = 0
    device: str = "cpu"
    max_steps: Optional[int] = None              # stop early (optimizer steps)


@dataclass
class History:
    step: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    tokens_seen: List[int] = field(default_factory=list)
    tokens_per_sec: float = 0.0
    step_losses: List[float] = field(default_factory=list)


def _loss(model, x, y, amp_dtype, device):
    x, y = x.to(device), y.to(device)
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(dev_type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = model(x)
    return F.cross_entropy(logits.float().flatten(0, 1), y.flatten())


@torch.no_grad()
def evaluate(model, loader, cfg: TrainConfig, n: int):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
        losses.append(_loss(model, x, y, cfg.amp_dtype, cfg.device).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train_model(model, train_loader, val_loader, optimizer, cfg: TrainConfig,
                start_step: int = 0, log=print) -> History:
    """Pretraining loop with warmup+cosine, grad accumulation, AMP, clipping, checkpoints."""
    hist = History()
    steps_per_epoch = len(train_loader) // cfg.grad_accum
    total = cfg.max_steps or cfg.epochs * steps_per_epoch
    warmup = int(cfg.warmup_frac * total)
    step, micro, tokens = start_step, 0, 0
    t0 = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for _ in range(cfg.epochs):
        for x, y in train_loader:
            loss = _loss(model, x, y, cfg.amp_dtype, cfg.device) / cfg.grad_accum
            loss.backward()
            micro += 1
            tokens += x.numel()
            if micro % cfg.grad_accum:
                continue
            lr = warmup_cosine_lr(step, total, cfg.peak_lr, warmup, cfg.initial_lr, cfg.min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            gn = clip_grad_norm_(model.parameters(), cfg.grad_clip) if cfg.grad_clip \
                else global_grad_norm(model.parameters())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            hist.step_losses.append(loss.item() * cfg.grad_accum)
            step += 1
            if step % cfg.eval_freq == 0:
                tr = evaluate(model, train_loader, cfg, cfg.eval_iter)
                va = evaluate(model, val_loader, cfg, cfg.eval_iter)
                hist.step.append(step); hist.train_loss.append(tr); hist.val_loss.append(va)
                hist.lr.append(lr); hist.grad_norm.append(gn); hist.tokens_seen.append(tokens)
                log(f"step {step:5d} | train {tr:.3f} | val {va:.3f} | lr {lr:.2e} | "
                    f"|g| {gn:.2f}")
            if cfg.ckpt_path and cfg.ckpt_every and step % cfg.ckpt_every == 0:
                save_checkpoint(cfg.ckpt_path, model, optimizer, step)
            if cfg.max_steps and step >= cfg.max_steps:
                hist.tokens_per_sec = tokens / (time.perf_counter() - t0)
                return hist
    hist.tokens_per_sec = tokens / (time.perf_counter() - t0)
    return hist


# =========================================================================== Task 8
def training_memory_estimate(n_params: float, n_layers: int, d_model: int, n_heads: int,
                             seq: int, batch: int, act_checkpointing: bool = False,
                             flash: bool = False) -> Dict[str, float]:
    """Bytes for mixed-precision AdamW training (no sharding).

    model states: 2 (bf16 w) + 2 (bf16 g) + 4 (fp32 master) + 8 (Adam m, v) = 16 B/param
    activations : s*b*h*(34 + 5*a*s/h) bytes/layer (bf16, Korthikanti et al. 2022);
                  with full checkpointing only the block input (2*s*b*h bytes) per layer.
    """
    states = 16 * n_params
    s, b, h, a = seq, batch, d_model, n_heads
    if act_checkpointing:
        act = n_layers * 2 * s * b * h
    else:
        act = n_layers * s * b * h * (34 + (0 if flash else 5 * a * s / h))
    return dict(model_states=states, activations=act, total=states + act)


def chinchilla_optimal(compute_flops: float, tokens_per_param: float = 20.0):
    """C = 6 N D and D = k N  =>  N = sqrt(C / (6k)), D = k N."""
    n = math.sqrt(compute_flops / (6 * tokens_per_param))
    return n, tokens_per_param * n


def mfu(tokens_per_sec: float, n_params: float, peak_flops: float) -> float:
    """Model FLOPs utilisation using the 6N FLOPs/token training estimate."""
    return 6 * n_params * tokens_per_sec / peak_flops


# =========================================================================== Task 9
def load_openai_gpt2_params(model, params: dict):
    """Load OpenAI's original GPT-2 checkpoint layout (as numpy arrays) into GPTModel.

    params = {"wte": (V,d), "wpe": (T,d), "g": (d,), "b": (d,),
              "blocks": [ {"ln_1": {"g","b"}, "ln_2": {"g","b"},
                           "attn": {"c_attn": {"w": (d,3d), "b": (3d,)},
                                    "c_proj": {"w": (d,d),  "b": (d,)}},
                           "mlp":  {"c_fc":   {"w": (d,4d), "b": (4d,)},
                                    "c_proj": {"w": (4d,d), "b": (d,)}}}, ... ]}
    OpenAI stores projection matrices as (in, out), PyTorch Linear as (out, in): transpose.
    """
    def put(param, arr):
        t = torch.as_tensor(arr, dtype=param.dtype)
        if t.shape != param.shape:
            raise ValueError(f"shape mismatch {tuple(param.shape)} vs {tuple(t.shape)}")
        with torch.no_grad():
            param.copy_(t)

    put(model.tok_emb.weight, params["wte"])
    put(model.pos_emb.weight, params["wpe"])
    for blk, p in zip(model.trf_blocks, params["blocks"]):
        import numpy as np
        qw, kw, vw = np.split(np.asarray(p["attn"]["c_attn"]["w"]), 3, axis=-1)
        qb, kb, vb = np.split(np.asarray(p["attn"]["c_attn"]["b"]), 3, axis=-1)
        put(blk.att.W_query.weight, qw.T); put(blk.att.W_query.bias, qb)
        put(blk.att.W_key.weight, kw.T); put(blk.att.W_key.bias, kb)
        put(blk.att.W_value.weight, vw.T); put(blk.att.W_value.bias, vb)
        put(blk.att.out_proj.weight, np.asarray(p["attn"]["c_proj"]["w"]).T)
        put(blk.att.out_proj.bias, p["attn"]["c_proj"]["b"])
        put(blk.ff.layers[0].weight, np.asarray(p["mlp"]["c_fc"]["w"]).T)
        put(blk.ff.layers[0].bias, p["mlp"]["c_fc"]["b"])
        put(blk.ff.layers[2].weight, np.asarray(p["mlp"]["c_proj"]["w"]).T)
        put(blk.ff.layers[2].bias, p["mlp"]["c_proj"]["b"])
        put(blk.norm1.scale, p["ln_1"]["g"]); put(blk.norm1.shift, p["ln_1"]["b"])
        put(blk.norm2.scale, p["ln_2"]["g"]); put(blk.norm2.shift, p["ln_2"]["b"])
    put(model.final_norm.scale, params["g"])
    put(model.final_norm.shift, params["b"])
    put(model.out_head.weight, params["wte"])      # weight tying
    return model
