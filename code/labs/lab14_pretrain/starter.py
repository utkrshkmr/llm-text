"""Lab 14 STARTER (fill in the TODOs): a complete, instrumented pretraining loop.

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
    raise NotImplementedError("TODO: cross_entropy_from_scratch")


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
    raise NotImplementedError("TODO: warmup_cosine_lr")


# =========================================================================== Task 3
def global_grad_norm(parameters) -> float:
    """L2 norm of all gradients viewed as one long vector."""
    raise NotImplementedError("TODO: global_grad_norm")


@torch.no_grad()
def clip_grad_norm_(parameters, max_norm: float, eps: float = 1e-6) -> float:
    """Rescale all gradients by min(1, max_norm / ||g||). Returns the pre-clip norm."""
    raise NotImplementedError("TODO: clip_grad_norm_")


# =========================================================================== Task 4
@torch.no_grad()
def generate_text(model, idx: torch.Tensor, max_new_tokens: int, context_size: int,
                  temperature: float = 0.0, top_k: Optional[int] = None,
                  eos_id: Optional[int] = None, generator: Optional[torch.Generator] = None):
    """Autoregressive sampling with temperature scaling and top-k filtering.

    temperature == 0 -> greedy (argmax). Stops early if every row emits eos_id.
    """
    raise NotImplementedError("TODO: generate_text")


# =========================================================================== Task 5
def save_checkpoint(path: str, model, optimizer, step: int, extra: Optional[dict] = None):
    """Everything needed for a *bit-exact* resume: weights, optimizer moments, step, RNG."""
    raise NotImplementedError("TODO: save_checkpoint")


def load_checkpoint(path: str, model, optimizer=None) -> int:
    raise NotImplementedError("TODO: load_checkpoint")


# =========================================================================== Task 6
def enable_activation_checkpointing(model) -> int:
    """Wrap every transformer block so its internals are recomputed during backward.

    Only each block's *input* is kept alive between forward and backward; the attention
    scores, MLP hidden states, etc. are rebuilt on the fly. Returns #blocks wrapped.
    """
    raise NotImplementedError("TODO: enable_activation_checkpointing")


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
    raise NotImplementedError("TODO: train_model")


# =========================================================================== Task 8
def training_memory_estimate(n_params: float, n_layers: int, d_model: int, n_heads: int,
                             seq: int, batch: int, act_checkpointing: bool = False,
                             flash: bool = False) -> Dict[str, float]:
    """Bytes for mixed-precision AdamW training (no sharding).

    model states: 2 (bf16 w) + 2 (bf16 g) + 4 (fp32 master) + 8 (Adam m, v) = 16 B/param
    activations : s*b*h*(34 + 5*a*s/h) bytes/layer (bf16, Korthikanti et al. 2022);
                  with full checkpointing only the block input (2*s*b*h bytes) per layer.
    """
    raise NotImplementedError("TODO: training_memory_estimate")


def chinchilla_optimal(compute_flops: float, tokens_per_param: float = 20.0):
    """C = 6 N D and D = k N  =>  N = sqrt(C / (6k)), D = k N."""
    raise NotImplementedError("TODO: chinchilla_optimal")


def mfu(tokens_per_sec: float, n_params: float, peak_flops: float) -> float:
    """Model FLOPs utilisation using the 6N FLOPs/token training estimate."""
    raise NotImplementedError("TODO: mfu")


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
    raise NotImplementedError("TODO: load_openai_gpt2_params")
