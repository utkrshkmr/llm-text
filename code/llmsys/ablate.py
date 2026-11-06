"""A configurable decoder-only LM for architecture ablations (Chapter 10).

Every design choice that Chapter 10 discusses is a field of :class:`ArchConfig`:
norm type and placement, positional encoding, MLP activation/gating, QK-norm,
z-loss, logit soft-capping, tied embeddings, parallel blocks, biases, and
initialisation. Linear and embedding layers are written from scratch with
``torch.einsum`` so that the parameter layout (``weight: (d_out, d_in)``) and the
initialisation are explicit.

Typical use::

    from llmsys.ablate import ArchConfig, ArchLM, train_lm, make_corpus
    ids = make_corpus(200_000)
    hist = train_lm(ArchConfig(norm_position="post"), ids, steps=300)
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import apply_rope, compute_rope_params


# --------------------------------------------------------------------------- config
@dataclass
class ArchConfig:
    vocab_size: int = 256
    context_length: int = 128
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: Optional[int] = None          # None -> derived from ffn kind (4d or 8/3 d)
    ff_multiple_of: int = 16            # round d_ff up to this (64 on real GPUs)
    norm: str = "rms"                   # rms | layer | none
    norm_position: str = "pre"          # pre | post | sandwich
    pos: str = "rope"                   # rope | learned | sinusoidal | alibi | nope
    ffn: str = "swiglu"                 # swiglu | geglu | reglu | silu | gelu | relu | relu2
    qk_norm: bool = False
    z_loss: float = 0.0                 # coefficient on mean(logsumexp^2)
    logit_softcap: Optional[float] = None
    attn_softcap: Optional[float] = None
    tie_embeddings: bool = False
    parallel_block: bool = False
    bias: bool = False
    init: str = "trunc_normal"          # trunc_normal | gpt2 | xavier
    depth_scaled_init: bool = True      # scale output projections by 1/sqrt(2L)
    emb_std: float = 0.02
    rope_base: float = 10_000.0
    use_sdpa: bool = False              # fused attention kernel (GPU); not with attn_softcap

    def ff_dim(self) -> int:
        return self.d_ff or ffn_hidden_dim(self.d_model, self.ffn, self.ff_multiple_of)


GATED = {"swiglu", "geglu", "reglu"}


def ffn_hidden_dim(d_model: int, kind: str, multiple_of: int = 64) -> int:
    """Parameter-matched MLP width: 4d for 2-matrix MLPs, ~8/3 d for 3-matrix GLUs."""
    raw = 8 * d_model / 3 if kind in GATED else 4 * d_model
    return int(multiple_of * math.ceil(raw / multiple_of))


# --------------------------------------------------------------------------- init
def trunc_normal_(w: torch.Tensor, std: float) -> torch.Tensor:
    """Normal(0, std^2) truncated at +-3 std (resampled, not clamped)."""
    return nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)


def init_std(d_in: int, d_out: int, scheme: str) -> float:
    if scheme == "trunc_normal":          # variance-preserving average of fan-in and fan-out
        return math.sqrt(2.0 / (d_in + d_out))
    if scheme == "xavier":                # Glorot: same variance target, used with uniform
        return math.sqrt(2.0 / (d_in + d_out))
    if scheme == "gpt2":                  # GPT-2's fixed small std
        return 0.02
    raise ValueError(scheme)


# --------------------------------------------------------------------------- layers
class Linear(nn.Module):
    """y = x W^T (+ b). Weight stored as (d_out, d_in), like nn.Linear."""

    def __init__(self, d_in: int, d_out: int, bias: bool = False, std: Optional[float] = None,
                 scheme: str = "trunc_normal"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        std = std if std is not None else init_std(d_in, d_out, scheme)
        if scheme == "xavier":
            a = math.sqrt(3.0) * std
            nn.init.uniform_(self.weight, -a, a)
        else:
            trunc_normal_(self.weight, std)
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    def forward(self, x):
        y = torch.einsum("...i,oi->...o", x, self.weight)
        return y if self.bias is None else y + self.bias


class Embedding(nn.Module):
    """Lookup table (num_embeddings, d). Indexing is a gather, not a matmul."""

    def __init__(self, num: int, d: int, std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_(torch.empty(num, d), std))

    def forward(self, ids):
        return self.weight[ids]


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight).to(x.dtype)


class LayerNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d)) if bias else None

    def forward(self, x):
        xf = x.float()
        xf = (xf - xf.mean(-1, keepdim=True)) * torch.rsqrt(xf.var(-1, keepdim=True, unbiased=False) + self.eps)
        y = xf * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)


def make_norm(kind: str, d: int, bias: bool = False) -> nn.Module:
    if kind == "rms":
        return RMSNorm(d)
    if kind == "layer":
        return LayerNorm(d, bias=bias)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


ACTS = {
    "relu": F.relu, "gelu": F.gelu, "silu": F.silu,
    "relu2": lambda x: F.relu(x).square(),
}


class MLP(nn.Module):
    """Position-wise MLP. GLU variants: down(act(gate(x)) * up(x))."""

    def __init__(self, cfg: ArchConfig, out_std: float):
        super().__init__()
        d, h, kind = cfg.d_model, cfg.ff_dim(), cfg.ffn
        self.kind = kind
        self.gated = kind in GATED
        self.act = {"swiglu": F.silu, "geglu": F.gelu, "reglu": F.relu}.get(kind) or ACTS[kind]
        self.w_in = Linear(d, h, cfg.bias, scheme=cfg.init)
        self.w_gate = Linear(d, h, cfg.bias, scheme=cfg.init) if self.gated else None
        self.w_out = Linear(h, d, cfg.bias, std=out_std, scheme=cfg.init)

    def forward(self, x):
        if self.gated:
            return self.w_out(self.act(self.w_gate(x)) * self.w_in(x))
        return self.w_out(self.act(self.w_in(x)))


# --------------------------------------------------------------------------- positions
def sinusoidal_positions(T: int, d: int, base: float = 10_000.0) -> torch.Tensor:
    """(T, d) table: dims 2i, 2i+1 = sin, cos of pos / base^(2i/d)."""
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Geometric slopes 2^(-8/n), 2^(-16/n), ... (Press et al., 2022) for power-of-2 n;
    otherwise interleave the next power of two as in the reference implementation."""
    def pow2(n):
        start = 2 ** (-8.0 / n)
        return [start ** (i + 1) for i in range(n)]
    if math.log2(n_heads).is_integer():
        return torch.tensor(pow2(n_heads))
    m = 2 ** math.floor(math.log2(n_heads))
    extra = pow2(2 * m)[0::2][: n_heads - m]
    return torch.tensor(pow2(m) + extra)


def alibi_bias(n_heads: int, T: int) -> torch.Tensor:
    """(H, T, T) additive bias: -slope_h * (i - j) for j <= i (future is masked separately)."""
    i = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    dist = (i - j).clamp(min=0).float()
    return -alibi_slopes(n_heads)[:, None, None] * dist


def softcap(x: torch.Tensor, cap: Optional[float]) -> torch.Tensor:
    """cap * tanh(x / cap): ~identity for |x| << cap, saturates smoothly at +-cap."""
    return x if cap is None else cap * torch.tanh(x / cap)


# --------------------------------------------------------------------------- attention
class Attention(nn.Module):
    def __init__(self, cfg: ArchConfig, out_std: float):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        assert d % H == 0
        self.cfg, self.H, self.dh = cfg, H, d // H
        self.wq = Linear(d, d, cfg.bias, scheme=cfg.init)
        self.wk = Linear(d, d, cfg.bias, scheme=cfg.init)
        self.wv = Linear(d, d, cfg.bias, scheme=cfg.init)
        self.wo = Linear(d, d, cfg.bias, std=out_std, scheme=cfg.init)
        self.q_norm = RMSNorm(self.dh) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.dh) if cfg.qk_norm else None

    def forward(self, x, rope=None, alibi=None):
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.H, self.dh).transpose(1, 2)   # (B,H,T,dh)
        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.cfg.use_sdpa and self.cfg.attn_softcap is None:
            if alibi is None:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
                bias = alibi[:, :T, :T].to(q.dtype).masked_fill(m, float("-inf"))
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            return self.wo(o.transpose(1, 2).reshape(B, T, -1))
        s = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(self.dh)
        s = softcap(s, self.cfg.attn_softcap)
        if alibi is not None:
            s = s + alibi[:, :T, :T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        p = torch.softmax(s.float().masked_fill(mask, float("-inf")), dim=-1).to(v.dtype)
        o = torch.einsum("bhqk,bhkd->bhqd", p, v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(o)


class Block(nn.Module):
    """One transformer block with configurable norm placement and parallel form.

    pre:      x + f(N(x))
    post:     N(x + f(x))
    sandwich: x + N2(f(N1(x)))         (extra norm on the branch output; 'peri-LN')
    parallel: x + attn(N(x)) + mlp(N(x)) with one shared norm (pre only)
    """

    def __init__(self, cfg: ArchConfig):
        super().__init__()
        out_std = init_std(cfg.d_model, cfg.d_model, cfg.init)
        if cfg.depth_scaled_init:
            out_std /= math.sqrt(2 * cfg.n_layers)
        self.cfg = cfg
        self.attn = Attention(cfg, out_std)
        self.mlp = MLP(cfg, out_std)
        mk = lambda: make_norm(cfg.norm, cfg.d_model, cfg.bias)
        self.n1, self.n2 = mk(), mk()
        if cfg.norm_position == "sandwich":
            self.n1b, self.n2b = mk(), mk()

    def forward(self, x, rope=None, alibi=None):
        c = self.cfg
        if c.parallel_block:
            h = self.n1(x)
            return x + self.attn(h, rope, alibi) + self.mlp(h)
        if c.norm_position == "pre":
            x = x + self.attn(self.n1(x), rope, alibi)
            return x + self.mlp(self.n2(x))
        if c.norm_position == "post":
            x = self.n1(x + self.attn(x, rope, alibi))
            return self.n2(x + self.mlp(x))
        if c.norm_position == "sandwich":
            x = x + self.n1b(self.attn(self.n1(x), rope, alibi))
            return x + self.n2b(self.mlp(self.n2(x)))
        raise ValueError(c.norm_position)


class ArchLM(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = Embedding(cfg.vocab_size, cfg.d_model, cfg.emb_std)
        if cfg.pos == "learned":
            self.pos_emb = Embedding(cfg.context_length, cfg.d_model, cfg.emb_std)
        if cfg.pos == "sinusoidal":
            self.register_buffer("pe", sinusoidal_positions(cfg.context_length, cfg.d_model),
                                 persistent=False)
        if cfg.pos == "rope":
            cos, sin = compute_rope_params(cfg.d_model // cfg.n_heads, cfg.rope_base,
                                           cfg.context_length)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        if cfg.pos == "alibi":
            self.register_buffer("alibi", alibi_bias(cfg.n_heads, cfg.context_length),
                                 persistent=False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # post-norm already ends every block with a norm; pre/sandwich need a final one
        self.final_norm = (make_norm(cfg.norm, cfg.d_model, cfg.bias)
                           if cfg.norm_position != "post" else nn.Identity())
        if cfg.tie_embeddings:
            self.head = None
        else:
            self.head = Linear(cfg.d_model, cfg.vocab_size, False, scheme=cfg.init)

    def forward(self, ids):
        c = self.cfg
        B, T = ids.shape
        x = self.tok(ids)
        if c.pos == "learned":
            x = x + self.pos_emb(torch.arange(T, device=ids.device))
        elif c.pos == "sinusoidal":
            x = x + self.pe[:T]
        rope = (self.cos, self.sin) if c.pos == "rope" else None
        alibi = self.alibi if c.pos == "alibi" else None
        for blk in self.blocks:
            x = blk(x, rope, alibi)
        x = self.final_norm(x)
        w = self.tok.weight if self.head is None else self.head.weight
        logits = torch.einsum("btd,vd->btv", x, w)
        return softcap(logits, c.logit_softcap)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, z_loss: float = 0.0):
    """Cross-entropy plus the optional z-loss  z * mean(logsumexp(logits)^2).

    Returns (total, ce, lse_mean) so the size of log Z can be logged.
    """
    lf = logits.float().flatten(0, -2)
    lse = torch.logsumexp(lf, dim=-1)
    ce = (lse - lf.gather(-1, targets.reshape(-1, 1)).squeeze(-1)).mean()
    total = ce + z_loss * lse.pow(2).mean() if z_loss > 0 else ce
    return total, ce.detach(), lse.mean().detach()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- data
_NOUNS = "cat dog bird fox owl frog bear wolf deer mouse".split()
_ADJS = "red big small quick lazy old young happy sad blue".split()
_VERBS = "sees chases likes finds helps follows meets calls".split()


def make_corpus(n_chars: int = 200_000, seed: int = 0) -> torch.Tensor:
    """A synthetic byte-level corpus whose statistics depend on *order and position*.

    Three line types are mixed: template sentences (local syntax), counting runs
    ("count 3 4 5 6 7.", which need position/order), and bracketed copy lines
    ("copy [abc] abc.", which need long-range attention). Returns a uint8-range LongTensor.
    """
    g = torch.Generator().manual_seed(seed)
    r = lambda n: int(torch.randint(n, (1,), generator=g))
    lines: List[str] = []
    total = 0
    while total < n_chars:
        kind = r(3)
        if kind == 0:
            s = (f"the {_ADJS[r(10)]} {_NOUNS[r(10)]} {_VERBS[r(8)]} "
                 f"the {_ADJS[r(10)]} {_NOUNS[r(10)]}.")
        elif kind == 1:
            a = r(20)
            s = "count " + " ".join(str(a + i) for i in range(3 + r(5))) + "."
        else:
            w = "".join(chr(97 + r(26)) for _ in range(3 + r(5)))
            s = f"copy [{w}] {w}."
        lines.append(s)
        total += len(s) + 1
    return torch.tensor(list("\n".join(lines).encode("ascii")), dtype=torch.long)


def get_batch(ids: torch.Tensor, batch: int, T: int, g: torch.Generator):
    """Random contiguous windows (the standard pretraining sampler)."""
    starts = torch.randint(0, len(ids) - T - 1, (batch,), generator=g)
    x = torch.stack([ids[s: s + T] for s in starts])
    y = torch.stack([ids[s + 1: s + T + 1] for s in starts])
    return x, y


# --------------------------------------------------------------------------- training
@dataclass
class AblationResult:
    name: str
    config: dict
    steps: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    lse: List[float] = field(default_factory=list)
    diverged: bool = False
    params: int = 0
    seconds: float = 0.0


def train_lm(cfg: ArchConfig, ids: torch.Tensor, steps: int = 300, batch: int = 16,
             lr: float = 3e-3, warmup: int = 30, weight_decay: float = 0.1, seed: int = 0,
             eval_every: int = 25, val_frac: float = 0.1, clip: float = 1.0,
             name: str = "run", log=None, device: str = "cpu", amp: bool = False,
             compile: bool = False) -> AblationResult:
    """Small but complete training loop for ablations (AdamW, warmup+cosine, clipping).

    Divergence (non-finite loss) stops the run and sets ``diverged``; this is a result,
    not an error, in an ablation study. On a GPU pass device="cuda", amp=True (bf16 autocast)
    and optionally compile=True; the data stays on the CPU and batches are copied over.
    """
    torch.manual_seed(seed)
    n_val = int(len(ids) * val_frac)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    model = ArchLM(cfg).to(device)
    fwd = torch.compile(model) if compile else model
    ctx = (lambda: torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)) \
        if amp else (lambda: torch.autocast(device_type="cpu", enabled=False))
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    res = AblationResult(name, asdict(cfg), params=count_params(model))
    g = torch.Generator().manual_seed(seed + 1)
    gv = torch.Generator().manual_seed(1234)
    T = cfg.context_length
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        lr_t = lr * step / warmup if step <= warmup else \
            0.1 * lr + 0.45 * lr * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        for grp in opt.param_groups:
            grp["lr"] = lr_t
        x, y = get_batch(train_ids, batch, T, g)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with ctx():
            logits = fwd(x)
        loss, ce, lse = lm_loss(logits, y, cfg.z_loss)
        if not torch.isfinite(loss):
            res.diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip else float("inf"))
        opt.step()
        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_ids, 32, T, gv)
                with ctx():
                    lv = fwd(xv.to(device))
                _, vce, _ = lm_loss(lv, yv.to(device))
            model.train()
            res.steps.append(step); res.train_loss.append(float(ce)); res.val_loss.append(float(vce))
            res.grad_norm.append(float(gn)); res.lse.append(float(lse))
            if not math.isfinite(float(vce)):
                res.diverged = True
                break
            if log:
                log(f"[{name}] step {step} train {float(ce):.3f} val {float(vce):.3f} |g| {float(gn):.2f}")
    res.seconds = time.perf_counter() - t0
    return res


STANDARD_ABLATIONS: Dict[str, dict] = {
    "baseline (pre-RMS, RoPE, SwiGLU)": {},
    "no norm": {"norm": "none"},
    "post-norm": {"norm_position": "post"},
    "NoPE": {"pos": "nope"},
    "learned abs. pos.": {"pos": "learned"},
    "SiLU MLP (4d)": {"ffn": "silu"},
    "GELU MLP (4d)": {"ffn": "gelu"},
    "QK-norm + z-loss": {"qk_norm": True, "z_loss": 1e-4},
}


def run_ablations(base: ArchConfig, ids: torch.Tensor, variants: Dict[str, dict] = None,
                  seeds=(0,), **train_kw) -> List[AblationResult]:
    variants = variants or STANDARD_ABLATIONS
    out = []
    for name, ov in variants.items():
        for s in seeds:
            out.append(train_lm(replace(base, **ov), ids, seed=s, name=name, **train_kw))
    return out
