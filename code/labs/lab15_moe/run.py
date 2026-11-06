"""Lab 15 training driver: a tiny dense model against an MoE of the same *active* size.

    python labs/lab15_moe/run.py --steps 300 --csv moe_vs_dense.csv
    python labs/lab15_moe/run.py --steps 300 --balance auxfree --gate sigmoid
    python labs/lab15_moe/run.py --steps 300 --experts 16 --topk 2 --capacity-factor 1.0

The corpus is synthetic and offline: ``--domains`` different bigram Markov chains over one shared
vocabulary, each sequence drawn from one chain. A dense SwiGLU FFN of hidden width ``--hidden``
is compared with an MoE whose ``--topk`` experts of width hidden/topk give the same active FLOPs.
Both models are otherwise identical (same attention, same embeddings, same seed). The CSV has one
row per step for each model: loss, the router's max/mean load, dead experts and dropped tokens.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from llmsys.layers import RMSNorm, SwiGLUFeedForward  # noqa: E402
from llmsys.moe_layer import MoELayer, MoEShape, moe_params  # noqa: E402


# --------------------------------------------------------------------------- data
class MarkovCorpus:
    """``n_domains`` bigram chains over ``vocab`` tokens; sequence i comes from domain i % n_domains.

    Each chain's transition matrix is a sparse random stochastic matrix (``branching`` successors
    per token), so the next token is predictable only if the model knows which domain it is in:
    the natural case for experts to specialise on.
    """

    def __init__(self, vocab: int, n_domains: int, branching: int = 4, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.vocab, self.n_domains = vocab, n_domains
        self.trans = torch.zeros(n_domains, vocab, vocab)
        for d in range(n_domains):
            succ = torch.rand(vocab, vocab, generator=g).topk(branching, dim=-1).indices
            logits = torch.randn(vocab, branching, generator=g)
            self.trans[d].scatter_(1, succ, torch.softmax(logits, -1))
        self.g = g

    def batch(self, batch: int, seq: int, device) -> torch.Tensor:
        tokens = torch.empty(batch, seq + 1, dtype=torch.long)
        tokens[:, 0] = torch.randint(0, self.vocab, (batch,), generator=self.g)
        domain = torch.arange(batch) % self.n_domains
        for t in range(seq):
            probs = self.trans[domain, tokens[:, t]]                  # (B, V)
            tokens[:, t + 1] = torch.multinomial(probs, 1, generator=self.g).squeeze(1)
        return tokens.to(device)

    def entropy_floor(self) -> float:
        """Mean conditional entropy of the chains: the loss an oracle that knows the domain reaches."""
        p = self.trans.clamp(min=1e-12)
        return float(-(self.trans * p.log()).sum(-1).mean())


# --------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn: nn.Module):
        super().__init__()
        self.n1, self.n2, self.ffn = RMSNorm(d), RMSNorm(d), ffn
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.n_heads = n_heads

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.n_heads, d // self.n_heads).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ffn(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int, n_layers: int, n_heads: int, make_ffn):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(512, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, make_ffn()) for _ in range(n_layers)])
        self.norm, self.head = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

    def moe_layers(self) -> list[MoELayer]:
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoELayer)]


def build(args, kind: str, device):
    torch.manual_seed(args.seed)
    if kind == "dense":
        make = lambda: SwiGLUFeedForward(args.d, args.hidden)  # noqa: E731
    else:
        d_expert = args.hidden // args.topk
        make = lambda: MoELayer(  # noqa: E731
            args.d, d_expert, args.experts, args.topk, gate=args.gate,
            capacity_factor=args.capacity_factor,
            aux_loss_coef=args.aux_coef if args.balance == "aux" else 0.0,
            z_loss_coef=args.z_coef, aux_free_gamma=args.gamma if args.balance == "auxfree" else 0.0)
    return TinyLM(args.vocab, args.d, args.layers, args.heads, make).to(device)


# --------------------------------------------------------------------------- training
def train(args, kind: str, corpus: MarkovCorpus, device, writer):
    model = build(args, kind, device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    corpus.g.manual_seed(args.seed + 1)                       # both models see the same batches
    moes = model.moe_layers()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        tokens = corpus.batch(args.batch, args.seq, device)
        logits = model(tokens[:, :-1])
        lm_loss = F.cross_entropy(logits.reshape(-1, args.vocab), tokens[:, 1:].reshape(-1))
        loss = lm_loss + sum((m.balance_loss() for m in moes), torch.zeros((), device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        row = dict(model=kind, step=step, loss=round(lm_loss.item(), 4))
        if moes:
            stats = [m.last_stats for m in moes]
            row.update(max_over_mean=round(sum(s["max_over_mean"] for s in stats) / len(stats), 3),
                       dead=sum(s["dead"] for s in stats), dropped=sum(m.last_dropped for m in moes),
                       aux=round(sum(m.last_aux_loss.item() for m in moes) / len(moes), 4))
        writer(row)
        if step % args.log_every == 0 or step == args.steps:
            extra = (f"  max/mean {row['max_over_mean']:.2f} dead {row['dead']} "
                     f"dropped {row['dropped']} aux {row['aux']:.3f}" if moes else "")
            print(f"[{kind:5s}] step {step:4d} loss {row['loss']:.4f}{extra}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.steps
    print(f"[{kind:5s}] {n_params / 1e6:.2f}M params, {dt * 1e3:.1f} ms/step")
    return row["loss"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--domains", type=int, default=8)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128, help="dense FFN width = active expert width")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--gate", choices=["softmax", "sigmoid"], default="softmax")
    ap.add_argument("--balance", choices=["aux", "auxfree", "none"], default="aux")
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--z-coef", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=1e-3, help="aux-free bias step")
    ap.add_argument("--capacity-factor", type=float, default=None, help="None = dropless")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--only", choices=["dense", "moe"], default=None)
    ap.add_argument("--csv", default=None, help="write per-step rows here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    corpus = MarkovCorpus(args.vocab, args.domains, seed=args.seed)
    shape = MoEShape(args.d, args.hidden // args.topk, args.experts, args.topk)
    p = moe_params(shape)
    print(f"corpus: {args.domains} domains, entropy floor {corpus.entropy_floor():.3f} nats/token")
    print(f"FFN per layer: dense {3 * args.d * args.hidden:,} params; MoE total {p['total']:,}, "
          f"active {p['active']:,} (E={args.experts}, k={args.topk}, d_expert={shape.d_expert})")
    rows = []
    kinds = [args.only] if args.only else ["dense", "moe"]
    finals = {k: train(args, k, corpus, device, rows.append) for k in kinds}
    print("final losses:", {k: round(v, 4) for k, v in finals.items()},
          f"(oracle floor {corpus.entropy_floor():.3f})")
    if args.csv:
        keys = ["model", "step", "loss", "max_over_mean", "dead", "dropped", "aux"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
