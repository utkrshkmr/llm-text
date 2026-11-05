"""Lab 13 tests (CPU, tiny GPT, synthetic data).

    LAB_IMPL=starter  python -m pytest -q labs/lab13_pretrain
    LAB_IMPL=solution python -m pytest -q labs/lab13_pretrain
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from _labutil import load_impl
from llmsys.data import SlidingWindowDataset
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel
from llmsys.utils import set_seed

lab = load_impl(__file__)


def _loaders(seed=0, n_tokens=4000, vocab=64, period=17):
    """A noisy periodic token stream: easy to learn, impossible to memorise instantly."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, vocab, (period,), generator=g)
    ids = base.repeat(n_tokens // period + 1)[:n_tokens].clone()
    noise = torch.rand(n_tokens, generator=g) < 0.05
    ids[noise] = torch.randint(0, vocab, (int(noise.sum()),), generator=g)
    split = int(0.9 * n_tokens)
    tr = SlidingWindowDataset(ids[:split], 32, 32)
    va = SlidingWindowDataset(ids[split:], 32, 32)
    return (DataLoader(tr, batch_size=8, shuffle=True, drop_last=True,
                       generator=torch.Generator().manual_seed(seed)),
            DataLoader(va, batch_size=8, shuffle=False))


def _model(seed=0):
    set_seed(seed)
    return GPTModel(dict(GPT_CONFIG_TINY, vocab_size=64, context_length=32))


# --------------------------------------------------------------------------- task 1
def test_cross_entropy_matches_torch():
    set_seed(0)
    logits = torch.randn(2, 5, 11) * 10
    targets = torch.randint(0, 11, (2, 5))
    ours = lab.cross_entropy_from_scratch(logits, targets)
    assert torch.allclose(ours, F.cross_entropy(logits.flatten(0, 1), targets.flatten()), atol=1e-5)
    # numerically stable for extreme logits
    big = torch.tensor([[[1000.0, -1000.0, 0.0]]])
    assert torch.isfinite(lab.cross_entropy_from_scratch(big, torch.tensor([[1]])))
    assert math.isclose(lab.perplexity(math.log(50257)), 50257, rel_tol=1e-9)


# --------------------------------------------------------------------------- task 2
def test_warmup_cosine():
    f = lambda s: lab.warmup_cosine_lr(s, 100, 1e-2, 20, 1e-4, 1e-3)
    assert math.isclose(f(0), 1e-4)
    assert math.isclose(f(10), 1e-4 + 0.5 * (1e-2 - 1e-4))
    assert math.isclose(f(20), 1e-2)
    assert math.isclose(f(60), 1e-3 + 0.5 * (1e-2 - 1e-3))     # halfway through decay
    assert math.isclose(f(100), 1e-3)
    lrs = [f(s) for s in range(20, 101)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


# --------------------------------------------------------------------------- task 3
def test_grad_clipping_matches_torch():
    m1, m2 = _model(), _model()
    x = torch.randint(0, 64, (2, 16))
    for m in (m1, m2):
        (m(x).float().pow(2).mean() * 100).backward()
    n1 = lab.clip_grad_norm_(m1.parameters(), 0.5)
    n2 = torch.nn.utils.clip_grad_norm_(m2.parameters(), 0.5)
    assert math.isclose(n1, float(n2), rel_tol=1e-4)
    assert math.isclose(lab.global_grad_norm(m1.parameters()), 0.5, rel_tol=1e-3)
    for p, q in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)


# --------------------------------------------------------------------------- task 4
def test_generate_text():
    m = _model().eval()
    idx = torch.tensor([[1, 2, 3]])
    greedy = lab.generate_text(m, idx, 6, context_size=32)
    assert greedy.shape == (1, 9)
    ref = idx
    for _ in range(6):
        ref = torch.cat([ref, m(ref)[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(greedy, ref)
    # top_k = 1 with any temperature is greedy
    s = lab.generate_text(m, idx, 6, 32, temperature=5.0, top_k=1)
    assert torch.equal(s, greedy)
    # eos stops generation immediately if it is the greedy token
    first = int(greedy[0, 3])
    assert lab.generate_text(m, idx, 6, 32, eos_id=first).shape == (1, 3)
    # sampling with a seeded generator is reproducible
    a = lab.generate_text(m, idx, 8, 32, 1.0, 10, generator=torch.Generator().manual_seed(1))
    b = lab.generate_text(m, idx, 8, 32, 1.0, 10, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- task 5+7
def test_training_reduces_loss_and_resumes_exactly(tmp_path):
    tr, va = _loaders()
    cfg = lab.TrainConfig(epochs=3, peak_lr=3e-3, initial_lr=3e-4, min_lr=3e-4,
                          warmup_frac=0.1, grad_clip=1.0, eval_freq=10, eval_iter=3)
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.peak_lr, weight_decay=0.1)
    before = lab.evaluate(m, va, cfg, 10)
    h = lab.train_model(m, tr, va, opt, cfg, log=lambda s: None)
    after = lab.evaluate(m, va, cfg, 10)
    assert after < before - 1.0, (before, after)
    assert h.tokens_per_sec > 0 and len(h.val_loss) >= 2

    # exact resume: train 4 steps, checkpoint, train 4 more == train 8 straight
    path = str(tmp_path / "ck.pt")
    cfg8 = lab.TrainConfig(epochs=1, peak_lr=1e-3, warmup_frac=0.0, eval_freq=10**9,
                           grad_clip=None, max_steps=8)
    ds = tr.dataset
    batches = [ds[i] for i in range(16)]
    stack = lambda bs: (torch.stack([b[0] for b in bs]), torch.stack([b[1] for b in bs]))
    loader = [stack(batches[i:i + 2]) for i in range(0, 16, 2)]

    def run(model, opt, steps, start):
        for i in range(start, start + steps):
            x, y = loader[i]
            opt.zero_grad()
            F.cross_entropy(model(x).flatten(0, 1), y.flatten()).backward()
            opt.step()

    a = _model(); oa = torch.optim.AdamW(a.parameters(), lr=1e-3)
    run(a, oa, 8, 0)
    b = _model(); ob = torch.optim.AdamW(b.parameters(), lr=1e-3)
    run(b, ob, 4, 0)
    lab.save_checkpoint(path, b, ob, step=4)
    c = _model(seed=99); oc = torch.optim.AdamW(c.parameters(), lr=1e-3)
    assert lab.load_checkpoint(path, c, oc) == 4
    run(c, oc, 4, 4)
    for p, q in zip(a.parameters(), c.parameters()):
        assert torch.allclose(p, q, atol=1e-6)


def test_grad_accumulation_equivalence():
    """4 micro-batches of 2 with accumulation == 1 batch of 8 (no dropout, mean loss)."""
    tr, _ = _loaders()
    x, y = next(iter(tr))
    m1, m2 = _model(), _model()
    F.cross_entropy(m1(x).flatten(0, 1), y.flatten()).backward()
    for xs, ys in zip(x.chunk(4), y.chunk(4)):
        (F.cross_entropy(m2(xs).flatten(0, 1), ys.flatten()) / 4).backward()
    for p, q in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)


def test_bf16_autocast_training_runs():
    tr, va = _loaders()
    cfg = lab.TrainConfig(epochs=2, peak_lr=3e-3, warmup_frac=0.1, eval_freq=10**9,
                          amp_dtype=torch.bfloat16)
    m = _model()
    before = lab.evaluate(m, va, cfg, 10)
    lab.train_model(m, tr, va, torch.optim.AdamW(m.parameters(), lr=3e-3), cfg, log=lambda s: None)
    assert lab.evaluate(m, va, cfg, 10) < before - 0.5


# --------------------------------------------------------------------------- task 6
def test_activation_checkpointing_same_gradients():
    m1, m2 = _model(), _model()
    m1.train(); m2.train()
    assert lab.enable_activation_checkpointing(m2) == len(m2.trf_blocks)
    x = torch.randint(0, 64, (2, 16))
    m1(x).float().pow(2).mean().backward()
    m2(x).float().pow(2).mean().backward()
    for p, q in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)


# --------------------------------------------------------------------------- task 8
def test_napkin_math():
    est = lab.training_memory_estimate(124e6, 12, 768, 12, 1024, 8)
    assert math.isclose(est["model_states"], 16 * 124e6)
    ck = lab.training_memory_estimate(124e6, 12, 768, 12, 1024, 8, act_checkpointing=True)
    assert ck["activations"] < est["activations"] / 20
    n, d = lab.chinchilla_optimal(5.76e23)
    assert 6.5e10 < n < 7.4e10 and math.isclose(d, 20 * n)      # ~70B params, ~1.4T tokens
    assert math.isclose(lab.mfu(1e4, 1e9, 6e13 / 0.5), 0.5)


# --------------------------------------------------------------------------- task 9
def test_load_openai_params_layout():
    cfg = dict(GPT_CONFIG_TINY, vocab_size=64, context_length=32, qkv_bias=True)
    m = GPTModel(cfg)
    d, rng = cfg["emb_dim"], np.random.default_rng(0)
    r = lambda *s: rng.standard_normal(s).astype(np.float32)
    blocks = [dict(ln_1=dict(g=r(d), b=r(d)), ln_2=dict(g=r(d), b=r(d)),
                   attn=dict(c_attn=dict(w=r(d, 3 * d), b=r(3 * d)), c_proj=dict(w=r(d, d), b=r(d))),
                   mlp=dict(c_fc=dict(w=r(d, 4 * d), b=r(4 * d)), c_proj=dict(w=r(4 * d, d), b=r(d))))
              for _ in range(cfg["n_layers"])]
    params = dict(wte=r(64, d), wpe=r(32, d), g=r(d), b=r(d), blocks=blocks)
    lab.load_openai_gpt2_params(m, params)
    blk, p = m.trf_blocks[1], blocks[1]
    assert np.allclose(blk.att.W_key.weight.detach().numpy(), p["attn"]["c_attn"]["w"][:, d:2 * d].T)
    assert np.allclose(blk.att.W_value.bias.detach().numpy(), p["attn"]["c_attn"]["b"][2 * d:])
    assert np.allclose(m.out_head.weight.detach().numpy(), params["wte"])
    # the layout must agree with a straightforward numpy forward of the first LayerNorm
    x = torch.randn(1, 4, d)
    ln = m.trf_blocks[0].norm1(x).detach().numpy()
    xn = x.numpy()
    mu, var = xn.mean(-1, keepdims=True), xn.var(-1, keepdims=True)
    expected = (xn - mu) / np.sqrt(var + 1e-5) * blocks[0]["ln_1"]["g"] + blocks[0]["ln_1"]["b"]
    assert np.allclose(ln, expected, atol=1e-4)
