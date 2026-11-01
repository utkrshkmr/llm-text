import copy
import math
import os

import numpy as np
import pytest
import torch

from _labutil import load_impl
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel

lab = load_impl(__file__)
torch.set_num_threads(1)   # tiny models: extra threads only add contention


def _quadratic_problem(seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(6, 5, generator=g))
    b = torch.nn.Parameter(torch.randn(6, generator=g))
    X = torch.randn(32, 5, generator=g)
    Y = torch.randn(32, 6, generator=g)
    return [W, b], lambda: ((X @ W.T + b - Y) ** 2).mean()


def _run(opt_cls, kw, steps=25, seed=0):
    params, loss_fn = _quadratic_problem(seed)
    opt = opt_cls(params, **kw)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    return [p.detach().clone() for p in params]


@pytest.mark.parametrize("kw", [dict(lr=0.05), dict(lr=0.05, momentum=0.9),
                                dict(lr=0.05, momentum=0.9, nesterov=True),
                                dict(lr=0.05, momentum=0.9, weight_decay=0.01)])
def test_sgd_matches_torch(kw):
    ours = _run(lab.SGD, kw)
    ref = _run(torch.optim.SGD, kw)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("foreach", [False, True])
def test_adamw_matches_torch(foreach):
    kw = dict(lr=0.03, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ours = _run(lab.AdamW, dict(kw, foreach=foreach))
    ref = _run(torch.optim.AdamW, kw)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_adam_l2_matches_torch_adam():
    kw = dict(lr=0.03, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05)
    ours = _run(lab.AdamW, dict(kw, decoupled=False))
    ref = _run(torch.optim.Adam, kw)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_adamw_first_step_is_sign_like():
    # bias correction makes the first update ~ lr * sign(g) (|m_hat|/sqrt(v_hat) = 1)
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    opt = lab.AdamW([p], lr=0.1, weight_decay=0.0, eps=1e-12)
    p.grad = torch.tensor([0.5, -4.0, 1e-3])
    opt.step()
    torch.testing.assert_close(p.detach(), torch.tensor([0.9, -1.9, 2.9]), rtol=0, atol=1e-6)


def test_schedules():
    s = lab.WarmupCosine(peak=1.0, total=100, warmup=10, floor=0.1)
    assert s(0) == pytest.approx(0.1) and s(9) == pytest.approx(1.0)
    assert s(55) == pytest.approx(0.1 + 0.45 * (1 + math.cos(math.pi * 0.5)))
    assert s(100) == pytest.approx(0.1) and s(500) == pytest.approx(0.1)
    w = lab.WSD(peak=1.0, total=100, warmup=10, decay_frac=0.2, floor=0.0)
    assert w(50) == pytest.approx(1.0) and w(79) == pytest.approx(1.0)
    assert w(90) == pytest.approx(0.5) and w(100) == pytest.approx(0.0)


def test_clip_matches_torch():
    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(3)]
    b = copy.deepcopy(a)
    for x, y in zip(a, b):
        x.grad = torch.randn_like(x) * 5
        y.grad = x.grad.clone()
    n1 = lab.clip_grad_norm_(a, 1.0)
    n2 = torch.nn.utils.clip_grad_norm_(b, 1.0)
    assert n1 == pytest.approx(float(n2), rel=1e-5)
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
    total = torch.sqrt(sum((x.grad ** 2).sum() for x in a))
    assert float(total) <= 1.0 + 1e-6
    # below the threshold nothing changes
    g0 = a[0].grad.clone()
    lab.clip_grad_norm_(a, 100.0)
    torch.testing.assert_close(a[0].grad, g0)


def test_batcher_shapes_shift_and_resume(tmp_path):
    toks = np.arange(1000, dtype=np.uint16)
    path = tmp_path / "toks.bin"
    toks.tofile(path)
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    b1 = lab.RandomWindowBatcher(mm, batch_size=4, block_size=16, seed=3)
    x, y = next(b1)
    assert x.shape == (4, 16) and x.dtype == torch.long
    assert torch.equal(y, x + 1)                        # targets are inputs shifted by one
    sd = b1.state_dict()
    nxt = [next(b1)[0] for _ in range(3)]
    b2 = lab.RandomWindowBatcher(mm, batch_size=4, block_size=16, seed=99)
    b2.load_state_dict(sd)
    for a in nxt:
        assert torch.equal(a, next(b2)[0])


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(32, 16) @ torch.diag(torch.logspace(-1, 0, 16))   # condition number ~10
    O = lab.newton_schulz_orthogonalize(G, steps=5)
    s = torch.linalg.svdvals(O)
    assert s.max() < 1.3 and s.min() > 0.5             # all singular values pulled to ~1
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    cos = (O * (U @ Vh)).sum() / (O.norm() * (U @ Vh).norm())
    assert cos > 0.95


def test_state_bytes_accounting():
    torch.manual_seed(0)
    m = GPTModel(GPT_CONFIG_TINY)
    opt = lab.AdamW(m.parameters(), lr=1e-3)
    m(torch.randint(0, 512, (2, 8))).sum().backward()
    opt.step()
    n = sum(p.numel() for p in m.parameters())
    assert lab.optimizer_state_bytes(opt) == 2 * 4 * n        # m and v in fp32
    parts = lab.training_state_bytes(n)
    assert parts["total"] == 16 * n
    mp = lab.training_state_bytes(n, weight_bytes=2, grad_bytes=2, master_bytes=4)
    assert mp["total"] == 16 * n and mp["weights"] == 2 * n


def test_logger_roundtrip(tmp_path):
    run = lab.RunLogger(str(tmp_path / "run"), dict(lr=1e-3), stdout=False)
    for s in range(3):
        run.log(s, loss=1.0 / (s + 1), lr=1e-3)
    run.close()
    recs = lab.load_jsonl(os.path.join(tmp_path, "run", "metrics.jsonl"))
    assert [r["step"] for r in recs] == [0, 1, 2]
    assert all("wall" in r for r in recs) and recs[2]["loss"] == pytest.approx(1 / 3)
    # a truncated final line (crash mid-write) is ignored
    with open(os.path.join(tmp_path, "run", "metrics.jsonl"), "a") as f:
        f.write('{"step": 3, "lo')
    assert len(lab.load_jsonl(os.path.join(tmp_path, "run", "metrics.jsonl"))) == 3


def test_resume_is_bit_exact(tmp_path):
    toks = (np.arange(20_000) * 7 % 512).astype(np.uint16)   # learnable: next = prev + 7
    cfg = lab.LoopConfig(steps=16, ckpt_every=8, eval_every=1000, batch_size=4, block_size=16)
    torch.manual_seed(1)
    m_full = GPTModel(GPT_CONFIG_TINY)
    m_crash = copy.deepcopy(m_full)
    full = lab.train_loop(m_full, toks, cfg, str(tmp_path / "full"))
    part1 = lab.train_loop(m_crash, toks, cfg, str(tmp_path / "crash"), stop_after=11)
    assert len(part1) == 11
    m_new = GPTModel(GPT_CONFIG_TINY)                    # fresh process: fresh weights
    part2 = lab.train_loop(m_new, toks, cfg, str(tmp_path / "crash"))
    assert len(part2) == 16 - 8                          # resumed from the step-8 checkpoint
    assert part2 == full[8:]                             # identical losses, bit for bit
    for a, b in zip(m_full.parameters(), m_new.parameters()):
        assert torch.equal(a, b)
    assert full[-1] < full[0]
