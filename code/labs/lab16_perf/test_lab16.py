"""Lab 16 tests.

CPU-only machines run the Triton kernels through Triton's interpreter. TRITON_INTERPRET must be
set before triton is imported anywhere in the process, which is why it is set at the very top
of this file and why ``code/run_all_tests.sh`` runs every lab in its own pytest process.
"""
import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")

from _labutil import load_impl  # noqa: E402

lab = load_impl(__file__)
DEV = "cuda" if HAS_CUDA else "cpu"
gpu = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")


# ------------------------------------------------------------------ Part A
def test_time_fn_excludes_warmup():
    calls = []
    stats = lab.time_fn(lambda: calls.append(1), warmup=3, iters=7, sync=lambda: None)
    assert len(calls) == 10
    assert len(stats["times"]) == 7
    assert stats["std"] >= 0 and stats["mean"] >= 0
    assert min(stats["times"]) <= stats["median"] <= max(stats["times"])


def test_tracker_counts_and_dedups():
    x = torch.randn(64, 32, requires_grad=True)
    y = torch.randn(64, 32, requires_grad=True)
    with lab.SavedTensorTracker() as tr:
        _ = x * y                         # mul saves both operands
    assert tr.total_bytes == 2 * 64 * 32 * 4
    with lab.SavedTensorTracker() as tr2:
        _ = x * x                         # same storage saved twice -> counted once
    assert tr2.total_bytes == 64 * 32 * 4
    lin = torch.nn.Linear(32, 8, bias=False)
    with lab.SavedTensorTracker() as tr3:
        _ = lin(x)                        # saves input x, weight is a Parameter (skipped)
    assert tr3.total_bytes == 64 * 32 * 4


@pytest.mark.parametrize("causal", [False, True])
def test_naive_attention_saved_bytes(causal):
    B, H, T, D = 2, 3, 12, 8
    q, k, v = (torch.randn(B, H, T, D, requires_grad=True) for _ in range(3))
    with lab.SavedTensorTracker() as tr:
        _ = lab.naive_attention(q, k, v, causal)
    assert tr.total_bytes == lab.naive_attention_saved_bytes(B, H, T, D, 4, causal)


# ------------------------------------------------------------------ Part B
def _blocks(n=6, d=32):
    torch.manual_seed(0)
    return [torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))
            for _ in range(n)]


def test_checkpoint_segments_saves_memory_and_matches_grads():
    blocks = _blocks()
    x = torch.randn(16, 32, requires_grad=True)
    measured = {}
    grads = {}
    for seg in (0, 1, 2, 3, 6):
        for b in blocks:
            b.zero_grad()
        x.grad = None
        with lab.SavedTensorTracker() as tr:
            y = lab.checkpoint_segments(blocks, x, seg)
        measured[seg] = tr.total_bytes
        y.square().sum().backward()
        grads[seg] = [x.grad.clone()] + [p.grad.clone() for b in blocks for p in b.parameters()]
    assert measured[1] < measured[0] and measured[6] < measured[0]
    for seg in (1, 2, 3, 6):
        for a, b in zip(grads[0], grads[seg]):
            assert torch.allclose(a, b, atol=1e-6)


# ------------------------------------------------------------------ Part C
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("T", [32, 37])
def test_fa2_forward_tiled(causal, T):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 2, T, 16, dtype=torch.float64) for _ in range(3))
    o, lse = lab.fa2_forward_tiled(q, k, v, causal, 8, 16)
    o_ref, lse_ref = lab.attention_with_lse(q, k, v, causal)
    assert torch.allclose(o, o_ref, atol=1e-10)
    assert torch.allclose(lse.double(), lse_ref, atol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_fa2_backward_tiled(causal):
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 2, 24, 8, dtype=torch.float64, requires_grad=True) for _ in range(3))
    o = lab.naive_attention(q, k, v, causal)
    do = torch.randn_like(o)
    o.backward(do)
    with torch.no_grad():
        o2, lse = lab.attention_with_lse(q, k, v, causal)
        g = lab.fa2_backward_tiled(q, k, v, o2, do, lse, causal, 8, 8)
    for a, b in zip(g, (q.grad, k.grad, v.grad)):
        assert torch.allclose(a.double(), b, atol=1e-5)


def test_fa2_autograd_gradcheck():
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 1, 10, 4, dtype=torch.float64, requires_grad=True) for _ in range(3))
    f = lambda a, b, c: lab.FlashAttention2Tiled.apply(a, b, c, True, 4, 4)
    assert torch.autograd.gradcheck(f, (q, k, v), atol=1e-5)


# ------------------------------------------------------------------ Part D
@pytest.mark.parametrize("causal", [False, True])
def test_triton_forward(causal):
    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 2, 48, 16, device=DEV) for _ in range(3))
    o, lse = lab.fa2_triton_forward(q, k, v, causal, 16, 16)
    o_ref, lse_ref = lab.attention_with_lse(q, k, v, causal)
    tol = 2e-2 if HAS_CUDA else 1e-5          # tf32 dots on GPU
    assert torch.allclose(o, o_ref, atol=tol)
    assert torch.allclose(lse, lse_ref, atol=tol)


@pytest.mark.parametrize("causal", [False, True])
def test_triton_backward(causal):
    torch.manual_seed(4)
    q, k, v = (torch.randn(1, 2, 32, 16, device=DEV, requires_grad=True) for _ in range(3))
    out = lab.FlashAttention2Triton.apply(q, k, v, causal, 16, 16)
    do = torch.randn_like(out)
    out.backward(do)
    q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
    lab.naive_attention(q2, k2, v2, causal).backward(do)
    tol = 3e-2 if HAS_CUDA else 1e-4
    for a, b in zip((q.grad, k.grad, v.grad), (q2.grad, k2.grad, v2.grad)):
        assert torch.allclose(a, b, atol=tol)


@gpu
def test_triton_bf16_matches_sdpa():
    torch.manual_seed(5)
    q, k, v = (torch.randn(2, 4, 512, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o = lab.FlashAttention2Triton.apply(q, k, v, True, 64, 64)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (o.float() - ref.float()).abs().max() < 2e-2
