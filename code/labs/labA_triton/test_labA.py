"""Lab A tests.

Without a GPU the Triton kernels run through Triton's CPU interpreter (TRITON_INTERPRET=1,
set below *before* triton is imported). That checks correctness, not speed. Tests marked
``gpu`` additionally need CUDA and are skipped otherwise.
"""
import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")

from _labutil import load_impl  # noqa: E402
from llmsys.kernels import reference as ref  # noqa: E402

lab = load_impl(__file__)
DEV = "cuda" if HAS_CUDA else "cpu"
gpu = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")


def t(*shape, dtype=torch.float32):
    return torch.randn(*shape, dtype=dtype, device=DEV)


def test_vector_add_ragged():
    torch.manual_seed(0)
    x, y = t(3001), t(3001)
    assert torch.allclose(lab.vector_add(x, y, block=256), x + y)


def test_softmax_rows():
    torch.manual_seed(0)
    x = t(19, 300) * 10
    out = lab.softmax_rows(x)
    assert torch.allclose(out.cpu(), ref.softmax_rows(x.cpu()), atol=1e-6)
    assert torch.allclose(out.sum(-1).cpu(), torch.ones(19), atol=1e-5)


def test_rmsnorm():
    torch.manual_seed(0)
    x, w = t(2, 5, 96), t(96)
    assert torch.allclose(lab.rmsnorm(x, w).cpu(), ref.rmsnorm(x.cpu(), w.cpu()), atol=1e-5)


@pytest.mark.parametrize("M,N,K", [(64, 64, 64), (70, 45, 33)])
def test_matmul(M, N, K):
    torch.manual_seed(0)
    a, b = t(M, K), t(K, N)
    out = lab.matmul(a, b, bm=32, bn=32, bk=16, group_m=2)
    assert torch.allclose(out.cpu(), (a @ b).cpu(), atol=1e-3)


@pytest.mark.parametrize("causal", [True, False])
def test_flash_attention_triton(causal):
    torch.manual_seed(0)
    q, k, v = t(1, 2, 40, 16), t(1, 2, 40, 16), t(1, 2, 40, 16)
    out = lab.flash_attention(q, k, v, causal=causal, block_m=16, block_n=16)
    assert torch.allclose(out.cpu(), ref.attention(q.cpu(), k.cpu(), v.cpu(), causal), atol=1e-4)


@pytest.mark.parametrize("T,S", [(33, 33), (5, 21)])
def test_flash_attention_torch_tiled(T, S):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, T, 8), torch.randn(2, 3, S, 8), torch.randn(2, 3, S, 8)
    for causal in (True, False):
        out = lab.flash_attention_tiled_torch(q, k, v, causal, block_m=8, block_n=4)
        assert torch.allclose(out, ref.attention(q, k, v, causal), atol=1e-5)


def test_napkin_math():
    assert abs(lab.arithmetic_intensity("add", 10**6, 4) - 1 / 12) < 1e-12
    assert lab.arithmetic_intensity("matmul", 4096, 2) > 1000
    # memory bound: 1 GB at 1 TB/s = 1 ms even though FLOPs are tiny
    assert abs(lab.roofline_time(1e6, 1e9, 1e15, 1e12) - 1e-3) < 1e-12
    assert abs(lab.roofline_time(1e15, 1e3, 1e15, 1e12) - 1.0) < 1e-12


@gpu
def test_gpu_bf16_large():
    torch.manual_seed(0)
    a, b = t(1024, 512, dtype=torch.bfloat16), t(512, 768, dtype=torch.bfloat16)
    out = lab.matmul(a, b)
    assert torch.allclose(out, (a.float() @ b.float()), atol=0.5, rtol=2e-2)
    q = t(2, 8, 1024, 64, dtype=torch.bfloat16)
    o = lab.flash_attention(q, q, q, causal=True)
    r = torch.nn.functional.scaled_dot_product_attention(q.float(), q.float(), q.float(),
                                                         is_causal=True)
    assert torch.allclose(o, r, atol=2e-2)


def test_worked_examples_in_llmsys():
    """SwiGLU and the RMSNorm backward/autograd wrapper from Appendix A (provided code)."""
    from llmsys.kernels import triton_kernels as K
    torch.manual_seed(0)
    g, u = t(3, 333), t(3, 333)
    assert torch.allclose(K.swiglu(g, u, block=128).cpu(), ref.swiglu(g.cpu(), u.cpu()), atol=1e-6)
    x = t(2, 5, 64).requires_grad_(); w = t(64).requires_grad_()
    y = ref.rmsnorm(x, w); dy = torch.randn_like(y); y.backward(dy)
    gx, gw = x.grad.clone(), w.grad.clone(); x.grad = None; w.grad = None
    K.TritonRMSNorm.apply(x, w).backward(dy)
    assert torch.allclose(x.grad, gx, atol=1e-5) and torch.allclose(w.grad, gw, atol=1e-4)


def test_paged_decode_worked_example():
    from llmsys.kernels import triton_kernels as K
    torch.manual_seed(0)
    B, H, D, bs, NB = 3, 2, 16, 4, 20
    q = t(B, H, D); kc = t(NB, bs, H, D); vc = t(NB, bs, H, D)
    bt = torch.randperm(NB)[:15].reshape(3, 5).int().to(DEV)
    sl = torch.tensor([17, 5, 9], dtype=torch.int32, device=DEV)
    out = K.paged_decode_attention(q, kc, vc, bt, sl)
    expect = ref.paged_decode_attention(q.cpu(), kc.cpu(), vc.cpu(), bt.cpu(), sl.cpu())
    assert torch.allclose(out.cpu(), expect, atol=1e-5)
