"""Tests for llmsys.ops: backend switch and the compiled operators (Chapter 31).

The compiled backends are exercised when ``torch.ops.llmsys.rmsnorm`` exists: from the installed
extension (``pip install -e .``) or, with ``LLMSYS_JIT=1``, from a just-in-time build of the two
rmsnorm source files. Without either, those tests skip and the pure-Python parts still run.
"""
import pytest
import torch

from llmsys import ops
from llmsys.layers import RMSNorm
from llmsys.ops import rmsnorm as R

HAS_CUDA = torch.cuda.is_available()
BUILT = R.build()
cpp = pytest.mark.skipif(not BUILT, reason="compiled rmsnorm not available (build llmsys._C or set LLMSYS_JIT=1)")
gpu = pytest.mark.skipif(not (BUILT and HAS_CUDA and "cuda" in ops.available("rmsnorm", "cuda")),
                         reason="needs the CUDA build of the extension")


def test_registry_and_resolution():
    assert "python" in ops.available("rmsnorm")
    assert ops.resolve("rmsnorm", "cpu", "python") == "python"
    with pytest.raises(RuntimeError):
        ops.resolve("rmsnorm", "cpu", "cuda")          # a CUDA backend never runs CPU tensors
    assert ops.resolve("rmsnorm", "cpu") == ("cpp" if BUILT else "python")
    ops.set_backend("python")
    try:
        assert ops.resolve("rmsnorm", "cpu") == "python"
    finally:
        ops.set_backend("auto")


def test_python_backend_matches_layer():
    torch.manual_seed(0)
    m = RMSNorm(16)
    with torch.no_grad():
        m.scale.copy_(torch.rand(16) + 0.5)
    x = torch.randn(3, 5, 16, requires_grad=True)
    y = R.rmsnorm(x, m.scale, 1e-6, backend="python")
    ref = m(x)
    assert torch.allclose(y, ref, atol=1e-6)
    g = torch.randn_like(y)
    gx, gw = torch.autograd.grad(y, (x, m.scale), g)
    rx, rw = torch.autograd.grad(ref, (x, m.scale), g)
    assert torch.allclose(gx, rx, atol=1e-5) and torch.allclose(gw, rw, atol=1e-5)


@cpp
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cpp_matches_reference(dtype):
    torch.manual_seed(0)
    x = (torch.randn(4, 7, 96) * 3).to(dtype)
    w = torch.rand(96) + 0.5
    y = R.rmsnorm(x, w, 1e-6, backend="cpp")
    ref = R.rmsnorm_forward_ref(x, w, 1e-6)
    tol = 1e-6 if dtype == torch.float32 else 2e-2
    assert y.dtype == dtype and torch.allclose(y.float(), ref.float(), atol=tol, rtol=tol)


@cpp
def test_cpp_opcheck():
    x = torch.randn(2, 3, 32)
    w = torch.rand(32)
    torch.library.opcheck(torch.ops.llmsys.rmsnorm, (x, w, 1e-6))


@cpp
def test_cpp_backward_and_compile():
    x = torch.randn(2, 5, 32, requires_grad=True)
    w = torch.rand(32, requires_grad=True)
    y = R.rmsnorm(x, w, 1e-6, backend="cpp")
    y.sum().backward()
    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    R.rmsnorm(x2, w2, 1e-6, backend="python").sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-5) and torch.allclose(w.grad, w2.grad, atol=1e-5)
    fn = torch.compile(lambda a, b: torch.ops.llmsys.rmsnorm(a, b, 1e-6), fullgraph=True)
    assert torch.allclose(fn(x.detach(), w.detach()), y.detach(), atol=1e-6)


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("d", [32, 1024, 3072, 5000])
def test_cuda_matches_reference(dtype, d):
    torch.manual_seed(0)
    x = (torch.randn(3, 9, d, device="cuda") * 3).to(dtype)
    w = torch.rand(d, device="cuda") + 0.5
    y = R.rmsnorm(x, w, 1e-6, backend="cuda")
    ref = R.rmsnorm_forward_ref(x, w, 1e-6)
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(y.float(), ref.float(), atol=tol, rtol=tol)
    torch.library.opcheck(torch.ops.llmsys.rmsnorm, (x, w, 1e-6))
