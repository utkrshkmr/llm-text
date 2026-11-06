"""Tests for Lab 29: C++ for PyTorch.

The compiled parts (``torch.ops.llmsys.bpe_*`` and ``silu_mul``) come from the installed extension
or, with ``LLMSYS_JIT=1``, from a just-in-time build of their source files; without either those
tests skip with a reason. The CMake tests skip when cmake/ninja/c++ are missing.
"""
import os
import random

import pytest
import torch

from _labutil import load_impl

from llmsys.tokenizer import ByteBPETokenizer

lab = load_impl(__file__)

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "encoder_sample.txt")
HAS_CUDA = torch.cuda.is_available()
JIT = os.environ.get("LLMSYS_JIT", "0") == "1"


def _built(fn):
    try:
        return bool(fn())
    except NotImplementedError:
        return False


BPE_BUILT = _built(lab.build_bpe)
SILU_BUILT = _built(lab.build_silu_mul)
cpp_bpe = pytest.mark.skipif(not BPE_BUILT, reason="compiled bpe ops not available (set LLMSYS_JIT=1 or build llmsys._C)")
cpp_silu = pytest.mark.skipif(not SILU_BUILT, reason="compiled silu_mul not available (set LLMSYS_JIT=1 or build llmsys._C)")
gpu = pytest.mark.skipif(not (SILU_BUILT and HAS_CUDA and torch._C._dispatch_has_kernel_for_dispatch_key(
    "llmsys::silu_mul", "CUDA")), reason="needs a CUDA GPU and the CUDA build of silu_mul")
needs_cmake = pytest.mark.skipif(not lab.cmake_available(), reason="cmake, ninja and a C++ compiler are required")


@pytest.fixture(scope="module")
def corpus():
    text = open(SAMPLE, encoding="utf-8").read()
    return text + " Ünïcödé, emoji 🚀 and numbers 12345 -- all bytes are tokens.\n" + text[:1500]


@pytest.fixture(scope="module")
def tok(corpus):
    return ByteBPETokenizer.train(corpus, 420)


def random_byte_chunks(n: int, seed: int = 0):
    """Chunks of raw random bytes (not valid UTF-8): the encoder must treat every byte as a token."""
    rng = random.Random(seed)
    return [bytes(rng.randrange(256) for _ in range(rng.randint(0, 24))) for _ in range(n)]


# ----------------------------------------------------------------------------- Task 1
def test_merges_table_sorted_by_id(tok):
    table = lab.merges_table(tok.merges)
    assert table.dtype == torch.int64 and table.shape == (len(tok.merges), 3)
    ids = table[:, 2].tolist()
    assert ids == sorted(ids) and ids[0] == 256
    assert all(tok.merges[(int(a), int(b))] == int(i) for a, b, i in table.tolist())


def test_chunks_to_tensors_roundtrip():
    chunks = [b"ab", b"", b"xyz", b"\x00\xff"]
    data, offsets = lab.chunks_to_tensors(chunks)
    assert data.dtype == torch.uint8 and offsets.dtype == torch.int64
    assert offsets.tolist() == [0, 2, 2, 5, 7]
    assert [bytes(data[offsets[i]:offsets[i + 1]].tolist()) for i in range(4)] == chunks
    d0, o0 = lab.chunks_to_tensors([])
    assert d0.numel() == 0 and o0.tolist() == [0]


def test_python_backend_matches_tokenizer(tok, corpus):
    assert lab.encode(corpus, tok.merges, backend="python") == tok.encode(corpus)
    ids = list(b"lowest lower low")
    assert lab.encode_chunk_python(ids, tok.merges) == tok._encode_chunk("lowest lower low")


def test_backend_switch(tok):
    usable = lab.available_backends()
    assert "python" in usable
    assert ("cpp" in usable) == BPE_BUILT
    with pytest.raises(RuntimeError):
        lab.encode("abc", tok.merges, backend="triton")
    if not BPE_BUILT:
        with pytest.raises(RuntimeError):
            lab.encode("abc", tok.merges, backend="cpp")


# ----------------------------------------------------------------------------- Task 2
@cpp_bpe
def test_cpp_encode_matches_tokenizer_on_text(tok, corpus):
    assert lab.encode(corpus, tok.merges, backend="cpp") == tok.encode(corpus)
    assert lab.encode(corpus, tok.merges, backend="auto") == tok.encode(corpus)


@cpp_bpe
def test_cpp_encode_matches_on_random_bytes(tok):
    chunks = random_byte_chunks(300)
    data, offsets = lab.chunks_to_tensors(chunks)
    table = lab.merges_table(tok.merges)
    ids, out_off = lab.encode_cpp(data, offsets, table)
    ref_ids, ref_off = lab.encode_python(data, offsets, table)
    assert torch.equal(ids, ref_ids) and torch.equal(out_off, ref_off)
    lengths = [len(lab.encode_chunk_python(list(c), tok.merges)) for c in chunks]
    assert out_off.tolist() == [0] + torch.tensor(lengths).cumsum(0).tolist()


@cpp_bpe
def test_cpp_encode_batch_equals_single_chunks(tok, corpus):
    chunks = lab.pretokenize(corpus)[:200]
    data, offsets = lab.chunks_to_tensors(chunks)
    table = lab.merges_table(tok.merges)
    ids, out_off = lab.encode_cpp(data, offsets, table)
    for c, chunk in enumerate(chunks):
        single = torch.ops.llmsys.bpe_encode(torch.frombuffer(bytearray(chunk), dtype=torch.uint8), table) \
            if chunk else torch.empty(0, dtype=torch.int64)
        assert torch.equal(ids[out_off[c]:out_off[c + 1]], single)


@cpp_bpe
def test_cpp_encode_rejects_bad_offsets(tok):
    table = lab.merges_table(tok.merges)
    data = torch.zeros(10, dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        lab.encode_cpp(data, torch.tensor([0, 11]), table)          # past the end
    with pytest.raises(RuntimeError):
        lab.encode_cpp(data, torch.tensor([0, 6, 4, 10]), table)    # decreasing


@cpp_bpe
def test_cpp_encode_throughput(tok, corpus):
    text = corpus * 20
    py = lab.encode_throughput(text, tok.merges, "python", repeats=1)
    cpp = lab.encode_throughput(text, tok.merges, "cpp", repeats=3)
    print(f"\nencode throughput: python {py:.2f} MB/s, cpp {cpp:.2f} MB/s ({cpp / py:.1f}x)")
    assert cpp > py


# ----------------------------------------------------------------------------- Task 3
@cpp_bpe
def test_cpp_train_matches_tokenizer(corpus):
    ref = ByteBPETokenizer.train(corpus, 500)
    out = lab.train(corpus, 500)
    assert out.merges == ref.merges and out.special_tokens == ref.special_tokens
    assert out.encode(corpus) == ref.encode(corpus)


@cpp_bpe
def test_cpp_train_tie_breaking_and_early_stop():
    # Many equal counts: the tie-break (smaller first id, then smaller second id) must match.
    text = "ab ba ab ba cd dc cd dc ef fe ef fe"
    ref = ByteBPETokenizer.train(text, 300)
    out = lab.train(text, 300)
    assert out.merges == ref.merges
    # Fewer pairs than requested merges: stop early with the same number of merges.
    ref2 = ByteBPETokenizer.train("aaaa", 400)
    out2 = lab.train("aaaa", 400)
    assert out2.merges == ref2.merges and out2.special_tokens == ref2.special_tokens


# ----------------------------------------------------------------------------- Task 4
def test_silu_mul_backward_manual_matches_autograd():
    torch.manual_seed(0)
    g = torch.randn(3, 17, dtype=torch.float64, requires_grad=True)
    u = torch.randn(3, 17, dtype=torch.float64, requires_grad=True)
    grad = torch.randn(3, 17, dtype=torch.float64)
    y = lab.silu_mul_unfused(g, u)
    dg, du = torch.autograd.grad(y, (g, u), grad)
    mg, mu = lab.silu_mul_backward_manual(grad, g.detach(), u.detach())
    assert torch.allclose(dg, mg, atol=1e-12) and torch.allclose(du, mu, atol=1e-12)


@cpp_silu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(5, 33), (1, 1), (2, 3, 1027), (0,)])
def test_silu_mul_fused_matches_reference(dtype, shape):
    torch.manual_seed(0)
    g, u = (torch.randn(shape) * 4).to(dtype), (torch.randn(shape) * 4).to(dtype)
    y = lab.silu_mul_fused(g, u)
    ref = lab.silu_mul_unfused(g.float(), u.float())
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert y.dtype == dtype and y.shape == g.shape
    assert torch.allclose(y.float(), ref, atol=tol, rtol=tol)


@cpp_silu
def test_silu_mul_gradcheck_fp64():
    torch.manual_seed(0)
    g = torch.randn(4, 9, dtype=torch.float64, requires_grad=True)
    u = torch.randn(4, 9, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lab.silu_mul_fused, (g, u))
    y = lab.silu_mul_fused(g, u)
    assert y.grad_fn is not None and "SiluMul" in type(y.grad_fn).__name__   # the node built in C++


@cpp_silu
def test_silu_mul_backward_matches_manual_in_bf16():
    torch.manual_seed(0)
    g = (torch.randn(6, 64) * 3).bfloat16().requires_grad_(True)
    u = (torch.randn(6, 64) * 3).bfloat16().requires_grad_(True)
    grad = torch.randn(6, 64).bfloat16()
    lab.silu_mul_fused(g, u).backward(grad)
    mg, mu = lab.silu_mul_backward_manual(grad.float(), g.detach().float(), u.detach().float())
    assert torch.allclose(g.grad.float(), mg, atol=0.15, rtol=2e-2)
    assert torch.allclose(u.grad.float(), mu, atol=0.15, rtol=2e-2)


@cpp_silu
def test_silu_mul_opcheck():
    g, u = torch.randn(2, 7), torch.randn(2, 7)
    torch.library.opcheck(torch.ops.llmsys.silu_mul, (g, u))
    torch.library.opcheck(torch.ops.llmsys.silu_mul_backward, (torch.randn(2, 7), g, u))


@cpp_silu
def test_silu_mul_compile_fullgraph():
    torch._dynamo.reset()
    fn = lab.silu_mul_compiled()
    g, u = torch.randn(3, 40), torch.randn(3, 40)
    assert torch.allclose(fn(g, u), lab.silu_mul_unfused(g, u), atol=1e-6)
    g2, u2 = torch.randn(3, 40, requires_grad=True), torch.randn(3, 40, requires_grad=True)
    fn(g2, u2).sum().backward()                      # backward through the compiled graph
    mg, mu = lab.silu_mul_backward_manual(torch.ones(3, 40), g2.detach(), u2.detach())
    assert torch.allclose(g2.grad, mg, atol=1e-5) and torch.allclose(u2.grad, mu, atol=1e-5)


@cpp_silu
def test_silu_mul_shape_mismatch_raises():
    with pytest.raises(RuntimeError):
        lab.silu_mul_fused(torch.randn(2, 3), torch.randn(3, 2))


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_silu_mul_cuda(dtype):
    torch.manual_seed(0)
    g = (torch.randn(3, 5, 1031, device="cuda") * 4).to(dtype).requires_grad_(True)
    u = (torch.randn(3, 5, 1031, device="cuda") * 4).to(dtype).requires_grad_(True)
    y = lab.silu_mul_fused(g, u)
    ref = lab.silu_mul_unfused(g.detach().float(), u.detach().float())
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(y.float(), ref, atol=tol, rtol=tol)
    grad = torch.randn_like(y)
    y.backward(grad)
    mg, mu = lab.silu_mul_backward_manual(grad.float(), g.detach().float(), u.detach().float())
    tol_b = 1e-4 if dtype == torch.float32 else 0.2
    assert torch.allclose(g.grad.float(), mg, atol=tol_b, rtol=2e-2)
    assert torch.allclose(u.grad.float(), mu, atol=tol_b, rtol=2e-2)
    torch.library.opcheck(torch.ops.llmsys.silu_mul, (g.detach(), u.detach()))
    gd = torch.randn(3, 7, device="cuda", dtype=torch.float64, requires_grad=True)
    ud = torch.randn(3, 7, device="cuda", dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lab.silu_mul_fused, (gd, ud))


# ----------------------------------------------------------------------------- Task 5 and 6
@needs_cmake
def test_pybind11_example_builds_and_matches(tok, corpus):
    build_dir = lab.build_example("pybind11_bpe")
    enc = lab.load_pybind_encoder(build_dir, tok.merges)
    chunks = lab.pretokenize(corpus)
    assert [t for c in chunks for t in enc.encode(c)] == tok.encode(corpus)
    assert [t for seq in enc.encode_batch(chunks) for t in seq] == tok.encode(corpus)
    assert enc.num_merges == len(tok.merges)


@needs_cmake
def test_ctypes_example_builds_and_matches(tok, corpus):
    build_dir = lab.build_example("ctypes_bpe")
    enc = lab.load_ctypes_encoder(build_dir, tok.merges)
    chunks = lab.pretokenize(corpus)
    assert [t for c in chunks for t in enc.encode(c)] == tok.encode(corpus)
    assert enc.encode_batch(chunks) == tok.encode(corpus)
    assert enc.encode_batch(random_byte_chunks(50)) == [t for c in random_byte_chunks(50)
                                                        for t in lab.encode_chunk_python(list(c), tok.merges)]


@needs_cmake
def test_nanobind_example_builds_and_matches(tok, corpus):
    pytest.importorskip("nanobind", reason="stretch task: pip install nanobind")
    build_dir = lab.build_example("nanobind_bpe")
    enc = lab.load_nanobind_encoder(build_dir, tok.merges)
    chunks = lab.pretokenize(corpus)
    assert [t for c in chunks for t in enc.encode(c)] == tok.encode(corpus)
    assert [t for seq in enc.encode_batch(chunks) for t in seq] == tok.encode(corpus)


# ----------------------------------------------------------------------------- Extra: load_inline
@pytest.mark.skipif(not JIT, reason="compiles C++ at test time: set LLMSYS_JIT=1")
def test_load_inline_pair_count():
    mod = lab.build_inline()
    ids = torch.tensor([1, 1, 1, 2, 1, 2, 2, 1, 2])
    assert mod.pair_count(ids, 1, 2) == 3 and mod.pair_count(ids, 1, 1) == 1 and mod.pair_count(ids, 9, 9) == 0
    with pytest.raises(RuntimeError):
        mod.pair_count(ids.float(), 1, 2)
