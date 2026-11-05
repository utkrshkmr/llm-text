"""Acceptance tests for Project 1 (Chapter 42).

Skipped unless P1_IMPL points to your implementation of ``p1_template.py``:
    P1_IMPL=/path/to/p1_impl.py python -m pytest code/projects/p1_lm -q
"""
import collections
import importlib.util
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # code/ for llmsys

IMPL = os.environ.get("P1_IMPL")
pytestmark = pytest.mark.skipif(not IMPL, reason="set P1_IMPL=/path/to/your/p1_impl.py")
torch.set_num_threads(1)


@pytest.fixture(scope="module")
def p1():
    spec = importlib.util.spec_from_file_location("p1_impl", os.path.expanduser(IMPL))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p1_impl"] = mod
    spec.loader.exec_module(mod)
    return mod


CORPUS = ("the cat sat on the mat. the dog sat on the log. "
          "<|endoftext|>a cat and a dog met on the mat; they sat. ") * 30
SPECIAL = "<|endoftext|>"


# ----------------------------------------------------------------------------- tokenizer
def _expected_first_merge(text, specials):
    import regex
    pat = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    counts = collections.Counter()
    for seg in regex.split("|".join(map(regex.escape, specials)), text):
        for m in pat.finditer(seg):
            b = m.group().encode()
            for a, c in zip(b, b[1:]):
                counts[(bytes([a]), bytes([c]))] += 1
    best = max(counts.values())
    return max(p for p, c in counts.items() if c == best)


def test_bpe_structure_and_first_merge(p1):
    vocab, merges = p1.train_bpe(CORPUS, 300, [SPECIAL])
    assert len(vocab) == 256 + len(merges) + 1 <= 300      # stops early if nothing is left to merge
    assert all(vocab[i] == bytes([i]) for i in range(256))
    assert SPECIAL.encode() in vocab.values()
    for i, (a, b) in enumerate(merges):
        assert vocab[256 + i] == a + b                      # merge ids in creation order
    assert merges[0] == _expected_first_merge(CORPUS, [SPECIAL])
    assert p1.train_bpe(CORPUS, 300, [SPECIAL]) == (vocab, merges)   # deterministic


def test_bpe_special_tokens_are_hard_boundaries(p1):
    text = ("ab" + SPECIAL + "ba") * 50
    vocab, merges = p1.train_bpe(text, 270, [SPECIAL])
    learned = [vocab[i] for i in range(256, 256 + len(merges))]
    assert not any(b"<" in t or b"|" in t for t in learned)   # nothing learned from the token
    assert not any(t in (b"bb", b"aa") for t in learned)      # no merge across the boundary


def test_tokenizer_roundtrip_and_specials(p1):
    vocab, merges = p1.train_bpe(CORPUS, 320, [SPECIAL])
    tok = p1.Tokenizer(vocab, merges, [SPECIAL])
    for s in ["", "the cat sat", "héllo wörld 🙂 你好", "  spaces\n\nand\ttabs ",
              "a" + SPECIAL + "b" + SPECIAL]:
        assert tok.decode(tok.encode(s)) == s
    ids = tok.encode("cat" + SPECIAL + "dog")
    sid = [k for k, v in vocab.items() if v == SPECIAL.encode()][0]
    assert ids.count(sid) == 1
    assert len(tok.encode("the cat sat on the mat")) < len("the cat sat on the mat".encode())


def test_tokenizer_overlapping_specials(p1):
    vocab, merges = p1.train_bpe(CORPUS, 300, [SPECIAL])
    double = SPECIAL + SPECIAL
    tok = p1.Tokenizer(vocab, merges, [SPECIAL, double])
    ids = tok.encode("x" + double + "y" + SPECIAL)
    assert tok.decode(ids) == "x" + double + "y" + SPECIAL
    assert [tok.decode([i]) for i in ids if tok.decode([i]) in (SPECIAL, double)] == [double, SPECIAL]


def test_encode_iterable_matches_encode(p1, tmp_path):
    vocab, merges = p1.train_bpe(CORPUS, 320, [SPECIAL])
    tok = p1.Tokenizer(vocab, merges, [SPECIAL])
    text = CORPUS[:2000].replace(". ", ".\n") + "\n\nend  \n"
    path = tmp_path / "t.txt"
    path.write_text(text, encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        assert list(tok.encode_iterable(f)) == tok.encode(text)


def test_decode_replaces_invalid_utf8(p1):
    vocab, merges = p1.train_bpe(CORPUS, 300, [SPECIAL])
    tok = p1.Tokenizer(vocab, merges, [SPECIAL])
    assert tok.decode([0xE4]) == "�"


# ----------------------------------------------------------------------------- model pieces
def test_softmax_and_cross_entropy(p1):
    torch.manual_seed(0)
    x = torch.randn(3, 5, 7) * 20 + 1e4                       # large logits must not overflow
    torch.testing.assert_close(p1.softmax(x, -1), torch.softmax(x, -1))
    y = torch.randint(0, 7, (3, 5))
    torch.testing.assert_close(p1.cross_entropy(x, y), F.cross_entropy(x.flatten(0, 1), y.flatten()))


def test_rmsnorm(p1):
    from llmsys.layers import RMSNorm
    torch.manual_seed(0)
    ref = RMSNorm(16, eps=1e-6)
    with torch.no_grad():
        ref.scale.uniform_(0.5, 1.5)
    x = torch.randn(2, 3, 16)
    torch.testing.assert_close(p1.rmsnorm(x, ref.scale.detach(), 1e-6), ref(x))
    assert p1.rmsnorm(x.bfloat16(), ref.scale.detach(), 1e-6).dtype == torch.bfloat16


def test_rope_matches_reference_and_is_relative(p1):
    from llmsys.layers import apply_rope, compute_rope_params
    torch.manual_seed(0)
    x = torch.randn(2, 4, 9, 16)
    cos, sin = compute_rope_params(16, 10_000.0, 64)
    torch.testing.assert_close(p1.apply_rope(x, torch.arange(9), 10_000.0), apply_rope(x, cos, sin),
                               rtol=1e-5, atol=1e-5)
    q, k = torch.randn(16), torch.randn(16)
    def score(i, j):
        qi = p1.apply_rope(q.view(1, 1, 16), torch.tensor([i]))
        kj = p1.apply_rope(k.view(1, 1, 16), torch.tensor([j]))
        return float((qi * kj).sum())
    assert score(7, 3) == pytest.approx(score(14, 10), rel=1e-4, abs=1e-5)


def test_attention_matches_sdpa(p1):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 6, 8) for _ in range(3))
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    torch.testing.assert_close(p1.attention(q, k, v, mask),
                               F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
    torch.testing.assert_close(p1.attention(q, k, v), F.scaled_dot_product_attention(q, k, v))


TINY = dict(vocab_size=64, context_length=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)


def test_lm_shapes_causality_and_init_loss(p1):
    torch.manual_seed(0)
    m = p1.TransformerLM(**TINY)
    idx = torch.randint(0, 64, (2, 16))
    out = m(idx)
    assert out.shape == (2, 16, 64)
    idx2 = idx.clone(); idx2[:, 10:] = (idx2[:, 10:] + 1) % 64
    torch.testing.assert_close(m(idx2)[:, :10], out[:, :10])          # no peeking at the future
    loss = F.cross_entropy(out.flatten(0, 1), torch.randint(0, 64, (32,)))
    assert abs(loss.item() - math.log(64)) < 0.7                       # near-uniform at init


def test_lm_overfits_one_batch(p1):
    torch.manual_seed(0)
    m = p1.TransformerLM(**TINY)
    opt = p1.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    x = torch.randint(0, 64, (4, 16)); y = torch.randint(0, 64, (4, 16))
    for _ in range(200):
        loss = p1.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.3


# ----------------------------------------------------------------------------- training
def test_adamw_matches_torch(p1):
    def run(cls):
        g = torch.Generator().manual_seed(0)
        W = torch.nn.Parameter(torch.randn(5, 4, generator=g)); X = torch.randn(16, 4, generator=g)
        opt = cls([W], lr=0.05, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        for _ in range(20):
            opt.zero_grad(); ((X @ W.T) ** 2).mean().backward(); opt.step()
        return W.detach()
    torch.testing.assert_close(run(p1.AdamW), run(torch.optim.AdamW), rtol=1e-5, atol=1e-6)


def test_lr_schedule(p1):
    from llmsys.optim import WarmupCosine
    ref = WarmupCosine(peak=1e-3, total=100, warmup=10, floor=1e-4)
    for s in [0, 5, 9, 10, 37, 99, 100, 150]:
        assert p1.lr_schedule(s, 1e-3, 1e-4, 10, 100) == pytest.approx(ref(s), rel=1e-9, abs=1e-12)


def test_clip_grad_norm(p1):
    torch.manual_seed(0)
    a = [torch.nn.Parameter(torch.randn(6, 6)) for _ in range(3)]
    b = [torch.nn.Parameter(t.detach().clone()) for t in a]
    for x, y in zip(a, b):
        x.grad = torch.randn_like(x) * 4; y.grad = x.grad.clone()
    n = p1.clip_grad_norm(a, 1.0)
    assert n == pytest.approx(float(torch.nn.utils.clip_grad_norm_(b, 1.0)), rel=1e-5)
    for x, y in zip(a, b):
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)


def test_get_batch(p1):
    toks = np.arange(500, dtype=np.uint16)
    x, y = p1.get_batch(toks, 8, 12, "cpu", np.random.default_rng(0))
    assert x.shape == (8, 12) and x.dtype == torch.int64
    assert torch.equal(y, x + 1) and int(x.max()) < 499
    x2, _ = p1.get_batch(toks, 8, 12, "cpu", np.random.default_rng(0))
    assert torch.equal(x, x2)


def test_checkpoint_roundtrip(p1, tmp_path):
    torch.manual_seed(0)
    m = p1.TransformerLM(**TINY); opt = p1.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 8))
    for _ in range(3):
        opt.zero_grad(); p1.cross_entropy(m(x), x).backward(); opt.step()
    path = str(tmp_path / "ck.pt")
    p1.save_checkpoint(path, m, opt, 3)
    torch.manual_seed(1)
    m2 = p1.TransformerLM(**TINY); opt2 = p1.AdamW(m2.parameters(), lr=1e-3)
    assert p1.load_checkpoint(path, m2, opt2) == 3
    for o in (opt, opt2):
        o.zero_grad()
    p1.cross_entropy(m(x), x).backward(); opt.step()
    p1.cross_entropy(m2(x), x).backward(); opt2.step()
    for a, b in zip(m.parameters(), m2.parameters()):
        torch.testing.assert_close(a, b)
