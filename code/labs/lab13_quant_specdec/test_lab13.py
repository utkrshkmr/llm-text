import copy

import torch

from _labutil import load_impl
from llmsys.generate import generate
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model

lab = load_impl(__file__)


def rel(a, b):
    return float((a.detach() - b.detach()).norm() / b.detach().norm())


# ------------------------------------------------------------------ Part A
def test_absmax_per_channel_beats_per_tensor():
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    w[3] *= 50                                   # one outlier row
    qt, st = lab.quantize_absmax(w, 8, dim=None)
    qc, sc = lab.quantize_absmax(w, 8, dim=-1)
    assert qt.dtype == torch.int8 and qc.abs().max() <= 127
    assert sc.shape == (32, 1)
    assert rel(lab.dequantize_absmax(qc, sc), w) < rel(lab.dequantize_absmax(qt, st), w)
    assert rel(lab.dequantize_absmax(qc, sc), w) < 0.01


def test_groupwise_int4_and_packing():
    torch.manual_seed(0)
    w = torch.randn(16, 128)
    q, s, z = lab.quantize_groupwise(w, 4, 32)
    assert q.dtype == torch.uint8 and int(q.max()) <= 15 and s.shape == (16, 4)
    err = rel(lab.dequantize_groupwise(q, s, z, 32), w)
    assert err < 0.12
    p = lab.pack_int4(q)
    assert p.shape == (16, 64) and p.dtype == torch.uint8
    assert torch.equal(lab.unpack_int4(p), q)
    # smaller groups -> smaller error
    q2, s2, z2 = lab.quantize_groupwise(w, 4, 8)
    assert rel(lab.dequantize_groupwise(q2, s2, z2, 8), w) < err


def test_quantized_model_outputs_close_and_smaller():
    torch.manual_seed(0)
    m = GPTModel(GPT_CONFIG_TINY).eval()
    idx = torch.randint(0, 512, (2, 12))
    ref = m(idx)
    m8 = lab.quantize_linear_layers(m, 8, 32)
    m4 = lab.quantize_linear_layers(m, 4, 32)
    assert rel(m8(idx), ref) < 0.01
    assert rel(m4(idx), ref) < 0.1
    b_fp = lab.linear_weight_bytes(m.trf_blocks)  # out_head is kept in fp32
    assert lab.linear_weight_bytes(m8.trf_blocks) < 0.4 * b_fp
    assert lab.linear_weight_bytes(m4.trf_blocks) < 0.25 * b_fp
    assert isinstance(m.trf_blocks[0].att.W_query, torch.nn.Linear)  # original untouched


# ------------------------------------------------------------------ Part B
def test_accept_or_resample_preserves_target_distribution():
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5, 0.2, 0.2, 0.1])
    q = torch.tensor([0.1, 0.6, 0.1, 0.2])
    counts = torch.zeros(4)
    n = 20000
    for _ in range(n):
        d = int(torch.multinomial(q, 1, generator=g))
        _, t = lab.accept_or_resample(p, q, d, g)
        counts[t] += 1
    assert torch.allclose(counts / n, p, atol=0.015)


def test_verify_drafts_structure():
    g = torch.Generator().manual_seed(0)
    V = 5
    onehot = lambda i: torch.nn.functional.one_hot(torch.tensor(i), V).float()
    ps = torch.stack([onehot(1), onehot(2), onehot(4)])
    qs = [onehot(1), onehot(3)]
    out = lab.verify_drafts(ps, qs, [1, 3], g)   # accept 1, reject 3 -> resample 2
    assert out == [1, 2]
    out = lab.verify_drafts(ps, [onehot(1), onehot(2)], [1, 2], g)
    assert out == [1, 2, 4]                      # all accepted + bonus token


def test_speculative_greedy_equals_target_greedy():
    torch.manual_seed(0)
    # untied embeddings, otherwise a random tiny model just echoes its last token
    target = Qwen3Model(dict(QWEN3_CONFIG_TINY, tie_embeddings=False)).eval()
    draft = lab.layer_skip_draft(target, 1)
    prompt = torch.randint(0, 512, (1, 6))
    out, calls = lab.speculative_generate(target, draft, prompt, 24, gamma=4)
    assert torch.equal(out, generate(target, prompt, 24))
    assert calls < 25                             # fewer target calls than tokens


def test_speculative_sampling_runs_and_respects_length():
    torch.manual_seed(0)
    target = Qwen3Model(QWEN3_CONFIG_TINY).eval()
    draft = lab.noisy_copy(target, 0.5)
    g = torch.Generator().manual_seed(1)
    out, _ = lab.speculative_generate(target, draft, torch.tensor([[1, 2, 3]]), 17,
                                      gamma=3, temperature=1.0, generator=g)
    assert out.shape == (1, 20)


def test_expected_tokens_per_round():
    assert abs(lab.expected_tokens_per_round(0.0, 4) - 1.0) < 1e-9
    assert abs(lab.expected_tokens_per_round(1.0, 4) - 5.0) < 1e-9
    assert abs(lab.expected_tokens_per_round(0.5, 2) - 1.75) < 1e-9


# ------------------------------------------------------------------ Part C
def test_common_prefix_len():
    assert lab.common_prefix_len([1, 2, 3, 4], [1, 2, 3, 9, 9]) == 3
    assert lab.common_prefix_len([], [1]) == 0
    assert lab.common_prefix_len([1, 2], [1, 2, 3]) == 2


def test_incremental_detokenizer_handles_split_utf8():
    from chat_common import IncrementalDetokenizer
    from llmsys.tokenizer import ByteBPETokenizer
    tok = ByteBPETokenizer.train("plain ascii text only " * 5, vocab_size=260)
    text = "café \U0001F680 ok"
    ids = tok.encode(text)
    d = IncrementalDetokenizer(tok)
    pieces = [d.push(i) for i in ids]
    assert "".join(pieces) == text
    assert all("�" not in p for p in pieces)
