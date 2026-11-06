import math

import torch

from _labutil import load_impl
from llmsys.layers import MultiHeadAttention as CoreMHA

lab = load_impl(__file__)


def toy_inputs():
    # six tokens, 3-d embeddings (the chapter's running example)
    return torch.tensor([[0.40, 0.10, 0.80], [0.60, 0.90, 0.20], [0.50, 0.80, 0.70],
                         [0.10, 0.30, 0.90], [0.90, 0.20, 0.10], [0.20, 0.70, 0.40]])


def test_simple_attention():
    x = toy_inputs()
    ctx, w = lab.simple_attention(x)
    assert w.shape == (6, 6) and torch.allclose(w.sum(-1), torch.ones(6))
    manual = sum(torch.softmax(x[2] @ x.T, 0)[j] * x[j] for j in range(6))
    assert torch.allclose(ctx[2], manual, atol=1e-6)


def test_self_attention_matches_formula():
    torch.manual_seed(0)
    sa = lab.SelfAttention(3, 2)
    x = toy_inputs()
    q, k, v = sa.W_query(x), sa.W_key(x), sa.W_value(x)
    ref = torch.softmax(q @ k.T / math.sqrt(2), -1) @ v
    assert torch.allclose(sa(x), ref, atol=1e-6)


def test_causal_masks_agree():
    torch.manual_seed(0)
    s = torch.randn(5, 5)
    a, b = lab.causal_softmax(s), lab.mask_after_softmax(s)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.all(torch.triu(a, 1) == 0)
    assert torch.allclose(a.sum(-1), torch.ones(5))


def test_causal_attention_is_causal():
    torch.manual_seed(0)
    ca = lab.CausalAttention(3, 4, context_length=8, dropout=0.0).eval()
    x = torch.randn(2, 6, 3)
    y1 = ca(x)
    x2 = x.clone()
    x2[:, 4:] += 5.0  # perturb the future
    y2 = ca(x2)
    assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-6)
    assert not torch.allclose(y1[:, 4:], y2[:, 4:])


def test_dropout_only_in_training():
    torch.manual_seed(0)
    ca = lab.CausalAttention(3, 4, context_length=8, dropout=0.5)
    x = torch.randn(1, 6, 3)
    ca.eval()
    assert torch.equal(ca(x), ca(x))
    ca.train()
    assert not torch.equal(ca(x), ca(x))


def test_wrapper_shape():
    torch.manual_seed(0)
    w = lab.MultiHeadAttentionWrapper(3, 2, 8, 0.0, num_heads=3)
    assert w(torch.randn(2, 5, 3)).shape == (2, 5, 6)


def test_mha_matches_core_implementation():
    torch.manual_seed(0)
    mine = lab.MultiHeadAttention(16, 32, 12, 0.0, num_heads=4).eval()
    core = CoreMHA(16, 32, 12, 0.0, num_heads=4).eval()
    core.load_state_dict(mine.state_dict())
    x = torch.randn(3, 10, 16)
    assert torch.allclose(mine(x), core(x), atol=1e-5)


def test_mha_one_head_equals_causal_attention():
    torch.manual_seed(0)
    mha = lab.MultiHeadAttention(5, 4, 8, 0.0, num_heads=1).eval()
    ca = lab.CausalAttention(5, 4, 8, 0.0).eval()
    for n in ["W_query", "W_key", "W_value"]:
        getattr(ca, n).load_state_dict(getattr(mha, n).state_dict())
    x = torch.randn(2, 7, 5)
    assert torch.allclose(mha(x), mha.out_proj(ca(x)), atol=1e-6)


def test_online_softmax():
    torch.manual_seed(0)
    x = torch.randn(4, 37) * 10
    assert torch.allclose(lab.online_softmax(x, 5), torch.softmax(x, -1), atol=1e-6)


def test_tiled_attention_matches_naive():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 29, 8, dtype=torch.float64) for _ in range(3))
    for causal in (True, False):
        s = q @ k.transpose(-2, -1) / math.sqrt(8)
        if causal:
            s = s.masked_fill(torch.triu(torch.ones(29, 29, dtype=torch.bool), 1), float("-inf"))
        ref = torch.softmax(s, -1) @ v
        out = lab.tiled_attention(q, k, v, causal=causal, block_q=8, block_k=5)
        assert torch.allclose(out, ref, atol=1e-10)


def test_score_matrix_bytes():
    assert lab.score_matrix_bytes(1, 32, 8192, 2) == 32 * 8192 * 8192 * 2
