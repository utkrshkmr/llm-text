"""Tests for Lab 9. Run: LAB_IMPL=solution python -m pytest -q labs/lab09_attnalt"""
import pytest
import torch

from _labutil import load_impl
from llmsys import linear_attn as ref
from llmsys.ssm import MambaBlock, scan_sequential

lab = load_impl(__file__)
B, H, T, DK, DV = 2, 3, 29, 6, 5


def _qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = ref.elu_feature_map(torch.randn(B, H, T, DK, generator=g, dtype=torch.float64))
    k = ref.elu_feature_map(torch.randn(B, H, T, DK, generator=g, dtype=torch.float64))
    v = torch.randn(B, H, T, DV, generator=g, dtype=torch.float64)
    return q, k, v


GAMMAS = [None, torch.tensor([0.8, 0.9, 0.99], dtype=torch.float64),
          torch.rand(B, H, T, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.5 + 0.5]


@pytest.mark.parametrize("gi", [0, 1, 2])
def test_three_forms_agree(gi):
    q, k, v = _qkv()
    g = GAMMAS[gi]
    a = lab.linear_attention_quadratic(q, k, v, g)
    b, S = lab.linear_attention_recurrent(q, k, v, g)
    assert torch.allclose(a, ref.linear_attention_quadratic(q, k, v, g), atol=1e-10)
    assert torch.allclose(a, b, atol=1e-10)
    for c in (1, 4, 7, 64):
        assert torch.allclose(a, lab.linear_attention_chunked(q, k, v, g, chunk=c), atol=1e-10)
    assert S.shape == (B, H, DK, DV)


def test_recurrent_state_continues():
    q, k, v = _qkv()
    y_full, _ = lab.linear_attention_recurrent(q, k, v)
    y1, S1 = lab.linear_attention_recurrent(q[..., :10, :], k[..., :10, :], v[..., :10, :])
    y2, _ = lab.linear_attention_recurrent(q[..., 10:, :], k[..., 10:, :], v[..., 10:, :], state=S1)
    assert torch.allclose(y_full, torch.cat([y1, y2], 2), atol=1e-10)


def test_delta_rule_matches_reference_and_stores_exactly():
    q, k, v = _qkv(3)
    k = torch.nn.functional.normalize(k, dim=-1)
    beta = torch.rand(B, H, T, dtype=torch.float64)
    gam = GAMMAS[2]
    y_ref, S_ref = ref.delta_rule_recurrent(q, k, v, beta, gam)
    S = torch.zeros(B, H, DK, DV, dtype=torch.float64)
    ys = []
    for t in range(T):
        y, S = lab.delta_rule_step(S, q[..., t, :], k[..., t, :], v[..., t, :], beta[..., t], gam[..., t])
        ys.append(y)
    assert torch.allclose(torch.stack(ys, 2), y_ref, atol=1e-10) and torch.allclose(S, S_ref, atol=1e-10)
    # orthonormal keys, beta = 1, no decay: perfect associative memory, even when overwriting
    E = torch.eye(4, dtype=torch.float64)
    S = torch.zeros(1, 1, 4, 2, dtype=torch.float64)
    vals = torch.randn(6, 2, dtype=torch.float64)
    keys = [0, 1, 2, 3, 1, 2]                     # keys 1 and 2 are written twice
    for i, kk in enumerate(keys):
        _, S = lab.delta_rule_step(S, E[kk][None, None], E[kk][None, None], vals[i][None, None],
                                   torch.ones(1, 1, dtype=torch.float64))
    assert torch.allclose(S[0, 0, 1], vals[4]) and torch.allclose(S[0, 0, 0], vals[0])


def test_associative_scan():
    g = torch.Generator().manual_seed(0)
    for T_ in (1, 2, 7, 64, 100):
        a, b = torch.rand(3, T_, 4, generator=g), torch.randn(3, T_, 4, generator=g)
        assert torch.allclose(lab.associative_scan(a, b), scan_sequential(a, b), atol=1e-5)


def test_selective_ssm_step_matches_block():
    torch.manual_seed(0)
    m = MambaBlock(8, d_state=4)
    u = torch.randn(2, 11, 8)
    x, z = m.in_proj(u).chunk(2, -1)
    xc = torch.nn.functional.silu(m.conv(x.transpose(1, 2))[..., :11].transpose(1, 2))
    dt, Bm, Cm = m._ssm_params(xc)
    A = -torch.exp(m.A_log)
    h = torch.zeros(2, m.E, m.N)
    ys = []
    for t in range(11):
        y, h = lab.selective_ssm_step(h, xc[:, t], dt[:, t], A, Bm[:, t], Cm[:, t], m.D)
        ys.append(y)
    y = torch.stack(ys, 1)
    expect = m.out_proj(y * torch.nn.functional.silu(z))
    assert torch.allclose(expect, m(u), atol=1e-5)


def test_sliding_window_and_memory():
    m = lab.sliding_window_mask(6, 3)
    assert torch.equal(m, ref.sliding_window_mask(6, 3))
    assert m.sum().item() == 1 + 2 + 3 + 3 + 3 + 3
    kw = dict(n_layers=28, n_heads=16, n_kv_heads=8, head_dim=128)
    full = lab.decode_cache_bytes("full", 32768, **kw)
    assert full == 2 * 28 * 8 * 128 * 32768 * 2
    assert lab.decode_cache_bytes("swa", 32768, window=4096, **kw) == full / 8
    lin = lab.decode_cache_bytes("linear", 10, **kw)
    assert lin == lab.decode_cache_bytes("linear", 10 ** 6, **kw) == 28 * 16 * 128 * 128 * 2
    hyb = lab.decode_cache_bytes("hybrid", 32768, attn_every=4, **kw)
    assert hyb == 2 * 7 * 8 * 128 * 32768 * 2 + 21 * 16 * 128 * 128 * 2
