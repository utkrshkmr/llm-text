import torch

from _labutil import load_impl
from llmsys.kvcache import KVCache
from llmsys.layers import GroupedQueryAttention as RefGQA, apply_rope as ref_rope, compute_rope_params
from llmsys.moe import MoEFeedForward
from llmsys.qwen3 import QWEN3_CONFIG_TINY, QWEN3_CONFIGS, Qwen3Model as RefQwen3
from llmsys.utils import count_parameters

lab = load_impl(__file__)
CFG = dict(QWEN3_CONFIG_TINY)


def test_rmsnorm():
    x = torch.randn(2, 3, 8) * 5
    w = torch.rand(8)
    y = lab.rms_norm(x, w)
    ref = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(y, ref, atol=1e-5)
    yb = lab.rms_norm(x.bfloat16(), w)
    assert yb.dtype == torch.bfloat16


def test_rope_matches_core_and_is_relative():
    cos, sin = lab.compute_rope(16, 10_000.0, 64)
    c2, s2 = compute_rope_params(16, 10_000.0, 64)
    assert torch.allclose(cos, c2, atol=1e-6) and torch.allclose(sin, s2, atol=1e-6)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    dots = []
    for m in [3, 10, 40]:
        qm = lab.apply_rope(q, cos, sin, offset=m + 5)
        kn = lab.apply_rope(k, cos, sin, offset=m)
        dots.append((qm * kn).sum())
    assert torch.allclose(dots[0], dots[1], atol=1e-4) and torch.allclose(dots[1], dots[2], atol=1e-4)
    x = torch.randn(2, 3, 5, 16)
    assert torch.allclose(lab.apply_rope(x, cos, sin, 7), ref_rope(x, cos, sin, 7), atol=1e-6)


def test_interleaved_equivalence():
    D = 16
    cos, sin = lab.compute_rope(D, 10_000.0, 32)
    p = lab.halves_to_interleaved_perm(D)
    x = torch.randn(1, 2, 6, D)
    a = lab.apply_rope(x, cos, sin, 3)[..., p]
    b = lab.apply_rope_interleaved(x[..., p], 10_000.0, 3)
    assert torch.allclose(a, b, atol=1e-5)


def test_context_extension():
    D, s = 16, 4.0
    cos, sin = lab.compute_rope(D, 10_000.0, 64)
    pc, ps = lab.position_interpolation_rope(D, 10_000.0, 256, s)
    assert torch.allclose(pc[40], cos[10], atol=1e-5)
    nb = lab.ntk_aware_base(10_000.0, s, D)
    f_old = 1 / 10_000.0 ** (torch.arange(0, D, 2).float() / D)
    f_new = 1 / nb ** (torch.arange(0, D, 2).float() / D)
    assert torch.isclose(f_new[0], f_old[0]) and torch.isclose(f_new[-1], f_old[-1] / s, rtol=1e-4)


def test_swiglu_shape():
    m = lab.SwiGLU(8, 20)
    assert m(torch.randn(2, 3, 8)).shape == (2, 3, 8)
    assert sum(p.numel() for p in m.parameters()) == 3 * 8 * 20


def test_gqa_matches_core_with_cache():
    torch.manual_seed(0)
    ref = RefGQA(32, 4, 2, 8, qk_norm=True); ref.layer_idx = 0
    mine = lab.GroupedQueryAttention(32, 4, 2, 8, qk_norm=True); mine.layer_idx = 0
    mine.load_state_dict(ref.state_dict())
    cos, sin = compute_rope_params(8, 10_000.0, 64)
    x = torch.randn(2, 9, 32)
    assert torch.allclose(mine(x, cos, sin), ref(x, cos, sin), atol=1e-5)
    c = KVCache(1)
    out = torch.cat([mine(x[:, :5], cos, sin, cache=c), mine(x[:, 5:], cos, sin, cache=c)], 1)
    assert torch.allclose(out, ref(x, cos, sin), atol=1e-5)
    assert c.k[0].shape[1] == 2  # cache stores KV *groups*, not query heads


def test_full_model_matches_core():
    torch.manual_seed(0)
    ref = RefQwen3(CFG).eval()
    mine = lab.Qwen3Model(CFG).eval()
    mine.load_state_dict(ref.state_dict())
    idx = torch.randint(0, CFG["vocab_size"], (2, 13))
    assert torch.allclose(mine(idx), ref(idx), atol=1e-5)
    assert lab.count_params(CFG) == count_parameters(ref)


def test_kv_budget():
    c = QWEN3_CONFIGS["0.6B"]
    # 2 * 28 layers * 8 kv heads * 128 dim * 2 bytes = 114,688 bytes per token
    assert lab.kv_cache_bytes(c, 1) == 114_688
    assert lab.kv_cache_bytes(c, 4096, batch=8) == 114_688 * 4096 * 8
    n = lab.max_concurrent_sequences(c, 24e9, 8192)
    assert 15 <= n <= 25
    assert 0.55e9 < lab.count_params(c) < 0.65e9


def test_moe_dispatch_and_aux_loss():
    torch.manual_seed(0)
    moe = MoEFeedForward(16, 24, n_experts=6, top_k=2)
    x = torch.randn(3, 7, 16)
    assert torch.allclose(lab.moe_dispatch_forward(moe, x), moe.forward_reference(x), atol=1e-5)
    E = 4
    probs = torch.full((8, E), 1.0 / E)
    idx = torch.arange(8).remainder(E)[:, None]
    assert torch.isclose(lab.load_balancing_loss(probs, idx, E), torch.tensor(1.0))
    idx_c, w, _ = lab.route_topk(torch.randn(5, E), 2)
    assert idx_c.shape == (5, 2) and torch.allclose(w.sum(-1), torch.ones(5))
