"""Lab 13 tests: routers and balancing, capacity, grouped GEMM, upcycling, expert parallelism.

    LAB_IMPL=starter  python -m pytest -q labs/lab13_moe
    LAB_IMPL=solution python -m pytest -q labs/lab13_moe

Everything runs on CPU: the Triton grouped GEMM through Triton's interpreter (TRITON_INTERPRET=1,
set before triton is imported) and expert parallelism on two gloo processes. Tests marked ``gpu``
need CUDA and are skipped otherwise.
"""
import math
import os

import pytest
import torch

HAS_CUDA = torch.cuda.is_available()
if not HAS_CUDA:
    os.environ.setdefault("TRITON_INTERPRET", "1")

from _labutil import load_impl  # noqa: E402
from llmsys import moe as moe6  # noqa: E402
from llmsys.dist.expert_parallel import ExpertParallelMoE  # noqa: E402
from llmsys.dist.launch import run_distributed  # noqa: E402
from llmsys.kernels import grouped_gemm as gg  # noqa: E402
from llmsys.layers import SwiGLUFeedForward  # noqa: E402
from llmsys.moe_layer import Experts, MoELayer, MoEShape, moe_params  # noqa: E402

lab = load_impl(__file__)
gpu = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")
DEV = "cuda" if HAS_CUDA else "cpu"       # Triton kernels need device tensors unless interpreted


def _logits(n=64, e=8, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, e, generator=g) * scale


# --------------------------------------------------------------------------- Task 1
def test_softmax_router_matches_chapter6():
    logits = _logits()
    r = lab.topk_softmax_router(logits, 2)
    ref = moe6.route_topk(logits, 2)
    assert torch.equal(r.topk_idx, ref.topk_idx)
    assert torch.allclose(r.topk_weight, ref.topk_weight, atol=1e-6)
    assert torch.allclose(r.topk_weight.sum(-1), torch.ones(64), atol=1e-6)
    raw = lab.topk_softmax_router(logits, 2, normalize=False)
    assert torch.allclose(raw.topk_weight, torch.gather(ref.probs, 1, ref.topk_idx), atol=1e-6)
    assert r.topk_weight.dtype == torch.float32 and r.logits.dtype == torch.float32


def test_sigmoid_router_and_bias_selects_only():
    logits = _logits()
    s = torch.sigmoid(logits)
    r = lab.topk_sigmoid_router(logits, 3)
    assert torch.equal(r.topk_idx.sort(-1).values, s.topk(3, -1).indices.sort(-1).values)
    assert torch.allclose(r.topk_weight.sum(-1), torch.ones(64), atol=1e-6)
    assert torch.allclose(r.scores, s)
    # a large bias on expert 5 forces it into every token's set, but the combine weights still
    # come from the *unbiased* affinities
    bias = torch.zeros(8)
    bias[5] = 10.0
    rb = lab.topk_sigmoid_router(logits, 3, bias=bias)
    assert (rb.topk_idx == 5).any(-1).all()
    slot = (rb.topk_idx == 5).float().argmax(-1)
    w_raw = lab.topk_sigmoid_router(logits, 3, normalize=False, bias=bias).topk_weight
    assert torch.allclose(w_raw.gather(1, slot[:, None]).squeeze(1), s[:, 5], atol=1e-6)
    # bf16 input: routing identical to fp32 routing of the same (bf16-valued) logits
    rb16 = lab.topk_sigmoid_router(logits.to(torch.bfloat16), 3)
    assert torch.equal(rb16.topk_idx, lab.topk_sigmoid_router(logits.to(torch.bfloat16).float(), 3).topk_idx)


# --------------------------------------------------------------------------- Task 2
def test_group_limited_router():
    logits = _logits(n=128, e=16, seed=1)
    r = lab.group_limited_router(logits, 4, n_groups=4, topk_groups=2)
    groups = r.topk_idx // 4
    assert all(torch.unique(g).numel() <= 2 for g in groups)
    assert torch.allclose(r.topk_weight.sum(-1), torch.ones(128), atol=1e-6)
    # every token picks the two best groups by the sum of their two largest affinities
    s = torch.sigmoid(logits).view(128, 4, 4)
    gscore = s.topk(2, -1).values.sum(-1)
    best = gscore.topk(2, -1).indices.sort(-1).values
    for i in range(128):
        assert set(groups[i].tolist()) <= set(best[i].tolist())
    # with all groups allowed it is the plain sigmoid router
    full = lab.group_limited_router(logits, 4, n_groups=4, topk_groups=4)
    plain = lab.topk_sigmoid_router(logits, 4)
    assert torch.equal(full.topk_idx.sort(-1).values, plain.topk_idx.sort(-1).values)


# --------------------------------------------------------------------------- Task 3
def test_expert_choice_router():
    logits = _logits(n=40, e=5, seed=2)
    r = lab.expert_choice_router(logits, capacity=8)
    assert r.token_idx.shape == (5, 8) and r.weight.shape == (5, 8)
    p = torch.softmax(logits, -1)
    for e in range(5):
        expected = p[:, e].topk(8).indices
        assert torch.equal(r.token_idx[e].sort().values, expected.sort().values)
        assert torch.allclose(r.weight[e], p[r.token_idx[e], e])
    # load is exactly C per expert by construction, whatever the logits
    skew = logits.clone()
    skew[:, 0] += 50.0
    assert lab.expert_choice_router(skew, 8).token_idx.shape == (5, 8)


# --------------------------------------------------------------------------- Task 4
def test_aux_loss_on_crafted_inputs():
    E = 4
    # perfectly balanced assignments + uniform scores -> 1
    idx = torch.arange(16).remainder(E)[:, None]
    scores = torch.full((16, E), 0.25)
    assert math.isclose(lab.load_balancing_loss(scores, idx, E).item(), 1.0, abs_tol=1e-6)
    # total collapse: every token to expert 0 with all its mass -> E
    idx0 = torch.zeros(16, 1, dtype=torch.long)
    onehot = torch.zeros(16, E)
    onehot[:, 0] = 1.0
    assert math.isclose(lab.load_balancing_loss(onehot, idx0, E).item(), float(E), abs_tol=1e-6)
    # sigmoid scores (not summing to one) are renormalised per token
    sig = torch.full((16, E), 0.9)
    assert math.isclose(lab.load_balancing_loss(sig, idx, E).item(), 1.0, abs_tol=1e-6)
    # agrees with the Chapter 9 version on softmax scores
    logits = _logits()
    r = lab.topk_softmax_router(logits, 2)
    ours = lab.load_balancing_loss(r.scores, r.topk_idx, 8)
    ref = moe6.load_balancing_loss(r.scores, r.topk_idx, 8)
    assert torch.allclose(ours, ref, atol=1e-6)
    # gradient flows through the scores only
    sc = r.scores.clone().requires_grad_(True)
    lab.load_balancing_loss(sc, r.topk_idx, 8).backward()
    assert sc.grad is not None and torch.isfinite(sc.grad).all()


def test_router_z_loss():
    E = 8
    z = lab.router_z_loss(torch.zeros(10, E))
    assert math.isclose(z.item(), math.log(E) ** 2, rel_tol=1e-6)
    big = torch.zeros(10, E)
    big[:, 0] = 20.0
    assert lab.router_z_loss(big).item() > 390.0           # (20 + log(1 + 7e^-20))^2 ~ 400
    # scale-invariant inputs: shifting all logits by c shifts logsumexp by c
    x = _logits(scale=3.0)
    assert lab.router_z_loss(x + 2.0).item() > lab.router_z_loss(x).item()


# --------------------------------------------------------------------------- Task 5
def test_bias_balancer_moves_load_toward_uniform():
    E, k = 8, 2
    logits = _logits(n=512, e=E, seed=3)
    logits[:, 0] += 2.0                       # expert 0 is a "winner"; experts 6, 7 nearly dead
    logits[:, 6:] -= 2.0
    bal = lab.BiasBalancer(E, gamma=0.005)
    r0 = lab.topk_sigmoid_router(logits, k)
    m0 = lab.max_over_mean(r0.topk_idx, E)
    assert m0 > 2.0
    hist = []
    for _ in range(100):
        r = lab.topk_sigmoid_router(logits, k, bias=bal.bias)
        counts = torch.bincount(r.topk_idx.reshape(-1), minlength=E)
        bal.update(counts)
        hist.append(lab.max_over_mean(r.topk_idx, E))
    # the sign update oscillates by one gamma step around balance, so judge the tail's mean
    assert sum(hist[-20:]) / 20 < 1.1 and max(hist[-20:]) < m0 / 2
    # the bias is negative for the overloaded expert and positive for the starved ones
    assert bal.bias[0] < 0 and (bal.bias[6:] > 0).all()
    # combine weights never see the bias: for a token whose selection is unchanged, the weights
    # with and without bias are identical
    rb = lab.topk_sigmoid_router(logits, k, bias=bal.bias)
    same = (rb.topk_idx.sort(-1).values == r0.topk_idx.sort(-1).values).all(-1)
    assert same.any()
    i = int(same.nonzero()[0])
    wa = dict(zip(r0.topk_idx[i].tolist(), r0.topk_weight[i].tolist()))
    wb = dict(zip(rb.topk_idx[i].tolist(), rb.topk_weight[i].tolist()))
    for e in wa:
        assert math.isclose(wa[e], wb[e], abs_tol=1e-6)
    # perfectly balanced counts leave the bias alone
    before = bal.bias.clone()
    bal.update(torch.full((E,), 7))
    assert torch.equal(before, bal.bias)
    assert math.isclose(lab.max_over_mean(torch.arange(16).remainder(E)[:, None], E), 1.0)


# --------------------------------------------------------------------------- Task 6
def _layer(seed=0, **kw):
    torch.manual_seed(seed)
    return MoELayer(d_model=16, d_expert=12, n_experts=6, top_k=2, **kw)


def test_permute_unpermute_roundtrip_and_dropless_equivalence():
    layer = _layer()
    x = torch.randn(45, 16)
    r = layer.route(x)
    p = lab.permute_tokens(x, r.topk_idx, r.topk_weight, 6)
    assert p.xs.shape == (90, 16) and (p.expert_idx[1:] >= p.expert_idx[:-1]).all()
    assert torch.equal(p.counts, torch.bincount(r.topk_idx.reshape(-1), minlength=6))
    assert torch.equal(p.xs, x[p.token_idx])
    # the stable sort keeps token order inside each expert's segment
    for e in range(6):
        seg = p.token_idx[p.expert_idx == e]
        assert (seg[1:] > seg[:-1]).all()
    # identity experts: unpermute(permute(x)) sums the k gate weights (= 1) times x
    out = lab.unpermute_tokens(p.xs, p, 45)
    assert torch.allclose(out, x, atol=1e-5)
    # dropless pipeline equals the mask-based reference
    y = lab.moe_forward(layer.experts, x, r)
    ref = layer.forward_reference(x)
    assert torch.allclose(y, ref, atol=1e-5)
    # and the explicit tile-schedule GEMM gives the same numbers
    y2 = lab.moe_forward(layer.experts, x, r, gemm=lambda a, c, w: gg.grouped_gemm_torch(a, c, w, 8, 8))
    assert torch.allclose(y2, ref, atol=1e-5)


def test_capacity_dropping():
    layer = _layer(seed=1)
    x = torch.randn(60, 16)
    r = layer.route(x)
    p = lab.permute_tokens(x, r.topk_idx, r.topk_weight, 6)
    # capacity at least the largest segment: nothing dropped, equals dropless
    cap = int(p.counts.max())
    q = lab.apply_capacity(p, cap)
    assert q.n_dropped == 0 and torch.equal(q.counts, p.counts) and torch.equal(q.xs, p.xs)
    big = lab.moe_forward(layer.experts, x, r, capacity_factor=cap * 6 / (60 * 2))
    assert torch.allclose(big, layer.forward_reference(x), atol=1e-5)
    # a tight capacity drops exactly the overflow, earliest tokens win
    cap = 10
    q = lab.apply_capacity(p, cap)
    assert q.n_dropped == int((p.counts - cap).clamp(min=0).sum())
    assert torch.equal(q.counts, p.counts.clamp(max=cap))
    for e in range(6):
        kept = q.token_idx[q.expert_idx == e]
        assert torch.equal(kept, p.token_idx[p.expert_idx == e][:cap])
    # dropped slots contribute nothing: compare with a masked reference
    y = lab.moe_forward(layer.experts, x, r, capacity_factor=cap * 6 / (60 * 2))
    keep = torch.zeros(60, 2, dtype=torch.bool)
    for e in range(6):
        rows_e = (r.topk_idx == e).nonzero()                        # (n_e, 2) in token order
        for t, slot in rows_e[:cap].tolist():
            keep[t, slot] = True
    ref = torch.zeros_like(x)
    for t in range(60):
        for slot in range(2):
            if keep[t, slot]:
                e = int(r.topk_idx[t, slot])
                ref[t] += r.topk_weight[t, slot] * layer.experts.forward_one(e, x[t:t + 1])[0]
    assert torch.allclose(y, ref, atol=1e-5)
    assert keep.sum().item() == 120 - q.n_dropped


# --------------------------------------------------------------------------- Task 7
def test_tile_schedule():
    counts = torch.tensor([5, 0, 13, 32, 1])
    row_off, tile_off = lab.tile_schedule(counts, bm=8)
    assert row_off.tolist() == [0, 5, 5, 18, 50, 51]
    assert tile_off.tolist() == [0, 1, 1, 3, 7, 8]
    assert row_off.dtype == torch.int32 and tile_off.dtype == torch.int32
    assert int(tile_off[-1]) <= gg.max_tiles(int(counts.sum()), 5, 8)


@pytest.mark.parametrize("counts", [[5, 0, 13, 19], [0, 0, 7], [64], [1, 1, 1, 1, 1, 1, 1, 1, 1]])
def test_grouped_gemm_loop_and_triton(counts):
    torch.manual_seed(0)
    counts = torch.tensor(counts, device=DEV)
    M, E, K, N = int(counts.sum()), counts.numel(), 24, 20
    xs, w = torch.randn(M, K, device=DEV), torch.randn(E, K, N, device=DEV)
    ref = gg.grouped_gemm_reference(xs, counts, w)
    assert torch.allclose(lab.grouped_gemm_loop(xs, counts, w), ref, atol=1e-5)
    if not lab.HAS_TRITON:
        pytest.skip("triton not installed")
    out = lab.grouped_gemm_triton(xs, counts, w, bm=16, bn=16, bk=16)
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()
    # transposed weights (the dX of the backward pass) are just a stride swap
    dy = torch.randn(M, N, device=DEV)
    dx = lab.grouped_gemm_triton(dy, counts, w.transpose(1, 2), bm=16, bn=16, bk=16)
    assert torch.allclose(dx, gg.grouped_gemm_reference(dy, counts, w.transpose(1, 2)), atol=1e-4)


def test_grouped_gemm_autograd_wrapper():
    torch.manual_seed(0)
    counts = torch.tensor([3, 0, 9, 4], device=DEV)
    xs = torch.randn(16, 8, device=DEV, requires_grad=True)      # tl.dot has no fp64 path
    w = torch.randn(4, 8, 6, device=DEV, requires_grad=True)
    if not gg.HAS_TRITON:
        pytest.skip("triton not installed")
    y = gg.GroupedGEMM.apply(xs, counts, w)
    y.square().sum().backward()
    xr = xs.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    gg.grouped_gemm_reference(xr, counts, wr).square().sum().backward()
    assert torch.allclose(xs.grad, xr.grad, atol=1e-4)
    assert torch.allclose(w.grad, wr.grad, atol=1e-4)


def test_padding_efficiency_model():
    info = gg.grouped_gemm_tiles([5, 0, 13, 19], bm=16)
    assert info["tiles"] == 1 + 0 + 1 + 2 and info["padded_rows"] == 64
    assert math.isclose(info["efficiency"], 37 / 64)


@gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_grouped_gemm_triton_gpu(dtype):
    torch.manual_seed(0)
    counts = torch.tensor([70, 0, 130, 1, 311], device="cuda")
    xs = torch.randn(512, 96, device="cuda", dtype=dtype)
    w = torch.randn(5, 96, 80, device="cuda", dtype=dtype) / 10
    out = lab.grouped_gemm_triton(xs, counts, w)
    ref = gg.grouped_gemm_reference(xs.float(), counts, w.float())
    tol = 1e-4 if dtype == torch.float32 else 3e-2
    assert torch.allclose(out.float(), ref, atol=tol, rtol=tol)


# --------------------------------------------------------------------------- Task 8
def test_upcycling_preserves_dense_function():
    torch.manual_seed(0)
    ffn = SwiGLUFeedForward(16, 24)
    x = torch.randn(3, 7, 16)
    layer = lab.upcycle_copy(ffn, n_experts=4, top_k=2)
    assert layer.experts.w_gate.shape == (4, 16, 24)
    assert torch.allclose(layer(x), ffn(x), atol=1e-5)
    assert torch.allclose(layer.forward_reference(x), ffn(x), atol=1e-5)
    # a sigmoid gate without renormalisation does *not* reproduce the dense function
    sig = lab.upcycle_copy(ffn, 4, 2, gate="sigmoid", normalize=False)
    assert not torch.allclose(sig(x), ffn(x), atol=1e-3)
    split = lab.upcycle_split(ffn, n_experts=4, top_k=2)
    assert split.experts.w_gate.shape == (4, 16, 6)
    x2 = x.reshape(-1, 16)
    assert torch.allclose(split.experts.forward_all(x2).sum(0), ffn(x2), atol=1e-5)
    p_dense = sum(p.numel() for p in ffn.parameters())
    assert moe_params(MoEShape(16, 6, 4, 2))["total"] == p_dense + 16 * 4


# --------------------------------------------------------------------------- Task 9
def _ep_worker(rank, world, which):
    import torch.distributed as dist
    torch.manual_seed(0)
    layer = MoELayer(d_model=16, d_expert=12, n_experts=6, top_k=2, n_shared=1, gate="sigmoid")
    layer.eval()
    g = torch.Generator().manual_seed(7)
    x = torch.randn(world * 21, 16, generator=g)
    xs = x.chunk(world)[rank]
    with torch.no_grad():
        ref = layer(x).chunk(world)[rank]                       # single-process numerics
        r = layer.router(xs)
        e_local = 6 // world
        local = layer.experts.slice(rank * e_local, (rank + 1) * e_local)
        out = lab.ep_forward(local, xs, r, 6, world) + layer.shared(xs)
    assert torch.allclose(out, ref, atol=1e-5), f"[rank {rank}] EP mismatch {(out - ref).abs().max()}"
    # the plan is consistent: rows sent equal rows the peers receive
    p = lab.permute_tokens(xs, r.topk_idx, r.topk_weight, 6)
    plan = lab.plan_dispatch(p.counts, world)
    assert sum(plan.send_splits) == 42 and int(plan.local_counts.sum()) == sum(plan.recv_splits)
    assert torch.equal(plan.recv_counts.sum(0), plan.local_counts)
    # the library's wrapper, with autograd through both all-to-alls
    ep = ExpertParallelMoE(layer, rank, world)
    xg = xs.clone().requires_grad_(True)
    y = ep(xg)
    assert torch.allclose(y, ref, atol=1e-5)
    y.square().sum().backward()
    xr = x.clone().requires_grad_(True)
    layer(xr).square().sum().backward()
    assert torch.allclose(xg.grad, xr.grad.chunk(world)[rank], atol=1e-4)
    dist.barrier()


def test_expert_parallel_world2_matches_single_process():
    run_distributed(_ep_worker, 2, "ep")


# --------------------------------------------------------------------------- Task 10
def test_cost_models():
    # Qwen3-30B-A3B-like layer on this box: EP=2 over PCIe
    assert math.isclose(lab.ep_bytes_per_token(2048, 8, 2), 2 * 8 * 2048 * 2 * 0.5)
    assert math.isclose(lab.ep_bytes_per_token(2048, 8, 8), 2 * 8 * 2048 * 2 * 7 / 8)
    assert math.isclose(lab.tp_bytes_per_token(2048, 2), 2048 * 2)
    assert lab.ep_bytes_per_token(2048, 8, 2) > lab.tp_bytes_per_token(2048, 2)
    assert math.isclose(lab.expected_active_experts(128, 8, 1), 8.0)
    a64 = lab.expected_active_experts(128, 8, 64)
    assert 125 < a64 < 126.5 and a64 < 128
    assert lab.expected_active_experts(128, 8, 4) < 32      # overlaps make it fewer than 4k
    s = MoEShape(d_model=2048, d_expert=768, n_experts=128, top_k=8)
    frac = lab.active_fraction(s)
    assert math.isclose(frac, (8 * 3 * 2048 * 768 + 2048 * 128) / (128 * 3 * 2048 * 768 + 2048 * 128))
    full = lab.offload_miss_bytes(s, 48, 1, cache_fraction=0.0)
    assert math.isclose(full, 48 * 8 * 3 * 2048 * 768 * 2)
    assert math.isclose(lab.offload_miss_bytes(s, 48, 1, 0.5), full / 2)
    assert lab.offload_miss_bytes(s, 48, 1, 1.0) == 0.0
