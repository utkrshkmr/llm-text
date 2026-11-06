"""Lab C tests: CPU only, eight simulated XLA devices, no GPU, a few seconds of compile time.

    LAB_IMPL=starter  python -m pytest -q labs/labC_jax
    LAB_IMPL=solution python -m pytest -q labs/labC_jax

The device-count flag must be set before JAX is imported anywhere in the process; the import
of ``llmsys.jaxlab`` below does it (and the lab modules do it again, harmlessly).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_force_host_platform_device_count=8").strip()

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from _labutil import load_impl  # noqa: E402

jaxlab = pytest.importorskip("llmsys.jaxlab")
if not jaxlab.HAS_JAX:
    pytest.skip("jax is not installed", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from llmsys.gpt import GPT_CONFIG_TINY, GPTModel  # noqa: E402
from llmsys.jaxlab import basics, hlo, sharding as S  # noqa: E402

lab = load_impl(__file__)
pytestmark = pytest.mark.skipif(len(jax.devices()) < 8,
                                reason="needs 8 simulated devices (XLA_FLAGS set too late)")

MESH = S.make_mesh((2, 4), ("dp", "tp"))
RNG = np.random.default_rng(0)


def _ab(m=8, k=16, n=8):
    a = jnp.asarray(RNG.standard_normal((m, k)), jnp.float32)
    b = jnp.asarray(RNG.standard_normal((k, n)), jnp.float32)
    return a, b


def _close(x, y, tol=1e-5):
    return float(jnp.max(jnp.abs(jnp.asarray(x) - jnp.asarray(y)))) < tol


# --------------------------------------------------------------------------- Task 1 + 2: the model
def test_block_matches_reference():
    key = jax.random.key(0)
    p = basics.init_block_params(key, 32, qkv_bias=True)
    x = jax.random.normal(jax.random.key(1), (2, 7, 32))
    assert _close(lab.layer_norm(p["norm1"], x), basics.layer_norm(p["norm1"], x))
    assert _close(lab.causal_attention(p["att"], x, 4), basics.causal_attention(p["att"], x, 4))
    assert _close(lab.transformer_block(p, x, 4), basics.transformer_block(p, x, 4))


def test_causal_mask_is_causal():
    p = basics.init_block_params(jax.random.key(0), 16, qkv_bias=False)
    x = jax.random.normal(jax.random.key(1), (1, 6, 16))
    y_full = lab.causal_attention(p["att"], x, 2)
    y_prefix = lab.causal_attention(p["att"], x[:, :3], 2)
    assert _close(y_full[:, :3], y_prefix)       # the future must not change the past


def test_jax_gpt_equals_torch_gpt():
    torch.manual_seed(0)
    model = GPTModel(GPT_CONFIG_TINY).eval()
    idx = RNG.integers(0, GPT_CONFIG_TINY["vocab_size"], size=(2, 12))
    with torch.no_grad():
        ref = model(torch.as_tensor(idx)).numpy()
    params = lab.torch_state_dict_to_jax(model.state_dict())
    assert basics.count_params(params) == sum(p.numel() for p in model.parameters())
    out = lab.gpt_forward(params, jnp.asarray(idx), GPT_CONFIG_TINY["n_heads"])
    assert out.shape == ref.shape
    assert np.max(np.abs(np.asarray(out) - ref)) < 1e-5


def test_jit_gives_same_logits_and_traces_once():
    torch.manual_seed(0)
    model = GPTModel(GPT_CONFIG_TINY)
    params = lab.torch_state_dict_to_jax(model.state_dict())
    idx = jnp.asarray(RNG.integers(0, 512, size=(2, 8)))
    fwd = jax.jit(lab.gpt_forward, static_argnums=2)
    assert _close(fwd(params, idx, 4), lab.gpt_forward(params, idx, 4))
    # one trace per shape signature: same shape twice -> one trace, new shape -> a second
    hist = basics.count_traces(lambda i: lab.gpt_forward(params, i, 4),
                               [(idx,), (idx + 1,), (idx[:, :4],)])
    assert hist == [1, 1, 2]


# --------------------------------------------------------------------------- Task 3: vmap(grad)
def test_per_example_grads_match_loop():
    cfg = dict(GPT_CONFIG_TINY, n_layers=1)
    params = basics.init_gpt_params(jax.random.key(0), cfg)
    idx = jnp.asarray(RNG.integers(0, 512, size=(3, 6)))
    tgt = jnp.asarray(RNG.integers(0, 512, size=(3, 6)))
    # jit both sides: eager JAX dispatches op by op, which is slow for a whole GPT gradient
    g = jax.jit(lab.per_example_grads, static_argnums=3)(params, idx, tgt, 4)
    one_grad = jax.jit(jax.grad(basics.loss_fn), static_argnums=3)
    assert g["out_head"].shape == (3,) + params["out_head"].shape
    for i in range(3):
        gi = one_grad(params, idx[i:i + 1], tgt[i:i + 1], 4)
        assert _close(g["out_head"][i], gi["out_head"], 1e-5)
        assert _close(g["blocks"][0]["ff"]["fc1"]["w"][i], gi["blocks"][0]["ff"]["fc1"]["w"], 1e-5)


# --------------------------------------------------------------------------- Task 4: automatic sharding
@pytest.mark.parametrize("a_spec,b_spec,out_spec,collective", [
    (("dp", None), (None, "tp"), ("dp", "tp"), None),              # case 1: local
    (("dp", "tp"), ("tp", None), ("dp", None), "all-reduce"),      # case 3: both on K
    (("tp", None), (None, "tp"), ("tp", None), "all-gather"),      # case 4: conflict on tp
])
def test_auto_sharding_cases(a_spec, b_spec, out_spec, collective):
    a, b = _ab()
    A, B = S.shard(a, MESH, *a_spec), S.shard(b, MESH, *b_spec)
    fn = lab.sharded_matmul_auto(MESH, a_spec, b_spec, out_spec)
    out = fn(A, B)
    assert _close(out, a @ b, 1e-4)
    assert S.spec_of(out) == out_spec
    counts = hlo.count_collectives(hlo.compiled_text(fn, A, B))
    issued = {k for k, v in counts.items() if v}
    assert issued == ({collective} if collective else set())


def test_auto_sharding_free_output_matches_calculus():
    a, b = _ab()
    A, B = S.shard(a, MESH, "dp", "tp"), S.shard(b, MESH, "tp", None)
    fn = lab.sharded_matmul_auto(MESH, ("dp", "tp"), ("tp", None))
    out = fn(A, B)
    assert _close(out, a @ b, 1e-4)
    assert S.spec_of(out)[0] == "dp"     # rows follow A's dp sharding; tp is reduced away
    assert hlo.count_collectives(hlo.compiled_text(fn, A, B))["all-reduce"] == 1


# --------------------------------------------------------------------------- Task 5: shard_map cases
@pytest.mark.parametrize("name,a_spec,b_spec,out_spec", [
    ("case1_local", ("dp", None), (None, "tp"), ("dp", "tp")),
    ("case2_gather", ("dp", "tp"), (), ("dp", None)),
    ("case3_allreduce", ("dp", "tp"), ("tp", None), ("dp", None)),
    ("case3_reducescatter", ("dp", "tp"), ("tp", None), ("dp", "tp")),
    ("case4_gather", ("tp", None), (None, "tp"), ("tp", None)),
])
def test_shard_map_cases_equal_matmul(name, a_spec, b_spec, out_spec):
    a, b = _ab(8, 16, 8)
    fn = jax.jit(getattr(lab, name)(MESH))
    A, B = S.shard(a, MESH, *a_spec), S.shard(b, MESH, *b_spec)
    out = fn(A, B)
    assert _close(out, a @ b, 1e-4)
    assert S.spec_of(out) == out_spec


@pytest.mark.parametrize("name,expected", [
    ("case1_local", {}),
    ("case2_gather", {"all-gather": 1}),
    ("case3_allreduce", {"all-reduce": 1}),
    ("case3_reducescatter", {"reduce-scatter": 1}),
    ("case4_gather", {"all-gather": 1}),
])
def test_shard_map_cases_issue_declared_collective(name, expected):
    a, b = _ab(8, 16, 8)
    a_spec, b_spec = {
        "case1_local": (("dp", None), (None, "tp")), "case2_gather": (("dp", "tp"), ()),
        "case3_allreduce": (("dp", "tp"), ("tp", None)),
        "case3_reducescatter": (("dp", "tp"), ("tp", None)),
        "case4_gather": (("tp", None), (None, "tp"))}[name]
    fn = jax.jit(getattr(lab, name)(MESH))
    text = hlo.compiled_text(fn, S.shard(a, MESH, *a_spec), S.shard(b, MESH, *b_spec))
    counts = {k: v for k, v in hlo.count_collectives(text).items() if v}
    assert counts == expected


def test_grad_through_shard_map_equals_unsharded_grad():
    a, b = _ab(8, 16, 8)
    sharded = S.matmul_grad(lab.case3_allreduce(MESH))
    plain = S.matmul_grad(lambda x, y: x @ y)
    ga, gb = sharded(S.shard(a, MESH, "dp", "tp"), S.shard(b, MESH, "tp", None))
    ra, rb = plain(a, b)
    assert _close(ga, ra, 1e-4) and _close(gb, rb, 1e-4)
    # the transpose of psum is a broadcast; of all_gather, a reduce-scatter
    text = hlo.compiled_text(S.matmul_grad(lab.case4_gather(MESH)),
                             S.shard(a, MESH, "tp", None), S.shard(b, MESH, None, "tp"))
    counts = hlo.count_collectives(text)
    assert counts["all-gather"] >= 1 and counts["reduce-scatter"] >= 1


# --------------------------------------------------------------------------- Task 6: collective matmul
def test_collective_matmul_equals_matmul():
    x, w = _ab(16, 8, 16)
    fn = jax.jit(lab.collective_matmul_allgather(MESH))
    X, W = S.shard(x, MESH, "tp", None), S.shard(w, MESH, None, "tp")
    out = fn(X, W)
    assert _close(out, x @ w, 1e-4)
    assert S.spec_of(out) == (None, "tp")
    counts = hlo.count_collectives(hlo.compiled_text(fn, X, W))
    assert counts["collective-permute"] == MESH.shape["tp"] - 1
    assert counts["all-gather"] == 0


# --------------------------------------------------------------------------- Task 7: reading HLO
HLO_SAMPLE = """
ENTRY %main.0_spmd (param: f32[4,4], param.1: f32[4,8]) -> f32[4,8] {
  %param = f32[4,4]{1,0} parameter(0), sharding={devices=[2,4]<=[8]}
  %ynn_fusion = f32[4,8]{1,0} fusion(%param, %param.1), kind=kCustom, calls=%fused_computation
  %all-gather-start = (f32[4,8], f32[16,8]) all-gather-start(%ynn_fusion), dimensions={0}
  %all-gather-done = f32[16,8]{1,0} all-gather-done(%all-gather-start)
  %reduce_scatter_x = f32[4,8]{1,0} reduce-scatter(%all-gather-done), dimensions={1}, metadata={op_name="jit/all-reduce"}
  ROOT %all-reduce = f32[4,8]{1,0} all-reduce(%reduce_scatter_x), channel_id=1, to_apply=%add.clone
}
"""


def test_count_collectives_on_text():
    counts = lab.count_collectives(HLO_SAMPLE)
    assert counts["all-gather"] == 1          # start/done pair counts once
    assert counts["reduce-scatter"] == 1      # the metadata string does not count
    assert counts["all-reduce"] == 1          # the instruction name alone does not count
    assert counts["collective-permute"] == 0 and counts["all-to-all"] == 0
    shlo = '%2 = "stablehlo.all_reduce"(%1) <{channel_handle = ...}> ({'
    assert lab.count_collectives(shlo)["all-reduce"] == 1


def test_count_collectives_on_compiled_programs():
    a, b = _ab()
    one = lab.sharded_matmul_auto(MESH, ("dp", None), (None, "tp"), ("dp", "tp"))
    three = lab.sharded_matmul_auto(MESH, ("dp", "tp"), ("tp", None), ("dp", None))
    none = lab.count_collectives(lab.compiled_text(
        one, S.shard(a, MESH, "dp", None), S.shard(b, MESH, None, "tp")))
    assert sum(none.values()) == 0
    some = lab.count_collectives(lab.compiled_text(
        three, S.shard(a, MESH, "dp", "tp"), S.shard(b, MESH, "tp", None)))
    assert some == {**none, "all-reduce": 1}


# --------------------------------------------------------------------------- Task 8: cost model
def test_ring_overlap_model():
    seq, ovl = lab.ring_overlap_time_s(compute_s=1.0, total_bytes=0, n=4, alpha_s=0.0)
    assert seq == pytest.approx(1.0) and ovl == pytest.approx(1.0)   # nothing to hide
    seq, ovl = lab.ring_overlap_time_s(compute_s=4.0, total_bytes=4 * 22e9, n=4, alpha_s=0.0)
    assert seq == pytest.approx(3 + 4) and ovl == pytest.approx(4.0)  # balanced: hop == step
    seq, ovl = lab.ring_overlap_time_s(compute_s=1e-3, total_bytes=0, n=2, alpha_s=25e-6)
    assert ovl < seq and ovl == pytest.approx(max(0.5e-3, 25e-6) + 0.5e-3)
    assert lab.ring_overlap_time_s(1.0, 1 << 20, 1)[0] == pytest.approx(1.0)
