"""Lab 23 tests: fx fusion pass, Dynamo counts, torch.export, ONNX Runtime, Inductor output,
CUDA-graph decode. CPU-only except the last; Inductor compiles are kept tiny."""
import os

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                      os.path.join(os.environ.get("TMPDIR", "/tmp"), "llmsys_inductor_cache"))

import pytest  # noqa: E402
import torch  # noqa: E402

from _labutil import load_impl  # noqa: E402
from llmsys.compile import dynamo_tools, export_tools, fx_passes  # noqa: E402
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel  # noqa: E402
from llmsys.kvcache import StaticKVCache  # noqa: E402
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Block, Qwen3Model  # noqa: E402

lab = load_impl(__file__)
HAS_CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not HAS_CUDA, reason="needs a CUDA GPU")


def _block_graph(dtype=torch.float32):
    """A Qwen3 block as Dynamo captures it, plus the real inputs of its single graph."""
    torch.manual_seed(0)
    model = Qwen3Model(QWEN3_CONFIG_TINY).eval()
    blk = Qwen3Block(QWEN3_CONFIG_TINY, dtype=dtype).eval()
    x = torch.randn(2, 8, QWEN3_CONFIG_TINY["emb_dim"]).to(dtype)
    cap = dynamo_tools.capture_graphs(blk, x, model.cos[:8], model.sin[:8])
    assert cap.compiles == 1
    return cap.graphs[0], cap.example_inputs[0]


def _canon_counts(gm):
    counts = {}
    for n in gm.graph.nodes:
        c = fx_passes.canonical(n, gm)
        counts[c] = counts.get(c, 0) + 1
    return counts


# ----------------------------------------------------------------- Task 1
def test_canonical_op_normalises_spellings():
    def f(x):
        return (x * 2).pow(2).mean(-1, keepdim=True).rsqrt() + torch.rsqrt(x.float()).to(torch.float32)
    gm = dynamo_tools.capture_graphs(f, torch.randn(4, 8)).graphs[0]
    names = [lab.canonical_op(n) for n in gm.graph.nodes]
    assert names.count("mul") == 1 and names.count("pow") == 1 and names.count("mean") == 1
    assert names.count("rsqrt") == 2 and names.count("cast") == 2 and names.count("add") == 1
    assert names[0] is None and names[-1] is None          # placeholder and output are not ops


def test_match_rmsnorm_finds_all_four_norms():
    gm, _ = _block_graph()
    matches = [lab.match_rmsnorm(n) for n in gm.graph.nodes]
    found = [m for m in matches if m is not None]
    assert len(found) == 4                                  # norm1, q_norm, k_norm, norm2
    for x, w, eps in found:
        assert eps == 1e-6 and w.op == "placeholder"        # the scale is a lifted parameter


def test_fuse_rmsnorm_is_exact_and_removes_the_pattern():
    gm, inputs = _block_graph()
    ref = gm(*inputs)[0]
    assert lab.fuse_rmsnorm(gm) == 4
    gm.graph.eliminate_dead_code(); gm.recompile()
    out = gm(*inputs)[0]
    assert torch.equal(out, ref)                            # fp32 maths in both: bit-identical
    counts = _canon_counts(gm)
    assert "rsqrt" not in counts and "pow" not in counts and "mean" not in counts
    assert counts.get("fused_rmsnorm", 0) == 4


def test_fuse_rmsnorm_bf16_absorbs_the_casts():
    gm, inputs = _block_graph(torch.bfloat16)
    ref = gm(*inputs)[0]
    lab.fuse_rmsnorm(gm)
    gm.graph.eliminate_dead_code(); gm.recompile()
    out = gm(*inputs)[0]
    assert out.dtype == torch.bfloat16
    assert (out.float() - ref.float()).abs().max() <= 2e-2 * ref.float().abs().max()
    # 4 norms x (up-cast + down-cast) gone: the only casts left are the ones RoPE/softmax do
    assert _canon_counts(gm).get("cast", 0) <= 20 - 8


# ----------------------------------------------------------------- Task 2
def test_pipeline_matches_reference_and_shrinks_graph():
    gm, inputs = _block_graph()
    gm_ref, _ = _block_graph()
    ref_out = gm_ref(*inputs)[0]
    log = lab.run_pipeline(gm)
    assert [r["name"] for r in log] == ["fuse_rmsnorm", "fuse_silu_mul", "eliminate_dead_code"]
    assert [r["matches"] for r in log][:2] == [4, 1]
    assert log[0]["nodes_after"] == log[0]["nodes_before"] + 4      # fusions add before DCE removes
    _, ref_stats = fx_passes.run_passes(gm_ref)
    assert log[-1]["nodes_after"] == ref_stats[-1].nodes_after      # same final size as llmsys
    assert log[-1]["nodes_after"] < 0.75 * log[0]["nodes_before"]
    out = gm(*inputs)[0]
    assert torch.allclose(out, ref_out, atol=1e-6, rtol=1e-6)
    assert _canon_counts(gm).get("silu", 0) == 0


# ----------------------------------------------------------------- Task 3
def test_graph_breaks_are_counted():
    def clean(x):
        return (x * 2).relu().sum()

    def breaks(x):
        y = x * 2
        if y.sum() > 0:               # data-dependent branch: Dynamo must stop and ask Python
            y = y + 1
        return y.relu()

    assert lab.count_graph_breaks(clean, torch.randn(4))[:2] == (1, 0)
    graphs, nbreaks, reasons = lab.count_graph_breaks(breaks, torch.randn(4))
    assert (graphs, nbreaks) == (2, 1) and len(reasons) == 1


def test_recompiles_follow_the_guards():
    def f(x, k):
        return (x * k).relu()

    shapes = [(torch.randn(4), 2), (torch.randn(5), 2), (torch.randn(6), 2), (torch.randn(6), 2)]
    # static shapes: first call compiles, 5 recompiles once (Dynamo then marks the dim dynamic)
    assert lab.count_recompiles(f, shapes) == 2
    # a Python int is a constant in the graph: every new value is a new program
    ints = [(torch.randn(4), k) for k in (1, 2, 3)]
    assert lab.count_recompiles(f, ints) == 3
    # dynamic=True: one shape-polymorphic graph for all sizes
    assert lab.count_recompiles(f, shapes, dynamic=True) == 1
    # a dtype change is a guard failure too
    dts = [(torch.randn(4), 2), (torch.randn(4).double(), 2)]
    assert lab.count_recompiles(f, dts) == 2


# ----------------------------------------------------------------- Task 4
def test_export_gpt_dynamic_seq_runs_at_other_lengths():
    torch.manual_seed(0)
    model = GPTModel(GPT_CONFIG_TINY).eval()
    ids = torch.randint(0, 512, (2, 16))
    ep = lab.export_gpt_dynamic(model, ids, max_seq=GPT_CONFIG_TINY["context_length"])
    assert len(ep.range_constraints) >= 1                      # the sequence symbol and its range
    for T in (1, 7, 16, 40):
        other = torch.randint(0, 512, (2, T))
        with torch.no_grad():
            assert torch.allclose(ep.module()(other), model(other), atol=1e-6)
    with pytest.raises(Exception):                              # batch was not marked dynamic
        ep.module()(torch.randint(0, 512, (3, 16)))


# ----------------------------------------------------------------- Task 5
@pytest.mark.skipif(not (export_tools.HAS_ONNX and export_tools.HAS_ORT), reason="needs onnx + onnxruntime")
def test_onnx_runtime_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    model = GPTModel(GPT_CONFIG_TINY).eval()
    ids = torch.randint(0, 512, (2, 12))
    path = lab.export_to_onnx(model, ids, str(tmp_path / "tiny_gpt.onnx"))
    assert os.path.getsize(path) > 0
    for shape in ((2, 12), (1, 5), (3, 20)):                   # batch and sequence both dynamic
        other = torch.randint(0, 512, shape)
        out = lab.run_onnx(path, other)
        with torch.no_grad():
            ref = model(other)
        assert out.shape == ref.shape
        assert (out - ref).abs().max() < 1e-4
        assert torch.equal(out.argmax(-1), ref.argmax(-1))


# ----------------------------------------------------------------- Task 6
def test_inductor_fuses_norm_residual_activation():
    def norm_res_act(x, r, w):
        h = x + r
        y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        return torch.nn.functional.gelu(y), h

    args = (torch.randn(8, 64), torch.randn(8, 64), torch.ones(64))
    try:
        names, code = lab.inductor_kernels(norm_res_act, *args)
    except Exception as e:                                      # no C++ compiler, sandboxed tmp, ...
        pytest.skip(f"Inductor unavailable here: {type(e).__name__}: {str(e)[:120]}")
    if not names:
        pytest.skip("Inductor emitted no kernels (cache hit without a loadable module?)")
    assert names == dynamo_tools.kernel_names(code)
    fused = [n for n in names if "rsqrt" in n and "mean" in n]
    assert len(fused) == 1, names                               # one kernel for the whole chain
    ops = dynamo_tools.fused_ops(fused[0])
    assert {"add", "mean", "rsqrt", "mul", "gelu"} <= set(ops), ops


# ----------------------------------------------------------------- Task 7
@gpu
@pytest.mark.gpu
def test_cuda_graph_decode_equals_eager():
    torch.manual_seed(0)
    cfg = QWEN3_CONFIG_TINY
    model = Qwen3Model(cfg).cuda().eval()
    B, P, steps = 2, 6, 5
    prompt = torch.randint(0, cfg["vocab_size"], (B, P), device="cuda")
    cache = StaticKVCache(cfg["n_layers"], B, cfg["n_kv_groups"], 32, cfg["head_dim"], device="cuda")
    cache_ref = StaticKVCache(cfg["n_layers"], B, cfg["n_kv_groups"], 32, cfg["head_dim"], device="cuda")
    with torch.no_grad():
        logits = model(prompt, cache=cache)
        logits_ref = model(prompt, cache=cache_ref)
        dec = lab.capture_decode_graph(model, cache, start_pos=P)
        for _ in range(steps):
            nxt = logits_ref[:, -1].argmax(-1, keepdim=True)
            logits_ref = model(nxt, cache=cache_ref)
            logits = dec.step(nxt).clone()                      # the output buffer is overwritten by the next replay
            assert torch.allclose(logits, logits_ref, atol=1e-4, rtol=1e-4)
    assert cache.lengths[0] == P + steps == cache_ref.lengths[0]
    torch.testing.assert_close(cache.k[0][:, :, :P + steps], cache_ref.k[0][:, :, :P + steps], atol=1e-5, rtol=1e-5)
