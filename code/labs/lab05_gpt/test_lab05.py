import torch
from torch.utils.flop_counter import FlopCounterMode

from _labutil import load_impl
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel as RefGPT, gpt2_config

lab = load_impl(__file__)
CFG = dict(GPT_CONFIG_TINY)


def _n(model):
    return sum(p.numel() for p in model.parameters())


def test_layernorm_stats():
    torch.manual_seed(0)
    ln = lab.LayerNorm(16)
    y = ln(torch.randn(3, 5, 16) * 7 + 3)
    assert torch.allclose(y.mean(-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(y.var(-1, unbiased=False), torch.ones(3, 5), atol=1e-3)


def test_gelu_matches_torch_tanh_approx():
    x = torch.linspace(-6, 6, 101)
    assert torch.allclose(lab.gelu_tanh(x), torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-6)


def test_model_matches_reference_exactly():
    torch.manual_seed(0)
    ref = RefGPT(CFG).eval()
    mine = lab.GPTModel(CFG).eval()
    mine.load_state_dict(ref.state_dict())
    idx = torch.randint(0, CFG["vocab_size"], (2, 11))
    assert torch.allclose(mine(idx), ref(idx), atol=1e-5)


def test_block_preserves_shape():
    blk = lab.TransformerBlock(CFG)
    x = torch.randn(2, 7, CFG["emb_dim"])
    assert blk(x).shape == x.shape


def test_generate_greedy():
    torch.manual_seed(0)
    ref = RefGPT(CFG).eval()
    mine = lab.GPTModel(CFG).eval()
    mine.load_state_dict(ref.state_dict())
    idx = torch.tensor([[1, 2, 3]])
    out = lab.generate_greedy(mine, idx, 70, context_size=CFG["context_length"])
    assert out.shape == (1, 73)  # exceeds context_length -> must crop correctly


def test_param_count_exact():
    for cfg in [CFG, dict(CFG, qkv_bias=True, n_layers=3), gpt2_config("gpt2-small (124M)", qkv_bias=False)]:
        m = lab.GPTModel(cfg)
        assert lab.count_params_analytic(cfg)["total"] == _n(m)
    big = lab.count_params_analytic(gpt2_config("gpt2-small (124M)", qkv_bias=False))
    assert big["total"] == 163_009_536
    tied = lab.count_params_analytic(gpt2_config("gpt2-small (124M)", qkv_bias=True), tie_embeddings=True)
    assert tied["total"] == 124_439_808


def test_forward_flops_match_flop_counter():
    torch.manual_seed(0)
    m = lab.GPTModel(CFG).eval()
    for b, t in [(1, 8), (2, 16)]:
        idx = torch.randint(0, CFG["vocab_size"], (b, t))
        with FlopCounterMode(display=False) as fc:
            m(idx)
        assert fc.get_total_flops() == lab.forward_flops(CFG, b, t)


def test_activation_memory():
    torch.manual_seed(0)
    m = lab.GPTModel(CFG)
    a1 = lab.measure_saved_activation_bytes(m, torch.randint(0, 512, (1, 32)))
    a2 = lab.measure_saved_activation_bytes(m, torch.randint(0, 512, (2, 32)))
    assert 1.9 < a2 / a1 < 2.1
    est = lab.activation_bytes_estimate(CFG, 1, 32)
    assert est > 0 and 0.5 < a1 / est < 3.0
