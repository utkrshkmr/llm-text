import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from _labutil import load_impl
from llmsys.gpt import GPT_CONFIG_124M, GPT_CONFIG_TINY, GPTModel
from llmsys.kvcache import StaticKVCache
from llmsys.qwen3 import QWEN3_CONFIG_TINY, QWEN3_CONFIGS, Qwen3Model
from llmsys.utils import count_parameters

lab = load_impl(__file__)


def test_gpt_params_exact():
    for cfg in (GPT_CONFIG_TINY, dict(GPT_CONFIG_TINY, qkv_bias=True, n_layers=3)):
        assert lab.gpt_param_count(cfg) == count_parameters(GPTModel(cfg))


def test_gpt2_small_is_163M_untied():
    n = lab.gpt_param_count(GPT_CONFIG_124M)
    assert n == 163_009_536
    # removing the duplicated output head recovers the familiar "124M"
    assert n - GPT_CONFIG_124M["vocab_size"] * GPT_CONFIG_124M["emb_dim"] == 124_412_160


def test_qwen3_params_exact():
    for cfg in (QWEN3_CONFIG_TINY, dict(QWEN3_CONFIG_TINY, tie_embeddings=False, qk_norm=False)):
        assert lab.qwen3_param_count(cfg) == count_parameters(Qwen3Model(cfg))
    assert abs(lab.qwen3_param_count(QWEN3_CONFIGS["0.6B"]) - 596e6) < 2e6


def _measured_flops(model, T, vocab):
    idx = torch.randint(0, vocab, (1, T))
    with FlopCounterMode(display=False) as fc:
        model(idx)
    return fc.get_total_flops()


def test_flops_match_counter():
    T = 16
    for cfg, M in ((GPT_CONFIG_TINY, GPTModel), (QWEN3_CONFIG_TINY, Qwen3Model)):
        m = M(cfg).eval()
        # dense (masked) attention: every query is multiplied against all T keys
        predicted = T * lab.forward_flops_per_token(cfg, context=T)
        assert predicted == _measured_flops(m, T, cfg["vocab_size"])


def test_kv_cache_bytes_matches_buffer():
    cfg = QWEN3_CONFIG_TINY
    c = StaticKVCache(cfg["n_layers"], 3, cfg["n_kv_groups"], 40, cfg["head_dim"],
                      dtype=torch.bfloat16)
    assert lab.kv_cache_bytes(cfg, 40, batch=3, dtype="bf16") == c.nbytes()
    # Qwen3-0.6B: 28 layers * 8 kv heads * 128 dims * 2 (K,V) * 2 bytes = 112 KiB / token
    assert lab.kv_cache_bytes(QWEN3_CONFIGS["0.6B"], 1) == 114_688


def test_training_numbers():
    assert lab.training_flops(1e9, 2e10) == 1.2e20
    assert lab.training_state_bytes(7e9) == 112e9
    d = lab.training_days(7e9, 2e12, n_gpus=1024, hw="H100-SXM", mfu=0.4)
    assert 2.0 < d < 2.5


def test_decode_roofline():
    cfg = QWEN3_CONFIGS["8B"]
    r1 = lab.decode_step_roofline(cfg, batch=1, context=1024)
    assert r1["bound"] == "memory"
    big = lab.decode_step_roofline(cfg, batch=1024, context=0)
    assert big["bound"] == "compute"
    b = lab.critical_batch(cfg)
    assert 100 < b < 400  # near the H100 ridge point (~295 FLOP/byte)
    assert lab.max_batch_for_kv(cfg, "H100-SXM", 4096) > 0


def test_kaplan_convention_is_our_count_at_half_context():
    # Kaplan's context term is the causal average of ours: a token at a uniformly random position
    # attends to n_ctx / 2 keys on average. Removing the LM-head matmul recovers their C_forward.
    for cfg in (GPT_CONFIG_124M, QWEN3_CONFIGS["0.6B"]):
        n_ctx = 1024
        ours = lab.forward_flops_per_token(cfg, n_ctx // 2) - 2 * cfg["emb_dim"] * cfg["vocab_size"]
        assert lab.kaplan_forward_flops(cfg, n_ctx) == ours
    # GPT-2 small: N = 12 * 12 * 768^2 = 84.9M, so 2N = 170M FLOPs; the context term is 18.9M
    k = lab.kaplan_forward_flops(GPT_CONFIG_124M, 1024)
    assert k == 2 * 84_934_656 + 2 * 12 * 1024 * 768


def test_reference_box_numbers():
    hw = lab.HARDWARE["A100-40GB-PCIe"]
    assert hw["flops"] / hw["bw"] == pytest.approx(200.6, rel=1e-3)
