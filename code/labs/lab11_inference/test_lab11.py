import itertools

import pytest
import torch

from _labutil import load_impl
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel
from llmsys.qwen3 import QWEN3_CONFIG_TINY, QWEN3_CONFIGS, Qwen3Model
from llmsys.utils import set_seed

lab = load_impl(__file__)


@pytest.fixture(params=["gpt", "qwen"])
def model(request):
    set_seed(0)
    if request.param == "gpt":
        return GPTModel(GPT_CONFIG_TINY).eval()
    return Qwen3Model(QWEN3_CONFIG_TINY).eval()


def _run(gen, model, ids, n, eos=None):
    return torch.cat(list(gen(model, ids, n, eos)), dim=1)


def test_three_decoders_agree(model):
    ids = torch.randint(0, 512, (2, 6))
    a = _run(lab.generate_nocache, model, ids, 12)
    b = _run(lab.generate_kvcache, model, ids, 12)
    c = _run(lab.generate_static, model, ids, 12)
    assert a.shape == (2, 12)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_eos_stops(model):
    ids = torch.randint(0, 512, (1, 5))
    first = _run(lab.generate_kvcache, model, ids, 1)[0, 0].item()
    for gen in (lab.generate_nocache, lab.generate_kvcache, lab.generate_static):
        assert list(gen(model, ids, 10, first)) == []


def test_kv_bytes_qwen3_06b():
    # 2 (K,V) * 28 layers * 8 kv heads * 128 dims * 2 bytes = 114,688 B per token
    assert lab.kv_bytes_per_token(QWEN3_CONFIGS["0.6B"]) == 114_688
    assert lab.max_tokens_in_budget(QWEN3_CONFIGS["0.6B"], 114_688 * 10 + 5) == 10


def test_kv_bytes_matches_static_cache():
    from llmsys.kvcache import StaticKVCache
    cfg = QWEN3_CONFIG_TINY
    c = StaticKVCache(cfg["n_layers"], 1, cfg["n_kv_groups"], 50, cfg["head_dim"], torch.float32)
    assert lab.kv_bytes_per_token(cfg, 4) * 50 == c.nbytes()
    gcfg = GPT_CONFIG_TINY
    assert lab.kv_bytes_per_token(gcfg, 4) == 2 * 2 * 64 * 4


def test_roofline():
    r = lab.decode_roofline(8e9, 147_456, batch=1, context=1000)
    assert r["bound"] == "memory"
    assert abs(r["t_memory"] - (16e9 + 1000 * 147_456) / 3.35e12) < 1e-9
    big = lab.decode_roofline(8e9, 0, batch=4096, context=0)
    assert big["bound"] == "compute"
    cb = lab.critical_batch(8e9)
    assert 290 < cb < 300          # H100: 989e12 / 3.35e12 ~= 295 FLOP/byte
    at = lab.decode_roofline(8e9, 0, batch=int(cb) + 1, context=0)
    assert at["bound"] == "compute"


def test_measure_generation_with_fake_clock(model):
    ticks = itertools.count(0.0, 0.5)
    clock = lambda: next(ticks)
    ids = torch.randint(0, 512, (1, 4))
    m = lab.measure_generation(lab.generate_kvcache, model, ids, 5, clock=clock)
    assert m["n_tokens"] == 5
    assert m["ttft"] == pytest.approx(0.5) and m["tpot"] == pytest.approx(0.5)
    assert m["tokens_per_s"] == pytest.approx(5 / 2.5)
