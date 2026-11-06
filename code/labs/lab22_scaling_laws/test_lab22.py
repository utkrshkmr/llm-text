import math

import numpy as np
import pytest

from _labutil import load_impl
from llmsys.scaling import (GroundTruth, Run, RunConfig, SyntheticHardware,
                            SyntheticTrainingAPI, config_for_params)

lab = load_impl(__file__)

TRUE = dict(E=1.8, A=450.0, alpha=0.35, B=480.0, beta=0.29)


def chin(n, d, E=1.8, A=450.0, alpha=0.35, B=480.0, beta=0.29):
    return E + A / n ** alpha + B / d ** beta


def isoflop_runs(budgets=(1e17, 1e18, 1e19, 1e20), per=9, span=30.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    runs = []
    for c in budgets:
        mid = math.sqrt(c / 120)
        for n in np.geomspace(mid / math.sqrt(span), mid * math.sqrt(span), per):
            d = c / (6 * n)
            runs.append(Run(compute=c, params=n, tokens=d, loss=chin(n, d) + rng.normal(0, noise)))
    return runs


# ------------------------------------------------------------------ Task 1
def test_accounting_matches_model_and_12Ld2():
    cfg = RunConfig(n_layer=4, d_model=96, n_head=3, d_ff=256, tokens=512 * 8, batch_size=8)
    out = lab.count_params_and_flops(cfg)
    # build the same dense SwiGLU transformer with llmsys and count matmul weights
    from llmsys.qwen3 import Qwen3Model
    m = Qwen3Model(dict(vocab_size=cfg.vocab, context_length=64, emb_dim=96, n_heads=3,
                        n_layers=4, hidden_dim=256, head_dim=32, qk_norm=False, n_kv_groups=3,
                        rope_base=1e4, tie_embeddings=False))
    linear = sum(p.numel() for n_, p in m.trf_blocks.named_parameters() if p.dim() == 2)
    assert out["non_embedding"] == linear
    assert out["embedding"] == 2 * cfg.vocab * 96
    assert out["flops_per_token"] == pytest.approx(cfg.flops_per_token())
    assert out["train_flops"] == pytest.approx(cfg.flops_per_token() * cfg.tokens)
    sq = RunConfig(n_layer=6, d_model=384, n_head=6, d_ff=1024, tokens=512)
    assert lab.count_params_and_flops(sq)["non_embedding"] == 12 * 6 * 384 ** 2


# ------------------------------------------------------------------ Task 2
def test_power_law_exact():
    x = np.geomspace(1e3, 1e9, 7)
    k, a = lab.fit_power_law(x, 3.5 * x ** -0.27)
    assert k == pytest.approx(3.5, rel=1e-6) and a == pytest.approx(-0.27, abs=1e-9)


# ------------------------------------------------------------------ Task 3
def test_isoflop_optima_min_and_quadratic():
    runs = isoflop_runs(budgets=(1e18,), per=7)
    best = min(runs, key=lambda r: r.loss)
    mn = lab.isoflop_optima(runs, "min")[0]
    assert mn.n_opt == best.params and mn.loss == best.loss
    q = lab.isoflop_optima(isoflop_runs(budgets=(1e18,), per=15, span=8.0), "quadratic")[0]
    # analytic optimum along the IsoFLOP line, by dense search
    ns = np.geomspace(1e7, 1e10, 20000)
    true_n = ns[np.argmin(chin(ns, 1e18 / (6 * ns)))]
    assert abs(math.log(q.n_opt / true_n)) < 0.15


# ------------------------------------------------------------------ Task 4
def test_allocation_exponents():
    alloc = lab.fit_allocation(lab.isoflop_optima(isoflop_runs(per=15, span=10.0), "quadratic"))
    a_true = TRUE["beta"] / (TRUE["alpha"] + TRUE["beta"])
    assert alloc["N"][1] == pytest.approx(a_true, abs=0.04)
    assert alloc["N"][1] + alloc["D"][1] == pytest.approx(1.0, abs=0.02)   # C = 6ND
    assert alloc["L"][1] < 0


# ------------------------------------------------------------------ Task 5
def test_parametric_fit_recovers_truth():
    rng = np.random.default_rng(1)
    n = np.geomspace(1e6, 1e9, 6)
    d = np.geomspace(1e8, 1e11, 6)
    N, D = [x.ravel() for x in np.meshgrid(n, d)]
    L = chin(N, D) * np.exp(rng.normal(0, 0.002, N.shape))
    fit = lab.fit_chinchilla(N, D, L)
    assert fit.alpha == pytest.approx(0.35, abs=0.04)
    assert fit.beta == pytest.approx(0.29, abs=0.04)
    assert fit.E == pytest.approx(1.8, abs=0.08)


# ------------------------------------------------------------------ Task 6
def test_closed_form_optimum_matches_search():
    from llmsys.scaling import ChinchillaFit
    fit = ChinchillaFit(**TRUE)
    c = 3e21
    n_opt, d_opt = lab.chinchilla_optimal(fit, c)
    assert 6 * n_opt * d_opt == pytest.approx(c, rel=1e-6)
    ns = np.geomspace(1e7, 1e12, 200001)
    n_star = ns[np.argmin(fit.loss(ns, c / (6 * ns)))]
    assert n_opt == pytest.approx(n_star, rel=2e-3)


# ------------------------------------------------------------------ Task 7
def test_lr_basin_and_transfer():
    lrs = np.geomspace(1e-4, 1e-1, 9)
    loss = 3.0 + 0.05 * np.log(lrs / 2.5e-3) ** 2
    assert lab.fit_lr_optimum(lrs, loss) == pytest.approx(2.5e-3, rel=1e-6)
    ns = np.array([1e6, 1e7, 1e8])
    k, a = lab.fit_lr_scaling(ns, 0.4 * ns ** -0.3)
    assert a == pytest.approx(-0.3, abs=1e-9) and k == pytest.approx(0.4, rel=1e-6)


# ------------------------------------------------------------------ Task 8
def test_bootstrap_ci_covers_mean():
    x = list(np.random.default_rng(0).normal(5.0, 1.0, 200))
    lo, hi = lab.bootstrap_ci(lambda s: float(np.mean(s)), x, n_boot=300, level=0.95)
    assert lo < 5.0 < hi and hi - lo < 0.5


# ------------------------------------------------------------------ Task 9
def test_choose_final_config_respects_wallclock():
    from llmsys.scaling import ChinchillaFit
    hw = SyntheticHardware()
    fit = ChinchillaFit(**TRUE)
    small, _ = lab.choose_final_config(fit, (1.0, -0.3), 3600, hw)
    big, pred = lab.choose_final_config(fit, (1.0, -0.3), 48 * 3600, hw)
    assert hw.runtime(small) <= 3600 and hw.runtime(big) <= 48 * 3600
    assert big.non_embedding_params() > small.non_embedding_params()
    assert big.tokens > small.tokens and pred < float(fit.loss(small.non_embedding_params(), small.tokens))


# ------------------------------------------------------------------ Task 10
def test_run_study_end_to_end():
    api = SyntheticTrainingAPI(total_budget_seconds=6 * 3600)
    out = lab.run_study(api, 12 * 3600, lr_sizes=(3e6, 1e7), lr_grid=(1e-3, 3e-3, 9e-3, 2.7e-2),
                        budgets=(1e16, 4e16, 1.6e17), sizes_per_budget=5)
    b = api.budget()
    assert b["used_seconds"] <= b["total_budget_seconds"]
    score = api.score(12 * 3600)
    assert not score["over_time_budget"]
    assert score["regret"] < 0.03
    assert abs(score["prediction_error"]) < 0.1
