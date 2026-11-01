"""Lab B tests (pure NumPy, run in seconds).

    LAB_IMPL=starter  python -m pytest -q labs/labB_accel
    LAB_IMPL=solution python -m pytest -q labs/labB_accel
"""
import math

import numpy as np
import pytest

from _labutil import load_impl

lab = load_impl(__file__)
rng = np.random.default_rng(0)


@pytest.mark.parametrize("M,K,N", [(1, 1, 1), (5, 7, 4), (3, 2, 6), (8, 8, 8)])
def test_output_stationary(M, K, N):
    A, B = rng.standard_normal((M, K)), rng.standard_normal((K, N))
    r = lab.simulate_output_stationary(A, B)
    assert np.allclose(r.out, A @ B)
    assert r.cycles == M + N + K - 2 or (M + N + K - 2 == 1 and r.cycles == 1)
    assert math.isclose(r.utilization, M * N * K / (M * N * r.cycles))


@pytest.mark.parametrize("M,K,N", [(1, 3, 2), (6, 4, 5), (9, 2, 3)])
def test_weight_stationary(M, K, N):
    X, W = rng.standard_normal((M, K)), rng.standard_normal((K, N))
    r = lab.simulate_weight_stationary(X, W)
    assert np.allclose(r.out, X @ W)
    assert r.cycles == M + N + K - 2 and r.pes == K * N
    assert lab.simulate_weight_stationary(X, W, include_weight_load=True).cycles == r.cycles + K


def test_tiled_model():
    # one tile that exactly fits the array reproduces the simulator's cycle count
    M, K, N = 10, 4, 6
    t = lab.tiled_systolic_cycles(M, K, N, R=K, C=N)
    assert t["tiles"] == 1 and t["cycles"] == M + K + N - 2
    us = [lab.tiled_systolic_cycles(m, 4096, 4096)["utilization"] for m in (1, 64, 1024, 8192)]
    assert us == sorted(us) and us[0] < 0.01 and us[-1] > 0.95
    pad = lab.tiled_systolic_cycles(1000, 4100, 4096)
    assert pad["tiles"] == 33 * 32 and pad["padding_efficiency"] < 0.98


def test_tile_layout():
    x = np.arange(40 * 70, dtype=float).reshape(40, 70)
    buf = lab.tilize(x)
    assert buf.size == 64 * 96                          # padded to 32-multiples
    assert np.array_equal(buf[:16], x[0, :16])          # face 0, row 0
    assert np.array_equal(buf[16:32], x[1, :16])        # face 0, row 1
    assert np.array_equal(buf[256:272], x[0, 16:32])    # face 1 starts at column 16
    assert np.array_equal(buf[512:528], x[16, :16])     # face 2 starts at row 16
    assert np.array_equal(lab.untilize(buf, 40, 70), x)


def test_cannon_and_summa():
    p = 4
    A, B = rng.standard_normal((8, 12)), rng.standard_normal((12, 8))
    r = lab.cannon_matmul(A, B, p, dtype_bytes=2)
    assert np.allclose(r.out, A @ B) and r.steps == p
    a_blk, b_blk = 2 * 3 * 2, 3 * 2 * 2
    skew = sum(p * min(i, p - i) for i in range(p)) * (a_blk + b_blk)
    assert math.isclose(r.hop_bytes, skew + (p - 1) * p * p * (a_blk + b_blk))
    assert math.isclose(lab.summa_traffic(8, 12, 8, p), p * p * (p - 1) * (a_blk + b_blk))
    assert lab.summa_traffic(8, 12, 8, p, multicast=False) > lab.summa_traffic(8, 12, 8, p)


def test_bounds():
    assert math.isclose(lab.decode_tokens_per_s_bound(8e9, 1.0, 3.35e12), 3.35e12 / 8e9)
    assert math.isclose(lab.decode_tokens_per_s_bound(1e9, 2.0, 1e12, 1e9), 1e12 / 3e9)
    assert math.isclose(lab.ridge_point("H100-SXM (bf16)"), 989e12 / 3.35e12)
    assert math.isclose(lab.mesh_compute_bound_block(4e12, 64e9, 1), 62.5)
