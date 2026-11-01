"""Accelerator models for Appendix B: systolic arrays and 2-D meshes of cores.

* ``simulate_output_stationary`` / ``simulate_weight_stationary``: cycle-level
  simulations of an R x C systolic array (register shifts every cycle), returning the
  result, the cycle count and the utilisation. The TPU's matrix unit (MXU) is
  weight-stationary; output-stationary arrays are common in research accelerators.
* ``tiled_systolic_cycles``: the analytic model for matrices larger than the array
  (tiling, fill/drain, padding), used to reason about prefill vs. decode on TPUs.
* ``cannon_matmul`` / ``summa_traffic``: matmul on a p x p mesh of cores that each own a
  block in local SRAM and exchange blocks with neighbours over a network-on-chip, the
  execution model of Tenstorrent Tensix grids, Cerebras and many dataflow designs.
  The simulation counts link traffic so data movement can be compared between schemes.
* ``ACCELERATORS``: published headline numbers (dense, peak) used for rooflines.

Everything is pure NumPy so it runs anywhere; it models *timing and data movement*, not
the numerics of any particular chip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# Published peak numbers (dense; vendor datasheets/announcements, rounded). Treat as
# order-of-magnitude anchors: precisions and measurement conventions differ by vendor.
ACCELERATORS: Dict[str, Dict[str, float]] = {
    "H100-SXM (bf16)":        dict(flops=989e12, bw=3.35e12, mem=80e9),
    "TPU v4 (bf16)":          dict(flops=275e12, bw=1.2e12, mem=32e9),
    "TPU v5e (bf16)":         dict(flops=197e12, bw=0.82e12, mem=16e9),
    "TPU v5p (bf16)":         dict(flops=459e12, bw=2.76e12, mem=95e9),
    "Wormhole n150 (fp8)":    dict(flops=262e12, bw=0.288e12, mem=12e9),
    "Wormhole n300 (fp8)":    dict(flops=466e12, bw=0.576e12, mem=24e9),
    "GB10 / DGX Spark (fp4*)": dict(flops=1000e12, bw=0.273e12, mem=128e9),
}


# =========================================================================== systolic
@dataclass
class SystolicResult:
    out: np.ndarray
    cycles: int
    macs: int
    pes: int

    @property
    def utilization(self) -> float:
        return self.macs / (self.pes * self.cycles)


def simulate_output_stationary(A: np.ndarray, B: np.ndarray) -> SystolicResult:
    """M x N array of PEs computes C = A @ B; PE(i, j) accumulates C[i, j] in place.

    Row i of A enters the left edge delayed by i cycles; column j of B enters the top edge
    delayed by j cycles ("skewing"), so A[i, k] and B[k, j] meet in PE(i, j) at cycle
    t = i + j + k. Each cycle every PE multiplies its two input registers, adds into its
    accumulator, and passes a to the right and b downward.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    a_reg = np.zeros((M, N), dtype=np.float64)
    b_reg = np.zeros((M, N), dtype=np.float64)
    acc = np.zeros((M, N), dtype=np.float64)
    T = M + N + K - 2                      # last product at t = (M-1)+(N-1)+(K-1)
    for t in range(T):
        a_new = np.zeros_like(a_reg)
        b_new = np.zeros_like(b_reg)
        a_new[:, 1:] = a_reg[:, :-1]       # shift right
        b_new[1:, :] = b_reg[:-1, :]       # shift down
        for i in range(M):                 # inject skewed A at the left edge
            k = t - i
            if 0 <= k < K:
                a_new[i, 0] = A[i, k]
        for j in range(N):                 # inject skewed B at the top edge
            k = t - j
            if 0 <= k < K:
                b_new[0, j] = B[k, j]
        acc += a_new * b_new
        a_reg, b_reg = a_new, b_new
    return SystolicResult(acc, T, M * N * K, M * N)


def simulate_weight_stationary(X: np.ndarray, W: np.ndarray,
                               include_weight_load: bool = False) -> SystolicResult:
    """K x N array holds W (K x N) stationary; Y = X @ W for X of shape M x K.

    x[m, k] enters row k at cycle m + k and moves right; partial sums move down each
    column, so y[m, n] leaves the bottom of column n at cycle m + n + K - 1. Loading the
    weights takes K extra cycles (one row per cycle) unless double-buffered.
    """
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    x_reg = np.zeros((K, N))
    p_reg = np.zeros((K, N))
    Y = np.zeros((M, N))
    T = M + N + K - 2
    for t in range(T):
        x_new = np.zeros_like(x_reg)
        x_new[:, 1:] = x_reg[:, :-1]
        for k in range(K):
            m = t - k
            if 0 <= m < M:
                x_new[k, 0] = X[m, k]
        p_in = np.zeros_like(p_reg)
        p_in[1:, :] = p_reg[:-1, :]
        p_new = p_in + x_new * W
        for n in range(N):
            m = t - (K - 1) - n
            if 0 <= m < M:
                Y[m, n] = p_new[K - 1, n]
        x_reg, p_reg = x_new, p_new
    cycles = T + (K if include_weight_load else 0)
    return SystolicResult(Y, cycles, M * N * K, K * N)


def tiled_systolic_cycles(M: int, K: int, N: int, R: int = 128, C: int = 128,
                          weight_load_overlapped: bool = True) -> Dict[str, float]:
    """Cycles for Y(MxN) = X(MxK) W(KxN) on an R x C weight-stationary array.

    W is cut into ceil(K/R) x ceil(N/C) tiles; for each tile all M rows of X stream
    through (M + R + C - 2 cycles incl. fill/drain). Partial tiles are padded, which is
    why model dimensions are chosen as multiples of the array size.
    """
    tk, tn = math.ceil(K / R), math.ceil(N / C)
    per_tile = M + R + C - 2 + (0 if weight_load_overlapped else R)
    cycles = tk * tn * per_tile
    util = (M * K * N) / (R * C * cycles)
    return dict(tiles=tk * tn, cycles=cycles, utilization=util,
                padding_efficiency=(K * N) / (tk * R * tn * C))


# =========================================================================== mesh
@dataclass
class MeshResult:
    out: np.ndarray
    steps: int
    hop_bytes: float          # total bytes x hops moved over NoC links
    sram_bytes_per_core: float


def cannon_matmul(A: np.ndarray, B: np.ndarray, p: int, dtype_bytes: int = 2) -> MeshResult:
    """Cannon's algorithm on a p x p torus of cores.

    Core (i, j) owns blocks A_ij and B_ij. After an initial skew (row i of A shifted left
    by i, column j of B shifted up by j), each of p steps does a local block matmul and
    then passes its A block one hop left and its B block one hop up. Every transfer is
    nearest-neighbour, so per-step traffic is constant and links are evenly loaded.
    """
    M, K = A.shape
    _, N = B.shape
    assert M % p == 0 and K % p == 0 and N % p == 0
    bm, bk, bn = M // p, K // p, N // p
    Ab = [[A[i*bm:(i+1)*bm, j*bk:(j+1)*bk].copy() for j in range(p)] for i in range(p)]
    Bb = [[B[i*bk:(i+1)*bk, j*bn:(j+1)*bn].copy() for j in range(p)] for i in range(p)]
    Cb = [[np.zeros((bm, bn)) for _ in range(p)] for _ in range(p)]
    hop = 0.0
    a_blk, b_blk = bm * bk * dtype_bytes, bk * bn * dtype_bytes
    # initial skew; on a torus a shift by s costs min(s, p - s) hops per block
    Ab = [[Ab[i][(j + i) % p] for j in range(p)] for i in range(p)]
    Bb = [[Bb[(i + j) % p][j] for j in range(p)] for i in range(p)]
    for i in range(p):
        hop += p * min(i, p - i) * a_blk
    for j in range(p):
        hop += p * min(j, p - j) * b_blk
    for step in range(p):
        for i in range(p):
            for j in range(p):
                Cb[i][j] += Ab[i][j] @ Bb[i][j]
        if step < p - 1:
            Ab = [[Ab[i][(j + 1) % p] for j in range(p)] for i in range(p)]
            Bb = [[Bb[(i + 1) % p][j] for j in range(p)] for i in range(p)]
            hop += p * p * (a_blk + b_blk)
    C = np.block(Cb)
    sram = (a_blk + b_blk + bm * bn * 4) * 2          # double-buffered in/out blocks, fp32 acc
    return MeshResult(C, p, hop, sram)


def summa_traffic(M: int, K: int, N: int, p: int, dtype_bytes: int = 2,
                  multicast: bool = True) -> float:
    """Link bytes for SUMMA on a p x p mesh: at each of p steps, one column of cores
    broadcasts its A block along its row and one row broadcasts its B block down its
    column. With hardware multicast a broadcast along a line of p cores costs p-1 link
    traversals of one block; with unicast from the root it costs sum_{d=1}^{p-1} d."""
    a_blk = (M // p) * (K // p) * dtype_bytes
    b_blk = (K // p) * (N // p) * dtype_bytes
    per_line = (p - 1) if multicast else p * (p - 1) / 2
    return p * p * per_line * (a_blk + b_blk)    # p steps x p rows/cols x line cost


def mesh_compute_bound_block(flops_per_core: float, link_bw: float,
                             dtype_bytes: int = 2) -> float:
    """Smallest square block size b such that a b x b x b local matmul (2 b^3 FLOPs) takes
    at least as long as moving two b x b blocks over one link (Cannon step)."""
    # 2 b^3 / F >= 2 b^2 * bytes / BW  ->  b >= F * bytes / BW
    return flops_per_core * dtype_bytes / link_bw


# =========================================================================== tile layouts
def tilize(x: np.ndarray, tile: int = 32, face: int = 16) -> np.ndarray:
    """Row-major (H, W) matrix -> flat buffer in tile layout.

    Tiles of ``tile x tile`` are stored one after another in row-major tile order. Inside a
    tile, the four ``face x face`` faces are stored in the order top-left, top-right,
    bottom-left, bottom-right, each face row-major. This mirrors the 32x32 tile / 16x16 face
    layout that Tenstorrent's Tensix cores consume natively. Pads H, W up to multiples of tile.
    """
    H, W = x.shape
    Hp, Wp = -(-H // tile) * tile, -(-W // tile) * tile
    xp = np.zeros((Hp, Wp), dtype=x.dtype)
    xp[:H, :W] = x
    out = []
    for ti in range(Hp // tile):
        for tj in range(Wp // tile):
            t = xp[ti*tile:(ti+1)*tile, tj*tile:(tj+1)*tile]
            for fi in range(tile // face):
                for fj in range(tile // face):
                    out.append(t[fi*face:(fi+1)*face, fj*face:(fj+1)*face].reshape(-1))
    return np.concatenate(out)


def untilize(buf: np.ndarray, H: int, W: int, tile: int = 32, face: int = 16) -> np.ndarray:
    """Inverse of :func:`tilize` (drops padding)."""
    Hp, Wp = -(-H // tile) * tile, -(-W // tile) * tile
    xp = np.zeros((Hp, Wp), dtype=buf.dtype)
    k = 0
    for ti in range(Hp // tile):
        for tj in range(Wp // tile):
            for fi in range(tile // face):
                for fj in range(tile // face):
                    r0, c0 = ti*tile + fi*face, tj*tile + fj*face
                    xp[r0:r0+face, c0:c0+face] = buf[k:k+face*face].reshape(face, face)
                    k += face * face
    return xp[:H, :W]


def decode_tokens_per_s_bound(n_params: float, bytes_per_param: float, bw: float,
                              kv_bytes_per_token_step: float = 0.0) -> float:
    """Upper bound on single-stream decode speed: every step must stream all weights (and
    the KV cache) from off-chip memory once, so tokens/s <= bw / bytes_per_step."""
    return bw / (n_params * bytes_per_param + kv_bytes_per_token_step)


def ridge_point(name: str) -> float:
    """FLOPs per byte at which a kernel turns from memory- to compute-bound."""
    a = ACCELERATORS[name]
    return a["flops"] / a["bw"]
