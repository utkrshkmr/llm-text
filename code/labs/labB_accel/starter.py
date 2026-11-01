"""Lab B starter. Fill in every TODO.

Accelerator models for Appendix B: systolic arrays and 2-D meshes of cores.

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
    raise NotImplementedError("TODO (simulate_output_stationary): cycle-by-cycle: shift a right, b down, inject skewed A/B at the edges, accumulate; T=M+N+K-2")


def simulate_weight_stationary(X: np.ndarray, W: np.ndarray,
                               include_weight_load: bool = False) -> SystolicResult:
    """K x N array holds W (K x N) stationary; Y = X @ W for X of shape M x K.

    x[m, k] enters row k at cycle m + k and moves right; partial sums move down each
    column, so y[m, n] leaves the bottom of column n at cycle m + n + K - 1. Loading the
    weights takes K extra cycles (one row per cycle) unless double-buffered.
    """
    raise NotImplementedError("TODO (simulate_weight_stationary): x moves right (x[m,k] enters row k at cycle m+k), partial sums move down, y[m,n] exits at m+n+K-1")


def tiled_systolic_cycles(M: int, K: int, N: int, R: int = 128, C: int = 128,
                          weight_load_overlapped: bool = True) -> Dict[str, float]:
    """Cycles for Y(MxN) = X(MxK) W(KxN) on an R x C weight-stationary array.

    W is cut into ceil(K/R) x ceil(N/C) tiles; for each tile all M rows of X stream
    through (M + R + C - 2 cycles incl. fill/drain). Partial tiles are padded, which is
    why model dimensions are chosen as multiples of the array size.
    """
    raise NotImplementedError("TODO (tiled_systolic_cycles): ceil-tiles of W, per tile M+R+C-2 cycles (+R if load not overlapped); return tiles, cycles, utilization, padding_efficiency")


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
    raise NotImplementedError("TODO (cannon_matmul): skew A rows left by i and B cols up by j, then p steps of local matmul + one-hop shifts; count hop bytes")


def summa_traffic(M: int, K: int, N: int, p: int, dtype_bytes: int = 2,
                  multicast: bool = True) -> float:
    """Link bytes for SUMMA on a p x p mesh: at each of p steps, one column of cores
    broadcasts its A block along its row and one row broadcasts its B block down its
    column. With hardware multicast a broadcast along a line of p cores costs p-1 link
    traversals of one block; with unicast from the root it costs sum_{d=1}^{p-1} d."""
    raise NotImplementedError("TODO (summa_traffic): p steps x p lines x (p-1 hops with multicast, or p(p-1)/2 without) x (A block + B block bytes)")


def mesh_compute_bound_block(flops_per_core: float, link_bw: float,
                             dtype_bytes: int = 2) -> float:
    """Smallest square block size b such that a b x b x b local matmul (2 b^3 FLOPs) takes
    at least as long as moving two b x b blocks over one link (Cannon step)."""
    raise NotImplementedError("TODO (mesh_compute_bound_block): b = F * bytes / BW")


# =========================================================================== tile layouts
def tilize(x: np.ndarray, tile: int = 32, face: int = 16) -> np.ndarray:
    """Row-major (H, W) matrix -> flat buffer in tile layout.

    Tiles of ``tile x tile`` are stored one after another in row-major tile order. Inside a
    tile, the four ``face x face`` faces are stored in the order top-left, top-right,
    bottom-left, bottom-right, each face row-major. This mirrors the 32x32 tile / 16x16 face
    layout that Tenstorrent's Tensix cores consume natively. Pads H, W up to multiples of tile.
    """
    raise NotImplementedError("TODO (tilize): pad to tile multiples; emit tiles row-major, each as 4 faces (TL, TR, BL, BR), each face row-major")


def untilize(buf: np.ndarray, H: int, W: int, tile: int = 32, face: int = 16) -> np.ndarray:
    """Inverse of :func:`tilize` (drops padding)."""
    raise NotImplementedError("TODO (untilize): inverse of tilize, dropping the padding")


def decode_tokens_per_s_bound(n_params: float, bytes_per_param: float, bw: float,
                              kv_bytes_per_token_step: float = 0.0) -> float:
    """Upper bound on single-stream decode speed: every step must stream all weights (and
    the KV cache) from off-chip memory once, so tokens/s <= bw / bytes_per_step."""
    raise NotImplementedError("TODO (decode_tokens_per_s_bound): bw / (weight bytes + kv bytes per step)")


def ridge_point(name: str) -> float:
    """FLOPs per byte at which a kernel turns from memory- to compute-bound."""
    raise NotImplementedError("TODO (ridge_point): flops / bw of the named accelerator")
