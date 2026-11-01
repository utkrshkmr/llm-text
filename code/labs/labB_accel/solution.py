"""Lab B reference solution: accelerator models (Appendix B).

The implementations live in ``llmsys/systolic.py``; this file re-exports the API that
``starter.py`` asks you to build.
"""
from llmsys.systolic import (ACCELERATORS, MeshResult, SystolicResult,  # noqa: F401
                             cannon_matmul, decode_tokens_per_s_bound,
                             mesh_compute_bound_block, ridge_point,
                             simulate_output_stationary, simulate_weight_stationary,
                             summa_traffic, tiled_systolic_cycles, tilize, untilize)
