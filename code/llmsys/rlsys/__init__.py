"""RL training-systems utilities (Chapter 46).

packing   -- FFD bin packing, packed (varlen) forward, packed log-prob pass
sim       -- event-driven simulator: colocated / sync / async / partial rollouts
mismatch  -- train/inference mismatch metrics and truncated/masked IS corrections
sandbox   -- subprocess-based code-execution reward (teaching sandbox)
"""
from .mismatch import clipped_pg_loss, is_weights, mismatch_metrics, token_logprobs
from .packing import (build_packed_row, efficiency_report, ffd_pack, packed_completion_logprobs,
                      varlen_forward)
from .sandbox import run_python_reward, score_batch
from .sim import SimConfig, simulate, weight_sync_seconds
