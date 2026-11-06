"""Lab 65 reference solution: RL training-systems building blocks.

Re-exports the reference implementations from ``llmsys.rlsys``:

Task 1  ffd_pack, efficiency_report            -- packing the log-prob / training pass
Task 2  block_causal_mask, packed_completion_logprobs
Task 3  is_weights, mismatch_metrics, clipped_pg_loss, token_logprobs
Task 4  run_generation, simulate               -- colocated / sync / async / partial
Task 5  run_python_reward, score_batch         -- sandboxed code-execution reward
"""
from llmsys.rlsys.mismatch import clipped_pg_loss, is_weights, mismatch_metrics, token_logprobs
from llmsys.rlsys.packing import (block_causal_mask, build_packed_row, efficiency_report,
                                  ffd_pack, packed_completion_logprobs, varlen_forward)
from llmsys.rlsys.sandbox import run_python_reward, score_batch
from llmsys.rlsys.sim import (Rollout, SimConfig, generation_occupancy, run_generation,
                              simulate, weight_sync_seconds)

__all__ = ["clipped_pg_loss", "is_weights", "mismatch_metrics", "token_logprobs",
           "block_causal_mask", "build_packed_row", "efficiency_report", "ffd_pack",
           "packed_completion_logprobs", "varlen_forward", "run_python_reward", "score_batch",
           "Rollout", "SimConfig", "generation_occupancy", "run_generation", "simulate",
           "weight_sync_seconds"]
