"""Lab 18 reference solution: distributed training primitives from scratch.

The full implementations live in ``llmsys/dist`` (so later chapters can import them);
this file re-exports exactly the API the lab asks you to build in ``starter.py``.
Read the sources alongside Chapter 18:

    llmsys/dist/collectives.py      Task 1  naive + ring all-reduce
    llmsys/dist/costs.py            Task 2  alpha-beta model, ZeRO memory, bubbles
    llmsys/dist/ddp.py              Task 3  bucketed DDP with overlap
    llmsys/dist/zero.py             Task 4  ZeRO-1/2 optimizer
    llmsys/dist/tensor_parallel.py  Task 5  column/row parallel MLP
    llmsys/dist/pipeline.py         Task 6  GPipe / 1F1B schedules
"""
from llmsys.dist.collectives import (naive_allreduce, ring_all_gather,  # noqa: F401
                                     ring_allreduce, ring_reduce_scatter)
from llmsys.dist.costs import (allreduce_time, pipeline_bubble_fraction,  # noqa: F401
                               zero_memory_per_gpu)
from llmsys.dist.ddp import BucketedDDP, allreduce_gradients, broadcast_parameters  # noqa: F401
from llmsys.dist.pipeline import (bubble_fraction, gpipe_order,  # noqa: F401
                                  one_f_one_b_order, peak_inflight, simulate)
from llmsys.dist.tensor_parallel import (ColumnParallelLinear,  # noqa: F401
                                         RowParallelLinear, TensorParallelMLP)
from llmsys.dist.zero import ZeroAdamW, partition_parameters  # noqa: F401
