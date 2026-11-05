"""llmsys.compile: tools for looking inside ML compilers (Chapter 23).

    fx_passes      graph rewriting on torch.fx graphs: pattern matching, an RMSNorm fusion pass, a
                   SiLU*gate fusion pass, dead-code elimination, a pass pipeline, graph statistics
    dynamo_tools   inspecting torch.compile: captured graphs, graph breaks, recompiles, guards,
                   Inductor's generated code and the kernels it emitted
    export_tools   torch.export with dynamic shapes, ONNX export + ONNX Runtime, and a CUDA-graph
                   decode step over a static KV cache

Everything imports on a CPU-only machine; ONNX, ONNX Runtime and the CUDA-graph helpers expose
``HAS_*`` flags instead of failing at import time.
"""
from __future__ import annotations

from . import dynamo_tools, export_tools, fx_passes  # noqa: F401
from .dynamo_tools import (capture_graphs, count_recompiles, explain, inductor_output_code,  # noqa: F401
                           kernel_names, list_guards, summarize_kernels)
from .export_tools import (HAS_ONNX, HAS_ORT, CUDAGraphDecoder, compare_ort_torch,  # noqa: F401
                           export_onnx, export_program, graph_decode_step, ort_session)
from .fx_passes import (eliminate_dead_code, fuse_rmsnorm, fuse_silu_mul, fused_rmsnorm,  # noqa: F401
                        fused_silu_mul, fusion_backend, graph_stats, run_passes)
