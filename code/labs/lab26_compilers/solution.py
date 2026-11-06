"""Lab 26 (reference solution): looking inside ML compilers.

Seven tasks, each a piece of what a compiler stack does to our models:

    Task 1  canonical_op / match_rmsnorm / fuse_rmsnorm      pattern matching on an FX graph
    Task 2  fuse_silu_mul / run_pipeline                      a second fusion and a pass pipeline
    Task 3  count_graph_breaks / count_recompiles             what Dynamo captures and when it gives up
    Task 4  export_gpt_dynamic                                torch.export with a symbolic sequence length
    Task 5  export_to_onnx / run_onnx                         ONNX as an exchange format, ONNX Runtime
    Task 6  inductor_kernels                                  reading the code Inductor generated
    Task 7  capture_decode_graph                              a CUDA-graph decode step over a static cache

The reference implementations in ``llmsys.compile`` do the same things with more generality; the
tests compare this file against them.
"""
from __future__ import annotations

import operator
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch._dynamo as dynamo
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.node import Node

from llmsys.ops import rmsnorm as rmsnorm_op
from llmsys.ops import silu_mul as silu_mul_op


# ============================================================================ Task 1: RMSNorm fusion
_FUNCS = {operator.mul: "mul", torch.mul: "mul", operator.add: "add", torch.add: "add",
          operator.pow: "pow", torch.pow: "pow", torch.rsqrt: "rsqrt", torch.mean: "mean", F.silu: "silu"}
_METHODS = {"mul": "mul", "add": "add", "pow": "pow", "rsqrt": "rsqrt", "mean": "mean",
            "to": "cast", "float": "cast"}


def canonical_op(node: Node) -> Optional[str]:
    """One name per operation, however the graph spells it.

    Dynamo records ``x * y`` as ``call_function(operator.mul)``, ``x.pow(2)`` as
    ``call_method("pow")`` and ``x.to(torch.float32)`` as ``call_method("to")``. A pattern matcher
    that compared ``node.target`` directly would need one rule per spelling, so we normalise first.
    Returns None for nodes that are not operations (placeholders, outputs, attributes).
    """
    if node.op == "call_function":
        return _FUNCS.get(node.target, getattr(node.target, "__name__", None))
    if node.op == "call_method":
        return _METHODS.get(node.target, node.target)
    return None


def _skip_casts(node: Node) -> Node:
    while isinstance(node, Node) and canonical_op(node) == "cast":
        node = node.args[0]
    return node


def match_rmsnorm(node: Node) -> Optional[Tuple[Node, Node, float]]:
    """If ``node`` is the last multiply of ``(x * rsqrt(mean(x^2, -1, keepdim) + eps)) * w``,
    return ``(x, w, eps)`` (``x`` is the fp32 operand that feeds the square); else None.

    We walk the graph backwards from the candidate node, checking one operation per step and
    tolerating either operand order of the commutative multiplies.
    """
    if canonical_op(node) != "mul" or len(node.args) != 2:
        return None
    for inner, w in ((node.args[0], node.args[1]), (node.args[1], node.args[0])):
        if not isinstance(inner, Node) or canonical_op(inner) != "mul":
            continue
        a, b = inner.args
        x, r = (b, a) if canonical_op(a) == "rsqrt" else (a, b)
        if not isinstance(r, Node) or canonical_op(r) != "rsqrt":
            continue
        add = r.args[0]
        if canonical_op(add) != "add":
            continue
        mean, eps = add.args if isinstance(add.args[1], (int, float)) else add.args[::-1]
        if not isinstance(eps, (int, float)) or canonical_op(mean) != "mean":
            continue
        dim = mean.kwargs.get("dim", mean.args[1] if len(mean.args) > 1 else None)
        keep = mean.kwargs.get("keepdim", mean.args[2] if len(mean.args) > 2 else False)
        if dim != -1 or not keep:
            continue
        sq = mean.args[0]
        squared = (canonical_op(sq) == "pow" and sq.args[0] is x and sq.args[1] == 2) or \
                  (canonical_op(sq) == "mul" and sq.args[0] is x and sq.args[1] is x)
        if squared and isinstance(w, Node):
            return x, w, float(eps)
    return None


def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """What the pattern is replaced with: one call into ``llmsys.ops`` (python, cpp or cuda)."""
    return rmsnorm_op.rmsnorm(x, weight, eps)


def fuse_rmsnorm(gm: fx.GraphModule) -> int:
    """Replace every matched RMSNorm subgraph by one ``fused_rmsnorm`` node; return the count.

    The model's RMSNorm up-casts to fp32 on entry and casts back on exit. When the only user of
    the matched multiply is that cast back to the original dtype, the fused operator absorbs both
    casts (it computes in fp32 internally and returns the input dtype); otherwise it is fed the
    fp32 tensor and returns fp32, which keeps the dtype of every surviving node unchanged. The
    old nodes are left in place for the dead-code pass to remove.
    """
    n = 0
    for node in list(gm.graph.nodes):
        m = match_rmsnorm(node)
        if m is None:
            continue
        x, w, eps = m
        target, inp = node, x
        users = list(node.users)
        if len(users) == 1 and canonical_op(users[0]) == "cast":
            raw = _skip_casts(x)
            raw_dtype = getattr(raw.meta.get("example_value"), "dtype", None)
            want = next((a for a in users[0].args[1:] if isinstance(a, torch.dtype)), None)
            if want is not None and want == raw_dtype:
                target, inp = users[0], raw
        with gm.graph.inserting_before(target):
            new = gm.graph.call_function(fused_rmsnorm, (inp, _skip_casts(w), eps))
        new.meta = dict(target.meta)
        target.replace_all_uses_with(new)
        n += 1
    if n:
        gm.graph.lint()
        gm.recompile()
    return n


# ============================================================================ Task 2: SiLU*gate and the pipeline
def fused_silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return silu_mul_op.silu_mul(a, b)


def fuse_silu_mul(gm: fx.GraphModule) -> int:
    """``silu(a) * b`` -> ``fused_silu_mul(a, b)``; return how many were rewritten."""
    n = 0
    for node in list(gm.graph.nodes):
        if canonical_op(node) != "mul" or len(node.args) != 2:
            continue
        for s, b in ((node.args[0], node.args[1]), (node.args[1], node.args[0])):
            if isinstance(s, Node) and isinstance(b, Node) and canonical_op(s) == "silu":
                with gm.graph.inserting_before(node):
                    new = gm.graph.call_function(fused_silu_mul, (s.args[0], b))
                new.meta = dict(node.meta)
                node.replace_all_uses_with(new)
                n += 1
                break
    if n:
        gm.graph.lint()
        gm.recompile()
    return n


def eliminate_dead_code(gm: fx.GraphModule) -> int:
    before = len(gm.graph.nodes)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return before - len(gm.graph.nodes)


def run_pipeline(gm: fx.GraphModule, passes: Sequence[Callable[[fx.GraphModule], int]] = (
        fuse_rmsnorm, fuse_silu_mul, eliminate_dead_code)) -> List[Dict[str, object]]:
    """Apply the passes in order; return one record per pass: name, matches, nodes before/after, ms.

    Order matters: the fusions leave the old subgraphs dangling (no users), and only the final
    dead-code pass removes them, so node counts go *up* before they come down.
    """
    log = []
    for p in passes:
        before = len(gm.graph.nodes)
        t0 = time.perf_counter()
        matches = p(gm)
        log.append(dict(name=p.__name__, matches=int(matches), nodes_before=before,
                        nodes_after=len(gm.graph.nodes), ms=1e3 * (time.perf_counter() - t0)))
    return log


# ============================================================================ Task 3: Dynamo
def count_graph_breaks(fn: Callable, *args) -> Tuple[int, int, List[str]]:
    """(graphs, graph breaks, one-line reasons) for one call of ``fn`` under Dynamo."""
    dynamo.reset()
    ex = dynamo.explain(fn)(*args)
    reasons = [r.reason.splitlines()[0].strip() for r in ex.break_reasons]
    return ex.graph_count, ex.graph_break_count, reasons


def count_recompiles(fn: Callable, inputs: Sequence[tuple], dynamic: Optional[bool] = None) -> int:
    """How many times Dynamo compiled ``fn`` while calling it on every tuple in ``inputs``.

    A backend is invoked once per compiled graph. We pass a backend that counts its invocations
    and returns the graph unchanged, so the count is exactly the number of (re)compiles: a guard
    failure (new shape, new Python constant, new dtype) shows up as one more call.
    """
    dynamo.reset()
    calls = []

    def counting_backend(gm: fx.GraphModule, example_inputs):
        calls.append(len(example_inputs))
        return gm.forward

    compiled = torch.compile(fn, backend=counting_backend, dynamic=dynamic)
    for a in inputs:
        compiled(*a)
    return len(calls)


# ============================================================================ Task 4: torch.export
def export_gpt_dynamic(model: nn.Module, example_ids: torch.Tensor, max_seq: int):
    """Export ``model(ids)`` with ``ids`` of shape ``(B, T)`` and ``T`` symbolic in ``[1, max_seq]``.

    ``torch.export`` traces once with fake tensors whose sequence dimension is a symbol ``s``; every
    shape computation becomes an expression in ``s`` and every data-dependent branch on ``s``
    becomes a guard the exporter must prove from the declared range. The model raises when
    ``T > context_length``: the bound ``max_seq <= context_length`` lets the exporter discharge
    that check statically instead of failing with an unresolvable guard.
    """
    from torch.export import Dim

    seq = Dim("seq", min=1, max=max_seq)
    return torch.export.export(model.eval(), (example_ids,), dynamic_shapes={"in_idx": {1: seq}})


# ============================================================================ Task 5: ONNX
def export_to_onnx(model: nn.Module, example_ids: torch.Tensor, path: str) -> str:
    """Write ``model`` to ``path`` as ONNX with dynamic batch and sequence dimensions."""
    from torch.export import Dim

    shapes = {"in_idx": {0: Dim("batch", min=1), 1: Dim("seq", min=1, max=model.cfg["context_length"])}}
    prog = torch.onnx.export(model.eval(), (example_ids,), dynamo=True, dynamic_shapes=shapes,
                             opset_version=18, optimize=True)
    prog.save(path)
    return path


def run_onnx(path: str, ids: torch.Tensor) -> torch.Tensor:
    """Run the ONNX file in ONNX Runtime (CPU execution provider) and return the logits as a tensor."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return torch.from_numpy(sess.run(None, {name: ids.numpy()})[0])


# ============================================================================ Task 6: Inductor's code
def inductor_kernels(fn: Callable, *args) -> Tuple[List[str], str]:
    """Compile ``fn`` with Inductor (static shapes), run it once, return (kernel names, source).

    Inductor writes each compiled graph as a Python module: a wrapper that allocates buffers and
    launches kernels, with the kernel sources inline (Triton on a GPU, C++/OpenMP on the CPU).
    We intercept the module as it is loaded. Kernel names encode what was fused:
    ``cpp_fused_add_mean_mul_pow_rsqrt_0`` is one kernel for the whole RMSNorm.
    """
    import re
    from torch._inductor import codecache

    sources: List[str] = []
    orig = codecache.PyCodeCache.load_by_key_path

    def hook(key, path, *a, **k):
        with open(path) as fh:
            sources.append(fh.read())
        return orig(key, path, *a, **k)

    dynamo.reset()
    codecache.PyCodeCache.load_by_key_path = hook
    try:
        torch.compile(fn, dynamic=False)(*args)
    finally:
        codecache.PyCodeCache.load_by_key_path = orig
    code = "\n".join(sources)
    names = re.findall(r"^(\w+) = async_compile\.(?:triton|cpp_pybinding|cpp)\(", code, re.M)
    return names, code


# ============================================================================ Task 7: CUDA graphs
class DecodeGraph:
    """A captured decode step: static input buffer, device-side position, replay per token."""

    def __init__(self, model: nn.Module, cache, start_pos: int):
        from llmsys.compile.export_tools import graph_decode_step
        from llmsys.kvcache import StaticKVCache

        dev = cache.k[0].device
        B = cache.k[0].shape[0]
        self.cache = cache
        self.ids = torch.zeros(B, 1, dtype=torch.long, device=dev)
        self.pos = torch.tensor(start_pos, dtype=torch.long, device=dev)
        # Warm-up on a side stream against a scratch cache: lazy initialisation (cuBLAS handles,
        # allocator segments) must not be recorded, and must not touch the real cache.
        scratch = StaticKVCache(len(cache.k), B, cache.k[0].shape[1], cache.max_len, cache.k[0].shape[3],
                                dtype=cache.k[0].dtype, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                graph_decode_step(model, self.ids, self.pos, scratch)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.logits = graph_decode_step(model, self.ids, self.pos, cache)

    def step(self, next_ids: torch.Tensor) -> torch.Tensor:
        self.ids.copy_(next_ids)        # into the captured address
        self.graph.replay()             # one launch for the whole step
        self.pos += 1                   # device-side counter: no host sync, no recapture
        for i in range(len(self.cache.lengths)):
            self.cache.lengths[i] += 1
        return self.logits


def capture_decode_graph(model: nn.Module, cache, start_pos: int) -> DecodeGraph:
    """Capture one decode step of ``model`` over ``cache`` (already prefilled to ``start_pos``).

    Everything the step depends on must live at a fixed address and have a fixed shape: the token
    buffer, the position (a 0-dim device tensor, never a Python int), and the cache slots, which are
    written with ``index_copy_`` at ``pos`` and read in full with the slots beyond ``pos`` masked.
    """
    return DecodeGraph(model, cache, start_pos)
