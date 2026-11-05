"""Graph rewriting on torch.fx: a small fusion compiler for our models (Chapter 23).

The passes operate on any ``torch.fx.GraphModule``: the ones TorchDynamo hands to a backend
(``dynamo_tools.capture_graphs``), the ones ``torch.export`` produces, or a hand-built graph.
Node targets differ between these producers (``operator.mul`` vs ``torch.ops.aten.mul.Tensor``,
``Tensor.to`` vs ``prims.convert_element_type``), so every pass looks at nodes through
:func:`canonical`, which maps the many spellings of an operation to one name.

    fuse_rmsnorm      (x * rsqrt(mean(x^2, -1) + eps)) * w  ->  fused_rmsnorm(x, w, eps)
    fuse_silu_mul     silu(a) * b                           ->  fused_silu_mul(a, b)
    eliminate_dead_code
    run_passes        a pipeline with per-pass statistics
    fusion_backend    the pipeline as a torch.compile backend (eager execution of the fused graph)
    graph_stats       node counts and bytes of materialised intermediates (via ShapeProp)
"""
from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch.fx.node import Node

from llmsys.ops import rmsnorm as _rmsnorm_op

# ----------------------------------------------------------------------------- fusion targets
def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """The replacement for the RMSNorm subgraph: one ``llmsys.ops`` call (python/cpp/cuda backend)."""
    return _rmsnorm_op.rmsnorm(x, weight, eps)


def fused_silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """silu(a) * b as one node: the SwiGLU gate that Inductor (or a hand kernel) fuses into one pass."""
    return F.silu(a) * b


# ----------------------------------------------------------------------------- canonical op names
_FUNCTION_NAMES = {
    operator.mul: "mul", torch.mul: "mul", torch.Tensor.mul: "mul",
    operator.add: "add", torch.add: "add", torch.Tensor.add: "add",
    operator.pow: "pow", torch.pow: "pow", torch.Tensor.pow: "pow",
    torch.rsqrt: "rsqrt", torch.Tensor.rsqrt: "rsqrt",
    torch.mean: "mean", torch.Tensor.mean: "mean",
    F.silu: "silu", torch.nn.functional.silu: "silu",
    torch.Tensor.to: "cast", torch.Tensor.float: "cast",
    torch._C._nn.linear: "linear", F.linear: "linear", torch.matmul: "matmul", operator.matmul: "matmul",
    torch.softmax: "softmax", F.softmax: "softmax",
}
_METHOD_NAMES = {"mul": "mul", "add": "add", "pow": "pow", "rsqrt": "rsqrt", "mean": "mean",
                 "to": "cast", "float": "cast", "half": "cast", "bfloat16": "cast", "type": "cast",
                 "matmul": "matmul", "softmax": "softmax"}
_ATEN_NAMES = {"mul": "mul", "add": "add", "pow": "pow", "rsqrt": "rsqrt", "mean": "mean",
               "silu": "silu", "_to_copy": "cast", "convert_element_type": "cast",
               "linear": "linear", "mm": "matmul", "bmm": "matmul", "matmul": "matmul",
               "_softmax": "softmax", "softmax": "softmax"}


def canonical(node: Node, gm: Optional[fx.GraphModule] = None) -> Optional[str]:
    """Map a node to a producer-independent operation name (``"mul"``, ``"cast"``, ...) or None."""
    if not isinstance(node, Node):
        return None
    if node.op == "call_function":
        t = node.target
        if t in _FUNCTION_NAMES:
            return _FUNCTION_NAMES[t]
        if isinstance(t, torch._ops.OpOverload):            # aten.mul.Tensor -> "mul"
            return _ATEN_NAMES.get(t.overloadpacket.__name__)
        if isinstance(t, torch._ops.OpOverloadPacket):
            return _ATEN_NAMES.get(t.__name__)
        return getattr(t, "__name__", None)
    if node.op == "call_method":
        return _METHOD_NAMES.get(node.target, node.target)
    if node.op == "call_module" and gm is not None:
        mod = gm.get_submodule(node.target)
        if isinstance(mod, torch.nn.SiLU):
            return "silu"
        return type(mod).__name__
    return node.op                                           # placeholder / get_attr / output


def _strip_casts(node: Node) -> Node:
    while canonical(node) == "cast" and isinstance(node.args[0], Node):
        node = node.args[0]
    return node


def _dtype_of(node: Node):
    """Best-effort dtype of a node's value (Dynamo's example_value, ShapeProp's tensor_meta)."""
    ev = node.meta.get("example_value")
    if isinstance(ev, torch.Tensor):
        return ev.dtype
    tm = node.meta.get("tensor_meta")
    return getattr(tm, "dtype", None)


def _cast_target_dtype(node: Node):
    if canonical(node) != "cast":
        return None
    if node.op == "call_method" and node.target == "float":
        return torch.float32
    for a in list(node.args[1:]) + list(node.kwargs.values()):
        if isinstance(a, torch.dtype):
            return a
    return None


def _mean_over_last_dim(node: Node) -> bool:
    dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
    keep = node.kwargs.get("keepdim", node.args[2] if len(node.args) > 2 else False)
    if isinstance(dim, (list, tuple)):
        dim = dim[0] if len(dim) == 1 else None
    return dim == -1 and bool(keep)


def _is_square(node: Node, of: Node) -> bool:
    c = canonical(node)
    if c == "pow":
        return node.args[0] is of and node.args[1] == 2
    return c == "mul" and node.args[0] is of and node.args[1] is of


# ----------------------------------------------------------------------------- RMSNorm
def match_rmsnorm(node: Node) -> Optional[Tuple[Node, Node, float, bool]]:
    """Recognise ``(xc * rsqrt(mean(xc^2, -1, keepdim) + eps)) * w`` ending at ``node``.

    ``node`` may be the final multiply or a cast of it (RMSNorm casts back to the input dtype).
    Returns ``(x, w, eps, absorbed_cast)`` with ``x`` and ``w`` *before* any up-casts, or None.
    """
    absorbed = False
    final = node
    if canonical(final) == "cast" and isinstance(final.args[0], Node):
        final, absorbed = final.args[0], True
    if canonical(final) != "mul" or len(final.args) != 2:
        return None
    for inner, w in ((final.args[0], final.args[1]), (final.args[1], final.args[0])):
        if not (isinstance(inner, Node) and isinstance(w, Node)) or canonical(inner) != "mul":
            continue
        xc, r = inner.args
        if canonical(xc) == "rsqrt":
            xc, r = r, xc
        if not isinstance(xc, Node) or canonical(r) != "rsqrt":
            continue
        add = r.args[0]
        if canonical(add) != "add":
            continue
        mean, eps = add.args
        if isinstance(mean, (int, float)):
            mean, eps = eps, mean
        if not isinstance(eps, (int, float)) or canonical(mean) != "mean" or not _mean_over_last_dim(mean):
            continue
        sq = mean.args[0]
        if not isinstance(sq, Node) or not _is_square(sq, xc):
            continue
        x = _strip_casts(xc)
        if absorbed:                                       # only absorb a cast back to x's dtype
            want, have = _cast_target_dtype(node), _dtype_of(x)
            if want is not None and have is not None and want != have:
                return None
        return x, _strip_casts(w), float(eps), absorbed
    return None


def fuse_rmsnorm(gm: fx.GraphModule) -> int:
    """Rewrite every RMSNorm subgraph into one ``fused_rmsnorm`` node; return the number fused."""
    n = 0
    for node in list(gm.graph.nodes):
        m = match_rmsnorm(node)
        if m is None:
            continue
        x, w, eps, absorbed = m
        if not absorbed and canonical(node) == "cast":
            continue                                       # the mul itself will be visited too
        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(fused_rmsnorm, (x, w, eps))
        fused.meta = dict(node.meta)
        node.replace_all_uses_with(fused)
        n += 1
    if n:
        gm.graph.lint()
        gm.recompile()
    return n


# ----------------------------------------------------------------------------- SwiGLU gate
def match_silu_mul(node: Node, gm: fx.GraphModule) -> Optional[Tuple[Node, Node]]:
    if canonical(node) != "mul" or len(node.args) != 2:
        return None
    for s, b in ((node.args[0], node.args[1]), (node.args[1], node.args[0])):
        if isinstance(s, Node) and isinstance(b, Node) and canonical(s, gm) == "silu":
            return s.args[0], b
    return None


def fuse_silu_mul(gm: fx.GraphModule) -> int:
    n = 0
    for node in list(gm.graph.nodes):
        m = match_silu_mul(node, gm)
        if m is None:
            continue
        a, b = m
        with gm.graph.inserting_before(node):
            fused = gm.graph.call_function(fused_silu_mul, (a, b))
        fused.meta = dict(node.meta)
        node.replace_all_uses_with(fused)
        n += 1
    if n:
        gm.graph.lint()
        gm.recompile()
    return n


# ----------------------------------------------------------------------------- DCE and pipeline
def eliminate_dead_code(gm: fx.GraphModule) -> int:
    """Remove nodes whose outputs nobody uses (what the fusion passes leave behind)."""
    before = len(gm.graph.nodes)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return before - len(gm.graph.nodes)


@dataclass
class PassStat:
    name: str
    matches: int
    nodes_before: int
    nodes_after: int
    ms: float

    def __str__(self) -> str:
        return (f"{self.name:22s} matches={self.matches:3d}  nodes {self.nodes_before:4d} -> "
                f"{self.nodes_after:4d}  {self.ms:6.2f} ms")


DEFAULT_PASSES: Tuple[Callable[[fx.GraphModule], int], ...] = (fuse_rmsnorm, fuse_silu_mul,
                                                                eliminate_dead_code)


def run_passes(gm: fx.GraphModule, passes: Sequence[Callable[[fx.GraphModule], int]] = DEFAULT_PASSES,
               verbose: bool = False) -> Tuple[fx.GraphModule, List[PassStat]]:
    """Apply ``passes`` in order (each returns how many rewrites it made); record statistics."""
    stats: List[PassStat] = []
    for p in passes:
        before = len(gm.graph.nodes)
        t0 = time.perf_counter()
        matches = p(gm)
        stats.append(PassStat(p.__name__, int(matches or 0), before, len(gm.graph.nodes),
                              1e3 * (time.perf_counter() - t0)))
        if verbose:
            print(stats[-1])
    return gm, stats


def fusion_backend(gm: fx.GraphModule, example_inputs) -> Callable:
    """A ``torch.compile`` backend: run our passes, then execute the rewritten graph eagerly.

        compiled = torch.compile(block, backend=fusion_backend)
    """
    gm, _ = run_passes(gm)
    return gm.forward


def fusion_then_inductor_backend(gm: fx.GraphModule, example_inputs) -> Callable:
    """Our passes first, then hand the graph to Inductor (needs fake kernels for custom ops)."""
    from torch._inductor.compile_fx import compile_fx

    gm, _ = run_passes(gm)
    return compile_fx(gm, example_inputs)


# ----------------------------------------------------------------------------- statistics
def graph_stats(gm: fx.GraphModule, example_inputs: Sequence) -> Dict[str, object]:
    """Node counts by canonical op and bytes of every materialised intermediate tensor.

    Runs ``ShapeProp`` with the given inputs, so shapes and dtypes are exact for that input. The
    ``bytes`` figure is what an eager (unfused) execution writes to memory for the non-input,
    non-output nodes; a compiler's job is to make most of it disappear into registers.
    """
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(gm).propagate(*example_inputs)
    counts: Dict[str, int] = {}
    total = 0
    by_op: Dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output", "get_attr"):
            continue
        name = canonical(node, gm) or "?"
        counts[name] = counts.get(name, 0) + 1
        tm = node.meta.get("tensor_meta")
        if tm is not None and hasattr(tm, "shape"):
            nbytes = 1
            for s in tm.shape:
                nbytes *= int(s)
            nbytes *= torch.empty((), dtype=tm.dtype).element_size()
            total += nbytes
            by_op[name] = by_op.get(name, 0) + nbytes
    return dict(nodes=len(gm.graph.nodes), counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                bytes=total, bytes_by_op=dict(sorted(by_op.items(), key=lambda kv: -kv[1])))


def fusion_bytes(batch_tokens: int, d_model: int, dtype_bytes: int = 2, unfused_kernels: int = 7
                 ) -> Dict[str, int]:
    """Napkin math for a norm + residual + activation chain over a ``(tokens, d)`` activation.

    Unfused: each of the ``unfused_kernels`` elementwise/reduction kernels reads its inputs and
    writes its output, about 2 tensors of traffic per kernel (reductions write far less, so this
    is a mild overestimate). Fused: read the input(s) once, write the output once (3 tensors for
    norm + residual: x, residual in, out). Returns bytes for both and the ratio.
    """
    tensor = batch_tokens * d_model * dtype_bytes
    unfused = 2 * unfused_kernels * tensor
    fused = 3 * tensor
    return dict(tensor_bytes=tensor, unfused=unfused, fused=fused, ratio=unfused // max(fused, 1))
