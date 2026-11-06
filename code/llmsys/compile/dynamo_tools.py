"""Looking inside torch.compile: graphs, graph breaks, recompiles, guards, generated code (Chapter 23).

    capture_graphs(fn, *args)         the FX graph(s) Dynamo produced for one call, via a recording backend
    explain(fn, *args)                graph count, graph-break count and reasons (torch._dynamo.explain)
    count_recompiles(fn, inputs)      how many times the backend was invoked over a list of inputs
    list_guards(fn)                   the guards of every cache entry attached to ``fn``'s code object
    inductor_output_code(fn, *args)   the Python wrapper(s) Inductor generated, with the kernels inline
    kernel_names(code) / summarize_kernels(code)   which kernels exist and what each one fused
    partition_report(fn, *args)       what AOTAutograd saves between forward and backward, per partitioner
    triton_artifacts(cache_dir)       the IR stages Triton left in its cache for each kernel

Nothing here needs a GPU: on the CPU Inductor emits C++/OpenMP kernels instead of Triton ones,
and the same helpers read them.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch._dynamo as dynamo
import torch.fx as fx


# ----------------------------------------------------------------------------- graph capture
@dataclass
class Capture:
    graphs: List[fx.GraphModule] = field(default_factory=list)
    example_inputs: List[list] = field(default_factory=list)
    outputs: object = None

    @property
    def compiles(self) -> int:
        return len(self.graphs)


def recording_backend(store: Capture) -> Callable:
    """A backend that keeps every graph Dynamo hands it and runs the graph eagerly."""
    def backend(gm: fx.GraphModule, example_inputs):
        store.graphs.append(gm)
        store.example_inputs.append(list(example_inputs))
        return gm.forward
    return backend


def capture_graphs(fn: Callable, *args, dynamic: Optional[bool] = None, **kwargs) -> Capture:
    """Run ``fn(*args)`` under Dynamo and return the FX graph(s) it produced.

    One graph per graph-break segment; ``example_inputs`` are the real tensors (inputs and lifted
    parameters, in placeholder order), so ``cap.graphs[0](*cap.example_inputs[0])`` runs the graph.
    """
    dynamo.reset()
    cap = Capture()
    compiled = torch.compile(fn, backend=recording_backend(cap), dynamic=dynamic)
    cap.outputs = compiled(*args, **kwargs)
    return cap


# ----------------------------------------------------------------------------- graph breaks
@dataclass
class ExplainReport:
    graph_count: int
    graph_break_count: int
    op_count: int
    break_reasons: List[str]

    def __str__(self) -> str:
        s = f"{self.graph_count} graph(s), {self.graph_break_count} break(s), {self.op_count} ops"
        for r in self.break_reasons:
            s += f"\n  - {r}"
        return s


def explain(fn: Callable, *args, **kwargs) -> ExplainReport:
    """Where does Dynamo stop tracing, and why? (``torch._dynamo.explain`` with a tidy summary.)"""
    dynamo.reset()
    ex = dynamo.explain(fn)(*args, **kwargs)
    reasons = [r.reason.splitlines()[0].strip() for r in ex.break_reasons]
    return ExplainReport(ex.graph_count, ex.graph_break_count, ex.op_count, reasons)


# ----------------------------------------------------------------------------- recompiles
def count_recompiles(fn: Callable, inputs: Sequence[tuple], dynamic: Optional[bool] = None,
                     **compile_kwargs) -> int:
    """Compile ``fn`` and call it on every tuple in ``inputs``; return how many graphs were compiled.

    Every time a guard fails Dynamo compiles the frame again and the backend is invoked again, so
    counting backend invocations counts compiles (the first call included).
    """
    dynamo.reset()
    cap = Capture()
    compiled = torch.compile(fn, backend=recording_backend(cap), dynamic=dynamic, **compile_kwargs)
    for a in inputs:
        compiled(*a)
    return cap.compiles


def counters() -> Dict[str, Dict[str, int]]:
    """A copy of Dynamo's process-wide counters (frames, graph breaks, unique graphs, resumes)."""
    return {k: dict(v) for k, v in dynamo.utils.counters.items()}


# ----------------------------------------------------------------------------- guards
_GUARD_LINE = re.compile(r"\+- ([A-Z_]+): (.*)")


def list_guards(fn: Callable, full: bool = False) -> List[str]:
    """Guards of every compiled entry cached on ``fn``'s code object (``fn`` or ``fn.forward``).

    Returns ``"KIND: expression"`` strings; without ``full`` the bookkeeping guards that every
    frame carries (global state, torch-function mode stack, dynamic-marking attributes) are
    dropped so that the shape/dtype/constant guards stand out.
    """
    code = fn.forward.__code__ if isinstance(fn, torch.nn.Module) else fn.__code__
    entries = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(code)
    out: List[str] = []
    boring = ("GLOBAL_STATE", "TORCH_FUNCTION_MODE_STACK", "DEFAULT_DEVICE", "DIMENSION_DYNAMIC_MARKING_GUARD",
              "LAMBDA_GUARD")
    for e in entries:
        for line in str(e.guard_manager).splitlines():
            m = _GUARD_LINE.search(line)
            if not m:
                continue
            kind, expr = m.group(1), m.group(2).split("  #")[0].strip()
            if full or kind not in boring:
                out.append(f"{kind}: {expr}")
    return out


# ----------------------------------------------------------------------------- generated code
@contextlib.contextmanager
def _record_inductor_modules(store: List[str]):
    """Intercept every generated module Inductor loads (wrapper code with kernels inline)."""
    from torch._inductor import codecache

    orig = codecache.PyCodeCache.load_by_key_path

    def hook(key, path, *a, **k):
        try:
            with open(path) as fh:
                store.append(fh.read())
        except OSError:
            pass
        return orig(key, path, *a, **k)

    codecache.PyCodeCache.load_by_key_path = hook
    try:
        yield
    finally:
        codecache.PyCodeCache.load_by_key_path = orig


def inductor_output_code(fn: Callable, *args, dynamic: Optional[bool] = False, **compile_kwargs
                         ) -> Tuple[object, List[str]]:
    """Compile ``fn`` with Inductor, run it once, and return ``(output, [generated modules])``.

    The modules are the ``output_code`` artifacts: a Python wrapper that allocates buffers and
    launches kernels, with the Triton (GPU) or C++ (CPU) kernel sources inline. Works on cache hits
    as well as fresh compiles, because the cached module is still loaded from its file.
    """
    dynamo.reset()
    codes: List[str] = []
    with _record_inductor_modules(codes):
        compiled = torch.compile(fn, dynamic=dynamic, **compile_kwargs)
        out = compiled(*args)
    return out, codes


_KERNEL_DEF = re.compile(r"^(\w+) = async_compile\.(triton|cpp_pybinding|cpp)\(", re.M)
_KIND = {"poi": "pointwise", "red": "reduction", "per": "persistent reduction", "tem": "template",
         "for": "foreach"}
_COMPOUND = ("_to_copy", "native_layer_norm", "scaled_dot_product", "log_softmax", "_softmax",
             "_unsafe_view", "silu_backward", "new_zeros", "masked_fill", "nan_to_num", "index_select",
             "index_put", "repeat_interleave", "embedding_dense_backward", "sum_backward",
             "threshold_backward", "gelu_backward", "sigmoid_backward", "tanh_backward", "mul_backward",
             "prepare_softmax_online")


def kernel_names(code: str) -> List[str]:
    """Names of the kernels defined in one generated module, in definition order."""
    return [m.group(1) for m in _KERNEL_DEF.finditer(code)]


def fused_ops(kernel_name: str) -> List[str]:
    """``triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0`` -> ['to_copy', 'add', 'mean', ...]."""
    m = re.match(r"^(?:triton_\w{3}|cpp)_fused_(.*?)_\d+$", kernel_name)
    if not m:
        return []
    body = m.group(1)
    for c in _COMPOUND:                                  # protect op names that contain underscores
        body = body.replace(c, c.strip("_").replace("_", "-"))
    return [t.replace("-", "_") for t in body.split("_") if t]


def kernel_kind(kernel_name: str) -> str:
    m = re.match(r"^triton_(\w{3})_", kernel_name)
    if m:
        return _KIND.get(m.group(1), m.group(1))
    return "cpp" if kernel_name.startswith("cpp_") else "?"


def summarize_kernels(code: str) -> List[Dict[str, object]]:
    """One row per kernel: name, kind (pointwise/reduction/...), the ATen ops fused into it."""
    return [dict(name=n, kind=kernel_kind(n), ops=fused_ops(n)) for n in kernel_names(code)]


def extern_calls(code: str) -> List[str]:
    """Library kernels the wrapper calls instead of generating (cuBLAS/MKL GEMMs, SDPA, ...)."""
    return sorted(set(re.findall(r"extern_kernels\.(\w+)\(", code)) |
                  set(re.findall(r"(aten\.[\w.]+)\(", code)))


# ----------------------------------------------------------------------------- AOTAutograd
@dataclass
class PartitionReport:
    partitioner: str
    forward_ops: int
    backward_ops: int
    saved_tensors: int
    saved_bytes: int
    forward: fx.GraphModule
    backward: fx.GraphModule

    def __str__(self) -> str:
        return (f"{self.partitioner:8s} forward {self.forward_ops:3d} ops, backward "
                f"{self.backward_ops:3d} ops, saves {self.saved_tensors} tensors, "
                f"{self.saved_bytes} bytes")


def _value_bytes(node: fx.Node) -> int:
    v = node.meta.get("val")
    return v.numel() * v.element_size() if isinstance(v, torch.Tensor) else 0


def partition_report(fn: Callable, *args, partitioner: str = "min_cut",
                     params: Sequence[torch.Tensor] = ()) -> PartitionReport:
    """Trace ``fn`` and its backward jointly and report how the graph was cut.

    ``partitioner`` is ``"default"`` (everything runs in forward, save whatever backward reads) or
    ``"min_cut"`` (what torch.compile uses: recompute cheap ops in backward when that saves bytes).
    ``fn`` must return one tensor. The saved tensors are the forward graph's outputs beyond that
    result, and the forward's inputs that backward reads do not count: they are alive anyway.
    """
    from torch._functorch.aot_autograd import aot_function
    from torch._functorch.partitioners import default_partition, min_cut_rematerialization_partition

    choice = dict(default=default_partition, min_cut=min_cut_rematerialization_partition)[partitioner]
    graphs: Dict[str, fx.GraphModule] = {}

    def keep(name: str) -> Callable:
        def compiler(gm: fx.GraphModule, _):
            graphs[name] = gm
            return gm
        return compiler

    traced = aot_function(fn, fw_compiler=keep("fw"), bw_compiler=keep("bw"), partition_fn=choice)
    traced(*args).sum().backward()
    fw, bw = graphs["fw"], graphs["bw"]
    returned = next(n for n in fw.graph.nodes if n.op == "output").args[0]
    saved = [n for n in returned[1:] if isinstance(n, fx.Node) and n.op != "placeholder"]
    calls = lambda g: sum(1 for n in g.graph.nodes if n.op == "call_function")
    return PartitionReport(partitioner, calls(fw), calls(bw), len(saved),
                           sum(_value_bytes(n) for n in saved), fw, bw)


# ----------------------------------------------------------------------------- Triton's cache
def triton_artifacts(cache_dir: Optional[str] = None) -> List[Dict[str, object]]:
    """One row per compiled kernel in Triton's on-disk cache: stage sizes and machine-code metadata.

    Each kernel variant is a directory named by a hash of (source, compiler, target, constexprs),
    holding ``<kernel>.ttir .ttgir .llir .ptx .cubin .json``. ``cache_dir`` defaults to
    ``$TRITON_CACHE_DIR``, then ``~/.triton/cache``.
    """
    root = cache_dir or os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
    rows: List[Dict[str, object]] = []
    if not os.path.isdir(root):
        return rows
    for key in sorted(os.listdir(root)):
        folder = os.path.join(root, key)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(name)
            if ext != ".cubin":
                continue
            sizes = {e: os.path.getsize(os.path.join(folder, stem + "." + e))
                     for e in ("ttir", "ttgir", "llir", "ptx", "cubin")
                     if os.path.exists(os.path.join(folder, stem + "." + e))}
            meta: Dict[str, object] = {}
            path = os.path.join(folder, stem + ".json")
            if os.path.exists(path):
                with open(path) as fh:
                    meta = {k: v for k, v in json.load(fh).items() if k != "child_paths"}
            rows.append(dict(kernel=stem, key=key, sizes=sizes, meta=meta))
    return rows
