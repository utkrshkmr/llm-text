"""Single-device performance engineering toolkit (Chapter 16).

Benchmarking
    ``time_fn``            warm-up + synchronise + repeat; returns mean/std/median/min/all.
    ``cuda_event_time``    GPU-side timing with CUDA events (excludes host overhead).
    ``bench``              ``triton.testing.do_bench`` on GPU, ``time_fn`` elsewhere.
Annotation
    ``nvtx_range``         context manager: NVTX range (Nsight) + ``record_function`` (torch.profiler).
    ``annotate_modules``   forward/backward hooks that push an NVTX range per module call.
Memory
    ``SavedTensorTracker`` what autograd keeps for backward ("residuals"), deduplicated.
    ``memory_snapshot``    ``torch.cuda.memory._record_memory_history`` wrapper -> pickle for
                           https://pytorch.org/memory_viz (no-op without CUDA).
    ``peak_memory``        peak allocated bytes of a callable (CUDA only).
    ``offload_saved_tensors``  saved-tensor hooks that park residuals in (pinned) host memory.
Recomputation
    ``checkpoint_segments``    run a list of blocks, checkpointing every ``segment`` blocks.
    ``selective_checkpoint``   PyTorch selective activation checkpointing: save matmul outputs,
                               recompute everything else.
Reporting
    ``to_markdown`` / ``to_latex`` turn a list of result dicts into a table.
"""
from __future__ import annotations

import contextlib
import statistics
import timeit
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.checkpoint import checkpoint


# =========================================================================== timing
def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


def time_fn(fn: Callable[[], object], warmup: int = 5, iters: int = 10,
            sync: Callable[[], None] = synchronize) -> Dict[str, object]:
    """Time ``fn`` honestly: ``warmup`` untimed calls, then ``iters`` synchronised calls.

    Uses ``timeit.default_timer`` (the highest-resolution monotonic clock). Every timed call is
    followed by ``sync()`` because GPU work is launched asynchronously.
    """
    for _ in range(warmup):
        fn()
    sync()
    times: List[float] = []
    for _ in range(iters):
        t0 = timeit.default_timer()
        fn()
        sync()
        times.append(timeit.default_timer() - t0)
    return dict(mean=statistics.fmean(times), std=statistics.pstdev(times),
                median=statistics.median(times), min=min(times), times=times)


def cuda_event_time(fn: Callable[[], object], warmup: int = 5, iters: int = 20) -> float:
    """Median milliseconds measured with CUDA events (device-side clock)."""
    assert torch.cuda.is_available(), "CUDA events need a GPU"
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def bench(fn: Callable[[], object], warmup_ms: int = 25, rep_ms: int = 100) -> float:
    """Milliseconds per call: ``triton.testing.do_bench`` on GPU (flushes L2 between calls)."""
    if torch.cuda.is_available():
        import triton.testing
        return float(triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms))
    return 1e3 * time_fn(fn, warmup=2, iters=5)["median"]


# =========================================================================== annotation
@contextlib.contextmanager
def nvtx_range(name: str):
    """Mark a region for both Nsight Systems (NVTX) and torch.profiler (record_function)."""
    use_nvtx = torch.cuda.is_available()
    if use_nvtx:
        torch.cuda.nvtx.range_push(name)
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        if use_nvtx:
            torch.cuda.nvtx.range_pop()


def annotate_modules(model: torch.nn.Module, types: Sequence[type] = (),
                     prefix: str = "") -> List[torch.utils.hooks.RemovableHandle]:
    """Push an NVTX range around every forward (and backward) of selected submodules.

    ``types`` restricts annotation to some module classes (default: direct children of
    ``model`` and their children). Returns the hook handles so they can be removed.
    """
    handles = []
    if not torch.cuda.is_available():
        return handles
    nvtx = torch.cuda.nvtx
    for name, mod in model.named_modules():
        if mod is model or (types and not isinstance(mod, tuple(types))):
            continue
        if not types and name.count(".") > 1:
            continue
        label = f"{prefix}{name}"
        handles.append(mod.register_forward_pre_hook(lambda m, a, l=label: nvtx.range_push(f"fwd:{l}")))
        handles.append(mod.register_forward_hook(lambda m, a, o: nvtx.range_pop()))
        handles.append(mod.register_full_backward_pre_hook(lambda m, g, l=label: nvtx.range_push(f"bwd:{l}")))
        handles.append(mod.register_full_backward_hook(lambda m, gi, go: nvtx.range_pop()))
    return handles


# =========================================================================== memory
class SavedTensorTracker:
    """Record every tensor autograd saves for backward while the context is active.

    Tensors that share storage are counted once (e.g. the same input saved by two ops), and
    ``nn.Parameter``s are skipped by default (they live in memory anyway). Use it to see what a
    layer really keeps alive between forward and backward::

        with SavedTensorTracker() as tr:
            y = block(x)
        print(tr.total_bytes / 2**20, "MiB saved for backward")
        print(tr.table())
    """

    def __init__(self, skip_params: bool = True):
        self.skip_params = skip_params
        self.entries: List[Dict[str, object]] = []
        self._seen: Dict[tuple, int] = {}
        self._ctx = None

    def _key(self, t: torch.Tensor):
        try:
            st = t.untyped_storage()
            return (st.data_ptr(), st.nbytes())
        except Exception:  # tensors without storage (e.g. some subclasses)
            return (id(t), t.numel() * t.element_size())

    def _pack(self, t: torch.Tensor):
        is_param = isinstance(t, torch.nn.Parameter) or isinstance(getattr(t, "_base", None),
                                                                     torch.nn.Parameter)
        if self.skip_params and is_param:      # includes views such as weight.T saved by Linear
            return t
        key = self._key(t)
        nbytes = key[1] if key not in self._seen else 0
        self._seen.setdefault(key, len(self.entries))
        gf = type(t.grad_fn).__name__ if t.grad_fn is not None else None
        self.entries.append(dict(shape=tuple(t.shape), dtype=str(t.dtype).replace("torch.", ""),
                                 grad_fn=gf, bytes=nbytes))
        return t

    @staticmethod
    def _unpack(t):
        return t

    def __enter__(self):
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False

    @property
    def total_bytes(self) -> int:
        return sum(e["bytes"] for e in self.entries)

    def table(self, k: int = 10) -> str:
        rows = sorted(self.entries, key=lambda e: -e["bytes"])[:k]
        return "\n".join(f"{e['bytes']/2**20:9.2f} MiB  {str(e['shape']):24s} {e['dtype']:9s} {e['grad_fn']}"
                         for e in rows)


def saved_bytes(fn: Callable[..., torch.Tensor], *args, **kwargs) -> int:
    """Bytes autograd saves while running ``fn(*args)`` (forward only; graph is discarded)."""
    with SavedTensorTracker() as tr:
        out = fn(*args, **kwargs)
    del out
    return tr.total_bytes


@contextlib.contextmanager
def offload_saved_tensors(pin_memory: bool = True, min_bytes: int = 1 << 20):
    """Move large residuals to host memory during forward and back during backward.

    This is what ``torch.autograd.graph.save_on_cpu`` does; written out here to show the
    mechanism. The copy costs PCIe/NVLink-C2C bandwidth, so it only pays off when the step is
    compute-bound enough to hide the transfers (or when the alternative is not fitting at all).
    """
    def pack(t):
        if not t.is_cuda or t.numel() * t.element_size() < min_bytes:
            return t
        host = torch.empty(t.size(), dtype=t.dtype, layout=t.layout, pin_memory=pin_memory)
        host.copy_(t, non_blocking=True)
        return (t.device, host)

    def unpack(packed):
        if isinstance(packed, tuple):
            device, host = packed
            return host.to(device, non_blocking=True)
        return packed

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


@contextlib.contextmanager
def memory_snapshot(path: str = "memory_snapshot.pickle", max_entries: int = 1_000_000):
    """Record allocator history and dump it for https://pytorch.org/memory_viz."""
    if not torch.cuda.is_available():
        yield
        return
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)


def peak_memory(fn: Callable[[], object]) -> Optional[int]:
    """Peak bytes allocated by the CUDA caching allocator while running ``fn`` (else None)."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


# =========================================================================== recomputation
def checkpoint_segments(blocks: Sequence[torch.nn.Module], x: torch.Tensor, segment: int,
                        **block_kwargs) -> torch.Tensor:
    """Run ``blocks`` in order; every group of ``segment`` consecutive blocks is checkpointed.

    Only each group's *input* is saved; the group's internals are recomputed in backward.
    ``segment=0`` disables checkpointing. Peak residual memory is roughly
    (#groups) x (one block input) + segment x (one block's residuals).
    """
    if segment <= 0:
        for b in blocks:
            x = b(x, **block_kwargs)
        return x

    def run(lo, hi):
        def f(h):
            for b in blocks[lo:hi]:
                h = b(h, **block_kwargs)
            return h
        return f

    for lo in range(0, len(blocks), segment):
        x = checkpoint(run(lo, min(lo + segment, len(blocks))), x, use_reentrant=False)
    return x


def selective_checkpoint(fn: Callable, *args, save_ops: Optional[Iterable] = None):
    """Selective activation checkpointing: keep outputs of ``save_ops`` (default: matmuls),
    recompute every other op (norms, activations, softmax, elementwise) in backward."""
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    ops = set(save_ops) if save_ops is not None else {
        torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default}

    def policy(ctx, op, *a, **k):
        return CheckpointPolicy.MUST_SAVE if op in ops else CheckpointPolicy.PREFER_RECOMPUTE

    return checkpoint(fn, *args, use_reentrant=False,
                      context_fn=lambda: create_selective_checkpoint_contexts(policy))


# =========================================================================== analytics
def residual_stream_bytes(batch: int, seq: int, d_model: int, dtype_bytes: int = 4) -> int:
    """Size of one (B, T, d) activation tensor -- the unit of most activation-memory estimates."""
    return batch * seq * d_model * dtype_bytes


def attention_score_bytes(batch: int, heads: int, seq: int, dtype_bytes: int = 4) -> int:
    """One materialised (B, H, T, T) score/probability tensor."""
    return batch * heads * seq * seq * dtype_bytes


def adamw_step_bytes(n_params: int, param_bytes: int = 4, state_bytes: int = 4) -> int:
    """Minimum HBM traffic of an unfused AdamW step: read p, g, m, v; write p, m, v."""
    return n_params * (2 * param_bytes + param_bytes + 2 * state_bytes + 2 * state_bytes)


# =========================================================================== reporting
def to_markdown(rows: List[Dict[str, object]], fmt: str = "{:.3g}") -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    f = lambda v: fmt.format(v) if isinstance(v, float) else str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(f(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def to_latex(rows: List[Dict[str, object]], fmt: str = "{:.3g}") -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    f = lambda v: fmt.format(v) if isinstance(v, float) else esc(v)
    lines = [r"\begin{tabular}{" + "l" * len(cols) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    lines += [" & ".join(f(r.get(c, "")) for c in cols) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)
