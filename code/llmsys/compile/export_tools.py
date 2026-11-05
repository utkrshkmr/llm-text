"""Ahead-of-time graphs: torch.export, ONNX + ONNX Runtime, CUDA graphs for decoding (Chapter 23).

    export_program(model, ids, max_seq)  torch.export with a dynamic sequence dimension
    export_onnx(model, ids, path)        ONNX file (torch.export-based exporter; TorchScript fallback)
    ort_session / ort_run / compare_ort_torch     run it in ONNX Runtime and compare with PyTorch
    graph_decode_step(model, ids, pos, cache)     a Qwen3 decode step with no Python state: every
                                                   position-dependent value is a device tensor
    CUDAGraphDecoder                     captures that step in a torch.cuda.CUDAGraph and replays it

Only ``graph_decode_step`` and ``CUDAGraphDecoder`` know about our models; the export helpers take
any module whose forward has a single ``(B, T)`` integer input.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.export import Dim, ExportedProgram

from llmsys.kvcache import StaticKVCache
from llmsys.layers import _attend, apply_rope

try:  # pragma: no cover - depends on the environment
    import onnx
    HAS_ONNX = True
except Exception:
    onnx = None
    HAS_ONNX = False
try:  # pragma: no cover
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    ort = None
    HAS_ORT = False
try:  # pragma: no cover
    import onnxscript  # noqa: F401   (the torch.export-based ONNX exporter needs it)
    HAS_ONNXSCRIPT = True
except Exception:
    HAS_ONNXSCRIPT = False


# ----------------------------------------------------------------------------- torch.export
def export_program(model: nn.Module, example_ids: torch.Tensor, max_seq: Optional[int] = None,
                   dynamic_seq: bool = True, arg_name: str = "in_idx") -> ExportedProgram:
    """Export ``model(ids)`` ahead of time; the sequence dimension is symbolic when ``dynamic_seq``.

    ``max_seq`` bounds the symbol (default: the model's ``context_length``); a bound lets the
    exporter discharge ``start + T > context_length`` statically instead of guarding on it.
    """
    shapes = None
    if dynamic_seq:
        max_seq = max_seq or model.cfg["context_length"]
        shapes = {arg_name: {1: Dim("seq", min=1, max=max_seq)}}
    return torch.export.export(model.eval(), (example_ids,), dynamic_shapes=shapes)


def program_summary(ep: ExportedProgram) -> Dict[str, object]:
    """Node count, op histogram and the symbolic-shape constraints of an ExportedProgram."""
    ops = Counter()
    for n in ep.graph.nodes:
        if n.op == "call_function":
            t = n.target
            ops[getattr(t, "__name__", str(t))] += 1
    sig = ep.graph_signature
    return dict(nodes=len(ep.graph.nodes), params=len(sig.parameters), buffers=len(sig.buffers),
                user_inputs=len(sig.user_inputs), ops=dict(ops.most_common()),
                range_constraints={str(k): str(v) for k, v in ep.range_constraints.items()})


def run_exported(ep: ExportedProgram, ids: torch.Tensor) -> torch.Tensor:
    return ep.module()(ids)


# ----------------------------------------------------------------------------- ONNX
def export_onnx(model: nn.Module, example_ids: torch.Tensor, path: str, dynamic_seq: bool = True,
                opset: int = 18, dynamo: Optional[bool] = None, arg_name: str = "in_idx") -> str:
    """Write ``model`` to an ONNX file with a symbolic sequence dimension; return ``path``.

    ``dynamo=True`` uses the torch.export-based exporter (needs ``onnxscript``); ``False`` the older
    TorchScript tracer. Default: whichever is available, preferring the new one.
    """
    if dynamo is None:
        dynamo = HAS_ONNXSCRIPT
    model = model.eval()
    if dynamo:
        shapes = {arg_name: {1: Dim("seq", min=1, max=model.cfg["context_length"])}} if dynamic_seq else None
        prog = torch.onnx.export(model, (example_ids,), dynamo=True, dynamic_shapes=shapes,
                                 opset_version=opset, optimize=True)
        prog.save(path)
    else:
        import warnings
        axes = {arg_name: {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}} if dynamic_seq else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(model, (example_ids,), path, dynamo=False, input_names=[arg_name],
                              output_names=["logits"], dynamic_axes=axes, opset_version=min(opset, 17))
    return path


def onnx_summary(path: str) -> Dict[str, object]:
    if not HAS_ONNX:
        raise RuntimeError("onnx is not installed")
    m = onnx.load(path)
    dims = lambda v: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]
    return dict(nodes=len(m.graph.node), ops=dict(Counter(n.op_type for n in m.graph.node).most_common()),
                inputs={i.name: dims(i) for i in m.graph.input}, outputs={o.name: dims(o) for o in m.graph.output},
                opset=[(o.domain or "ai.onnx", o.version) for o in m.opset_import],
                initializers=len(m.graph.initializer))


def ort_session(path: str, providers: Sequence[str] = ("CPUExecutionProvider",),
                optimized_path: Optional[str] = None, threads: Optional[int] = None):
    """An ONNX Runtime session with all graph optimisations; optionally dump the optimised graph."""
    if not HAS_ORT:
        raise RuntimeError("onnxruntime is not installed")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if optimized_path:
        so.optimized_model_filepath = optimized_path
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=list(providers))


def ort_run(session, ids: torch.Tensor):
    name = session.get_inputs()[0].name
    return session.run(None, {name: ids.cpu().numpy()})[0]


@torch.no_grad()
def compare_ort_torch(model: nn.Module, session, ids: torch.Tensor) -> Dict[str, float]:
    ref = model.eval()(ids).float()
    out = torch.from_numpy(ort_run(session, ids)).float()
    diff = (out - ref).abs()
    return dict(max_abs=diff.max().item(), mean_abs=diff.mean().item(),
                argmax_agree=(out.argmax(-1) == ref.argmax(-1)).float().mean().item())


# ----------------------------------------------------------------------------- CUDA graphs
@torch.no_grad()
def graph_decode_step(model: nn.Module, ids: torch.Tensor, pos: torch.Tensor,
                      cache: StaticKVCache) -> torch.Tensor:
    """One Qwen3 decode step written so that a CUDA graph can replay it at any position.

    ``ids`` is ``(B, 1)`` and ``pos`` a 0-dim int64 tensor *on the device*: the RoPE row, the cache
    slot and the attention mask are all derived from ``pos`` with tensor ops (``index_select``,
    ``index_copy_``, a comparison against ``arange``), never from a Python integer. The eager
    model's ``cache.update`` slices with Python ints, which a captured graph would freeze.
    Attention runs over all ``max_len`` slots with the slots beyond ``pos`` masked out, so every
    tensor shape is independent of the position. Numerically identical to ``model(ids, cache=...)``.
    """
    B = ids.shape[0]
    p1 = pos.reshape(1)
    cos, sin = model.cos.index_select(0, p1), model.sin.index_select(0, p1)
    forbidden = (torch.arange(cache.max_len, device=ids.device) > pos)[None, None, None, :]
    x = model.tok_emb(ids)
    for i, blk in enumerate(model.trf_blocks):
        att = blk.att
        h = blk.norm1(x)
        q = att.W_query(h).view(B, 1, att.num_heads, att.head_dim).transpose(1, 2)
        k = att.W_key(h).view(B, 1, att.num_kv_groups, att.head_dim).transpose(1, 2)
        v = att.W_value(h).view(B, 1, att.num_kv_groups, att.head_dim).transpose(1, 2)
        if att.q_norm is not None:
            q, k = att.q_norm(q), att.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        cache.k[i].index_copy_(2, p1, k)                  # write slot `pos`, address fixed
        cache.v[i].index_copy_(2, p1, v)
        K = cache.k[i].repeat_interleave(att.group_size, dim=1)
        V = cache.v[i].repeat_interleave(att.group_size, dim=1)
        ctx = _attend(q, K, V, forbidden, 0.0, False, att.use_sdpa)
        x = x + att.out_proj(ctx.transpose(1, 2).reshape(B, 1, att.d_out))
        x = x + blk.ff(blk.norm2(x))
    return model.out_head(model.final_norm(x))


class CUDAGraphDecoder:
    """Capture :func:`graph_decode_step` once and replay it for every generated token.

        cache = StaticKVCache(L, B, G, max_len, D, dtype, "cuda")
        logits = model(prompt, cache=cache)                 # prefill eagerly (fills slots 0..P-1)
        dec = CUDAGraphDecoder(model, cache, start_pos=prompt.shape[1])
        for _ in range(n):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            logits = dec.step(nxt)                          # one graph launch per token

    The static input buffer, the position counter and the cache buffers are the graph's only
    state; ``step`` copies the token in, replays, and advances the position on the device.
    """

    def __init__(self, model: nn.Module, cache: StaticKVCache, start_pos: int, batch_size: Optional[int] = None,
                 warmup: int = 3):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA graphs need a CUDA device")
        self.model, self.cache = model, cache
        dev = cache.k[0].device
        B = batch_size or cache.k[0].shape[0]
        self.ids = torch.zeros(B, 1, dtype=torch.long, device=dev)
        self.pos = torch.tensor(start_pos, dtype=torch.long, device=dev)
        # Warm up on a side stream (lazy init, cuBLAS handles, allocator) with a scratch cache so
        # the real cache is untouched, then capture against the real cache at the real position.
        scratch = StaticKVCache(len(cache.k), B, cache.k[0].shape[1], cache.max_len, cache.k[0].shape[3],
                                dtype=cache.k[0].dtype, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                graph_decode_step(model, self.ids, self.pos, scratch)
        torch.cuda.current_stream().wait_stream(s)
        del scratch
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = graph_decode_step(model, self.ids, self.pos, cache)

    def step(self, next_ids: torch.Tensor) -> torch.Tensor:
        self.ids.copy_(next_ids)
        self.graph.replay()
        self.pos += 1                                       # device-side; the Python side never syncs
        for i in range(len(self.cache.lengths)):            # keep the eager bookkeeping consistent
            self.cache.lengths[i] += 1
        return self.logits
