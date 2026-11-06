"""Fully sharded data parallelism (ZeRO-3) from scratch (Chapter 19).

Design (the "flat-parameter" scheme used by PyTorch FSDP1 and DeepSpeed ZeRO-3):

* The model is cut into **units**: typically one per transformer block, plus a *root*
  unit that owns every parameter not inside a block (embeddings, final norm, head).
* Each unit's parameters are flattened into one fp32 vector, padded to a multiple of the
  world size, and split into equal **shards**. Rank r keeps only shard r, as an
  ``nn.Parameter`` that the (unmodified) optimizer updates. The unit's original
  parameters are deleted from the module.
* **Forward**: a pre-forward hook all-gathers the shards (optionally cast to a low-precision
  ``compute_dtype`` *before* communication) into a flat buffer, then re-attaches each
  original parameter name as a *view* into that buffer. A post-forward hook frees the
  buffer's storage (``untyped_storage().resize_(0)``) so only one or two units are ever
  materialised.
* **Backward**: a hook on the unit's output fires when backward reaches the unit; it
  re-gathers the same storage so the views autograd saved are valid again. When the flat
  buffer's gradient is complete (``register_post_accumulate_grad_hook``) it is
  **reduce-scattered**: every rank receives the averaged gradient of its own shard, the
  full gradient is freed, and the storage is released again.
* **Prefetch** (optional): during forward, unit i+1's all-gather is issued asynchronously
  while unit i computes; during backward, the unit before is prefetched.

Everything works with the gloo backend on CPU (for development and tests) and unchanged
with NCCL on GPUs. The code favours clarity over the last bit of speed: production FSDP
avoids the extra copy in :meth:`_FlatUnit._fill_storage` and uses dedicated CUDA streams.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


# =========================================================================== flat helpers
def padded_numel(numel: int, world: int) -> int:
    """Smallest multiple of ``world`` that is >= ``numel``."""
    return int(math.ceil(numel / world) * world)


def flatten_tensors(tensors: Sequence[torch.Tensor], world: int,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Concatenate tensors into one 1-D vector padded with zeros to a multiple of ``world``."""
    flat = torch.cat([t.detach().reshape(-1).to(dtype) for t in tensors]) if tensors \
        else torch.zeros(0, dtype=dtype)
    pad = padded_numel(flat.numel(), world) - flat.numel()
    return torch.cat([flat, flat.new_zeros(pad)]) if pad else flat


def shard_range(total: int, rank: int, world: int) -> Tuple[int, int]:
    """[start, end) of rank's equal-size shard of a padded flat vector of length ``total``."""
    assert total % world == 0, "pad the flat vector first"
    s = total // world
    return rank * s, (rank + 1) * s


def unflatten_views(flat: torch.Tensor, shapes: Sequence[torch.Size]) -> List[torch.Tensor]:
    """Split a flat vector back into views with the given shapes (padding is ignored)."""
    out, off = [], 0
    for shp in shapes:
        n = int(torch.Size(shp).numel())
        out.append(flat[off: off + n].view(shp))
        off += n
    return out


# =========================================================================== one unit
class _FlatUnit:
    """Bookkeeping for one FSDP unit: which module attributes it owns and its shard."""

    def __init__(self, name: str, module: nn.Module,
                 slots: List[Tuple[nn.Module, str, int]], params: List[nn.Parameter],
                 rank: int, world: int, compute_dtype: Optional[torch.dtype]):
        self.name, self.module = name, module
        self.slots = slots              # (owning module, attribute name, index into params)
        self.shapes = [p.shape for p in params]
        self.numel = sum(p.numel() for p in params)
        self.rank, self.world = rank, world
        self.compute_dtype = compute_dtype or params[0].dtype
        flat = flatten_tensors(params, world, torch.float32)
        self.padded = flat.numel()
        a, b = shard_range(self.padded, rank, world)
        self.shard = nn.Parameter(flat[a:b].clone())          # fp32 master shard
        # gathered full flat buffer in compute dtype; its storage is freed when unused
        self.flat = torch.empty(self.padded, dtype=self.compute_dtype,
                                device=self.shard.device)
        self.flat.untyped_storage().resize_(0)
        self.full: Optional[torch.Tensor] = None              # autograd leaf over self.flat
        self.pending: Optional[Tuple[torch.Tensor, object]] = None   # prefetch (buf, work)
        self.grad_work: Optional[Tuple[object, torch.Tensor]] = None
        self.backward_ready = False
        self.stats = {"all_gathers": 0, "reduce_scatters": 0, "peak_gathered": 0}

    # ------------------------------------------------------------------ storage helpers
    @property
    def gathered(self) -> bool:
        return self.flat.untyped_storage().size() > 0

    def _launch_gather(self, async_op: bool):
        src = self.shard.detach().to(self.compute_dtype)      # cast BEFORE communicating
        buf = torch.empty(self.padded, dtype=self.compute_dtype, device=src.device)
        work = dist.all_gather_into_tensor(buf, src, async_op=async_op)
        self.stats["all_gathers"] += 1
        return buf, work

    def _fill_storage(self, buf: torch.Tensor):
        # Copy at the *storage* level: this does not bump autograd version counters, so
        # tensors saved for backward (views of self.flat) stay valid after a re-gather.
        self.flat.untyped_storage().resize_(buf.untyped_storage().size())
        self.flat.untyped_storage().copy_(buf.untyped_storage())

    def prefetch(self):
        if not self.gathered and self.pending is None:
            self.pending = self._launch_gather(async_op=True)

    def unshard(self):
        if self.gathered:
            return
        if self.pending is not None:
            buf, work = self.pending
            work.wait()
            self.pending = None
        else:
            buf, _ = self._launch_gather(async_op=False)
        self._fill_storage(buf)

    def reshard(self):
        self.flat.untyped_storage().resize_(0)

    def attach_views(self, requires_grad: bool):
        """Expose the gathered buffer as the module's parameters (plain tensors)."""
        full = self.flat.detach()
        if requires_grad:
            full.requires_grad_(True)
        self.full = full
        views = unflatten_views(full, self.shapes)
        for mod, attr, idx in self.slots:
            setattr(mod, attr, views[idx])
        return full


# =========================================================================== the wrapper
class FullyShardedDP(nn.Module):
    """Wrap ``module`` so that every unit's parameters are sharded across ranks.

    Args:
        module: the full model (identical initialisation is NOT required; rank 0's
            weights are broadcast before sharding).
        units: submodules that become their own FSDP unit (e.g. ``model.trf_blocks``).
            Everything else forms the root unit.
        compute_dtype: dtype for communication and compute (e.g. ``torch.bfloat16``);
            master shards, gradients after reduce-scatter and optimizer state stay fp32.
        reshard_after_forward: free gathered parameters after each unit's forward
            (ZeRO-3). ``False`` keeps them until backward (more memory, fewer gathers).
        prefetch: overlap the next unit's all-gather with the current unit's compute.

    Use exactly like a module, then call :meth:`finish_gradient_synchronization` after
    ``loss.backward()`` and before ``optimizer.step()``; build the optimizer over
    ``fsdp.parameters()`` (the shards).
    """

    def __init__(self, module: nn.Module, units: Iterable[nn.Module] = (),
                 compute_dtype: Optional[torch.dtype] = None,
                 reshard_after_forward: bool = True, prefetch: bool = True):
        super().__init__()
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.reshard_after_forward = reshard_after_forward
        self.prefetch_enabled = prefetch
        self.compute_dtype = compute_dtype
        # broadcast rank 0's initial weights so all ranks shard the same model
        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, src=0)
        unit_modules = list(units)
        self._units: List[_FlatUnit] = []
        claimed: Dict[int, nn.Parameter] = {}
        for i, um in enumerate(unit_modules):
            u = self._make_unit(f"unit{i}", um, claimed)
            if u is not None:
                self._units.append(u)
        root = self._make_unit("root", module, claimed, root=True)
        if root is not None:
            self._units.append(root)
        else:                         # no root parameters: still end the recording pass
            module.register_forward_hook(lambda *a: setattr(self, "_recording", False))
        self.module = module
        # shards are the only parameters the optimizer sees
        self.shards = nn.ParameterList([u.shard for u in self._units])
        self._fwd_order: List[_FlatUnit] = []
        self._recording = True
        for u in self._units:
            self._install_hooks(u)

    # ------------------------------------------------------------------ construction
    def _make_unit(self, name, um: nn.Module, claimed: Dict[int, nn.Parameter],
                   root: bool = False) -> _FlatUnit:
        slots, params, index = [], [], {}
        for mod in um.modules():
            for attr, p in list(mod._parameters.items()):
                if p is None or id(p) in claimed and id(p) not in index:
                    continue                                  # owned by an inner unit
                if id(p) not in index:                        # tied params appear once
                    index[id(p)] = len(params)
                    params.append(p)
                slots.append((mod, attr, index[id(p)]))
        if not params:
            return None
        for p in params:
            claimed[id(p)] = p
        for mod, attr, _ in slots:                            # params become plain attrs
            del mod._parameters[attr]
        unit = _FlatUnit(name, um, slots, params, self.rank, self.world, self.compute_dtype)
        return unit

    def set_compute_dtype(self, dtype: torch.dtype):
        for u in self._units:
            u.compute_dtype = dtype
            u.flat = torch.empty(u.padded, dtype=dtype, device=u.shard.device)
            u.flat.untyped_storage().resize_(0)

    # ------------------------------------------------------------------ hooks
    def _install_hooks(self, u: _FlatUnit):
        is_root = u.name == "root"

        def pre_forward(mod, args, kwargs):
            if self._recording and not is_root:
                self._fwd_order.append(u)
            u.unshard()
            u.stats["peak_gathered"] = max(u.stats["peak_gathered"],
                                           self.gathered_numel())
            full = u.attach_views(requires_grad=torch.is_grad_enabled())
            if full.requires_grad:
                u.backward_ready = False
                full.register_post_accumulate_grad_hook(lambda t, u=u: self._on_grad(u))
            if self.prefetch_enabled and not is_root:
                nxt = self._next_in_order(u, +1)
                if nxt is not None:
                    nxt.prefetch()
            return None

        def post_forward(mod, args, kwargs, out):
            grad_on = torch.is_grad_enabled()
            if not is_root and (self.reshard_after_forward or not grad_on):
                u.reshard()
            if grad_on:
                for t in _tensors(out):
                    if t.requires_grad:
                        t.register_hook(lambda g, u=u: self._pre_backward(u))
            if is_root:
                self._recording = False
                if not grad_on and self.reshard_after_forward:
                    u.reshard()
            return None

        u.module.register_forward_pre_hook(pre_forward, with_kwargs=True)
        u.module.register_forward_hook(post_forward, with_kwargs=True)

    def _next_in_order(self, u: _FlatUnit, step: int) -> Optional[_FlatUnit]:
        if u not in self._fwd_order:
            return None
        i = self._fwd_order.index(u) + step
        return self._fwd_order[i] if 0 <= i < len(self._fwd_order) else None

    def _pre_backward(self, u: _FlatUnit):
        if not u.backward_ready:
            u.backward_ready = True
            u.unshard()                     # refill the storage saved views point into
            if self.prefetch_enabled:
                prev = self._next_in_order(u, -1)
                if prev is not None:
                    prev.prefetch()

    def _on_grad(self, u: _FlatUnit):
        g = u.full.grad.detach().to(torch.float32)
        shard_grad = torch.empty_like(u.shard)
        work = dist.reduce_scatter_tensor(shard_grad, g, op=dist.ReduceOp.SUM,
                                          async_op=True)
        u.stats["reduce_scatters"] += 1
        u.grad_work = (work, shard_grad, g)
        u.full.grad = None
        u.full = None
        if u.name != "root" or self.reshard_after_forward:
            u.reshard()

    # ------------------------------------------------------------------ public API
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all reduce-scatters and install averaged gradients on the shards."""
        for u in self._units:
            if u.grad_work is None:
                continue
            work, shard_grad, _ = u.grad_work
            work.wait()
            shard_grad /= self.world
            if u.shard.grad is None:
                u.shard.grad = shard_grad
            else:                           # gradient accumulation across micro-batches
                u.shard.grad += shard_grad
            u.grad_work = None
        for u in self._units:              # root may still be gathered after backward
            if u.gathered and u.pending is None:
                u.reshard()

    def gathered_numel(self) -> int:
        return sum(u.padded for u in self._units if u.gathered)

    def sharded_numel(self) -> int:
        return sum(u.shard.numel() for u in self._units)

    def unit_names(self) -> List[str]:
        return [u.name for u in self._units]

    def comm_stats(self) -> Dict[str, Dict[str, int]]:
        return {u.name: dict(u.stats) for u in self._units}

    @torch.no_grad()
    def full_state(self) -> Dict[str, torch.Tensor]:
        """All-gather every unit and return {qualified attr name: fp32 tensor} on every rank.

        Used for checkpointing and for testing equivalence with single-process training.
        """
        names = {id(m): n for n, m in self.module.named_modules()}
        out = {}
        for u in self._units:
            buf = torch.empty(u.padded, dtype=torch.float32, device=u.shard.device)
            dist.all_gather_into_tensor(buf, u.shard.detach())
            views = unflatten_views(buf, u.shapes)
            for mod, attr, idx in u.slots:
                key = (names[id(mod)] + "." if names[id(mod)] else "") + attr
                out[key] = views[idx].clone()
        return out


def _tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _tensors(y)


# =========================================================================== accounting
def fsdp_persistent_bytes(n_params: float, world: int, master_bytes: int = 4,
                          optim_bytes: int = 8, grad_bytes: int = 4) -> float:
    """Per-rank bytes that live for the whole step: master shard + grad shard + Adam m, v."""
    return n_params * (master_bytes + grad_bytes + optim_bytes) / world


def fsdp_transient_bytes(unit_params: Sequence[float], compute_bytes: int = 2,
                         units_in_flight: int = 2) -> float:
    """Peak gathered-parameter memory: the largest ``units_in_flight`` units in compute dtype
    (current unit + one prefetched), plus their full gradients during backward."""
    biggest = sorted(unit_params, reverse=True)[:units_in_flight]
    return sum(biggest) * compute_bytes * 2       # params + full grad of the same unit


def layer_parallel_costs(n_params: float, tokens: float, d_model: int, n: int,
                         flops: float, bw: float, strategy: str = "dp",
                         bytes_per: int = 2, tp_allreduces: int = 1) -> Dict[str, float]:
    """Compute vs. communication time for ONE dense layer (or block) under a strategy.

    The layer has ``n_params`` weights with input/output width ``d_model`` and processes ``tokens`` tokens per step across ``n`` devices. ``flops`` is
    per-device FLOP/s and ``bw`` per-device egress bandwidth in bytes/s. Ring collectives
    move (n-1)/n of the data per rank (all-reduce: twice that).

    dp   : batch split; backward all-reduces the weight gradients.
    fsdp : batch split; weights all-gathered in forward and again in backward, gradients
           reduce-scattered.
    tp   : weights split Megatron-style; every column->row group ends in an all-reduce of a
           (tokens, d_model) activation, ``tp_allreduces`` times per forward (1 for an
           attention block or an MLP), and as many again in backward.
    """
    ring = (n - 1) / n
    w_bytes = n_params * bytes_per
    c_fwd = 2 * n_params * tokens / n / flops          # every strategy divides the FLOPs by n
    c_bwd = 2 * c_fwd
    if strategy == "dp":
        m_fwd, m_bwd = 0.0, 2 * ring * w_bytes / bw
    elif strategy == "fsdp":
        m_fwd = ring * w_bytes / bw
        m_bwd = 2 * ring * w_bytes / bw                # all-gather + reduce-scatter
    elif strategy == "tp":
        act = tokens * d_model * bytes_per
        m_fwd = tp_allreduces * 2 * ring * act / bw
        m_bwd = m_fwd
    else:
        raise ValueError(strategy)
    return dict(compute_fwd=c_fwd, compute_bwd=c_bwd, comm_fwd=m_fwd, comm_bwd=m_bwd,
                fwd_comm_bound=m_fwd > c_fwd, bwd_comm_bound=m_bwd > c_bwd)


def max_degree(n_params: float, tokens: float, d_model: int, flops: float, bw: float,
               strategy: str, bytes_per: int = 2, tp_allreduces: int = 1,
               limit: int = 1 << 16) -> int:
    """Largest power-of-two degree n for which the backward pass is not communication bound."""
    best, n = 1, 2
    while n <= limit:
        c = layer_parallel_costs(n_params, tokens, d_model, n, flops, bw, strategy,
                                 bytes_per, tp_allreduces)
        if c["bwd_comm_bound"] or c["fwd_comm_bound"]:
            break
        best, n = n, n * 2
    return best
