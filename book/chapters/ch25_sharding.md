# Sharded Data Parallelism from Scratch {#ch:sharding}

[@ch:distributed] surveyed the whole parallelism landscape at the level of cost models: the
$\alpha$--$\beta$ model, ring all-reduce, ZeRO's memory formulas, tensor and pipeline parallelism.
This chapter zooms into the one axis that every large training run uses first, the *data-parallel*
axis, and builds its production forms end to end: a process-group development setup you can run on a
laptop, the progression of DDP implementations, optimizer-state sharding done the way DeepSpeed and
FSDP do it, and a complete fully sharded data-parallel (FSDP) implementation with prefetching and
mixed-precision communication. It ends with an accounting method for deciding how far DP, FSDP and
TP can be scaled before communication dominates. That method is the tool you need to design a
parallel layout, or to answer an interviewer's "why not just add more GPUs?".

Everything here runs on CPU processes with the gloo backend, so you can develop and unit-test it
anywhere and then switch one string to `"nccl"` on a GPU node. The lab builds the same components
and checks them against single-process training to $10^{-4}$.

![Roadmap of the chapter. Each stage removes one source of redundancy from data parallelism and pays
for it with a precisely accountable amount of
communication.](figures/ch25_sharding-1.svg){#fig:sharding:roadmap}

## A distributed development environment {#sec:sharding:env}

### Processes, ranks and process groups

A PyTorch distributed job is a set of ordinary operating-system processes, usually one per GPU, that
join a *process group* and then call collectives on it. The vocabulary is small but it is worth
fixing precisely, because most launch bugs are vocabulary bugs:

::: description
a machine. A node has a *local world* of processes (typically 8, one per GPU).

the total number of processes in the default group, across all nodes.

a process's global id in $0,\dots,\text{world}-1$. Rank 0 conventionally does logging and checkpoint
writing.

the id of a process within its node, from 0 up to the local world size minus one. It selects the GPU
the process drives.

the library that moves the bytes: NCCL for CUDA tensors, gloo for CPU tensors (and for development),
MPI rarely.

how processes find each other: a TCP store at `MASTER_ADDR:MASTER_PORT` (or a c10d/etcd rendezvous
with `torchrun`).
:::

![Ranks and local ranks for two nodes with four GPUs each. Collectives inside a node ride NVLink;
anything that crosses the dashed boundary is bounded by the (5--20$\times$ slower) network
interface.](figures/ch25_sharding-2.svg){#fig:sharding:ranks}

### One script, two launchers

The single most useful habit for distributed work is to make every script runnable in two ways:
spawned from Python on a laptop with gloo, and launched with `torchrun` on a GPU node with NCCL.
[@lst:sharding:template] is the template the lab's `bench.py` follows. The rest of the
chapter assumes it.

``` {#lst:sharding:template .python caption="A launcher-agnostic distributed entry point."}
import os, torch, torch.distributed as dist

def setup():
    """Works under torchrun (env vars set) and under mp.spawn (we set them)."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)            # reads RANK/WORLD_SIZE/MASTER_*
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = setup()
    rank, world = dist.get_rank(), dist.get_world_size()
    x = torch.full((4,), float(rank), device=device)
    dist.all_reduce(x)                          # sum over ranks: 0+1+...+(world-1)
    if rank == 0:
        print("sum", x.tolist())
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

``` bash
# GPU node: one process per GPU
torchrun --standalone --nproc-per-node 8 demo.py
# laptop: llmsys.dist.launch.run_distributed spawns gloo processes and sets MASTER_*;
# or emulate torchrun on CPU:
torchrun --standalone --nproc-per-node 3 demo.py      # works: backend falls back to gloo
```

::: pitfall
[The three ways a distributed job hangs]{.box-title}

1.  **Mismatched collectives.** Every rank must issue the same collectives in the same order with
    matching shapes. An `if rank == 0: dist.all_reduce(...)` deadlocks the others forever. Set
    `TORCH_DISTRIBUTED_DEBUG=DETAIL` to have PyTorch check collective fingerprints, and
    `NCCL_DEBUG=INFO` to see the NCCL ring/tree setup.

2.  **One rank dies.** Its peers block in the next collective until the timeout (by default 10
    minutes for NCCL). Pass a shorter `timeout=` to `init_process_group` during development, and
    enable `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`.

3.  **Data-dependent control flow.** Early stopping when *this* rank's loss is NaN, an uneven last
    batch, or a hook that fires only if a parameter received a gradient all produce mismatched
    collectives. Make decisions collectively: all-reduce the flag first.
:::

### Benchmarking distributed code

Collectives are asynchronous twice over. With NCCL, even a "synchronous" call returns when the
operation is *enqueued* on the GPU stream, not when it has finished, so a wall-clock timer around
`dist.all_reduce` measures the launch overhead only. With `async_op=True` the call returns a work
handle immediately, and `handle.wait()` only makes the current stream wait for it. The measurement
protocol in [@lst:sharding:bench] handles both, and it is the same discipline as single-GPU
benchmarking ([@ch:machine]):

``` {#lst:sharding:bench .python caption="Timing a collective honestly and aggregating across ranks."}
import time, torch, torch.distributed as dist

def time_allreduce(numel, dtype=torch.float32, warmup=5, iters=20, device="cpu"):
    x = torch.randn(numel, dtype=dtype, device=device)
    for _ in range(warmup):                     # NCCL builds rings/channels lazily
        dist.all_reduce(x)
    sync = torch.cuda.synchronize if x.is_cuda else (lambda: None)
    sync(); dist.barrier()                      # start everyone together
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    sync()                                      # wait for the GPU, not just the launch
    dt = (time.perf_counter() - t0) / iters
    times = [None] * dist.get_world_size()
    dist.all_gather_object(times, dt)           # the slowest rank defines the step
    nbytes = x.numel() * x.element_size()
    p = dist.get_world_size()
    busbw = nbytes / max(times) * 2 * (p - 1) / p   # NCCL-tests convention
    return max(times), busbw
```

Report *bus bandwidth* (algorithm bandwidth scaled by $2(p-1)/p$ for all-reduce, see
[@sec:dist:ring]): it is directly comparable to the link's rated bandwidth and is
independent of the number of ranks in the bandwidth-bound regime. Sweep message sizes from kilobytes
to gigabytes. Small messages measure $\alpha$ (latency), large ones measure $\beta$ (bandwidth), and
the knee between them tells you what bucket size your DDP should use.

::: napkinmath
[Where is the knee?]{.box-title}

In the $\alpha$--$\beta$ model an all-reduce of $n$ bytes over $p$ ranks costs
$2(p-1)\alpha + 2\frac{p-1}{p}n\beta$. The two terms are equal at $n^\star = p\,\alpha/\beta$. For 8
GPUs on NVLink with an effective $\alpha\approx10\,\mathrm{\micro\second}$ and
$1/\beta \approx 450$ GB/s per direction, $n^\star = 8\times10^{-5}\times 4.5\times10^{11}
\approx 36$ MB. Messages much smaller than tens of MB are latency-bound, which is why DDP's default
bucket size is 25 MB and why one all-reduce per small tensor (a LayerNorm scale has a few kilobytes)
is a bad idea. Across nodes $\alpha$ is larger and the knee moves further out.
:::

## The DDP progression {#sec:sharding:ddp}

Plain data parallelism has one job: after every backward pass, replace each rank's local gradient by
the average over ranks. Since every rank starts from the same weights (broadcast from rank 0) and
applies the same optimizer to the same averaged gradient, the replicas stay identical without ever
exchanging weights again. The interesting engineering is all in *how* and *when* the averaging
happens. There are four classic designs, and their timelines are in [@fig:sharding:ddp].

1.  **Per-parameter, after backward.** Loop over parameters and all-reduce each gradient
    (`llmsys.dist.ddp.allreduce_gradients`). Simple and correct, but it pays one latency term per
    tensor, and communication starts only after the whole backward has finished.

2.  **Flat, after backward.** Concatenate every gradient into one buffer, all-reduce once, and
    scatter the result back (`allreduce_flat` in [@lab:sharding]). This pays the latency once and moves the
    same bytes, which helps a lot when there are many small tensors. It still does not overlap with
    compute, and it needs a temporary buffer the size of the gradients.

3.  **Per-parameter, overlapped.** Register a hook that fires when a parameter's gradient has been
    accumulated (`Tensor.register_post_accumulate_grad_hook`) and launch an *asynchronous*
    all-reduce right there. The last layer's gradients are ready first, so their communication runs
    while autograd is still computing earlier layers. Before `optimizer.step()` we wait on all
    handles.

4.  **Bucketed, overlapped.** Group parameters in reverse registration order into buckets of
    $\sim$`<!-- -->`{=html}25 MB and launch one async all-reduce per full bucket. This combines the
    latency amortisation of (2) with the overlap of (3). It is what
    `torch.nn.parallel.DistributedDataParallel` does and what our `BucketedDDP`
    ([@lst:dist:zero]'s neighbour in `llmsys/dist/ddp.py`) implements. Setting
    `bucket_cap_mb=0` degenerates it into design (3).

![Four ways to average gradients, drawn for the same model. Communication (purple) that lies under
the backward bar (dark blue) is hidden; communication to its right adds to the step time. Design (3)
launches many tiny messages whose latencies add up; (4) sends a few latency-amortised messages and
hides most of them behind backward.](figures/ch25_sharding-3.svg){#fig:sharding:ddp}

::: systemslens
[A step-time model for the four designs]{.box-title}

Let $T_f, T_b$ be forward and backward compute, $m$ the number of gradient tensors, $G$ the gradient
bytes and $A(n)=2(p-1)\alpha + 2\frac{p-1}{p}n\beta$ the all-reduce cost. Then roughly
$$\begin{align*}
T^{(1)} &\approx T_f + T_b + \textstyle\sum_{i=1}^{m} A(g_i) = T_f+T_b + 2m(p-1)\alpha + 2\tfrac{p-1}{p}G\beta,\\
T^{(2)} &\approx T_f + T_b + A(G) + T_\text{copy},\\
T^{(3,4)} &\approx T_f + \max\!\big(T_b,\; \text{comm. of all but the last bucket}\big) + A(g_\text{last}).
\end{align*}$$ Overlap can hide at most $T_b$ worth of communication: if
$2\frac{p-1}{p}G\beta > T_b$, the job is communication-bound no matter how clever the scheduling is,
and the remedy is fewer bytes (bf16 gradients, sharding, compression) or more compute per byte (a
larger per-GPU batch). The smallest bucket, the *last* one launched, is the exposed tail. PyTorch
DDP therefore makes the first bucket (the last layers) small so that communication starts early.
:::

``` {#lst:sharding:overlap .python caption="Design (3): overlapping per-parameter all-reduces with post-accumulate hooks."}
class OverlapDDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module, self.handles = module, []
        broadcast_parameters(module)                  # identical start (Ch. 14)
        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p):                               # runs inside autograd's backward
        p.grad /= dist.get_world_size()               # pre-divide: sum == mean
        self.handles.append(dist.all_reduce(p.grad, async_op=True))

    def forward(self, *args, **kw):
        return self.module(*args, **kw)

    def finish_gradient_synchronization(self):        # call before optimizer.step()
        for h in self.handles:
            h.wait()
        self.handles.clear()
```

::: pitfall
[Gradient accumulation with DDP]{.box-title}

With $k$ micro-batches per optimizer step, every hook fires $k$ times, so a naive implementation
all-reduces $k$ times. PyTorch's `no_sync()` context disables the hooks for the first $k-1$
micro-batches. In a from-scratch wrapper, keep a flag that the hook checks. Also remember that
`p.grad` is *accumulated* across micro-batches. Pre-dividing inside the hook, as in
[@lst:sharding:overlap], is only correct if it runs once per optimizer step.
:::

::: napkinmath
[How many messages does a real model send?]{.box-title}

Qwen3-0.6B has 28 blocks with 11 weight tensors each (4 attention projections, 2 QK-norm scales, 3
MLP matrices, 2 block norms) plus the embedding and final norm: $28\times11+2=310$ tensors. Design
(1) pays $310\times2(p-1)\alpha$. With $p=8$ and $\alpha=10\,\mathrm{\micro\second}$ that is
$\approx43\,\mathrm{\milli\second}$ of pure latency per step, comparable to the backward pass of a
small batch. The same 1.2 GB of bf16 gradients in a single flat message costs
$2\cdot\frac{7}{8}\cdot 1.2\times10^{9}/4.5\times10^{11}\approx4.7\,\mathrm{\milli\second}$ of
bandwidth time plus one latency.
:::

## Optimizer-state sharding, element-wise {#sec:sharding:zero}

With mixed-precision AdamW every parameter costs 16 bytes of model state ([@ch:pretrain]): 2
for the bf16 weight, 2 for the bf16 gradient, and 12 for the fp32 master copy and the two Adam
moments. Plain DP replicates all 16 bytes on every rank. ZeRO stage 1 removes the replication of the
12 optimizer bytes. [@ch:distributed] implemented this with *parameter ownership*: each whole
tensor belongs to one rank, gradients are reduced to it and the updated tensor is broadcast back
([@lst:dist:zero]). That design is pleasant to read but has three weaknesses. Load balance
is only as good as the largest tensor allows: the embedding matrix of a small model can be a third
of all parameters. All traffic for a tensor passes through one root. And the optimizer state is laid
out per tensor, so a checkpoint saved on 8 ranks cannot easily be loaded on 16.

The production design flattens instead. [@fig:sharding:flat] shows it for three ranks.

![Element-wise sharding of a flattened parameter group across three ranks. Shard boundaries ignore
tensor boundaries (rank 1's shard holds the tail of $W_k$ and the head of $W_1$), so the balance is
perfect up to the padding.](figures/ch25_sharding-4.svg){#fig:sharding:flat}

The step is three collectives that together move exactly the bytes of one all-reduce:

1.  flatten the gradients and **reduce-scatter** them: rank $r$ receives the rank-averaged gradient
    of exactly its shard;

2.  run the *unmodified* wrapped optimizer on the shard, so its state exists only for $1/p$ of the
    elements;

3.  **all-gather** the updated shards and copy them back into the model's parameters.

``` {#lst:sharding:flatzero .python caption="The step of `FlatShardedOptimizer` (`llmsys/dist/zero_wrapper.py`)."}
@torch.no_grad()
def step(self):
    for g in self._groups:                       # one flat buffer per param group
        flat_g = flatten_tensors([p.grad for p in g["params"]], self.world)
        shard_g = torch.empty_like(g["shard"])
        dist.reduce_scatter_tensor(shard_g, flat_g)      # sum of my slice
        g["shard"].grad = shard_g / self.world
    self.inner.step()                            # e.g. AdamW over the shards only
    for g in self._groups:
        full = torch.empty(g["numel"], device=g["shard"].device)
        dist.all_gather_into_tensor(full, g["shard"].detach())
        for p, v in zip(g["params"], unflatten_views(full, g["shapes"])):
            p.copy_(v.to(p.dtype))               # write updated weights back
```

Because the wrapper builds its inner optimizer from parameter *groups* whose `params` are the shard
tensors, everything that manipulates `param_groups` (learning-rate schedulers, per-group weight
decay) keeps working. Weight decay is the reason to keep one flat buffer per group: decayed and
non-decayed parameters must not share a shard.

::: pitfall
[Global-norm clipping needs one more collective]{.box-title}

`clip_grad_norm_` computes $\lVert g\rVert_2$ over the gradients it can see. After a reduce-scatter
each rank sees only its shard, so the local norm is wrong. The fix is one scalar all-reduce:
$\lVert g\rVert_2^2 = \sum_r \lVert g^{(r)}\rVert_2^2$. Forgetting it gives every rank a different
clip coefficient and the replicas silently diverge. The same applies to any statistic that spans
parameters, such as gradient-noise-scale estimates or Adam-mini's per-block statistics.
:::

::: napkinmath
[ZeRO-1 for a 1.7B model on 4 GPUs]{.box-title}

$\Psi = 1.7\times10^9$. Plain DP holds $16\Psi = 27.2$ GB of model state per GPU. With element-wise
ZeRO-1 the replicated part is the bf16 weights and gradients, $4\Psi = 6.8$ GB, and the sharded part
is $12\Psi/4 = 5.1$ GB, for 11.9 GB in total. The communication per step is still
$2\cdot\frac{3}{4}\cdot 2\Psi = 5.1$ GB per rank, the same as an all-reduce. On a 24 GB card the
difference decides whether the model trains at all.
:::

::: {#tab:sharding:zero .table-float}
|  | parameter ownership ([@ch:distributed]) | element-wise flat shards (this chapter) |
|:---|:---|:---|
| balance | greedy; limited by largest tensor | perfect up to $p-1$ padding elements |
| collectives | reduce to owner, broadcast from owner | reduce-scatter, all-gather (ring-friendly) |
| temporary memory | none | one flat gradient buffer per group |
| checkpoints | per-tensor, owner-dependent | one shard per rank; reshard by re-cutting the flat vector |
| optimizer | must work tensor-by-tensor | any `torch.optim` class, unmodified |

: Two implementations of optimizer-state sharding.
:::

## Fully sharded data parallelism from scratch {#sec:sharding:fsdp}

ZeRO-1 still replicates the bf16 weights and gradients: $4\Psi$ bytes per GPU, 32 GB for an
8B-parameter model, before a single activation is stored. FSDP (ZeRO stage 3) shards those too. At
rest every rank holds only its $1/p$ slice of every parameter. Just before a layer runs, its weights
are all-gathered; just after, they are freed. In backward the weights are gathered again, and the
resulting full gradients are reduce-scattered back to the shards. [@ch:distributed] gave the
memory formula ($16\Psi/p$) and the communication volume ($3\Psi$ instead of $2\Psi$). This section
builds the machinery that makes those formulas true. The code is `llmsys/dist/fsdp.py`, about 300
lines.

### Units and flat parameters

Gathering one tensor at a time would pay the collective latency hundreds of times per pass, so FSDP
groups parameters into **units**. Our implementation makes each transformer block one unit and puts
everything left over (embeddings, final norm, LM head) into a *root* unit. Each unit's parameters
are flattened into one fp32 vector, padded to a multiple of $p$ and sharded element-wise exactly as
in [@fig:sharding:flat]. The shards are the *only* `nn.Parameter`s left in the model, so a
standard optimizer built over `fsdp.parameters()` updates fp32 master shards and holds Adam state
for $1/p$ of the model.

The construction is a small piece of module surgery ([@lst:sharding:makeunit]). For every
module inside the unit we record *slots*, triples (module, attribute name, index into the unit's
parameter list), and then delete the parameters from the module's `_parameters` dictionary. Tied
weights (Qwen3's input and output embeddings are the same tensor) produce two slots pointing at the
same index. Parameters already claimed by an inner unit are skipped, which is how the root unit ends
up with only the leftovers.

``` {#lst:sharding:makeunit .python caption="Turning a submodule into an FSDP unit (abridged)."}
def _make_unit(self, name, um, claimed):
    slots, params, index = [], [], {}
    for mod in um.modules():
        for attr, p in list(mod._parameters.items()):
            if p is None or (id(p) in claimed and id(p) not in index):
                continue                          # belongs to an inner unit
            if id(p) not in index:                # tied weights appear once
                index[id(p)] = len(params); params.append(p)
            slots.append((mod, attr, index[id(p)]))
    for p in params:
        claimed[id(p)] = p
    for mod, attr, _ in slots:
        del mod._parameters[attr]                 # params become plain attributes
    return _FlatUnit(name, um, slots, params, self.rank, self.world,
                     self.compute_dtype)          # builds flat vector + my shard
```

### The life cycle of a unit

Every unit cycles through the states of [@fig:sharding:lifecycle] once per training step.
Four hooks drive the transitions, and none of them requires changing the model's code.

![The per-step life cycle of one FSDP unit. Only the GATHERED and RE-GATHERED states hold the full
weights, and only one or two units are in them at any
time.](figures/ch25_sharding-5.svg){#fig:sharding:lifecycle}

##### Pre-forward: all-gather and attach views.

A forward pre-hook on the unit's module casts the local shard to the compute dtype, *then*
all-gathers it (so a bf16 run communicates 2 bytes per element, not 4), and stores the result in the
unit's flat buffer. It then creates an autograd leaf over that buffer and `setattr`s each slot to a
*view* of the leaf. To the module's own `forward` nothing has changed: `self.weight` is a tensor of
the right shape. To autograd, all of the unit's weights are slices of one leaf, so their gradients
accumulate into one flat `.grad`, which is exactly the layout the reduce-scatter wants.

##### Post-forward: free the storage.

After the forward, the hook resizes the flat buffer's untyped storage to zero bytes. The tensor
objects survive (autograd has saved views of them for backward), but their bytes are returned to the
allocator. This is the trick that makes FSDP's memory claim true, and it is also what PyTorch FSDP1
does internally.

##### Pre-backward: re-gather into the same storage.

The post-forward hook also registers a gradient hook on the unit's *output*. Autograd calls it when
the gradient with respect to the output is complete, which is the moment the unit's own backward is
about to run. It re-allocates the storage and refills it with the gathered weights, so the views
autograd saved now point at valid data again.

::: pitfall
[Refilling storage without upsetting autograd]{.box-title}

Autograd protects saved tensors with *version counters*: if a tensor saved for backward is modified
in place, backward raises "one of the variables needed for gradient computation has been modified by
an inplace operation". Writing the re-gathered weights with `flat.copy_(buf)` bumps the version
counter of `flat` and of every view that shares it, and the error follows. The fix is to copy at the
*storage* level, `flat.untyped_storage().copy_(buf.untyped_storage())`, which bypasses the
tensor-level bookkeeping. This is safe because we are restoring the identical values. Production
FSDP avoids even this copy by gathering directly into the storage on a side stream.
:::

##### Post-accumulate: reduce-scatter and free.

A `register_post_accumulate_grad_hook` on the flat leaf fires once the unit's complete gradient is
available. The leaf receives contributions from all of its views, and autograd sums them before the
hook runs, so tied weights are handled for free. The hook casts the gradient to fp32, launches an
asynchronous `reduce_scatter_tensor` into a shard-sized buffer, drops the full gradient and frees
the weight storage. `finish_gradient_synchronization()` waits on all handles, divides by $p$ and
installs the result as `shard.grad`, adding to it if a gradient is already there, which is gradient
accumulation. The optimizer then steps on fp32 master shards with fp32 gradients, regardless of the
compute dtype.

``` {#lst:sharding:hooks .python caption="The hooks that drive a unit (abridged from `FullyShardedDP`)."}
def pre_forward(mod, args, kwargs):
    u.unshard()                                   # all-gather (or wait for prefetch)
    full = u.attach_views(requires_grad=torch.is_grad_enabled())
    if full.requires_grad:
        full.register_post_accumulate_grad_hook(lambda t: self._on_grad(u))
    if self.prefetch_enabled:
        nxt = self._next_in_order(u, +1)          # recorded during the first step
        if nxt is not None:
            nxt.prefetch()                        # async all-gather of the next unit

def post_forward(mod, args, kwargs, out):
    if self.reshard_after_forward or not torch.is_grad_enabled():
        u.reshard()                               # storage().resize_(0)
    for t in tensors_in(out):
        if t.requires_grad:                       # fires when backward reaches u
            t.register_hook(lambda g: self._pre_backward(u))

def _on_grad(self, u):                            # u's full gradient is complete
    g = u.full.grad.to(torch.float32)
    shard_grad = torch.empty_like(u.shard)
    work = dist.reduce_scatter_tensor(shard_grad, g, async_op=True)
    u.grad_work = (work, shard_grad, g)           # keep g alive until the wait
    u.full.grad = None; u.reshard()
```

### Prefetching: taking communication off the critical path

Without prefetching each unit's all-gather sits on the critical path: compute waits for
communication, which waits for compute. With prefetching, the pre-forward hook of unit $i$ issues an
*asynchronous* all-gather for unit $i+1$ before unit $i$ computes. When unit $i+1$'s own hook runs,
it only waits on the handle. In backward the order reverses: unit $i$'s pre-backward hook prefetches
unit $i-1$. The forward order is recorded during the first step, since modules may be called in an
order unrelated to their registration order, so prefetching starts from the second step.
[@fig:sharding:prefetch] shows the effect. The price is memory: two units are materialised
at a time instead of one.

![Forward pass of four FSDP units. Without prefetch every all-gather (AG) is exposed; with prefetch,
AG$_{i+1}$ overlaps F$_i$ and only the first gather is exposed, as long as a gather is shorter than
a unit's compute.](figures/ch25_sharding-6.svg){#fig:sharding:prefetch}

::: systemslens
[When does prefetching hide the all-gather?]{.box-title}

A unit with $P_u$ parameters processing $T$ tokens per rank costs $2P_uT/C$ seconds of forward
compute on a device with $C$ FLOP/s. Gathering it in bf16 costs $\frac{p-1}{p}\,2P_u/W$ seconds with
ring all-gather at egress bandwidth $W$. The gather hides behind compute when
$$\frac{2P_u T}{C} \;\ge\; \frac{2P_u}{W}\quad\Longleftrightarrow\quad T \;\ge\; \frac{C}{W},$$
which is independent of the unit's size. The tokens *per rank per step* must exceed the hardware's
FLOP-to-byte ratio. For an H100 ($C=989$ TFLOP/s) on NVLink ($W\approx450$ GB/s) this is
$\approx2200$ tokens; across nodes with 50 GB/s per GPU it is $\approx20{,}000$ tokens. The same
ratio governs plain DP ([@sec:sharding:accounting]). FSDP simply pays it in forward too.
:::

### Mixed precision, checkpoints and what not to shard

Three practical decisions round out the design.

- **Cast before you communicate.** Master shards are fp32. Casting the shard to bf16 *before* the
  all-gather halves the traffic and the gathered memory. Gradients are reduce-scattered in fp32 in
  our implementation (the sum of many bf16 numbers loses precision). PyTorch's
  `MixedPrecisionPolicy(reduce_dtype=...)` exposes the same choice.

- **Checkpointing.** `full_state()` gathers every unit and returns a normal state dict; that is fine
  for small models and for evaluation. For large models save one file of shards per rank and the
  flat layout metadata (shapes, padding), or use `torch.distributed.checkpoint`, which can reshard
  on load.

- **Granularity.** Every unit costs one all-gather and one reduce-scatter latency, and the largest
  one or two units set the transient memory. One unit per block is the usual balance. Tiny tensors
  such as norm scales go into their block's unit rather than their own (a 4 KB gather is pure
  latency). Units should not be so large that two of them do not fit next to the activations.

::: napkinmath
[An 8B model on 8 GPUs with FSDP]{.box-title}

Take Qwen3-8B dimensions: $\Psi\approx8.2\times10^9$, made of 36 blocks of $\approx0.19\times10^9$
parameters each plus an untied embedding and LM head of $0.62\times10^9$ parameters each, which form
the root unit ($1.24\times10^9$). The persistent state is $16\Psi/8 = 16.4$ GB per GPU (fp32 master,
fp32 gradient shard and Adam's two moments). The transient state is two gathered blocks in bf16 plus
one block's full gradient, $3\times0.19\times10^9\times2 \approx 1.1$ GB, plus the root unit, which
stays gathered from the first forward to the end of backward because the embedding is used first and
the head last: $1.24\times10^9\times2 = 2.5$ GB of weights and as much again for its gradient. That
is about 22 GB before activations, against 131 GB for plain DP, which does not fit on an 80 GB GPU
at all. The root unit is the largest transient term, which is why production setups wrap the
embedding and the head as *separate* units.
:::

### Testing it

The test that matters is equivalence. Train the same model for a few steps on the full batch in a
single process, then train it under FSDP with each of $p$ ranks seeing $1/p$ of the batch, and
compare all parameters. [@lst:sharding:test] is the core of [@lab:sharding]'s test. It also asserts
the invariants that catch most bugs: after every step nothing is gathered, every unit did exactly
one reduce-scatter, and two all-gathers per step (forward and backward).

``` {#lst:sharding:test .python caption="Equivalence and invariant checks for FSDP (inside each of 3 ranks)."}
torch.manual_seed(0 if rank == 0 else 1234)   # FSDP must broadcast rank 0's init
m = Qwen3Model(cfg)
fsdp = FullyShardedDP(m, units=list(m.trf_blocks), prefetch=True)
opt = torch.optim.AdamW(fsdp.parameters(), lr=1e-2)
for _ in range(3):
    opt.zero_grad()
    loss(fsdp, x.chunk(world)[rank], y.chunk(world)[rank]).backward()
    fsdp.finish_gradient_synchronization()
    opt.step()
    assert fsdp.gathered_numel() == 0              # everything freed
state = fsdp.full_state()                          # all-gather for comparison
for name, ref in reference_params.items():        # single-process, full batch
    assert torch.allclose(state[name], ref, atol=1e-4)
assert fsdp.comm_stats()["unit1"] == {"all_gathers": 6, "reduce_scatters": 3, ...}
```

The tolerance is $10^{-4}$ rather than exact equality: a reduce-scatter sums gradients in a
different order from a single-process backward, and Adam's normalisation by $\sqrt{v}$ amplifies
those last-bit differences in the first steps.

## FSDP2, DTensor and hybrid sharding in production {#sec:sharding:fsdp2}

PyTorch's current FSDP ("FSDP2", `torch.distributed.fsdp.fully_shard`) keeps the semantics we built
but changes the data structure. Instead of one flat parameter per unit, every parameter becomes a
**DTensor**, a tensor annotated with a *device mesh* and a *placement* per mesh dimension:

- `Shard(d)`: the tensor is split along dimension $d$ across that mesh dimension;

- `Replicate()`: every rank holds the full tensor;

- `Partial()`: every rank holds a summand; the logical value is the sum (this is what a gradient is
  before its reduce-scatter).

FSDP2 shards every parameter along dimension 0 (`Shard(0)`). A unit still all-gathers its parameters
together (into one buffer, via a "copy-in/copy-out" step), so the latency amortisation remains, but
each parameter keeps its identity. That makes per-parameter settings (freezing, different dtypes,
LoRA adapters), state-dict handling, and composition with tensor parallelism much simpler.
Composition with TP is the reason for DTensor in the first place: a TP weight is `Shard(0)` on the
`tp` mesh dimension and FSDP adds `Shard(0)` on the `dp` dimension ([@lst:dist:prod]).

``` {#lst:sharding:fsdp2 .python caption="FSDP2 on our Qwen3 model with hybrid sharding (HSDP): shard inside a node, replicate across nodes."}
import torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from llmsys.qwen3 import Qwen3Model, QWEN3_CONFIGS

dist.init_process_group("nccl")
nodes, gpus = dist.get_world_size() // 8, 8
mesh = init_device_mesh("cuda", (nodes, gpus), mesh_dim_names=("replicate", "shard"))
with torch.device("meta"):                          # no memory until sharded
    model = Qwen3Model(QWEN3_CONFIGS["8B"])
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for blk in model.trf_blocks:                        # one unit per block
    fully_shard(blk, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
fully_shard(model, mesh=mesh, mp_policy=mp)         # root unit
model.to_empty(device="cuda")                       # allocate local shards only
init_weights(model)                                 # or load a sharded checkpoint (DCP)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
```

A 2-D mesh passed to `fully_shard` means *hybrid* sharding. Parameters are sharded over the inner
(`shard`) dimension, where NVLink makes the frequent all-gathers cheap, and replicated over the
outer (`replicate`) dimension, where only an all-reduce of the gradient shards crosses the slower
network once per step. HSDP is the default choice when a model's state fits in one node's aggregate
memory. Its cross-node traffic is that of DP on a $1/8$-size gradient, and it never puts per-layer
all-gathers on the inter-node links.

::: deepdive
[`reshard_after_forward` as a dial]{.box-title}

Setting `reshard_after_forward=False` for the last few blocks (or for all of them, which is ZeRO-2
behaviour for parameters) keeps their gathered weights alive until backward: one fewer all-gather
per unit, at the price of holding those units in full. Because the first units in backward are the
last ones in forward, keeping just the final unit gathered removes one exposed all-gather at the
forward/backward boundary for very little memory. FSDP2 also accepts an integer here, resharding to
a smaller intra-node group after forward, which is a middle ground for multi-node runs.
:::

## Parallelism accounting: a reusable method {#sec:sharding:accounting}

How many GPUs can a strategy use before communication, not compute, sets the step time? The question
has a clean answer once we commit to a simple model and apply it systematically. The method has four
steps, and it works for any layer and any strategy:

1.  **Write the layer as matmuls** with explicit shapes, forward and backward. A matmul
    $(a\times b)(b\times c)$ costs $2abc$ FLOPs. Backward costs twice the forward: one matmul for
    the input gradient and one for the weight gradient per forward matmul.

2.  **Decide what each device holds** (which dimension of which tensor is split over the $n$
    devices) and therefore which tensors must be communicated, and when, to make every matmul local.

3.  **Cost the collectives** with the ring model of a fully connected set of devices, each with
    egress bandwidth $W$: all-gather and reduce-scatter of a tensor of $S$ bytes take
    $\frac{n-1}{n}\frac{S}{W}$, an all-reduce twice that. Ignore latency, which is valid for large
    messages.

4.  **Compare per-device compute time** ($\text{FLOPs}/C$) with communication time, phase by phase,
    assuming perfect overlap. The strategy stops scaling when communication exceeds compute.

##### Data parallelism for a generic dense layer.

Let a layer have $P$ parameters (any number of matmuls) and process $T$ tokens per step in total, so
$T/n$ per device. In forward each weight element is used in one multiply-add per token: $2P\,T/n$
FLOPs per device. Backward costs $4P\,T/n$. DP needs no communication in forward. In backward it
all-reduces the $P$ weight gradients ($2P$ bytes in bf16):
$$t_\text{comp}^\text{bwd} = \frac{4PT}{nC},\qquad
  t_\text{comm}^\text{bwd} = 2\,\frac{n-1}{n}\,\frac{2P}{W}\approx\frac{4P}{W}
  \quad\Longrightarrow\quad
  \text{compute-bound iff}\quad \frac{T}{n} \gtrsim \frac{C}{W}.$$ $P$ cancels. Whether DP is
communication-bound depends only on the *tokens per device per step* compared with the hardware
ratio $C/W$, which has units of FLOPs per byte, the same quantity as a roofline ridge point
([@ch:machine]) but for the network instead of HBM. Adding GPUs at a fixed global batch
shrinks $T/n$ until the inequality fails. That is the communication reason, alongside the
optimisation reason in [@ch:scalinglaws], why the global batch must grow with the
data-parallel degree.

##### FSDP.

The forward now all-gathers $P$ weights ($\frac{n-1}{n}\frac{2P}{W}$) against $2PT/(nC)$ of compute;
the backward all-gathers again and reduce-scatters instead of all-reducing. Both phases give the
same condition, $T/n \gtrsim C/W$ up to a constant factor, so FSDP scales as far as DP does and
costs about $1.5\times$ the bytes. Its benefit is memory, not speed.

##### Tensor parallelism.

Split each weight matrix Megatron-style ([@sec:dist:tp]): a column-parallel matmul followed
by a row-parallel one needs a single all-reduce of the $(T, d_{\text{model}})$ activation per
forward and one more in backward. For a group of matmuls with $P_g$ parameters ending in one
all-reduce: $$t_\text{comp}^\text{fwd} = \frac{2P_gT}{nC},\qquad
  t_\text{comm}^\text{fwd} = 2\,\frac{n-1}{n}\,\frac{2Td_{\text{model}}}{W}
  \quad\Longrightarrow\quad
  \text{compute-bound iff}\quad n - 1 \lesssim \frac{P_g}{2\,d_{\text{model}}}\cdot\frac{W}{C}.$$
Now $T$ cancels, and the limit is a property of the *model shape*: the parameters per all-reduce,
divided by the width, measured in units of $C/W$. TP degree cannot be raised by increasing the
batch, which is why it is used only inside a high-bandwidth domain.

::: napkinmath
[Accounting for a Qwen3-8B block on H100s]{.box-title}

$d_{\text{model}}=4096$; the attention projections hold
$P_\text{attn} = 2\cdot4096^2 + 2\cdot4096\cdot1024
\approx 4.19\times10^7$ parameters (GQA makes $W_K, W_V$ narrow), and the SwiGLU MLP holds
$3\cdot4096\cdot12288\approx1.51\times10^8$. With $C=989$ TFLOP/s and NVLink $W=450$ GB/s,
$C/W\approx2200$.

- **DP/FSDP**: with $T = 2^{20}\approx10^6$ tokens per step, $T/n \ge 2200$ allows $n\le 477$; on
  50 GB/s inter-node links $C/W\approx19{,}800$ and $n\le 53$.

- **TP, attention group**: $n-1 \le \frac{4.19\times10^7}{2\cdot4096}\cdot\frac{1}{2200}\approx2.3$.
  The attention all-reduce is exposed already at $n=4$.

- **TP, whole block**: two all-reduces for $1.93\times10^8$ parameters gives
  $n-1\le\frac{9.65\times10^7}{8192\cdot2200}\approx5.4$.

So TP-8 on an 8B model is at the edge of compute-bound even on NVLink, and it only works in practice
because frameworks overlap the all-reduces with neighbouring compute ("async TP") or replace them
with reduce-scatter/all-gather pairs (sequence parallelism, [@sec:dist:cp]). [@lab:sharding]'s
`layer_parallel_costs` and `max_degree` reproduce these numbers.
:::

![The accounting of [@sec:sharding:accounting] for the attention projections of one
Qwen3-8B block ($T=2^{20}$ tokens per step, H100 at 989 TFLOP/s, $W=450$ GB/s). Compute falls as
$1/n$. DP's communication is almost constant ($\approx4P/W$), so the curves cross near
$n\approx480$. TP's communication is *larger than compute already at $n=4$*, because it moves
activations ($T\times d_{\text{model}}$) rather than weights. Computed with
`llmsys.dist.fsdp.layer_parallel_costs`.](figures/ch25_sharding-7.svg){#fig:sharding:accounting}

##### Two-dimensional layouts.

Combining FSDP of degree $n_f$ with TP of degree $n_t$ on $n = n_f n_t$ devices, each device holds
$1/(n_f n_t)$ of the weights and $T/n_f$ tokens. The TP group's activation all-reduces now carry
only $T/n_f$ tokens, against compute that is also divided by $n_f$, so the TP condition is
unchanged: $n_t - 1\lesssim P_g W_t/(2d_{\text{model}}C)$ with $W_t$ the TP group's bandwidth. The
FSDP all-gathers move only the $1/n_t$ slice each TP rank needs, against compute $2PT/(n_f n_t C)$,
which gives $T/n_f \gtrsim C/W_f$. In words, the two conditions decouple. Choose $n_t$ from the
model shape and the fast-domain bandwidth, then choose $n_f$ from the batch and the slow-domain
bandwidth. Put the TP dimension innermost in the device mesh so that it rides NVLink, which is
exactly the layout of [@lst:dist:prod]. The exercises ask you to redo this derivation for
the MLP of a GPT-2-style block, which uses a plain GELU MLP.

### Choosing a collective algorithm

The accounting above uses the ring because it is bandwidth-optimal: every rank sends and receives
$2\frac{p-1}{p}n$ bytes, the minimum possible for an all-reduce. Its latency term $2(p-1)\alpha$
grows linearly with $p$, however, and at thousands of ranks the latency of small messages matters.
NCCL and other libraries therefore keep several algorithms and pick one per message size and
topology:

- **Bidirectional ring**: send half the data clockwise and half anticlockwise. On full-duplex links
  both directions are busy, which halves the time of a unidirectional ring.

- **Trees**: reduce up a binary tree and broadcast down it, with $O(\log p)$ latency. NCCL's *double
  binary tree* uses two complementary trees so that every rank is a leaf in one and an interior node
  in the other, recovering full bandwidth for large messages.

- **Recursive halving--doubling** (Rabenseifner): reduce-scatter by exchanging halves with partners
  at distance $p/2, p/4,\dots$, then all-gather in reverse. It is bandwidth-optimal *and* has
  $O(\log p)$ latency, but it needs power-of-two groups and good bisection bandwidth.

- **Hierarchical (2-D)**: reduce-scatter inside each node over NVLink, all-reduce the $1/g$ slices
  across nodes (each GPU using its own NIC, "rail-optimised"), then all-gather inside the node.
  Inter-node traffic per GPU falls by the node size $g$.

![The $\alpha$--$\beta$ model for 64 ranks (`llmsys.dist.costs`). The first three curves assume
every hop is a network link ($\alpha=20\,\mathrm{\micro\second}$, 50 GB/s). The ring pays
$126\alpha$ of latency for small messages, while tree and halving--doubling pay $12\alpha$. For
large messages ring and halving--doubling are bandwidth-optimal and the simple tree is $\sim2\times$
worse. The hierarchical curve assumes 8 nodes of 8 GPUs ($\alpha=10\,\mathrm{\micro\second}$,
450 GB/s inside a node, one 50 GB/s NIC per GPU). Keeping most hops on NVLink cuts large-message
time by the node size. NCCL's multi-ring and tree algorithms achieve the same effect without an
explicit hierarchy. Analytic model, not
measurements.](figures/ch25_sharding-8.svg){#fig:sharding:collectives}

## Debugging and validating distributed training {#sec:sharding:debug}

Distributed bugs are expensive because they surface late (a slowly diverging loss), far away (on
rank 113 of 512), or as hangs with no stack trace. A short checklist catches most of them before
they cost a cluster-day.

1.  **Equivalence first.** Every new wrapper gets the test of [@lst:sharding:test]: same
    seed, a few steps, compare all parameters with single-process full-batch training. Run it with
    $p=1$ (catches bookkeeping bugs), $p=2$, and an odd $p$ such as 3 (catches padding and
    uneven-split bugs).

2.  **Replica checksums.** For DP and ZeRO-1/2, all replicas must hold bit-identical weights. Every
    $N$ steps, all-gather a cheap checksum (sum of a few parameters cast to fp64) and assert
    equality; a mismatch means non-deterministic or rank-dependent code ran.

3.  **Collective fingerprints.** `TORCH_DISTRIBUTED_DEBUG=DETAIL` verifies that all ranks call
    collectives with the same shapes and order, which turns hangs into errors.

4.  **Flight recorder.** PyTorch can dump the last collectives each rank issued when a timeout fires
    (`TORCH_NCCL_TRACE_BUFFER_SIZE`). Comparing dumps across ranks shows who is behind.

5.  **NaN provenance.** Check for non-finite values in the *local* gradient before the reduction.
    After an all-reduce one rank's NaN is everyone's NaN and the culprit is gone.

6.  **Memory as a test.** Assert the memory invariants: once gradient synchronisation has finished
    nothing is gathered, and the parameter bytes per rank equal the sharded formula within padding.
    The lab tests do both.

::: systemslens
[Profiling overlap on a GPU]{.box-title}

To *see* overlap, wrap units in NVTX ranges (`torch.cuda.nvtx.range_push("unit3/fwd")`) and run
under Nsight Systems (`nsys profile -t cuda,nvtx,osrt python train.py`). NCCL kernels appear on
their own stream; a healthy FSDP trace shows `ncclDevKernel_AllGather` on the communication stream
*under* the matmul kernels of the previous unit. When the all-gathers line up end to end with the
compute and the compute stream has gaps between units, prefetching is not happening or $T/n < C/W$.
[@ch:perf] covers the profiler workflow in detail.
:::

## Summary {#sec:sharding:summary}

- Make every distributed script launchable with `torchrun` (NCCL) and from Python with gloo; time
  collectives after warm-up and synchronisation, aggregate the slowest rank, and report bus
  bandwidth.

- DDP implementations differ in *when* gradients are averaged and *how many messages* that takes.
  Overlap hides at most the backward pass; the last bucket is the exposed tail.

- Element-wise ZeRO-1 flattens parameter groups, reduce-scatters gradients into shards, steps any
  optimizer on the shard, and all-gathers the result. It has the same bytes as DP and
  $12\Psi(1-1/p)$ less memory, and global-norm clipping needs one extra scalar all-reduce.

- FSDP shards everything. Units are all-gathered just in time (cast to bf16 first), freed by
  resizing their storage, re-gathered for backward, and reduced with a reduce-scatter triggered by a
  post-accumulate hook. Prefetching hides the gathers when $T/n \ge C/W$.

- FSDP2 represents shards as DTensors over a device mesh; a 2-D mesh gives hybrid sharding.

- Accounting method: write matmuls, assign shards, cost ring collectives, and compare phase by
  phase. DP and FSDP need $T/n \gtrsim C/W$, a batch condition; TP needs
  $n_t - 1\lesssim P_g W/(2d_{\text{model}}C)$, a shape condition.

## Exercises {#sec:sharding:exercises}

::: exercise
[Bus bandwidth]{.box-title}

Show that for a ring all-reduce in the bandwidth-bound regime the bus bandwidth
$\frac{n}{t}\cdot\frac{2(p-1)}{p}$ equals the per-rank link bandwidth, independent of $p$. Why does
the same normalisation use $\frac{p-1}{p}$ for all-gather?
:::

::: exercise
[Bucket size]{.box-title}

Using the step-time model of [@sec:sharding:ddp], write the exposed communication time as a
function of the bucket size $b$ for a model whose gradients become ready at a uniform rate over
$T_b$. Find the $b$ that minimises it, and evaluate it for $G=1.2$ GB,
$T_b=50\,\mathrm{\milli\second}$, $p=8$, $\alpha=10\,\mathrm{\micro\second}$, 450 GB/s.
:::

::: exercise
[Clipping under sharding]{.box-title}

Extend `FlatShardedOptimizer` with a gradient-norm clipping method that is correct under sharding.
How many bytes does it communicate? Write a 3-rank test that compares it with PyTorch's
`clip_grad_norm_` in a single process.
:::

::: exercise
[Resharding a checkpoint]{.box-title}

The element-wise layout lets a checkpoint written on $p$ ranks be loaded on $p'$ ranks. Describe the
algorithm: what metadata must be saved, and which parts of each old shard does new rank $r'$ read?
Is the same true for the parameter-ownership layout of [@ch:distributed]?
:::

::: exercise
[Why storage, not tensors]{.box-title}

Replace `_fill_storage`'s storage-level copy with `self.flat.copy_(buf)` and run the [@lab:sharding] FSDP
test. Explain the error message precisely in terms of autograd's saved-tensor version counters, and
explain why the storage-level copy is safe here.
:::

::: exercise
[Unit granularity]{.box-title}

For a model with $L$ equal blocks of $P_b$ parameters, write peak transient memory and total
collective latency per step as functions of the number of blocks per unit, $k$. Which $k$ minimises
step time subject to transient memory $\le M$? Evaluate for Qwen3-1.7B on 16 GB GPUs.
:::

::: exercise
[Accounting: a GELU MLP]{.box-title}

Apply the four-step method to a GPT-2-style MLP, $y = \operatorname{GELU}(xW_1)W_2$ with
$W_1\in\mathbb{R}^{D\times4D}$ and $W_2\in\mathbb{R}^{4D\times D}$, for DP, FSDP (forward and
backward separately) and TP (column then row). Give the FLOPs, the communication time, and the
largest degree before each phase becomes communication-bound, in terms of $T, D, n, C, W$.
:::

::: exercise
[Accounting: 2-D]{.box-title}

Redo the previous exercise for a 2-D layout with FSDP degree $n_f$ and TP degree $n_t$. Which
collectives run over which mesh dimension? Verify the decoupling claim of
[@sec:sharding:accounting], and identify any term that couples the two degrees.
:::

::: exercise
[Hierarchical all-reduce]{.box-title}

Derive the time of the hierarchical all-reduce in [@fig:sharding:collectives] as a function
of $g$ (GPUs per node), $N$ (nodes), and the two $(\alpha,\beta)$ pairs. For what message sizes is
it faster than a flat ring over all $gN$ ranks with network links? Check your formula against
`llmsys.dist.costs`.
:::

::: exercise
[HSDP vs FSDP]{.box-title}

For a 70B model on 4 nodes of 8 GPUs (80 GB each, 50 GB/s per GPU network), compare FSDP over all 32
GPUs with HSDP sharded inside each node: per-GPU persistent memory, bytes crossing the network per
step, and whether each layout fits. What changes if the model is 13B?
:::

::: exercise
[Measurement]{.box-title}

Run [@lab:sharding]'s `bench.py` on 2--8 GPUs. Plot step time for per-parameter, flat and bucketed DDP,
element-wise ZeRO-1 and FSDP with and without prefetch, for a model whose per-GPU token count you
vary over $C/W$. Does the crossover appear where the accounting predicts?
:::

## Lab 19: Sharded data parallelism from scratch {#sec:sharding:lab}

::: labbox
[Sharded data parallelism from scratch]{.box-title}

**Code:** `code/labs/lab25_fsdp`

**Goals**

Build the data-parallel family from first principles and prove each member equivalent to
single-process training: a one-message flat all-reduce, an element-wise sharded optimizer around any
`torch.optim` class, and a complete FSDP with prefetching and mixed-precision communication. Then
turn the accounting method into code.

**Setup**

No GPU is needed: the tests start one 3-rank gloo process group with
`llmsys.dist.launch.run_distributed` and train a 3-block Qwen3-shaped model.

``` bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab25_fsdp     # your implementation
LAB_IMPL=solution python -m pytest -q labs/lab25_fsdp     # reference
python labs/lab25_fsdp/bench.py --world 3                  # CPU, gloo
torchrun --nproc-per-node 4 labs/lab25_fsdp/bench.py       # GPUs, NCCL
```

**Tasks**

1.  **Flat helpers**: `padded_numel`, `flatten_tensors`, `shard_range`, `unflatten_views`. The views
    must alias the flat buffer (the test writes through one).

2.  **`allreduce_flat`**: average all gradients with one all-reduce.

3.  **`FlatShardedOptimizer.step` and `state_numel`**: reduce-scatter, inner step, all-gather, write
    back. The test wraps `torch.optim.AdamW` and checks $2\lceil\Psi/p\rceil$ state elements per
    rank.

4.  **FSDP**: the storage methods of `_FlatUnit` (`_launch_gather`, `_fill_storage`, `unshard`,
    `reshard`, `attach_views`), `FullyShardedDP._make_unit`, `_on_grad` and
    `finish_gradient_synchronization`. The hooks are provided; read them first.

5.  **Accounting**: `layer_parallel_costs` for `"dp"`, `"fsdp"`, `"tp"` and `fsdp_persistent_bytes`;
    `max_degree` is provided.

**Measurements**

- `bench.py` times a step under per-parameter DDP, flat DDP, bucketed DDP, element-wise ZeRO-1, and
  FSDP with and without prefetch, and prints peak memory per rank (on GPU,
  `torch.cuda.max_memory_allocated`). Expect: flat beats per-parameter when tensors are many and
  small; ZeRO-1 matches DDP's time with less memory; FSDP uses the least memory and is slower
  without prefetch.

- Vary `--tokens-per-rank` across $C/W$ for your hardware and find the point where FSDP's step time
  departs from DDP's. Compare with [@sec:sharding:accounting].

- On a GPU node, capture an Nsight trace of FSDP with and without prefetch and identify the exposed
  all-gathers.

**Deliverables**

Passing tests; a table of step time and peak memory for the five variants; one plot of step time
against tokens per rank; a short paragraph reconciling the measurements with the accounting.

**Stretch goals**

Gradient accumulation without redundant reduce-scatters (a `no_sync` flag); separate units for
embedding and head; a sharded checkpoint save/load that reshards to a different world size; and
HSDP: shard over one subgroup and all-reduce shard gradients over another (`dist.new_group`).
:::
