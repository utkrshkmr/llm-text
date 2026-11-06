# Distributed Training Systems {#ch:distributed}

[@ch:pretrain] ended with two uncomfortable numbers. A 7B-parameter model needs about 112 
just for its weights, gradients and Adam moments---more than any single GPU holds---and a frontier
pretraining run needs $10^{24}$--$10^{26}$ FLOPs, which at a realistic $4\times10^{14}$ FLOP/s per
GPU is thousands of GPU-years. Both numbers force the same conclusion: training must be spread
across many devices, and the way it is spread decides whether the cluster computes or waits.

Training at this scale needs more than one device, and this chapter is about how. It is written for
someone who already knows what an all-reduce is and wants the *cost model*: how each parallelism
strategy partitions memory, how many bytes it moves per step and over which links, how those
transfers overlap with compute, and how the strategies compose into the 3D/4D layouts that real LLM
training uses. Every algorithm is implemented from scratch in `llmsys.dist` and tested in Lab 17 on
CPU processes with the gloo backend, which exposes the same collective API as NCCL.

![Roadmap. We first build the communication vocabulary (1), then partition the *batch* (2--3), the
*weights inside a layer* (4), the *layers* (5), the *sequence* (6) and the *experts* (7), and
finally compose them (8) and keep the resulting job alive
(9).](figures/ch22_distributed-1.svg){#fig:dist:roadmap}

## The problem in numbers {#sec:dist:numbers}

Consider training a 70B dense model on 15T tokens, roughly the Llama 3 70B recipe. Training compute
is $6ND\approx6.3\times10^{24}$ FLOPs ([@eq:pretrain:6nd]). On H100s at 40% MFU
($\approx396\,\mathrm{TFLOP/s}$ each) that is $1.6\times10^{10}$ GPU-seconds, about 500 GPU-years: a
1-year run on 500 GPUs or a 2-month run on 3,000. Memory is just as binding: model states alone are
$16\times70\text{B}=1.12\,\mathrm{TB}$, fourteen 80 GB GPUs before counting activations.

There are only a few things one can cut into pieces ([@fig:dist:axes]):

- the **batch**: every device holds the full model and processes different examples (data
  parallelism, DP); ZeRO/FSDP additionally shard the model states over the DP group;

- the **weight matrices within a layer**: each device holds a slice of every matrix and computes a
  slice of every activation (tensor parallelism, TP);

- the **stack of layers**: each device holds a contiguous group of layers and activations flow
  between devices (pipeline parallelism, PP);

- the **sequence**: each device holds part of every sequence; attention needs the rest
  (sequence/context parallelism, SP/CP);

- the **experts** of a mixture-of-experts layer (expert parallelism, EP).

![What each parallelism cuts. Blue blocks are layers (a tall block is the full stack), green boxes
are the inputs each device sees. Expert parallelism (not shown) places different experts of an MoE
layer on different devices and routes tokens to them with
all-to-all.](figures/ch22_distributed-2.svg){#fig:dist:axes}

They differ in what they save, what they cost in communication, and which network links that
communication can tolerate. To compare them we need a communication cost model.

## Communication fundamentals {#sec:dist:comm}

### Collectives

All distributed training is built from a handful of collective operations over a group of $p$ ranks
([@fig:dist:collectives]). Each rank starts with a buffer of $n$ bytes.

Broadcast

: one rank's buffer is copied to all others.

Reduce

: the element-wise sum of all buffers ends up on one rank.

All-reduce

: every rank ends up with the element-wise sum. (Gradient averaging.)

Reduce-scatter

: the sum is computed, but rank $i$ only receives the $i$-th $1/p$ slice of it.

All-gather

: rank $i$ contributes slice $i$; every rank ends up with the concatenation.

All-to-all

: rank $i$ sends a distinct slice to every rank $j$ (a transpose of the data across ranks). Used by
  MoE dispatch and Ulysses sequence parallelism.

The identity *all-reduce = reduce-scatter + all-gather* underlies both the ring algorithm and ZeRO.

![Collective semantics for three ranks (rows). $r_k$ is chunk $k$ of rank $r$'s buffer,
$\Sigma_k=\sum_r r_k$. Reduce-scatter leaves rank $r$ with only $\Sigma_r$; all-to-all transposes
chunk ownership.](figures/ch22_distributed-3.svg){#fig:dist:collectives}

### The $\alpha$--$\beta$ cost model

A point-to-point message of $n$ bytes costs $$\begin{equation}
t(n) = \alpha + n\,\beta,
\end{equation}$$ where $\alpha$ is a fixed per-message latency (software, NIC, switch; a few to tens
of microseconds) and $\beta=1/B$ is the inverse bandwidth of the link. Small messages are
latency-bound, large ones bandwidth-bound; the crossover is at $n^\star=\alpha/\beta$,
e.g. $10\,\mathrm{\micro s}\times50\,\mathrm{GB/s}=500\,\mathrm{KB}$ on an InfiniBand NDR link.
Gradient buckets of tens of MB are therefore firmly bandwidth-bound, which is why DDP bucketing
([@sec:dist:dp]) is about *overlap*, not latency.

::: proposition
[Bandwidth lower bound for all-reduce]{.box-title}

In an all-reduce over $p$ ranks, each rank must send at least $\frac{p-1}{p}n$ bytes and receive at
least $\frac{p-1}{p}n$ bytes for the reduce-scatter half, and the same again for the all-gather
half---$2\frac{p-1}{p}n$ in total per rank.
:::

*Sketch.* Each output element depends on $p$ inputs held on $p$ different ranks; for rank $i$ to own
its reduced $1/p$ slice, the other $p-1$ contributions to that slice ($\frac{p-1}{p}n$ bytes in
total, after any partial pre-reduction) must reach it. Symmetric arguments apply to distributing the
result.

### Ring all-reduce {#sec:dist:ring}

The ring algorithm meets this bound. Arrange the ranks in a ring and split the buffer into $p$
chunks. In the *reduce-scatter* phase, at step $s=0,\dots,p-2$ rank $r$ sends chunk $(r-s)\bmod p$
to its right neighbor and receives chunk $(r-s-1)\bmod p$ from its left neighbor, adding it to its
own copy. After $p-1$ steps, rank $r$ holds the fully reduced chunk $(r+1)\bmod p$. The *all-gather*
phase circulates the reduced chunks in the same pattern without adding ([@fig:dist:ring]).
Each of the $2(p-1)$ steps moves $n/p$ bytes on every link simultaneously: $$\begin{equation}
t_{\text{ring}} = 2(p-1)\,\alpha + 2\,\frac{p-1}{p}\,n\,\beta.
\label{eq:dist:ring}
\end{equation}$$ The bandwidth term is independent of $p$ for large $p$---the property that made
ring all-reduce (Patarasuk & Yuan, 2009; popularized for deep learning by Baidu and Horovod) the
default for data parallelism. Its weakness is the latency term, linear in $p$; for small messages or
very large rings, tree or recursive halving/doubling algorithms with $O(\log p)$ latency win. NCCL
chooses between ring, tree and (on NVSwitch systems) in-switch reduction (NVLS) based on message
size and topology.

![Ring reduce-scatter on four ranks (rows) and four chunks (columns). Entries list which ranks'
contributions a chunk already contains; $\Sigma$ means fully reduced. After $p-1=3$ steps rank $r$
owns the reduced chunk $(r+1)\bmod 4$. Three more steps of all-gather copy each $\Sigma$ around the
ring.](figures/ch22_distributed-4.svg){#fig:dist:ring}

``` {#lst:dist:ring .python caption="Ring all-reduce from point-to-point messages (`llmsys/dist/collectives.py`, [@lab:distributed] Task 1)."}
def _sendrecv(send_buf, recv_buf, dst, src):
    reqs = [dist.isend(send_buf.contiguous(), dst), dist.irecv(recv_buf, src)]
    for r in reqs:                       # post both before waiting: no ring deadlock
        r.wait()

def ring_allreduce(t):
    rank, p = dist.get_rank(), dist.get_world_size()
    right, left = (rank + 1) % p, (rank - 1) % p
    chunks = list(torch.tensor_split(t.view(-1), p))     # views into t
    for s in range(p - 1):                               # reduce-scatter
        send_i, recv_i = (rank - s) % p, (rank - s - 1) % p
        buf = torch.empty_like(chunks[recv_i])
        _sendrecv(chunks[send_i], buf, right, left)
        chunks[recv_i] += buf
    owned = (rank + 1) % p
    for s in range(p - 1):                               # all-gather
        send_i, recv_i = (owned - s) % p, (owned - s - 1) % p
        buf = torch.empty_like(chunks[recv_i])
        _sendrecv(chunks[send_i], buf, right, left)
        chunks[recv_i].copy_(buf)
    return t
```

![$\alpha$--$\beta$ all-reduce cost for $p=64$, $\alpha=10\,\mathrm{\micro s}$,
$B=50\,\mathrm{GB/s}$ (analytic, `llmsys.dist.costs.allreduce_time`). Tree algorithms win for small
messages (latency), ring and halving/doubling for large ones (bandwidth); the naive root-based
scheme is $\approx p/2$ times slower for large
messages.](figures/ch22_distributed-5.svg){#fig:dist:arcost}

### The hardware the bytes travel on

Inside an HGX H100 node, eight GPUs are fully connected through NVSwitch with NVLink 4: about 450 
per GPU in each direction. Between nodes, each GPU typically has its own 400  InfiniBand or RoCE
NIC, i.e. 50  per direction---an order of magnitude less. PCIe Gen5 x16 sits at about 64  per
direction. The large intra-/inter-node bandwidth gap is the single most important fact for
parallelism layout: communication-heavy strategies (TP, SP) stay inside the NVLink domain;
communication-light ones (DP, PP) span nodes. NCCL's reported *bus bandwidth* for all-reduce
rescales the measured algorithm bandwidth by $2(p-1)/p$ ([@eq:dist:ring]) so that it can be
compared directly with link speed.

::: deepdive
[Inside NCCL]{.box-title}

NCCL splits a collective across several *channels* (each a ring or tree mapped onto a subset of
links and driven by its own CUDA thread block) and pipelines each channel's chunk in small slices so
that sending, receiving and reducing overlap. It picks among three wire *protocols*: *LL* (8-byte
stores with an embedded flag: lowest latency, about half the bandwidth), *LL128* (128-byte lines
with flags, near-full bandwidth on NVLink) and *Simple* (large chunks with explicit fences: full
bandwidth, higher latency). On NVSwitch systems, *NVLS* performs the reduction inside the switch,
and InfiniBand *SHARP* does the same in the network fabric, cutting the data each GPU must send for
an all-reduce roughly in half. Because NCCL kernels occupy SMs, heavy communication also steals a
few percent of compute from overlapping GEMMs---one reason comm/compute overlap is never perfectly
free.
:::

::: napkinmath
[Gradient all-reduce for a 7B model]{.box-title}

bf16 gradients of 7B parameters are 14 . Ring all-reduce among the 8 GPUs of one node:
$2\cdot\frac78\cdot14\,\mathrm{GB}/450\,\mathrm{GB/s}\approx54\,\mathrm{ms}$. Across 64 GPUs over IB
at 50  per GPU: $\approx0.55\,\mathrm{s}$. Compare with the compute of one step: at 32k tokens per
GPU and 40% MFU, $6\cdot7\text{B}\cdot32{,}768/396\,\mathrm{TFLOP/s}\approx3.5\,\mathrm{s}$. Even
across nodes, the all-reduce can be hidden behind the backward pass---if it is overlapped.
:::

## Data parallelism {#sec:dist:dp}

Data parallelism replicates the model on $p$ ranks, gives each rank a different $1/p$ of the global
batch, and averages gradients before the optimizer step. If every rank starts from the same weights
and applies the same averaged gradient with the same optimizer, the replicas stay identical forever,
and the result is mathematically the same as single-device training on the whole batch (for
mean-reduced losses over equal-sized shards). Lab 17 checks this equivalence to $10^{-6}$.

The simplest implementation all-reduces each `.grad` after `backward()`. It has two problems. First,
a model has hundreds of parameter tensors, many tiny (biases, norms), so the step pays the latency
$\alpha$ hundreds of times. Second, communication starts only after the *whole* backward pass, so
the network idles during compute and the GPUs idle during communication.

##### Bucketing and overlap.

Gradients become available in roughly reverse order of the forward pass: the last layer's gradients
are ready first. PyTorch DDP (Li et al., 2020) groups parameters, in reverse order, into buckets of
about 25 , registers a hook on every parameter, and launches an asynchronous all-reduce for a bucket
as soon as all its gradients exist. While NCCL reduces bucket $k$ on a separate CUDA stream, the
backward pass keeps computing the gradients of bucket $k+1$ ([@fig:dist:overlap]). Before
the optimizer step, the training loop waits for all outstanding reductions.

![Bucketing overlaps gradient communication with the remaining backward computation. Only the
reduction of the last bucket (the first layers) stays exposed. Buckets are formed in reverse layer
order because that is the order in which gradients
appear.](figures/ch22_distributed-6.svg){#fig:dist:overlap}

``` {#lst:dist:ddp .python caption="Bucketed DDP from scratch (`llmsys/dist/ddp.py`, abridged)."}
class BucketedDDP(nn.Module):
    def __init__(self, module, bucket_cap_mb=25.0):
        super().__init__()
        self.module, self.world = module, dist.get_world_size()
        broadcast_parameters(module)                    # identical start everywhere
        self.buckets, cur, size = [], [], 0
        for p in reversed([p for p in module.parameters() if p.requires_grad]):
            cur.append(p); size += p.numel() * p.element_size()
            if size >= bucket_cap_mb * 2**20:
                self.buckets.append(_Bucket(cur)); cur, size = [], 0
        if cur: self.buckets.append(_Bucket(cur))
        for b in self.buckets:
            for p in b.params:
                self._bucket_of[p] = b
                p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _on_grad_ready(self, p):                         # runs inside backward()
        b = self._bucket_of[p]; b.ready += 1
        if b.ready == len(b.params):                     # bucket complete
            b.buffer = torch.cat([q.grad.reshape(-1) for q in b.params])
            b.work = dist.all_reduce(b.buffer, async_op=True)

    def finish_gradient_synchronization(self):
        for b in self.buckets:
            b.work.wait(); b.buffer /= self.world
            # ... copy slices of b.buffer back into each q.grad
```

::: systemslens
[What production DDP adds]{.box-title}

- **Gradient-as-bucket-view:** `.grad` tensors are views into the flat bucket buffer, so no
  `torch.cat`/copy-back is needed (our version copies twice).

- **`no_sync()`** for gradient accumulation: skip the all-reduce for all but the last micro-batch,
  otherwise accumulation multiplies communication by $k$.

- **Unused parameters** (e.g. inactive MoE experts) never fire their hook; DDP must detect them
  (`find_unused_parameters`) or the bucket never completes---exactly the error our implementation
  raises.

- **Bucket order** is re-learned after the first iteration from the actual order in which hooks
  fired.

- **Compression hooks** (fp16/bf16 casting, PowerSGD) reduce bytes at some accuracy risk.
:::

##### How far can DP scale?

DP alone scales the batch: with $p$ replicas and a per-GPU micro-batch of $b$ sequences, the global
batch is $pb$ sequences. Beyond a *critical batch size* (McCandlish et al., 2018), larger batches
stop reducing the number of steps needed, so extra replicas waste compute. LLM pretraining typically
uses global batches of 1--16M tokens. And DP does nothing for memory: every replica still holds all
16 bytes per parameter. The next section removes that redundancy.

### A step-time model for data parallelism {#sec:dist:dpmodel}

Let $t_c$ be the compute time of one step on one GPU and $t_m$ the time of the gradient all-reduce.
Without overlap, the step takes $t_c+t_m$; with perfect overlap, $\max(t_c,t_m)$ (in practice the
last bucket is always exposed). The *scaling efficiency* relative to a single GPU is therefore
between $$\begin{equation}
\eta_{\text{no overlap}}=\frac{t_c}{t_c+t_m}\qquad\text{and}\qquad
\eta_{\text{overlap}}=\frac{t_c}{\max(t_c,t_m)}.
\end{equation}$$ For ring all-reduce $t_m\approx 2\Psi/B$ is almost independent of $p$, while $t_c$
is proportional to the *local* batch. Weak scaling (fixed local batch, growing global batch)
therefore scales almost perfectly; strong scaling (fixed global batch split over more GPUs) shrinks
$t_c$ until communication dominates. [@fig:dist:dpeff] plots both bounds for a 7B model as
the local batch shrinks.

![Analytic DP efficiency for a 7B model (bf16 gradients, 14 GB; ring all-reduce with large $p$;
compute at 40% MFU of an H100, so $t_c\approx0.106\,\mathrm{ms}$ per token). With overlap,
efficiency stays at 1 until the local batch drops below $t_m/t_c$ tokens (about 5k tokens at 50 );
without overlap it degrades smoothly. A slower network (e.g. 100 Gb Ethernet, 12.5 ) moves the knee
$4\times$ to the right.](figures/ch22_distributed-7.svg){#fig:dist:dpeff}

##### Feeding data to many ranks.

Every DP rank must see a *different* slice of each global batch, and the union must be the same as
in single-device training. PyTorch's `DistributedSampler` does this by permuting indices with a
shared seed per epoch (`sampler.set_epoch(e)`) and giving rank $r$ every $p$-th index starting at
$r$. At pretraining scale, data is pre-tokenized into many binary shards; each rank streams its own
subset of shards through a memory-mapped reader (`llmsys.data.TokenMemmapDataset`) and records its
position so that a resumed job continues exactly where it stopped. Forgetting `set_epoch` silently
repeats the same order every epoch; giving two ranks the same seed for shard selection silently
duplicates data.

##### Global batch bookkeeping.

With micro-batch $b$ sequences of length $s$, gradient accumulation $k$ and DP degree $d$, one
optimizer step consumes $b\cdot k\cdot d\cdot s$ tokens. Changing the number of GPUs while keeping
the optimization identical means adjusting $k$ so the product stays fixed; the learning rate
schedule is defined over optimizer steps, not GPU-hours.

## ZeRO and fully sharded data parallelism {#sec:dist:zero}

In DP every rank stores identical copies of weights, gradients and optimizer moments. ZeRO
(Rajbhandari et al., 2020) observes that each rank only *needs* the full set of some of these, some
of the time, and shards the rest over the $N_d$ data-parallel ranks in three stages. With $\Psi$
parameters, 2-byte weights and gradients and $K=12$ bytes of fp32 optimizer state per parameter:
$$\begin{align}
\text{DP:}\quad & (2+2+K)\,\Psi \\
\text{ZeRO-1 (optimizer states):}\quad & (2+2+K/N_d)\,\Psi \\
\text{ZeRO-2 (+ gradients):}\quad & (2+(2+K)/N_d)\,\Psi \\
\text{ZeRO-3 / FSDP (+ parameters):}\quad & (2+2+K)\,\Psi/N_d
\end{align}$$

::: {#tab:dist:zero .table-float}
| stage         |  bytes/param | 7.5B on 64 GPUs | communication per step                            |
|:--------------|-------------:|----------------:|:--------------------------------------------------|
| DP (stage 0)  |           16 |        120.0 GB | all-reduce grads: $2\Psi$                         |
| ZeRO-1        | $4 + 12/N_d$ |         31.4 GB | reduce-scatter grads + all-gather params: $2\Psi$ |
| ZeRO-2        | $2 + 14/N_d$ |         16.6 GB | same: $2\Psi$                                     |
| ZeRO-3 / FSDP |     $16/N_d$ |          1.9 GB | all-gather $\times2$ + reduce-scatter: $3\Psi$    |

: Model-state memory per GPU (computed with `llmsys.dist.costs.zero_memory_per_gpu`) and
communication volume per rank (in units of the gradient size $\Psi$ bytes) for the ZeRO stages.
:::

![ZeRO stages on four GPUs. Bar heights are proportional to bytes per parameter (2, 2 and 12). Each
stage shards one more kind of state so that every GPU keeps only its $1/N_d$ slice; ZeRO-3 shards
everything.](figures/ch22_distributed-8.svg){#fig:dist:zero}

##### ZeRO-1 and ZeRO-2.

Each rank *owns* a $1/N_d$ shard of the parameters and keeps fp32 master weights and Adam moments
only for that shard. After backward, gradients are *reduce-scattered* (each rank receives the summed
gradient of its shard), the owner runs Adam on its shard, and the updated parameters are
*all-gathered*. Since reduce-scatter + all-gather moves exactly the bytes of one all-reduce, stages
1 and 2 cost the same communication as plain DP while cutting memory by up to $4\times$ and
$8\times$. Stage 2 also frees the non-owned gradient slices immediately after their reduction.

Our teaching implementation ([@lst:dist:zero]) assigns whole parameter tensors to owners
and uses reduce/broadcast, because the gloo backend lacked reduce-scatter in older PyTorch versions;
production versions flatten parameters into one buffer and shard it element-wise for perfect
balance.

``` {#lst:dist:zero .python caption="ZeRO-1/2 with parameter-level ownership (`llmsys/dist/zero.py`, abridged)."}
class ZeroAdamW:
    def __init__(self, params, lr, stage=1, **kw):
        self.params = [p for p in params if p.requires_grad]
        self.owner = partition_parameters(self.params, dist.get_world_size())
        mine = [p for p, o in zip(self.params, self.owner) if o == dist.get_rank()]
        self.local = torch.optim.AdamW(mine, lr=lr, **kw)   # states only for my shard

    @torch.no_grad()
    def step(self):
        for p, o in zip(self.params, self.owner):
            if self.stage == 1:
                dist.all_reduce(p.grad); p.grad /= self.world
            else:                                  # stage 2: only the owner keeps it
                dist.reduce(p.grad, dst=o)
                if dist.get_rank() == o: p.grad /= self.world
                else: p.grad = None
        self.local.step()                          # Adam on owned parameters only
        for p, o in zip(self.params, self.owner):
            dist.broadcast(p.data, src=o)          # everyone gets updated weights
```

##### ZeRO-3 and FSDP.

Stage 3 also shards the bf16 parameters. The model is wrapped in *units* (typically one transformer
block each). Just before a unit's forward, its parameters are all-gathered from all ranks; after the
forward they are freed. The backward pass all-gathers them again, computes gradients, and
reduce-scatters the gradients so each rank keeps only its shard. Communication is $1.5\times$ that
of DP ($3\Psi$ vs. $2\Psi$ bytes), but only one or two units are ever materialized, so per-GPU
memory falls to $16\Psi/N_d$ plus a unit or two of full parameters. PyTorch's FSDP (Zhao et al.,
2023) implements this design; FSDP2 shards each parameter individually as a `DTensor`, which
composes cleanly with tensor parallelism.

::: systemslens
[Making FSDP fast]{.box-title}

- **Prefetching.** Issue the all-gather for unit $i+1$ while unit $i$ computes (forward), and for
  unit $i-1$ during unit $i$'s backward. With enough compute per unit, the all-gathers are
  completely hidden.

- **Hybrid sharding (HSDP).** Shard within a node (fast NVLink) and replicate across nodes,
  combining FSDP's memory savings with DP's lighter cross-node traffic (a plain all-reduce of
  gradient shards across replicas).

- **Mixed-precision communication.** All-gather bf16 parameters and reduce-scatter bf16 (or fp32 for
  stability) gradients; keep fp32 master shards locally.

- **Offload.** ZeRO-Offload/Infinity move optimizer states and even parameters to CPU memory or
  NVMe, trading PCIe bandwidth for capacity---useful for fine-tuning big models on few GPUs, rarely
  for pretraining throughput.
:::

![FSDP with prefetching for a 4-unit model. The all-gather (AG) of unit $i+1$ runs while unit $i$
computes forward (F); in backward (B) units are re-gathered in reverse order and each unit's
gradients are reduce-scattered (RS) as soon as they exist. Only one or two units' full parameters
are ever resident.](figures/ch22_distributed-9.svg){#fig:dist:fsdp}

## Tensor parallelism {#sec:dist:tp}

ZeRO shards *storage*, but every layer still runs on one device with full-size activations. Tensor
parallelism (Megatron-LM; Shoeybi et al., 2019) splits the *computation* of each layer. Take a
linear layer $\mathbf{Y}=\mathbf{X}\mathbf{W}^\top$ with
$\mathbf{W}\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ and $t$ ranks.

Column-parallel

: Split $\mathbf{W}$ by output rows, $\mathbf{W}=[\mathbf{W}_1;\dots;\mathbf{W}_t]$. Rank $r$
  computes $\mathbf{Y}_r=\mathbf{X}\mathbf{W}_r^\top$, a slice of the output features. Input
  $\mathbf{X}$ must be replicated. In backward, each rank's $\partial\mathcal L/\partial\mathbf{X}$
  is only the contribution of its slice, so the input gradient needs an **all-reduce**.

Row-parallel

: Split $\mathbf{W}$ by input columns. Rank $r$ consumes the matching slice $\mathbf{X}_r$ of the
  input and produces a partial result $\mathbf{X}_r\mathbf{W}_r^\top$; the true output is the sum
  over ranks, an **all-reduce** in the forward pass. Its backward needs no communication: the output
  gradient is already replicated.

The trick is to chain them. In the MLP
$\mathbf{Y}=\operatorname{GELU}(\mathbf{X}\mathbf{A}^\top)\mathbf{B}^\top$, make $\mathbf{A}$
column-parallel and $\mathbf{B}$ row-parallel. Each rank then holds a slice of the hidden features,
applies GELU locally (element-wise, so no communication), and feeds that slice straight into its
row-parallel slice of $\mathbf{B}$. One all-reduce at the end of the forward pass and one at the
start of the backward pass suffice ([@fig:dist:tp]). Megatron calls the two conjugate
operators $f$ (identity forward, all-reduce backward) and $g$ (all-reduce forward, identity
backward). Self-attention splits the same way: heads are distributed column-parallel through the
Q/K/V projections (each rank runs complete heads, so softmax stays local) and the output projection
is row-parallel. A transformer layer thus needs **two all-reduces in forward and two in backward**.

![Megatron-style tensor-parallel MLP on two ranks. The hidden activation $\mathbf{H}$ never leaves
its rank; only the $f$ and $g$ operators
communicate.](figures/ch22_distributed-10.svg){#fig:dist:tp}

``` {#lst:dist:tp .python caption="The $f$/$g$ operators and parallel linear layers (`llmsys/dist/tensor_parallel.py`, abridged)."}
class CopyToTP(torch.autograd.Function):          # f
    @staticmethod
    def forward(ctx, x): return x
    @staticmethod
    def backward(ctx, g):
        g = g.clone(); dist.all_reduce(g); return g

class ReduceFromTP(torch.autograd.Function):      # g
    @staticmethod
    def forward(ctx, x):
        x = x.clone(); dist.all_reduce(x); return x
    @staticmethod
    def backward(ctx, g): return g

class ColumnParallelLinear(nn.Module):            # holds out_features / t rows
    def forward(self, x):
        return F.linear(CopyToTP.apply(x), self.weight, self.bias)

class RowParallelLinear(nn.Module):               # holds in_features / t columns
    def forward(self, x_local):
        y = ReduceFromTP.apply(F.linear(x_local, self.weight))
        return y + self.bias                      # bias added once, after the sum
```

::: napkinmath
[Why TP stays inside a node]{.box-title}

A 70B layer ($h=8192$) with $s=4096$, $b=1$ and $t=8$: each all-reduce carries the $s\cdot b\cdot h$
bf16 activation, 67 ; four per layer per step at $2\cdot\frac78\cdot67\,\mathrm{MB}$ each
$\approx0.47\,\mathrm{GB}$ per rank. At NVLink's 450 : 1.0  per layer, versus
$\approx5.4\,\mathrm{ms}$ of compute per layer per GPU at 50% MFU. Over 50  IB the same traffic
takes 9.4 ---longer than the compute. TP is therefore almost always confined to the 8 GPUs of an
NVLink domain (or the 72 of an NVL72 rack).
:::

##### Hiding TP communication.

The all-reduces in [@fig:dist:tp] sit on the critical path: the next layer cannot start
until $g$ completes. Two families of techniques overlap them anyway. *Decomposition* (Wang et al.,
2023; "async TP" in PyTorch, Flux, and NVIDIA's userbuffers) splits the GEMM adjacent to a
collective into chunks along the sequence dimension, so that chunk $i$'s reduce-scatter or
all-gather proceeds while chunk $i+1$ is multiplied. *Micro-batch interleaving* runs two independent
micro-batches through the layer out of phase, so one computes while the other communicates (the
approach behind DeepSeek-V3's dual micro-batch overlap). Both turn the $1.0\,\mathrm{ms}$ of the
napkin math above into almost-free communication when the chunked GEMMs remain large enough to run
efficiently.

::: keyidea
[TP degree is a trade-off, not a free lunch]{.box-title}

Each doubling of $t$ halves every GEMM's size along one dimension. A $4096\times8192$ GEMM split 16
ways becomes $4096\times512$, which may no longer reach peak tensor-core efficiency, while the
activation all-reduce stays the same size. Choose the smallest $t$ that makes memory fit.
:::

##### Sequence parallelism.

TP leaves LayerNorm, dropout and the residual stream replicated: every rank stores the full
$s\times b\times h$ activations there. Korthikanti et al. (2022) split these regions along the
*sequence* dimension instead and replace each all-reduce with a reduce-scatter (entering the
sequence-parallel region) plus an all-gather (leaving it). Communication volume is unchanged, since
all-reduce = reduce-scatter + all-gather, while the replicated activations shrink by $t$.

##### Vocabulary parallelism.

The embedding table and output head ($V\times h$, up to 1.2B parameters for $V=152$k, $h=8192$) are
split along the vocabulary. The cross-entropy is then computed without gathering logits: each rank
computes its local max and local $\sum\exp$ over its vocabulary slice, two small all-reduces produce
the global normalizer, and the target logit comes from whichever rank owns the target ID.

## Pipeline parallelism {#sec:dist:pp}

Pipeline parallelism assigns contiguous groups of layers (stages) to different devices. Only the
activations at stage boundaries cross devices---$s\cdot b\cdot h$ values per micro-batch,
point-to-point---so PP tolerates slow inter-node links well. The catch is idleness: stage 2 cannot
start until stage 1 finishes. GPipe (Huang et al., 2019) splits each batch into $m$ micro-batches
and streams them through the stages: all forwards, then all backwards, then one synchronous
optimizer step. With $p$ stages, per-micro-batch forward time $t_f$ and backward time $t_b$, each
stage is idle for $(p-1)(t_f+t_b)$ out of $(m+p-1)(t_f+t_b)$: $$\begin{equation}
\text{bubble fraction} = \frac{p-1}{m+p-1}.
\label{eq:dist:bubble}
\end{equation}$$ To keep the bubble small one needs $m\gg p$ (e.g. $m\ge4p$ gives $<20\%$). But
GPipe holds the activations of *all* $m$ micro-batches at once before the first backward.

##### 1F1B.

PipeDream-Flush / Megatron's *one-forward-one-backward* schedule (Narayanan et al., 2021) has the
same bubble but starts each micro-batch's backward as early as possible: stage $s$ runs $p-s-1$
warm-up forwards, then alternates one forward and one backward, then drains. A stage never holds
more than $p-s$ micro-batches of activations, independent of $m$. [@fig:dist:pipe] shows
both schedules as produced by `llmsys.dist.pipeline.simulate`.

![Pipeline schedules simulated with `llmsys.dist.pipeline`. Both take 27 time units and idle for
$(p-1)/(m+p-1)=1/3$ of the time (white gaps), but 1F1B frees each micro-batch's activations as soon
as its backward completes.](figures/ch22_distributed-11.svg){#fig:dist:pipe}

##### Shrinking the bubble.

*Interleaved* 1F1B gives each device $v$ non-contiguous "virtual" stages (e.g. layers 1--4 and
17--20), shrinking the bubble by $v$ at the cost of $v$ times more point-to-point messages.
*Zero-bubble* schedules (Qi et al., 2024) split the backward pass into the input-gradient part $B$
(on the critical path) and the weight-gradient part $W$ (which can be deferred), and fill the
bubbles with $W$ work. DeepSeek-V3's *DualPipe* feeds micro-batches from both ends of the pipeline
and overlaps computation with the all-to-all communication of its MoE layers.

::: systemslens
[Practical pipeline concerns]{.box-title}

- **Balance.** The slowest stage sets the pace. The embedding layer and the output head with its
  large softmax make the first and last stages heavier; Megatron lets you assign fewer transformer
  layers to them.

- **Micro-batch count vs. global batch.** $m$ is bounded by global batch $/$ (DP degree $\times$
  micro-batch size). With a fixed global batch, more DP means fewer micro-batches per pipeline and a
  bigger bubble---a central tension of 3D parallelism.

- **Activation memory is uneven**: stage 0 holds $p$ micro-batches under 1F1B, the last stage only
  one.

- **Weight updates are synchronous** (flush at the end of each batch), which keeps the math
  identical to non-pipelined training; asynchronous PipeDream variants with weight stashing are
  rarely used for LLMs.
:::

## Sequence and context parallelism for long contexts {#sec:dist:cp}

At 128k-token contexts the activations of even one sequence overwhelm a GPU, and attention's
$O(s^2)$ compute dominates. *Context parallelism* (CP) splits the sequence of every layer---not just
the LayerNorm regions---across $c$ devices. The MLP and projections are per-token and need no
communication; attention needs every query to see every (earlier) key.

##### Ring attention

(Liu et al., 2023). Each device keeps its query chunk and passes its key/value chunk around a ring.
At each of $c$ steps it computes attention of its queries against the visiting K/V chunk and folds
the result into its output with the online-softmax rescaling of FlashAttention
([@ch:attention]): keep a running max $m$ and normalizer $\ell$ per query, and rescale the
partial output when a new block raises the max. Sending the next K/V chunk overlaps with computing
on the current one. With causal masking, naive contiguous chunking makes the last device do $c$
times more work than the first; *zig-zag* assignment (device $i$ gets chunks $i$ and $2c-1-i$)
balances it.

![Ring attention on four devices with contiguous chunks. Row $q$ is device $q$'s query block; at
step $t$ it holds K/V block $(q-t)\bmod 4$ (dark). White cells are causally masked: device 0 has
useful work only at step 0 while device 3 works at every step---the imbalance that zig-zag chunk
assignment removes.](figures/ch22_distributed-12.svg){#fig:dist:ringattn}

##### Ulysses

(Jacobs et al., 2023). Alternatively, an all-to-all before attention re-partitions from "all heads,
$1/c$ of the sequence" to "$1/c$ of the heads, the full sequence", runs ordinary (Flash)attention
locally, and a second all-to-all switches back. Communication per device is $O(s\,h/c)$ and
independent of the number of steps, but the degree is limited by the number of (KV) heads.

## Expert parallelism {#sec:dist:ep}

Mixture-of-experts layers ([@ch:modern]) replace the MLP with $E$ expert MLPs and a router
that sends each token to its top-$k$ experts. With expert parallelism, experts are spread over $e$
devices. Each MoE layer then needs two all-to-alls: *dispatch* (send each token's hidden state to
the devices hosting its chosen experts) and *combine* (send results back). Per token and layer,
about $2k\cdot h$ values cross the network in each direction. Systems issues include load imbalance
(a popular expert becomes a straggler; auxiliary balancing losses or bias-based balancing mitigate
it), token dropping when an expert exceeds its *capacity* ($\text{capacity factor}\times$
tokens$/E$), and topology-aware routing that limits each token to experts on a few nodes so that
most all-to-all traffic stays on NVLink (as in DeepSeek-V3).

::: napkinmath
[All-to-all volume of a large MoE]{.box-title}

Take a DeepSeek-V3-like layer ($h=7168$, top-$k=8$ routed experts) and 8,192 tokens per GPU.
Dispatch sends $8192\cdot8\cdot7168$ bf16 values $\approx0.94\,\mathrm{GB}$ per GPU per MoE layer,
and combine sends the same back---every layer, forward and backward. At 50  that is
$\approx19\,\mathrm{ms}$ per direction per layer, comparable to the expert compute itself. Hence the
emphasis on node-limited routing (most traffic on NVLink), FP8 dispatch, and schedules that overlap
one micro-batch's all-to-all with another's computation.
:::

## Putting it together: 3D and 4D parallelism {#sec:dist:3d}

Large runs compose these strategies on a *device mesh*. The ordering follows communication
intensity: TP (and SP) innermost on NVLink; CP next; PP across nodes; DP/FSDP outermost
([@fig:dist:mesh]). In PyTorch, `init_device_mesh("cuda", (dp, pp, tp))` creates the
sub-groups and FSDP2/TP/PP APIs each take one mesh dimension.

![A 32-GPU mesh with TP $=8$ (one node), PP $=2$ (light/dark nodes are stages 0 and 1) and DP $=2$.
Rank $=$ dp$\cdot16$ + pp$\cdot8$ + tp: TP neighbors are adjacent ranks on the same NVSwitch; PP
sends activations to the next node; DP all-reduces span the whole
cluster.](figures/ch22_distributed-13.svg){#fig:dist:mesh}

::: {#tab:dist:summary .table-float}
| strategy    | splits   | saves memory of      | communication                    | typical link |
|:------------|:---------|:---------------------|:---------------------------------|:-------------|
| DP          | batch    | ---                  | all-reduce grads, $2\Psi$/step   | IB           |
| ZeRO-1/2    | batch    | optimizer (+grads)   | same as DP                       | IB           |
| FSDP/ZeRO-3 | batch    | everything           | $3\Psi$/step, per-unit gathers   | IB / NVLink  |
| TP (+SP)    | matrices | weights, activations | 4 all-reduces of $sbh$ per layer | NVLink       |
| PP          | layers   | weights, activations | p2p $sbh$ per micro-batch        | IB           |
| CP          | sequence | activations          | K/V ring or all-to-all           | NVLink / IB  |
| EP          | experts  | expert weights       | 2 all-to-alls per MoE layer      | NVLink / IB  |

: Parallelism strategies at a glance.
:::

##### A planning recipe.

1.  Pick the smallest TP degree (within a node) that, together with PP and ZeRO, makes model states
    and activations fit; larger TP shrinks per-GPU matmuls and lowers efficiency.

2.  Add PP when the model does not fit within one NVLink domain; choose $p$ so that layers divide
    evenly and $m\ge4p$ micro-batches are available.

3.  Add CP only when sequence length requires it.

4.  Spend all remaining GPUs on DP (with ZeRO-1 or FSDP), and check that the global batch stays
    below the critical batch size.

5.  Verify with napkin math: memory per GPU, bubble fraction, exposed communication.

::: napkinmath
[A 70B run on 1,024 H100s]{.box-title}

Choose TP $=8$, PP $=4$, DP $=32$ with ZeRO-1, $s=4096$, global batch 4M tokens.

- Parameters per GPU: $70\text{B}/(8\cdot4)=2.19$B. Model states with ZeRO-1:
  $(2+2+12/32)\times2.19\text{B}\approx9.6\,\mathrm{GB}$.

- Activations: [@eq:pretrain:act] with FlashAttention gives 1.14  per layer per sequence;
  TP+SP divides by 8; 20 layers per stage; stage 0 holds up to $p=4$ micro-batches under 1F1B:
  $1.14/8\times20\times4\approx11.4\,\mathrm{GB}$. Total $\approx21\,\mathrm{GB}$---comfortable on
  80 GB, leaving room for larger micro-batches or less recomputation.

- Bubble: 1,024 sequences per step / DP 32 = 32 micro-batches per pipeline; $(4-1)/(32+3)=8.6\%$.
  With PP $=8$ it would be 17.9% (or 9.9% interleaved with $v=2$).

- DP gradient traffic: each rank reduces its 4.4  bf16 shard over 32 replicas:
  $\approx0.17\,\mathrm{s}$ at 50 , hidden behind $\approx4.1\,\mathrm{s}$ of compute per step at
  40% MFU.
:::

## From scratch to production APIs {#sec:dist:apis}

Everything above has a production counterpart in PyTorch. A training script is launched with one
process per GPU by `torchrun`, which sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE` and the rendezvous
address:

``` bash
torchrun --nnodes 4 --nproc-per-node 8 --rdzv-backend c10d \
         --rdzv-endpoint $HEAD_NODE:29500 train.py --tp 8 --dp 4
```

``` {#lst:dist:prod .python caption="Composing TP and FSDP2 on a 2-D device mesh (sketch; PyTorch 2.4+ APIs)."}
import os, torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor.parallel import (parallelize_module,
    ColwiseParallel, RowwiseParallel)

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (4, 8), mesh_dim_names=("dp", "tp"))

model = build_model()                               # e.g. a Qwen3Model on "meta"
for blk in model.trf_blocks:                        # Megatron-style TP inside a node
    parallelize_module(blk, mesh["tp"], {
        "att.W_query": ColwiseParallel(), "att.W_key": ColwiseParallel(),
        "att.W_value": ColwiseParallel(), "att.out_proj": RowwiseParallel(),
        "ff.fc1": ColwiseParallel(), "ff.fc2": ColwiseParallel(),
        "ff.fc3": RowwiseParallel()})
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for blk in model.trf_blocks:                        # FSDP units = blocks, over dp
    fully_shard(blk, mesh=mesh["dp"], mp_policy=mp)
fully_shard(model, mesh=mesh["dp"], mp_policy=mp)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
```

The mapping to this chapter is direct: `ColwiseParallel`/`RowwiseParallel` are our column/row layers
(the attention module must also be told that each rank now holds $H/t$ heads); `fully_shard` is
ZeRO-3 with per-block units, prefetching and reduce-scatter; the mesh encodes the rank layout of
[@fig:dist:mesh]. Pipeline parallelism is available through `torch.distributed.pipelining`
(schedules `ScheduleGPipe`, `Schedule1F1B`, `ScheduleInterleaved1F1B`), and
`torch.distributed.checkpoint` saves and reshards DTensor-based state. Megatron-LM/Megatron-Core,
DeepSpeed and TorchTitan are the reference systems that package these pieces for LLM pretraining.

::: pitfall
[Test the small configuration first]{.box-title}

Every parallel layout must reproduce the single-device loss curve. Before a large launch, run the
same seed on 1 GPU and on a tiny mesh (e.g. $\text{TP}=2$, $\text{DP}=2$) for a few hundred steps
and compare losses to several decimal places. This is exactly what Lab 17's tests do, in miniature.
:::

## Reliability at scale {#sec:dist:reliability}

At thousands of GPUs, failure is the steady state. Meta reported 466 job interruptions during a
54-day period of Llama 3 405B pretraining on 16k H100s, 419 of them unexpected, most attributed to
GPU or HBM faults and other hardware issues (Llama Team, 2024)---roughly one every three hours.
Training infrastructure therefore treats recovery as a first-class path:

- **Fast, frequent checkpoints** (asynchronous, sharded, [@sec:pretrain:save]) and fast
  restarts (pre-warmed spare nodes, cached container images, quick NCCL re-initialization).

- **Elastic launch.** `torchrun` restarts the worker group on failure; with resharding-capable
  checkpoints the job can resume on fewer or more nodes.

- **Hang and straggler detection.** A single slow GPU (thermal throttling, a flaky link) slows every
  synchronous collective. Collective timeouts, per-rank step-time telemetry and NCCL flight
  recorders identify the culprit.

- **Silent data corruption.** Rare hardware faults flip bits without crashing. Symptoms are loss
  spikes or NaNs on one replica; defenses include comparing gradient norms across DP replicas (they
  should match exactly after the all-reduce) and periodic known-answer tests.

::: pitfall
[Determinism across ranks]{.box-title}

DP correctness assumes every replica applies *bitwise identical* updates. Non-deterministic kernels
or different reduction orders across ranks make replicas drift apart silently. Broadcast parameters
at start, use the same seeds for anything that must agree (dropout masks are fine to differ;
parameter initialization is not), and periodically check a weight checksum across ranks.
:::

## Summary {#sec:dist:summary}

- Collectives are the vocabulary of distributed training; the $\alpha$--$\beta$ model with the ring
  identity all-reduce = reduce-scatter + all-gather explains most of their costs.

- DP scales throughput but not memory; bucketing and overlap hide gradient communication.

- ZeRO-1/2 shard optimizer states and gradients at no extra communication; ZeRO-3/FSDP shard
  everything for 1.5$\times$ the traffic.

- TP splits matrices with paired column/row layers (two all-reduces per pass per sub-block) and
  needs NVLink-class bandwidth; SP removes the remaining replicated activations.

- PP splits layers; its bubble is $(p-1)/(m+p-1)$; 1F1B bounds activation memory, interleaving and
  zero-bubble schedules shrink idle time.

- CP (ring attention, Ulysses) handles long sequences; EP distributes MoE experts with all-to-all.

- Real jobs compose these on a device mesh ordered by bandwidth needs and must be engineered for
  frequent failure.

## Exercises {#sec:dist:exercises}

::: exercise
[Ring all-reduce cost]{.box-title}

Derive [@eq:dist:ring] by counting the bytes each rank sends in each of the $2(p-1)$ steps.
Then show that the recursive halving/doubling algorithm achieves the same bandwidth term with only
$2\log_2p$ latency terms when $p$ is a power of two.
:::

::: exercise
[Two-level all-reduce]{.box-title}

A cluster has $N$ nodes of 8 GPUs, with intra-node bandwidth $B_1$ and per-GPU inter-node bandwidth
$B_2\ll B_1$. Design a hierarchical all-reduce (reduce-scatter in the node, all-reduce across nodes
on $1/8$ shards, all-gather in the node) and give its cost. Compare with a flat ring over all $8N$
GPUs for $B_1=450$, $B_2=50$ GB/s.
:::

::: exercise
[Bucket size]{.box-title}

With $\alpha=20\,\mathrm{\micro s}$ and 50 , what bucket size keeps the latency overhead below 5% of
the transfer time? Why might a very large bucket still be bad for overlap?
:::

::: exercise
[No-sync accumulation]{.box-title}

Modify `BucketedDDP` to support a `no_sync()` context manager. Measure, with the gloo backend, the
time for 4-step gradient accumulation with and without it.
:::

::: exercise
[ZeRO arithmetic]{.box-title}

Reproduce [@tab:dist:zero]. For a 13B model on 16 GPUs with 80 GB each, which is the lowest
ZeRO stage that leaves at least 30 GB per GPU for activations?
:::

::: exercise
[TP backward]{.box-title}

Show that for a column-parallel linear layer the gradient with respect to its input is
$\sum_r \frac{\partial\mathcal L}{\partial \mathbf{Y}_r}\mathbf{W}_r$, which justifies the
all-reduce in the backward of $f$. Why does the row-parallel layer need no communication in
backward?
:::

::: exercise
[Attention under TP]{.box-title}

Extend `TensorParallelMLP` to a `TensorParallelAttention` module (heads split across ranks,
column-parallel QKV, row-parallel output). Verify against `MultiHeadAttention` from `llmsys.layers`
with gloo on two processes. What constraint on the number of heads arises? How does GQA
([@ch:modern]) complicate it when $t$ exceeds the number of KV heads?
:::

::: exercise
[Bubbles]{.box-title}

Using `llmsys.dist.pipeline`, plot bubble fraction and peak in-flight micro-batches for GPipe and
1F1B with $p=8$ and $m\in\{8,16,32,64\}$. Then implement an interleaved schedule with $v=2$ and
confirm that the bubble roughly halves.
:::

::: exercise
[Causal load balance in ring attention]{.box-title}

With $c$ devices and contiguous chunks, count the number of (query block, key block) pairs each
device computes under a causal mask. Show that the zig-zag assignment equalizes the work.
:::

::: exercise
[Plan a run]{.box-title}

Plan the parallel layout for a 405B dense model ($L=126$, $h=16{,}384$, $s=8192$) on 16,384 H100s
with a 16M-token global batch. Report TP, PP, CP, DP, per-GPU memory, bubble and the exposed DP
communication, stating every assumption.
:::

## Lab 18: Distributed training from scratch {#sec:dist:lab}

::: labbox
[Distributed training from scratch]{.box-title}

**Code:** `code/labs/lab22_distributed`

**Goals**

Implement the core distributed-training primitives yourself, verify that each is *mathematically
equivalent* to single-process training, and connect measured behavior to the cost models of this
chapter.

**Setup**

No GPUs needed. `llmsys.dist.launch.run_distributed(fn, world_size)` spawns CPU processes with the
gloo backend; every collective you use also exists in NCCL. Run the tests with
`LAB_IMPL=starter python -m pytest -q labs/lab22_distributed` (one process group of 3 ranks runs all
distributed checks; a non-power-of-two world catches indexing bugs).

**Tasks**

1.  `naive_allreduce`, `ring_reduce_scatter`, `ring_all_gather`, `ring_allreduce` using only
    `isend`/`irecv`; must handle any tensor shape and sizes not divisible by $p$.

2.  `allreduce_time` ($\alpha$--$\beta$, ring and naive), `zero_memory_per_gpu` (stages 0--3),
    `pipeline_bubble_fraction`.

3.  `BucketedDDP`: reverse-order buckets, post-accumulate-grad hooks, async all-reduce,
    `finish_gradient_synchronization`. Weights after 3 steps must equal full-batch single-process
    AdamW, even when ranks start from different initializations.

4.  `ZeroAdamW` stage 1 and 2 with balanced `partition_parameters`: same equivalence; each rank
    stores Adam state only for the parameters it owns.

5.  `CopyToTP`, `ReduceFromTP`, `ColumnParallelLinear`, `RowParallelLinear`: the tensor-parallel MLP
    must match the full MLP in outputs, input gradients and the corresponding weight-gradient
    slices.

6.  `gpipe_order` and `one_f_one_b_order`: the provided simulator must report bubble $(p-1)/(m+p-1)$
    and peak in-flight $m$ (GPipe) and $p-s$ (1F1B).

**Measurements**

- `python bench.py allreduce --world 4`: time naive, ring and library all-reduce for 1 KB--16 MB and
  fit $\alpha$, $\beta$ for each. Expect the library to win at small sizes (better latency) and your
  ring to beat the naive scheme at large sizes. On a multi-GPU node switch to `backend="nccl"` and
  compare the fitted bandwidth with NVLink specs. (Timings on a shared or oversubscribed CPU are
  noisy; take the minimum of several repetitions.)

- `python bench.py ddp --world 2`: per-parameter all-reduce vs. bucketed DDP.

- `python bench.py pipeline --stages 8 --mb 32`: compare with [@eq:dist:bubble].

**Deliverables**

Passing tests; a plot of the fitted $\alpha$--$\beta$ curves against your measurements; a short
table comparing DDP variants; and a written 3D-parallel plan for a 70B model on 256 GPUs with the
napkin math of [@sec:dist:3d].

**Stretch goals**

Implement ZeRO-3 for an `nn.Sequential` (gather each layer's parameters in a forward pre-hook, free
them afterwards, re-gather in backward); a hierarchical all-reduce; a real two-stage pipeline with
`dist.send`/`dist.recv` executing your 1F1B order; ring attention on top of `llmsys.flash` from
[@ch:attention].
:::
