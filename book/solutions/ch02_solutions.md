## Chapter 2: PyTorch Essentials {#chapter-2-pytorch-essentials .unnumbered}

#### Exercise 2.1

Contiguous $(2,3,4)$ has strides $(12,4,1)$. After `transpose(0, 2)` the shape is $(4,3,2)$ and the
strides are $(1,4,12)$, over the same storage. `.view(-1)` requires that the flattened order equal
the storage order, which no longer holds, so it raises an error. `.reshape(-1)` falls back to a
copy: one kernel that reads and writes all 24 elements (in general, $2\times$ the tensor's bytes of
traffic plus a new allocation).

#### Exercise 2.2

bf16 has 7 explicit mantissa bits, so the spacing of representable numbers in $[1,2)$ is
$2^{-7}=0.0078125$. With round-to-nearest-even, $1+2^{-8}$ rounds back to 1, so any $\delta$
slightly larger than $2^{-8}\approx0.0039$ is needed to change the value. An update of $10^{-4}$ is
lost entirely: the weight never moves. fp32 master weights have spacing
$2^{-23}\approx1.2\times10^{-7}$ near 1, so the update is applied exactly in fp32 and the bf16 copy
is re-derived from the master after each step; small updates accumulate until they cross a bf16
rounding boundary.

#### Exercise 2.3

With $a=\sigma(z)$ and $\mathcal{L}=-[y\log a+(1-y)\log(1-a)]$:
$\partial\mathcal{L}/\partial a = -y/a + (1-y)/(1-a) = (a-y)/(a(1-a))$ and
$\partial a/\partial z=a(1-a)$, so $\partial\mathcal{L}/\partial z=a-y$. For a batch with mean loss,
$\nabla_{\boldsymbol{w}}=\frac1N\mathbf{X}^\top(\boldsymbol{a}-\boldsymbol{y})$ and
$\partial/\partial b=\frac1N\sum_i(a_i-y_i)$. This is exactly `logistic_regression_grads` (Task 3) in the lab
solution, which matches autograd to $10^{-6}$.

#### Exercise 2.4

(a) The Jacobian is $m\times n$. Forward mode builds it one column at a time: each
Jacobian--vector product pushes one tangent through every layer, and for the linear layers that
costs one matrix product the size of the forward one, so $n$ columns cost about $n$ forward passes
plus the primal pass, $(1+n)F$. Reverse mode builds it one row at a time: each vector--Jacobian
product with respect to the *input* needs only $\mathbf{G}\mathbf{W}$ per layer (no weight
gradient), again one forward-sized product, so $m$ rows cost about $(1+m)F$, with the primal pass
recorded once. (b) With layers $n\to1024\to1024\to m$, one forward pass is
$2(1024n+1024^2+1024m)$ FLOPs, 3.2 MFLOP for $(512, 8)$. Forward mode costs about $513F\approx1.6$
GFLOP and reverse mode about $9F\approx29$ MFLOP, so in FLOPs `jacrev` should be about 57 times
cheaper for $(512, 8)$ and `jacfwd` about 57 times cheaper for $(8, 512)$. (c) Expect the measured
ratio on a GPU to be far smaller than 57. Both functions vectorise their passes with `vmap`, so the
512 tangents (or cotangents) become one matrix product per layer that reads each weight once:
the expensive mode is a GEMM with $M=512$, at a much higher operational intensity than the
matrix--vector products of the cheap mode, which are bound by memory and by kernel launches
([@ch:machine]). The cheap mode cannot go below a fixed cost of a few dozen launches, and the
expensive one runs its extra FLOPs at a far better rate, so the time ratio is a small factor. On a
CPU, where the GEMMs are compute-bound at much smaller sizes, the ratio moves towards the FLOP
ratio. The direction never changes: with few outputs, use reverse mode.

#### Exercise 2.5

(a) Per block the saved tensors become the block input (saved by the first `nn.Linear` for its
weight gradient) and $\mathbf{U}$; $\mathbf{W}_2$ is a parameter and is not counted. With 8,192
tokens, $d=4096$, hidden 16,384 and bf16, that is $8192\times(4096+16384)\times2 = 335.5$ MB per
block and 5.37 GB for 16 blocks, against 9.66 GB without the function and 1.07 GB with full
checkpointing. `saved_activation_bytes` reports exactly $16\times335{,}544{,}320$ bytes, because it
counts the tensors the hooks see and deduplicates by data pointer. (b) The backward pass now
recomputes one GELU per element of $\mathbf{U}$, about a dozen FLOPs for each of
$8192\times16384$ elements, roughly 1.6 GFLOP per block, against 6.6 TFLOP of GEMM work in the
block's forward and backward: negligible. The bytes matter more: reading $\mathbf{U}$ and writing
$\operatorname{GELU}(\mathbf{U})$ moves 537 MB per block, about 0.4 ms at 1,369 GB/s, or about 6 ms
for the stack, roughly 1% of the 501 ms step (a kernel that applies GELU while loading the second
GEMM's operand avoids even that write). Full checkpointing instead recomputes both GEMMs, 2.2
TFLOP and about 10 ms per block at 217 TFLOP/s. The lean block saves 4.3 GB for about 1% of the
step; full checkpointing saves 8.6 GB for a third more FLOPs. Recomputing cheap, memory-bound
operations is the idea of selective recomputation (Korthikanti et al., 2022).

#### Exercise 2.6

(a) Write $s$ for a block's input and $r$ for its residuals: $s=8192\times4096\times2=67.1$ MB and
$r=s+2\times268.4=604.0$ MB. `checkpoint_sequential` with $S=L/k$ segments checkpoints the first
$S-1$ segments and runs the last one normally, so after the forward pass it holds
$(S-1)s+kr$ bytes, which is also about the peak at the start of backward (the last segment's
residuals are already there; each earlier segment, when recomputed, holds at most the same $kr$).
For $L=16$: $k=1$: $15s+r=1.61$ GB; $k=2$: $7s+2r=1.68$ GB; $k=4$: $3s+4r=2.62$ GB; $k=8$:
$s+8r=4.90$ GB; $k=16$: $16r=9.66$ GB. (The lab's `saved_activation_bytes` checkpoints every
block, including the last, which is why it reports $16s=1.07$ GB.) (b) $k^\star=\sqrt{Ls/r}$. For
$L=64$, $k^\star=\sqrt{64\times67.1/604}\approx2.7$; among divisors, $k=2$ gives $31s+2r=3.29$ GB,
$k=4$ gives $15s+4r=3.42$ GB, and $k=1$ gives $63s+r=4.83$ GB. For blocks with $r=s$ and $L=16$,
$k^\star=4=\sqrt{L}$ and the memory is $(L/k-1+k)s=7s$ against $16s$, the $O(\sqrt{L})$ regime of
the paper. (c) The saved bytes match exactly. The measured peak is higher than $(S-1)s+kr$ by the
backward pass's own tensors: the gradient flowing between blocks (block-input-sized), the
gradients of $\mathbf{U}$ and of the GELU output (268 MB each, transient), the weight gradients if
they are not allocated before the measurement starts, cuBLAS workspace, and the allocator's
rounding of every block. With full checkpointing these add about 1 GB to the 1.07 GB of stored
inputs in [@tab:pytorch:ckpt].

#### Exercise 2.7

Parameters:
$(784\cdot512+512)+(512\cdot256+256)+(256\cdot10+10)=401{,}920+131{,}328+2{,}570=535{,}818$. In
fp32: weights 2.14 MB, gradients 2.14 MB, Adam's two moments 4.29 MB, for a total of 8.57 MB (16
bytes/parameter). Forward FLOPs for a batch of 128:
$2\times128\times(784\cdot512+512\cdot256+256\cdot10)\approx1.37\times10^8$ (biases and ReLU add
well under 1%).

#### Exercise 2.8

(a) The step consumes $32\times2048=65{,}536$ tokens every 0.25 s, 262,144 tokens/s. One worker
produces a batch in $65{,}536/400{,}000=164$ ms, faster than the 250 ms step, so one worker keeps
up with 1.5$\times$ headroom; two give room for jitter. The loader delivers batches in order,
assigning them to workers round-robin, so a stalled worker delays the batch it owes even if the
other worker is idle. With two workers each owes every second batch, one per 0.5 s, and a 1 s stall
is hidden only if that worker already has two finished batches queued: `prefetch_factor` of at
least 2 (the default), better 3, which is 4 to 6 batches, 1--1.5 s of steps, in flight. (b) Stored
as `uint16`, the loader reads $262{,}144\times2=0.5$ MB/s, nothing for a disk or the page cache. What
limits it is per-batch Python work (indexing, building an `int64` tensor, passing it from the worker
through shared memory), on the order of a millisecond per batch, still far below 250 ms. (c) The
inputs are $65{,}536\times8=524$ KB of `int64` (the shifted targets as many again), about 26 μs
each at 20 GB/s, negligible next to the step. `non_blocking=True` matters for another reason: a
blocking copy makes the CPU wait until the copy has run, and the copy runs only after every kernel
already queued on the stream, so the CPU stops running ahead of the GPU and the GPU then idles
while the next step's kernels are launched ([@sec:machine:async]).

#### Exercise 2.9

Gradients are 14 GB in bf16. A ring all-reduce sends and receives $2(P-1)/P\times14\approx24.5$ GB
per GPU for $P=8$. Over NVLink at 300 GB/s this takes $\approx82$ ms, easily hidden behind a 1.5 s
step. Over 100 Gb/s Ethernet (12.5 GB/s) it takes $\approx2.0$ s, longer than the compute, so
communication cannot be hidden and the step at least doubles. The model does not fit for DDP anyway:
7B parameters $\times$ 16 bytes $=112$ GB of weights, gradients and Adam state exceeds 80 GB before
activations. This requires ZeRO/FSDP sharding ([@ch:distributed]).

#### Exercise 2.10

(a) The gradients are $350\times10^6\times2=700$ MB. With $P=2$, a ring all-reduce sends
$2(P-1)/P=1$ times the gradient volume per GPU in each direction, 700 MB, so the transfers alone take
$700/20=35$ ms, well under the 120 ms backward pass. With $n=700/B$ buckets of $B$ MB, assume
gradients become ready at a uniform rate, one bucket every $120/n$ ms, and that the all-reduces run
one after another, each taking $\alpha+B/\beta$. If each all-reduce is shorter than the interval
between buckets, communication keeps up and only the last bucket, which becomes ready when backward
ends, is exposed: $\alpha+B/\beta$. Otherwise a backlog builds and the exposed time is about
$n(\alpha+B/\beta)-120$ ms. Taking $\alpha=0.1$ ms as an assumption to be measured: $B=1$ MB gives
700 calls of 0.15 ms against an interval of 0.17 ms, so it barely keeps up and exposes about
0.15 ms, but any $\alpha$ above 0.12 ms tips it into a backlog (at $\alpha=0.2$ ms, $700\times0.25-120=55$
ms exposed); $B=25$ MB exposes about 1.35 ms; $B=100$ MB about 5.1 ms; $B=1{,}000$ MB holds all
700 MB in one bucket that starts only when backward ends, exposing the full 35 ms. The prediction
is a broad optimum between a few and a few tens of megabytes, with small buckets punished by
$\alpha$ and large ones by lost overlap. (b) Measure the step with `torch.cuda.Event` timing after
a warm-up, for each `bucket_cap_mb`, and check once that both ranks hold identical weights after a
step, since on this box direct peer copies are unreliable while NCCL is correct. (c) The
assumptions to check first: gradients are not produced uniformly (DDP orders buckets in reverse
parameter order, and its first bucket is small by design); NCCL's kernels use SMs and slow the
backward pass they overlap with; with no peer-to-peer path, NCCL stages through host memory, so
$\beta$ may be well below 20 GB/s; and $\alpha$ includes the CPU cost of launching each call, which
competes with the backward pass's own launches.
