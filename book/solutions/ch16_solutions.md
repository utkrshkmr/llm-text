## Chapter 16: Performance Engineering on One GPU {#chapter-16-performance-engineering-on-one-gpu .unnumbered}

#### Exercise 16.1

One-time costs: CUDA context creation and kernel-module loading; cuBLAS handle creation and
heuristic or autotuning selection per new GEMM shape; Triton JIT compilation and `triton.autotune`
sweeps; `torch.compile` tracing (Dynamo), joint-graph partitioning (AOTAutograd) and code generation
(Inductor); caching-allocator growth (one `cudaMalloc` per new segment); CUDA-graph capture if
enabled; GPU clock ramp-up. Experiment: wrap "import + model construction", "first forward", "first
backward", "first optimizer step", "second step" and "steps 3--20" in separate NVTX ranges and
profile without a capture range. Compile and autotune appear as long host-side ranges with no
kernels; allocator growth appears as `cudaMalloc` calls in the CUDA API row; clock ramp-up shows as
identical kernels getting faster over the first iterations (also visible with `nvidia-smi dmon`).

#### Exercise 16.2

Any launch-bound workload: for example, 1000 small elementwise kernels on $10^4$-element tensors.
Each kernel runs for about $2\,\mu$s on the device, but the host needs about $5$--$10\,\mu$s per
launch, so CUDA events (device time, roughly the sum of kernel durations when the queue never
empties) report much less than wall-clock time, which includes the launch bottleneck. Report device
time (events or `do_bench`) for a kernel paper, because the claim is about the kernel; report
synchronised wall-clock time for training throughput, because host overhead is part of the real
cost.

#### Exercise 16.3

For `fc3(silu(fc1(x)) * fc2(x))` in eager mode: `fc1` and `fc2` save $x$ (the same storage, counted
once): $BTd\cdot4$ bytes. `silu` saves its input $g=\texttt{fc1}(x)$: $BTh\cdot4$. The product saves
both operands, `silu`$(g)$ and $u=\texttt{fc2}(x)$: $2BTh\cdot4$. `fc3` saves its input (the
product): $BTh\cdot4$. Total $4BT(d+4h)$ bytes, i.e. $1+4h/d$ units of the residual stream (about 13
for $h=3d$). Under `torch.compile` the partitioner typically keeps $x$, $g$ and $u$ and recomputes
`silu`$(g)$ and the product in the backward kernel, dropping two $h$-wide tensors (about $1+2h/d$
units). Verify by printing `tr.table()` in both modes.

#### Exercise 16.4

With outer groups of $k_1$ blocks, each recomputed with inner checkpoints every $k_2$ blocks, the
peak (in units of $a$, with $A=30a$) is $M=\tfrac{L}{k_1}+\tfrac{k_1}{k_2}+30k_2$: outer
checkpoints, the inner checkpoints of the outer group being recomputed, and one live inner group.
Each block runs forward three times (original, outer recomputation, inner recomputation) versus
twice for one level. Because $A\gg a$, $k_2=1$ is optimal, giving $M=64/k_1+k_1+30$, minimised at
$k_1=8$: $M=46a$. The best one-level schedule ($k=1$ or $2$) gives $94a$ or $92a$. One extra forward
per block roughly halves peak activation memory here.

#### Exercise 16.5

Per parameter, AdamW does about 12--15 FLOPs (two moment updates, bias corrections, a square root, a
division, weight decay, the update) and moves about 28 bytes ([@sec:perf:anatomy]): roughly
0.5 FLOP/byte, far below any GPU's ridge point. Qwen3-1.7B has $1.72\times10^9$ parameters, so
48.2 GB of traffic, or 14.4 ms at 3.35 TB/s. The model has about 310 parameter tensors (11 per layer
over 28 layers, plus the embedding and final norm); at about ten kernels per tensor and $5\,\mu$s
per launch, launching alone takes 15.5 ms of host time, and kernels on small tensors (norm weights)
are too short to reach bandwidth. Expect 25--40 ms unfused. A fused (or multi-tensor) implementation
launches a handful of kernels and approaches $14.4/0.8\approx18$ ms at 80% of peak bandwidth.

#### Exercise 16.6

$16\times4096=65{,}536$ tokens. Logits in bf16: $65{,}536\times151{,}936\times2$ B $=19.9$ GB; the
fp32 upcast doubles that to 39.8 GB; the gradient of the logits is another 19.9--39.8 GB. That is
80--100 GB at the peak, more than an 80 GB GPU. To keep the fp32 slab under 1 GB: chunk
$\le 10^9/(151{,}936\times4)\approx1645$ tokens, so use 1024 (0.62 GB fp32 plus 0.31 GB bf16).
Recomputation repeats the LM-head forward ($2d_{\text{model}}V=622$ MFLOP per token),
i.e. $4.1\times10^{13}$ extra FLOPs per step (about 41 ms at the H100's bf16 peak): one third more
than the LM head's normal training cost.

#### Exercise 16.7

With tile size $b$ and $n=T/b$ tiles per side, a causal forward visits $n(n+1)/2$ tiles and does two
GEMMs of $2b^2d_{h}$ FLOPs on each, giving $2\,BHd_{h}\,T(T+b)$ FLOPs ($\approx2BHd_{h}T^2$, half of
non-causal). The two-kernel backward does seven GEMMs per visited tile: $7\,BHd_{h}\,T(T+b)$,
$3.5\times$ the forward. A backward reading a stored $\mathbf{P}$ does four GEMMs
($\mathbf{dV},\mathbf{dP},\mathbf{dQ},\mathbf{dK}$), $2\times$ the forward, but must store and read
$BHT^2$ elements. The diagonal tiles are computed in full and masked, so their wasted FLOPs are the
$T\cdot b$ term.

#### Exercise 16.8

Visited tiles $n(n+1)/2$, masked tiles $n$: a fraction of $2/(n+1)$, e.g. about 3% for $n=64$. With
`BLOCK_M`$=2\,$`BLOCK_N` and $n_q=T/\texttt{BLOCK\_M}$ query tiles, query tile $i$ visits $2i+2$ key
tiles, the last two of which straddle the diagonal and need masks. Visited $=n_q(n_q+1)$, masked
$=2n_q$, so again $2/(n_q+1)$, now in terms of the (fewer) query tiles.

#### Exercise 16.9

Tiles $=16\lceil M/128\rceil$ and waves $=\lceil\text{tiles}/132\rceil$; achieved fraction
$=\text{tiles}/(132\cdot\text{waves})$. At $M=4096$: 512 tiles, 4 waves, 97%. Throughput drops just
above $M=4224$ (528 tiles fill exactly 4 waves; 544 need 5, falling to 82%), $M=5248$ (656 to 672
tiles: 99% to 85%), $M=6272$ (784 to 800 tiles) and $M=7296$ (912 to 928 tiles), with smaller steps
at every multiple of 128 from tile quantisation. A sweep of measured throughput shows the same
sawtooth.

#### Exercise 16.10

With `fullgraph=True`, Dynamo raises at `loss.item()` ("data-dependent" or "Tensor.item" graph
break) because converting a tensor to a Python number needs a device synchronisation and leaves the
graph. `TORCH_LOGS=graph_breaks` on the original code lists remaining breaks, typically from
NVTX/profiler calls inside the compiled region, `print`s, Python-side conditionals on tensor values
and calls into unsupported libraries. The remedies: move logging outside the compiled function
(return the loss tensor and call `.item()` every $k$ steps), replace data-dependent branches with
`torch.where`, mark helper functions with `torch.compiler.disable` if they must stay in Python, and
compile per block so that remaining breaks fall between compiled regions.
