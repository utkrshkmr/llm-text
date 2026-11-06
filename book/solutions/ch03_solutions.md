## Chapter 3: The Machine Underneath: GPUs, Rooflines and Measurement {#chapter-3-the-machine-underneath .unnumbered}

#### Exercise 3.1

A100: $I^\star=312/2.04\approx153$ FLOP/byte; a bf16 square GEMM has $I=n/3$, so it becomes
compute-bound at $n\approx460$. RTX 4090: $I^\star=165/1.008\approx164$, so $n\approx490$. The
decode critical batch is $B^\star\approx bI^\star/2=I^\star$ in bf16, so the 4090 needs a slightly
larger batch (about 164 versus 153). Both need far smaller batches than an H100
($\approx$`<!-- -->`{=html}295), whose compute has outgrown its bandwidth.

#### Exercise 3.2

$d=4096$, hidden $4d=16{,}384$, bf16. *$BT=1$*: each GEMM does
$2\times4096\times16{,}384\approx1.34\times10^8$ FLOPs and moves about
$2\times4096\times16{,}384\approx134$ MB (the weight dominates), so $I\approx1$: memory-bound, about
40 μs each on an H100. GELU touches 16,384 elements, which is negligible but still costs a launch.
*$BT=4096$*: GEMM 1 does $5.5\times10^{11}$ FLOPs and moves
$2(4096^2+2\cdot4096\cdot16{,}384)\approx302$ MB, so $I\approx1{,}800$: compute-bound,
$\approx0.56$ ms at peak; GEMM 2 is the same. GELU reads and writes $4096\times16{,}384$ bf16
values, 268 MB, $\approx0.08$ ms, memory-bound. Fusing GELU into GEMM 1's epilogue removes that
round trip (the GEMM output is activated in registers before being stored), saving $\approx0.08$ ms
of $\approx1.2$ ms, about 7%. Fusion matters far more for chains of elementwise ops without a GEMM
to hide behind.

#### Exercise 3.3

By Little's law the H100 needs $3.35\times10^{12}\times500\times10^{-9}\approx1.68$ MB of loads in
flight, $1.68\times10^6/132\approx12.7$ KB per SM, about 25 outstanding warp loads of 512 bytes per
SM at every moment. The A100 needs 6.3 KB, about a dozen. Per SM the H100 must therefore keep
twice as many bytes in flight, while the per-SM resources that hold them grew far less: the
register file is the same 256 KB and shared memory went from 164 KB to 228 KB. A kernel whose
occupancy and per-warp parallelism were just enough on the A100 keeps roughly half of the bytes in
flight that the H100 needs, and reaches roughly half of its bandwidth roof, with no change in its
code. This is one reason Hopper added the Tensor Memory Accelerator, which issues large
asynchronous bulk copies without occupying a thread per load ([@ch:gpuarch]).

#### Exercise 3.4

(a) $\lceil1408/128\rceil^2=11^2=121$ output tiles; with one tile per SM at a time, 108 SMs need
$\lceil121/108\rceil=2$ waves, and the second wave has 13 tiles, so 12% of its SM-time does useful
work. (b) The GEMM occupies $2\times108=216$ tile-slots for 121 tiles of work, so quantisation alone
caps it at $121/216=56\%$ of the compute roof (before the fixed costs of
[@sec:machine:hides]). (c) In the sweep's range the tile count steps with $\lceil n/128\rceil$:
$n\le1{,}280$ gives 100 tiles in one wave (93% of the slots); $1{,}281\le n\le1{,}408$ gives 121
tiles in two waves (56%); $1{,}409\le n\le1{,}536$ gives 144 tiles in two waves (67%). Expect a
sawtooth: throughput falls sharply between 1,280 and 1,312, where the work grows by 5% and the time
almost doubles, and recovers partly towards 1,536. In practice cuBLAS picks among tile shapes and may
split the $K$ dimension across SMs for shapes like these, which smooths the steps; how much of the
sawtooth survives tells you how well its heuristics cover the shape.

#### Exercise 3.5

For 1,024 elements, both eager and compiled versions are dominated by per-op overhead: eager time is
about $100\times$ the per-launch cost (hundreds of μs to milliseconds); this is regime (b) of
[@fig:machine:async]. For $10^8$ elements, each op is a memory-bound kernel taking
$\approx0.24$ ms in fp32 on an H100, so eager time is about 24 ms and overhead is hidden; this is
regime (a). `torch.compile` fuses the chain into one kernel that reads and writes once. The large
tensor speeds up by up to $\approx100\times$ in traffic terms, and the small tensor speeds up
because 100 launches became one; this is regime (c). With `reduce-overhead`, CUDA graphs remove the
remaining launch cost.

#### Exercise 3.6

There are four synchronisation points. (1) `x.cuda()` and `y.cuda()` without `non_blocking=True`
are blocking copies: the CPU waits until they have run, which is after all previously queued
kernels. (2) `if torch.isnan(loss):` converts a GPU tensor to a Python `bool`. (3) `.item()` on the
correct-count. (4) `loss.item()` in the `print`. None is needed on every step. The step also runs
a second forward pass, on weights already changed by `opt.step()`, just to count correct
predictions; the training forward's logits serve. A version that waits once every 100 steps:

``` python
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
n_correct = torch.zeros((), dtype=torch.long, device="cuda")
loss_sum = torch.zeros((), device="cuda")
bad = torch.zeros((), dtype=torch.bool, device="cuda")
for step, (x, y) in enumerate(loader):
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    bad |= ~torch.isfinite(loss.detach())              # device-side flags and sums
    loss_sum += loss.detach()
    n_correct += (logits.detach().argmax(-1) == y).sum()
    if step % 100 == 99:                               # the only synchronisation
        if bad.item():
            raise RuntimeError("diverged")
        print(step, (loss_sum / 100).item())
        loss_sum.zero_()
```

The price is that a NaN is detected up to 100 steps late, after the optimiser has applied it, so
the run must restart from a checkpoint; a guard that skips the update on the device instead (as the
gradient scaler of mixed-precision training does with its found-infinity flag, [@sec:quant:mixed])
keeps the weights clean without a synchronisation.

#### Exercise 3.7

At batch 8, the profile is dominated by many small kernels: `addmm`/`mm` at low intensity
(memory-bound, weight reads dominate), plus optimiser elementwise kernels (`add_`, `mul_`, foreach
ops) that touch every parameter and are memory-bound. At batch 4096, `addmm`/`mm` in forward and
backward take the large majority of device time and are compute-bound; elementwise ops shrink to a
small fraction. Optimiser cost is independent of batch size, so its share falls as the batch grows.

#### Exercise 3.8

Achieved model FLOP/s $=2.1\times10^6\times6\times3\times10^9=3.78\times10^{16}$; peak
$=64\times989\times10^{12}=6.33\times10^{16}$; MFU $\approx60\%$. That is at the very top of what
well-tuned dense training achieves, so ask: are padding tokens counted as tokens? Is $N$ the total
or non-embedding count? Are attention FLOPs or recomputation included (that would make it hardware
FLOPs utilisation, HFU, not MFU)? Is the peak taken as the bf16 or the fp8 rate? Is the throughput a
steady-state peak or an average including evaluation and checkpoint stalls?

#### Exercise 3.9

(a) An L2 probe keeps source and destination inside the 40 MB L2 (for example two 8 MiB buffers)
and runs the copy many times inside one timed call (a Python loop of 100 copies, or a CUDA graph of
them), dividing the bytes moved by the total time; with the per-call fixed cost of about 14 μs
amortised, the probe measures the L2's bandwidth, several times the 1.37 TB/s HBM roof. Report it
with the same median and p10/p90 as `time_stats`. (b) Only kernels whose working set fits in L2
and is re-read can be limited by it. The streaming kernels of [@tab:machine:points] (add, row sum,
softmax, the GEMV and the decode attention) touch hundreds of megabytes once each, so HBM binds.
The tiled GEMMs and the prefill attention re-read their tiles of $\mathbf{A}$, $\mathbf{B}$ or
$\mathbf{K}$, $\mathbf{V}$ many times, mostly from L2: their L2 intensity is far below their HBM
intensity, and for the $M=256$ GEMM, which sits just right of the HBM ridge, the L2 roof is the
candidate. (c) Nsight Compute's `dram__bytes_read.sum` and `dram__bytes_write.sum` give
$Q_{\text{HBM}}$, and `lts__t_bytes.sum` gives the L2 traffic. Divide the kernel's $F$ by each to get
$I_{\text{HBM}}$ and $I_{\text{L2}}$, and plot the kernel's achieved FLOP/s at both intensities
against both roofs; the roof the kernel is closest to is the one that binds. For a well-tiled GEMM
the HBM point lies near $Q_{\min}$ and the L2 point far to the left but still under the much higher
L2 roof, so neither memory level binds and the kernel is limited by the tensor cores and the
pipeline that feeds them.

#### Exercise 3.10

(a, b) Measured on 2x A100-40GB (PCIe), one GPU, 8,192-cubed bf16 GEMMs run for eight seconds
each: random normal data, 215 TFLOP/s at an SM clock of 1,035 MHz and 250 W (the limit); random
data with the low 4 of the 7 mantissa bits zeroed, 230 TFLOP/s at 1,095 MHz and 249 W; zeros,
290 TFLOP/s at 1,380 MHz and 234 W. Zeroing mantissa bits reduces the number of bits that toggle in
the multipliers, so each FLOP costs less energy and the power manager allows a higher clock; the
exponent and the upper mantissa bits of random data still toggle, so the gain is partial. Zeros
toggle almost nothing and the card is no longer power-limited. (c) Treating the full board power as
the cost (the card idles near 35 W, which a more careful study subtracts), the random-data GEMM
costs $250/(214.7\times10^{12}) \approx 1.2$ pJ per FLOP, and the 1 GiB copy, which also runs at
the 250 W limit at 1,379 GB/s, costs $250/(1.379\times10^{12}) \approx 180$ pJ per byte moved.
The two energies are equal at an operational intensity of about $180/1.2 \approx 155$ FLOP/byte,
almost exactly the empirical ridge point: on a power-limited card running at its limit, time and
energy are proportional and the energy roofline and the time roofline coincide. (d) An H100 SXM
has a 700 W limit for 989 TFLOP/s, about the same peak FLOP/s per watt of limit as the A100 PCIe
(312 TFLOP/s for 250 W), so it is also expected to be power-limited on dense random-data GEMMs, and
reports of data-dependent GEMM throughput on H100s (He, 2024) agree; the size of the effect depends
on how much clock headroom the power manager has at the limit and has to be measured.
