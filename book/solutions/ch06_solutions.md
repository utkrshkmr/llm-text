## Chapter 4: Project 10, Know Your Machine {.unnumbered}

#### Exercise 4.1

From $\pi = N_{\text{SM}}\,f\,\phi$, $f = 989\times10^{12}/(132\times4{,}096) = 1.83$ GHz. The
datasheet number therefore assumes every SM runs its tensor cores at 1.83 GHz for the whole
computation. Compare it with the maximum clock `nvidia-smi -q -d CLOCK` reports, and then with the
clock under a sustained GEMM (the method of [@sec:proj10:power]). On our A100 a large GEMM's
throughput was proportional to the clock, so the burst-to-sustained ratio equals the clock ratio,
$1{,}035/1{,}410 = 0.73$. A card with more tensor throughput per watt of power limit will throttle
further; predict the ratio from (peak FLOP/s) / (power limit) relative to the A100's $312/250$,
then measure it with the sampler, pairing samples by timestamp.

#### Exercise 4.2

$n_{1/2} = \alpha\beta = 6.3\times10^{-6}\times23.9\times10^{9} = 151$ KB. Since
$B(n)/\beta = n/(n+n_{1/2})$, 90% needs $n = 9\,n_{1/2} \approx 1.36$ MB. A batch of
$8\times2048$ int64 ids is 131 KB: $t = 6.3\ \mu\text{s} + 131{,}072/23.9\times10^{9}\ \text{s} =
11.8\ \mu$s, an effective 11.1 GB/s (46% of $\beta$). As int32 it is 65.5 KB and takes 9.0 μs. The
3 μs saved is invisible next to a training step of tens of milliseconds, and with `non_blocking`
the copy overlaps compute anyway; the dtype matters for memory, not for transfer time.

#### Exercise 4.3

$\beta_{\text{memcpy}} = 1/(1/8.1 - 1/23.5) = 12.4$ GB/s, against $1/(1/10.0 - 1/24.0) = 17.1$
GB/s inferred from the H2D direction. The two-stage model with one host-copy rate does not fit both
directions: the D2H pageable path is slower than the model explains (its host copy runs after the
DMA into the staging buffer, and the driver may overlap the two stages less in this direction). A
direct test: time a 256 MiB `pageable.copy_(pinned)` on the CPU (one thread, then several, with
both buffers on the GPU's NUMA node) to measure $\beta_{\text{memcpy}}$ in each direction, then
predict the pageable transfer from ([-@eq:proj10:pageable]) with no overlap and with full overlap
($\min(\beta_{\text{pinned}}, \beta_{\text{memcpy}})$); the measurement should fall between the
two.

#### Exercise 4.4

With $k = n/c$ chunks, $T(c) = (n/c + 1)(\alpha + c/\beta) = n\alpha/c + n/\beta + \alpha +
c/\beta$. Setting $dT/dc = -n\alpha/c^2 + 1/\beta = 0$ gives $c^\star = \sqrt{n\alpha\beta} =
\sqrt{n\,n_{1/2}}$. For $n = 256$ MiB: with the H2D stage ($n_{1/2} = 151$ KB) $c^\star = 6.4$ MB;
with the D2H stage ($n_{1/2} = 356$ KB) $c^\star = 9.8$ MB. Both are close to the 8 MiB we used,
and at the optimum $T^\star = n/\beta + 2\sqrt{n\alpha/\beta} + \alpha \approx 11.2 + 0.5 =
11.8$ ms, about 23 GB/s. The measured 18.5 GB/s (two-GPU run) is lower because both DMA streams
share host memory, one of them necessarily crosses the inter-socket link (the two GPUs are on
different nodes, the pinned buffers on one), our measurement did not place the buffers
deliberately, and the Python loop issues four operations per chunk at several microseconds each.
`staged_bandwidth(n, 0, 1, c)` for $c$ = 1, 4, 16, 64 MiB gives the measured curve; expect the
small chunks to lose most (they pay $\alpha$ and the host's enqueue cost 256 times).

#### Exercise 4.5

For Gaussian samples the median's standard error is $\sqrt{\pi/2}\,\sigma/\sqrt{n} \approx
1.2533\,\sigma/\sqrt{n}$. A 95% interval of half-width 1% with $\sigma = 3\%$ of the median needs
$1.96\times1.2533\times0.03/\sqrt{n} \le 0.01$, so $\sqrt{n} \ge 7.4$ and $n \ge 55$. The
order-statistic rule ([-@eq:proj10:ci]) with $n = 55$ takes ranks 20 and 36. A bootstrap check:
draw 55 log-normal samples with $\sigma_{\log} = 0.03$, call `stats.bootstrap_ci`, and repeat a few
hundred times; the interval's half-width averages about 1% and it covers the true median in about
95% of repetitions. For timing distributions with a heavy right tail, the median needs fewer
samples than the mean for the same precision, one more reason to report it.

#### Exercise 4.6

Eager: the host enqueues 420 kernels at 7.4 μs each, 3.1 ms, while the GPU's own work (about
0.9 ms of weight streaming) hides underneath: the step is host-bound. Fusing half the kernels
gives 1.6 ms, still host-bound. From a CUDA graph the host costs one launch; the device runs 420
kernels with about 1.3 μs of gap each (0.55 ms) and streams the weights in 0.9 ms, so the step
takes between $\max(0.9, 0.55) = 0.9$ ms and $0.9 + 0.55 = 1.45$ ms, depending on how much of the
gaps overlap with work. Eager decoding stops being overhead-bound when the GPU's work exceeds
3.1 ms: ignoring the KV cache, the matmul FLOPs are $2\times0.6\times10^9\times B$, which take
3.1 ms at 217 TFLOP/s for $B \approx 560$. Reading the KV cache grows with batch and context and
lowers that crossover in practice.

#### Exercise 4.7

(a) Work division: PyTorch's elementwise kernels launch a grid sized from the element count; a
size that leaves a partly filled last wave of blocks over the 108 SMs wastes a fraction of the
final wave, and at a few microseconds per kernel the tail is a visible share. (b) Address mapping:
L2 slices and the two partitions are selected by hashing physical addresses; buffers whose sizes
are not powers of two can load the slices unevenly or straddle partitions differently. To
distinguish them, sweep the size finely (32 points per octave between 2 and 16 MiB): mechanism (a)
predicts dips at sizes where the block count crosses multiples of 108 times the resident blocks per
SM, and the pattern should also appear in the cold sweep; mechanism (b) predicts a dependence on
alignment, so repeat each size with the buffers offset by 1 KiB and by 64 KiB, which leaves (a)
unchanged and moves (b). The two mechanisms are not exclusive; report the fraction of the dip each
explains.

#### Exercise 4.8

With $F = 1.0995\times10^{12}$: the "burst" time is $F/261\times10^{12} = 4.213$ ms at 1,410 MHz
and the sustained time $F/217\times10^{12} = 5.067$ ms at 1,035 MHz. Subtracting,
$W_{\text{SM}}(1/1{,}035 - 1/1{,}410) = 0.854$ ms gives $W_{\text{SM}} \approx 3{,}320$ MHz$\cdot$ms
and $W_{\text{mem}} = 4.213 - 3{,}320/1{,}410 = 1.86$ ms, 44% of the burst time apparently
independent of the clock. At 765 MHz the fit predicts $3{,}320/765 + 1.86 = 6.2$ ms, 177 TFLOP/s.
But the clock-scaled peak at 765 MHz is $312\times765/1{,}410 = 169$ TFLOP/s: the fit predicts a
GEMM that completes more FLOPs per clock cycle than the tensor cores can issue, which is
impossible. The faulty assumption is that the "burst" sample ran entirely at 1,410 MHz. The clock
readings lag the GPU by up to about 100 ms, and the throttle reason "software power cap" was
already active at the first sample; the first 10-GEMM group (44 ms) partly ran at a lower clock,
so its throughput understates what 1,410 MHz delivers, and the fit invents a clock-independent
part to explain the shortfall. The correctly timestamped trace of [@fig:proj10:sustained] settles
it: throughput divided by the clock at each moment is constant at about 294 TFLOP/s per 1,410 MHz
across the steady state, so $W_{\text{mem}} \approx 0$ for this GEMM, and the proportional model
predicts $294\times765/1{,}410 = 160$ TFLOP/s at 765 MHz, below the peak as it must be.

#### Exercise 4.9

The rule multiplies compute-bound kernel times by $1{,}410/1{,}035 = 1.36$ and leaves
memory-bound ones unchanged. For the large GEMMs it should work: [@sec:proj10:power] found their
throughput proportional to the clock, so the $4096^3$ GEMM should go from 549 μs to about 750 μs.
It under-predicts latency-bound memory kernels (softmax, LayerNorm, small kernels), whose time
depends on how fast the SMs issue loads and on fixed costs that scale with the clock, and it may
over-predict the slowdown of small GEMMs, whose time includes clock-independent fixed costs.
Streaming kernels (copy, add, GEMV) should not move. To measure in the low-clock state without
locking clocks, time each kernel immediately after two seconds of back-to-back GEMMs, record the
clock with a timestamped sampler, and discard samples whose clock is not near 1,035 MHz. Habitat's
own evaluation compares different GPUs, where SM count, bandwidth and cache size change together;
this box isolates the clock, which is the cleanest test of its scaling rule.

#### Exercise 4.10

A prediction for an H100 SXM from our fractions: HBM copy $0.85\times3.35 = 2.85$ TB/s; BF16 per
clock $0.94\times989 \approx 930$ TFLOP/s at the boost clock, less under the power limit; PCIe 5.0
x16 pinned about $0.76\times63 = 48$ GB/s from the local NUMA node; GPU-to-GPU over NVLink rather
than PCIe, hundreds of GB/s, but checked for correctness all the same; fixed costs similar to ours
(they are set by the host CPU and the driver, not the GPU). The fractions most likely to transfer
are the DRAM and PCIe ones ([@tab:proj10:repro] already shows them carrying over from V100 to
A100) and the fixed costs; the least likely are the burst GEMM fraction (it depends on library
maturity for the architecture) and the burst--sustained gap (it depends on the power limit relative
to throughput, 700 W on the SXM part). The model of ([-@eq:proj10:model]) needs an interconnect
term for NVLink paths and, for FP8, a compute class whose peak depends on the scaling granularity
of [@ch:quant].

#### Exercise 4.11

A minimal test, skipped on machines with fewer than two GPUs:

``` python
@pytest.mark.parametrize("n", [1024, 16 * 2**20])
@pytest.mark.parametrize("src,dst", [(0, 1), (1, 0)])
def test_peer_copy_delivers_data(n, src, dst):
    x = (torch.arange(n, device=f"cuda:{src}") % 65521).float()
    y = x.to(f"cuda:{dst}")
    torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    assert torch.equal(y.cpu(), x.cpu())      # compare through the host, not the path under test
```

Add a non-contiguous case and the framework's own send/receive path, and record the IOMMU group
type next to the result (`env.iommu_info`): a translating IOMMU (`DMA`, `DMA-FQ`) is the
configuration NVIDIA does not support for PCIe peer-to-peer. NCCL does not need the test on our
box because it does not use peer-to-peer DMA across a `SYS` path by default: its topology search
reads the PCI hierarchy, assigns each pair of GPUs a path type, and enables the P2P transport only
up to a configured path distance (`NCCL_P2P_LEVEL`), falling back to shared host memory (the "via
SHM/direct" channels in its log) otherwise. Choosing a transport from the measured topology rather
than from the driver's yes/no answer to "can access peer" is the robust design.
