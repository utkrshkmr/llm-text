# Accelerators Beyond the GPU {#app:accel}

The book's systems material is written in the vocabulary of NVIDIA GPUs because that is where most
LLM training and serving runs today. But the underlying questions are the same on every accelerator:
where do the weights live, who decides when bytes move, how fast can the chips talk to each other,
and what must the programmer say explicitly that the hardware will not infer? This appendix answers
those questions for four other designs: Google's TPUs (systolic arrays and a torus interconnect),
Tenstorrent's Wormhole (a mesh of small programmable cores with software-managed SRAM and Ethernet
scale-out), wafer-scale and SRAM-only designs (Cerebras, Groq), and unified-memory desktop machines
such as NVIDIA's GB10. Each comes with the model or simulator needed to reason about LLM workloads
on it.

Specifications quoted here are headline numbers from vendors' public material (rounded, dense unless
marked) as of 2025. Precisions and measurement conventions differ between vendors, so treat
cross-vendor ratios as order-of-magnitude guides and check current documentation before relying on a
number. The code is `llmsys/systolic.py`, exercised in Lab B.

![Four accelerator families in one abstract machine: compute units, on-chip memory, off-chip memory
and interconnect. Moving right, more of the data movement that GPUs leave to caches and the hardware
scheduler is planned explicitly by the programmer or the
compiler.](figures/appB_accelerators-1.svg){#fig:accel:taxonomy}

## One roofline, many machines {#sec:accel:roofline}

The roofline of [@ch:machine] applies to every design. A kernel with arithmetic intensity $I$
(FLOPs per byte of off-chip traffic) runs at most at $\min(C, I\cdot B)$ FLOP/s, where $C$ is peak
compute and $B$ the off-chip bandwidth. The ridge point $C/B$ is the intensity a kernel needs to be
compute-bound. [@fig:accel:roofline] overlays four accelerators.

![Rooflines from published peak numbers (`llmsys.systolic.ACCELERATORS`). Batch-1 decode has an
intensity of about 2 FLOPs per weight byte at bf16 (1 multiply-add per weight), so every machine
runs it far below peak and its speed is set by memory bandwidth alone. Designs with a high ridge
point (Wormhole's GDDR6, GB10's LPDDR5X) depend even more on batching, or on keeping weights in
on-chip SRAM, for decode.](figures/appB_accelerators-2.svg){#fig:accel:roofline}

::: napkinmath
[Single-stream decode ceilings]{.box-title}

Decode must stream every weight from off-chip memory once per token, so tokens/s
$\le B/(\text{bytes of
weights})$ (`decode_tokens_per_s_bound`). For an 8B model at 1 byte per parameter: H100
$3.35\times10^{12}/8\times10^9 \approx 419$, TPU v5p $\approx 345$, Wormhole n300 $\approx 72$, GB10
$\approx 34$ tokens/s. GB10's selling point is the other side of the ledger: its 128 GB of unified
memory holds a 70B model at 4 bits ($\approx35$ GB) that no 24 GB card can, but that model then
decodes at $\le 273\times10^{9}/35\times10^{9}\approx 8$ tokens/s per stream. Capacity and bandwidth
are separate purchases.
:::

## Systolic arrays {#sec:accel:systolic}

A systolic array is a grid of multiply-accumulate processing elements (PEs) that pass operands to
their neighbours every cycle, like blood pulsing through a body. Each operand fetched from memory is
reused once per PE it traverses, so a $128\times128$ array performs 16,384 MACs per cycle while
reading only 128 new values per edge per cycle. That is a structural answer to the
arithmetic-intensity problem: the reuse is wired into the dataflow rather than obtained from caches.

### Weight-stationary dataflow

The TPU's matrix unit (MXU) is *weight-stationary*. To compute $Y = XW$ with
$W\in\mathbb{R}^{K\times N}$ on a $K\times N$ array:

1.  load $W$ so that PE$(k,n)$ holds $w_{kn}$ (one row per cycle, or double-buffered behind the
    previous tile's computation);

2.  feed row $m$ of $X$ in from the left, *skewed*: $x_{mk}$ enters row $k$ at cycle $m+k$ and moves
    one PE to the right per cycle;

3.  partial sums flow down each column: PE$(k,n)$ adds $x_{mk}w_{kn}$ to the partial sum arriving
    from above and passes it on, so $y_{mn}$ leaves the bottom of column $n$ at cycle $m+n+K-1$.

![A $K\times N=3\times4$ weight-stationary systolic array. Inputs are skewed so that $x_{mk}$ and
the partial sum for $y_{mn}$ meet in PE$(k,n)$ at the same cycle. `simulate_weight_stationary`
performs exactly these register shifts.](figures/appB_accelerators-3.svg){#fig:accel:ws}

``` {#lst:accel:ws .python caption="One cycle of the weight-stationary array (`llmsys/systolic.py`)."}
for t in range(M + N + K - 2):
    x_new = np.zeros_like(x_reg); x_new[:, 1:] = x_reg[:, :-1]   # shift right
    for k in range(K):                                           # skewed injection
        m = t - k
        if 0 <= m < M:
            x_new[k, 0] = X[m, k]
    p_in = np.zeros_like(p_reg); p_in[1:, :] = p_reg[:-1, :]     # sums from above
    p_new = p_in + x_new * W                                     # every PE: one MAC
    for n in range(N):                                           # bottom row exits
        m = t - (K - 1) - n
        if 0 <= m < M:
            Y[m, n] = p_new[K - 1, n]
    x_reg, p_reg = x_new, p_new
```

### Utilisation: fill, drain and padding

A full pass takes $M+K+N-2$ cycles and performs $MKN$ MACs on $KN$ PEs, so the utilisation is
$$U \;=\; \frac{MKN}{KN\,(M+K+N-2)} \;=\; \frac{M}{M+K+N-2}.$$ For matrices larger than the
$R\times C$ array, $W$ is tiled into $\lceil K/R\rceil\lceil
N/C\rceil$ tiles and all $M$ rows stream through each tile. With weight loads overlapped,
$$U \;=\; \frac{MKN}{RC\,\lceil K/R\rceil\lceil N/C\rceil\,(M+R+C-2)}.$$ Two lessons follow, and
both shape how models are served on TPUs ([@fig:accel:util]):

- **Small $M$ wastes the array.** In decode, $M$ is the batch size. With $M=8$ on a $128\times128$
  MXU, $U = 8/262 \approx 3\%$: fill and drain dominate. Prefill with $M$ in the thousands reaches
  above 90%. This is the systolic version of decode being memory-bound.

- **Dimensions should be multiples of the array.** $K=4100$ instead of 4096 needs 33 tiles along $K$
  instead of 32, wasting 3% of the work on padding, and XLA pads silently. Model widths, head
  dimensions and vocabulary sizes are chosen as multiples of 128 (or 256 on newer TPUs) for this
  reason.

![Utilisation of a $128\times128$ weight-stationary array computing $X W$ with
$W\in\mathbb{R}^{4096\times4096}$, from `tiled_systolic_cycles`. The curve is $M/(M+254)$: half
utilisation needs $M\approx R+C$.](figures/appB_accelerators-4.svg){#fig:accel:util}

::: deepdive
[Output-stationary and other dataflows]{.box-title}

In an *output-stationary* array each PE owns one output $c_{ij}$ and accumulates it in place while
$a$ values flow right and $b$ values flow down (`simulate_output_stationary`). It avoids moving
partial sums but must drain the outputs at the end. *Row-stationary* (Eyeriss) and other dataflows
trade which operand is reused locally, a design space that matters most for convolutions. For
transformer matmuls, weight-stationary with large tiles is the common choice because the weights are
the largest operand and are reused across many rows of activations.
:::

## TPUs {#sec:accel:tpu}

### The chip

A TPU chip contains one or two *TensorCores*. Each TensorCore combines several MXUs ($128\times128$
systolic arrays on v2--v5; $256\times256$ on v6e "Trillium"), a vector unit (VPU) for elementwise
work such as softmax, norms and activations, a scalar unit that runs the compiler-generated control
program, and tens of megabytes of software-managed on-chip memory (VMEM). There are no hardware
caches between HBM and the compute units in the GPU sense. The XLA compiler decides which tiles live
in VMEM when, and it inserts the DMAs to move them.

::: {#tab:accel:tpu .table-float}
| chip               | bf16 TFLOP/s |   HBM |    HBM BW | pod topology                                    |
|:-------------------|-------------:|------:|----------:|:------------------------------------------------|
| TPU v4             |          275 | 32 GB |  1.2 TB/s | 3-D torus, 4096 chips, optical circuit switches |
| TPU v5e            |          197 | 16 GB | 0.82 TB/s | 2-D torus, 256 chips                            |
| TPU v5p            |          459 | 95 GB | 2.76 TB/s | 3-D torus, up to 8960 chips                     |
| TPU v6e (Trillium) |          918 | 32 GB |  1.6 TB/s | 2-D torus, 256 chips                            |

: Headline numbers for recent TPU generations from Google's public material (rounded). Newer
generations exist; check current documentation.
:::

### The interconnect: a torus, not a switch

TPU chips connect directly to their neighbours through inter-chip interconnect (ICI) links, forming
a 2-D or 3-D torus: every chip has links to its $\pm x$, $\pm y$ (and $\pm z$) neighbours, with
wrap-around. There is no central switch. A collective along one torus axis is a bidirectional ring
over that axis ([@sec:sharding:accounting]), and a multi-axis all-reduce runs one ring per
axis in sequence. Because a ring over a torus dimension of size $d$ uses only the links of that
dimension, collectives along *different* axes can proceed concurrently.

This topology explains the TPU programming style. A device mesh in JAX is not an abstraction over a
switch; it maps (nearly) directly onto the physical torus, and choosing which tensor dimension to
shard along which mesh axis is choosing which physical links carry the traffic. GPUs offer
all-to-all bandwidth inside an NVLink domain (up to 72 GPUs in recent NVL72 racks) and a slower
switched network outside it. TPUs offer uniform neighbour bandwidth across a pod of thousands of
chips but non-uniform distance.

![A $4\times4$ slice of a 2-D torus (wrap-around links dashed). Sharding a tensor along a mesh axis
puts its collectives on that axis's rings.](figures/appB_accelerators-5.svg){#fig:accel:torus}

### Programming model: compile everything, annotate shardings

TPU programs are traced into XLA's HLO graph and compiled *as a whole* for fixed shapes. Three
consequences matter for LLM code:

- **Static shapes.** Every new sequence length is a new compilation. Serving systems therefore
  bucket sequence lengths (pad to 128, 256, 512, ...) and use fixed-size, paged KV caches; training
  pads or packs to a fixed context length.

- **Fusion is the compiler's job.** XLA fuses elementwise chains into the surrounding matmuls
  automatically. Hand-written kernels (Pallas, JAX's Triton-like kernel language) are reserved for
  attention variants and other operations that XLA does not fuse well.

- **Sharding is declarative.** You annotate how arrays are laid out on the mesh. The compiler
  (GSPMD) propagates shardings through the graph and inserts the collectives. FSDP, TP and their
  combinations are a few annotations rather than a wrapper library.

``` {#lst:accel:jax .python caption="FSDP $\\times$ TP for one MLP in JAX: shardings declared, collectives inferred (illustrative; not run in this environment)."}
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(jax.devices().reshape(4, 2), axis_names=("fsdp", "tp"))
x_s  = NamedSharding(mesh, P("fsdp", None))       # batch split over fsdp axis
w1_s = NamedSharding(mesh, P("fsdp", "tp"))       # D sharded (FSDP), F sharded (TP)
w2_s = NamedSharding(mesh, P("tp", "fsdp"))

@jax.jit
def mlp(x, w1, w2):
    h = jax.nn.gelu(x @ w1)                       # XLA all-gathers w1 over "fsdp"
    return h @ w2                                 # ... and reduce-scatters over "tp"

x  = jax.device_put(jnp.ones((256, 1024)), x_s)
w1 = jax.device_put(jnp.ones((1024, 4096)), w1_s)
w2 = jax.device_put(jnp.ones((4096, 1024)), w2_s)
y = mlp(x, w1, w2)    # inspect with jax.jit(mlp).lower(x, w1, w2).compile().as_text()
```

The accounting method of [@sec:sharding:accounting] carries over unchanged. Only $W$
changes: it becomes the ICI bandwidth along the relevant torus axis (or axes, if a collective spans
several).

### Writing a TPU kernel: Pallas

When XLA's fusion is not enough, which in practice means attention variants, paged-KV decode and
some quantised matmuls, JAX offers *Pallas*, a kernel language in the spirit of Triton
([@app:triton]). A kernel is a Python function over *references* to blocks of its operands. A
`BlockSpec` says how the grid of program instances maps to blocks of each array, and the compiler
generates the HBM$\leftrightarrow$VMEM DMAs and pipelines them. [@lst:accel:pallas] is the
canonical tiled matmul. Compare it with the Triton version in [@app:triton]: the difference
is that the block movement is declared, not written.

``` {#lst:accel:pallas .python caption="A tiled matmul in Pallas (illustrative; follows the public JAX documentation, not run here)."}
import functools, jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_kernel(x_ref, w_ref, o_ref, acc_ref, *, nk):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)        # VMEM scratch accumulator
    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...],
                            preferred_element_type=jnp.float32)   # runs on the MXU
    @pl.when(pl.program_id(2) == nk - 1)
    def _():
        o_ref[...] = acc_ref[...].astype(o_ref.dtype)

def matmul(x, w, bm=256, bn=256, bk=512):
    M, K = x.shape; _, N = w.shape
    return pl.pallas_call(
        functools.partial(matmul_kernel, nk=K // bk),
        grid=(M // bm, N // bn, K // bk),
        in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                  pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],  # VMEM scratch
    )(x, w)
```

Block sizes must be multiples of the hardware tile (for example $8\times128$ for 32-bit data on
recent TPUs), and they must fit double-buffered in VMEM, which is the same budget reasoning as
shared memory in Triton. To see where time goes, wrap a step in `jax.profiler.trace` and open the
trace in TensorBoard or Perfetto: XLA's fused ops, the DMA activity and the collectives on each ICI
axis appear as separate rows, the TPU equivalent of an Nsight timeline.

## Tenstorrent: a mesh of programmable cores {#sec:accel:tt}

Tenstorrent's chips take a different path from both GPUs and TPUs: a two-dimensional grid of fairly
small, fully programmable cores, each with its own SRAM, connected by on-chip networks, and chips
connected to each other with plain Ethernet. Because the same software stack spans one card
(n150/n300), a workstation (TT-QuietBox), and racks (Galaxy), the programming model scales from one
core to many chips without changing paradigm. This section describes the Wormhole generation as
documented in Tenstorrent's open-source TT-Metalium repository and product pages (2024--2025). Exact
core counts, SRAM sizes and API names vary between products and releases, so verify against the
current documentation.

### The Tensix core

The unit of compute is the **Tensix** core ([@fig:accel:tensix]). Each contains:

- **five small in-order RISC-V processors** ("baby cores"). Two are *data-movement* processors that
  issue reads and writes over the two networks-on-chip (NoC 0 and NoC 1); three cooperate on
  *compute*, one each for *unpack* (move tiles from SRAM into the math engine's source registers,
  converting formats), *math* (issue instructions to the matrix and vector engines), and *pack*
  (write results back to SRAM, converting formats);

- a **matrix engine** (FPU) that operates on $32\times32$ *tiles*, and a **vector engine** (SFPU)
  for elementwise and transcendental functions;

- about **1.5 MB of SRAM** ("L1"), directly addressable and *not* a cache. Nothing moves in or out
  unless a data-movement kernel moves it;

- **NoC routers** linking the core to its neighbours on two 2-D torus networks, with hardware
  multicast: one write can deliver a tile to a whole rectangle of cores.

![Anatomy of a Tensix core (schematic, after Tenstorrent's public documentation). Two processors
move data between the NoCs and SRAM; three drive the compute pipeline unpack $\to$ math $\to$ pack.
Circular buffers in SRAM decouple the two
sides.](figures/appB_accelerators-6.svg){#fig:accel:tensix}

::: {#tab:accel:tt .table-float}
| product | public headline figures (Wormhole generation, rounded) |
|:---|:---|
| n150 card | 1 chip, 72 Tensix cores, $\approx$`<!-- -->`{=html}108 MB SRAM, 12 GB GDDR6 at 288 GB/s, up to $\approx$`<!-- -->`{=html}262 TFLOP/s (8-bit) |
| n300 card | 2 chips, 128 Tensix cores, $\approx$`<!-- -->`{=html}192 MB SRAM, 24 GB GDDR6 at 576 GB/s, up to $\approx$`<!-- -->`{=html}466 TFLOP/s (8-bit) |
| TT-QuietBox | workstation with four n300 cards: 8 chips, 96 GB GDDR6, chips linked by Ethernet |
| Galaxy | rack server with 32 chips in a 2-D Ethernet mesh |

: Wormhole products as described by Tenstorrent (2024--2025). The successor generation, Blackhole,
adds more cores, more GDDR and large general-purpose RISC-V cores on chip.
:::

![Schematic floor plan of a Wormhole-class chip: a grid of Tensix cores on two torus NoCs, DRAM
controllers and Ethernet cores at the edges. A data-movement kernel reads a tile from any DRAM bank
or any core's SRAM by NoC address; dimension-ordered routing makes the hop count, and therefore the
latency, depend on placement.](figures/appB_accelerators-7.svg){#fig:accel:wormhole}

### Programming model: reader, compute, writer

TT-Metalium, the low-level SDK, exposes the hardware directly. A *program* assigns three cooperating
kernels to each participating core:

::: description
runs on a data-movement processor and issues asynchronous NoC reads from DRAM, or from other cores'
SRAM, into a circular buffer (CB) in local SRAM;

runs on the three compute processors (compiled once, specialised into unpack/math/pack). It waits
for tiles in its input CBs, runs tile operations such as *matmul tiles* or *add tiles*, and pushes
results into an output CB;

runs on the other data-movement processor and drains the output CB to DRAM or to other cores.
:::

CBs are bounded producer--consumer queues of tiles with explicit *reserve/push* and *wait/pop*
operations, so double buffering is simply a CB with two slots. Host code creates buffers in DRAM or
SRAM (*interleaved* across banks or *sharded* across cores), compiles the kernels, sets per-core
runtime arguments (which tiles this core owns) and enqueues the program.

![The three-kernel structure of a TT-Metalium program on one core. Data movement and compute run
concurrently on different processors and synchronise only through circular buffers in local SRAM.
This is software pipelining made explicit.](figures/appB_accelerators-8.svg){#fig:accel:rcw}

``` {#lst:accel:ttkernel .cpp caption="The shape of a tiled matmul's compute kernel on a Tensix core. This is *schematic* pseudocode in the style of TT-Metalium's C++ kernel API; consult the SDK's examples for exact names and signatures."}
// compute kernel: C[tile] = sum_k A[tile, k] * B[k, tile] for this core's output tiles
void kernel_main() {
    uint32_t Kt = get_arg(0);                 // K / 32 tiles along the reduction dim
    for (uint32_t out = 0; out < num_out_tiles; ++out) {
        acquire_dst_registers();               // accumulate in math registers
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(CB_A, 1);           // reader has delivered A[., kt]
            cb_wait_front(CB_B, 1);           //               ... and B[kt, .]
            matmul_tiles(CB_A, CB_B, 0, 0, /*dst=*/0);   // 32x32x32 on the FPU
            cb_pop_front(CB_A, 1);
            cb_pop_front(CB_B, 1);
        }
        cb_reserve_back(CB_OUT, 1);
        pack_tile(/*dst=*/0, CB_OUT);         // registers -> SRAM (format convert)
        cb_push_back(CB_OUT, 1);              // writer may now send it to DRAM
        release_dst_registers();
    }
}
```

On top of TT-Metalium sits **TT-NN**, a PyTorch-like library of operations for tensors on device
(with explicit layouts and memory configurations), and above that compiler front ends (TT-Forge on
TT-MLIR) that lower PyTorch or JAX graphs. Model bring-up usually happens in TT-NN. The things you
must decide that a GPU would hide are the *layout* (row-major or tiled), the *memory* (DRAM or
SRAM), the *sharding* across the core grid (height, width or block), and, for matmuls, the program
configuration (core grid, block sizes, whether to multicast).

``` {#lst:accel:ttnn .python caption="TT-NN at a glance (illustrative; API as documented around 2025, not run here)."}
import torch, ttnn
device = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.randn(1024, 4096), dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=device)   # tilize on transfer
b = ttnn.from_torch(torch.randn(4096, 4096), dtype=ttnn.bfloat8_b,  # block-fp8 weights
                    layout=ttnn.TILE_LAYOUT, device=device)
c = ttnn.matmul(a, b, memory_config=ttnn.L1_MEMORY_CONFIG)       # keep result in SRAM
out = ttnn.to_torch(c)
ttnn.close_device(device)
```

### Tiles and formats

Tensix engines consume $32\times32$ tiles stored as four $16\times16$ *faces*, so a tensor in
`TILE_LAYOUT` is not row-major in memory. `llmsys.systolic.tilize` reproduces that layout
([@fig:accel:tile]), and Lab B asks you to verify it and to measure what fraction of a
tensor is padding for awkward shapes. Wormhole also supports *block floating point* formats (such as
`bfloat8_b`), in which a group of values within a tile shares one exponent. That halves weight
traffic relative to bf16 for a modest precision cost, the same trade as the microscaling formats of
[@ch:efficient].

![Tile layout: a matrix is cut into $32\times32$ tiles, each stored as four $16\times16$ faces.
Converting to and from this layout ("tilize"/"untilize") is a real cost at model
boundaries.](figures/appB_accelerators-9.svg){#fig:accel:tile}

### Matmul on a grid of cores

A large matmul is distributed over the core grid by *block sharding*: core $(i,j)$ owns output block
$C_{ij}$ and needs the $i$-th block-row of $A$ and the $j$-th block-column of $B$. Two classic
schedules from distributed-memory computing map directly onto a 2-D NoC:

- **Cannon's algorithm**: pre-skew the blocks, then repeat $p$ times: multiply locally, pass the $A$
  block one hop left and the $B$ block one hop up. All transfers are nearest-neighbour and every
  link is busy (`cannon_matmul`).

- **SUMMA**: at step $k$, the cores in column $k$ multicast their $A$ blocks along their rows and
  the cores in row $k$ multicast their $B$ blocks down their columns. With hardware multicast a
  broadcast along a line of $p$ cores costs $p-1$ link traversals (`summa_traffic`). Tenstorrent's
  multi-core matmul configurations use this "multicast" pattern.

Either way, the question is the GPU tiling question of [@app:triton] one level up: is a
core's block large enough that its local matmul takes longer than receiving the next blocks? With
$C_c$ FLOP/s per core and NoC link bandwidth $W_n$, a $b\times b\times b$ block step costs
$2b^3/C_c$ of compute against $2b^2\cdot\text{bytes}/W_n$ of transfer, so
$b\gtrsim C_c\cdot\text{bytes}/W_n$ (`mesh_compute_bound_block`). The SRAM budget bounds $b$ from
above: two input blocks, double-buffered, plus an accumulator must fit in about 1.5 MB.

![One SUMMA step on a mesh of cores with NoC multicast. Each core accumulates its output block in
local SRAM; only $A$ and $B$ blocks travel.](figures/appB_accelerators-10.svg){#fig:accel:summa}

::: systemslens
[What changes when you port an LLM to a Tensix mesh]{.box-title}

- **Placement is the program.** Choosing which core owns which block of which weight, and which
  activations stay in SRAM between operations, is the main optimisation. A small model's weights can
  be *entirely SRAM-resident*: about 192 MB on an n300 holds a few hundred million 8-bit parameters,
  which removes DRAM from the decode loop.

- **Decode is DRAM-bound as everywhere else.** At 288--576 GB/s per card, decoding a larger model is
  bandwidth-limited ([@sec:accel:roofline]), so block-fp8 weights, batching and tensor
  parallelism across chips are the levers, the same as on GPUs.

- **Scale-out is Ethernet.** Chips form meshes over many 100 GbE links (16 per Wormhole chip).
  Collectives are implemented by the software stack over those links, and their cost obeys the same
  ring arithmetic as [@ch:sharding] with $W$ the per-link bandwidth times the number of links
  used in parallel.

- **Static scheduling helps.** Shapes known ahead of time let the stack precompute NoC routes and CB
  sizes. As on TPUs, serving uses bucketed sequence lengths and a fixed KV-cache layout.
:::

### Many chips: meshes over Ethernet

Wormhole chips have no NVLink-style switch. Each chip exposes many 100 GbE ports, and a multi-chip
system wires them into a line, ring or 2-D mesh (the n300 board connects its two chips directly; a
TT-QuietBox connects four boards; a Galaxy connects 32 chips as a mesh). The software presents the
chips as a *mesh device*, and TT-NN provides collective operations (all-gather, reduce-scatter,
all-reduce) implemented on dedicated Ethernet cores. Tensors are distributed over the mesh by *mesh
mappers*, either replicated or sharded along a dimension, exactly the DTensor placements of
[@sec:sharding:fsdp2].

::: napkinmath
[Tensor-parallel decode on an 8-chip workstation]{.box-title}

Shard an 8B model over 8 Wormhole chips with tensor parallelism and 8-bit weights: about 1 GB of
weights per chip, streamed at 288 GB/s, gives a bandwidth ceiling of
$288\times10^9/1.02\times10^9\approx280$ steps/s if communication were free (the aggregate-bandwidth
view: $8\times288$ GB/s over 8.2 GB). Each decode step needs two all-reduces per layer of a
$d_{\text{model}}=4096$ activation, 8 KB in bf16, $36\times2=72$ in total. These messages are tiny,
so they are *latency*-bound. At, say, $10\,\mathrm{\micro\second}$ per hop and a ring of 8, one
all-reduce costs $\approx 2\cdot7\cdot10\,\mathrm{\micro\second}=140\,\mathrm{\micro\second}$, and
72 of them cost 10 ms per token. Added to the $\approx3.6\,\mathrm{\milli\second}$ of weight
streaming, that cuts the ceiling from $\sim$`<!-- -->`{=html}280 to roughly 70 tokens/s. The lesson
carries over to every multi-chip decode: per-token collective *latency*, not bandwidth, decides how
far TP pays off, which is why decode stacks fuse all-reduces with neighbouring matmuls, use
lower-latency algorithms (trees, two-shot on 2-D meshes), or prefer fewer, larger TP groups plus
data parallelism across requests. (The $10\,\mathrm{\micro\second}$ is a placeholder: measure your
own system's per-hop latency.)
:::

### Getting started on a Tenstorrent machine

The software is open source, and the bring-up path is well trodden. In outline (names as in the
public repositories around 2025; the installation guides are authoritative):

1.  **Check the hardware.** Install the system tools and run `tt-smi` to list cards, firmware
    versions, temperatures and the chip topology. Use it to reset a card after a crashed program.

2.  **Install TT-Metalium / TT-NN** from a release wheel or by building the `tt-metal` repository.
    Run its programming examples (hello-world kernels, an eltwise add, a single-core and a
    multi-core matmul) before anything else, because they exercise exactly the reader/compute/writer
    structure of [@fig:accel:rcw].

3.  **Run a model demo.** The repository ships TT-NN implementations of popular open models
    (Llama-family and Qwen-family LLMs among them) with performance targets per device type. Run one
    on your card or mesh to get a known-good baseline for tokens/s and time-to-first-token.

4.  **Serve.** For an OpenAI-compatible endpoint, Tenstorrent maintains an inference server and a
    vLLM integration; the serving concepts of [@ch:serving] (continuous batching, paged KV
    cache) carry over directly.

5.  **Profile.** The SDK includes a device profiler that records per-core, per-RISC timestamps and a
    Tracy-based host profiler. Look first at whether data-movement or compute kernels dominate each
    op, which is the Tensix analogue of memory- versus compute-bound.

### Bringing up a transformer block in TT-NN

Porting our from-scratch Qwen3 block ([@ch:modern]) is mostly a mapping exercise.
[@tab:accel:ttnn] lists the correspondence. The work that does not appear in the table is
the *memory planning*: which weights are DRAM-interleaved and which are sharded into SRAM, which
activations stay in SRAM between ops, and the core grid for each matmul.

::: {#tab:accel:ttnn .table-float}
| `llmsys` (PyTorch) | TT-NN counterpart (names indicative; check the docs) |
|:---|:---|
| `nn.Embedding` lookup | `ttnn.embedding` on a row-major weight |
| `RMSNorm` | `ttnn.rms_norm` (fused, tile layout) |
| `W_query/W_key/W_value` | one fused QKV `ttnn.linear` (weights concatenated offline) |
| QK-norm + RoPE | `ttnn.rms_norm` per head + a rotary-embedding op with precomputed cos/sin |
| causal attention / KV cache | transformer SDPA ops (prefill and decode variants) over a paged or static cache |
| `SwiGLU` | two `ttnn.linear`s, `ttnn.silu`, `ttnn.mul`, one `ttnn.linear` |
| residual add | `ttnn.add` (often fused into the next op's input) |
| TP across chips | column/row-sharded weights on a mesh device + CCL all-gather / reduce-scatter |

: Mapping the Qwen3 block of [@ch:modern] onto TT-NN operations.
:::

::: pitfall
[Validate numerics before optimising]{.box-title}

Bring a port up in three passes: (1) run every op on device but compare each block's output with the
PyTorch reference block by block (Pearson correlation above 0.99 per layer is a common acceptance
bar), using bf16 everywhere; (2) switch weights to block-fp8 and re-check the end-to-end perplexity
and a few generations; (3) only then tune core grids and memory placement. A wrong tile layout or a
forgotten untilize yields plausible-looking garbage, not an error.
:::

## Wafer-scale and SRAM-only designs {#sec:accel:sram}

Two companies push the "keep weights on chip" idea to its limit.

##### Cerebras.

The Wafer-Scale Engine uses an entire silicon wafer as one chip. The third generation (WSE-3) is
described as having on the order of 900,000 cores and 44 GB of on-chip SRAM with petabytes per
second of aggregate memory bandwidth, connected by an on-wafer 2-D mesh. For inference the weights
of a model are spread over the SRAM of one or more wafers, which removes off-chip bandwidth from the
decode loop, the bottleneck of every design in [@fig:accel:roofline]. For training very
large models Cerebras describes a "weight streaming" mode in which weights live in an external
memory system and are streamed through the wafer layer by layer, with activations kept on the wafer.

##### Groq.

The LPU is a deterministic, compiler-scheduled processor with only a few hundred megabytes of SRAM
per chip and no caches; the compiler schedules every data movement and computation cycle by cycle,
including across chips. A large model is therefore spread over hundreds of chips connected directly
to one another. The payoff is very low and predictable per-token latency, because there is no DRAM
and no dynamic scheduling jitter. The cost is the number of chips needed to hold the weights, which
makes it a capacity-for-latency trade.

::: napkinmath
[How many SRAM chips for a 70B model?]{.box-title}

At 8 bits a 70B model needs 70 GB of weights. With about 230 MB of SRAM per chip (the figure Groq
has published for its first-generation chip) that is $70\times10^9/2.3\times10^8\approx 300$ chips
before any KV cache, against one or two 80 GB GPUs. On a 44 GB wafer it is two wafers. The
architectures differ less in *whether* data movement costs than in *where* they choose to pay for
it: off-chip bandwidth (GPU, TPU, Tensix with DRAM) or chip count and inter-chip links (SRAM-only).
:::

## Unified-memory machines: GB10 and friends {#sec:accel:gb10}

NVIDIA's GB10 (sold as DGX Spark) pairs an Arm CPU with a Blackwell-generation GPU sharing 128 GB of
coherent LPDDR5X memory at about 273 GB/s. The headline compute figure is around one petaFLOP at FP4
with sparsity. Apple-silicon machines and AMD's Strix Halo class share the design point: *capacity
first*.

For LLM work this is an unusual roofline ([@fig:accel:roofline]). The ridge point is very
high, thousands of FLOPs per byte, because compute is plentiful and bandwidth is modest. Prefill and
fine-tuning with large batches use the compute well. Single-stream decode of a large model is
limited to $273\,\text{GB/s}/\text{bytes per token}$, single-digit tokens per second for a 4-bit 70B
model. What these machines do uniquely well is *fit*: a 4-bit 70B model with a long KV cache, or
LoRA/QLoRA fine-tuning of 30B-class models ([@ch:instruct]), fits in one box with no model
parallelism at all. Two units can be linked over a high-speed NIC for more capacity, with the usual
collective arithmetic over that link. For the labs in this book, a GB10 is an excellent development
machine: everything up to Qwen3-8B in bf16 fits with room for activations, and the CUDA software
stack is the same as on a datacenter GPU.

::: {#tab:accel:gb10 .table-float}
|            |        |      |      |      |            |                             |
|:-----------|-------:|-----:|-----:|-----:|-----------:|----------------------------:|
| model      | params | bf16 |  fp8 |  fp4 |   KV @ 32k | GB10 decode ceiling (tok/s) |
|            |        | (GB) | (GB) | (GB) | (GB, bf16) |            bf16 / fp8 / fp4 |
| Qwen3-0.6B |  0.60B |  1.2 |  0.6 |  0.3 |        3.8 |             229 / 458 / 916 |
| Qwen3-1.7B |  1.72B |  3.4 |  1.7 |  0.9 |        3.8 |              79 / 159 / 317 |
| Qwen3-4B   |  4.02B |  8.0 |  4.0 |  2.0 |        4.8 |               34 / 68 / 136 |
| Qwen3-8B   |  8.19B | 16.4 |  8.2 |  4.1 |        4.8 |                17 / 33 / 67 |
| Qwen3-14B  |  14.8B | 29.5 | 14.8 |  7.4 |        5.4 |                 9 / 18 / 37 |
| Qwen3-32B  |  32.8B | 65.5 | 32.8 | 16.4 |        8.6 |                  4 / 8 / 17 |

: Planning a 128 GB, 273 GB/s unified-memory machine (computed with `llmsys.flops` and
`decode_tokens_per_s_bound`; weights only in the ceiling). Everything fits, even bf16 32B with a
long cache, but single-stream speed falls with size exactly as bandwidth predicts. Batching raises
total throughput until compute or cache capacity binds.
:::

## Interconnects compared {#sec:accel:interconnect}

Chips differ less in how they multiply matrices than in how they talk to each other.
[@fig:accel:topologies] contrasts the three topologies of this appendix, and the cost model
of [@ch:sharding] turns each into a statement about which parallelism fits.

![Three interconnect topologies. A switch gives uniform all-to-all bandwidth inside a domain; a
torus gives uniform neighbour bandwidth over thousands of chips; an Ethernet mesh uses commodity
links bundled per edge.](figures/appB_accelerators-11.svg){#fig:accel:topologies}

- **Switched domains** make every collective cheap inside the domain and make all-to-all (expert
  parallelism, [@sec:dist:ep]) natural. Leaving the domain costs an order of magnitude in
  bandwidth, so the layout puts TP and EP inside and DP/FSDP outside.

- **Tori** make *ring* collectives along an axis cheap at enormous scale. All-to-all must hop across
  many links, so MoE routing is relatively more expensive, and layouts favour sharding patterns that
  decompose into per-axis rings.

- **Ethernet meshes** trade per-link bandwidth and latency for cost and flexibility. Bandwidth comes
  from bundling many links per neighbour, and small-message latency matters for decode (the napkin
  math above).

## Other accelerators in brief {#sec:accel:others}

Several other designs are in production use for LLMs. One paragraph each, with the same four
questions in mind.

::: description
(MI300X and successors) is a GPU in the same SIMT family as NVIDIA's, with 64-wide wavefronts,
matrix cores and very large HBM capacity (192 GB at about 5.3 TB/s on MI300X). The software stack is
ROCm/HIP. PyTorch, Triton and vLLM support it, so the methods of this book apply almost unchanged.
Large capacity per GPU shifts the balance toward less model parallelism for inference.

chips contain NeuronCores built around systolic tensor engines plus vector, scalar and
general-purpose engines, with software-managed on-chip memory. Programs are compiled by the Neuron
compiler from PyTorch/JAX graphs, and custom kernels are written in NKI, a Python tile-level kernel
language, so the TPU discussion of static shapes and compiler-managed memory applies.

combines matrix engines with programmable tensor processor cores and integrates RoCE Ethernet ports
on chip for scale-out, which places it close to the Ethernet-mesh philosophy of Wormhole at the
interconnect level.

and other client SoCs share the unified-memory trade-offs of GB10: large capacity, moderate
bandwidth. Frameworks such as MLX and llama.cpp target them for local inference and fine-tuning.
:::

## Case study: one Qwen3-8B decode step on four machines {#sec:accel:case}

Putting the pieces together, [@tab:accel:case] applies the decode roofline to Qwen3-8B with
a 4096-token context on four machines, at batch 1 and batch 64. Weights are 2 bytes on H100 and TPU
and 1 byte (fp8 or block-fp8) on Wormhole and GB10, and the KV cache is bf16 ($147$ KB per token for
this model's 8 KV heads $\times$ 128 dims $\times$ 36 layers). Each step must read all weights once
plus every sequence's cache. Compute is $2\Psi$ FLOPs per token plus attention.

::: {#tab:accel:case .table-float}
| machine       | batch        | weights |      KV | memory time | compute time | tokens/s bound |
|:--------------|:-------------|--------:|--------:|------------:|-------------:|---------------:|
| H100 SXM      | 1            | 16.4 GB |  0.6 GB |     5.07 ms |      0.02 ms |            197 |
|               | 64           | 16.4 GB | 38.7 GB |     16.4 ms |      1.14 ms |          3,900 |
| TPU v5p       | 1            | 16.4 GB |  0.6 GB |     6.15 ms |      0.04 ms |            162 |
|               | 64           | 16.4 GB | 38.7 GB |     19.9 ms |      2.45 ms |          3,200 |
| Wormhole n300 | 1            |  8.2 GB |  0.6 GB |     15.3 ms |      0.04 ms |             65 |
|               | 24$^\dagger$ |  8.2 GB | 14.5 GB |     39.4 ms |       0.9 ms |            610 |
| GB10          | 1            |  8.2 GB |  0.6 GB |     32.2 ms |      0.02 ms |             31 |
|               | 64           |  8.2 GB | 38.7 GB |      172 ms |      1.12 ms |            370 |

: Roofline bounds for one decode step of Qwen3-8B at 4096 tokens of context (computed with
`llmsys.flops` and the `ACCELERATORS` table). $^\dagger$Batch 64 does not fit in the n300's 24 GB;
about 24 sequences do. Real systems reach a fraction of these bounds, but the ratios hold.
:::

Four observations generalise:

1.  **Every machine is memory-bound in decode**, by one to two orders of magnitude at batch 1 and
    still by an order of magnitude at batch 64. Peak FLOP/s barely matters for decode; bytes per
    second and bytes per token do.

2.  **At large batch the KV cache, not the weights, dominates traffic**: 38.7 GB of cache against
    16.4 GB of weights at batch 64. This is why GQA ([@ch:modern]), KV-cache quantisation
    ([@ch:efficient]) and paged caches ([@ch:serving]) matter as much as weight
    quantisation.

3.  **Capacity caps batch, and batch is throughput.** The n300 cannot hold 64 sequences of 4096
    tokens, so its throughput bound is set by how many sequences fit, not by its bandwidth alone.

4.  **Cheaper bytes per token beat faster chips**: halving weight bytes (fp8) doubles the batch-1
    ceiling on every machine, the same factor as doubling HBM bandwidth.

::: {#tab:accel:formats .table-float}
| format | what it is | where accelerated (as publicly documented) |
|:---|:---|:---|
| bf16 | 8-bit exponent, 7-bit mantissa | all machines in this appendix |
| fp8 (E4M3/E5M2) | per-tensor or per-block scaled 8-bit floats | H100 and later GPUs, MI300-class, newer TPUs |
| block fp8/fp4 ("bfp8_b", "bfp4_b") | groups of values within a tile share one exponent | Tenstorrent Tensix |
| MX formats (MXFP8/6/4) | OCP microscaling: 32-value blocks with a shared power-of-two scale | Blackwell-generation GPUs and others adopting the OCP spec |
| int8 / int4 | integer weights (and activations) with scales | widely; int4 mostly as weight-only with dequantisation |

: Low-precision formats you will meet when porting (see [@ch:efficient] for the quantisation
methods). Support varies by generation; check before relying on a format.
:::

## A porting checklist {#sec:accel:checklist}

::: {#tab:accel:port .table-float}
|  | GPU | TPU | Tenstorrent |
|:---|:---|:---|:---|
| kernel language | CUDA, Triton | XLA (auto), Pallas | TT-Metalium kernels, TT-NN ops |
| on-chip memory | caches + shared memory | compiler-managed VMEM | explicitly managed SRAM per core |
| shapes | dynamic OK (graphs help) | static; bucket lengths | static preferred; tile multiples of 32 |
| parallelism API | FSDP/DTensor, NCCL | mesh + sharding annotations | mesh device + TT-NN multi-device ops |
| best decode lever | batching, quantisation | batching, larger $M$ per MXU | SRAM-resident weights, block-fp8 |

: What changes for the programmer.
:::

Porting an LLM workload to a new accelerator follows the same steps every time.

1.  **Compute the rooflines** of prefill and decode on the target ([@sec:accel:roofline]).
    They tell you which of compute, off-chip bandwidth or capacity will bind.

2.  **Fix the shapes**: pick padding multiples for the matrix engine (128, 256 or 32) and
    sequence-length buckets; check vocabulary, head and MLP sizes for padding waste
    (`tiled_systolic_cycles` reports padding efficiency).

3.  **Choose number formats** the hardware accelerates (bf16, fp8, block-fp8, int8, fp4) and
    validate accuracy as in [@ch:efficient].

4.  **Map the parallelism** onto the physical topology: TP inside the fastest domain, FSDP/DP across
    the slower one, collectives on the axes with the most links ([@ch:sharding]).

5.  **Measure, then specialise**: profile with the vendor's tools and write custom kernels only for
    the operations that dominate (usually attention and fused norms).

## Using non-GPU hardware with this book's labs {#sec:accel:labs}

The labs are written against PyTorch on CPU or CUDA, but most of their *measurement* sections can be
repeated on other hardware, and doing so is the best way to make this appendix concrete.

- **GB10 / DGX Spark.** It runs the CUDA stack, so every lab runs unchanged. Use it for the
  long-context and large-model variants (Qwen3-8B and 14B inference in Lab 26, LoRA/QLoRA in Lab 25,
  GRPO on 1.7B in Lab 37), and use the planning numbers of [@tab:accel:gb10] to predict
  decode speed before measuring it.

- **Tenstorrent (n150/n300, TT-QuietBox).** Repeat the decode-roofline experiment of Lab 26 with a
  TT-NN model demo: measure batch-1 tokens/s with DRAM-resident weights, then with block-fp8
  weights, and compare with `decode_tokens_per_s_bound`. Port the Qwen3 block of Lab 6 op by op
  using [@tab:accel:ttnn], checking per-block correlation against the PyTorch reference. On
  a multi-chip box, measure an all-gather's time against message size, as in
  [@lst:sharding:bench], to obtain your mesh's $\alpha$ and $\beta$ for the TP napkin math
  of [@sec:accel:tt].

- **TPU VMs.** Re-implement the Lab 12 training loop in JAX with the sharding annotations of
  [@lst:accel:jax] and measure MXU utilisation against batch size to reproduce
  [@fig:accel:util]. The Lab 18 accounting functions apply with $W$ set to the ICI
  bandwidth of the axis you shard over.

In each case the method is the book's: predict with a cost model, measure, then explain the gap. The
gap usually comes from fixed overheads (kernel launch or dispatch, collective latency, format
conversion such as tilize/untilize) that the roofline leaves out. Measuring them on your own
hardware is what turns these models into engineering tools.

## Summary {#sec:accel:summary}

- Every accelerator is compute units, on-chip memory, off-chip memory and an interconnect; designs
  differ in who schedules data movement (hardware, compiler or programmer) and in how chips are
  connected.

- Systolic arrays build operand reuse into the dataflow. Utilisation is $M/(M+R+C-2)$ per tile, so
  decode (small $M$) wastes the array and shapes should be multiples of the array size.

- TPUs pair systolic MXUs with compiler-managed memory and a torus interconnect. Programs are
  compiled whole for static shapes, and parallelism is expressed as sharding annotations on a mesh.

- Tenstorrent's Tensix mesh exposes five RISC-V processors per core, explicit SRAM and NoC data
  movement through reader/compute/writer kernels, $32\times32$ tiles, and Ethernet scale-out.
  Placement is the program, and per-token collective latency limits multi-chip decode.

- SRAM-only and wafer-scale designs remove off-chip bandwidth from decode at the cost of chip count;
  unified-memory machines such as GB10 buy capacity, not bandwidth.

- Decode is memory-bound everywhere. Bytes per token (weights plus KV cache) and the batch that fits
  are the levers on every machine.

## Exercises {#sec:accel:exercises}

::: exercise
[Utilisation formula]{.box-title}

Derive the cycle count $M+K+N-2$ of the weight-stationary array from the injection schedule in
[@sec:accel:systolic]. How does it change if weights are loaded one row per cycle and
loading is not overlapped? Verify both with `simulate_weight_stationary`.
:::

::: exercise
[Batching on an MXU]{.box-title}

Using `tiled_systolic_cycles`, find the smallest decode batch that reaches 50% MXU utilisation for
Qwen3-8B's MLP up-projection ($4096\times12288$) on a $128\times128$ array and on a $256\times256$
array. What does this imply for the batch sizes TPU serving systems target?
:::

::: exercise
[Padding waste]{.box-title}

Compute the padding efficiency of every weight matrix of Qwen3-0.6B (vocabulary 151,936) and of
GPT-2 small (vocabulary 50,257) on a $128\times128$ array and with $32\times32$ tiles (`report.py`
prints the table). Why does Qwen3 waste nothing? GPT-2-reproduction codebases pad the vocabulary to
50,304; check that this fixes both layouts, and explain why the padded rows cost nothing at
inference.
:::

::: exercise
[Torus all-reduce]{.box-title}

On a $16\times16$ 2-D torus with bidirectional links of bandwidth $W$ each way, derive the time of
an all-reduce of $S$ bytes performed as reduce-scatter along $x$, all-reduce along $y$, then
all-gather along $x$. Compare with a single bidirectional ring over all 256 chips.
:::

::: exercise
[Cannon vs SUMMA]{.box-title}

Run `cannon_matmul` and `summa_traffic` for $p\in\{2,4,8\}$ with square matrices. Which moves fewer
link-bytes, and why? When would you still prefer Cannon's algorithm on a real NoC (think about link
contention and multicast support)?
:::

::: exercise
[Block size on a core grid]{.box-title}

A core sustains $C_c=4$ TFLOP/s on 8-bit tiles, and a NoC link moves 64 GB/s. What is the smallest
block edge $b$ for which a Cannon step is compute-bound? Does a double-buffered pair of $b\times b$
8-bit input blocks plus an fp32 accumulator fit in 1.5 MB?
:::

::: exercise
[SRAM-resident decode]{.box-title}

A 300M-parameter model in 8-bit weights is served on an n300 card with weights resident in SRAM. If
on-chip bandwidth were unlimited, what would bound tokens/s? Estimate it from the card's 8-bit
compute figure, and compare with the DRAM-bound estimate if the weights lived in GDDR6.
:::

::: exercise
[Choosing hardware]{.box-title}

For each workload, which machine in [@fig:accel:roofline] is the best fit and why: (a)
serving an 8B chat model to 1000 concurrent users; (b) one developer running a 70B model locally;
(c) GRPO training of a 1.7B model; (d) a latency-critical single-user 8B assistant.
:::

::: exercise
[KV versus weights]{.box-title}

Using the numbers of [@tab:accel:case], find the batch size at which Qwen3-8B's KV-cache
traffic equals its weight traffic at 4096 tokens of context, for bf16 weights and for fp8 weights.
How does quantising the KV cache to 8 bits move it?
:::

::: exercise
[Latency-bound TP]{.box-title}

Redo the 8-chip tensor-parallel napkin math of [@sec:accel:tt] for a per-hop latency of
$2\,\mathrm{\micro\second}$ and $20\,\mathrm{\micro\second}$, and for TP degrees 2, 4 and 8. At
which TP degree does adding chips stop increasing single-stream tokens/s? Propose a layout for
serving many users.
:::

## Lab B: Accelerator models {#sec:accel:lab}

::: labbox
[Systolic arrays, tile layouts and meshes of cores]{.box-title}

**Code:** `code/labs/labB_accel`

**Goals**

Build the cycle-level and traffic models of this appendix and use them to reason about LLM workloads
on non-GPU hardware. Everything is NumPy and runs anywhere.

**Setup**

``` bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/labB_accel
LAB_IMPL=solution python -m pytest -q labs/labB_accel
python labs/labB_accel/report.py          # tables for the exercises
```

**Tasks**

1.  `simulate_output_stationary` and `simulate_weight_stationary`: register-level simulations that
    must reproduce `A @ B` exactly and report the cycle count.

2.  `tiled_systolic_cycles`: the tiling, fill/drain and padding model, checked against the
    simulators on small cases.

3.  `tilize` / `untilize`: the $32\times32$-tile, $16\times16$-face layout, as an exact round trip
    with padding.

4.  `cannon_matmul`: block matmul on a $p\times p$ torus with link-traffic accounting, and
    `summa_traffic` for comparison.

5.  `decode_tokens_per_s_bound` and `ridge_point` for the accelerator table.

**Measurements**

- Plot utilisation against $M$ for $128^2$ and $256^2$ arrays and mark typical decode and prefill
  sizes.

- Tabulate padding efficiency for the matrices of Qwen3-0.6B and GPT-2.

- If you have access to a Tenstorrent card: run a TT-NN matmul at several shapes with DRAM and with
  L1 memory configurations, and compare the achieved TFLOP/s with the roofline. On a TPU VM: time
  `jax.jit` matmuls with $M\in\{8,\dots,8192\}$ and compare with [@fig:accel:util].

**Deliverables**

Passing tests; the two tables; one paragraph on which non-GPU design would best serve your own
research workload, justified with the models.

**Stretch goals**

Add an input-stationary dataflow; simulate SUMMA with explicit per-link contention; extend the mesh
model with a DRAM bank per edge core and compute the time to load a weight-stationary block layout.
:::
