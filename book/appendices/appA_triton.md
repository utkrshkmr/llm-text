# GPU Kernels with Triton {#app:triton}

Throughout the book we have argued with rooflines: decode is memory-bound, softmax is memory-bound,
attention without tiling moves $O(T^2)$ bytes, and a weight-only int4 GEMM is fast only if
dequantization happens *inside* the kernel. This appendix makes those arguments executable. We write
the kernels ourselves in Triton (Tillet et al., 2019), a Python-embedded language that compiles
block-level programs to GPU code. Triton is what `torch.compile`'s Inductor backend emits for most
fused operations, so reading Triton is also how you inspect what the compiler did to your model.

Every kernel here has a PyTorch reference in `llmsys/kernels/reference.py`. The Triton versions live
in `llmsys/kernels/triton_kernels.py`. Triton ships with an *interpreter*: set `TRITON_INTERPRET=1`
before importing `triton`, and kernels run on the CPU with NumPy semantics. It is slow, but it means
correctness can be tested on a laptop, and it is how this appendix's lab is checked.

![Roadmap of this appendix: from the programming model to kernels that implement
[@ch:attention]'s FlashAttention and [@ch:efficient]'s quantized
GEMM.](figures/appA_triton-1.svg){#fig:triton:roadmap}

## Why write a kernel at all?

A PyTorch eager program launches one kernel per operator. Each kernel reads its inputs from HBM and
writes its outputs back to HBM. For matmuls this is fine, since their arithmetic intensity is high.
For chains of elementwise and reduction operations (softmax, norms, activation functions,
dequantization, rotary embeddings) the intermediate tensors make round trips through memory, and the
chain runs at a small fraction of what the hardware could do. A *fused* kernel loads each input
once, keeps intermediates in registers or shared memory, and writes the result once.

::: keyidea
Write a custom kernel when (1) a sequence of memory-bound operations can be fused, (2) a
computation's working set can be *tiled* into on-chip memory to change its asymptotic memory traffic
(FlashAttention), or (3) a data format is not supported by library kernels (packed int4 weights,
custom KV layouts). Otherwise, cuBLAS, cuDNN and `torch.compile` usually beat a hand-written kernel.
:::

## The Triton programming model

### From CUDA threads to Triton programs

CUDA exposes a three-level hierarchy. A *grid* of thread blocks; each block is scheduled onto one
streaming multiprocessor (SM) and contains up to 1024 threads grouped into 32-wide warps; each
thread owns registers, and threads in a block share up to a few hundred KB of shared memory (SMEM).
The programmer reasons about individual threads: which element each one loads, how they cooperate
through SMEM, and how to synchronise.

Triton removes the thread level. You write a *program* that operates on whole *blocks* (tiles) of
data as if they were small arrays, and launch a grid of program instances. The compiler decides how
to spread each block across threads and warps, when to stage data through shared memory, how to
vectorise memory accesses, and how to issue tensor-core instructions for `tl.dot`.
[@fig:triton:hierarchy] contrasts the two views.

![CUDA's thread-level view versus Triton's block-level view. A Triton program corresponds to a CUDA
thread block. Inside it you write NumPy-like operations on tiles, with `num_warps` controlling how
many warps execute it.](figures/appA_triton-2.svg){#fig:triton:hierarchy}

### The memory hierarchy a kernel sees

Kernel design is mostly about *where data lives*. For an H100 SXM:

::: center
| level     | capacity                                            | scope        | managed by     |
|:----------|:----------------------------------------------------|:-------------|:---------------|
| registers | 256 KB per SM (64k $\times$ 32-bit)                 | one thread   | compiler       |
| SMEM / L1 | 256 KB per SM combined, up to 228 KB usable as SMEM | thread block | you / compiler |
| L2 cache  | 50 MB                                               | whole GPU    | hardware       |
| HBM3      | 80 GB at 3.35 TB/s                                  | whole GPU    | you            |
:::

There are 132 SMs. Each step down the hierarchy is roughly an order of magnitude more capacity and
an order of magnitude less bandwidth per byte. Tiles must therefore be sized so that their working
set (inputs, accumulators, pipeline buffers) fits in registers and SMEM, while still leaving enough
room for several programs per SM to hide latency (*occupancy*). In Triton you control this through
block sizes, `num_warps` and `num_stages`. Exceed the register budget and the compiler spills to
local memory, which lives in HBM, and performance collapses ([@fig:triton:memory]).

![The memory hierarchy as seen by one kernel. Triton tiles live in registers and SMEM. Register
spills silently fall all the way to HBM.](figures/appA_triton-3.svg){#fig:triton:memory}

### A first kernel: vector addition

[@lst:triton:add] adds two vectors. The `@triton.jit` kernel runs once per program
instance:

- `tl.program_id(0)` tells the instance which block it owns;

- `tl.arange(0, BLOCK)` creates a vector of offsets within the block;

- `mask` guards the ragged last block ([@fig:triton:add]);

- pointers are integers plus offsets, so `x_ptr + offsets` is a block of addresses and `tl.load`
  gathers them.

`BLOCK` is a `tl.constexpr`: it is known at compile time (and must be a power of two), so the
compiler can specialise register allocation and vector widths. A new value triggers a recompilation,
which is cached.

``` {#lst:triton:add .python caption="Vector addition in Triton, with its launcher."}
@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # which block am I?
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # BLOCK element indices
    mask = offsets < n_elements                  # guard the ragged tail
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def vector_add(x, y, block=1024):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block),)              # number of program instances
    _add_kernel[grid](x, y, out, n, BLOCK=block)
    return out
```

![Each program instance handles one contiguous block of `BLOCK` elements. The last block is
partially out of bounds and is masked: masked loads return `other` (default undefined, so always
pass it when the value matters), and masked stores are
dropped.](figures/appA_triton-4.svg){#fig:triton:add}

### Debugging with the interpreter

Kernels compile lazily on first launch, with arguments' dtypes and `constexpr` values as the cache
key. Errors at that point tend to be cryptic. Two habits help. First, develop with
`TRITON_INTERPRET=1`: kernels then execute eagerly in Python, so you can `print` a tile or drop into
a debugger inside the kernel, on a machine without a GPU. Second, test against a PyTorch reference
on awkward shapes (prime sizes, sizes smaller than one block, non-contiguous strides) before
measuring speed.

``` bash
TRITON_INTERPRET=1 python -m pytest -q labs/labA_triton     # correctness, CPU
python labs/labA_triton/bench.py --hw H100-SXM               # speed, GPU
```

::: pitfall
[The interpreter flag must be set first]{.box-title}

`TRITON_INTERPRET` is read when `@triton.jit` decorates a function, so it must be in the environment
*before* the module defining your kernels is imported. Our lab's test file sets it at the top when
no GPU is present.
:::

## Fused softmax and RMSNorm {#sec:triton:softmax}

A row-wise softmax in eager PyTorch, written out (`x - max`, `exp`, `sum`, divide), launches several
kernels, each streaming the whole matrix through HBM. [@lst:triton:softmax] assigns one
program per row, loads the row once, performs every step in registers, and writes the result once.

``` {#lst:triton:softmax .python caption="Fused row softmax: one program per row."}
@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, x_row_stride, out_row_stride,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    z = x - tl.max(x, axis=0)             # padding is -inf, so it never wins the max
    num = tl.exp(z)                       # exp(-inf) = 0: padding adds nothing
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, num / den, mask=mask)
```

Note how the padding value is chosen so that masked lanes are neutral for both reductions: $-\infty$
for the max, and therefore $e^{-\infty}=0$ for the sum.

![Memory traffic of an unfused versus a fused softmax. Red arrows are reads from HBM, green arrows
are writes. For a memory-bound operation, run time is proportional to the
arrows.](figures/appA_triton-5.svg){#fig:triton:softmax}

::: napkinmath
[Softmax over a 4096 $\times$ 8192 fp32 matrix on an H100]{.box-title}

The matrix is 134 MB. Fused: one read plus one write, 268 MB, so $\ge 80\,\mu$s at 3.35 TB/s. The
five-kernel eager version moves roughly $9\times134$ MB $\approx1.2$ GB, so $\ge360\,\mu$s.
PyTorch's `torch.softmax` is itself a fused kernel; our Triton kernel should match it, and the lab's
benchmark reports achieved GB/s against the 3.35 TB/s peak. A good fused kernel reaches 80--90%.
:::

The one-row-per-program design assumes a row fits in registers. Beyond roughly 32k fp32 columns,
register pressure forces spills, and the kernel must loop over chunks of the row. It then either
does two passes (max, then sum) or keeps *running* statistics with the online-softmax rescaling of
[@sec:triton:flash]. RMSNorm ([@ch:modern]) has the same shape: one reduction (the
mean of squares) followed by an elementwise scale. The lab implements it exactly like softmax.

### A second fusion: the SwiGLU activation

Qwen3's MLP computes
$\mathrm{down}\big(\operatorname{SiLU}(\mathrm{gate}(x))\odot\mathrm{up}(x)\big)$
([@ch:modern]). The two projections are GEMMs, but between them sit three memory-bound steps:
the SiLU, the elementwise product, and the dtype handling. Fusing them into one pass is the simplest
useful kernel there is ([@lst:triton:swiglu]).

``` {#lst:triton:swiglu .python caption="Fused SiLU-and-multiply (`llmsys/kernels/triton_kernels.py`)."}
@triton.jit
def _swiglu_kernel(g_ptr, u_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu = g / (1.0 + tl.exp(-g))                 # g * sigmoid(g)
    tl.store(out_ptr + offs, silu * u, mask=mask)
```

::: napkinmath
[SwiGLU fusion during prefill]{.box-title}

Take Qwen3-8B ($d_{\text{ff}}=12288$) and a prefill chunk of 8192 tokens in bf16. Each of `gate` and
`up` is $8192\cdot12288\cdot2$ B$=201$ MB. Unfused (`silu`, then `mul`) reads 201 MB and writes
201 MB, then reads 402 MB and writes 201 MB: about 1.0 GB, or 300 $\mu$s at 3.35 TB/s, per layer.
Fused, it reads 402 MB and writes 201 MB: 603 MB, or 180 $\mu$s. Over 36 layers that saves about
4 ms per chunk. An even better option is to fuse the activation into the *epilogue* of the gate/up
GEMM, so the 201 MB intermediates never reach HBM at all.
:::

### A backward kernel: RMSNorm

Training needs gradients, and fused forward kernels need fused backward kernels. For one row, let
$r=(\frac1n\sum_j x_j^2+\epsilon)^{-1/2}$, $\hat x=r\,x$ and $y=w\odot\hat x$. Given the upstream
gradient $\boldsymbol{dy}$, define $\boldsymbol{g}=\boldsymbol{dy}\odot w$. Then $$\begin{equation}
  \frac{\partial \mathcal L}{\partial x} \;=\; r\Big(\boldsymbol{g} - \hat x\,\frac{1}{n}\sum_j g_j\hat x_j\Big),
  \qquad
  \frac{\partial \mathcal L}{\partial w} \;=\; \sum_{\text{rows}} \boldsymbol{dy}\odot\hat x .
  \label{eq:triton:rmsbwd}
\end{equation}$$ The input gradient is again a one-row reduction followed by elementwise work: one
program per row, exactly like the forward. The weight gradient is a reduction *across* rows. The
kernel writes per-row partial products and a second, tiny reduction sums them. Production kernels
write one partial per program (not per row) and use a second Triton kernel or atomics. Note that $r$
is *recomputed* from $x$ rather than saved: recomputing a reduction over data we must load anyway
costs nothing, while storing it costs memory.

``` {#lst:triton:rmsbwd .python caption="Fused RMSNorm backward and its autograd wrapper."}
@triton.jit
def _rmsnorm_bwd_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, gw_ptr, n_cols, stride, eps,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)   # recomputed
    xhat = x * rstd
    g = dy * w
    dx = rstd * (g - xhat * (tl.sum(g * xhat, axis=0) / n_cols))
    tl.store(dx_ptr + row * stride + cols, dx, mask=mask)
    tl.store(gw_ptr + row * stride + cols, dy * xhat, mask=mask)  # dW partials

class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        ctx.save_for_backward(x, weight); ctx.eps = eps
        return rmsnorm(x, weight, eps)
    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx, dw = rmsnorm_backward(x, weight, dy, ctx.eps)
        return dx, dw, None
```

The lab's tests check both gradients against PyTorch autograd, under the interpreter.

::: systemslens
[Fusion is where `torch.compile` earns its keep]{.box-title}

Inductor, `torch.compile`'s default backend, performs exactly this kind of fusion automatically. It
groups pointwise and reduction nodes of the captured graph and emits Triton kernels like
[@lst:triton:softmax]. Run with `TORCH_LOGS=output_code` to see the generated Triton for
your model. Hand-written kernels still win where the compiler cannot see the opportunity, such as
tiling attention, custom quantized formats, or cross-operation algebraic rewrites.
:::

## Tiled matrix multiplication {#sec:triton:matmul}

### Tiling and arithmetic intensity

Computing $C=AB$ with $A\in\mathbb{R}^{M\times K}$, $B\in\mathbb{R}^{K\times N}$, each program
produces one $B_M\times B_N$ tile of $C$. It loops over $K$ in slabs of $B_K$, loading a
$B_M\times B_K$ tile of $A$ and a $B_K\times B_N$ tile of $B$ and accumulating their product in fp32
registers ([@fig:triton:tiling]). Per slab it moves $(B_MB_K+B_KB_N)$ elements and performs
$2B_MB_NB_K$ FLOPs, so its intensity with respect to the loads is $$\begin{equation}
  I_{\text{tile}} = \frac{2B_MB_NB_K}{(B_M+B_N)B_K\cdot b} = \frac{2B_MB_N}{(B_M+B_N)\,b},
\end{equation}$$ where $b$ is bytes per element. With $128\times128$ bf16 tiles,
$I=2\cdot128^2/(256\cdot2)=64$ FLOP/byte. That is below the H100's ridge of $\approx295$ *if those
loads went to HBM*. They do not all go there: tiles of $A$ are reused by every program in the same
row of the output grid, and tiles of $B$ by every program in the same column. The L2 cache serves
most re-reads, which is why program *ordering* matters (next subsection). Larger tiles raise $I$ but
consume more registers and SMEM, lowering occupancy. Autotuning searches this trade-off.

![Tiled GEMM. The program owning the orange tile of $C$ streams the light-blue row panel of $A$ and
the light-green column panel of $B$ through on-chip memory, one $B_K$-wide slab at a time (dark),
accumulating in fp32 registers.](figures/appA_triton-6.svg){#fig:triton:tiling}

### Program ordering for L2 reuse

Programs are launched in increasing `pid` order, and only as many run concurrently as fit on the
SMs. With naive row-major ordering, the concurrently running programs cover one full row of output
tiles. They share one panel of $A$, but each needs a *different* panel of $B$, so the working set is
the whole of $B$. *Grouped* ("swizzled") ordering walks $G$ rows of tiles in column-major order
within each group, so concurrent programs share a few panels of both $A$ and $B$
([@fig:triton:swizzle]). The remapping is pure integer arithmetic on the program id, at the
top of [@lst:triton:matmul].

![Launch order of output tiles. If 9 programs run at once, grouped ordering touches 6 input panels
instead of 8. For large matrices the saving in L2 misses is
substantial.](figures/appA_triton-7.svg){#fig:triton:swizzle}

``` {#lst:triton:matmul .python caption="Grouped, tiled matmul. `tl.dot` compiles to tensor-core MMA instructions on GPUs."}
@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m, num_pid_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
```

### Pipelining and autotuning

Inside the $K$ loop, a naive kernel would load a slab, wait, compute, and load the next. Triton's
`num_stages` parameter software-pipelines the loop: the compiler issues asynchronous copies
(`cp.async` on Ampere, the Tensor Memory Accelerator on Hopper) for slabs $k+1,\dots,k+s-1$ while
the tensor cores work on slab $k$, keeping $s$ buffers in shared memory
([@fig:triton:pipeline]). `num_warps` sets how many warps cooperate on one program. Good
values depend on the GPU, the shapes and the dtype, so production kernels declare a list of
candidate configurations and let `@triton.autotune` benchmark them the first time each `key`
(e.g. $(M,N,K)$) is seen:

``` {#lst:triton:autotune .python caption="Declaring autotuning configurations (GPU only)."}
@triton.autotune(
    configs=[triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8},
                           num_stages=3, num_warps=8),
             triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8},
                           num_stages=4, num_warps=4),
             triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8},
                           num_stages=4, num_warps=4)],
    key=["M", "N", "K"])
@triton.jit
def _matmul_kernel_tuned(...):  # same body as above
    ...
```

![Software pipelining of the $K$ loop. With multiple shared-memory buffers, asynchronous loads of
future slabs overlap tensor-core work on the current one, hiding memory
latency.](figures/appA_triton-8.svg){#fig:triton:pipeline}

::: napkinmath
[How close to cuBLAS should you get?]{.box-title}

A square bf16 GEMM with $n=4096$ performs $2n^3=1.37\cdot10^{11}$ FLOPs and touches
$3n^2\cdot2$ B$=100$ MB, an intensity of $\approx1365$ FLOP/B, which is deep in compute-bound
territory. At 989 TFLOP/s the floor is 139 $\mu$s. cuBLAS typically reaches 70--80% of peak at this
size on Hopper. A tutorial-style Triton kernel built on `tl.dot` reaches a large fraction of that on
Ampere. Matching cuBLAS on Hopper needs warp-specialised kernels with TMA and asynchronous `wgmma`,
which newer Triton versions expose. Do not expect to beat the vendor library on plain GEMMs. Write
GEMMs when you need a *fused prologue or epilogue*, as in the next section.
:::

## A fused W8A16 GEMM {#sec:triton:w8a16}

[@ch:efficient]'s `QuantLinear` dequantized the whole weight in PyTorch before multiplying,
which moves *more* bytes than the bf16 baseline. The fix is to load the int8 weight tile, widen it
in registers, and apply the per-output-channel scale once to the accumulator in the epilogue. The
scale factors out of the $K$ sum because it depends only on the output channel:
$y_{mn}=\sum_k x_{mk}\,s_n q_{nk}=s_n\sum_k x_{mk}q_{nk}$.

``` {#lst:triton:w8a16 .python caption="Weight-only int8 GEMM: only one byte per weight crosses HBM."}
@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, K,
                  stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :] * stride_xk,
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[None, :] * stride_wn + (k0 + rk)[:, None] * stride_wk,
                    mask=(rn[None, :] < N) & ((k0 + rk)[:, None] < K), other=0)
        acc += tl.dot(x.to(tl.float32), w.to(tl.float32))   # widen in registers
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * s[None, :]                                  # epilogue: apply scale
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn, acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))
```

Two remarks for real deployments. First, the weight tile is loaded transposed (`w[k, n]` from a
row-major `(N, K)` array); a production kernel pre-arranges the layout offline so that loads are
contiguous and vectorised. Second, decode has $M=B$ (a handful of rows), so a $B_M=16$ tile wastes
most of each tensor-core instruction. Specialised kernels (Marlin, Machete) use split-$K$
parallelism so that enough programs exist to keep every SM streaming weights. For group-wise int4,
the scale *does* depend on $k$, so it is applied per group inside the loop.

## FlashAttention forward in Triton {#sec:triton:flash}

[@ch:attention] derived FlashAttention: process keys and values in blocks, keep a running row
maximum $m$ and normaliser $\ell$, and rescale the partial output whenever the maximum increases.
The $T\times T$ score matrix then never touches HBM. In Triton the algorithm is short, because tiles
are first-class ([@lst:triton:flash]).

::: {#prop:triton:online .proposition}
\[Online softmax\] Process the scores $s_1,\dots,s_S$ of one query row in blocks. Initialise
$m=-\infty$, $\ell=0$, $\boldsymbol{a}=\boldsymbol{0}$. For each block $\mathcal B$ with values
$\boldsymbol{v}_j$ set $m'=\max(m,\max_{j\in\mathcal B}s_j)$, then
$\ell\leftarrow e^{m-m'}\ell+\sum_{j\in\mathcal B}e^{s_j-m'}$ and
$\boldsymbol{a}\leftarrow e^{m-m'}\boldsymbol{a}+\sum_{j\in\mathcal B}e^{s_j-m'}\boldsymbol{v}_j$,
and finally $m\leftarrow m'$. After the last block,
$\boldsymbol{a}/\ell=\sum_j\operatorname{softmax}(\boldsymbol{s})_j\,\boldsymbol{v}_j$.
:::

::: proof
*Proof.* By induction, after processing a set $\mathcal P$ of blocks the invariants
$\ell=\sum_{j\in\mathcal P}e^{s_j-m}$ and
$\boldsymbol{a}=\sum_{j\in\mathcal P}e^{s_j-m}\boldsymbol{v}_j$ hold with
$m=\max_{j\in\mathcal P}s_j$. Multiplying by $e^{m-m'}$ re-expresses the old sums relative to the
new maximum, and adding the new block's terms extends them to $\mathcal P\cup\mathcal B$. At the
end, $\boldsymbol{a}/\ell=\sum_je^{s_j-m}\boldsymbol{v}_j/\sum_je^{s_j-m}$, and the common factor
$e^{-m}$ cancels. Every exponent is $\le 0$, so nothing overflows. ◻
:::

``` {#lst:triton:flash .python caption="FlashAttention forward kernel: one program per (query block, batch$\\times$head)."}
@triton.jit
def _flash_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, T, S, sm_scale, stride_b, stride_t,
                      CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr, D: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    base = bh * stride_b
    q = tl.load(q_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :],
                mask=offs_m[:, None] < T, other=0.0).to(tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    shift = S - T
    hi = S
    if CAUSAL:                                  # skip blocks entirely in the future
        hi = tl.minimum(S, (pid_m + 1) * BLOCK_M + shift)
    for n0 in range(0, hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + base + offs_n[:, None] * stride_t + offs_d[None, :],
                    mask=offs_n[:, None] < S, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + base + offs_n[:, None] * stride_t + offs_d[None, :],
                    mask=offs_n[:, None] < S, other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        valid = offs_n[None, :] < S
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None] + shift)
        s = tl.where(valid, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)             # rescale old statistics
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new
    tl.store(o_ptr + base + offs_m[:, None] * stride_t + offs_d[None, :],
             acc / l_i[:, None], mask=offs_m[:, None] < T)
```

![Work decomposition of the FlashAttention forward kernel. Each program owns one query block of one
head and streams the key/value blocks to its left. With causal masking, roughly half the blocks are
skipped and only diagonal blocks need masks.](figures/appA_triton-9.svg){#fig:triton:flash}

::: napkinmath
[Why tiling attention matters]{.box-title}

For $T=8192$ and 32 heads, a materialised fp16 score matrix takes $8192^2\cdot32\cdot2$ B$=4.3$ GB
*per layer*. It must be written by one kernel and read by the next (twice more in the backward
pass). The tiled kernel's HBM traffic is only $Q$, $K$, $V$ and $O$,
i.e. $4\cdot8192\cdot32\cdot128\cdot2$ B$=268$ MB, plus re-reads of $K$ and $V$ by each query block
that hit in L2. Attention changes from memory-bound to compute-bound, and its memory footprint from
$O(T^2)$ to $O(T)$.
:::

##### Backward pass.

FlashAttention's backward recomputes the probabilities block by block from $Q$, $K$ and the saved
per-row log-sum-exp $m+\log\ell$ (one float per row), instead of storing $P$. It trades a second
$QK^\top$ for $O(T^2)$ memory. The gradients with respect to $Q$, $K$ and $V$ need a separate pass
(or atomics), because $dK$ and $dV$ accumulate over query blocks.

##### What later versions add.

FlashAttention-2 (Dao, 2023) swaps the loop order to parallelise over the sequence dimension even at
batch 1, reduces non-matmul FLOPs (the rescaling), and partitions work between warps to avoid
shared-memory communication. FlashAttention-3 (Shah et al., 2024) targets Hopper: warp
specialisation (producer warps issue TMA loads, consumer warps issue asynchronous `wgmma`),
ping-pong scheduling that overlaps the softmax of one tile with the matmul of another, and FP8
support. For decode, where $T_q=1$, split-KV variants ("flash-decoding") parallelise over the key
dimension and reduce partial results, because a single query block cannot fill the GPU.

::: pitfall
[Our teaching kernel's limits]{.box-title}

`flash_attention` in `llmsys` requires $T=S$ (prefill), a power-of-two head dimension of at least
16, and contiguous `(B`$\cdot$`H, T, D)` layouts. It accumulates in fp32 but loads whatever dtype it
is given. It does not handle GQA head sharing, paged KV, sliding windows or the backward pass. Use
`F.scaled_dot_product_attention` or the FlashAttention library in production.
:::

## A paged-attention decode kernel {#sec:triton:paged}

[@ch:serving] stores the KV cache in fixed-size physical blocks and gives each sequence a
*block table* that maps logical block $j$ to a physical block id. This eliminates fragmentation, but
it means that attention can no longer assume contiguous keys. The kernel must chase the indirection
itself. Decode is the simplest case: one new query per sequence. [@lst:triton:paged]
assigns one program per (sequence, head), walks the sequence's logical blocks, loads the physical
block id from the table, and applies the online softmax of [@prop:triton:online] block by
block ([@fig:triton:paged]).

![Paged decode attention. The block table translates logical KV blocks into scattered physical
blocks. The kernel performs the gather on the fly, so no contiguous copy of the cache is ever
made.](figures/appA_triton-10.svg){#fig:triton:paged}

``` {#lst:triton:paged .python caption="Decode attention over a paged KV cache (one query token per sequence)."}
@triton.jit
def _paged_decode_kernel(q_ptr, k_cache_ptr, v_cache_ptr, block_table_ptr, seq_lens_ptr,
                         o_ptr, sm_scale, max_blocks, H: tl.constexpr,
                         BLOCK_SIZE: tl.constexpr, D: tl.constexpr):
    b, h = tl.program_id(0), tl.program_id(1)
    offs_d, offs_t = tl.arange(0, D), tl.arange(0, BLOCK_SIZE)
    q = tl.load(q_ptr + (b * H + h) * D + offs_d).to(tl.float32)
    seq_len = tl.load(seq_lens_ptr + b)
    m_i, l_i = -float("inf"), 0.0
    acc = tl.zeros((D,), dtype=tl.float32)
    for j in range(0, tl.cdiv(seq_len, BLOCK_SIZE)):
        phys = tl.load(block_table_ptr + b * max_blocks + j)     # indirection
        base = (phys * BLOCK_SIZE + offs_t[:, None]) * H * D + h * D + offs_d[None, :]
        valid = (j * BLOCK_SIZE + offs_t) < seq_len
        k = tl.load(k_cache_ptr + base, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + base, mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.where(valid, tl.sum(k * q[None, :], axis=1) * sm_scale, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    tl.store(o_ptr + (b * H + h) * D + offs_d, acc / l_i)
```

Two design notes. First, with a single query the score computation is a matrix--vector product, so
the kernel uses `tl.sum` over an elementwise product rather than `tl.dot`. Decode attention is
purely memory-bound: every KV byte is read once for about two FLOPs. The block size therefore
matters for *memory* efficiency. 16--32 tokens $\times$ `H_kv` $\times$ `D` per block gives
contiguous loads of several KB, enough to approach full bandwidth. Second, the grid has only
$B\cdot H$ programs. With a batch of 4 and 8 KV heads, that is 32 programs on 132 SMs, so most of
the GPU idles on long sequences. *Split-KV* (flash-decoding) adds a third grid axis over chunks of
the sequence. Each program returns a partial $(m,\ell,\boldsymbol{a})$ triple, and a small second
kernel merges them with the same rescaling rule. Production kernels also share the K/V load across
the query heads of a GQA group, which cuts KV traffic by the group size.

::: systemslens
[Kernels are the engine's contract]{.box-title}

A serving engine's memory manager ([@ch:serving]) and its attention kernel must agree on
everything: block size, KV layout (token-major or head-major), dtype and scale format for quantized
KV, and how the block table and sequence lengths are passed. This is why engines ship their own
attention backends, and why a change such as "store KV in FP8" or "share blocks across sequences for
prefix caching" is as much a kernel change as a scheduler change. When you prototype a new cache
policy, write the PyTorch reference (`reference.paged_decode_attention`) first. It defines the
semantics that any fast kernel must reproduce.
:::

## Numerics inside kernels {#sec:triton:numerics}

Hand-written kernels own their numerics, and the defaults that PyTorch chose for you no longer
apply.

- **Accumulate in fp32.** Tensor cores take bf16/fp16/fp8 inputs but accumulate in fp32. Keep `acc`
  in `tl.float32` and round once when storing. A bf16 accumulator over $K=4096$ products loses most
  of the low-order bits: bf16 has an 8-bit significand, so once the running sum is 256 times larger
  than a term, the term vanishes.

- **Upcast before reductions.** Our softmax and norm kernels convert loaded tiles with
  `.to(tl.float32)` before `tl.max`/`tl.sum`; fp16 sums of squares overflow at 65504.

- **Subtract the max.** `exp` of fp32 overflows above $\approx88.7$. Every softmax, online or not,
  must work with $s-m$.

- **Mind the $-\infty$ arithmetic.** A row whose scores are all masked has $m=-\infty$, and
  $e^{m-m'}=e^{-\infty-(-\infty)}$ is NaN. Engines guard this case (padding rows in batched decode,
  fully masked sliding windows), for example by clamping $m$ to a finite floor.

- **Determinism.** Floating-point addition is not associative. Split-K GEMMs, split-KV attention and
  atomic accumulations (`tl.atomic_add` for weight gradients) sum in an order that depends on
  scheduling, so results can differ bit-for-bit between runs. For RL training this matters: rollouts
  generated by an inference kernel and log-probabilities recomputed by a training kernel can
  disagree slightly even for identical weights ([@ch:rlsys]). Deterministic modes use fixed
  reduction trees at some cost in speed.

::: pitfall
[Testing tolerances]{.box-title}

Compare against an fp32 (or fp64) reference, never against another low-precision kernel, and scale
tolerances with the reduction length. Our tests use $10^{-5}$ for fp32 interpreter runs. The GPU
bf16 test allows $2\times10^{-2}$ relative error. A kernel that passes at $K=64$ and fails at
$K=8192$ usually accumulates in the wrong dtype.
:::

### What to look at in a profile

::: center
| kernel type | bound by | first metrics to check (Nsight Compute) |
|:---|:---|:---|
| elementwise, softmax, norms | HBM bandwidth | DRAM throughput (% of peak); global load efficiency (coalescing) |
| GEMM, prefill attention | tensor cores | tensor pipe utilisation; SMEM bank conflicts; occupancy and register spills |
| decode attention, W4A16 GEMV | HBM bandwidth + parallelism | DRAM throughput; number of programs vs. SMs (grid too small?) |
| anything tiny | launch latency | Nsight Systems timeline gaps; use CUDA graphs or fuse |
:::

## Under the hood: from Python to SASS {#sec:triton:compiler}

`@triton.jit` does not interpret your Python on the GPU. On the first launch with a new signature it
parses the function's AST and lowers it through a stack of MLIR dialects
([@fig:triton:compiler]). *Triton IR* (TTIR) is a hardware-independent, block-level SSA
form. *TritonGPU IR* (TTGIR) attaches *layouts* to every tensor: which thread owns which elements,
whether a tile lives in registers or in shared memory, and which MMA instruction shape a `tl.dot`
uses. Most of Triton's optimisations happen at this level: coalescing, pipelining, layout
conversions, and the insertion of barriers and asynchronous copies. Next come LLVM IR, then PTX, and
finally the vendor assembler produces a cubin (SASS). Every stage can be inspected:

``` {#lst:triton:asm .python caption="Dumping the intermediate representations of a compiled kernel (GPU)."}
compiled = _softmax_kernel[(n_rows,)](x, out, n_cols, x.stride(0), out.stride(0),
                                      BLOCK=1024)
print(compiled.asm.keys())          # e.g. 'ttir', 'ttgir', 'llir', 'ptx', 'cubin'
print(compiled.asm["ttgir"])        # look for #blocked / #shared / #mma layouts
print(compiled.n_regs, compiled.n_spills)   # register pressure at a glance
```

![Triton's compilation pipeline. The optimisations that make block-level code fast happen when
tensors acquire GPU layouts in TTGIR.](figures/appA_triton-11.svg){#fig:triton:compiler}

The same applies to code you did not write. `torch.compile` with the Inductor backend emits Triton
kernels for fused pointwise and reduction regions (and sometimes GEMMs, via templates). Set
`TORCH_LOGS=output_code` to print them, or set `TORCH_COMPILE_DEBUG=1` to dump them to disk. Reading
those kernels is a quick way to learn what was fused, which regions fell back to library calls, and
whether a `view` or `.contiguous()` in your model code broke a fusion.

::: systemslens
[When the compiler needs help]{.box-title}

Kernels with very large tiles (big `BLOCK` for long rows, large GEMM tiles) are limited by
registers. If `n_spills` is non-zero, shrink the tile, lower `num_stages`, or raise `num_warps` so
that each thread holds fewer elements. Excessive `constexpr` specialisation (e.g. passing the
sequence length as `constexpr`) recompiles for every new value: keep shapes as runtime arguments
unless the compiler genuinely needs them.
:::

## Triton and CUDA: a translation table

::: center
| CUDA concept                     | Triton equivalent / who handles it                      |
|:---------------------------------|:--------------------------------------------------------|
| thread block                     | program instance (`tl.program_id`)                      |
| grid dims                        | launch grid tuple `kernel[grid](...)`                   |
| threads per block                | `num_warps` $\times$ 32 (compiler distributes the tile) |
| shared memory, `__syncthreads`   | implicit; inserted by the compiler (layouts, swizzling) |
| coalescing, vector loads         | inferred from contiguous `tl.arange` offsets            |
| `mma`/`wgmma` fragments          | `tl.dot` on tiles $\ge16$ in each dimension             |
| `cp.async`, TMA, multi-buffering | `num_stages`; tensor descriptors in recent versions     |
| warp shuffles for reductions     | `tl.sum`, `tl.max` along an axis                        |
| atomics                          | `tl.atomic_add`, `tl.atomic_max`, ...                   |
| templates / compile-time params  | `tl.constexpr` arguments (re-JIT per value)             |
:::

When does a project drop below Triton? When it needs explicit control over warp roles, register
allocation or inter-SM cooperation (thread-block clusters, distributed shared memory), or the very
last 10--20% on a flagship GEMM or attention kernel. Then use CUDA C++ with CUTLASS/CuTe, or tile
DSLs such as ThunderKittens.

## Measuring kernels

- **Wall-clock**: `triton.testing.do_bench(fn)` handles warm-up, L2 flushing and CUDA-event timing.
  Never time a GPU kernel with `time.time()` without synchronising.

- **Achieved rates**: convert time into GB/s for memory-bound kernels and TFLOP/s for compute-bound
  ones, and report the percentage of the roofline (the lab's `bench.py` does this). A 2$\times$
  speedup that reaches only 15% of peak bandwidth still has headroom.

- **Timelines**: Nsight Systems (`nsys profile python script.py`) shows gaps between kernels: launch
  overhead, CPU stalls, synchronisations. For decode, these gaps are often the real problem, and the
  fix is CUDA graphs ([@ch:inference]), not a faster kernel.

- **Inside one kernel**: Nsight Compute (`ncu --set full`) reports DRAM throughput, SM and
  tensor-pipe utilisation, occupancy, bank conflicts and stall reasons, and plots the kernel on a
  roofline.

### Using a Triton kernel from PyTorch

A launcher like `softmax_rows` is already callable on CUDA tensors. To make it a first-class
operator that `torch.compile` can trace through and autograd can differentiate, register it as a
custom op (`torch.library.custom_op`) with a fake (meta) implementation for shape inference, and
give it a backward formula, either with `register_autograd` or by wrapping it in a
`torch.autograd.Function`. Recent PyTorch versions can also capture user-defined Triton kernels
directly inside compiled graphs.

## Summary

- Triton programs operate on tiles. You choose the decomposition (grid, block sizes, loop order),
  and the compiler handles threads, shared memory and tensor cores.

- Masks with neutral padding values (`0`, $-\infty$) make ragged edges and reductions correct.

- Fusion turns memory-bound chains into single passes. Tiling changes asymptotic traffic (GEMM,
  FlashAttention). Program ordering and pipelining decide how close a tiled kernel gets to the
  roofline.

- Quantized GEMMs and attention are where custom kernels pay off most in LLM inference, because
  their data formats and fusions are not library defaults.

- Develop with the interpreter and PyTorch references, then measure against the roofline with
  `do_bench`, Nsight Systems and Nsight Compute.

## Exercises

::: exercise
[grid arithmetic]{.box-title}

For `vector_add` with $n=10^6$ and `BLOCK`=1024, how many programs are launched, and how many lanes
are masked off? Why must `BLOCK` be a power of two?
:::

::: exercise
[padding values]{.box-title}

In [@lst:triton:softmax], what goes wrong if masked lanes are loaded with `other=0.0`?
Construct a row of length 3 with `BLOCK`=4 where the result is incorrect.
:::

::: exercise
[long rows]{.box-title}

Rewrite the softmax kernel for rows longer than one block, using a loop with online max/sum
rescaling followed by a second normalisation pass. How many HBM passes does your kernel make? Can
you get away with one write?
:::

::: exercise
[tile intensity]{.box-title}

Compute $I_{\text{tile}}$ for $(B_M,B_N)\in\{(64,64),(128,128),(128,256)\}$ in bf16 and in fp8.
Which of them can saturate an H100's tensor cores purely from HBM bandwidth, and why does L2 reuse
change the answer?
:::

::: exercise
[swizzling]{.box-title}

Generalise [@fig:triton:swizzle]: for an $R\times C$ grid of tiles, $P$ concurrent programs
and group size $G$, count the distinct $A$ and $B$ panels touched by the first $P$ programs, and
find the $G$ that minimises the total.
:::

::: exercise
[shared memory budget]{.box-title}

With `num_stages`$=s$ and bf16 tiles $B_M\times B_K$ and $B_K\times B_N$, how much shared memory
does the pipelined GEMM need? With 228 KB per SM on an H100, what is the largest $s$ for
$128\times128\times64$ tiles, and how many such programs fit per SM?
:::

::: exercise
[int4 epilogue]{.box-title}

Modify the W8A16 kernel for group-wise int4 with packed weights: where do you unpack, and where must
the scale and zero point be applied when the group size equals $B_K$?
:::

::: exercise
[FlashAttention memory]{.box-title}

Count the HBM bytes read and written by the forward kernel for $B=1$, $H=32$, $T=S=8192$, $d=128$,
bf16, assuming no L2 hits, as a function of $B_M$. Compare with naive attention.
:::

::: exercise
[decode attention]{.box-title}

Explain why the kernel of [@lst:triton:flash] uses only $B\cdot H$ programs when $T=1$, and
sketch a split-KV variant: what extra state must each partial result carry so the reduction is
exact?
:::

::: exercise
[SwiGLU in the epilogue]{.box-title}

Sketch how to fuse SiLU-and-multiply into the epilogue of a GEMM that computes `gate` and `up`
together from a concatenated weight $[W_g;W_u]$. Which tile shape constraint does this impose, and
how many HBM bytes per token does it save for Qwen3-8B?
:::

::: exercise
[RMSNorm backward]{.box-title}

Derive [@eq:triton:rmsbwd], starting from
$\partial\hat x_i/\partial x_j=r\big(\delta_{ij}-\hat x_i\hat x_j/n\big)$. Why is it safe, and
cheaper, to recompute $r$ in the backward kernel rather than save it?
:::

::: exercise
[paged decode]{.box-title}

For Qwen3-8B (8 KV heads, $d=128$, bf16) and a block size of 16 tokens, how many bytes does one
program load per block? With a batch of 2 and 32k-token contexts, how many programs does
[@lst:triton:paged] launch, and how many split-KV chunks per sequence would you use on a
132-SM GPU?
:::

## Lab A: GPU kernels with Triton {#sec:triton:lab}

::: labbox
[GPU kernels with Triton]{.box-title}

**Code:** `code/labs/labA_triton`

**Goals**

Write five Triton kernels (vector add, fused softmax, RMSNorm, grouped tiled matmul, FlashAttention
forward) and a PyTorch online-softmax attention reference. Verify them on any machine with the
interpreter, then measure them against the roofline on a GPU.

**Setup**

`cd code; LAB_IMPL=starter python -m pytest -q labs/labA_triton`. Without CUDA, the test file sets
`TRITON_INTERPRET=1` automatically (use small shapes; it is slow). `_w8a16_kernel` in `starter.py`
is complete: read it as a worked example before you start.

**Tasks**

1.  `_add_kernel` and `vector_add`: the grid, offsets and masks.

2.  `_softmax_kernel` and `softmax_rows`: one program per row; choose padding values that are
    neutral for max and sum.

3.  `_rmsnorm_kernel` and `rmsnorm`: mean of squares in fp32; support arbitrary leading dimensions
    by flattening in the launcher.

4.  `_matmul_kernel` and `matmul`: tiles, a $K$ loop with masks, fp32 accumulation with `tl.dot`,
    then grouped program ordering.

5.  `flash_attention_tiled_torch`: online softmax in PyTorch, including $T\neq S$ (queries aligned
    to the end of the keys, as in decode).

6.  `_flash_fwd_kernel` and `flash_attention`: port your PyTorch version to Triton, with causal
    block skipping.

7.  `arithmetic_intensity` and `roofline_time`: the napkin math used in `bench.py`.

**Measurements (GPU)**

Run `bench.py --impl starter --hw <your GPU>`. Report, for each kernel and size, time, achieved GB/s
or TFLOP/s and percentage of peak. Expect vector add and softmax at 70--90% of bandwidth for large
sizes, and the unfused eager softmax several times slower. For matmul, compare with and without
grouped ordering (`group_m=1`) and for 2--3 tile configurations; then add `@triton.autotune`. For
FlashAttention, compare against `F.scaled_dot_product_attention` and plot time against $T$ for your
kernel and for naive attention: one curve should be quadratic, and you should explain why the other
is too, despite linear memory.

**Deliverables**

Passing tests (interpreter or GPU); a table of achieved rates vs. roofline; one paragraph on the
biggest gap you found and its cause, backed by an Nsight Compute metric.

**Stretch goals**

- Fuse RoPE into the FlashAttention kernel's $Q$ and $K$ loads.

- Add GQA: let several query heads (`bh` indices) map to one KV head without materialising repeated
  K/V.

- Implement a split-KV decode kernel and compare it with the prefill kernel at $T=1$.

- Write the group-wise int4 variant of the W8A16 kernel and plug it into `QuantLinear` from [@lab:efficient].
  Measure decode tokens/s for Qwen3-0.6B.
:::
