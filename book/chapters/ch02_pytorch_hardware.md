# PyTorch and the Machine Underneath {#ch:pytorch}

Every model in this book is a PyTorch program, and every PyTorch program is a stream of *kernels*
launched onto an accelerator. This chapter covers both levels. The first half is a working
introduction to PyTorch: tensors, automatic differentiation, modules, data loaders, the training
loop, checkpoints, and running on one or several GPUs. A reader fluent in PyTorch can skim it, but
should read the Systems Lens boxes, which connect each API to what happens on the hardware. The
second half opens the box: how a GPU is organised, why most kernels are limited by memory bandwidth
rather than arithmetic, how the *roofline model* predicts which, how PyTorch's asynchronous
execution model hides (or fails to hide) latency, and how to measure all of this honestly. We make
the roofline exact enough to compute with: operational intensity at a given level of the memory
hierarchy, worked intensities for every class of kernel an LLM runs, and an empirical roofline
measured on this book's reference machine with the lab's kernels placed on it. These are the tools
we use to reason about every optimisation in later chapters; the deep treatments of number formats
and of the GPU itself follow in [@ch:numerics; @ch:gpuarch].

![Roadmap of [@ch:pytorch]: four stages of PyTorch fundamentals followed by four stages that
explain what those programs do on the
hardware.](figures/ch02_pytorch_hardware-1.svg){#fig:pytorch:roadmap}

## What is PyTorch? {#sec:pytorch:what}

PyTorch is an open-source Python library for deep learning, originally released by Meta AI in 2016
and now governed by the PyTorch Foundation. It has been the most widely used framework in
machine-learning research for years, and it is the substrate of nearly every open LLM codebase, from
Hugging Face Transformers to vLLM, SGLang, Megatron-LM and veRL. Its popularity comes from a design
that feels like ordinary Python: you write imperative code, run it line by line, inspect any value,
and get gradients for free.

### The three core components

It is helpful to see PyTorch as three libraries in one ([@fig:pytorch:stack]):

1.  A **tensor library**: an array library in the spirit of NumPy, whose arrays can live on CPUs,
    NVIDIA and AMD GPUs, or Apple silicon, with a vast set of operations implemented as optimised
    kernels.

2.  An **automatic differentiation engine** (*autograd*): it records the operations applied to
    tensors and computes gradients by traversing that record backwards.

3.  A **deep-learning library**: modules (layers), loss functions, optimisers, data loading and
    distributed training utilities built on top of the first two.

![The PyTorch stack. A single Python-level operation passes through autograd and the dispatcher
before a device kernel is enqueued. Each layer adds a few microseconds of overhead, which is why
tiny operations are slow in eager mode and why kernel fusion and compilation matter
([@sec:pytorch:async]).](figures/ch02_pytorch_hardware-2.svg){#fig:pytorch:stack}

### Deep learning in one paragraph

Artificial intelligence aims at systems that perform tasks associated with human intelligence;
machine learning is the subfield that achieves this by learning from data; deep learning is the part
of machine learning that uses neural networks with many layers, which learn both the features and
the decision rule from raw inputs. Training a deep network follows a fixed recipe: a model with
parameters $\theta$ maps inputs to predictions; a loss function measures how wrong the predictions
are on labelled training data; gradients of the loss with respect to $\theta$ are computed by
backpropagation; and an optimiser nudges $\theta$ against the gradient. After training, the model is
used for *inference* on new inputs. Everything in this book, including LLM pretraining and RL, is an
instance of this recipe with a particular model, loss and data stream.

### Installing PyTorch

PyTorch is installed with `pip` or `conda`. CPU-only wheels work everywhere; GPU builds must match
the CUDA (or ROCm) version supported by the installed driver, and the selector on `pytorch.org`
prints the right command. Two lines confirm what is available:

``` {#lst:pytorch:check .python caption="Checking the installation and available accelerators."}
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available(),
      "| MPS:", torch.backends.mps.is_available())   # Apple silicon GPU
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, f"{p.total_memory/2**30:.0f} GiB,", p.multi_processor_count, "SMs")
```

Apple-silicon Macs expose their GPU through the `mps` backend, which is convenient for the small
experiments in this book but lacks some kernels and all of the CUDA-specific tooling; we write code
that selects a device with `llmsys.utils.get_device()` and runs on whatever is present.
[@app:setup] gives a complete environment recipe, including cloud GPUs.

## Tensors {#sec:pytorch:tensors}

### Scalars, vectors, matrices and beyond

A tensor is an $n$-dimensional array with a single element type. Its *rank* (or order) is the number
of dimensions: a scalar has rank 0, a vector rank 1, a matrix rank 2, and LLM activations are
usually rank 3 with shape `(batch, sequence, features)` or rank 4 inside attention,
`(batch, heads, sequence, head_dim)`. Creating tensors is straightforward:

``` {#lst:pytorch:create .python caption="Creating tensors of different ranks."}
s = torch.tensor(3)                      # rank 0, shape ()
v = torch.tensor([1.0, 2.0, 3.0])        # rank 1, shape (3,)
m = torch.tensor([[1, 2], [3, 4]])       # rank 2, shape (2, 2), int64
x = torch.zeros(8, 1024, 768)            # rank 3: (batch, seq, d_model)
print(x.ndim, x.shape, x.dtype, x.device, x.numel())
```

### Tensor data types

PyTorch infers the data type (*dtype*) from the input: Python integers become `torch.int64`, Python
floats become `torch.float32`. Token ids are integers (`int64` by default, since embedding lookups
expect it); everything that is learned or computed is floating point. Converting is explicit:
`x.to(torch.bfloat16)` or `x.float()`. The default `float32` is a sensible balance of precision and
cost on CPUs, but on modern accelerators most LLM arithmetic runs in 16-bit or even 8-bit formats,
which we examine in [@sec:pytorch:formats].

### Common tensor operations

A handful of operations recur throughout the book:

``` {#lst:pytorch:ops .python caption="Operations used constantly in LLM code."}
a = torch.arange(6.0).reshape(2, 3)     # [[0,1,2],[3,4,5]]
a.view(3, 2)                            # reinterpret same memory (needs contiguity)
a.reshape(3, 2)                         # view if possible, else copy
a.T                                     # transpose of a 2-D tensor (a view)
a.transpose(0, 1)                       # swap any two dims (a view)
a @ a.T                                 # matrix multiplication, (2,3)@(3,2) -> (2,2)
q = torch.randn(4, 12, 128, 64)         # (batch, heads, seq, head_dim)
scores = q @ q.transpose(-2, -1)        # batched matmul -> (4, 12, 128, 128)
x = torch.randn(2, 5, 8)
x.mean(dim=-1, keepdim=True)            # reductions along a dimension
torch.softmax(x, dim=-1)                # normalise the last dimension
torch.cat([x, x], dim=1)                # (2, 10, 8)
x[:, -1, :]                             # last position of every sequence, (2, 8)
```

::: systemslens
[Shapes, strides and the cost of a "free" transpose]{.box-title}

A tensor is a pointer into a block of memory (its *storage*) plus a shape and a *stride* per
dimension: the number of elements to skip to move one step along that dimension. A contiguous
$2\times3$ tensor has strides $(3,1)$; its transpose is the *same storage* with shape $(3,2)$ and
strides $(1,3)$. That is why `.T`, `.transpose`, slicing and `.view` are free: they allocate nothing
and launch no kernel. The bill arrives later. A kernel that reads a non-contiguous tensor makes
strided, uncoalesced memory accesses, and some operations (`.view` after a transpose, many custom
kernels) refuse non-contiguous inputs, forcing a `.contiguous()` copy. In attention code, the
`transpose(1, 2)` between `(B,T,H,D)` and `(B,H,T,D)` layouts is the classic source of hidden
copies. [@lab:pytorch] asks you to detect views versus copies by comparing storage pointers.
:::

### Number formats {#sec:pytorch:formats}

Floating-point formats trade *range* (exponent bits) against *precision* (mantissa bits)
([@fig:pytorch:formats]). The formats that matter for LLMs are:

- **fp32** (1 sign, 8 exponent, 23 mantissa bits): the reference format; master weights and
  optimiser states are usually kept in fp32.

- **tf32**: not a storage format but a tensor-core *compute mode* that reads fp32 inputs and
  multiplies them with a 10-bit mantissa, enabled in PyTorch by setting
  `torch.backends.cuda.matmul.allow_tf32` to `True`.

- **fp16** (1/5/10): good precision but a narrow range (maximum 65,504), so gradients can underflow
  or overflow; training in fp16 needs *loss scaling* ([@ch:pretrain]).

- **bf16** (1/8/7): the same range as fp32 with much less precision (machine epsilon $2^{-7}$). Its
  range makes it the default for LLM training and inference.

- **fp8** in two flavours, E4M3 (max 448, more precision) and E5M2 (max 57,344, more range),
  supported by Hopper and later GPUs for matmuls with per-tensor or per-block scaling factors
  ([@ch:numerics]; their use for cheaper inference is in [@ch:efficient]).

![Bit layouts of the floating-point formats used for LLMs. bf16 keeps fp32's eight exponent bits and
sacrifices mantissa; fp16 does the opposite. Fewer bits mean fewer bytes to move and more
tensor-core throughput: on an H100, dense bf16 matmul peaks at about 989 TFLOP/s and fp8 at about
twice that.](figures/ch02_pytorch_hardware-3.svg){#fig:pytorch:formats}

::: napkinmath
[What a dtype buys you]{.box-title}

Halving the bytes per element halves memory capacity and traffic for every tensor stored in that
format, and on tensor cores roughly doubles peak matmul throughput (H100 SXM, dense: tf32
$\approx$`<!-- -->`{=html}495, bf16 $\approx$`<!-- -->`{=html}989, fp8
$\approx$`<!-- -->`{=html}1,979 TFLOP/s). A 7B model's weights take 28 GB in fp32, 14 GB in bf16,
7 GB in fp8, and 3.5 GB in int4. Precision is the cost: bf16's relative spacing of
$2^{-7}\approx0.8\%$ means that adding a small update to a large weight can round to no change at
all. This is why optimisers keep fp32 master weights even when the forward pass runs in bf16
([@ch:pretrain]).
:::

::: keyidea
[Number formats: what this chapter needs, and what comes later]{.box-title}

A floating-point format is a triple (sign, exponent bits $e$, mantissa bits $m$). The exponent sets
the *range*, the mantissa sets the *spacing*: between consecutive powers of two the format has
$2^m$ evenly spaced values, so its relative rounding error is at most $2^{-(m+1)}$ (half the
machine epsilon $\epsilon = 2^{-m}$; [@lab:pytorch] reads both from `torch.finfo`). Three facts
carry us through this chapter. (1) Bytes per element set memory traffic, so the format enters
every operational intensity below as the $b$ in the denominator. (2) Tensor cores multiply in a
narrow format but *accumulate* in fp32, which is why a bf16 GEMM is accurate enough to train with.
(3) The A100 has bf16, fp16, tf32 and int8 tensor-core paths but no fp8 ones; fp8 arrived with the
H100. Everything else (round-to-nearest-even and stochastic rounding, how error grows in a long
sum, subnormals and overflow, loss scaling, block-scaled fp8 and 4-bit formats, integer
quantisation and why bitwise reproducibility is hard) is the subject of [@ch:numerics].
:::

## Models as computation graphs {#sec:pytorch:graphs}

A neural network is a composition of operations, and it is useful to draw it as a directed acyclic
graph whose nodes are operations and whose edges carry tensors: the *computation graph*. Take
logistic regression on one example, the smallest model with every ingredient of a deep network:
$$\begin{equation}
z = \boldsymbol{w}^\top\boldsymbol{x} + b,\qquad a = \sigma(z) = \frac{1}{1+e^{-z}},\qquad
\mathcal{L} = -\bigl[y\log a + (1-y)\log(1-a)\bigr].
\end{equation}$$

``` {#lst:pytorch:logreg .python caption="Logistic regression forward pass; autograd records the graph as it runs."}
import torch.nn.functional as F
x = torch.tensor([1.5, -0.5])       # input features
y = torch.tensor(1.0)               # true label
w = torch.tensor([0.4, 0.9], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
z = x @ w + b                       # weighted sum
a = torch.sigmoid(z)                # predicted probability
loss = F.binary_cross_entropy(a, y)
print(loss.grad_fn)                 # <BinaryCrossEntropyBackward0 ...>
```

PyTorch builds this graph *dynamically* ("define-by-run"): each operation on a tensor with
`requires_grad=True` appends a node, recorded in the result's `grad_fn`, that knows how to compute
its local derivative. [@fig:pytorch:graph] shows the graph for
[@lst:pytorch:logreg] together with the backward pass that the next section runs over it.

![Computation graph for logistic regression. Solid arrows: forward pass (tensors flow left to
right). Dashed arrows: the backward pass multiplies local derivatives from the loss back to the
leaves that require gradients ($\boldsymbol{w}$ and $b$, green). For sigmoid followed by BCE, the
product simplifies to
$\partial\mathcal{L}/\partial z = a-y$.](figures/ch02_pytorch_hardware-4.svg){#fig:pytorch:graph}

## Automatic differentiation {#sec:pytorch:autograd}

Training needs the gradient of the loss with respect to every parameter. By the chain rule, the
gradient of $\mathcal{L}$ with respect to $\boldsymbol{w}$ is a product of local derivatives along
the path from $\boldsymbol{w}$ to $\mathcal{L}$. *Reverse-mode* automatic differentiation evaluates
that product from the loss backwards, which costs a small constant multiple of the forward pass
(about $2\times$ for networks dominated by matrix multiplications, as the Systems Lens below
derives) no matter how many parameters there are. This is the right mode for a scalar loss and
millions of parameters; [@sec:pytorch:vjp] makes the comparison with forward mode precise.

PyTorch exposes reverse mode in two ways. `torch.autograd.grad` returns gradients of an output with
respect to chosen inputs; `.backward()` accumulates gradients into the `.grad` field of every leaf
that requires them:

``` {#lst:pytorch:grad .python caption="Two ways to get gradients from the graph of Listing [-@lst:pytorch:logreg]."}
from torch.autograd import grad
gw = grad(loss, w, retain_graph=True)     # keep the graph for a second call
gb = grad(loss, b, retain_graph=True)
loss.backward()                           # fills w.grad and b.grad (and frees the graph)
print(w.grad, b.grad)                     # equal to gw[0], gb[0]
print((a - y).detach() * x)               # the analytic gradient for w
```

Three details cause most autograd bugs. First, gradients *accumulate*: `.backward()` adds to
`.grad`, so training loops must zero gradients every step (or rely on accumulation deliberately for
gradient accumulation, [@ch:pretrain]). Second, the graph is freed after one backward pass
unless `retain_graph=True`. Third, operations under `torch.no_grad()` or on `.detach()`ed tensors
are not recorded, which is what we want during evaluation and generation.

::: systemslens
[Why backward costs twice the forward, and where activation memory comes from]{.box-title}

For a linear layer $\mathbf{Y}=\mathbf{X}\mathbf{W}$ with $\mathbf{X}\in\mathbb{R}^{m\times k}$ and
$\mathbf{W}\in\mathbb{R}^{k\times n}$, the forward pass costs $2mkn$ FLOPs. Backward needs two
products of the same size: $\partial\mathcal{L}/\partial\mathbf{X}=\mathbf{G}\mathbf{W}^\top$ and
$\partial\mathcal{L}/\partial\mathbf{W}=\mathbf{X}^\top\mathbf{G}$, where
$\mathbf{G}=\partial\mathcal{L}/\partial\mathbf{Y}$, for $4mkn$ FLOPs. Hence forward plus backward
$\approx 3\times$ forward, and with $2N$ FLOPs per token forward we get the $6N$ FLOPs per training
token used throughout the book. The second product needs $\mathbf{X}$, the layer's *input*, so
autograd must keep every such input alive from the forward pass until its backward step: these
*saved activations* are why training memory grows with batch size $\times$ sequence length $\times$
depth, and why activation checkpointing (recomputing them during backward) trades FLOPs for memory;
[@sec:pytorch:vjp] measures the effect and [@sec:perf:ckpt] treats checkpointing in depth.
:::

### Vector--Jacobian products: the two modes and what they cost {#sec:pytorch:vjp}

Write a network as a composition $\boldsymbol{h}_0=\boldsymbol{x}$,
$\boldsymbol{h}_i = f_i(\boldsymbol{h}_{i-1})$ for $i=1,\dots,L$, and a scalar loss
$\mathcal{L}=\ell(\boldsymbol{h}_L)$. With $\mathbf{J}_i=\partial\boldsymbol{h}_i/\partial
\boldsymbol{h}_{i-1}$ the Jacobian of step $i$, the chain rule is a product of matrices:
$$\begin{equation}
\frac{\partial\mathcal{L}}{\partial\boldsymbol{x}}^{\!\top} \;=\;
\nabla\ell^{\top}\,\mathbf{J}_L\,\mathbf{J}_{L-1}\cdots\mathbf{J}_1 .
\label{eq:pytorch:chain}
\end{equation}$$ Matrix multiplication is associative, so the product can be evaluated from either
end, and the two orders are the two modes of automatic differentiation. *Forward mode* goes right
to left: it pushes a tangent vector $\boldsymbol{u}$ through Jacobian--vector products
$\boldsymbol{u}\leftarrow\mathbf{J}_i\boldsymbol{u}$ alongside the forward pass, and yields one
directional derivative per pass; the gradient with respect to $n$ inputs needs $n$ passes.
*Reverse mode* goes left to right: starting from $\boldsymbol{v}=\nabla\ell$, it applies
vector--Jacobian products (VJPs) $\boldsymbol{v}^\top\leftarrow\boldsymbol{v}^\top\mathbf{J}_i$ from
the last step back to the first, and one pass yields the derivative with respect to every input and
every parameter at once. If one VJP costs $c$ times its step's forward cost, a gradient costs
$c\,C_{\text{fwd}}$ in reverse mode and about $n\,c\,C_{\text{fwd}}$ in forward mode. With $n=10^9$
parameters and one scalar loss the choice is not close; the classical result that the gradient of
a scalar function costs a small constant multiple of the function itself is the "cheap gradient
principle" (Griewank and Walther, 2008).

No Jacobian is ever formed; each primitive supplies its VJP directly. For a linear layer
$\mathbf{Y}=\mathbf{X}\mathbf{W}^\top+\boldsymbol{b}$ with upstream gradient $\mathbf{G}$, the VJP is
$\partial\mathcal{L}/\partial\mathbf{X}=\mathbf{G}\mathbf{W}$, $\partial\mathcal{L}/\partial\mathbf{W}
=\mathbf{G}^\top\mathbf{X}$ and $\partial\mathcal{L}/\partial\boldsymbol{b}=\sum_{\text{rows}}\mathbf{G}$:
two matmuls the size of the forward one, so $c=2$, which is where the Systems Lens above gets
backward $\approx2\times$ forward. For an elementwise function the Jacobian is diagonal and the VJP
is a multiplication by $f'(\boldsymbol{h}_{i-1})$, with $c\approx1$. The price of reverse mode is
memory: the VJP of step $i$ needs $\boldsymbol{h}_{i-1}$ (or whatever $f_i'$ depends on), so the
forward pass must keep these *residuals* until the backward pass reaches them.
[@lst:pytorch:manualbwd] writes the whole reverse sweep by hand for the MLPs of this chapter; the
test in [@lab:pytorch] checks it against autograd to $10^{-6}$.

``` {#lst:pytorch:manualbwd .python caption="A reverse-mode sweep by hand for a Linear/ReLU stack under mean cross-entropy (from `lab02_pytorch_roofline/solution.py`)."}
def manual_mlp_backward(model, x, y):
    saved, h = [], x
    for layer in model:                          # forward, keeping each layer's input
        saved.append(h)
        h = layer(h)
    p = torch.softmax(h, dim=-1)
    g = (p - F.one_hot(y, p.shape[-1]).to(p.dtype)) / x.shape[0]   # dL/dlogits
    grads = []
    for layer, inp in zip(reversed(list(model)), reversed(saved)):
        if isinstance(layer, nn.Linear):         # VJP of Y = X W^T + b
            grads += [g.sum(0), g.T @ inp]       # db, dW
            g = g @ layer.weight                 # dX = G W
        elif isinstance(layer, nn.ReLU):         # diagonal Jacobian
            g = g * (inp > 0).to(g.dtype)
    return grads[::-1]                           # W1, b1, W2, b2, ... (parameters() order)
```

How much memory the residuals take, and what recomputing them saves, is easy to measure.
`saved_activation_bytes` in [@lab:pytorch] installs `torch.autograd.graph.saved_tensors_hooks`,
which see every tensor an operation stores for its backward, and adds up their bytes. For a stack
of 16 MLP blocks (width 4,096, hidden 16,384, GELU, bf16) over 8,192 tokens, each block saves its
input (67 MB), the first matmul's output (268 MB, for the GELU's derivative) and the GELU's output
(268 MB, for the second matmul's weight gradient): 9.66 GB in all, more than the 4.3 GB of weights.
Running each block under `torch.utils.checkpoint` keeps only the block inputs, 1.07 GB, and
recomputes the rest during backward. [@tab:pytorch:ckpt] shows the effect on the peak memory of one
forward and backward pass, measured with `torch.cuda.max_memory_allocated`.

::: {#tab:pytorch:ckpt .table-float}
| Variant | Saved activations | Peak above weights, fwd + bwd | Time per step |
|:---|---:|---:|---:|
| no checkpointing | 9.66 GB | 10.33 GB | 501 ms |
| every block checkpointed | 1.07 GB | 2.08 GB | 587 ms |

: Residuals of a 16-block bf16 MLP stack ($d=4096$, hidden 16,384, 8,192 tokens) with and without
per-block activation checkpointing, measured on 2x A100-40GB (PCIe), one GPU; torch 2.14.1, CUDA
13.0. The peak includes the weight gradients and the transient buffers of the backward pass.
Produced by `bench.py` of [@lab:pytorch].
:::

Checkpointing pays one extra forward pass, a third more FLOPs, to free most of the residual memory;
which tensors to save and which to recompute is a scheduling problem that [@sec:perf:ckpt] solves.

## Implementing multilayer neural networks {#sec:pytorch:mlp}

Real models are built by subclassing `torch.nn.Module`. A module registers its parameters (as
`nn.Parameter` attributes or submodules) in `__init__` and defines the computation in `forward`.
Calling the module runs `forward` and, via hooks, integrates with autograd, device movement,
`state_dict`s and more.

``` {#lst:pytorch:mlp .python caption="A multilayer perceptron as an `nn.Module`."}
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(30, 20)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))   # outputs logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

torch.manual_seed(123)                    # reproducible random initialisation
model = MLP(50, 3)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))   # 2213
```

The count is $(50\cdot30+30)+(30\cdot20+20)+(20\cdot3+3)=2{,}213$: each `nn.Linear(in, out)` has an
$\text{out}\times\text{in}$ weight matrix and an `out`-dimensional bias. The weights are initialised
randomly (by default from a scaled uniform distribution), so fixing the seed makes runs
reproducible.

Two habits are worth forming early. First, the network returns *logits*, unnormalised scores, not
probabilities: PyTorch's `cross_entropy` applies a numerically stable log-softmax internally, and
applying softmax yourself first is both wasteful and less accurate. Call
`torch.softmax(logits, dim=-1)` only when you actually need probabilities, for instance for
sampling. Second, wrap inference in `torch.no_grad()` (or `torch.inference_mode()`), so that no
graph is recorded and no activations are kept:

``` {#lst:pytorch:nograd .python caption="Inference without building a graph."}
X = torch.rand(4, 50)
with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=-1)   # rows sum to 1
```

## Setting up efficient data loaders {#sec:pytorch:loaders}

PyTorch separates *what* the data are from *how* they are batched. A `Dataset` implements `__len__`
and `__getitem__(i)`, returning one example. A `DataLoader` draws indices (shuffled or not), fetches
examples (optionally in worker processes), collates them into batch tensors, and yields them:

``` {#lst:pytorch:dataset .python caption="A map-style dataset and its loader."}
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features, self.labels = X, y
    def __getitem__(self, i):
        return self.features[i], self.labels[i]
    def __len__(self):
        return self.labels.shape[0]

train_loader = DataLoader(ToyDataset(X_train, y_train), batch_size=2,
                          shuffle=True,      # new order every epoch
                          drop_last=True,    # avoid a tiny final batch
                          num_workers=0)     # 0 = load in the main process
for epoch in range(2):
    for xb, yb in train_loader:
        ...
```

`shuffle=True` matters for SGD, which assumes batches are roughly independent samples;
`drop_last=True` avoids a final, much smaller batch whose noisy gradient can destabilise training
(and which would also break fixed-shape compilation). `num_workers` controls parallel loading.

![A data pipeline. Worker processes produce batches ahead of time into a bounded queue; with
`pin_memory=True` batches land in page-locked host memory so the host-to-device copy can be
asynchronous and overlap with compute.](figures/ch02_pytorch_hardware-5.svg){#fig:pytorch:loader}

::: systemslens
[Feeding the accelerator]{.box-title}

A DataLoader with `num_workers=k` forks $k$ processes (side-stepping Python's global interpreter
lock), each producing whole batches, and keeps `prefetch_factor` batches per worker in flight
([@fig:pytorch:loader]). The goal is simple: the GPU should never wait for input. For LLM
pretraining the per-batch CPU work is tiny if, and only if, tokenisation happened offline: a batch
is then just a slice of a memory-mapped array of token ids (`llmsys.data.TokenMemmapDataset`,
[@ch:text]). Two further flags matter on GPUs: `pin_memory=True` allocates batches in
page-locked memory, which the DMA engine can copy from directly, and
`x.to(device, non_blocking=True)` then makes the copy asynchronous so it overlaps with the previous
step's compute. In Jupyter notebooks, multi-process loading is unreliable; use `num_workers=0` there
and in tiny experiments, where process start-up dominates.
:::

## A typical training loop {#sec:pytorch:loop}

With a model, a loss and a loader, the training loop has the same skeleton in every chapter of this
book ([@fig:pytorch:loop]):

``` {#lst:pytorch:loop .python caption="The canonical training loop."}
model = MLP(in_dim=2, out_dim=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
for epoch in range(3):
    model.train()                          # enable dropout etc.
    for xb, yb in train_loader:
        logits = model(xb)                 # 1. forward
        loss = F.cross_entropy(logits, yb) # 2. loss
        optimizer.zero_grad()              # 3. clear old gradients
        loss.backward()                    # 4. backward: fill .grad
        optimizer.step()                   # 5. update parameters
    model.eval()                           # disable dropout for evaluation
    print(epoch, compute_accuracy(model, test_loader))

@torch.no_grad()
def compute_accuracy(model, loader):
    correct = total = 0
    for xb, yb in loader:
        correct += (model(xb).argmax(dim=-1) == yb).sum().item()
        total += yb.numel()
    return correct / total
```

![The training loop. Each iteration loads a batch, runs forward, computes the loss, clears stale
gradients, runs backward, and steps the optimiser. Evaluation and checkpointing happen periodically
with `model.eval()` and `torch.no_grad()`.](figures/ch02_pytorch_hardware-6.svg){#fig:pytorch:loop}

A few remarks generalise to LLM training. `model.train()` and `model.eval()` switch the behaviour of
layers such as dropout; forgetting `eval()` during generation makes outputs random for the wrong
reason. `optimizer.zero_grad()` can sit before `backward()` or after `step()`, but must happen once
per optimiser step; `zero_grad(set_to_none=True)` (the default) frees the gradient tensors instead
of writing zeros. The learning rate is the most important hyperparameter, and in
[@ch:pretrain] we add warmup, cosine decay and gradient clipping to this loop. Evaluating on
a held-out set separates fitting from generalisation; for LLMs we will track validation loss and,
later, task accuracy.

## Saving and loading models {#sec:pytorch:save}

A module's `state_dict()` is an ordered mapping from parameter (and buffer) names to tensors. Saving
and restoring it is the standard way to checkpoint:

``` {#lst:pytorch:save .python caption="Saving and restoring weights (and, for resuming, optimiser state)."}
torch.save(model.state_dict(), "model.pth")
model = MLP(2, 2)                                  # same architecture
model.load_state_dict(torch.load("model.pth", weights_only=True))

# resuming training needs the optimiser state too (e.g. Adam moments)
torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
            "step": step}, "ckpt.pth")
```

`torch.save` uses Python's pickle format, which can execute arbitrary code on load, so the flag
`weights_only=True` should be used for files you did not create. The ecosystem has largely moved to
**safetensors** for distributing weights: a simple header plus raw tensor bytes, safe to load and
memory-mappable, so a 16 GB checkpoint can be opened instantly and paged in lazily. Both the GPT-2
and Qwen3 loaders in `llmsys` read safetensors files.

::: napkinmath
[Checkpoint sizes]{.box-title}

A training checkpoint holds far more than the weights. For a 7B model trained with mixed-precision
AdamW, the fp32 master weights and two Adam moments total $7\times10^9\times12=84$ GB, plus 14 GB of
bf16 weights if stored. At 2 GB/s of write bandwidth to a parallel file system, saving takes about
50 s, so frequent synchronous checkpointing at scale wastes real GPU time. Sharded and asynchronous
checkpointing ([@ch:pretrain; @ch:distributed]) exist for this reason.
:::

## Training on GPUs {#sec:pytorch:gpu}

### Computations on GPU devices

Every tensor lives on a device, and an operation runs on the device of its inputs. Moving a tensor
is explicit, and mixing devices in one operation is an error:

``` {#lst:pytorch:device .python caption="Device placement rules."}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.tensor([1., 2., 3.]).to(device)
b = torch.tensor([4., 5., 6.]).to(device)
print(a + b)            # computed on the GPU; result lives there too
c = torch.tensor([1., 1., 1.])
# a + c  -> RuntimeError: tensors on different devices (cuda:0 and cpu)
model = model.to(device)  # moves all parameters and buffers in place
```

On a machine with several GPUs, `"cuda:0"`, `"cuda:1"`, ... name individual devices, and the
environment variable `CUDA_VISIBLE_DEVICES` restricts and renumbers what a process can see.
Copying a tensor from one GPU to another with `.to("cuda:1")` is legal PyTorch, but do not assume
it works on every machine: on this book's reference box, whose two GPUs sit on different CPU sockets
behind an IOMMU, a direct GPU-to-GPU copy silently delivers zeros, while NCCL collectives are
correct. We therefore move data between GPUs only through `torch.distributed`, and any multi-GPU
code should check a transfer against its source once before trusting it.

### Single-GPU training

Moving the loop of [@lst:pytorch:loop] to a GPU needs three changes: move the model once,
move each batch inside the loop, and leave everything else alone. For a tiny MLP the GPU will not be
faster (each kernel does almost nothing and launch overhead dominates), but for anything
matrix-heavy, including every LLM, the speedups are one to two orders of magnitude over a CPU.

``` {#lst:pytorch:gpuloop .python caption="The training loop on a GPU."}
model = MLP(2, 2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
for epoch in range(3):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Training with multiple GPUs {#sec:pytorch:ddp}

The simplest way to use several GPUs is *data parallelism* with PyTorch's `DistributedDataParallel`
(DDP). One process is launched per GPU; each holds a full replica of the model and receives a
different slice of every global batch. After the backward pass, gradients are averaged across
processes with an *all-reduce* collective, so every replica applies the same update and the replicas
stay identical ([@fig:pytorch:ddp]). With $P$ GPUs, an epoch takes roughly $1/P$ of the
time, minus communication overhead.

![Distributed data parallelism. A `DistributedSampler` gives each rank a disjoint shard; each rank
computes gradients on its shard; an all-reduce averages them; identical updates keep replicas
identical.](figures/ch02_pytorch_hardware-7.svg){#fig:pytorch:ddp}

The code changes are modest. A process group is initialised with a rendezvous address, the model is
wrapped in `DDP`, and the loader uses a `DistributedSampler`. The complete script is
`labs/lab02_pytorch_roofline/ddp_demo.py`; its heart is:

``` {#lst:pytorch:ddpcode .python caption="The essential pieces of DDP training (excerpt of `ddp_demo.py`)."}
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")   # rendezvous host
    os.environ.setdefault("MASTER_PORT", "29512")       # any free port
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def worker(rank, world_size, epochs):
    setup(rank, world_size)
    torch.manual_seed(0)                       # same init on every rank
    model = build_mlp(2, [16, 16], 2).to(device)
    ddp = DDP(model, device_ids=[rank] if device.type == "cuda" else None)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    loader = DataLoader(ds, batch_size=16, sampler=sampler, pin_memory=True)
    for epoch in range(epochs):
        sampler.set_epoch(epoch)               # different shuffle each epoch
        for x, y in loader:
            loss = F.cross_entropy(ddp(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    dist.destroy_process_group()

torch.multiprocessing.spawn(worker, args=(world_size, 3), nprocs=world_size)
```

In practice, scripts are usually launched with `torchrun --nproc_per_node=N script.py`, which spawns
the processes and sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`. DDP
requires real processes, so it does not work from inside a Jupyter notebook. The NCCL backend
implements the collectives on NVIDIA GPUs over NVLink and InfiniBand; the Gloo backend runs them
over TCP and is what the lab uses on CPU.

::: systemslens
[What DDP actually does during backward]{.box-title}

DDP does not wait for the full backward pass and then all-reduce everything. It registers autograd
hooks that fire as each parameter's gradient becomes ready, groups gradients into *buckets* (25 MB
by default), and launches an asynchronous all-reduce per bucket while backward continues on earlier
layers. Communication thus overlaps with computation. A ring all-reduce moves about $2(P-1)/P$ times
the gradient size per GPU, so for a 1B-parameter model with bf16 gradients each step moves about
4 GB per GPU; over NVLink at hundreds of GB/s this hides easily behind compute, over 25 Gb/s
Ethernet it does not. DDP also assumes every replica fits the full model, gradients and optimiser
state on one GPU. [@ch:distributed] removes that assumption with ZeRO/FSDP, tensor and
pipeline parallelism, and derives these communication costs properly.
:::

## The machine underneath: how a GPU is organised {#sec:pytorch:gpuarch}

We now switch from the API to the hardware. The details differ across generations and vendors, but
the organisation of an NVIDIA datacentre GPU is representative ([@fig:pytorch:hierarchy]);
we use the H100 SXM as the running example.

**Streaming multiprocessors.** The chip consists of many identical *streaming multiprocessors*
(SMs), 132 on an H100 SXM. Each SM runs many threads concurrently, organised in *warps* of 32
threads that execute the same instruction in lock-step (SIMT), and groups of warps form *thread
blocks* that are scheduled onto one SM and can share fast on-chip memory. Each SM contains
scalar/vector units and *tensor cores*, specialised units that compute small matrix products (for
example $16\times8\times16$ tiles) in one instruction. Nearly all of an H100's 989 TFLOP/s of bf16
throughput comes from tensor cores; the non-tensor fp32 throughput is more than an order of
magnitude lower. A kernel that cannot be phrased as matrix multiplication cannot use most of the
chip's FLOPs.

**The memory hierarchy.** Data move through a hierarchy whose levels trade capacity for bandwidth
and latency: per-thread *registers* (256 KB per SM in total); per-SM *shared memory*/L1 (up to about
228 KB of configurable shared memory per SM on H100), explicitly managed by the kernel; a chip-wide
*L2 cache* (50 MB); and off-chip *high-bandwidth memory* (HBM: 80 GB at 3.35 TB/s). Host memory sits
beyond PCIe (tens of GB/s), and other GPUs are reachable over NVLink (900 GB/s aggregate per H100).

![The memory hierarchy of an H100 SXM (approximate published figures). Every kernel's performance is
determined by how often it must go down this pyramid. Fast kernels such as FlashAttention
([@ch:attention]) restructure computation so that data loaded from HBM are reused many times
in shared memory and registers.](figures/ch02_pytorch_hardware-8.svg){#fig:pytorch:hierarchy}

**Kernels.** A GPU kernel is a function executed by a grid of thread blocks. A typical PyTorch
operation, say `torch.add(x, y)`, launches one kernel that reads $x$ and $y$ from HBM, adds them in
registers, and writes the result back to HBM. A matrix multiplication launches a cuBLAS kernel that
tiles the output, stages input tiles in shared memory, and feeds tensor cores from registers. The
difference in how much work each loaded byte supports is the subject of the next section.

**Our machine.** Every measured number in this book comes from a workstation with two A100-40GB
PCIe cards. Each has 108 SMs, a 40 MB L2 cache, up to 164 KB of shared memory per SM, 40 GB of HBM2
at 1.555 TB/s and a dense bf16 tensor-core peak of 312 TFLOP/s, all inside a 250 W power limit. The
two cards are connected only by PCIe Gen4 x16, about 25 GB/s per direction on paper and 20--22 GB/s
measured, with no NVLink. [@ch:gpuarch] takes the SM apart; here we need only its two roofs.

## The roofline model {#sec:pytorch:roofline}

A kernel needs time for its arithmetic and time to move its data. If the two overlap perfectly (a
good first approximation for well-written kernels), its runtime is bounded below by the larger:
$$\begin{equation}
t \;\ge\; \max\!\left(\frac{F}{\pi},\; \frac{Q}{\beta}\right),
\label{eq:pytorch:roofline}
\end{equation}$$ where $F$ is the number of FLOPs, $Q$ the bytes moved to and from HBM, $\pi$ the
peak FLOP/s and $\beta$ the peak bandwidth. Dividing $F$ by $t$ gives the attainable throughput as a
function of the kernel's *operational intensity* $I = F/Q$ (FLOPs per byte, also called arithmetic
intensity): $$\begin{equation}
\text{attainable FLOP/s} \;=\; \min(\pi,\; I\cdot\beta).
\label{eq:pytorch:attain}
\end{equation}$$ Plotted on log--log axes, this is a sloped line (bandwidth-bound region) meeting a
flat line (compute-bound region) at the *ridge point* $I^\star=\pi/\beta$: the *roofline* (Williams,
Waterman and Patterson, 2009). Kernels left of the ridge are memory-bound; making their arithmetic
faster does nothing, only moving fewer bytes helps. Kernels right of it are compute-bound.

![Rooflines for H100 and A100 in bf16 (published peak numbers) with typical LLM kernels placed at
their arithmetic intensity (computed, not measured). Only large matrix multiplications reach the
compute roof; elementwise operations, normalisations and batch-1 decoding live on the bandwidth
slope.](figures/ch02_pytorch_hardware-9.svg){#fig:pytorch:roofline}

### Operational intensity, precisely {#sec:pytorch:oi}

The two-line model hides three choices that decide whether its predictions are right.

*Which FLOPs.* $F$ counts the arithmetic the algorithm needs, with a multiply--add counted as two
FLOPs, not the instructions the kernel happens to issue. Index arithmetic, address computation and
work wasted on masked or padded elements are not in $F$, so a kernel that wastes work shows up as
a kernel that falls short of its roof.

*Which bytes.* $Q$ is the traffic across one boundary of the memory hierarchy. Williams et al.
deliberately chose the name *operational* intensity to say that $Q$ is the DRAM traffic *after*
the caches have filtered the program's loads and stores; the same kernel has a different intensity
with respect to L2 or shared memory. We write $I_\ell = F/Q_\ell$ for level $\ell$ and, unless
stated otherwise, mean HBM. For each level there is a roof, $\min(\pi, I_\ell\beta_\ell)$, and the
kernel's attainable rate is the minimum over levels (the *hierarchical* roofline; Yang, Kurth and
Williams, 2020).

*Which traffic count.* Every input must be read once and every output written once, so
$Q \ge Q_{\min}$, the *compulsory* traffic. We compute intensities from $Q_{\min}$; a real kernel
that re-reads data because its working set does not fit on chip moves more bytes, has a lower
true intensity, and lands below the analytic point. A fused kernel moves fewer bytes than the
chain of kernels it replaces, which is the whole argument for fusion.

::: proposition
[The roofline is tight to within a factor of two]{.box-title}

Suppose a kernel's only costs are $F$ FLOPs and $Q$ bytes; that the compute units never exceed
$\pi$ nor the memory system $\beta$; that each runs at its peak rate whenever it is busy; and that
at every instant at least one of the two is busy. Then
$$\max\!\left(\frac{F}{\pi},\frac{Q}{\beta}\right) \;\le\; t \;\le\; \frac{F}{\pi}+\frac{Q}{\beta}
\;\le\; 2\max\!\left(\frac{F}{\pi},\frac{Q}{\beta}\right).$$ The left inequality holds because
neither unit can exceed its peak; the middle one because each unit is busy for exactly $F/\pi$ or
$Q/\beta$ seconds when it runs at its peak, and with at least one unit busy at every instant $t$ is
at most the sum of the two busy times. Hence a kernel's achieved FLOP/s
lies between half the roofline and the roofline, and the factor of two is reached exactly at the
ridge point with no overlap at all. A kernel far *below* half its roof therefore has a cost the
model does not contain: latency, launch overhead, too little parallelism, or traffic above
$Q_{\min}$ ([@sec:pytorch:hides]).
:::

The ridge point is a property of the machine alone. For the A100-40GB PCIe,
$I^\star = 312\times10^{12}/1.555\times10^{12} \approx 201$ FLOP/byte from the datasheet; for the
H100 SXM, $989/3.35 \approx 295$. A kernel needs more than 200 FLOPs for every byte it brings from
HBM before the A100's tensor cores, rather than its memory, set its speed.

### Operational intensity of the kernels that matter {#sec:pytorch:intensities}

We now compute $I$ for every class of kernel an LLM runs, with $b$ bytes per element.

##### Elementwise operations.

An operation that reads $n_{\text{in}}$ tensors and writes $n_{\text{out}}$, doing $f$ FLOPs per
element, has $I = f/(b(n_{\text{in}}+n_{\text{out}}))$. `y = x + z` in bf16 does 1 FLOP per element
and moves 6 bytes: $I=1/6$. Activation functions, residual additions, dropout, and the scaling in
attention are all like this; even a GELU with a dozen FLOPs per element stays near 3 FLOP/byte.
All are deeply memory-bound, so the only way to speed them up is to avoid materialising their
inputs and outputs in HBM, by *fusing* them into neighbouring kernels.

##### Reductions and normalisations.

A row-wise sum of an $r\times c$ tensor does $rc$ adds, reads $rc$ elements and writes $r$:
$I = rc/(b(rc+r)) \approx 1/b$. Softmax over the same rows needs a maximum, a subtraction, an
exponential, a sum and a division, about 5 FLOPs per element, and at minimum reads and writes each
element once: $I\approx 5/(2b) = 1.25$ in bf16. RMSNorm and LayerNorm are similar, $I$ of order 1.
Reductions are as memory-bound as elementwise operations; what makes them harder is that each
output depends on a whole row, so a kernel must combine partial results across threads, which
costs synchronisation that an elementwise kernel never pays. We will see that cost in our
measurements.

##### Matrix multiplication.

For $\mathbf{C}=\mathbf{A}\mathbf{B}$ with $\mathbf{A}\in\mathbb{R}^{M\times K}$,
$\mathbf{B}\in\mathbb{R}^{K\times N}$, $F=2MNK$ and, at best, $Q=b(MK+KN+MN)$ (each operand read
once, the result written once). Dividing numerator and denominator by $MNK$ gives a form that is
easier to reason with: $$\begin{equation}
I_{\text{GEMM}} = \frac{2MNK}{b(MK+KN+MN)} = \frac{2}{b}\left(\frac{1}{M}+\frac{1}{N}+\frac{1}{K}\right)^{-1}.
\label{eq:pytorch:gemmI}
\end{equation}$$ The intensity is $2/b$ times a harmonic mean of the three dimensions (up to a factor
of 3), so *the smallest dimension decides it*. For a square GEMM, $I = 2n/(3b)$: in bf16, $I=n/3$,
which crosses the H100 ridge near $n\approx885$ and the A100 ridge near $n\approx600$. Large
training GEMMs are compute-bound; small ones are not.

##### Matrix--vector products and the batch that reaches the ridge.

With $M=1$ (one token's activations times a weight matrix), ([-@eq:pytorch:gemmI]) gives
$I\approx 2/b$: about 1 FLOP/byte in bf16, and 4 with int4 weights. This is the operation at the
heart of batch-1 decoding, and it sits at about $1/200$ of the A100's ridge point and $1/300$ of the
H100's. With a batch of $M$ tokens sharing the same weights, $I\approx 2M/b$ for $M\ll K,N$;
batching is the lever that slides decoding up the roofline. Setting ([-@eq:pytorch:gemmI]) equal
to $I^\star$ and solving for $M$ gives the batch at which a layer becomes compute-bound:
$$\begin{equation}
M^\star = \left(\frac{2}{bI^\star} - \frac{1}{N} - \frac{1}{K}\right)^{-1}
\;\approx\; \frac{bI^\star}{2}.
\label{eq:pytorch:mstar}
\end{equation}$$ For an $8192\times8192$ bf16 weight on the A100, $M^\star = (1/200.6 -
2/8192)^{-1} \approx 211$ tokens. This is the same critical batch the napkin calculator of
[@lab:landscape] finds for whole models.

##### Attention: prefill and decode.

A fused (FlashAttention-style) kernel reads $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ once and writes
the output once; the scores never reach HBM. With $H$ query heads, $G$ key--value heads, head
dimension $d_h$, $T_q$ queries and $T_k$ keys, it does $4HT_qT_kd_h$ FLOPs ($QK^\top$ and $PV$)
and moves $b\,d_h(2HT_q + 2GT_k)$ bytes. Two regimes follow. In *prefill*, $T_q=T_k=T$, and with
the causal mask halving the work, $$\begin{equation}
I_{\text{prefill}} \approx \frac{2HT^2d_h}{b\,d_h\,T(2H+2G)} = \frac{HT}{b(H+G)}
\;\xrightarrow{G=H}\; \frac{T}{2b},
\label{eq:pytorch:prefillI}
\end{equation}$$ about 1,024 FLOP/byte at $T=4096$ in bf16: compute-bound for any prompt longer than a
few hundred tokens. In *decode*, $T_q=1$ and the cache of $T$ tokens dominates the bytes:
$$\begin{equation}
I_{\text{decode}} \approx \frac{4HTd_h}{2b\,G\,T\,d_h} = \frac{2}{b}\cdot\frac{H}{G}.
\label{eq:pytorch:decodeI}
\end{equation}$$ In bf16 this is exactly the grouped-query ratio $H/G$: 1 for multi-head attention,
4 or 8 for typical GQA models, independent of the context length *and of the batch*, because every
sequence reads its own cache. Batching cannot rescue decode attention the way it rescues the weight
GEMMs; only sharing the cache across heads can. Multi-head latent attention (DeepSeek-V2 and V3)
takes this furthest: all $H$ heads read one latent of $d_c+d_r$ values per token and do about
$4d_c+2d_r$ FLOPs each on it, so $I \approx H(4d_c+2d_r)/(b(d_c+d_r))$, about 240 FLOP/byte for
$H=128$, $d_c=512$, $d_r=64$, past the A100's ridge ([@ch:anatomy; @ch:kvcache]).

Without fusion the picture changes. An unfused implementation writes the $T\times T$ score matrix,
reads it for the softmax, writes the probabilities and reads them again for $PV$: $4bHT^2$ extra
bytes. Its intensity becomes $Td_h/(b(T+d_h))$, which tends to $d_h/b = 64$ FLOP/byte for long
sequences: memory-bound on every GPU in this chapter, however long the prompt. FlashAttention
([@ch:attention]) is, in roofline terms, the move from 64 to $T/(2b)$.

::: {#tab:pytorch:intensities .table-float}
| Kernel (bf16) | $F$ | $Q_{\min}$ | $I$ (FLOP/byte) | A100 PCIe |
|:---|:---|:---|---:|:---|
| `x + y`, $n$ elements | $n$ | $6n$ | 0.17 | memory |
| row sum, $r\times c$ | $rc$ | $2(rc+r)$ | 0.5 | memory |
| softmax, $r\times c$ | $5rc$ | $4rc$ | 1.25 | memory |
| GEMV, $1\times8192$ by $8192\times8192$ | $2NK$ | $2(K+NK+N)$ | 1.0 | memory |
| GEMM, $M=16$, $N=K=8192$ | $2MNK$ | $2(MK+KN+MN)$ | 15.9 | memory |
| GEMM, $M=256$, $N=K=8192$ | $2MNK$ | $2(MK+KN+MN)$ | 241 | compute (barely) |
| GEMM, $8192^3$ | $2n^3$ | $6n^2$ | 2,731 | compute |
| attention prefill, causal, $T=4096$, $H=G=32$ | $\approx 2HT^2d_h$ | $8HTd_h$ | 1,024 | compute |
| attention prefill, unfused, same shape | $4HT^2d_h$ | $8HTd_h + 8HT^2$ | 62 | memory |
| attention decode, $H/G=4$, $T=4096$ | $4HTd_h$ | $\approx 4GTd_h$ | 4 | memory |

: Operational intensities of the kernel catalogue at minimum HBM traffic ($b=2$), and the side of
the A100-40GB PCIe ridge (201 FLOP/byte from the datasheet) each falls on. The lab's
`matmul_intensity`, `reduction_intensity` and `attention_intensity` compute every row.
:::

[@fig:pytorch:intensity] plots two of these formulas against their parameters. The GEMM curve
rises with slope one until $M$ approaches the other dimensions, crossing both ridges at a few
hundred tokens; the attention curves show prefill rising linearly with context while decode stays
flat at $H/G$.

![Operational intensity at minimum HBM traffic (analytic). (a) A GEMM of $M$ tokens against a
$K\times N$ weight: intensity grows linearly in $M$ until the harmonic mean of
([-@eq:pytorch:gemmI]) saturates; it crosses the A100 and H100 ridges at a few hundred tokens. (b)
Fused attention with 32 query heads: causal prefill grows linearly in the context length, decode
stays at the grouped-query ratio $H/G$ for every
length.](figures/ch02_pytorch_hardware-12.svg){#fig:pytorch:intensity}

::: napkinmath
[Fusion pays for memory-bound chains]{.box-title}

Take a bf16 activation tensor of $8\times4096\times4096$ elements (256 MiB) passing through
bias-add, GELU and dropout as three separate kernels: each reads and writes the tensor, about
$3\times2\times256$ MiB $\approx$ 1.6 GB of traffic, $\approx$`<!-- -->`{=html}0.48 ms at 3.35 TB/s.
Fused into one kernel it reads once and writes once: 0.54 GB, $\approx$`<!-- -->`{=html}0.16 ms, a
$3\times$ speedup with identical FLOPs. This is exactly what `torch.compile`'s code generator does
automatically ([@sec:pytorch:async]) and what we do by hand in Triton in
[@app:triton].
:::

### An empirical roofline on our machine {#sec:pytorch:empirical}

Datasheet peaks are upper bounds that no program reaches. An *empirical* roofline replaces $\pi$
and $\beta$ by the best rates a simple program achieves, so that the distance between a kernel and
its roof measures the kernel, not the marketing. We measure each roof with the simplest kernel that
saturates it. For bandwidth, a copy, `y.copy_(x)` on byte buffers: no arithmetic at all, perfectly
sequential access, and buffers up to 1 GiB, far larger than the 40 MB L2, so every byte comes from
HBM. For compute, a large bf16 matrix multiplication, which cuBLAS has tuned for years. Each
measurement follows the rules of [@sec:pytorch:profiling], with two additions: the warm-up runs for
a fixed time rather than a fixed count, because the GPU's clocks take a fraction of a second to
settle, and we keep the 10th and 90th percentiles of 20 repetitions as well as the median. An idle
A100's SM clock sits far below its 1,410 MHz maximum and needs about half a second of load to reach
it, so every probe and every kernel first runs for one second (`WARMUP_S` in the lab).

``` {#lst:pytorch:roofprobe .python caption="The two roof probes and one roofline point (from `lab02_pytorch_roofline/solution.py`)."}
def measure_copy_bandwidth(sizes_bytes, device="cpu", reps=20):
    out = []
    for n in sizes_bytes:
        x = torch.empty(n, dtype=torch.uint8, device=device).random_(0, 255)
        y = torch.empty_like(x)
        st = time_stats(lambda: y.copy_(x), reps=reps)      # median, p10, p90 seconds
        out.append(dict(n_bytes=n, bytes=2 * n, bytes_per_s=2 * n / st["seconds"], **st))
    return out

def _point(name, fn, flops, nbytes, reps):                  # one kernel on the roofline
    st = time_stats(fn, reps=reps)
    return dict(name=name, flops=flops, bytes=nbytes, intensity=flops / nbytes,
                flops_per_s=flops / st["seconds"], bytes_per_s=nbytes / st["seconds"], **st)

pts.append(_point("attn_decode",                            # 16 sequences, GQA 32/8 heads
                  lambda: F.scaled_dot_product_attention(q, kk, vv, enable_gqa=True),
                  B * attention_flops(1, T, H, D), B * attention_bytes(1, T, H, G, D, b), reps))
```

[@tab:pytorch:roofs] gives the two roofs. The copy reaches 1,369 GB/s, 88% of the datasheet's
1,555 GB/s, but only for large buffers: a 1 MiB copy runs at 139 GB/s, because moving 2 MiB at full
bandwidth takes 1.5 μs while each call carries about 14 μs of fixed cost (Python dispatch, the
launch, and the synchronisation in our timer). The best bf16 GEMM reaches 217 TFLOP/s, 70% of the
312 TFLOP/s peak, and the empirical ridge point is $217/1.37 \approx 158$ FLOP/byte.

::: {#tab:pytorch:roofs .table-float}
| Probe | Size | Achieved | Fraction of datasheet |
|:---|:---|---:|---:|
| copy | 1 MiB / 16 MiB / 64 MiB | 139 / 1,045 / 1,222 GB/s | 9% / 67% / 79% |
| copy | 256 MiB / 1 GiB | 1,343 / 1,369 GB/s | 86% / 88% |
| bf16 GEMM, random normal data | $n$ = 1,024 / 2,048 / 4,096 | 62 / 131 / 201 TFLOP/s | 20% / 42% / 64% |
| bf16 GEMM, random normal data | $n$ = 8,192 / 16,384 | 217 / 215 TFLOP/s | 70% / 69% |
| bf16 GEMM, all-zero data | $n$ = 8,192 / 16,384 | 296 / 290 TFLOP/s | 95% / 93% |

: The two roofs, measured on 2x A100-40GB (PCIe), one GPU; torch 2.14.1, CUDA 13.0, driver
595.91.07; median of 20 repetitions after a 1 s warm-up (p10 and p90 within 2% of the median for
every random-data GEMM of size 4,096 or more). Produced by `bench.py` of [@lab:pytorch].
:::

::: pitfall
[The compute roof depends on the data]{.box-title}

The last two rows of [@tab:pytorch:roofs] run the same cuBLAS kernel on the same shapes, with the
same instructions and the same bytes; only the values differ. With all-zero inputs the GEMM runs
36% faster. The cause is power, not arithmetic. The A100 PCIe card is limited to 250 W; switching
transistors costs energy in proportion to how many bits toggle, random data toggles many of them
on every multiply, and the power manager responds by lowering the clock. With zeros almost
nothing toggles and the clock stays high. Sampling `nvidia-smi` while each GEMM runs for eight
seconds shows it directly: on random data the card sits at its 250 W limit with the SM clock at
1,035 MHz; on zeros it draws 234 W at 1,380 MHz, close to the 1,410 MHz maximum (measured on 2x
A100-40GB (PCIe)). The ratio of the clocks, 1.33, is the ratio of the throughputs. Horace He (2024) reports the same effect on H100s. Two
lessons follow. Benchmark on data shaped like the real workload (random normal values are a
reasonable proxy for weights and activations; zeros, ones and `torch.empty` are not), and treat
the compute roof of a power-limited card as a property of the workload as much as of the chip.
:::

We can now place the kernel catalogue of [@tab:pytorch:intensities] on the measured roof.
[@tab:pytorch:points] lists each kernel's achieved rate and its fraction of the attainable rate
$\min(\pi, I\beta)$ at its intensity, and [@fig:pytorch:empirical] draws the result.

::: {#tab:pytorch:points .table-float}
| Kernel (bf16) | $I$ (FLOP/byte) | Time (μs) | Achieved | Fraction of roof |
|:---|---:|---:|---:|---:|
| `x + y`, $2^{27}$ elements | 0.17 | 606 | 1,328 GB/s | 97% |
| row sum, $2^{15}\times4096$ | 0.5 | 203 | 1,324 GB/s | 97% |
| softmax, $2^{15}\times4096$ | 1.25 | 805 | 667 GB/s | 49% |
| GEMV, $M=1$ | 1.0 | 118 | 1,136 GB/s | 83% |
| GEMM, $M=16$ | 15.9 | 119 | 18.0 TFLOP/s | 83% |
| GEMM, $M=256$ | 241 | 216 | 159 TFLOP/s | 73% |
| GEMM, $8192^3$ | 2,731 | 5,084 | 216 TFLOP/s | 100% |
| attention prefill (SDPA, causal) | 1,020 | 1,020 | 135 TFLOP/s | 62% |
| attention decode (SDPA, GQA) | 4 | 237 | 1,135 GB/s | 83% |

: The kernel catalogue measured on 2x A100-40GB (PCIe), one GPU; torch 2.14.1, CUDA 13.0; median of
20 repetitions after a 1 s warm-up. "Fraction of roof" divides the achieved FLOP/s by $\min(\pi, I\beta)$ with the
measured $\pi=217$ TFLOP/s and $\beta=1{,}369$ GB/s; the square GEMM defines the compute roof.
Shapes are those of [@tab:pytorch:intensities].
:::

![The empirical roofline of one A100-40GB PCIe with the kernel catalogue placed at its operational
intensity; percentages are fractions of the measured roof at that intensity. The dashed line is
the datasheet roofline. Measured on 2x A100-40GB (PCIe), one GPU, torch 2.14.1, CUDA 13.0, drawn by
`figures.py` of [@lab:pytorch] from the JSON that `bench.py`
writes.](figures/ch02_pytorch_hardware-11.svg){#fig:pytorch:empirical}

Four observations are worth the trouble of the measurement. First, the streaming kernels (add, row
sum) sit on the bandwidth roof, as the model predicts: their cost is their bytes. Second, the GEMV
and the $M=16$ GEMM take the same 119 μs. Sixteen tokens cost the same as one, because both read
the same 134 MB weight and the extra arithmetic is free below the ridge; this is the batching
argument of the previous section, measured. Third, the GEMM at $M=256$ sits just right of the
ridge at 73% of its roof: near the ridge neither unit can idle while the other works, and the
max-model of ([-@eq:pytorch:roofline]) is at its most optimistic, as the proposition above warned.
Fourth, two kernels are well below their roofs for reasons the model cannot see. PyTorch's fused
prefill attention reaches 135 TFLOP/s, 62% of the GEMM roof (Dao, 2023, reports 50--73% of the A100
peak for FlashAttention-2 across shapes; the online softmax and the masking cost non-matmul
instructions that a GEMM does not issue). The softmax reaches half the bandwidth roof even though
Nsight Compute shows it moving the compulsory traffic and no more (268 MB read, 257 MB written):
the missing half is time, not bytes, which is the subject of the next section.

### What the roofline hides {#sec:pytorch:hides}

The roofline is a bound on throughput in steady state. Everything that is not a steady stream of
FLOPs or bytes is invisible to it, and on a GPU four such costs are routinely larger than the
kernel's useful work.

**Latency and Little's law.** A load from HBM takes several hundred nanoseconds to return. To
sustain bandwidth $\beta$ with latency $\lambda$, a memory system must have $\beta\lambda$ bytes in
flight at all times (Little, 1961). A kernel that issues a load and waits for it before issuing the
next keeps one request in flight per thread and reaches a small fraction of $\beta$ no matter how
few bytes it moves in total.

::: napkinmath
[How many bytes must be in flight on an A100]{.box-title}

Assume a loaded HBM latency of about 500 ns (the right value for our card is measured in
[@ch:gpuarch]). To sustain the measured 1.37 TB/s the chip needs
$1.37\times10^{12}\times500\times10^{-9}\approx 685$ KB of loads outstanding, about 6.3 KB on each of
the 108 SMs. A warp issuing 16-byte vector loads requests 512 bytes per instruction, so every SM
must keep about a dozen such warp-loads outstanding at every moment. A softmax block that spends
part of its life in barrier-separated reductions, issuing no loads, drops below that line, and the
bandwidth it achieves drops with it.
:::

**Occupancy.** The way a GPU hides latency is to switch among many resident warps. How many warps
fit on an SM is limited by the registers and shared memory each block uses; a kernel with too few
resident warps (low *occupancy*) or too little independent work per warp cannot keep enough
requests in flight. [@ch:cuda] measures this trade-off directly.

**Launch overhead and small kernels.** Each kernel launch costs several microseconds of CPU and
driver time, and a kernel whose useful work is shorter than that is bound by neither roof. The
1 MiB copy above is the measured example: 90% of its time is fixed cost. [@sec:pytorch:async]
returns to this regime, and CUDA graphs and fusion are its remedies.

**Quantisation, tails and clocks.** A GEMM is cut into output tiles that are distributed over the
108 SMs in *waves*. A problem with fewer tiles than SMs leaves SMs idle, and one whose tile count
is slightly above a multiple of 108 runs a last, nearly empty wave (*wave quantisation*). The
$n=1{,}024$ GEMM of [@tab:pytorch:roofs] is both kinds of small: 10 μs of work at the measured roof,
only 64 output tiles of $128\times128$ for 108 SMs, plus the fixed per-call cost, which together
hold it to 20% of peak. The clock itself varies with power, as the pitfall above showed, and with
temperature.

Each of these costs has a counter that measures it, and the tools that read those counters (Nsight
Compute, CUPTI and their equivalents on other vendors' hardware) are the subject of
[@ch:profiling]. The roofline tells you how far a kernel is from where it could be; the counters
tell you why.

::: deepdive
[Reading the research: Williams, Waterman and Patterson (2009), "Roofline"]{.box-title}

**The cost model, in our notation.** The paper models a kernel on a multicore processor by two
numbers, its floating-point work and its DRAM traffic after cache filtering, and a machine by two
more, peak FLOP/s and peak *sustainable* DRAM bandwidth (measured with a stream-like benchmark, not
taken from the datasheet). Attainable performance is ([-@eq:pytorch:attain]), $\min(\pi, I\beta)$,
with $I$ the operational intensity of [@sec:pytorch:oi]. Two refinements make it a diagnostic
rather than a bound. *Ceilings* below each roof mark what is attainable without a given
optimisation (on the compute side, for example, without SIMD or without balanced multiplies and
adds; on the bandwidth side, without unit-stride access or software prefetching), so the ceiling a
kernel sits under names the optimisation it lacks. And the intensity of a kernel is not fixed:
better blocking for caches moves it to the right.

**What our code reproduces.** The paper's method is exactly the one in `bench.py` of
[@lab:pytorch]: a streaming probe for $\beta$, a dense kernel for $\pi$, analytic intensities for
the kernels, and the kernels' measured rates plotted against the roof. The paper's machines,
four multicore processors of 2008 from Intel, AMD, Sun and IBM, sustained less than their nominal
DRAM bandwidth on the streaming probe; ours sustains 88% (1,369 of 1,555 GB/s), and its compute
roof is 70% of nominal on realistic data. The paper's kernels (sparse matrix--vector product, a
lattice-Boltzmann code, a stencil and a 3-D FFT) had low operational intensities, between about
0.2 and 1.6 FLOP/byte, the range of our elementwise, reduction and GEMV kernels, and the paper's
central point carries over unchanged to LLM decode: for such kernels, effort spent on
arithmetic is wasted until the bytes are reduced.

**What the evaluation leaves out.** The model has no notion of time other than throughput: latency,
launch overhead, synchronisation and load imbalance are absent, which is why our softmax sits at
half its roof with no ceiling to explain it. It has one compute roof, while a GPU has several
(tensor cores per data type, the much slower non-tensor-core units, special-function units for
exponentials), and one memory level, while fused kernels live or die by L2 and shared-memory
bandwidth; hierarchical rooflines (Yang, Kurth and Williams, 2020) add the levels. Its peak is a
constant, while ours depends on the data through the power limit. And it is a single-node model:
communication between devices, the bottleneck of [@ch:distributed] on our PCIe-only box, needs a
roof of its own.
:::

::: systemslens
[Reading the rest of the book through the roofline]{.box-title}

Many optimisations in later chapters are roofline arguments. FlashAttention ([@ch:attention])
raises attention's intensity by never writing the $T\times T$ score matrix to HBM. The KV cache
([@ch:inference]) trades recomputation (FLOPs) for memory traffic. Grouped-query attention
([@ch:modern]) shrinks the bytes read per decoded token. Batching and continuous batching
([@ch:serving]) raise decoding's $I$ by sharing weight reads. Weight quantisation
([@ch:efficient]) cuts $Q$ directly. Speculative decoding verifies several tokens per weight
read. Each is a move to the right on [@fig:pytorch:roofline], or a reduction of $Q$ at
fixed $F$.
:::

## The execution model: asynchrony, overhead and compilation {#sec:pytorch:async}

### Kernels are launched asynchronously

When Python calls a CUDA operation, PyTorch does the bookkeeping on the CPU (argument checks,
dispatch, allocation), enqueues the kernel on a *CUDA stream*, and returns immediately, before the
GPU has done any work. The CPU races ahead, enqueuing more kernels, while the GPU drains the queue
in order. When the CPU is ahead, launch overhead is hidden and the GPU is always busy. When
individual kernels are tiny (small tensors, many elementwise ops), the GPU finishes each kernel
faster than the CPU can enqueue the next, and the program becomes *overhead-bound*: the GPU idles
between kernels ([@fig:pytorch:async]).

![Asynchronous execution. (a) With large kernels, the CPU enqueues work faster than the GPU consumes
it, so launch overhead is hidden. (b) With tiny kernels, each launch costs more CPU time than the
GPU spends executing, leaving the GPU idle. (c) Fusion and CUDA graphs collapse many launches into
one.](figures/ch02_pytorch_hardware-10.svg){#fig:pytorch:async}

::: napkinmath
[When does launch overhead matter?]{.box-title}

A Python-level PyTorch op costs on the order of 5--20 μs of CPU time; a kernel that moves 1 MB at
3.35 TB/s runs in 0.3 μs. Decoding one token of Qwen3-0.6B touches 28 layers with a dozen or more
small kernels each: several hundred launches, i.e. milliseconds of CPU time, versus a bandwidth
floor of about 0.36 ms ($1.2\,\text{GB}/3.35\,\text{TB/s}$). Eager-mode decoding of small models is
therefore often CPU-bound by a large factor. This is why serving engines capture the decode step in
a CUDA graph and why [@ch:inference] shows large speedups from `torch.compile` on small
models.
:::

### Synchronisation points

Because work is asynchronous, anything that needs a *value* computed on the GPU forces the CPU to
wait for the queue to drain: `.item()`, `.tolist()`, `print(tensor)`, `.cpu()`, and Python control
flow that depends on tensor values (`if loss > 0:`). An innocent `loss.item()` for logging every
step serialises CPU and GPU and destroys overlap. Log every $k$ steps, or accumulate on the device.
The same asynchrony makes naive timing wrong: `time.time()` around a CUDA call measures enqueue
time, not execution time, unless the device is synchronised first
([@sec:pytorch:profiling]).

### Streams and overlap

A stream is an in-order queue of kernels and copies. Work in different streams may overlap, which is
how communication (NCCL kernels on their own stream), host-to-device copies (with pinned memory) and
compute run concurrently. DDP's bucketed all-reduce and the loader's `non_blocking` copies exploit
this; in [@ch:distributed; @ch:serving] we schedule overlap explicitly.

### The caching allocator

`cudaMalloc` and `cudaFree` are slow and synchronising, so PyTorch keeps freed blocks in a cache and
reuses them. Consequently `nvidia-smi` reports the cache's size rather than live tensors, and
`torch.cuda.memory_allocated()` versus `torch.cuda.memory_reserved()` tell the two apart. Allocation
patterns with many differently sized tensors (variable sequence lengths are the classic case)
fragment the cache, which can produce out-of-memory errors with plenty of nominally free memory. The
KV-cache designs of [@ch:inference; @ch:serving] preallocate to avoid exactly this.

### Graphs and compilation

Eager execution is flexible but pays per-op overhead and cannot optimise across operations. Two
mechanisms recover performance. **CUDA graphs** record a fixed sequence of kernel launches once and
replay it with a single launch; they require static shapes and static memory addresses, which is why
they suit the fixed-shape decode step of a serving engine. **`torch.compile`** captures Python-level
programs into a graph (TorchDynamo), derives the backward graph ahead of time (AOTAutograd), and
generates fused kernels (TorchInductor, emitting Triton on GPUs and C++/OpenMP on CPUs), optionally
using CUDA graphs (`mode="reduce-overhead"`). The first call is slow, since it compiles, and
changing input shapes can trigger recompilation. We use compilation for inference in
[@ch:inference] and for training in [@ch:pretrain].

``` {#lst:pytorch:compile .python caption="Compiling a module. Same semantics, fewer and larger kernels."}
model = MLP(4096, 4096, hidden=(16384,)).to(device)
fast = torch.compile(model)                    # default: fuse ops, Triton kernels
x = torch.randn(64, 4096, device=device)
fast(x)                                        # first call compiles (seconds)
fast(x)                                        # later calls are fast
fast_rt = torch.compile(model, mode="reduce-overhead")  # + CUDA graphs
```

## Measuring and profiling {#sec:pytorch:profiling}

Performance claims without careful measurement are folklore. Three rules make timings trustworthy:
**warm up** (the first calls pay for allocator growth, cuBLAS heuristic selection, or compilation,
and an idle GPU's clock needs a fraction of a second of load to ramp up, so on a GPU warm up for a
fixed time rather than a fixed number of calls);
**synchronise** before reading the clock (asynchrony again); and **repeat and take the median**
(robust to noise from other processes). `llmsys.profiling.benchmark` implements exactly this:

``` {#lst:pytorch:bench .python caption="Honest timing (from `llmsys/profiling.py`)."}
def benchmark(fn, warmup=3, iters=10, min_time=0.0):
    for _ in range(warmup):
        fn()
    _sync()                                    # drain queued GPU work
    times, spent = [], 0.0
    while len(times) < iters or spent < min_time:
        t0 = time.perf_counter()
        fn()
        _sync()                                # wait for this call's kernels
        dt = time.perf_counter() - t0
        times.append(dt); spent += dt
    return statistics.median(times)
```

From a time, derive rates and compare them with the roofline. For a GEMM, achieved FLOP/s $= 2mnk/t$
and achieved bandwidth $= b(mk+kn+mn)/t$:

``` {#lst:pytorch:place .python caption="Placing a measured kernel on the roofline."}
from llmsys.profiling import benchmark, matmul_stats, Roofline
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
t = benchmark(lambda: a @ b)
s = matmul_stats(4096, 4096, 4096, t, bytes_per_el=2)
roof = Roofline(peak_flops=989e12, peak_bw=3.35e12)
print(f"{s['flops_per_s']/1e12:.0f} TFLOP/s of "
      f"{roof.attainable(s['intensity'])/1e12:.0f} attainable "
      f"({roof.bound(s['intensity'])}-bound)")
```

When a whole training step is slow, the question is *which* kernels dominate. `torch.profiler`
records every operator and kernel with CPU and GPU times and can export a Chrome/Perfetto trace for
timeline inspection; `llmsys.profiling.top_ops` returns the most expensive ops as a table:

``` {#lst:pytorch:profiler .python caption="Op-level profile of one training step."}
from torch.profiler import profile, ProfilerActivity
acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
with profile(activities=acts, record_shapes=True) as prof:
    loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("step.json")          # open in ui.perfetto.dev
```

For kernel-level questions (achieved occupancy, memory throughput, tensor-core utilisation, stalls),
NVIDIA's Nsight Systems (timelines across CPU, GPU, NCCL and NVTX ranges) and Nsight Compute
(per-kernel hardware counters, including a roofline view) are the tools of record;
[@app:triton] uses both and [@ch:profiling] builds the whole toolchain, down to the counter
interfaces underneath them.

::: pitfall
[Common measurement mistakes]{.box-title}

\(1\) Timing without synchronisation measures launch time. (2) Timing the first iteration measures
compilation and allocator warm-up. (3) Benchmarking with inputs that fit in L2 cache (50 MB on H100)
overstates bandwidth; use buffers several times larger than L2 or flush it. (4) Reporting a mean
over runs that include a stray outlier: use medians and report spread. (5) Comparing dtypes without
checking that the fast path (tensor cores, tf32) was actually enabled. (6) On shared machines, noisy
neighbours make CPU numbers erratic; repeat until stable.
:::

::: systemslens
[Model FLOPs utilisation]{.box-title}

For an entire training run the headline metric is *model FLOPs utilisation* (MFU): useful model
FLOPs per second (for a dense transformer, $6N$ per training token, ignoring recomputation) divided
by the hardware peak. If a 1.3B model trains at 150,000 tokens/s on 8 H100s, MFU
$=150{,}000\times6\times1.3\times10^9/(8\times989\times10^{12})
\approx 15\%$, a sign of overheads (small kernels, communication, data loading) worth profiling.
Well-tuned large-scale runs report 35--55%. MFU is the training analogue of the roofline fraction
for a single kernel, and `llmsys.flops.mfu` computes it.
:::

## Research frontier {#sec:pytorch:frontier}

This chapter's research area is the performance model itself: how a program written in an eager
framework becomes kernels, and how to predict and explain what those kernels achieve.

### Open problems

**Irregular and dynamic work.** The roofline places a kernel at one intensity. Mixture-of-experts
layers route a different number of tokens to every expert on every step, serving batches are
ragged, and sparse attention reads a data-dependent subset of the cache, so the intensity of the
same layer moves from step to step. A performance model for such workloads predicts a
distribution, and how to build and validate one is open ([@ch:moe; @ch:serving]).

**The memory wall.** Peak FLOP/s has grown much faster than memory bandwidth for two decades
(Gholami et al., 2024). At a fixed precision the ridge point does not rise monotonically: in bf16
it went from 201 FLOP/byte (A100 PCIe) to 295 (H100 SXM), back to about 206 on the H200
(989 TFLOP/s over 4.8 TB/s) and 281 on the B200 (2.25 PFLOP/s over 8 TB/s), because each
generation also bought faster memory. It rises steeply once the lower-precision tensor cores are
counted: the B200's fp8 and fp4 peaks put its ridge near 562 and 1,125 FLOP/byte, and kernels
that move to those formats without shrinking their bytes in proportion fall further to its left. Lower-precision formats, larger caches and processing near memory are all responses; which
of them wins for LLM inference is not settled ([@ch:numerics; @ch:memory]).

**Data movement as the cost.** Ivanov et al. (2021) measured that in a BERT encoder layer, tensor
contractions were 99.8% of the FLOPs but only 61% of the runtime; the rest was memory-bound
normalisation, softmax and elementwise work. Fusion recovers much of that, and how far automatic
fusion can go before hand-written kernels are needed is an active question
([@ch:compilers; @ch:cudakernels]).

**Eager versus captured execution.** PyTorch won by executing eagerly (Paszke et al., 2019) and
now recovers fusion and launch amortisation by capturing graphs (Ansel et al., 2024). Dynamic
shapes, graph breaks and the boundary between captured and eager code are where the remaining
overhead lives ([@ch:compilers]).

**Power and energy.** On a power-capped chip the compute roof depends on the data
([@tab:pytorch:roofs]). An energy roofline, joules per FLOP and per byte (Choi et al., 2013), is
the natural model when the datacentre's power, not its chips, is the budget.

### The works that defined the area

- Williams, Waterman and Patterson (2009): the roofline and operational intensity; performance
  bounded by $\min(\pi, I\beta)$ with ceilings that name missing optimisations.
- Volkov (2016): latency hiding on GPUs as Little's law; occupancy and instruction-level
  parallelism as two ways to keep enough requests in flight.
- Yang, Kurth and Williams (2020): the hierarchical roofline for GPUs, one roof per memory level and
  per compute unit, measured with hardware counters.
- Choi et al. (2013): a roofline for energy, separating the cost of a FLOP from the cost of a byte.
- Paszke et al. (2019): PyTorch's define-by-run design with an asynchronous device queue, the
  execution model of [@sec:pytorch:async].
- Ansel et al. (2024): PyTorch 2's graph capture and compiler, which bring fusion and CUDA graphs
  to eager programs.
- Ivanov et al. (2021): data movement, not FLOPs, dominates transformer training time on GPUs.
- Dao et al. (2022): FlashAttention, the canonical example of raising operational intensity by
  restructuring an exact algorithm around the memory hierarchy.
- Gholami et al. (2024): the memory wall quantified for AI hardware and models.

### Assumptions that are likely to break

- *One compute roof.* Each data type has its own tensor-core roof, and newer GPUs keep raising them
  (fp8 doubles the bf16 roof on an H100) while the units that compute exponentials and reductions
  grow slowly: an H100 SXM has 989 TFLOP/s of 16-bit matmul but only 3.9 TFLOP/s of special
  functions (Shah et al., 2024). Kernels such as attention become bound by the non-matmul roof.
- *HBM is the level that matters.* Fused kernels are bound by shared memory and L2; multi-GPU jobs
  by the interconnect; offloaded inference by PCIe, which on our box is 60 times slower than HBM.
- *Peak is a constant.* Power caps make it data-dependent; clocks vary with temperature.
- *Overlap must be found by the hardware.* Hopper-class GPUs expose asynchronous bulk copies and
  warp specialisation, so overlapping memory and compute becomes something the kernel programs
  explicitly ([@ch:gpuarch; @ch:cudakernels]).

### How to recognise the next variant

Place a new performance paper by asking: (1) which roof or level it is about (a compute unit and
data type; HBM, L2, shared memory; the interconnect, the host link or power); (2) whether it
changes $F$ (a different algorithm), $Q$ (fusion, compression, caching) or neither (scheduling,
overlap, launch amortisation); (3) which hidden cost of [@sec:pytorch:hides] it attacks;
(4) its granularity (one kernel, an operator graph, a training step, a cluster); (5) the hardware
feature it assumes (a data type, an asynchronous copy engine, a cache size); and (6) whether its
evidence is a model, counters, or end-to-end time. A paper that moves a kernel to the right on the
roofline and one that moves it up towards its roof are different contributions, and the best
papers say which they are.

## Summary {#sec:pytorch:summary}

- PyTorch combines a tensor library, reverse-mode autograd, and deep-learning utilities. Models are
  `nn.Module`s; data flow through `Dataset`/`DataLoader`; training is the five-step loop of forward,
  loss, zero-grad, backward, step.

- Tensors are views onto storage with strides; transposes and slices are free until a kernel needs
  contiguous memory. bf16 is the default LLM format: fp32 range, low precision.

- Backward costs about twice the forward and requires saved activations, hence $6N$ FLOPs per
  training token and activation memory that scales with batch and sequence length.

- DDP replicates the model per GPU, shards the data, and overlaps bucketed gradient all-reduces with
  backward.

- GPUs deliver most FLOPs through tensor cores and feed them through a memory hierarchy. The
  roofline $\min(\pi, I\beta)$ says whether a kernel is limited by compute or bandwidth; most LLM
  operations other than large GEMMs are bandwidth-bound.

- Execution is asynchronous: hidden launch overhead becomes visible with small kernels, and
  synchronisation points serialise CPU and GPU. Fusion, CUDA graphs and `torch.compile` reduce
  launches and memory traffic.

- Measure with warm-up, synchronisation and medians; profile to find dominant kernels; summarise
  training efficiency with MFU.

- Operational intensity is FLOPs per byte at one level of the memory hierarchy. A GEMM's intensity
  is $2/b$ times a harmonic mean of its dimensions, so its smallest dimension decides it; decode
  attention sits at $H/G$ FLOP/byte in bf16 whatever the batch; fused prefill attention grows as
  $T/(2b)$.

- Measured on our A100-40GB PCIe, the roofs are 1,369 GB/s (copy) and 217 TFLOP/s (bf16 GEMM on
  random data; 295 on zeros, because of the 250 W power limit), so the ridge is about 158 FLOP/byte.
  Streaming kernels reach 96--97% of their roof; latency, launch overhead and occupancy explain
  the kernels that do not.

## Exercises {#sec:pytorch:exercises}

::: exercise
[Strides]{.box-title}

For a contiguous tensor of shape $(2,3,4)$, write down its strides. What are the shape and strides
after `.transpose(0, 2)`? Explain why `.view(-1)` fails on the transposed tensor while
`.reshape(-1)` succeeds, and what the latter costs.
:::

::: exercise
[bf16 rounding]{.box-title}

Compute the smallest update $\delta$ such that `torch.tensor(1.0, dtype=torch.bfloat16) + delta`
differs from 1.0. What does this imply for applying an update of size $\eta\, g = 10^{-4}$ to a
weight of magnitude 1 stored in bf16? How do fp32 master weights fix this?
:::

::: exercise
[Manual gradients]{.box-title}

Derive $\partial\mathcal{L}/\partial z = a-y$ for sigmoid followed by binary cross-entropy, and
extend it to a batch with the mean loss. Verify your formula against autograd numerically (this is
Task 1 of Part A in [@lab:pytorch]).
:::

::: exercise
[Parameter counting]{.box-title}

Count the parameters of an MLP with layer sizes $784\to512\to256\to10$. How many bytes does it need
for weights, gradients and Adam states in fp32? How many FLOPs does one forward pass on a batch of
128 cost?
:::

::: exercise
[Ridge points]{.box-title}

Compute the ridge point of an A100-80GB SXM (312 TFLOP/s, 2.04 TB/s) and an RTX 4090 (165 TFLOP/s,
1.01 TB/s) in bf16. At what square GEMM size $n$ does each become compute-bound? Which GPU needs a larger
decode batch to stop being memory-bound?
:::

::: exercise
[Intensity of a transformer MLP]{.box-title}

A GPT-style MLP computes $\operatorname{GELU}(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2$ with
$\mathbf{X}\in\mathbb{R}^{BT\times d}$, $\mathbf{W}_1\in\mathbb{R}^{d\times4d}$,
$\mathbf{W}_2\in\mathbb{R}^{4d\times d}$, $d=4096$, bf16. Compute FLOPs and minimal bytes for the
two GEMMs and the GELU as separate kernels for $BT=1$ and $BT=4096$. Which kernels are memory-bound
in each case? What is the saving from fusing GELU into the first GEMM's epilogue?
:::

::: exercise
[DDP communication]{.box-title}

A 7B-parameter model is trained with DDP on 8 GPUs connected by NVLink (assume 300 GB/s effective
all-reduce bandwidth per GPU) and, alternatively, on 8 nodes connected by 100 Gb/s Ethernet.
Estimate the all-reduce time per step for bf16 gradients with a ring algorithm. If a step's compute
takes 1.5 s, which setting can hide communication behind backward? (Also: would the model even fit
for DDP?)
:::

::: exercise
[Launch overhead]{.box-title}

Write a function applying 100 elementwise operations to a $1024$-element tensor and time it eagerly
on a GPU (or CPU) with proper synchronisation. Repeat with a $10^8$-element tensor. Then apply
`torch.compile`. Explain the three timings using [@fig:pytorch:async].
:::

::: exercise
[Profiling]{.box-title}

Profile one training step of the MLP of [@lst:pytorch:compile] with batch sizes 8 and 4096.
Report the top five ops by device time in each case and classify each as compute- or memory-bound
using the roofline. How does the fraction of time in `aten::mm`/`addmm` change?
:::

::: exercise
[MFU sanity check]{.box-title}

A team reports training a 3B model at 2.1 million tokens/s on 64 H100s. Compute the implied MFU. Is
the claim plausible? What would you ask them about how "tokens" and "FLOPs" were counted?
:::

::: exercise
[An L2 roofline]{.box-title}

Extend the lab's empirical roofline to the hierarchical setting of Yang, Kurth and Williams (2020).
(a) Write an L2 probe: copy a buffer small enough that source and destination stay in the 40 MB
L2, many times inside one timed call so that launch overhead is amortised, and report the L2
bandwidth. (b) Predict which kernels of [@tab:pytorch:points] the L2 roof could limit, using their
working sets. (c) With Nsight Compute's DRAM and L2 byte counters, measure $Q_{\text{HBM}}$ and
$Q_{\text{L2}}$ of the $M=256$ GEMM and of the prefill attention kernel, place both on both roofs,
and say which roof binds.
:::

::: exercise
[Predict, then measure: power and the compute roof]{.box-title}

(a) Predict how the SM clock and the power draw differ between the bf16 GEMM of `bench.py` on
random normal data and on zeros; then measure both by sampling
`nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv -lms 100` while each runs for ten
seconds. (b) Repeat with inputs whose values are random but whose low 4 mantissa bits are zero,
and explain the result with the toggling argument of the pitfall in
[@sec:pytorch:empirical]. (c) In the spirit of the energy roofline of Choi et al. (2013), estimate
from your measurements the energy per bf16 FLOP and per HBM byte on the A100, and the operational
intensity at which a kernel spends equal energy on the two. (d) Predict how much of this effect
you would see on an H100 SXM with a 700 W limit, and why.
:::

## Lab 2: PyTorch essentials and an empirical roofline {#sec:pytorch:lab}

::: labbox
[PyTorch essentials and an empirical roofline]{.box-title}

**Code:** `code/labs/lab02_pytorch_roofline`

**Goals**

Part A makes the PyTorch workflow second nature: gradients by hand versus autograd, an `nn.Module`,
a `DataLoader`, a training loop, checkpoints, and DDP on CPU processes. Part B measures your own
machine: achieved FLOP/s and bandwidth, an empirical roofline, number formats, views versus copies,
and an op-level profile of a training step. Part C makes the roofline rigorous: the operational
intensity of every kernel class, a copy-kernel bandwidth probe, and the kernel catalogue of
[@tab:pytorch:intensities] measured and placed on your own roofline.

**Setup**

``` bash
cd code
pytest -q labs/lab02_pytorch_roofline                      # grades starter.py
LAB_IMPL=solution pytest -q labs/lab02_pytorch_roofline    # reference
```

**Tasks**

*Part A: PyTorch.*

1.  `logistic_regression_grads(x, y, w, b)`: hand-derived gradients of mean BCE; the test compares
    them with autograd to $10^{-6}$.

2.  `build_mlp(in_dim, hidden, out_dim)`: an `nn.Sequential` returning logits; the test checks
    shapes and the exact parameter count.

3.  `train_classifier` and `accuracy`: the loop of [@lst:pytorch:loop] with SGD; it must
    reach over 90% test accuracy on the two-blob `ToyDataset`.

4.  `save_and_reload`: a `state_dict` round trip that reproduces the accuracy exactly.

5.  Run `ddp_demo.py --world-size 2`. The provided test checks that the two replicas end with
    bit-identical weights. Explain why they do.

6.  `manual_mlp_backward`: the reverse sweep of [@lst:pytorch:manualbwd]; the test compares every
    parameter's gradient with autograd to $10^{-6}$.

7.  `saved_activation_bytes`: count the bytes autograd saves for backward with saved-tensor hooks,
    with and without `torch.utils.checkpoint`; the test checks both counts exactly. `bench.py`
    adds the peak-memory measurement of [@tab:pytorch:ckpt] on a GPU.

*Part B: the machine.*

1.  `matmul_intensity`, `elementwise_intensity`, `roofline_attainable`: the formulas of
    [@sec:pytorch:roofline].

2.  `measure_matmul_sweep` and `measure_bandwidth_sweep`: timed sweeps using
    `llmsys.profiling.benchmark`; `empirical_roofline` picks the best observed FLOP/s and bytes/s.

3.  `shares_storage`: detect whether a tensor is a view, using storage pointers.

4.  `format_table`: max, epsilon and smallest normal of fp32/bf16/fp16 from `torch.finfo`.

5.  `profile_train_step`: profile forward, backward and step and return the top ops.

*Part C: the roofline, rigorously.*

1.  `ridge_point`: $I^\star=\pi/\beta$; the test checks 200.6 FLOP/byte for the A100-40GB PCIe
    datasheet.

2.  `reduction_intensity`: a row-wise sum, $rc/(b(rc+r))$.

3.  `attention_flops`, `attention_bytes`, `attention_intensity`: fused attention at minimum
    traffic, with grouped KV heads and an exact causal count (including chunked prefill, where
    $T_q<T_k$). The tests check the limits ([-@eq:pytorch:prefillI]) and
    ([-@eq:pytorch:decodeI]).

4.  `measure_copy_bandwidth`: the copy probe of [@lst:pytorch:roofprobe], timed with the provided
    `time_stats` (time-based warm-up; median, p10 and p90).

5.  `measure_kernel_points`: build and time the kernel catalogue (add, row sum, softmax, GEMV,
    GEMMs with $M\in\{16, 256, K\}$, causal prefill attention and GQA decode attention) and
    return one roofline point per kernel. A `scale` argument shrinks every shape so that the test
    runs on a CPU.

**Measurements**

Run `python labs/lab02_pytorch_roofline/run.py` on your machine (`--device cuda --dtype bf16` on a
GPU) to produce `roofline.png`. Then run `python labs/lab02_pytorch_roofline/bench.py --out-dir
results`, which measures both roofs and the kernel catalogue and writes `roofline.json` and
`roofline.md` with the hardware and software versions; `figures.py --json results/roofline.json
--out-dir results` draws [@fig:pytorch:empirical] for your machine. On our A100-40GB PCIe the
whole run takes about 15 seconds and reproduces [@tab:pytorch:roofs; @tab:pytorch:points]; the
reference output is in `reference_a100/`. Expect GEMM points to climb the bandwidth slope and flatten at the
compute roof for $n$ in the low thousands, and elementwise points to sit near $I=1/8$ (fp32) or
$1/4$ (bf16) on the bandwidth slope. On a GPU, compare your empirical peaks with the datasheet.
Reaching 60--80% of peak bandwidth and 60--80% of peak bf16 FLOP/s with plain PyTorch is typical. On
a laptop CPU, expect noisy results; repeat runs and note the variance.

**Deliverables**

Passing tests; your roofline plot annotated with the ridge point; a short table of the top five ops
in a training step at batch sizes 8 and 4096 with a one-line roofline explanation of each; your
empirical roofline with the kernel catalogue, and for every kernel below 80% of its roof one
sentence naming the cost of [@sec:pytorch:hides] you think is responsible.

**Stretch goals**

\(1\) Add a bf16 and tf32 sweep and quantify the tensor-core speedup. (2) Time 100 tiny elementwise
ops eagerly, with `torch.compile`, and with `mode="reduce-overhead"`. (3) Use Nsight Compute's
roofline section on one GEMM and one elementwise kernel and compare with your empirical roofline.
(4) Measure DDP scaling efficiency for 1, 2 and 4 processes on a larger MLP and relate it to
all-reduce volume. (5) Add an unfused attention (explicit `softmax(Q @ K.T) @ V`) to the catalogue
and confirm that it lands near the 62 FLOP/byte of [@tab:pytorch:intensities], left of the ridge.
:::
