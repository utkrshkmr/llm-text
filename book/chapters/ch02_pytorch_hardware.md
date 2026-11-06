# PyTorch and the Machine Underneath {#ch:pytorch}

Every model in this book is a PyTorch program, and every PyTorch program is a stream of *kernels*
launched onto an accelerator. This chapter covers both levels. The first half is a working
introduction to PyTorch: tensors, automatic differentiation, modules, data loaders, the training
loop, checkpoints, and running on one or several GPUs. A reader fluent in PyTorch can skim it, but
should read the Systems Lens boxes, which connect each API to what happens on the hardware. The
second half opens the box: how a GPU is organised, why most kernels are limited by memory bandwidth
rather than arithmetic, how the *roofline model* predicts which, how PyTorch's asynchronous
execution model hides (or fails to hide) latency, and how to measure all of this honestly. These are
the tools we use to reason about every optimisation in later chapters.

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
copies. Lab 2 asks you to detect views versus copies by comparing storage pointers.
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
  ([@ch:efficient]).

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
that product from the loss backwards, which costs about the same as one more forward pass no matter
how many parameters there are. This is the right mode for a scalar loss and millions of parameters.

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
depth, and why activation checkpointing (recomputing them during backward, [@ch:pretrain])
trades FLOPs for memory.
:::

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

## The roofline model {#sec:pytorch:roofline}

A kernel needs time for its arithmetic and time to move its data. If the two overlap perfectly (a
good first approximation for well-written kernels), its runtime is bounded below by the larger:
$$\begin{equation}
t \;\ge\; \max\!\left(\frac{F}{\pi},\; \frac{Q}{\beta}\right),
\label{eq:pytorch:roofline}
\end{equation}$$ where $F$ is the number of FLOPs, $Q$ the bytes moved to and from HBM, $\pi$ the
peak FLOP/s and $\beta$ the peak bandwidth. Dividing $F$ by $t$ gives the attainable throughput as a
function of the kernel's *arithmetic intensity* $I = F/Q$ (FLOPs per byte): $$\begin{equation}
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

### Arithmetic intensity of the kernels that matter

##### Matrix multiplication.

For $\mathbf{C}=\mathbf{A}\mathbf{B}$ with $\mathbf{A}\in\mathbb{R}^{m\times k}$,
$\mathbf{B}\in\mathbb{R}^{k\times n}$ stored with $b$ bytes per element, $F=2mnk$ and, at best,
$Q=b(mk+kn+mn)$ (each operand read once, the result written once), so $$\begin{equation}
I_{\text{GEMM}} = \frac{2mnk}{b(mk+kn+mn)}\;\xrightarrow{m=n=k}\;\frac{2n}{3b}.
\end{equation}$$ In bf16 a square GEMM has $I=n/3$: it crosses the H100 ridge near $n\approx885$.
Large training GEMMs are compute-bound; small ones are not.

##### Matrix--vector products.

With $m=1$ (one token's activations times a weight matrix), $I=2nk/(b(k+kn+n))\approx 2/b$: about
1 FLOP/byte in bf16. This is the operation at the heart of batch-1 decoding, and it sits at about
$1/300$ of the H100's ridge point. With a batch of $B$ tokens sharing the same weights, $m=B$ and
$I\approx 2B/b$ for $B\ll k,n$; batching is the lever that slides decoding up the roofline.

##### Elementwise operations.

`y = x + z` in bf16 does 1 FLOP per element and moves 6 bytes: $I=1/6$. Activation functions,
residual additions, dropout, and the scaling in attention are all like this. Normalisations and
softmax perform a handful of FLOPs per element over two passes, $I$ of order 1--10. All are deeply
memory-bound, so the only way to speed them up is to avoid materialising their inputs and outputs in
HBM, by *fusing* them into neighbouring kernels.

::: napkinmath
[Fusion pays for memory-bound chains]{.box-title}

Take a bf16 activation tensor of $8\times4096\times4096$ elements (256 MiB) passing through
bias-add, GELU and dropout as three separate kernels: each reads and writes the tensor, about
$3\times2\times256$ MiB $\approx$ 1.6 GB of traffic, $\approx$`<!-- -->`{=html}0.48 ms at 3.35 TB/s.
Fused into one kernel it reads once and writes once: 0.54 GB, $\approx$`<!-- -->`{=html}0.16 ms, a
$3\times$ speedup with identical FLOPs. This is exactly what `torch.compile`'s code generator does
automatically ([@sec:pytorch:async]) and what we do by hand in Triton in
[@app:triton].
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
**warm up** (the first calls pay for allocator growth, cuBLAS heuristic selection, or compilation);
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
[@app:triton] uses both.

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
Lab 2, Task A1).
:::

::: exercise
[Parameter counting]{.box-title}

Count the parameters of an MLP with layer sizes $784\to512\to256\to10$. How many bytes does it need
for weights, gradients and Adam states in fp32? How many FLOPs does one forward pass on a batch of
128 cost?
:::

::: exercise
[Ridge points]{.box-title}

Compute the ridge point of an A100 (312 TFLOP/s, 2.04 TB/s) and an RTX 4090 (165 TFLOP/s, 1.01 TB/s)
in bf16. At what square GEMM size $n$ does each become compute-bound? Which GPU needs a larger
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

## Lab 2: PyTorch essentials and an empirical roofline {#sec:pytorch:lab}

::: labbox
[PyTorch essentials and an empirical roofline]{.box-title}

**Code:** `code/labs/lab02_pytorch_roofline`

**Goals**

Part A makes the PyTorch workflow second nature: gradients by hand versus autograd, an `nn.Module`,
a `DataLoader`, a training loop, checkpoints, and DDP on CPU processes. Part B measures your own
machine: achieved FLOP/s and bandwidth, an empirical roofline, number formats, views versus copies,
and an op-level profile of a training step.

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

*Part B: the machine.*

1.  `matmul_intensity`, `elementwise_intensity`, `roofline_attainable`: the formulas of
    [@sec:pytorch:roofline].

2.  `measure_matmul_sweep` and `measure_bandwidth_sweep`: timed sweeps using
    `llmsys.profiling.benchmark`; `empirical_roofline` picks the best observed FLOP/s and bytes/s.

3.  `shares_storage`: detect whether a tensor is a view, using storage pointers.

4.  `format_table`: max, epsilon and smallest normal of fp32/bf16/fp16 from `torch.finfo`.

5.  `profile_train_step`: profile forward, backward and step and return the top ops.

**Measurements**

Run `python labs/lab02_pytorch_roofline/run.py` on your machine (`--device cuda --dtype bf16` on a
GPU) to produce `roofline.png`. Expect GEMM points to climb the bandwidth slope and flatten at the
compute roof for $n$ in the low thousands, and elementwise points to sit near $I=1/8$ (fp32) or
$1/4$ (bf16) on the bandwidth slope. On a GPU, compare your empirical peaks with the datasheet.
Reaching 60--80% of peak bandwidth and 60--80% of peak bf16 FLOP/s with plain PyTorch is typical. On
a laptop CPU, expect noisy results; repeat runs and note the variance.

**Deliverables**

Passing tests; your roofline plot annotated with the ridge point; a short table of the top five ops
in a training step at batch sizes 8 and 4096 with a one-line roofline explanation of each.

**Stretch goals**

\(1\) Add a bf16 and tf32 sweep and quantify the tensor-core speedup. (2) Time 100 tiny elementwise
ops eagerly, with `torch.compile`, and with `mode="reduce-overhead"`. (3) Use Nsight Compute's
roofline section on one GEMM and one elementwise kernel and compare with your empirical roofline.
(4) Measure DDP scaling efficiency for 1, 2 and 4 processes on a larger MLP and relate it to
all-reduce volume.
:::
