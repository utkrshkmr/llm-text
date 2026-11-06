# Preface {#preface .unnumbered}

This book follows a language model from raw text to a deployed reasoning system, and keeps asking,
at every step, what the machine underneath is doing.

The first arc goes from text to a working GPT: tokenisation, attention, the transformer block,
pretraining, and fine-tuning for classification and for following instructions. The second arc
starts from a pretrained model and turns it into a *reasoning* model: generating text efficiently,
grading answers with verifiers, spending more compute at inference time, scoring and refining
answers, training with policy gradients, RLHF, preference optimization and reinforcement learning
with verifiable rewards, and distilling the result into something cheaper. Ideas that both arcs need
(sampling, loading pretrained weights, evaluation) appear once, where they fit best, and later
chapters refer back to them.

Implementing a model from scratch shows you *what* it computes. A systems engineer also needs to
know *where the time, memory and bandwidth go*, and how real training and serving stacks are built
around that answer. So every chapter carries a systems layer:

- **Systems Lens** boxes connect each modelling idea to its cost on real hardware: arithmetic
  intensity, memory traffic, kernel fusion, communication volume.

- **Napkin Math** boxes do back-of-the-envelope calculations with real accelerator numbers, so you
  build intuition for orders of magnitude.

- Whole chapters cover the systems themselves: single-GPU performance engineering
  ([@ch:perf]), distributed and sharded training ([@ch:distributed; @ch:sharding]),
  serving engines with continuous batching and paged KV caches ([@ch:serving]), quantisation
  and speculative decoding ([@ch:efficient]), on-device and CPU inference
  ([@ch:ondevice]), retrieval and compound systems ([@ch:retrieval]), the
  infrastructure behind RL for reasoning ([@ch:rlsys]), and GPU kernels in Triton
  ([@app:triton]) and accelerators beyond the GPU ([@app:accel]).

- A whole part goes beneath Python: C++ extensions and custom operators for PyTorch
  ([@ch:cpp]), CUDA kernels from first principles ([@ch:cuda]), attention and
  mixture-of-experts kernels ([@ch:cudakernels]), and ML compilers and runtimes
  ([@ch:compilers]), with JAX and XLA for comparison ([@app:jax]). The
  performance-critical parts of the companion code are written in C++ and CUDA, and the book teaches
  how to connect them to Python: bindings, builds, testing, packaging.

Throughout, the emphasis is on how things are actually *set up and run*: code, configurations, data
formats, launch scripts, logging, and the systems around the maths.

## How to read this book {#how-to-read-this-book .unnumbered}

The book has eight parts. Part I sets up vocabulary and a performance mental model. Part II builds
GPT-2 and then a modern Qwen3-style architecture from scratch, maps the design space around it,
looks beyond softmax attention, and then widens the view: encoders and encoder--decoders,
mixture-of-experts models end to end, and the anatomy of today's open models. Part III is
pretraining at scale: optimizers, the training loop, data, scaling laws, single-GPU performance,
distributed and sharded training, and a calculus for sharded computation. Part IV goes from Python
to the metal: C++ for PyTorch, CUDA kernels from first principles, attention and MoE kernels, and ML
compilers. Part V fine-tunes the model for classification and for following instructions. Part VI is
about inference: how a single request is generated, how a server multiplexes thousands of them, how
to make each token cheaper, how to run models on CPUs and devices, and how retrieval, tools and
agents turn a model into a compound system. Part VII is post-training: evaluation, inference-time
scaling, policy gradients, RLHF and PPO, DPO, GRPO, RL systems, distillation, and alignment.
Part VIII is nine research-grade capstone projects, each with acceptance tests and a reference
solution, sized for two A100 GPUs connected over PCIe.

If you already know transformers well, you can skim Part II for the Systems Lens boxes and the labs.
If your goal is inference serving, a fast path is [@ch:pytorch] $\rightarrow$
[@ch:modern] $\rightarrow$ [@ch:perf] $\rightarrow$ Part VI $\rightarrow$
[@ch:scaling] $\rightarrow$ [@ch:rlsys]. If your goal is kernels and systems
programming, read [@ch:pytorch] $\rightarrow$ [@ch:perf] $\rightarrow$ Part IV
$\rightarrow$ [@ch:shardcalc] $\rightarrow$ [@app:triton]. Each capstone project in
Part VIII lists the chapters and labs to work through first.

## Labs {#labs .unnumbered}

Every chapter ends with a lab. Labs live in `code/labs/labNN_*` and share the `llmsys` package in
`code/llmsys`. Each lab has a `starter.py` with `TODO`s, a reference `solution.py`, and a test file.
The tests run on a laptop CPU in under a minute with tiny model configurations:

``` bash
cd code
pip install -r requirements.txt
LAB_IMPL=starter  python -m pytest labs/lab05_attention   # grade your work
LAB_IMPL=solution python -m pytest labs/lab05_attention   # check the reference
```

Several labs have a C++ or CUDA extension: the performance-critical operators live in `code/csrc`
and are built into the `llmsys._C` module with `pip install -e .` ([@app:setup]); every such
operator keeps a pure-PyTorch reference and a backend switch, so the labs run without a compiler and
the C++ task is an explicit extension.

Each lab also has a *Measurements* section. Those experiments are meant to be run on real hardware,
ideally a GPU, with real checkpoints such as GPT-2 and Qwen3-0.6B, and they ask you to explain what
you see using the cost models from the chapter. Numbers in this book are either derived analytically
from published hardware specifications or explicitly labelled as measured; the reference machine for
measured numbers in this edition is a workstation with two A100-40GB GPUs connected over PCIe,
without NVLink. Your own measurements are the ones that count.

## Conventions {#conventions .unnumbered}

Tensors are written with their shapes, as in $\mathbf{X}\in\mathbb{R}^{B\times T\times d}$, where
$B$ is the batch size, $T$ the sequence length, and $d$ the model width $d_{\text{model}}$. We write
$L$ for the number of layers, $H$ for query heads, $G$ for key/value heads, $d_{h}$ for the head
dimension, $d_{\text{ff}}$ for the MLP width, and $V$ for the vocabulary size. "FLOPs" counts
floating-point operations, with a multiply--add counted as two. Bandwidths are in bytes per second
and use decimal prefixes (1 TB/s $=10^{12}$ B/s).
