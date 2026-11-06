# Preface {#preface .unnumbered}

*Big Book of LLMs: A Systems-First Textbook* follows a language model from raw text to a deployed
reasoning system, and keeps asking, at every step, what the machine underneath is doing. Its goal is larger than explaining today's
systems: it is to leave you able to follow the research in LLM systems, judge it, reproduce it on
your own hardware, and add to it.

The first arc goes from text to a working GPT: tokenisation, attention, the transformer block,
pretraining, and fine-tuning for classification and for following instructions. The second arc
starts from a pretrained model and turns it into a *reasoning* model: generating text efficiently,
grading answers with verifiers, spending more compute at inference time, scoring and refining
answers, training with policy gradients, RLHF, preference optimisation and reinforcement learning
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

- Whole chapters cover the systems themselves: the machine and how to measure it
  ([@ch:machine]), the numbers it computes with and how to scale them ([@ch:numerics; @ch:quant]), single-GPU performance engineering ([@ch:perf]), distributed and sharded
  training ([@ch:distributed; @ch:sharding; @ch:shardcalc]), serving engines with continuous
  batching ([@ch:serving]), KV-cache management ([@ch:kvcache]), quantisation and speculative
  decoding ([@ch:efficient]), on-device and CPU inference ([@ch:ondevice]), retrieval and compound
  systems ([@ch:retrieval]), a measured study of data movement in long-running reasoning workloads
  ([@ch:movestudy]), the infrastructure behind RL for reasoning ([@ch:rlsys]), GPU kernels in
  Triton ([@app:triton]) and accelerators beyond the GPU ([@app:accel]).

- A whole part goes beneath Python: C++ extensions and custom operators for PyTorch
  ([@ch:cpp]), the architecture of GPUs and other accelerators ([@ch:gpuarch]), CUDA kernels from
  first principles ([@ch:cuda]), attention and mixture-of-experts kernels ([@ch:cudakernels]), the
  profiling toolchain ([@ch:profiling]), ML compilers ([@ch:compilers]), and memory, data movement
  and interconnects ([@ch:memory]), with JAX and XLA for comparison ([@app:jax]). The
  performance-critical parts of the companion code are written in C++ and CUDA, and the book teaches
  how to connect them to Python: bindings, builds, testing, packaging.

Throughout, the emphasis is on how things are actually *set up and run*: code, configurations, data
formats, launch scripts, logging, containers, and the systems around the maths.

## Learning to do the research {#learning-to-do-the-research .unnumbered}

Every topic is developed from first principles, with working code and a number computed or
measured on a real machine, because that is what lets you recognise the same idea when it returns
in a new paper. Every chapter ends, before its summary, with a *Research frontier* section: the
open problems, the handful of papers that define the area and what each changed, the assumptions
that new hardware or models are likely to break, and the axes along which new papers vary, so that
you can place a new one in a minute. Every chapter also works through one landmark paper in a
*Reading the research* box: its cost model in our notation, which of its numbers our lab code
reproduces and how closely, and what its evaluation leaves out. The exercises include
research-style problems that extend a lab to the setting of a paper, or ask you to predict a
measurement on one machine and then make it. The appendix on doing research ([@app:research])
gathers the method: how to follow the field, read a paper for its cost model, reproduce a result
on a small budget, and write up a finding.

## How to read this book {#how-to-read-this-book .unnumbered}

The book has seven parts, and every part ends with its own capstone projects. Part I sets up the
vocabulary and a performance mental model: the landscape, PyTorch, the machine underneath and how to
measure it, the numbers the machine computes with, and the scales, quantisation grids and
mixed-precision recipes that make narrow formats usable. Part II builds GPT-2 and then a modern
Qwen3-style architecture from scratch, maps the design space around it, looks beyond softmax
attention, and then widens the view: encoders and encoder--decoders, mixture-of-experts models end
to end, and the anatomy of today's open models. Part III is pretraining at scale: optimisers, the
training loop, data, scaling laws, single-GPU performance, distributed and sharded training, and a
calculus for sharded computation. Part IV goes from Python to the metal: C++ for PyTorch, GPU and
accelerator architecture, CUDA kernels, attention and MoE kernels, profiling, compilers, and memory
and interconnects. Part V fine-tunes the model for classification and for following instructions.
Part VI is about inference: how a single request is generated, how a server multiplexes thousands of
them, how the KV cache is managed, how to make each token cheaper, how to run models on CPUs and
devices, how retrieval, tools and agents turn a model into a compound system, and what data movement
costs in a long-running reasoning workload. Part VII is post-training: evaluation, inference-time
scaling, self-refinement, the reinforcement-learning setup for LLMs, policy gradients, RLHF and PPO,
DPO, GRPO, RL systems, distillation, and alignment. The appendices cover Triton, accelerators, JAX,
the lab setup, containerising an ML project, the references, doing LLM systems research, and
solutions to the exercises.

The capstone projects are research-grade work framed as research questions. Each has milestones
with automated acceptance tests, baselines, a compute budget sized for two A100 GPUs connected over
PCIe, and a reference solution, and each uses only the material of its own part and the parts
before it, so you can do the capstones of a part as soon as you have read it:

- Part I: knowing your machine ([@ch:proj10]) and a low-precision training study ([@ch:proj11]).
- Part II: bringing up a new open model ([@ch:proj12]) and sequence mixers at long context
  ([@ch:proj13]).
- Part III: a language model from scratch ([@ch:proj1]), training systems on PCIe GPUs
  ([@ch:proj2]), a budgeted scaling study ([@ch:proj3]) and a web-data pipeline ([@ch:proj4]).
- Part IV: mixture-of-experts systems ([@ch:proj8]), a tiny tensor compiler ([@ch:proj14]) and a
  kernel library measured against vendor libraries ([@ch:proj15]).
- Part V: a parameter-efficient fine-tuning study ([@ch:proj16]) and a domain specialist built end
  to end ([@ch:proj17]).
- Part VI: a serving engine ([@ch:proj6]), a CPU runtime ([@ch:proj7]), a profiler ([@ch:proj9])
  and long-context multi-agent serving ([@ch:proj18]).
- Part VII: post-training a reasoning model ([@ch:proj5]), agentic RL with tools ([@ch:proj19]) and
  reasoning distillation under a budget ([@ch:proj20]).

If you already know transformers well, you can skim Part II for the Systems Lens boxes and the labs.
[@sec:landscape:roadmap] gives four reading paths, for systems researchers, inference engineers,
training engineers and post-training researchers, each ending at the capstones that test it.

## Labs {#labs .unnumbered}

Every chapter ends with a lab. Labs live in `code/labs/labNN_*` and share the `llmsys` package in
`code/llmsys`. Each lab has a `starter.py` with `TODO`s, a reference `solution.py`, and a test file.
The tests run on a laptop CPU in under a minute with tiny model configurations:

``` bash
cd code
pip install -r requirements.txt
LAB_IMPL=starter  python -m pytest labs/lab09_attention   # grade your work
LAB_IMPL=solution python -m pytest labs/lab09_attention   # check the reference
```

Several labs have a C++ or CUDA extension: the performance-critical operators live in `code/csrc`
and are built into the `llmsys._C` module with `pip install -e .` ([@app:setup]); every such
operator keeps a pure-PyTorch reference and a backend switch, so the labs run without a compiler and
the C++ task is an explicit extension.

Each lab also has a *Measurements* section. Those experiments are meant to be run on real hardware,
ideally a GPU, with real checkpoints such as GPT-2 and Qwen3-0.6B, and they ask you to explain what
you see using the cost models from the chapter. Numbers in this book are either derived analytically
from published hardware specifications or explicitly labelled as measured; the reference machine for
measured numbers is a workstation with two A100-40GB GPUs connected over PCIe, without NVLink.
Your own measurements are the ones that count.

## Conventions {#conventions .unnumbered}

Tensors are written with their shapes, as in $\mathbf{X}\in\mathbb{R}^{B\times T\times d}$, where
$B$ is the batch size, $T$ the sequence length, and $d$ the model width $d_{\text{model}}$. We write
$L$ for the number of layers, $H$ for query heads, $G$ for key/value heads, $d_{h}$ for the head
dimension, $d_{\text{ff}}$ for the MLP width, and $V$ for the vocabulary size. "FLOPs" counts
floating-point operations, with a multiply--add counted as two. Bandwidths are in bytes per second
and use decimal prefixes (1 TB/s $=10^{12}$ B/s).
