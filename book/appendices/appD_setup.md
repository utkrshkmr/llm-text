# Lab Environment, Hardware, and Checkpoints {#app:setup}

This appendix collects the practical details that the labs assume: how the code is laid out, how to
install it, which hardware is enough for which lab, and how to get the pretrained weights that the
measurement sections use.

## Repository layout

    code/
      llmsys/             shared package, built up chapter by chapter
        layers.py         norms, MLPs, RoPE, masks, MHA, GQA
        kvcache.py        dynamic and static KV caches
        gpt.py qwen3.py   model definitions + Hugging Face weight loaders
        models/           Llama, Qwen3-MoE, Mixtral, DeepSeek (MLA), Gemma, sliding-window, hybrids
        tokenizer.py      word-level and byte-level BPE tokenizers
        data.py train.py  datasets, loaders, training loop, LR schedule
        generate.py       sampling filters, cached / batched generation, log-probs
        flops.py          napkin-math calculators (params, FLOPs, bytes, rooflines)
        ops/              one module per fused operator: Python reference, Triton, C++ and CUDA
                          backends behind a switch (LLMSYS_BACKEND=auto|python|triton|cpp|cuda)
        ...               later chapters add eval/, rl/, serving/, dist/, kernels/, compile/,
                          retrieval/, jaxlab/, ...
      csrc/               C++ and CUDA sources of the llmsys._C extension (cpu/*.cpp, cuda/*.cu)
      setup.py            builds llmsys._C (pip install -e .)
      runtime/            standalone C++ CPU inference runtime (CMake; Project 7)
      tools/              environment checks: PCIe bandwidth, NCCL all-reduce curves
      labs/labNN_*/       README.md, starter.py, solution.py, test_labNN.py, run/bench scripts
                          (lab NN belongs to chapter NN; labA/labB/labC to appendices A/B/C)
      projects/pK_*/      capstone project kits (Part VIII): specification tooling, acceptance
                          tests, smoke and full configurations, and the reference solution
                          in reference/
      run_all_tests.sh    runs the core tests and each lab and project in its own process
      tests/              tests for the core package and the operator backends
      data/               small, original sample datasets used by the tests

The core modules (`layers`, `kvcache`, `gpt`, `qwen3`, `tokenizer`, `data`, `train`, `generate`) all
use one model interface:

``` python
logits = model(idx, cache=None, pad_mask=None)   # idx: (B, T) -> logits: (B, T, V)
```

Passing a `KVCache` turns the same forward pass into an incremental, cached decode step
([@ch:inference]). Passing `pad_mask` (True for real tokens) masks out the left padding used
for batched generation ([@ch:serving]).

## Installation

``` bash
python -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt         # torch, tokenizers, safetensors, sympy, ...
cd code && python -m pytest tests            # core package self-test (CPU, ~10 s)
LAB_IMPL=solution ./run_all_tests.sh         # every lab and project against its reference solution
```

Run labs one directory at a time, or with `run_all_tests.sh`, rather than as a single `pytest labs`
invocation. The Triton labs use Triton's CPU interpreter, which must be configured before `triton`
is first imported, so they cannot share a process with labs that import Triton normally. For GPU
work, install the PyTorch build that matches your CUDA driver
(<https://pytorch.org/get-started/locally/>). Triton ships with recent Linux CUDA builds of PyTorch.
The labs in [@app:triton] need an NVIDIA or AMD GPU, and their tests skip automatically on
CPU.

### The C++ and CUDA toolchain {#sec:setup:toolchain}

The performance-critical operators of Parts IV and VI
([@ch:cpp; @ch:cuda; @ch:cudakernels; @ch:ondevice; @ch:retrieval]) are written in C++ and
CUDA and compiled into one extension module, `llmsys._C`. Every operator keeps a pure-PyTorch
reference, so nothing in the book *requires* the extension; the tests of the compiled parts skip
when it is absent.

``` bash
# toolchain: a C++20 compiler (g++ >= 11 or clang >= 14), CMake >= 3.22, ninja, and for the
# CUDA kernels a CUDA toolkit whose MAJOR version equals torch.version.cuda (e.g. 13.x for a
# cu130 wheel). pybind11 and nanobind are pure pip installs.
pip install ninja cmake pybind11 nanobind
python -c "import torch; print(torch.version.cuda)"     # pick the toolkit to match
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.0"   # 8.0 = A100; 9.0 = H100
cd code && pip install -e . --no-build-isolation -v     # builds llmsys._C (CPU + CUDA ops)
LLMSYS_NO_CUDA=1 pip install -e . --no-build-isolation  # CPU-only build
python -c "from llmsys import _C; print(_C.has_cuda(), _C.cpu_flags())"
```

While developing a single operator you do not rebuild the whole extension: with `LLMSYS_JIT=1`, each
operator module's `build()` compiles only its own source files through
`torch.utils.cpp_extension.load` and registers them into the same `torch.ops.llmsys` namespace
([@ch:cpp]). `LLMSYS_BACKEND` selects the backend globally (`auto`, `python`, `triton`,
`cpp`, `cuda`).

::: pitfall
[Toolchain mismatches]{.box-title}

Three errors account for most failed builds. A compiler too old for the C++20 headers of recent
PyTorch reports "C++20 or later compatible compiler is required". A CUDA toolkit whose major version
differs from the one PyTorch was built with fails at configure or link time; install the matching
toolkit or wheel. And `-march=native` binaries copied to another machine die with an illegal
instruction; set `LLMSYS_CXXFLAGS` to a portable target before building a wheel. A standalone CMake
build must also pass `-D_GLIBCXX_USE_CXX11_ABI` consistently with PyTorch ([@ch:cpp]);
`cpp_extension` sets it for you.
:::

The standalone CPU runtime of Project 7 (`code/runtime`) is a plain CMake project with no PyTorch
dependency; its README gives the build commands.

## Which hardware for which lab

| **Hardware** | **What runs comfortably** |
|:---|:---|
| Laptop CPU | All tests; tiny-model training and RL demos; tokenizer, napkin-math, engine-simulation labs |
| Apple-silicon laptop (MPS) | GPT-2 small pretraining and fine-tuning; Qwen3-0.6B inference at short contexts |
| Single 16--24 GB GPU | Everything up to GRPO and distillation on Qwen3-0.6B with short responses; LoRA on 1.7B |
| Single 80 GB GPU (A100/H100) | Full-length GRPO runs on 0.6B--1.7B; inference on 8B; realistic serving-engine benchmarks |
| 2$\times$ A100-40GB over PCIe (the reference box of this edition) | All measurements in the book, the two-GPU labs ([@ch:distributed; @ch:sharding; @ch:shardcalc; @ch:rlsys]), and the capstone projects |
| 2--8 GPUs with NVLink | The same, with tensor and expert parallelism paying off ([@ch:shardcalc; @ch:moe]) |

: Rough guide to the hardware each lab needs.

::: napkinmath
[Will it fit?]{.box-title}

Inference needs roughly the weights plus the KV cache: for Qwen3-0.6B in bf16 that is about
$0.6\times10^9\times 2\approx 1.2$ GB of weights plus
$2\cdot L\cdot G\cdot d_{h}\cdot 2\,\text{bytes} = 2\cdot28\cdot8\cdot128\cdot2 \approx 115$ KB of
cache per token, so about 0.47 GB for a 4096-token response. Full fine-tuning with AdamW needs about
16 bytes per parameter before activations ([@ch:pretrain]), or about 9.6 GB for 0.6B
parameters. Use `llmsys.flops` to repeat these estimates for any configuration.
:::

## Getting checkpoints and datasets

The tests never touch the network. The `run.py`/`bench.py` scripts expect checkpoints that you
download once:

``` bash
pip install huggingface_hub
huggingface-cli download openai-community/gpt2 --local-dir ckpt/gpt2
huggingface-cli download Qwen/Qwen3-0.6B-Base --local-dir ckpt/qwen3-0.6b-base
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ckpt/qwen3-0.6b   # reasoning/chat
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ckpt/qwen3-1.7b   # LoRA/QLoRA, post-training
```

``` {#lst:setup:load .python caption="Loading the downloaded checkpoints into our from-scratch modules."}
from safetensors.torch import load_file
from llmsys.gpt import GPTModel, gpt2_config, load_gpt2_hf_state_dict
from llmsys.qwen3 import load_qwen3_from_dir, Qwen3Tokenizer

gpt2 = GPTModel(gpt2_config("gpt2-small (124M)"))
load_gpt2_hf_state_dict(gpt2, load_file("ckpt/gpt2/model.safetensors"))

qwen = load_qwen3_from_dir("ckpt/qwen3-0.6b-base", size="0.6B", device="cuda")
tok = Qwen3Tokenizer("ckpt/qwen3-0.6b-base/tokenizer.json")
```

Datasets used by the measurement scripts include MATH-500 and the MATH training split
([@ch:eval; @ch:grpo; @ch:distill]), the SMS Spam Collection ([@ch:classify]), an
Alpaca-style instruction set ([@ch:instruct]), and any public-domain short story for
pretraining ([@ch:pretrain]). Each script's `--help` names the source it expects and where to
put the file. Small sample files written for this book live in `code/data/` so the tests can run
offline.

## Reproducibility and measurement hygiene

- Call `llmsys.utils.set_seed`. On GPU, also set `torch.backends.cudnn.deterministic=True` when you
  compare numerics. Remember that batched and unbatched runs can differ in the last bits because
  reduction order changes.

- Time GPU code only after `torch.cuda.synchronize()` (the `llmsys.utils.timer` context manager does
  this). Discard warm-up iterations, which include compilation, autotuning and allocator growth.

- Report the configuration with every number: hardware, dtype, batch size, sequence length, PyTorch
  version, and whether `torch.compile` or CUDA graphs were used.

- Prefer distributions to single numbers. Report medians and percentiles (p50/p90/p99) for
  latencies, and confidence intervals for accuracies ([@ch:eval]).
