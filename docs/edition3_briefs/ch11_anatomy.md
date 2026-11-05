# Brief: Chapter 11 "Anatomy of Modern Open Models" (NEW) + Lab 11

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.1 item 11, book/chapters/ch06_modern_arch.tex,
ch07_arch_design.tex, ch08_attention_alternatives.tex (what already exists: Qwen3 dense, GQA, RoPE, MoE intro,
design-space ablations, SSMs/linear attention in llmsys/ssm.py and llmsys/linear_attn.py), code/llmsys/qwen3.py
(the model interface `logits = model(idx, cache=None, pad_mask=None)`, HF weight loader pattern), llmsys/kvcache.py,
llmsys/flops.py, llmsys/moe.py. GPU for your tests: CUDA_VISIBLE_DEVICES=1 (only if needed; tests are CPU).

Files you own (create):
- book/chapters/ch11_anatomy.tex           \chapter{Anatomy of Modern Open Models}\label{ch:anatomy}
- book/solutions/ch11_solutions.tex
- code/llmsys/models/__init__.py           registry: name -> (config dataclass, model class, hf loader)
- code/llmsys/models/llama.py              Llama-3-style dense (RMSNorm, SwiGLU, GQA, RoPE with Llama-3 frequency
                                           scaling, untied head), HF state-dict loader
- code/llmsys/models/qwen3_moe.py          Qwen3-MoE (dense attention of llmsys.qwen3 + routed experts, no shared
                                           expert), HF loader
- code/llmsys/models/mixtral.py            Mixtral-style sparse MoE (top-2 of 8, softmax-renormalised), optional
                                           sliding window, HF loader
- code/llmsys/models/deepseek.py           DeepSeek-V2/V3-style: multi-head latent attention with joint KV
                                           compression, absorbed up-projections at decode (cache only the latent
                                           c_KV and the decoupled RoPE key), decoupled RoPE dims, fine-grained MoE
                                           with shared experts (reuse llmsys.moe), multi-token-prediction module
                                           (one extra transformer block predicting token t+2 from the shifted
                                           embedding + hidden state, with its loss); two equivalent attention
                                           paths (naive and absorbed) with a test that they match
- code/llmsys/models/gemma.py              Gemma-2/3-style: interleaved local (sliding) / global attention layers,
                                           logit soft-capping (final and attention), GeGLU, embedding scaling by
                                           sqrt(d), post-norms, HF loader
- code/llmsys/models/sliding.py            sliding-window attention with a rolling KV cache (Mistral-style), and
                                           attention-sink variant; cache size independent of context
- code/llmsys/models/hybrid.py             hybrid attention / state-space model (e.g. Jamba- or Qwen3-Next-like:
                                           some layers are Mamba-2 / gated linear attention from llmsys.ssm and
                                           llmsys.linear_attn, others GQA); states + KV cache together
- code/llmsys/models/accounting.py         for every architecture: parameter count (by component), FLOPs/token
                                           (prefill and decode), KV-cache (or state) bytes per token, inference
                                           cost profile (bytes/step at batch B, arithmetic intensity), from the
                                           config alone; one function `profile(name, cfg, hw)` and a table printer
- code/labs/lab11_anatomy/{README.md,starter.py,solution.py,test_lab11.py,run.py}
- scratchpad refs file: refs/ch11_anatomy.tex

Chapter: side-by-side anatomy of the seven families with one figure per architecture (block diagram) and a
master comparison table (norms, activation, attention type, positions, KV/token, params, active params,
context). For each: the equations where they differ from Qwen3 (\cref{ch:modern}), the implementation
listing (abridged from your module), the parameter/FLOP/KV accounting with worked numbers for a published
config (Llama-3.1-8B, Qwen3-30B-A3B, Mixtral-8x7B, DeepSeek-V3 (671B/37B active), Gemma-3-12B, Mistral-7B
with 4k window, a hybrid such as Jamba-1.5-Mini or Qwen3-Next), and its inference cost profile on one A100
(bytes per decode step at batch 1 and 32; which are bandwidth-bound and where MLA's cache compression pays).
Dedicated section on MLA with the absorbed-projection algebra and what exactly is cached (the ch06 addition
later cross-references yours; make your treatment the complete one). Section on checkpoint loading: HF
safetensors naming, weight permutations (RoPE layout), tied vs untied embeddings, how to verify a loader
(logit agreement test). Systems Lens boxes: why MLA trades FLOPs for bytes; local/global interleaving and the
KV budget; MoE serving memory. Summary, exercises, Lab 11.

Lab 11 (CPU tests): tiny configs for every family; tests check (a) each model runs and its cached decode equals
full forward (use the pattern of code/tests/test_core.py::_check_cache_equivalence), (b) accounting functions
equal the counts of the instantiated tiny models (params exactly; KV bytes; FLOPs vs a direct count), (c) MLA
naive == absorbed path, (d) sliding window with rolling cache equals full attention with the window mask, (e)
HF loaders: build a *synthetic* HF-format state dict (random tensors with the official key names and shapes for
the tiny config) and check load -> our logits == an independent plain reference forward written inside the test
for two families (Llama and DeepSeek-MLA). run.py: download two real checkpoints (e.g. meta-llama/Llama-3.2-1B
or Qwen/Qwen3-1.7B, and deepseek-ai/DeepSeek-V2-Lite or google/gemma-3-1b-pt), load, and report max |logit diff|
against `transformers` if installed (optional dependency; skip gracefully) plus tokens/s; CPU/GPU.

Constraints: no network in tests; keep every existing llmsys module unchanged; configs as dataclasses with
`from_hf_config(dict)`; the model interface must be `model(idx, cache=None, pad_mask=None) -> logits`.
Compile checks (CHNUM=11) for chapter and solutions; run lab11 tests and `python -m pytest -q tests`.
