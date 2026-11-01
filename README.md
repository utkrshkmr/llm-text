# LLM Systems from Scratch

**Building, Training, Serving, and Teaching Language Models to Reason: a systems-first textbook with labs.**

🌐 **Read online: <https://utkrshkmr.github.io/llm-text/>** (light and dark mode, search)  
📕 **[LLM-Systems-from-Scratch.pdf](LLM-Systems-from-Scratch.pdf)**

This book follows a language model from raw text to a deployed reasoning system, and asks at
every step what the machine underneath is doing. Each chapter pairs the modelling ideas with their
systems costs (Systems Lens and Napkin Math boxes), runnable code, and practical guidance on setting
things up: configs, data formats, launch scripts, logging and debugging. Each chapter ends with a lab.

| Part | Chapters |
|---|---|
| I Foundations | 1 Landscape · 2 PyTorch and the machine underneath |
| II Building an LLM | 3 Text & tokenisation · 4 Attention → FlashAttention · 5 GPT-2 · 6 Qwen3, GQA, RoPE, MoE · 7 Architecture design space · 8 Linear attention, SSMs, hybrids |
| III Pretraining at Scale | 9 Optimizers & the training loop · 10 Pretraining · 11 Pretraining data · 12 Scaling laws · 13 Single-GPU performance engineering · 14 Distributed training · 15 FSDP from scratch |
| IV Fine-Tuning | 16 Classification · 17 Instruction tuning, LoRA, QLoRA |
| V Inference & Serving | 18 Inference fundamentals · 19 Batching & serving engines · 20 Quantisation, speculative decoding, deployment |
| VI Reasoning & Post-Training | 21 Evaluation · 22 Inference-time scaling · 23 Self-refinement · 24 Policy gradients · 25 RLHF: reward models & PPO · 26 DPO and its family · 27 GRPO · 28 Improving GRPO · 29 RL training systems · 30 Distillation · 31 Alignment, safety, multimodality |
| VII Capstone Projects | 32 A Transformer LM from scratch · 33 Make it fast, make it fit, make it scale · 34 A budgeted scaling study · 35 Filtering web data · 36 Reasoning RL and alignment |
| Appendices | A Triton kernels · B Accelerators beyond the GPU · C Lab setup · D References · E Solutions |

```
book/    LaTeX source (main.tex)
code/    llmsys package, labs (starter / solution / tests / bench scripts), project kits
web/     generator for the online edition (GitHub Pages)
```

## Build the book
```
cd book && latexmk -pdf main.tex
```
Requires TeX Live with tcolorbox, listings, tikz, pgfplots and cleveref. No shell-escape is needed.
To compile a single chapter: `CHNUM=4 ./check_chapter.sh chapters/ch04_attention.tex`.

## Run the labs
```
cd code
pip install -r requirements.txt
python -m pytest tests                                    # core package
LAB_IMPL=starter  python -m pytest labs/lab04_attention   # your work (TODOs fail until done)
LAB_IMPL=solution ./run_all_tests.sh                      # every lab, each in its own process
```
Tests run on a CPU with tiny models and need no network access. Each lab's `run.py` or `bench.py`
holds the experiments that need real hardware or checkpoints (GPT-2, Qwen3-0.6B, MATH-500, ...);
Appendix C explains how to set those up. The capstone project kits in `code/projects/` provide setup
tooling and acceptance tests, but the projects themselves are yours to build.

## Build the website
```
python3 web/build_site.py --book book --aux book --pdf book/main.pdf --out site
```
A GitHub Actions workflow rebuilds the PDF and the website on every push to `main` and publishes
the site to the `gh-pages` branch (see `web/README.md`).
