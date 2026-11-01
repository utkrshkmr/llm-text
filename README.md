# LLM Systems from Scratch

**Building, Training, Serving, and Teaching Language Models to Reason: a systems-first textbook with labs.**

📕 **[LLM-Systems-from-Scratch.pdf](LLM-Systems-from-Scratch.pdf)** (487 pages)

This book merges *Build a Large Language Model (From Scratch)* and *Build a Reasoning Model
(From Scratch)* (Raschka) into one sequence, covering every section of both. It adds the systems
layer underneath: GPU performance and the roofline model, FlashAttention, mixed precision and memory
budgets, distributed training (DDP, ZeRO/FSDP, tensor, pipeline, context and expert parallelism),
serving engines (continuous batching, PagedAttention, prefix caching, chunked prefill,
disaggregation), quantisation, speculative decoding, RL training infrastructure, and Triton kernels.
The prose, code and figures are original. Margin tags such as `LLM 3.5` or `RM E.3` show which
source section each passage covers, and the front matter has the complete coverage map.

| Part | Chapters |
|---|---|
| I Foundations | 1 Landscape · 2 PyTorch and the machine underneath |
| II Building an LLM | 3 Text & tokenisation · 4 Attention → FlashAttention · 5 GPT-2 · 6 Qwen3, GQA, RoPE, MoE |
| III Training | 7 Pretraining · 8 Distributed training systems · 9 Classification FT · 10 Instruction FT, LoRA, QLoRA |
| IV Inference & Serving | 11 Inference fundamentals · 12 Batching & serving engines · 13 Quantisation, spec. decoding, deployment |
| V Reasoning | 14 Evaluation · 15 Inference-time scaling · 16 Self-refinement · 17 GRPO · 18 Improving GRPO · 19 RL training systems · 20 Distillation |
| Appendices | A Triton kernels · B Lab setup · C References · D Exercise solutions |

```
book/    LaTeX source (main.tex)
code/    llmsys package + 21 labs (starter / solution / tests / bench scripts)
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
LAB_IMPL=solution python -m pytest labs                   # all reference solutions
```
Tests run on CPU in seconds with tiny models and need no network access. Each lab's `run.py` or
`bench.py` holds the experiments that need real hardware or checkpoints, such as GPT-2,
Qwen3-0.6B or MATH-500; see Appendix B.
