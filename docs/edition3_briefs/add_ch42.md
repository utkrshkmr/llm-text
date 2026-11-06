# Brief: additions to Chapter 42 "Instruction Tuning, LoRA, QLoRA" + Lab 42 extension

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.2 row "29", the existing chapter
book/chapters/ch42_instruction_lora.md (keep ALL content), book/solutions/ch42_solutions.md,
code/labs/lab42_instruct_lora/*, code/llmsys/finetune.py, code/llmsys/lora.py, code/llmsys/qwen3.py
(Qwen3Tokenizer chat template). GPU: not needed.

Files you own: book/chapters/ch42_instruction_lora.md (add), book/solutions/ch42_solutions.md (add),
code/labs/lab42_instruct_lora/* (add tasks/tests), new module code/llmsys/multiturn.py, docs/edition3_briefs/refs/add_ch42.md.

Addition: a section "Multi-turn instruction persistence through training-data construction" (~150–200 lines):
the failure mode (system/first-turn instructions forgotten after several turns), why it happens (loss on
later turns rarely depends on the early instruction; distribution of turn counts in SFT data), data
construction remedies: injecting persistent instructions into multi-turn conversations with later-turn
compliance checks, instruction-following verifiers that generate targets, loss masking choices (assistant-only
masking, weighting late turns, masking turns that violate the instruction), synthetic conversation
generation recipes, "system prompt distillation" (train to follow without the prompt), evaluation (a
compliance-by-turn curve). Code in llmsys/multiturn.py: conversation dataclass, chat-template packing with
per-turn loss masks (reuse the Qwen3 chat format), injection transforms, a rule-based compliance checker for
simple constraints (language, format, forbidden words) used both to build data and to evaluate. Napkin math:
tokens per conversation vs context budget, fraction of loss tokens. Systems Lens: packing multi-turn
sequences and masks efficiently. 2–3 exercises + solutions; lab tasks: build masked multi-turn examples (test:
mask positions exactly), injection transform (test: instruction present in every conversation, later turns
checked), compliance-by-turn evaluation on a toy generator (test: deterministic numbers).

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab42_instruct_lora` and `python -m pytest -q tests`.


## Expanded scope (owner, 2026-11; staged plan). Same depth rules: BRIEF_conventions.md §7 (Research frontier section, Reading-the-research box)
- LoRA in depth: the low-rank update and why it works (intrinsic dimension), initialisation (A random, B zero) and
  the α/r scaling (rsLoRA), which matrices to adapt, rank choice experiments, merging and unmerging, memory and FLOP
  accounting; variants: DoRA, LoRA+, PiSSA, VeRA, AdaLoRA; QLoRA with NF4 + double quantisation + paged optimizers
  (the NF4 construction from [@ch:quant]); multi-LoRA serving (S-LoRA, Punica's segmented gather matmul / BGMV,
  adapter paging in the KV/weight memory pool) with a batched multi-adapter forward implemented and tested.
