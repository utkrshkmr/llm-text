# Brief: additions to Chapter 25 "Instruction Tuning, LoRA, QLoRA" + Lab 25 extension

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.2 row "25", the existing chapter
book/chapters/ch25_instruction_lora.tex (keep ALL content), book/solutions/ch25_solutions.tex,
code/labs/lab25_instruct_lora/*, code/llmsys/finetune.py, code/llmsys/lora.py, code/llmsys/qwen3.py
(Qwen3Tokenizer chat template). GPU: not needed.

Files you own: book/chapters/ch25_instruction_lora.tex (add), book/solutions/ch25_solutions.tex (add),
code/labs/lab25_instruct_lora/* (add tasks/tests), new module code/llmsys/multiturn.py, scratchpad refs/add_ch25.tex.

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

Constraints: keep all content/tests; compile check CHNUM=25 (chapter + solutions); run
`LAB_IMPL=solution python -m pytest -q labs/lab25_instruct_lora` and `python -m pytest -q tests`.
