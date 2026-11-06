# Brief: additions to Chapter 9 "Modern Architectures: Qwen3 from Scratch" + Lab 9 extension

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.2 row "7", the existing chapter
book/chapters/ch09_modern_arch.md (keep ALL content), book/solutions/ch09_solutions.md, code/labs/lab09_qwen3/*,
code/llmsys/{layers.py,kvcache.py,qwen3.py}, and from the first wave: book/chapters/ch14_anatomy.md (its MLA
section) and code/llmsys/models/deepseek.py (the complete MLA implementation; REUSE it: import from
llmsys.models.deepseek in the lab/tests rather than writing a second MLA; if a small standalone class is needed
for the chapter listing, derive it from that module's functions).

Files you own: book/chapters/ch09_modern_arch.md (add), book/solutions/ch09_solutions.md (add),
code/labs/lab09_qwen3/* (add tasks/tests, keep existing passing), optional thin module code/llmsys/mla.py
re-exporting/adapting from llmsys.models.deepseek, docs/edition3_briefs/refs/add_ch09.md.

Addition: a new section "Latent attention in depth" placed after the GQA section (before the block
assembly), 150–250 lines: joint KV compression (down-projection to a latent c_KV of rank r, up-projections
W_UK, W_UV), the algebra of absorbing W_UK into the query path and W_UV into the output path at decode time,
decoupled RoPE (a separate small rotary key dimension shared across heads, why RoPE cannot be absorbed),
exactly what is cached (c_KV and k_RoPE per token) and the bytes per token vs GQA and MHA for
DeepSeek-V2/V3 configurations (table), FLOPs of the absorbed path vs naive path at prefill and decode
(when each is cheaper), a working implementation listing (abridged from llmsys.models.deepseek), and a
Systems Lens on why MLA is a bandwidth trade. Cross-reference [@ch:anatomy] for the full model. Add 2–3
exercises (KV bytes for MLA vs GQA at 128k context; absorbed vs naive FLOP crossover; decoupled RoPE dims)
with solutions, and lab tasks: (a) implement naive MLA attention, (b) the absorbed decode path, test they
match each other and the llmsys.models.deepseek reference with a cache; (c) KV-bytes planner extension.

Constraints: keep all existing text/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab09_qwen3` and `python -m pytest -q tests`.
