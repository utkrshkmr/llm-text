# Brief: additions to Chapter 7 "The Architecture Design Space" + Lab 7 extension

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.2 row "7", the existing chapter
book/chapters/ch07_arch_design.md (keep ALL content; note how its ablation framework is organised),
book/solutions/ch07_solutions.md, code/labs/lab07_ablations/* (incl. reference_results.json and run.py),
code/llmsys/ablate.py, code/llmsys/layers.py. Also skim book/chapters/ch09_encoders.md (first wave) for the
T5 relative-bias mention so you don't contradict it.

Files you own: book/chapters/ch07_arch_design.md (add), book/solutions/ch07_solutions.md (add),
code/labs/lab07_ablations/* (add tasks/tests/run options; keep existing tests passing), new module
code/llmsys/mixers.py, docs/edition3_briefs/refs/add_ch07.md.

Additions (sections inserted at the matching places of the design-space tour, plus exercises and lab tasks):
1. Relative position biases: T5 bucketed biases, ALiBi slopes, learned relative biases; how they compose
   with or replace RoPE; KV-cache and kernel consequences (bias tensors in FlashAttention). Code in
   llmsys/mixers.py (`RelativeBias`, `alibi_bias`).
2. Alternative normalisations: LayerNorm vs RMSNorm (link ch6), post-norm vs pre-norm vs sandwich/peri-norm
   (existing in lab7? extend), QK-norm (exists), DeepNorm scaling, nGPT-style normalised representations
   (brief), what each does to activation growth with depth (small measured plot on a tiny model: label
   illustrative).
3. Head-mixing (talking-heads attention: mixing logits across heads before and after softmax) and
   synthetic-attention variants (Synthesizer dense/random: attention weights not computed from Q·K) as
   token-mixer ablations: equations, parameter/FLOP costs, implementation in llmsys/mixers.py
   (`TalkingHeadsAttention`, `SynthesizerDense`, `SynthesizerRandom`) with the same interface as
   llmsys.layers.MultiHeadAttention so they plug into the chapter's ablation harness.
4. Systems Lens: why most of these did not survive (kernel availability, FlashAttention compatibility,
   KV-cache cost).
5. Lab 7 extension: add the new mixers and position schemes as ablation options (tests: shapes, causality
   preserved, parameter counts match formulas, relative bias is translation invariant, talking-heads with
   identity mixing equals MHA), extend run.py grid.

Constraints: keep all existing content and tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab07_ablations` and `python -m pytest -q tests`.
