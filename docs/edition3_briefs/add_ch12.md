# Brief: additions to Chapter 12 "The Architecture Design Space" + Lab 12 extension

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.2 row "8", the existing chapter
book/chapters/ch12_arch_design.md (keep ALL content; note how its ablation framework is organised),
book/solutions/ch12_solutions.md, code/labs/lab12_ablations/* (incl. reference_results.json and run.py),
code/llmsys/ablate.py, code/llmsys/layers.py. Also skim book/chapters/ch14_encoders.md (first wave) for the
T5 relative-bias mention so you don't contradict it.

Files you own: book/chapters/ch12_arch_design.md (add), book/solutions/ch12_solutions.md (add),
code/labs/lab12_ablations/* (add tasks/tests/run options; keep existing tests passing), new module
code/llmsys/mixers.py, docs/edition3_briefs/refs/add_ch12.md.

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
5. Lab 12 extension: add the new mixers and position schemes as ablation options (tests: shapes, causality
   preserved, parameter counts match formulas, relative bias is translation invariant, talking-heads with
   identity mixing equals MHA), extend run.py grid.

Constraints: keep all existing content and tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab12_ablations` and `python -m pytest -q tests`.


## DeepSeek contributions (owner request, 2026-11-05). Cover each from the primary paper or repository, at §7 depth
- Hyper-connections and DeepSeek's manifold-constrained hyper-connections (mHC): widening the residual stream into
  several parallel streams with learned mixing, why unconstrained mixing breaks the identity mapping and destabilises
  training at scale, and mHC's projection of the mixing matrix onto doubly stochastic matrices (Sinkhorn); implement
  both as block variants in the ablation workbench and compare with the standard residual.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Adaptive computation: mixture-of-depths (token-level routing around blocks, capacity), early exit and LayerSkip
  (self-speculative decoding from early layers), looped / recurrent-depth transformers (universal transformers,
  recent recurrent-depth reasoning models) and latent reasoning in continuous space (Coconut); implement
  mixture-of-depths and a looped block in the ablation workbench; FLOP/byte accounting for each.
- Minor block variants: LayerScale, differential attention, squared-ReLU FFNs, QK-norm and soft-capping recap.
