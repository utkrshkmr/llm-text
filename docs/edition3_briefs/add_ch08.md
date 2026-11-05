# Brief: additions to Chapter 8 "Beyond Softmax Attention" + Lab 8 extension

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.2 row "8", the existing chapter
book/chapters/ch08_attention_alternatives.tex (keep ALL content: linear attention, SSMs/Mamba, hybrids already
there), book/solutions/ch08_solutions.tex, code/labs/lab08_attnalt/*, code/llmsys/{linear_attn.py,ssm.py,layers.py},
and from the first wave book/chapters/ch11_anatomy.tex + code/llmsys/models/{sliding.py,hybrid.py} (sliding window
and hybrids exist there; cross-reference, do not duplicate). GPU: not needed.

Files you own: book/chapters/ch08_attention_alternatives.tex (add), book/solutions/ch08_solutions.tex (add),
code/labs/lab08_attnalt/* (add tasks/tests; keep existing passing), new module code/llmsys/attn_variants.py,
scratchpad refs/add_ch08.tex.

Additions (this is the largest addition set; ~300–400 lines of new chapter text with figures):
1. Low-rank sequence projection (Linformer): project keys/values along the sequence axis; cost, why it
   breaks causality and how causal variants cope.
2. Random-feature kernels (Performer/FAVOR+): softmax kernel approximation, positive random features,
   variance, the causal prefix-sum form; link to the linear-attention recurrence already in the chapter.
3. Dilated windows with global tokens (Longformer/BigBird): mask patterns (figure), FLOPs, kernel support
   (block-sparse), when sparsity pays on GPUs.
4. Trainable block-sparse attention with compression / selection / sliding-window branches (NSA-style):
   compressed block keys, top-n block selection per query, the gated combination, backward
   considerations, cost model vs full attention and vs sliding window; implementation (PyTorch reference
   with explicit block indexing) in llmsys/attn_variants.py.
5. Blockwise "lightning" linear attention (Lightning/MiniMax-style): intra-block quadratic + inter-block
   recurrent form, why it maps well to tensor cores; hybrid stacks interleaving softmax attention layers
   (ratio choices; link \cref{ch:anatomy}).
6. Cross-layer KV sharing (CLA/YOCO-style): sharing K/V across layer groups, cache bytes saved, quality
   notes.
7. A design taxonomy across all axes: a table/figure classifying every mechanism in the chapter by
   (state size vs context, compute per token, parallel training form, kernel availability, recall
   capability), and guidance.
8. Systems Lens / Napkin Math boxes: cache bytes per token for each variant at 128k context on a 7B model;
   prefill FLOPs; what decoders can actually fuse.
9. Lab 8 extension tasks: implement Linformer projection, FAVOR+ features (test: kernel approximation error
   decreases with features; causal form equals quadratic form), dilated/global mask builder (test:
   mask structure), NSA-style block selection attention (test: with all blocks selected equals full
   attention; selected-only matches a masked reference), lightning blockwise linear attention (test:
   equals the recurrent linear attention of the chapter), cross-layer KV sharing wrapper (test: cache
   size halves, outputs unchanged when layers share weights).

Constraints: keep all content/tests; compile check CHNUM=8 (chapter + solutions); run
`LAB_IMPL=solution python -m pytest -q labs/lab08_attnalt` and `python -m pytest -q tests`.
