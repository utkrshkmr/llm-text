# Brief: additions to Chapter 9 "Beyond Softmax Attention" + Lab 9 extension

Read first: BRIEF_conventions.md (one level up), docs/PLAN.md §1.2 row "9", the existing chapter
book/chapters/ch09_attention_alternatives.md (keep ALL content: linear attention, SSMs/Mamba, hybrids already
there), book/solutions/ch09_solutions.md, code/labs/lab09_attnalt/*, code/llmsys/{linear_attn.py,ssm.py,layers.py},
and from the first wave book/chapters/ch12_anatomy.md + code/llmsys/models/{sliding.py,hybrid.py} (sliding window
and hybrids exist there; cross-reference, do not duplicate). GPU: not needed.

Files you own: book/chapters/ch09_attention_alternatives.md (add), book/solutions/ch09_solutions.md (add),
code/labs/lab09_attnalt/* (add tasks/tests; keep existing passing), new module code/llmsys/attn_variants.py,
docs/edition3_briefs/refs/add_ch09.md.

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
   (ratio choices; link [@ch:anatomy]).
6. Cross-layer KV sharing (CLA/YOCO-style): sharing K/V across layer groups, cache bytes saved, quality
   notes.
7. A design taxonomy across all axes: a table/figure classifying every mechanism in the chapter by
   (state size vs context, compute per token, parallel training form, kernel availability, recall
   capability), and guidance.
8. Systems Lens / Napkin Math boxes: cache bytes per token for each variant at 128k context on a 7B model;
   prefill FLOPs; what decoders can actually fuse.
9. Lab 9 extension tasks: implement Linformer projection, FAVOR+ features (test: kernel approximation error
   decreases with features; causal form equals quadratic form), dilated/global mask builder (test:
   mask structure), NSA-style block selection attention (test: with all blocks selected equals full
   attention; selected-only matches a masked reference), lightning blockwise linear attention (test:
   equals the recurrent linear attention of the chapter), cross-layer KV sharing wrapper (test: cache
   size halves, outputs unchanged when layers share weights).

Constraints: keep all content/tests; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab09_attnalt` and `python -m pytest -q tests`.


## Expanded scope (owner, 2026-11; staged plan). Same depth rules: BRIEF_conventions.md §7 (Research frontier section, Reading-the-research box)
- Attention sinks and streaming LLMs in depth (architecture side): the empirical phenomenon, the explanation (softmax
  must sum to 1; no-op heads dump mass on the first tokens), massive activations, fixes at training time (a learnable
  sink token, sink logits / off-by-one softmax as in gpt-oss, gated attention), StreamingLLM's sink + window
  cache with RoPE position handling (re-indexing within the cache), measured perplexity of window-only vs sink+window
  on our model. Block-sparse attention masks (BigBird/Longformer patterns, native sparse attention / NSA, MoBA) with
  a block-mask implementation. Serving-side eviction policies live in [@ch:kvcache]; link both ways.


## DeepSeek contributions (owner request, 2026-11-05). Cover each from the primary paper or repository, at §7 depth
- DeepSeek Native Sparse Attention (NSA, 2025): three gated branches (compressed coarse-grained tokens, selected
  top-n blocks chosen from the compressed scores, sliding window), hardware-aligned block selection shared within a
  GQA group, natively trainable; implement a reference NSA forward and check it against dense attention when the
  selection covers everything.
- DeepSeek Sparse Attention (DSA, DeepSeek-V3.2): the lightning indexer (a few low-precision heads scoring every
  preceding token), top-k token selection per query, how it is trained (dense warm-up that aligns the indexer to the
  dense attention distribution, then sparse training), its operation on top of MLA, and the resulting long-context
  cost curve; implement indexer + top-k attention and measure the cost against dense attention on our model.
  The serving side of both (what the KV cache must still keep) goes to [@ch:kvcache].
