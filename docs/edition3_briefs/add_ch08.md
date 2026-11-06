# Brief: additions to Chapter 8 "Text and Tokenisation" + Lab 8 extension

Read first: BRIEF_conventions.md (one level up, incl. §6), docs/PLAN.md §1.2 row "4", the existing chapter
book/chapters/ch08_text_data.md (keep ALL existing content; you add sections), book/solutions/ch08_solutions.md,
code/labs/lab08_tokenizer/*, code/llmsys/tokenizer.py, code/llmsys/qwen3.py (Qwen3Tokenizer, embedding
layout), and the finished Chapter 31 material from the first wave: book/chapters/ch31_cpp_pytorch.md (its BPE
section), code/csrc/cpu/bpe.cpp, code/llmsys/ops/bpe.py, code/labs/lab31_cpp/ (the C++ BPE encoder/trainer ops
and their Python wrappers; reuse them, do not write new C++ for BPE).

Files you own: book/chapters/ch08_text_data.md (add), book/solutions/ch08_solutions.md (add), code/labs/lab08_tokenizer/*
(add tasks/tests; keep existing tests passing), new module code/llmsys/vocab_ext.py, docs/edition3_briefs/refs/add_ch08.md.

Additions (new sections before "Summary", new exercises, new lab tasks):
1. Adding tokens to a trained tokenizer: special tokens vs regular tokens, how merges and the regex interact,
   added-token matching order; the embedding and LM-head consequences.
2. Resizing and initialising embeddings for new tokens: mean of existing embeddings, initialisation from
   constituent sub-token embeddings (average of the old tokenisation), small-variance random; tied vs untied
   heads; what happens to logits of the new tokens at step 0 (napkin math on the softmax mass); code in
   llmsys/vocab_ext.py (`add_tokens`, `resize_embeddings(model, new_vocab, init="subtoken_mean")`) for
   llmsys.gpt.GPTModel and llmsys.qwen3.Qwen3Model.
3. Extending a tokenizer to a new language/domain: fertility (tokens per word) measurements, training
   additional merges on the new corpus and appending them (vocabulary extension vs retraining), the cost per
   token of a bad vocabulary (bytes per token, context budget), continued pretraining notes. Figure: fertility
   per language for a byte-level BPE with and without extension (compute on small synthetic/sample texts and
   label as illustrative).
4. Systems Lens: tokenizer throughput as a data-pipeline stage (MB/s per core; Python vs C++ from Lab 31),
   pre-tokenisation regex cost, batched/streaming encoding.
5. Lab 8 extension (explicit "C++ extension" task group at the end of the lab): use the C++ BPE trainer and
   encoder from Chapter 31 through `llmsys.ops.bpe` with the backend switch; tasks: (a) vocabulary extension
   with the C++ trainer and fertility before/after on a sample text; (b) embedding resize with sub-token mean
   init and a test that the new-token logits start near the mean; (c) throughput comparison Python vs C++.
   Tests must skip the C++ parts cleanly when the op is not built (use `llmsys.ops.bpe.build()` as the gate).

Constraints: keep every existing section, exercise, figure and test; append exercises (continue numbering
implicitly) and their solutions; Site check (see BRIEF_conventions.md §2); run
`LAB_IMPL=solution python -m pytest -q labs/lab08_tokenizer` (with and without LLMSYS_JIT=1) and `python -m pytest -q tests`.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Tokenizer families side by side: BPE vs SentencePiece (Unigram LM with its EM training and Viterbi segmentation,
  and SentencePiece's BPE mode), WordPiece (likelihood-based merges, ## continuation), tiktoken-style regex
  pre-tokenisation; implement Unigram training and segmentation in the lab and compare compression and fertility.
- Tokenizer-free and byte-level models: ByT5, MegaByte, Byte Latent Transformer (entropy-based patching), what they
  cost in sequence length and FLOPs, and why tokenisation is still a systems trade-off.
