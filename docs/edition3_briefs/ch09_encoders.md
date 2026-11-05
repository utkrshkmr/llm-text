# Brief: Chapter 9 "Encoders, Encoder–Decoders and Cross-Attention" (NEW) + Lab 9

Read first: BRIEF_conventions.md (same directory as this file, one level up), docs/PLAN.md §1.1 item 9,
book/chapters/ch04_attention.tex and ch06_modern_arch.tex (style), code/llmsys/layers.py, code/llmsys/gpt.py,
code/labs/lab06_qwen3/ (lab layout), code/labs/lab04_attention/test_lab04.py.

Files you own (create):
- book/chapters/ch09_encoders.tex          \chapter{Encoders, Encoder--Decoders and Cross-Attention}\label{ch:encoders}
- book/solutions/ch09_solutions.tex
- code/llmsys/encoder.py                  bidirectional encoder blocks, MLM / span-corruption / prefix-LM
                                           objectives (masking + target construction), pooling, sentence
                                           embedding head, InfoNCE / contrastive loss, cross-encoder reranker head
- code/llmsys/encdec.py                   encoder–decoder: cross-attention module with a cross-attention KV
                                           cache computed once per encoder output (distinct from the growing
                                           self-attention KVCache in llmsys.kvcache), a small T5-style model
- code/llmsys/vit.py                      patchify + ViT encoder (reuse llmsys.layers), image→tokens
- code/labs/lab09_encoders/{README.md,starter.py,solution.py,test_lab09.py,run.py}
- scratchpad refs file: refs/ch09_encoders.tex

Chapter content (all of PLAN §1.1 item 9, in depth):
1. Decoder-only vs encoder: masking (causal vs none vs prefix), what the loss trains, why
   bidirectionality helps understanding tasks and hurts generation. Padding masks and attention over
   [CLS]/[SEP] tokens. Figures: the three mask patterns side by side; loss positions.
2. Pretraining objectives with exact target construction: masked-token prediction (BERT 80/10/10 rule),
   span corruption with sentinel tokens (T5), prefix-LM (UL2 mixture of denoisers in brief). Napkin math:
   fraction of tokens that produce gradient signal per objective, tokens/s needed to match a causal LM.
3. Encoder–decoder architecture: cross-attention derivation; what is cached (K,V of the encoder output per
   layer, computed once; decoder self-attn cache grows); relative position bias in T5 (brief; details in
   \cref{ch:archdesign}). Systems Lens: encoder–decoder serving cost = one prefill of the source + decode with
   two attention reads per step; compare bytes/step with a decoder-only model of equal parameters.
4. Encoders as embedding and reranking models: bi-encoder vs cross-encoder, pooling (CLS, mean, last-token
   for decoder-based embedders), contrastive training with in-batch negatives (InfoNCE), temperature, hard
   negatives, Matryoshka dimensions (brief), late interaction (ColBERT, brief). Decoder-only LLMs as embedders.
   Forward reference to \cref{ch:retrieval} where the embeddings feed an ANN index.
5. Vision transformers as encoders: patch embedding, [CLS] vs pooled, position embeddings for images,
   resolution/patch-size cost (tokens = (H/P)(W/P)), as the vision tower of VLMs (\cref{ch:alignment}).
6. Systems: encoder inference is compute-bound (whole sequence at once, no KV cache), dynamic batching with
   length bucketing/sorting, padding waste vs packed ("unpadded") attention, typical deployment (ONNX
   Runtime / TensorRT; see \cref{ch:compilers}), throughput napkin math for BERT-base and a 7B decoder
   embedder on one A100 (FLOPs/sequence, bytes/sequence, batch needed to reach the compute roof).
7. Summary, 8–12 exercises, Lab 9.

Lab 9 (tests on CPU, tiny sizes): tasks implement (a) MLM masking + targets, (b) span-corruption targets with
sentinels, (c) prefix-LM mask, (d) a bidirectional encoder block/model (reuse layers), (e) CrossAttention with the
one-shot cross KV cache, test: incremental decoding with cached cross K/V equals full recompute, (f) pooling +
InfoNCE, (g) patchify/ViT forward shape and equivalence to an unfold-based reference. run.py: train a tiny
encoder with MLM on code/data/ text (create a small original sample file if needed, few KB) then contrastive
fine-tune on synthetic paraphrase pairs; report nearest-neighbour accuracy; optional --gpu timing of encoder
throughput vs batch size.

Constraints: CPU-only tests, offline, deterministic; keep llmsys.layers/gpt/qwen3 untouched (import from them).
Chapter must compile: CHNUM=9 ./book/check_chapter.sh chapters/ch09_encoders.tex (exit 0, no "!" lines),
also the solutions file. Run `LAB_IMPL=solution python -m pytest -q labs/lab09_encoders` and `python -m pytest -q tests`.
