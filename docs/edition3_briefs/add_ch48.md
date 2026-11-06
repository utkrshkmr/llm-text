# Brief: additions to Chapter 48 "Alignment, Safety, and Multimodality"

Read first: BRIEF_conventions.md (all, esp. §7), the existing chapter book/chapters/ch48_alignment_multimodal.md
(keep all content), code/llmsys/vlm.py, code/labs/lab48_alignment/*.

## DeepSeek contributions (owner request, 2026-11-05)
- Janus / Janus-Pro: one autoregressive transformer for both image understanding and generation, with decoupled
  visual encoders (a semantic encoder for understanding, a VQ tokenizer for generation); why decoupling helps.
- DeepSeek-VL2: MoE vision-language model with dynamic tiling of high-resolution images; tokens per image and
  serving cost.
- DeepSeek-OCR: "contexts optical compression" — rendering long text as images so a vision encoder compresses it
  into far fewer tokens; the measured compression/accuracy trade-off and what it suggests for long-context memory.
  Compute the token and KV-cache savings for a long document with our accounting tools.
- Research frontier section and Reading-the-research box (§7) if missing.
