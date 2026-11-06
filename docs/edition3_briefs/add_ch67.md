# Brief: additions to Chapter 67 "Alignment, Safety, and Multimodality"

Read first: BRIEF_conventions.md (all, esp. §7), the existing chapter book/chapters/ch67_alignment_multimodal.md
(keep all content), code/llmsys/vlm.py, code/labs/lab67_alignment/*.

## DeepSeek contributions (owner request, 2026-11-05)
- Janus / Janus-Pro: one autoregressive transformer for both image understanding and generation, with decoupled
  visual encoders (a semantic encoder for understanding, a VQ tokenizer for generation); why decoupling helps.
- DeepSeek-VL2: MoE vision-language model with dynamic tiling of high-resolution images; tokens per image and
  serving cost.
- DeepSeek-OCR: "contexts optical compression" — rendering long text as images so a vision encoder compresses it
  into far fewer tokens; the measured compression/accuracy trade-off and what it suggests for long-context memory.
  Compute the token and KV-cache savings for a long document with our accounting tools.
- Research frontier section and Reading-the-research box (§7) if missing.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Safety in practice: jailbreak families and evaluation (red-teaming datasets, automated attacks such as GCG), prompt
  injection against agents and defences (privilege separation, spotlighting), guardrail classifiers (Llama Guard
  style) as a serving component and their latency cost, watermarking of generated text (green-list schemes, SynthID)
  with detection statistics, machine unlearning and its evaluation.
- Interpretability for practitioners (new section): probing, logit lens / tuned lens, induction heads, activation
  patching and attribution, sparse autoencoders (training one on a small model's residual stream), steering vectors;
  the systems side (activation capture at scale, storage).
- Speech and audio: Whisper-style encoder-decoder ASR, audio tokenisers (neural codecs), speech-in/speech-out models and
  their serving latency budget.
