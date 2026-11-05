# Lab 41: Generalist post-training, evaluation at scale, and a tiny VLM

Chapter 41 (`ch:alignment`). You build the plumbing that sits between the methods of
Chapters 25 and 35 to 27 and a model you can ship:

| Task | Function | What it teaches |
|---|---|---|
| 1 | `mixture_weights` | temperature-style sampling over instruction sources |
| 2 | `pack_stream` | streaming concat-and-chunk SFT packing with prompt masking |
| 3 | `debiased_preference` | removing an LLM judge's position bias |
| 4 | `safety_report` | two-sided safety: unsafe compliance *and* over-refusal |
| 5 | `splice_image_embeddings` | how image features enter a decoder LM |
| 6 | `clip_loss`, `siglip_loss` | contrastive image-text pretraining objectives |
| 7 | `anyres_tokens` | vision-token accounting for high-resolution images |
| 8 | `multimodal_block_hashes` | prefix caching that is correct for images |
| 9 | `ImageEmbeddingCache.get_or_compute` | skipping the vision tower on repeated images |

The synthetic VLM task (coloured shapes -> two-word captions) and its training loop are provided.

```
cd code
LAB_IMPL=starter  python -m pytest labs/lab41_alignment     # your work
LAB_IMPL=solution python -m pytest labs/lab41_alignment     # reference
python labs/lab41_alignment/run.py vlm --steps 400          # stage-1 vs stage-2 VLM training
python labs/lab41_alignment/run.py eval --ckpt ckpt/qwen3-0.6b --suites data/suites.json
```
`run.py eval` needs a downloaded Qwen3 checkpoint and your own JSONL eval files (see `--help`);
the optional judge uses a local Ollama server.
