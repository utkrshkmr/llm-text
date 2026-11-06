# Lab 12 — A sentence encoder, from masked prediction to contrastive embeddings

Chapter 12 (`ch:encoders`). Implement in `starter.py`:

1. `mlm_corrupt`                                            (Task 1)
2. `span_corrupt_from_mask`                                 (Task 2)
3. `prefix_lm_mask`, `prefix_lm_targets`                    (Task 3)
4. `BidirectionalAttention.forward`, `EncoderBlock.forward`, `EncoderModel.forward` (Task 4)
5. `CrossAttention.forward`, `incremental_decode`           (Task 5)
6. `pool`, `info_nce_loss`                                  (Task 6)
7. `patchify`, `ViTEncoder.forward`                         (Task 7)
8. `encoder_flops_per_sequence`, `batch_for_compute_bound`, `encdec_step_bytes` (Task 8)

The references are `llmsys/encoder.py`, `llmsys/encdec.py` and `llmsys/vit.py`.

```bash
cd code
python -m pytest -q labs/lab12_encoders                     # grades starter.py
LAB_IMPL=solution python -m pytest -q labs/lab12_encoders   # reference

# Train with MLM on data/encoder_sample.txt, fine-tune with InfoNCE on synthetic paraphrases,
# report nearest-neighbour accuracy at each stage. About a minute on a CPU, seconds on a GPU.
python labs/lab12_encoders/run.py --impl solution

# Throughput against batch size, and random against length-sorted batching (needs a GPU).
python labs/lab12_encoders/run.py --impl solution --gpu --json-out lab12_gpu.json
```

`run.py --gpu` reports the median, p10 and p90 over 20 repetitions after warm-up, for a
BERT-base-shaped encoder and for a 7B decoder-shaped model (14 GB in bf16; skip it with
`--no-7b` on a small GPU). It uses random weights, because throughput does not depend on them.
