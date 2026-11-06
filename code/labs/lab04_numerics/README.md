# Lab 4 — Numbers on the machine: formats, rounding, accumulation and stable algorithms

Chapter 4 (`ch:numerics`). Implement in `starter.py`:

1. `decode_float`, `encode_float`: bit patterns of any `FloatFormat` with round-to-nearest-even,
   saturation and the IEEE / OCP "fn" / no-special encodings                       (Task 1)
2. `stochastic_round`                                                                  (Task 2)
3. `pairwise_sum`, `kahan_sum` in a simulated precision                                (Task 3)
4. `online_logsumexp`, `online_softmax`, `welford`                                     (Task 4)
5. `accumulate_gemm`: a GEMM with a limited-precision accumulator, chunked MMAs, a rounding rule
   and periodic promotion to FP32                                                      (Task 5)

The reference library is `llmsys/numerics.py`; format specs (`FP16`, `BF16`, `E4M3`, `E5M2`,
`E2M1`, ...) come from there. The tests compare Task 1 bit for bit with PyTorch's own casts
(`torch.float16`, `torch.bfloat16`, `torch.float8_e4m3fn`, `torch.float8_e5m2`) over every code
and every tie, and the other tasks with the library or with exact (`fractions.Fraction`, float64)
results. The last block of tests checks the formats, rounding and reduction parts of the library
API itself. Block scaling (MX, NVFP4), rotations, quantisers and the block-scaled FP8 GEMM are the
next lab, `labs/lab05_quant` (Chapter 5).

```bash
cd code
python -m pytest -q labs/lab04_numerics                     # grades starter.py
LAB_IMPL=solution python -m pytest -q labs/lab04_numerics   # reference (a few seconds, CPU)

# Measurements (GPU optional). Writes results/bench_results.{json,md}
CUDA_VISIBLE_DEVICES=0 python labs/lab04_numerics/bench.py
python labs/lab04_numerics/bench.py --only updates accum --device cpu --n-weights 65536
python labs/lab04_numerics/bench.py --impl starter         # time your own Task 5 code

# Figures of the chapter (reads results/bench_results.json for the measured ones)
python labs/lab04_numerics/figures.py
```

`bench.py` runs six experiments: GEMM throughput by dtype (median, p10, p90 over 30 repetitions),
10,000 BF16 weight updates with and without master weights and with stochastic rounding
(`updates`), dot-product error against K for simulated accumulators and for cuBLAS (`accum`), the
per-tensor FP8 GEMM with a 14-bit accumulator against K and against the promotion interval
(`promotion`), FP8 E4M3 vs E5M2 flush-to-zero on heavy-tailed data (`formats`), and probes of the
tensor-core accumulator's rounding (`tcprobe`).
