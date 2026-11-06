# Lab 3 — Numbers on the machine: formats, rounding, scaling and stable algorithms

Chapter 3 (`ch:numerics`). Implement in `starter.py`:

1. `decode_float`, `encode_float`: bit patterns of any `FloatFormat` with round-to-nearest-even,
   saturation and the IEEE / OCP "fn" / no-special encodings                       (Task 1)
2. `stochastic_round`                                                                  (Task 2)
3. `mx_quantize`, `mx_dequantize`: OCP microscaling with a shared E8M0 exponent per 32   (Task 3)
4. `pairwise_sum`, `kahan_sum` in a simulated precision                                (Task 4)
5. `online_logsumexp`, `online_softmax`, `welford`                                     (Task 5)
6. `quantize_affine`, `dequantize_affine`: per-group symmetric / asymmetric integers   (Task 6)
7. `hessian_weighted_error`, `obs_update`                                              (Task 7)
8. `accumulate_gemm`, `fp8_block_gemm`: limited-precision accumulators, promotion,
   DeepSeek-V3 style tile/block scaling                                                (Task 8)
9. `nvfp4_quantize`, `nvfp4_dequantize`: E2M1 elements, E4M3 scales per 16, FP32 tensor scale
                                                                                       (Task 9)

The reference library is `llmsys/numerics.py`; format specs (`FP16`, `BF16`, `E4M3`, `E5M2`,
`E2M1`, ...) come from there. The tests compare Task 1 bit for bit with PyTorch's own casts
(`torch.float16`, `torch.bfloat16`, `torch.float8_e4m3fn`, `torch.float8_e5m2`) over every code
and every tie, and the other tasks with the library or with exact (`fractions.Fraction`, float64)
results. The last block of tests checks the library API itself.

```bash
cd code
python -m pytest -q labs/lab03_numerics                     # grades starter.py
LAB_IMPL=solution python -m pytest -q labs/lab03_numerics   # reference (a few seconds, CPU)

# Measurements (GPU optional; about two minutes on one A100). Writes results/bench_results.{json,md}
CUDA_VISIBLE_DEVICES=0 python labs/lab03_numerics/bench.py
python labs/lab03_numerics/bench.py --only updates accum --device cpu --n-weights 65536
python labs/lab03_numerics/bench.py --impl starter         # time your own Task 8 code

# Figures of the chapter (reads results/bench_results.json for the measured ones)
python labs/lab03_numerics/figures.py
```

`bench.py` runs ten experiments: GEMM throughput by dtype (median, p10, p90 over 30 repetitions),
10,000 BF16 weight updates with and without master weights and with stochastic rounding, GEMM error
against the scale granularity for FP8/FP4/INT8/INT4/MX/NVFP4, dot-product error against K for
simulated accumulators and for cuBLAS, the block-scaled FP8 GEMM with a 14-bit accumulator against
K and against the promotion interval (`deepseek`), FP8 flush-to-zero on heavy-tailed data and FP16
loss scaling (`formats`), delayed vs current vs block FP8 scaling with an activation spike
(`scaling`), FP4 GEMM error with randomised Hadamard transforms (`hadamard`), re-quantisation for
the backward pass (`backward`), and probes of the tensor-core accumulator's rounding (`tcprobe`).
