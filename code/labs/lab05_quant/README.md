# Lab 5 — Scaling, quantisation and mixed precision

Chapter 5 (`ch:quant`). Implement in `starter.py`:

1. `mx_quantize`, `mx_dequantize`: OCP microscaling with a shared E8M0 exponent per 32   (Task 1)
2. `nvfp4_quantize`, `nvfp4_dequantize`: E2M1 elements, E4M3 scales per 16, FP32 tensor scale
                                                                                       (Task 2)
3. `hadamard_transform`: the fast block Walsh-Hadamard transform with random signs      (Task 3)
4. `quantize_affine`, `dequantize_affine`: per-group symmetric / asymmetric integers   (Task 4)
5. `hessian_weighted_error`, `obs_update`                                              (Task 5)
6. `fp8_block_gemm`: DeepSeek-V3 style 1x128 / 128x128 FP8 scaling, a limited-precision
   accumulator and promotion every 128                                                 (Task 6)

The reference library is `llmsys/numerics.py`; the formats and the bit-exact rounding routine
`round_to_format` are the ones built in the previous lab, `labs/lab04_numerics` (Chapter 4). The
tests compare every task with the library (`numerics.mx_quantize`, `nvfp4_quantize`,
`hadamard_transform`, `quantize`, `fp8_gemm_sim`) or with exact results (least squares, the actual
layer-output loss). The last block of tests checks the scaling and quantisation parts of the
library API: the NF4 grid, block shapes and metadata, the loss scaler and amax history, delayed
scaling on an activation spike, and re-quantisation with power-of-two scales.

```bash
cd code
python -m pytest -q labs/lab05_quant                     # grades starter.py
LAB_IMPL=solution python -m pytest -q labs/lab05_quant   # reference (a few seconds, CPU)

# Measurements (all emulated; a GPU only makes them faster). Writes results/bench_results.{json,md}
CUDA_VISIBLE_DEVICES=0 python labs/lab05_quant/bench.py
python labs/lab05_quant/bench.py --only hadamard backward --device cpu
python labs/lab05_quant/bench.py --impl starter         # use your own Task 6 code

# Figures of the chapter (reads results/bench_results.json for the measured ones)
python labs/lab05_quant/figures.py
```

`bench.py` runs six experiments: GEMM error against the scale granularity for FP8/FP4/INT8/INT4 and
for the MX and NVFP4 formats (`blocks`), FP4 GEMM error with randomised Hadamard transforms
(`hadamard`), re-quantisation for the backward pass (`backward`), the block-scaled FP8 GEMM with a
14-bit accumulator against per-tensor scaling and against K (`deepseek`), delayed vs current vs
block FP8 scaling with an activation spike (`scaling`), and FP16 loss scaling of a gradient-like
tensor (`lossscale`).
