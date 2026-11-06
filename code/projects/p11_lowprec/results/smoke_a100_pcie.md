# Project 11 results: smoke

Hardware: 2x A100-40GB (PCIe) (NVIDIA A100-PCIE-40GB); torch 2.14.1+cu130, CUDA 13.0. Wall time 3194 s.

Cast columns: share of the gradient elements entering GEMMs that were flushed to zero, by count and by magnitude, as the mean over logged intervals; activation overflow as the worst logged interval. Update columns: mean over the second half of training.

| run | recipe | final val | gap | verdict | g flushed, count (mean) | g flushed, mass (mean) | x overflow (max) | lost updates | pred. lost (bf16) | ms/step |
|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|
| gpt/fp32 | x=fp32 w=fp32 g=fp32 storage=fp32 sens=fp32 params=fp32 | 2.0237 | +0.000 | baseline | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.000 | 0.493 | 13.5 |
| gpt/bf16 | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=fp32 | 2.0195 | -0.020 | ok | 6.60e-07 | 8.51e-44 | 0.00e+00 | 0.000 | 0.492 | 44.2 |
| gpt/bf16_pure | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=bf16/rne | 2.0472 | +0.107 | degraded | 6.59e-07 | 1.66e-43 | 0.00e+00 | 0.491 | 0.491 | 44.3 |
| gpt/bf16_pure_sr | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=bf16/stochastic | 2.0320 | +0.039 | ok | 5.73e-06 | 1.14e-42 | 0.00e+00 | 0.442 | 0.492 | 46.4 |
| gpt/bf16_lowsens | x=bf16 w=bf16 g=bf16 storage=bf16 sens=bf16 params=fp32 | 2.0213 | -0.011 | ok | 3.66e-06 | 7.38e-43 | 0.00e+00 | 0.000 | 0.493 | 48.3 |
| gpt/fp16_noscale | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 | 2.0317 | +0.038 | ok | 1.10e-01 | 5.58e-05 | 0.00e+00 | 0.000 | 0.493 | 44.2 |
| gpt/fp16_dynamic | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 loss-scale=dynamic | 2.0345 | +0.051 | ok | 1.73e-02 | 3.77e-11 | 0.00e+00 | 0.000 | 0.493 | 46.0 |
| gpt/fp16_lowsens | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp16 params=fp32 loss-scale=dynamic | 2.0636 | +0.176 | degraded | 2.59e-02 | 1.41e-10 | 0.00e+00 | 0.000 | 0.490 | 49.2 |
| gpt/bf16_acc | x=bf16 w=bf16 g=bf16 acc=bf16/32 storage=bf16 sens=fp32 params=fp32 | 2.0228 | -0.004 | ok | 7.02e-06 | 1.43e-42 | 0.00e+00 | 0.000 | 0.492 | 63.1 |
| gpt/bf16_acc_promote | x=bf16 w=bf16 g=bf16 acc=bf16/32/p128 storage=bf16 sens=fp32 params=fp32 | 2.0334 | +0.046 | ok | 3.46e-07 | 1.04e-43 | 0.00e+00 | 0.000 | 0.493 | 66.6 |
| gpt/fp8_noscale | x=e4m3 w=e4m3 g=e5m2 storage=bf16 sens=fp32 params=fp32 | 3.4587 | +2.068 | broken, stalled | 8.84e-01 | 1.54e-01 | 0.00e+00 | 0.000 | 0.762 | 44.6 |
| gpt/fp8_tensor | x=e4m3/tensor w=e4m3/tensor g=e5m2/tensor storage=bf16 sens=fp32 params=fp32 | 2.3899 | +1.016 | broken | 2.11e-03 | 1.22e-12 | 0.00e+00 | 0.000 | 0.492 | 47.9 |
| gpt/fp8_tensor_hp | x=e4m3/tensor w=e4m3/tensor g=e5m2/tensor storage=bf16 sens=fp32 params=fp32 hp=head | 2.0468 | +0.106 | degraded | 1.50e-02 | 4.99e-12 | 0.00e+00 | 0.000 | 0.495 | 48.2 |
| gpt/fp8_tensor_sr | x=e4m3/tensor w=e4m3/tensor g=e5m2/tensor/sr storage=bf16 sens=fp32 params=fp32 | 2.0519 | +0.128 | degraded | 1.54e-02 | 5.04e-12 | 0.00e+00 | 0.000 | 0.493 | 51.2 |
| gpt/fp8_delayed | x=e4m3/delayed w=e4m3/delayed g=e5m2/delayed storage=bf16 sens=fp32 params=fp32 | 2.4119 | +1.054 | broken | 2.40e-03 | 2.14e-12 | 1.75e-06 | 0.000 | 0.493 | 47.5 |
| gpt/fp8_e4m3_tensor | x=e4m3/tensor w=e4m3/tensor g=e4m3/tensor storage=bf16 sens=fp32 params=fp32 | 2.1000 | +0.313 | degraded | 8.19e-02 | 4.48e-07 | 0.00e+00 | 0.000 | 0.489 | 47.9 |
| gpt/fp8_block | x=e4m3/block w=e4m3/block g=e4m3/block storage=bf16 sens=fp32 params=fp32 | 2.0544 | +0.138 | degraded | 4.67e-02 | 6.32e-08 | 0.00e+00 | 0.000 | 0.496 | 79.6 |
| gpt/mxfp4 | x=e2m1/mx w=e2m1/mx g=e2m1/mx storage=bf16 sens=fp32 params=fp32 | 3.4748 | +2.077 | broken, stalled | 3.29e-01 | 2.01e-01 | 1.40e-02 | 0.000 | 0.578 | 85.7 |
| gpt/mxfp4_sr | x=e2m1/mx w=e2m1/mx g=e2m1/mx/sr storage=bf16 sens=fp32 params=fp32 | nan | nan | diverged | 2.81e-01 | 1.13e-01 | 1.26e-02 | 0.000 | 0.212 | 86.6 |
| gpt/mxfp4_sr_ceil | x=e2m1/mx w=e2m1/mx g=e2m1/mx-ceil/sr storage=bf16 sens=fp32 params=fp32 | 2.2316 | +0.693 | broken | 3.86e-01 | 3.99e-02 | 8.20e-03 | 0.000 | 0.499 | 84.5 |
| gpt/mxfp4_fwd | x=e2m1/mx w=e2m1/mx g=bf16 bwd-x=bf16 bwd-w=bf16 storage=bf16 sens=fp32 params=fp32 | 2.2319 | +0.694 | broken | 6.87e-04 | 1.71e-40 | 8.02e-03 | 0.000 | 0.514 | 54.9 |
| gpt/nvfp4 | x=e2m1/nvfp4 w=e2m1/nvfp4 g=e2m1/nvfp4 storage=bf16 sens=fp32 params=fp32 | 5.6369 | +2.912 | broken | 3.36e-01 | 3.88e-02 | 0.00e+00 | 0.000 | 0.506 | 104.9 |
| gpt/nvfp4_sr | x=e2m1/nvfp4 w=e2m1/nvfp4 g=e2m1/nvfp4/sr storage=bf16 sens=fp32 params=fp32 | 2.1475 | +0.467 | degraded | 2.98e-01 | 1.55e-02 | 9.77e-06 | 0.000 | 0.503 | 105.8 |
| gpt/nvfp4_sr_hp | x=e2m1/nvfp4 w=e2m1/nvfp4 g=e2m1/nvfp4/sr storage=bf16 sens=fp32 params=fp32 hp=head | 2.1566 | +0.495 | degraded | 2.86e-01 | 9.23e-03 | 6.26e-06 | 0.000 | 0.501 | 98.8 |
| mlp/fp32 | x=fp32 w=fp32 g=fp32 storage=fp32 sens=fp32 params=fp32 | 0.0051 | +0.000 | baseline | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.000 | 0.826 | 2.8 |
| mlp/bf16 | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=fp32 | 0.0051 | +0.001 | ok | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.000 | 0.826 | 21.8 |
| mlp/bf16_pure | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=bf16/rne | 0.0125 | +0.898 | broken, stalled | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.816 | 0.816 | 22.2 |
| mlp/bf16_pure_sr | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=bf16/stochastic | 0.0052 | +0.018 | ok | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.742 | 0.824 | 27.7 |
| mlp/fp16_noscale | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 | 0.0051 | -0.000 | ok | 1.54e-02 | 3.62e-05 | 0.00e+00 | 0.000 | 0.826 | 22.6 |
| mlp/fp16_static | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 loss-scale=static | 0.0051 | +0.000 | ok | 3.12e-05 | 7.24e-11 | 0.00e+00 | 0.000 | 0.826 | 25.8 |
| mlp/fp16_dynamic | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 loss-scale=dynamic | 0.0051 | +0.000 | ok | 1.37e-06 | 4.05e-14 | 0.00e+00 | 0.000 | 0.826 | 26.6 |
| mlp/fp8_noscale | x=e4m3 w=e4m3 g=e5m2 storage=bf16 sens=fp32 params=fp32 | 0.0076 | +0.407 | degraded | 6.85e-01 | 2.10e-01 | 0.00e+00 | 0.000 | 0.823 | 22.5 |
| mlp/fp8_tensor | x=e4m3/tensor w=e4m3/tensor g=e5m2/tensor storage=bf16 sens=fp32 params=fp32 | 0.0058 | +0.138 | degraded | 9.34e-06 | 9.57e-15 | 0.00e+00 | 0.000 | 0.824 | 30.9 |
| mlp/fp8_e4m3_tensor | x=e4m3/tensor w=e4m3/tensor g=e4m3/tensor storage=bf16 sens=fp32 params=fp32 | 0.0057 | +0.122 | degraded | 2.07e-04 | 3.27e-09 | 0.00e+00 | 0.000 | 0.826 | 30.7 |
| mlp/fp8_block | x=e4m3/block w=e4m3/block g=e4m3/block storage=bf16 sens=fp32 params=fp32 | 0.0058 | +0.134 | degraded | 8.81e-05 | 3.84e-10 | 0.00e+00 | 0.000 | 0.827 | 47.8 |
| mlp/mxfp4 | x=e2m1/mx w=e2m1/mx g=e2m1/mx storage=bf16 sens=fp32 params=fp32 | 0.0362 | +1.962 | broken, stalled | 3.06e-01 | 2.54e-02 | 7.59e-03 | 0.000 | 0.825 | 48.6 |
| mlp/mxfp4_sr | x=e2m1/mx w=e2m1/mx g=e2m1/mx/sr storage=bf16 sens=fp32 params=fp32 | 0.0380 | +2.010 | broken, stalled | 2.84e-01 | 2.81e-02 | 7.58e-03 | 0.000 | 0.824 | 47.8 |
| mlp/mxfp4_sr_ceil | x=e2m1/mx w=e2m1/mx g=e2m1/mx-ceil/sr storage=bf16 sens=fp32 params=fp32 | 0.0371 | +1.987 | broken, stalled | 3.35e-01 | 4.21e-02 | 7.58e-03 | 0.000 | 0.824 | 47.4 |
| mlp/mxfp4_fwd | x=e2m1/mx w=e2m1/mx g=bf16 bwd-x=bf16 bwd-w=bf16 storage=bf16 sens=fp32 params=fp32 | 0.0579 | +2.433 | broken, stalled | 0.00e+00 | 0.00e+00 | 7.55e-03 | 0.000 | 0.827 | 21.6 |
| mlp/nvfp4 | x=e2m1/nvfp4 w=e2m1/nvfp4 g=e2m1/nvfp4 storage=bf16 sens=fp32 params=fp32 | 0.0141 | +1.017 | broken | 2.65e-01 | 1.74e-02 | 0.00e+00 | 0.000 | 0.819 | 54.7 |
| mlp/nvfp4_sr | x=e2m1/nvfp4 w=e2m1/nvfp4 g=e2m1/nvfp4/sr storage=bf16 sens=fp32 params=fp32 | 0.0150 | +1.079 | broken, stalled | 2.46e-01 | 1.93e-02 | 0.00e+00 | 0.000 | 0.821 | 56.3 |
| mlp/fp32@lr=0.003 | x=fp32 w=fp32 g=fp32 storage=fp32 sens=fp32 params=fp32 | 0.0007 | +0.000 | baseline | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.000 | 0.647 | 2.8 |
| mlp/bf16_pure@lr=0.003 | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=bf16/rne | 0.0014 | +0.676 | degraded | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.613 | 0.613 | 25.1 |
| mlp/bf16_pure_sr@lr=0.003 | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=bf16/stochastic | 0.0007 | +0.054 | ok | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.572 | 0.643 | 25.7 |
| mlp/fp32@lr=0.0003 | x=fp32 w=fp32 g=fp32 storage=fp32 sens=fp32 params=fp32 | 0.0271 | +0.000 | baseline | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.000 | 0.938 | 3.1 |
| mlp/bf16_pure@lr=0.0003 | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=bf16/rne | 0.0588 | +0.773 | broken, stalled | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.944 | 0.944 | 25.9 |
| mlp/bf16_pure_sr@lr=0.0003 | x=bf16 w=bf16 g=bf16 storage=bf16 sens=fp32 params=bf16/stochastic | 0.0274 | +0.008 | ok | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.879 | 0.937 | 27.1 |
| mlp/fp32@loss_weight=0.01 | x=fp32 w=fp32 g=fp32 storage=fp32 sens=fp32 params=fp32 | 0.0051 | +0.000 | baseline | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.000 | 0.826 | 2.4 |
| mlp/fp16_noscale@loss_weight=0.01 | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 | 0.0054 | +0.056 | ok | 3.58e-01 | 5.57e-02 | 0.00e+00 | 0.000 | 0.826 | 22.7 |
| mlp/fp16_static@loss_weight=0.01 | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 loss-scale=static | 0.0051 | -0.000 | ok | 2.15e-03 | 5.03e-07 | 0.00e+00 | 0.000 | 0.826 | 24.3 |
| mlp/fp16_dynamic@loss_weight=0.01 | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 loss-scale=dynamic | 0.0051 | -0.000 | ok | 4.63e-05 | 1.71e-10 | 0.00e+00 | 0.000 | 0.826 | 30.1 |
| mlp/fp32@loss_weight=0.001 | x=fp32 w=fp32 g=fp32 storage=fp32 sens=fp32 params=fp32 | 0.0051 | +0.000 | baseline | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.000 | 0.826 | 2.5 |
| mlp/fp16_noscale@loss_weight=0.001 | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 | 0.0118 | +0.839 | broken | 5.21e-01 | 1.37e-01 | 0.00e+00 | 0.000 | 0.831 | 22.5 |
| mlp/fp16_static@loss_weight=0.001 | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 loss-scale=static | 0.0051 | -0.000 | ok | 1.51e-02 | 3.47e-05 | 0.00e+00 | 0.000 | 0.826 | 27.6 |
| mlp/fp16_dynamic@loss_weight=0.001 | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 loss-scale=dynamic | 0.0051 | -0.000 | ok | 3.99e-04 | 1.52e-08 | 0.00e+00 | 0.000 | 0.826 | 24.9 |
| mlp/fp32@loss_weight=0.0001 | x=fp32 w=fp32 g=fp32 storage=fp32 sens=fp32 params=fp32 | 0.0051 | +0.000 | baseline | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.000 | 0.826 | 2.5 |
| mlp/fp16_noscale@loss_weight=0.0001 | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 | 0.0751 | +2.693 | broken, stalled | 5.91e-01 | 2.61e-01 | 0.00e+00 | 0.000 | 0.877 | 22.8 |
| mlp/fp16_static@loss_weight=0.0001 | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 loss-scale=static | 0.0051 | +0.000 | ok | 9.12e-02 | 1.99e-03 | 0.00e+00 | 0.000 | 0.826 | 23.5 |
| mlp/fp16_dynamic@loss_weight=0.0001 | x=fp16 w=fp16 g=fp16 storage=fp16 sens=fp32 params=fp32 loss-scale=dynamic | 0.0051 | -0.000 | ok | 3.15e-03 | 1.15e-06 | 0.00e+00 | 0.000 | 0.826 | 24.5 |

## Half-ulp rule: measured vs predicted lost updates

| run | seed | fmt | measured (mean) | predicted (mean) | max abs diff |
|:--|--:|:--|--:|--:|--:|
| gpt/bf16_pure | 0 | bf16 | 0.2985 | 0.2985 | 1.91e-05 |
| mlp/bf16_pure | 0 | bf16 | 0.6741 | 0.6741 | 2.12e-05 |
| mlp/bf16_pure@lr=0.003 | 0 | bf16 | 0.4471 | 0.4471 | 7.08e-06 |
| mlp/bf16_pure@lr=0.0003 | 0 | bf16 | 0.8613 | 0.8613 | 7.08e-06 |

## Failures and fixes (gap to fp32; ln 2 = 0.69 is twice the excess loss)

| model | failure | gap | fixes |
|:--|:--|--:|:--|
| gpt | bf16_pure | +0.107 (degraded) | bf16_pure_sr +0.039 (ok); bf16 -0.020 (ok) |
| gpt | bf16_lowsens | -0.011 (ok) | bf16 -0.020 (ok) |
| gpt | fp16_noscale | +0.038 (ok) | fp16_dynamic +0.051 (ok) |
| gpt | bf16_acc | -0.004 (ok) | bf16_acc_promote +0.046 (ok); bf16 -0.020 (ok) |
| gpt | fp8_noscale | +2.068 (broken, stalled) | fp8_tensor +1.016 (broken); fp8_delayed +1.054 (broken); fp8_block +0.138 (degraded) |
| gpt | fp8_tensor | +1.016 (broken) | fp8_tensor_hp +0.106 (degraded); fp8_tensor_sr +0.128 (degraded); fp8_block +0.138 (degraded) |
| gpt | fp8_e4m3_tensor | +0.313 (degraded) | fp8_tensor +1.016 (broken); fp8_block +0.138 (degraded) |
| gpt | mxfp4 | +2.077 (broken, stalled) | mxfp4_sr nan (diverged); mxfp4_sr_ceil +0.693 (broken); mxfp4_fwd +0.694 (broken) |
| gpt | nvfp4 | +2.912 (broken) | nvfp4_sr +0.467 (degraded); nvfp4_sr_hp +0.495 (degraded) |
| mlp | bf16_pure | +0.898 (broken, stalled) | bf16_pure_sr +0.018 (ok); bf16 +0.001 (ok) |
| mlp | fp16_noscale | -0.000 (ok) | fp16_static +0.000 (ok); fp16_dynamic +0.000 (ok) |
| mlp | fp8_noscale | +0.407 (degraded) | fp8_tensor +0.138 (degraded); fp8_block +0.134 (degraded) |
| mlp | fp8_tensor | +0.138 (degraded) | fp8_block +0.134 (degraded) |
| mlp | fp8_e4m3_tensor | +0.122 (degraded) | fp8_tensor +0.138 (degraded); fp8_block +0.134 (degraded) |
| mlp | mxfp4 | +1.962 (broken, stalled) | mxfp4_sr +2.010 (broken, stalled); mxfp4_sr_ceil +1.987 (broken, stalled); mxfp4_fwd +2.433 (broken, stalled) |
| mlp | nvfp4 | +1.017 (broken) | nvfp4_sr +1.079 (broken, stalled) |
