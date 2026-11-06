# Lab 3 bench results (2x A100-40GB (PCIe))

```
{
 "hardware": "2x A100-40GB (PCIe)",
 "torch": "2.14.1+cu130",
 "cuda": "13.0",
 "python": "3.11.18",
 "device": "cuda",
 "gpu": "NVIDIA A100-PCIE-40GB",
 "driver": "595.91.07",
 "seconds": 3.6
}
```

## GEMM throughput (measured)

| dtype | n | median TFLOP/s | p10 | p90 |
|:--|--:|--:|--:|--:|
| fp32 | 8192 | 18.1 | 17.9 | 18.2 |
| tf32 | 8192 | 101.9 | 92.2 | 103.4 |
| fp16 | 8192 | 210.0 | 203.8 | 258.0 |
| bf16 | 8192 | 217.2 | 214.0 | 266.5 |
| int8_b_rowmajor | 8192 | 57.7 | 56.1 | 58.1 |
| int8_b_colmajor | 8192 | 335.0 | 330.2 | 376.7 |

## BF16 updates

fraction unchanged at step 1 (RNE): 0.870

| mode | final relative displacement error |
|:--|--:|
| bf16_rne | 0.9154 |
| bf16_sr | 0.0733 |
| fp32_master | 0.0018 |

## GEMM error vs scale granularity

| format | block | scale | bits/elem | GEMM rel err | act rel err |
|:--|:--|:--|--:|--:|--:|
| fp8_e4m3 | tensor | fp32 | 8.000 | 0.0374 | 0.0265 |
| fp8_e4m3 | row | fp32 | 8.008 | 0.0347 | 0.0225 |
| fp8_e4m3 | 1x128 | fp32 | 8.250 | 0.0302 | 0.0156 |
| fp8_e4m3 | 1x32 | fp32 | 9.000 | 0.0279 | 0.0145 |
| fp8_e4m3 | 1x16 | fp32 | 10.000 | 0.0260 | 0.0135 |
| fp4_e2m1 | tensor | fp32 | 4.000 | 0.6185 | 0.6090 |
| fp4_e2m1 | row | fp32 | 4.008 | 0.5669 | 0.5584 |
| fp4_e2m1 | 1x128 | fp32 | 4.250 | 0.2145 | 0.1853 |
| fp4_e2m1 | 1x32 | fp32 | 5.000 | 0.1458 | 0.1052 |
| fp4_e2m1 | 1x16 | fp32 | 6.000 | 0.1242 | 0.0823 |
| int8 | tensor | fp32 | 8.000 | 0.1563 | 0.1561 |
| int8 | row | fp32 | 8.008 | 0.0769 | 0.0765 |
| int8 | 1x128 | fp32 | 8.250 | 0.0217 | 0.0207 |
| int8 | 1x32 | fp32 | 9.000 | 0.0118 | 0.0106 |
| int8 | 1x16 | fp32 | 10.000 | 0.0089 | 0.0076 |
| int4 | tensor | fp32 | 4.000 | 0.6394 | 0.6144 |
| int4 | row | fp32 | 4.008 | 0.6116 | 0.5990 |
| int4 | 1x128 | fp32 | 4.250 | 0.2590 | 0.2317 |
| int4 | 1x32 | fp32 | 5.000 | 0.1570 | 0.1240 |
| int4 | 1x16 | fp32 | 6.000 | 0.1251 | 0.0917 |
| mxfp8_e4m3 | 1x32 | e8m0 | 8.250 | 0.0476 | 0.0372 |
| mxfp8_e4m3_ceil | 1x32 | e8m0 | 8.250 | 0.0374 | 0.0264 |
| mxfp4 | 1x32 | e8m0 | 4.250 | 0.1912 | 0.1540 |
| mxfp4_ceil | 1x32 | e8m0 | 4.250 | 0.1887 | 0.1480 |
| nvfp4 | 1x16 | e4m3+fp32 | 4.500 | 0.1271 | 0.0854 |

## Accumulation error vs K

| dist | K | sim_fp16_acc | sim_bf16_acc | sim_fp32_acc | sim_fp16_acc_splitk8 | cublas_fp16_out_fp32 | cublas_fp16_out_fp16 | cublas_bf16_out_fp32 | bound_fp16_naive | bound_fp32_naive |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| uniform | 256 | 4.83e-04 | 3.82e-03 | 5.84e-08 | 8.69e-05 | 2.41e-06 | 2.24e-04 | 1.90e-07 | 1.42e-01 | 1.52e-05 |
| uniform | 1024 | 8.70e-04 | 7.15e-03 | 1.09e-07 | 1.27e-04 | 1.06e-06 | 2.17e-04 | 1.87e-07 | 9.98e-01 | 6.10e-05 |
| uniform | 4096 | 1.72e-03 | 1.34e-02 | 2.09e-07 | 2.28e-04 | 2.40e-06 | 2.25e-04 | 1.31e-07 | inf | 2.44e-04 |
| uniform | 16384 | 3.60e-03 | 5.00e-01 | 4.13e-07 | 4.39e-04 | 5.19e-06 | 2.26e-04 | 7.65e-07 | inf | 9.77e-04 |
| normal | 256 | 6.15e-04 | 4.63e-03 | 7.35e-08 | 2.51e-04 | 4.82e-07 | 2.01e-04 | 1.22e-07 | 1.42e-01 | 1.52e-05 |
| normal | 1024 | 1.17e-03 | 9.30e-03 | 1.44e-07 | 4.47e-04 | 2.77e-07 | 2.12e-04 | 1.34e-07 | 9.98e-01 | 6.10e-05 |
| normal | 4096 | 2.33e-03 | 1.84e-02 | 2.83e-07 | 8.22e-04 | 4.81e-07 | 2.93e-04 | 1.60e-07 | inf | 2.44e-04 |
| normal | 16384 | 4.57e-03 | 3.71e-02 | 5.64e-07 | 1.65e-03 | 9.02e-07 | 2.07e-04 | 3.19e-07 | inf | 9.77e-04 |

## Block-scaled FP8 GEMM (total / accumulation-only rel err)

| K | tensor_acc14_nopromote | tensor_acc14_promote128 | block_acc14_promote128 | block_fp32acc |
|--:|--:|--:|--:|--:|
| 512 | 3.66e-02 / 2.97e-04 | 3.66e-02 / 1.27e-04 | 3.11e-02 / 1.26e-04 | 3.11e-02 / 4.64e-08 |
| 1024 | 3.68e-02 / 6.01e-04 | 3.67e-02 / 1.24e-04 | 3.06e-02 / 1.26e-04 | 3.06e-02 / 5.57e-08 |
| 2048 | 3.73e-02 / 1.68e-03 | 3.72e-02 / 1.28e-04 | 3.22e-02 / 1.28e-04 | 3.22e-02 / 8.31e-08 |
| 4096 | 3.73e-02 / 3.49e-03 | 3.72e-02 / 1.15e-04 | 3.43e-02 / 1.16e-04 | 3.43e-02 / 1.15e-07 |
| 8192 | 3.81e-02 / 6.61e-03 | 3.75e-02 / 1.17e-04 | 3.50e-02 / 1.17e-04 | 3.50e-02 / 1.50e-07 |

| promotion interval | accumulation rel err |
|--:|--:|
| 32 | 4.61e-05 |
| 64 | 6.95e-05 |
| 128 | 1.17e-04 |
| 256 | 2.12e-04 |
| 512 | 4.08e-04 |
| 1024 | 8.02e-04 |
| 2048 | 1.60e-03 |
| 8192 | 6.47e-03 |

## FP8 flush on a heavy-tailed tensor

| block | format | flushed | median rel err |
|:--|:--|--:|--:|
| tensor | fp8_e4m3 | 0.5940 | 1.0000 |
| tensor | fp8_e5m2 | 0.0011 | 0.0435 |
| 1x128 | fp8_e4m3 | 0.0216 | 0.0239 |
| 1x128 | fp8_e5m2 | 0.0000 | 0.0430 |
| 128x128 | fp8_e4m3 | 0.1830 | 0.0411 |
| 128x128 | fp8_e5m2 | 0.0001 | 0.0433 |

| FP16 loss scale | flushed | subnormal | overflow |
|--:|--:|--:|--:|
| 1 | 1.04e-01 | 8.81e-01 | 0 |
| 1024 | 1.66e-04 | 1.65e-01 | 0 |
| 65536 | 2.86e-06 | 5.28e-03 | 0 |
| 16777216 | 0.00e+00 | 1.81e-05 | 21 |

## Delayed vs current vs block FP8 scaling

spike step: delayed saturated 996 elements, errors delayed 0.9143 / current 0.0271 / block 0.0257

calm steps mean: {"delayed_err": 0.026471967677213087, "current_err": 0.026458092333982466, "block_err": 0.02572232678601732}

16 steps after the spike, mean: {"delayed_err": 0.026499586241093245, "current_err": 0.02645626766111555, "block_err": 0.02572077182656772, "delayed_flushed": 649.1875, "current_flushed": 10.0}

## FP4 GEMM error with a randomised Hadamard transform

| Hadamard block | act_amax_over_rms | mxfp4 | nvfp4 | int4_g32 |
|--:|--:|--:|--:|--:|
| none | 68.6907 | 0.1912 | 0.1271 | 0.1570 |
| 16 | 18.0681 | 0.1615 | 0.1449 | 0.1256 |
| 32 | 13.3038 | 0.1586 | 0.1440 | 0.1201 |
| 128 | 7.5476 | 0.1582 | 0.1397 | 0.1227 |
| 4096 | 5.0707 | 0.1622 | 0.1341 | 0.1348 |

## Re-quantising for the backward pass (rel err vs source)

| case | forward_copy | backward_direct | backward_requantised | requant_vs_forward_copy | two_d_blocks |
|:--|--:|--:|--:|--:|--:|
| E4M3 fp32 scales | 1.56e-02 | 2.58e-02 | 3.01e-02 | 2.58e-02 | 2.61e-02 |
| E4M3 pow2 scales | 2.64e-02 | 2.64e-02 | 2.64e-02 | 1.71e-08 | 2.64e-02 |
| E2M1 E4M3 scales | 8.55e-02 | 9.47e-02 | 1.27e-01 | 9.35e-02 | 1.41e-01 |

## Tensor-core accumulator probes

```
{
 "probes": {
  "fp16": {
   "1+0.75ulp": -0.75,
   "1+15*0.5ulp": -0.5,
   "-1+0.75ulp": -0.25
  },
  "bf16": {
   "1+0.75ulp": -0.75,
   "1+15*0.5ulp": -0.5,
   "-1+0.75ulp": -0.25
  },
  "fp32_sgemm": {
   "1+0.75ulp": 0.25,
   "1+15*0.5ulp": -3.5,
   "-1+0.75ulp": 0.25
  }
 },
 "kernels": {
  "fp16": [
   "void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_f16_64x64_64x6_nn_align8>",
   "void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, "
  ],
  "bf16": [
   "ampere_s16816gemm_bf16_64x64_ldg8_stages_64x6_nn",
   "void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, "
  ]
 },
 "same_data": [
  {
   "K": 256,
   "fp16_kernel": 5.240628191185794e-07,
   "bf16_kernel": 2.0531828916795888e-07,
   "model_rne16": 5.940988441135302e-08
  },
  {
   "K": 4096,
   "fp16_kernel": 5.127322034531514e-07,
   "bf16_kernel": 1.3697116227255696e-07,
   "model_rne16": 2.11717486985291e-07
  },
  {
   "K": 16384,
   "fp16_kernel": 1.3650685725420702e-06,
   "bf16_kernel": 8.137346497948444e-07,
   "model_rne16": 4.2382018611101243e-07
  }
 ]
}
```
