# Spec sheet: 1x A100-PCIE-40GB

driver 595.91.07, CUDA 13.0, torch 2.14.1+cu130, config `smoke`

| quantity | measured | p10-p90 | datasheet | fraction | how |
|---|---|---|---|---|---|
| HBM bandwidth, copy | 1,325 GB/s | 1,311-1,378 | 1,555 GB/s | 85% | cold copy, plateau |
| HBM bandwidth, read | 1,326 GB/s | 1,272-1,351 | 1,555 GB/s | 85% | cold sum of 64 rows, plateau |
| HBM bandwidth, full reduction | 1,293 GB/s | 1,228-1,322 | 1,555 GB/s | 83% | cold sum(), plateau |
| HBM bandwidth, write | 1,498 GB/s | 1,478-1,508 | 1,555 GB/s | 96% | cold fill_(), plateau |
| L2 bandwidth (copy, warm), lower bound | 3,103 GB/s | - | - | - | graph burst; includes per-kernel gap |
| L2 bandwidth, gap-corrected | 4,055 GB/s | - | - | - | graph burst minus measured gap |
| L2 capacity (knee) | 23.89 MiB | - | 40 MiB | 60% | warm copy sweep knee |
| H2D pinned | 24.25 GB/s | - | 31.5 GB/s | 77% | best size; plateau 23.9 |
| H2D pageable | 14.64 GB/s | - | 31.5 GB/s | 46% | best size; plateau 10.1 |
| D2H pinned | 24.34 GB/s | - | 31.5 GB/s | 77% | best size; plateau 23.5 |
| D2H pageable | 14.5 GB/s | - | 31.5 GB/s | 46% | best size; plateau 8.1 |
| GPU0 -> GPU1 (P2P) | n/a | - | 31.5 GB/s | - | copy_, best size |
| GPU0 <-> GPU1 (both ways) | n/a | - | 63 GB/s | - | two streams, bytes of both |
| GPU0 -> host -> GPU1 (staged) | n/a | - | 31.5 GB/s | - | pipelined through pinned host buffers |
| matmul fp32 | 17.98 TFLOP/s | 17.98-17.98 | 19.5 TFLOP/s | 92% | best square GEMM (burst) |
| matmul tf32 | 116.3 TFLOP/s | 116-116.7 | 156 TFLOP/s | 75% | best square GEMM (burst) |
| matmul fp16 | 229.8 TFLOP/s | 229.6-254.8 | 312 TFLOP/s | 74% | best square GEMM (burst) |
| matmul bf16 | 233 TFLOP/s | 233-234 | 312 TFLOP/s | 75% | best square GEMM (burst) |
| matmul int8 | 416.7 TOP/s | 377.8-417.6 | 624 TOP/s | 67% | best square GEMM (burst) |
| matmul bf16, sustained | 217.2 TFLOP/s | - | 312 TFLOP/s | 70% | 8192^3 back to back, second half of the run |
|   same, scaled to the boost clock | 293.8 TFLOP/s | - | 312 TFLOP/s | 94% | per-sample FLOP/s x boost / clock |
| SIMT fp32 (FMA chain) | 18.97 TFLOP/s | - | 19.5 TFLOP/s | 97% | compiled FMA chain |
| tiny kernel, device side | 5.12 us | 4.096-5.222 | - | - | events, queue kept ahead |
| empty event pair | 2.048 us | 2.048-3.072 | - | - | timer floor |
| host enqueue per kernel | 7.417 us | 7.369-7.495 | - | - | no sync in timed loop |
| kernel in a CUDA graph | 1.27 us | 1.27-1.27 | - | - | graph replay / n |
| synchronize(), idle | 10.09 us | 9.937-11.11 | - | - | wall clock |
| .item() round trip | 9.231 us | 9.082-9.581 | - | - | wall clock |
| 4-byte H2D, pinned, + sync | 18.33 us | 18.08-19.07 | - | - | wall clock |

## Predicted vs measured kernel times

| kernel | bound | measured_us | datasheet_us | datasheet_err | roofline_us | roofline_err | roofline+launch_us | roofline+launch_err | fitted_loo_us | fitted_loo_err |
|---|---|---|---|---|---|---|---|---|---|---|
| copy_bf16 | memory | 394.2 | 345.3 | -12% | 405.2 | +3% | 410.3 | +4% | 428.8 | +9% |
| add_bf16 | memory | 604.7 | 517.9 | -14% | 607.7 | +1% | 612.9 | +1% | 645.3 | +7% |
| gelu_bf16 | memory | 411.1 | 345.3 | -16% | 405.2 | -1% | 410.3 | -0% | 484.4 | +18% |
| scale_fp32_4MiB | memory | 12.3 | 5.4 | -56% | 6.3 | -48% | 11.5 | -7% | 13.1 | +7% |
| add_fp32_16KiB | launch | 6.1 | 0.0 | -99% | 0.0 | -99% | 5.2 | -16% | 7.1 | +16% |
| sum_fp32 | memory | 400.4 | 345.3 | -14% | 405.2 | +1% | 410.3 | +2% | 435.3 | +9% |
| softmax_bf16 | memory | 372.7 | 172.6 | -54% | 202.6 | -46% | 207.7 | -44% | 221.9 | -40% |
| layernorm_bf16 | memory | 235.0 | 172.6 | -27% | 202.6 | -14% | 207.7 | -12% | 242.8 | +3% |
| rmsnorm_eager_bf16 | memory | 1,954.8 | 1,553.8 | -21% | 1,823.4 | -7% | 1,837.4 | -6% | 1,910.1 | -2% |
| rmsnorm_compiled_bf16 | memory | 197.6 | 172.6 | -13% | 202.6 | +3% | 207.7 | +5% | 232.2 | +17% |
| gemm_bf16_1024 | compute | 23.6 | 6.9 | -71% | 9.2 | -61% | 14.3 | -39% | 19.2 | -19% |
| gemm_bf16_4096 | compute | 549.4 | 440.5 | -20% | 589.8 | +7% | 594.9 | +8% | 653.8 | +19% |
| gemm_bf16_8192 | compute | 4,903.9 | 3,524.1 | -28% | 4,718.6 | -4% | 4,723.7 | -4% | 4,703.9 | -4% |
| gemm_fp32_4096 | compute | 7,888.9 | 7,048.2 | -11% | 7,644.7 | -3% | 7,649.8 | -3% | 7,190.3 | -9% |
| gemm_tf32_4096 | compute | 1,127.4 | 881.0 | -22% | 1,182.1 | +5% | 1,187.2 | +5% | 1,297.1 | +15% |
| gemv_bf16 | memory | 120.3 | 86.3 | -28% | 101.3 | -16% | 106.4 | -12% | 110.7 | -8% |
| gemm_decode16_bf16 | memory | 107.5 | 75.9 | -29% | 89.1 | -17% | 94.2 | -12% | 105.8 | -2% |
| gemm_mlp_bf16 | compute | 2,035.7 | 1,541.8 | -24% | 2,064.4 | +1% | 2,069.5 | +2% | 2,170.8 | +7% |
| sdpa_flash_bf16 | compute | 754.7 | 440.5 | -42% | 589.8 | -22% | 594.9 | -21% | 649.8 | -14% |
| attn_naive_bf16 | memory | 590.8 | 388.4 | -34% | 455.8 | -23% | 463.5 | -22% | 621.9 | +5% |

## Model error summary

| model | n | mape | max_abs | within | log_rmse | geo_mean_ratio | band |
|---|---|---|---|---|---|---|---|
| datasheet | 20 | 0.317 | 0.995 | 0.35 | 1.26 | 0.543 | 0.2 |
| roofline | 20 | 0.191 | 0.994 | 0.7 | 1.18 | 0.661 | 0.2 |
| roofline+launch | 20 | 0.113 | 0.443 | 0.8 | 0.201 | 0.903 | 0.2 |
| fitted_loo | 20 | 0.115 | 0.405 | 0.95 | 0.159 | 1 | 0.2 |
| datasheet (out of sample) | 20 | 0.317 | 0.995 | 0.35 | 1.26 | 0.543 | 0.2 |
| roofline (out of sample) | 20 | 0.191 | 0.994 | 0.7 | 1.18 | 0.661 | 0.2 |
| roofline+launch (out of sample) | 20 | 0.113 | 0.443 | 0.8 | 0.201 | 0.903 | 0.2 |
| fitted_loo (out of sample) | 20 | 0.115 | 0.405 | 0.95 | 0.159 | 1 | 0.2 |
