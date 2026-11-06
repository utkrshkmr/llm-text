# Spec sheet: 2x A100-PCIE-40GB

driver 595.91.07, CUDA 13.0, torch 2.14.1+cu130, config `smoke`

| quantity | measured | datasheet | fraction | how |
|---|---|---|---|---|
| HBM bandwidth, copy | 1,328 GB/s | 1,555 GB/s | 85% | cold copy, largest sizes |
| HBM bandwidth, read | 1,254 GB/s | 1,555 GB/s | 81% | cold sum() |
| HBM bandwidth, write | 1,489 GB/s | 1,555 GB/s | 96% | cold fill_() |
| L2 bandwidth (copy, warm) | 3,127 GB/s | - | - | graph burst, working set in L2 |
| L2 capacity (knee) | 24.07 MiB | 40 MiB | 60% | copy sweep knee |
| H2D pinned | 22.45 GB/s | 31.5 GB/s | 71% | best size |
| H2D pageable | 14.24 GB/s | 31.5 GB/s | 45% | best size |
| D2H pinned | 24.31 GB/s | 31.5 GB/s | 77% | best size |
| D2H pageable | 12.1 GB/s | 31.5 GB/s | 38% | best size |
| GPU0 -> GPU1 (P2P) | 8.338 GB/s | 31.5 GB/s | 26% | copy_, best size: DATA WRONG |
| GPU0 <-> GPU1 (both ways) | 16.63 GB/s | 63 GB/s | 26% | two streams, bytes of both: DATA WRONG |
| GPU0 -> host -> GPU1 (staged) | 18.48 GB/s | 31.5 GB/s | 59% | pipelined through pinned host buffers |
| matmul fp32 | 18.29 TFLOP/s | 19.5 TFLOP/s | 94% | best square GEMM |
| matmul tf32 | 123.7 TFLOP/s | 156 TFLOP/s | 79% | best square GEMM |
| matmul fp16 | 244 TFLOP/s | 312 TFLOP/s | 78% | best square GEMM |
| matmul bf16 | 251.3 TFLOP/s | 312 TFLOP/s | 81% | best square GEMM |
| matmul int8 | 397.1 TOP/s | 624 TOP/s | 64% | best square GEMM |
| matmul bf16, sustained | 216.7 TFLOP/s | 312 TFLOP/s | 69% | 8192^3 back to back, second half of the run |
| SIMT fp32 (FMA chain) | 18.9 TFLOP/s | 19.5 TFLOP/s | 97% | compiled FMA chain |
| tiny kernel, device side | 4.608 us | - | - | events, queue kept ahead |
| empty event pair | 2.56 us | - | - | timer floor |
| host enqueue per kernel | 7.763 us | - | - | no sync in timed loop |
| kernel in a CUDA graph | 1.27 us | - | - | graph replay / n |
| synchronize(), idle | 10.95 us | - | - | wall clock |
| .item() round trip | 9.674 us | - | - | wall clock |
| 4-byte H2D, pinned, + sync | 18.64 us | - | - | wall clock |

## Predicted vs measured kernel times

| kernel | bound | measured_us | datasheet_us | datasheet_err | roofline_us | roofline_err | roofline+launch_us | roofline+launch_err | fitted_loo_us | fitted_loo_err |
|---|---|---|---|---|---|---|---|---|---|---|
| copy_bf16 | memory | 394.2 | 345.3 | -12% | 404.2 | +3% | 408.8 | +4% | 429.2 | +9% |
| add_bf16 | memory | 605.2 | 517.9 | -14% | 606.3 | +0% | 610.9 | +1% | 646.5 | +7% |
| gelu_bf16 | memory | 410.6 | 345.3 | -16% | 404.2 | -2% | 408.8 | -0% | 490.1 | +19% |
| scale_fp32_4MiB | memory | 12.3 | 5.4 | -56% | 6.3 | -49% | 10.9 | -11% | 13.0 | +6% |
| add_fp32_16KiB | launch | 6.1 | 0.0 | -99% | 0.0 | -99% | 4.6 | -24% | 6.8 | +11% |
| sum_fp32 | memory | 399.4 | 345.3 | -14% | 404.2 | +1% | 408.8 | +2% | 436.5 | +9% |
| softmax_bf16 | memory | 372.7 | 172.6 | -54% | 202.1 | -46% | 206.7 | -45% | 224.0 | -40% |
| layernorm_bf16 | memory | 235.5 | 172.6 | -27% | 202.1 | -14% | 206.7 | -12% | 245.6 | +4% |
| rmsnorm_eager_bf16 | memory | 1,954.8 | 1,553.8 | -21% | 1,819.2 | -7% | 1,832.7 | -6% | 1,914.4 | -2% |
| rmsnorm_compiled_bf16 | memory | 197.6 | 172.6 | -13% | 202.1 | +2% | 206.7 | +5% | 233.7 | +18% |
| gemm_bf16_1024 | compute | 22.5 | 6.9 | -69% | 8.5 | -62% | 13.2 | -42% | 19.6 | -13% |
| gemm_bf16_4096 | compute | 548.9 | 440.5 | -20% | 546.8 | -0% | 551.4 | +0% | 660.4 | +20% |
| gemm_bf16_8192 | compute | 5,044.7 | 3,524.1 | -30% | 4,374.5 | -13% | 4,379.1 | -13% | 4,728.0 | -6% |
| gemm_fp32_4096 | compute | 7,809.5 | 7,048.2 | -10% | 7,514.2 | -4% | 7,518.8 | -4% | 7,830.3 | +0% |
| gemm_tf32_4096 | compute | 1,142.8 | 881.0 | -23% | 1,111.0 | -3% | 1,115.6 | -2% | 1,325.8 | +16% |
| gemv_bf16 | memory | 121.9 | 86.3 | -29% | 101.1 | -17% | 105.7 | -13% | 110.6 | -9% |
| gemm_decode16_bf16 | memory | 108.0 | 75.9 | -30% | 88.9 | -18% | 93.5 | -13% | 105.8 | -2% |
| gemm_mlp_bf16 | compute | 2,116.6 | 1,541.8 | -27% | 1,913.9 | -10% | 1,918.5 | -9% | 2,177.4 | +3% |
| sdpa_flash_bf16 | compute | 818.7 | 440.5 | -46% | 546.8 | -33% | 551.4 | -33% | 647.6 | -21% |
| attn_naive_bf16 | memory | 607.7 | 388.4 | -36% | 454.8 | -25% | 461.9 | -24% | 622.9 | +2% |

## Model error summary

| model | n | mape | max_abs | within | log_rmse | geo_mean_ratio | band |
|---|---|---|---|---|---|---|---|
| datasheet | 20 | 0.323 | 0.995 | 0.35 | 1.26 | 0.539 | 0.2 |
| roofline | 20 | 0.204 | 0.994 | 0.7 | 1.19 | 0.64 | 0.2 |
| roofline+launch | 20 | 0.132 | 0.445 | 0.75 | 0.23 | 0.867 | 0.2 |
| fitted_loo | 20 | 0.11 | 0.399 | 0.85 | 0.158 | 1 | 0.2 |
