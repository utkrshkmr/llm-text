# Empirical roofline (bf16)

Measured on 2x A100-40GB (PCIe) (NVIDIA A100-PCIE-40GB); torch 2.14.1+cu130, CUDA 13.0, driver 595.91.07. Median of 20 repetitions after a timed warm-up; p10-p90 in brackets.

Roof: 216.9 TFLOP/s (best bf16 GEMM), 1369 GB/s (copy), ridge 158 FLOP/byte.

| copy buffer | GB/s |
|---:|---:|
| 1 MiB | 139 |
| 4 MiB | 534 |
| 16 MiB | 1045 |
| 64 MiB | 1222 |
| 256 MiB | 1343 |
| 1024 MiB | 1369 |

| GEMM n | data | TFLOP/s (p90-p10 time) |
|---:|:--|---:|
| 1024 | randn | 62.2 (61.7-62.6) |
| 2048 | randn | 131.2 (130.6-131.6) |
| 4096 | randn | 201.3 (200.9-201.4) |
| 8192 | randn | 216.9 (216.8-217.0) |
| 16384 | randn | 215.3 (211.9-217.4) |
| 8192 | zeros | 295.5 (289.2-295.6) |
| 16384 | zeros | 290.4 (289.1-292.4) |

| kernel | FLOP/byte | time (us) | TFLOP/s | GB/s | % of roof |
|:--|---:|---:|---:|---:|---:|
| add | 0.167 | 606.4 | 0.22 | 1328 | 97 |
| rowsum | 0.5 | 202.9 | 0.66 | 1324 | 97 |
| softmax | 1.25 | 804.5 | 0.83 | 667 | 49 |
| gemv | 1 | 118.2 | 1.14 | 1136 | 83 |
| gemm_m16 | 15.9 | 119.0 | 18.04 | 1132 | 83 |
| gemm_m256 | 241 | 216.1 | 159.03 | 660 | 73 |
| gemm_square | 2.73e+03 | 5084.2 | 216.26 | 79 | 100 |
| attn_prefill | 1.02e+03 | 1020.2 | 134.75 | 132 | 62 |
| attn_decode | 4 | 236.7 | 4.54 | 1135 | 83 |

Activation checkpointing, 16 MLP blocks (d=4096, hidden 16384, 8192 tokens, weights 4.30 GB):

| variant | saved activations (GB) | peak fwd+bwd above weights (GB) | step (s) |
|:--|---:|---:|---:|
| full | 9.66 | 10.33 | 0.501 |
| checkpoint | 1.07 | 2.08 | 0.587 |
