# Q5090 V3 Prefill Length Nsys Report After GQA Optimization

> Date: 2026-07-02
> Scope: current `qus_bench` prefill-only sweep after the GQA prefill optimization,
> Qwen3.6-27B q5090 v3, one RTX 5090.
> Current build commit: `fbccbd86ff6b7f02d71a04e082887c52626cffc2`;
> `worktree_dirty=false` in benchmark artifacts.
> Baseline comparison: `profiles/nsys-prefill-length-sweep/`, captured at
> `7669670fe3bff3544e65d1d20bab3eec32367b5e` for
> `docs/bench/q5090-v3-prefill-length-nsys-kernel-breakdown.md`.
> Nsight Systems: `2025.5.2.266-255236693005v0`.

## Summary

The optimization moved the long-prefill bottleneck away from GQA attention and
back to low-bit linear GEMM:

- `pp4096` host prefill mean improved from `4829.771 ms` to `1538.345 ms`
  (`3.14x`).
- `pp4096` summed GPU kernel time improved from `4533.471 ms` to `1506.836 ms`
  (`3.01x`).
- `gqa_attention_prefill` at `pp4096` dropped from `2996.758 ms` and `66.1%`
  of kernel time to `22.413 ms` and `1.5%` (`133.7x` kernel-time reduction).
- The current `pp4096` top kernels are `rowsplit_gemm_mma_q4` (`44.0%`) and
  `rowsplit_gemm_mma_q5` (`42.7%`). Together with `dense_gemm`, linear work is
  `91.5%` of kernel time.
- Extended `pp8192` and `pp16384` runs keep the same shape: `pp16384` is
  `6216.728 ms` at `2635.5 tok/s`, with linear work still `87.5%` of kernel
  time and GQA attention at `5.0%`.
- Kernel launch count dropped from `1558` to `1462` for each prefill. The
  removed surface corresponds to the old GDN cast kernels no longer present in
  the native-bf16 chunked path.

## Method

The throughput sanity run used the same shape matrix and repetition policy as
the previous report:

```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
  -p 128,256,512,1024,2048,4096 \
  -r 3 --warmup 1 \
  -o json \
  --output-file profiles/nsys-prefill-length-sweep-2026-07-02/qus_bench_prefill_sweep_r3_warmup1.json
```

The `pp8192` and `pp16384` extension used the same repetition policy in a
separate run to avoid replacing the original 128-4096 artifact:

```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
  -p 8192,16384 \
  -r 3 --warmup 1 \
  -o json \
  --output-file profiles/nsys-prefill-length-sweep-2026-07-02/qus_bench_prefill_sweep_8192_16384_r3_warmup1.json
```

Nsight Systems used one prompt length per trace:

```bash
nsys profile --force-overwrite=true --stats=false \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  -o profiles/nsys-prefill-length-sweep-2026-07-02/pp<P>_prefill_r1_warmup0 \
  ./build/bench/qus_bench \
    --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
    -p <P> -r 1 --warmup 0 \
    -o json \
    --output-file profiles/nsys-prefill-length-sweep-2026-07-02/pp<P>_prefill_r1_warmup0.json
```

Current traces include an NVTX range named `qus_bench.prefill.pp<P>`. The
one-length-per-trace layout is still kept so `cuda_gpu_kern_sum` remains directly
comparable with the previous report.

Stats/export command:

```bash
python3 /home/neroued/.codex/skills/nsys-inference-analysis/scripts/nsys_inference_summary.py \
  profiles/nsys-prefill-length-sweep-2026-07-02/pp<P>_prefill_r1_warmup0.nsys-rep \
  --out profiles/nsys-prefill-length-sweep-2026-07-02/pp<P>_prefill_r1_warmup0.nsys-summary.md

nsys stats --force-export=true --force-overwrite=true \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format csv \
  --output profiles/nsys-prefill-length-sweep-2026-07-02/pp<P>_prefill_r1_warmup0 \
  profiles/nsys-prefill-length-sweep-2026-07-02/pp<P>_prefill_r1_warmup0.nsys-rep
```

## Artifacts

All new raw and derived artifacts are under:

`profiles/nsys-prefill-length-sweep-2026-07-02/`

Important files:

| Artifact | Path |
| --- | --- |
| Throughput JSON | `profiles/nsys-prefill-length-sweep-2026-07-02/qus_bench_prefill_sweep_r3_warmup1.json` |
| 8k/16k throughput JSON | `profiles/nsys-prefill-length-sweep-2026-07-02/qus_bench_prefill_sweep_8192_16384_r3_warmup1.json` |
| Per-length nsys reports | `profiles/nsys-prefill-length-sweep-2026-07-02/pp*_prefill_r1_warmup0.nsys-rep` |
| Per-length SQLite exports | `profiles/nsys-prefill-length-sweep-2026-07-02/pp*_prefill_r1_warmup0.sqlite` |
| Per-length summaries | `profiles/nsys-prefill-length-sweep-2026-07-02/pp*_prefill_r1_warmup0.nsys-summary.md` |
| Official kernel stats | `profiles/nsys-prefill-length-sweep-2026-07-02/pp*_prefill_r1_warmup0_cuda_gpu_kern_sum.csv` |
| Official NVTX stats | `profiles/nsys-prefill-length-sweep-2026-07-02/pp*_prefill_r1_warmup0_nvtx_sum.csv` |
| Derived timing summary | `profiles/nsys-prefill-length-sweep-2026-07-02/prefill_summary.csv` |
| Derived per-kernel breakdown | `profiles/nsys-prefill-length-sweep-2026-07-02/prefill_kernel_breakdown.csv` |
| Derived category breakdown | `profiles/nsys-prefill-length-sweep-2026-07-02/prefill_category_breakdown.csv` |
| Derived comparison | `profiles/nsys-prefill-length-sweep-2026-07-02/comparison_vs_2026_07_01.csv` |

The full nsys capture still includes model load and weight upload. The prefill
timing rows below use `qus_bench` host timing and the current NVTX
`qus_bench.prefill.pp<P>` range; kernel percentages use summed GPU kernel time.

## Before And After

| pp | old bench ms | new bench ms | speedup | old kernel ms | new kernel ms | kernel speedup | old MiB | new MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 150.661 | 137.243 | 1.10x | 143.437 | 129.461 | 1.11x | 38.1 | 22.6 |
| 256 | 199.307 | 167.681 | 1.19x | 169.352 | 157.280 | 1.08x | 76.2 | 45.2 |
| 512 | 313.575 | 237.138 | 1.32x | 303.890 | 226.283 | 1.34x | 152.4 | 90.4 |
| 1024 | 648.093 | 408.090 | 1.59x | 637.848 | 395.971 | 1.61x | 304.8 | 180.8 |
| 2048 | 1647.076 | 780.992 | 2.11x | 1650.935 | 767.277 | 2.15x | 609.5 | 361.5 |
| 4096 | 4829.771 | 1538.345 | 3.14x | 4533.471 | 1506.836 | 3.01x | 1219.0 | 723.0 |

## Current Timing

| pp | bench r3 mean ms | tok/s | nsys ms | NVTX ms | kernel sum ms | kernel/host | work peak MiB | kernels |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 137.243 | 932.7 | 142.938 | 142.865 | 129.461 | 90.6% | 22.6 | 1462 |
| 256 | 167.681 | 1526.8 | 191.761 | 191.698 | 157.280 | 82.0% | 45.2 | 1462 |
| 512 | 237.138 | 2159.2 | 240.901 | 240.781 | 226.283 | 93.9% | 90.4 | 1462 |
| 1024 | 408.090 | 2509.3 | 408.924 | 408.851 | 395.971 | 96.8% | 180.8 | 1462 |
| 2048 | 780.992 | 2622.3 | 784.492 | 784.412 | 767.277 | 97.8% | 361.5 | 1462 |
| 4096 | 1538.345 | 2662.6 | 1531.943 | 1531.724 | 1506.836 | 98.4% | 723.0 | 1462 |
| 8192 | 3055.059 | 2681.5 | 3059.536 | 3059.413 | 3025.658 | 98.9% | 1446.1 | 1462 |
| 16384 | 6216.728 | 2635.5 | 6204.421 | 6204.063 | 6146.795 | 99.1% | 2892.1 | 1462 |

## GQA Kernel Delta

| pp | old gqa ms | old share | new gqa ms | new share | gqa speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3.553 | 2.5% | 0.343 | 0.3% | 10.3x |
| 256 | 12.500 | 7.4% | 0.504 | 0.3% | 24.8x |
| 512 | 51.953 | 17.1% | 1.082 | 0.5% | 48.0x |
| 1024 | 203.119 | 31.8% | 2.333 | 0.6% | 87.1x |
| 2048 | 812.769 | 49.2% | 7.172 | 0.9% | 113.3x |
| 4096 | 2996.758 | 66.1% | 22.413 | 1.5% | 133.7x |

## Current Category Breakdown

| category | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4096 | pp8192 | pp16384 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear` | 96.0% | 95.2% | 93.8% | 93.9% | 92.6% | 91.5% | 89.7% | 87.5% |
| `gqa_attention` | 0.3% | 0.3% | 0.5% | 0.6% | 0.9% | 1.5% | 2.8% | 5.0% |
| `gdn` | 2.1% | 2.4% | 3.0% | 3.0% | 3.1% | 3.4% | 3.4% | 3.3% |
| `normalization` | 0.9% | 1.1% | 1.4% | 1.3% | 1.5% | 1.6% | 1.8% | 1.9% |
| `elementwise_conv_rope` | 0.7% | 1.0% | 1.2% | 1.3% | 1.8% | 2.0% | 2.3% | 2.3% |
| `io_sampling_bookkeeping` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Current Per-Kernel Share Matrix

| kernel label | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4096 | pp8192 | pp16384 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rowsplit_gemm_mma_q5` | 49.2% | 42.5% | 42.4% | 42.9% | 43.1% | 42.7% | 41.6% | 40.5% |
| `rowsplit_gemm_mma_q4` | 33.4% | 42.2% | 44.1% | 43.7% | 44.0% | 44.0% | 43.5% | 42.7% |
| `dense_gemm` | 12.9% | 10.2% | 7.1% | 7.2% | 5.5% | 4.8% | 4.6% | 4.2% |
| `gqa_attention_prefill` | 0.3% | 0.3% | 0.5% | 0.6% | 0.9% | 1.5% | 2.8% | 5.0% |
| `gdn_state_passing` | 0.6% | 0.9% | 1.1% | 1.3% | 1.5% | 1.7% | 1.8% | 1.8% |
| `rmsnorm` | 0.5% | 0.7% | 1.1% | 1.0% | 1.3% | 1.3% | 1.3% | 1.3% |
| `gdn_prepare_wy_wu` | 1.0% | 0.9% | 1.3% | 1.0% | 0.9% | 0.9% | 0.9% | 0.8% |
| `silu_and_mul` | 0.3% | 0.4% | 0.5% | 0.6% | 1.0% | 1.1% | 1.2% | 1.1% |
| `gdn_chunk_output` | 0.5% | 0.6% | 0.6% | 0.6% | 0.8% | 0.8% | 0.8% | 0.7% |
| `residual_add` | 0.2% | 0.2% | 0.3% | 0.3% | 0.3% | 0.4% | 0.6% | 0.6% |
| `lm_head_gemv_q6` | 0.5% | 0.4% | 0.3% | 0.2% | 0.1% | 0.0% | 0.0% | 0.0% |
| `rmsnorm_d5120` | 0.3% | 0.3% | 0.2% | 0.2% | 0.2% | 0.2% | 0.4% | 0.5% |
| `causal_conv1d_pairs` | 0.2% | 0.2% | 0.3% | 0.3% | 0.3% | 0.4% | 0.4% | 0.4% |
| `sigmoid_gate_mul` | 0.0% | 0.0% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% |
| `l2norm` | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% | 0.1% |
| `rope` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.1% |
| `causal_conv1d_state` | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `gdn_gating` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `gqa_attention_kv_fill` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `embed_gather_q6` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `argmax` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `fill_positions` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `set_pos` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Full demangled names are preserved in
`profiles/nsys-prefill-length-sweep-2026-07-02/prefill_kernel_breakdown.csv`.

## Current Top Kernels By Length

| pp | rank | kernel | launches | total ms | % kernel | avg us |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 1 | `rowsplit_gemm_mma_q5` | 256 | 63.748 | 49.2% | 249.017 |
| 128 | 2 | `rowsplit_gemm_mma_q4` | 256 | 43.181 | 33.4% | 168.674 |
| 128 | 3 | `dense_gemm` | 96 | 16.686 | 12.9% | 173.816 |
| 128 | 4 | `gdn_prepare_wy_wu` | 48 | 1.251 | 1.0% | 26.065 |
| 128 | 5 | `gdn_state_passing` | 48 | 0.781 | 0.6% | 16.264 |
| 256 | 1 | `rowsplit_gemm_mma_q5` | 256 | 66.808 | 42.5% | 260.969 |
| 256 | 2 | `rowsplit_gemm_mma_q4` | 256 | 66.296 | 42.2% | 258.968 |
| 256 | 3 | `dense_gemm` | 96 | 16.038 | 10.2% | 167.061 |
| 256 | 4 | `gdn_state_passing` | 48 | 1.392 | 0.9% | 29.009 |
| 256 | 5 | `gdn_prepare_wy_wu` | 48 | 1.360 | 0.9% | 28.325 |
| 512 | 1 | `rowsplit_gemm_mma_q4` | 256 | 99.775 | 44.1% | 389.744 |
| 512 | 2 | `rowsplit_gemm_mma_q5` | 256 | 95.907 | 42.4% | 374.636 |
| 512 | 3 | `dense_gemm` | 96 | 15.965 | 7.1% | 166.302 |
| 512 | 4 | `gdn_prepare_wy_wu` | 48 | 2.896 | 1.3% | 60.334 |
| 512 | 5 | `rmsnorm` | 80 | 2.540 | 1.1% | 31.747 |
| 1024 | 1 | `rowsplit_gemm_mma_q4` | 256 | 172.880 | 43.7% | 675.312 |
| 1024 | 2 | `rowsplit_gemm_mma_q5` | 256 | 169.717 | 42.9% | 662.958 |
| 1024 | 3 | `dense_gemm` | 96 | 28.407 | 7.2% | 295.906 |
| 1024 | 4 | `gdn_state_passing` | 48 | 5.278 | 1.3% | 109.964 |
| 1024 | 5 | `rmsnorm` | 80 | 4.077 | 1.0% | 50.959 |
| 2048 | 1 | `rowsplit_gemm_mma_q4` | 256 | 337.241 | 44.0% | 1317.346 |
| 2048 | 2 | `rowsplit_gemm_mma_q5` | 256 | 330.845 | 43.1% | 1292.362 |
| 2048 | 3 | `dense_gemm` | 96 | 41.856 | 5.5% | 435.997 |
| 2048 | 4 | `gdn_state_passing` | 48 | 11.385 | 1.5% | 237.188 |
| 2048 | 5 | `rmsnorm` | 80 | 9.732 | 1.3% | 121.652 |
| 4096 | 1 | `rowsplit_gemm_mma_q4` | 256 | 662.824 | 44.0% | 2589.155 |
| 4096 | 2 | `rowsplit_gemm_mma_q5` | 256 | 642.820 | 42.7% | 2511.014 |
| 4096 | 3 | `dense_gemm` | 96 | 72.302 | 4.8% | 753.143 |
| 4096 | 4 | `gdn_state_passing` | 48 | 26.039 | 1.7% | 542.484 |
| 4096 | 5 | `gqa_attention_prefill` | 16 | 22.413 | 1.5% | 1400.824 |
| 8192 | 1 | `rowsplit_gemm_mma_q4` | 256 | 1316.659 | 43.5% | 5143.198 |
| 8192 | 2 | `rowsplit_gemm_mma_q5` | 256 | 1258.306 | 41.6% | 4915.257 |
| 8192 | 3 | `dense_gemm` | 96 | 137.838 | 4.6% | 1435.814 |
| 8192 | 4 | `gqa_attention_prefill` | 16 | 83.287 | 2.8% | 5205.427 |
| 8192 | 5 | `gdn_state_passing` | 48 | 53.424 | 1.8% | 1113.002 |
| 16384 | 1 | `rowsplit_gemm_mma_q4` | 256 | 2626.448 | 42.7% | 10259.561 |
| 16384 | 2 | `rowsplit_gemm_mma_q5` | 256 | 2491.931 | 40.5% | 9734.105 |
| 16384 | 3 | `gqa_attention_prefill` | 16 | 308.680 | 5.0% | 19292.505 |
| 16384 | 4 | `dense_gemm` | 96 | 257.559 | 4.2% | 2682.910 |
| 16384 | 5 | `gdn_state_passing` | 48 | 108.431 | 1.8% | 2258.974 |

## Readout

- GQA prefill is no longer the long-context limiter in this sweep. It remains
  visible and grows with length, but is still only `5.0%` of `pp16384` kernel
  time.
- Current long-prefill work is overwhelmingly linear. The next broad
  optimization target is the rowsplit prefill GEMM path, especially Q4/Q5
  tensor-core kernels at `pp8192` and `pp16384`.
- GDN is no longer dominated by bf16/fp32 cast kernels. Its remaining share is
  `3.3%` at `pp16384`, mostly `gdn_state_passing`, `gdn_prepare_wy_wu`, and
  `gdn_chunk_output`.
- The long-prefill throughput curve stays flat through 16k: `pp4096` is
  `2663 tok/s`, `pp8192` is `2681 tok/s`, and `pp16384` is `2635 tok/s` in
  the r3 benchmark, compared with the previous collapse to `848 tok/s` at
  `pp4096`.

Nsys shows the new hotspot location, but not occupancy, memory throughput, or
stall reasons. The next Nsight Compute targets should be the current top linear
kernels:

```bash
ncu --force-overwrite --target-processes all --replay-mode application \
  --set detailed \
  --kernel-name regex:'.*linear_rowsplit_gemm_mma_kernel.*Q4Codec.*' \
  --launch-count 1 \
  -o profiles/ncu-prefill-length-sweep-2026-07-02/pp16384_rowsplit_gemm_mma_q4 \
  ./build/bench/qus_bench \
    --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
    -p 16384 -r 1 --warmup 0 -o json \
    --output-file profiles/ncu-prefill-length-sweep-2026-07-02/pp16384_rowsplit_gemm_mma_q4.json

ncu --force-overwrite --target-processes all --replay-mode application \
  --set detailed \
  --kernel-name regex:'.*linear_rowsplit_gemm_mma_kernel.*Q5Codec.*' \
  --launch-count 1 \
  -o profiles/ncu-prefill-length-sweep-2026-07-02/pp16384_rowsplit_gemm_mma_q5 \
  ./build/bench/qus_bench \
    --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
    -p 16384 -r 1 --warmup 0 -o json \
    --output-file profiles/ncu-prefill-length-sweep-2026-07-02/pp16384_rowsplit_gemm_mma_q5.json
```
