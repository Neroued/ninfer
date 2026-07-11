# Q5090 V3 Prefill Length Nsys Kernel Breakdown

> Date: 2026-07-01
> Scope: `qus_bench` prefill-only sweep for Qwen3.6-27B q5090 v3 on one RTX 5090.
> Build commit: `7669670fe3bff3544e65d1d20bab3eec32367b5e`; `worktree_dirty=false`
> in the benchmark JSON artifacts at capture time.
> Nsight Systems: `2025.5.2.266-255236693005v0`.

## Summary

The prefill hotspot changes with sequence length:

- `pp128` through `pp512` are dominated by low-bit linear GEMM kernels.
- `pp1024` is the crossover region: linear work is still larger in aggregate, but
  `gqa_attention_prefill` becomes the single largest kernel.
- `pp2048` and `pp4096` are attention dominated. At `pp4096`,
  `gqa_attention_prefill` is `66.1%` of summed GPU kernel time.

The key per-kernel share shift is:

| kernel label | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gqa_attention_prefill` | 2.5% | 7.4% | 17.1% | 31.8% | 49.2% | 66.1% |
| `rowsplit_gemm_mma_q5` | 46.9% | 39.5% | 35.1% | 28.8% | 21.7% | 14.4% |
| `rowsplit_gemm_mma_q4` | 32.1% | 38.6% | 36.5% | 29.9% | 22.4% | 15.0% |
| `dense_gemm` | 13.9% | 9.5% | 6.1% | 4.6% | 2.7% | 1.6% |

## Method

The throughput sanity run used the requested shape matrix with the sample
repetition policy:

```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
  -p 128,256,512,1024,2048,4096 \
  -r 3 --warmup 1 \
  -o json \
  --output-file profiles/nsys-prefill-length-sweep/qus_bench_prefill_sweep_r3_warmup1.json
```

Nsight Systems profiling used one prompt length per trace:

```bash
nsys profile --force-overwrite=true --stats=false \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  -o profiles/nsys-prefill-length-sweep/pp<P>_prefill_r1_warmup0 \
  ./build/bench/qus_bench \
    --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
    -p <P> -r 1 --warmup 0 \
    -o json \
    --output-file profiles/nsys-prefill-length-sweep/pp<P>_prefill_r1_warmup0.json
```

Reason: current `qus_bench` does not emit per-test NVTX ranges; the generated
`*_nvtx_sum.csv` files are empty. Separate traces avoid mixing different
prefill lengths in one `cuda_gpu_kern_sum` table. Kernel percentages below use
official `nsys stats --report cuda_gpu_kern_sum` output, with summed GPU kernel
duration as the denominator.

The stats/export command was:

```bash
python3 /home/neroued/.codex/skills/nsys-inference-analysis/scripts/nsys_inference_summary.py \
  profiles/nsys-prefill-length-sweep/pp<P>_prefill_r1_warmup0.nsys-rep \
  --out profiles/nsys-prefill-length-sweep/pp<P>_prefill_r1_warmup0.nsys-summary.md

nsys stats --force-export=true --force-overwrite=true \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format csv \
  --output profiles/nsys-prefill-length-sweep/pp<P>_prefill_r1_warmup0 \
  profiles/nsys-prefill-length-sweep/pp<P>_prefill_r1_warmup0.nsys-rep
```

## Artifacts

All raw and derived profiling artifacts are under:

`profiles/nsys-prefill-length-sweep/`

Important files:

| Artifact | Path |
| --- | --- |
| Throughput JSON | `profiles/nsys-prefill-length-sweep/qus_bench_prefill_sweep_r3_warmup1.json` |
| Per-length nsys reports | `profiles/nsys-prefill-length-sweep/pp*_prefill_r1_warmup0.nsys-rep` |
| Per-length SQLite exports | `profiles/nsys-prefill-length-sweep/pp*_prefill_r1_warmup0.sqlite` |
| Per-length nsys summaries | `profiles/nsys-prefill-length-sweep/pp*_prefill_r1_warmup0.nsys-summary.md` |
| Official kernel stats | `profiles/nsys-prefill-length-sweep/pp*_prefill_r1_warmup0_cuda_gpu_kern_sum.csv` |
| Official CUDA API stats | `profiles/nsys-prefill-length-sweep/pp*_prefill_r1_warmup0_cuda_api_sum.csv` |
| Official memop stats | `profiles/nsys-prefill-length-sweep/pp*_prefill_r1_warmup0_cuda_gpu_mem_time_sum.csv` |
| Derived timing summary | `profiles/nsys-prefill-length-sweep/prefill_summary.csv` |
| Derived per-kernel breakdown | `profiles/nsys-prefill-length-sweep/prefill_kernel_breakdown.csv` |
| Derived category breakdown | `profiles/nsys-prefill-length-sweep/prefill_category_breakdown.csv` |

The nsys capture span includes model load and weight upload. Do not read the
CUDA API or memcpy summaries as prefill-only totals. The kernel tables are used
for prefill attribution because the profiled command has a single prefill test
and model loading does not contribute meaningful GPU kernels.

## Timing

| pp | bench r3 mean ms | bench tok/s | nsys single ms | kernel sum ms | kernel/host | work peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 150.661 | 849.7 | 180.968 | 143.437 | 79.3% | 38.1 |
| 256 | 199.307 | 1285.0 | 181.284 | 169.352 | 93.4% | 76.2 |
| 512 | 313.575 | 1633.2 | 318.507 | 303.890 | 95.4% | 152.4 |
| 1024 | 648.093 | 1580.0 | 656.426 | 637.848 | 97.2% | 304.8 |
| 2048 | 1647.076 | 1243.4 | 1668.527 | 1650.935 | 98.9% | 609.5 |
| 4096 | 4829.771 | 848.1 | 4558.258 | 4533.471 | 99.5% | 1219.0 |

Small prompt lengths show more single-run variance under nsys. The `bench r3`
columns are the steadier throughput reference; the nsys single-run columns are
for profiler attribution.

### Workspace Update: GDN Chunked BF16 Scratch

After moving `gated_delta_rule_chunked` to native bf16 boundary I/O and bf16
storage for the chunked scratch buffers that passed `gdn_state_fp32`, the
workspace peak is:

| pp | work peak MiB | KiB/token | work bytes | artifact |
| ---: | ---: | ---: | ---: | --- |
| 4096 | 723.0 | 180.8 | 4 GiB | `profiles/bench/gdn_task_check_final_4096.json` |
| 16384 | 2892.1 | 180.8 | 4 GiB | `profiles/bench/gdn_task_check_final_16384_w4g.json` |

Baseline `pp4096` was 1219.0 MiB, or 304.8 KiB/token. The final `pp4096`
delta is -496.0 MiB, exactly -124.0 KiB/token. With a 4 GiB workspace arena,
the resulting peak model implies about 23203 tokens, or 23168 tokens rounded
down to the chunk multiple.

Scratch storage decisions:

| buffer | final storage | accumulator/math |
| --- | --- | --- |
| `q/k/v/out` boundary temporaries | removed; native bf16 I/O | convert to fp32 in registers/shared memory |
| `g_cumsum` | FP32 | FP32 gating and `exp(g)` |
| `W` | bf16 | T_inv and downstream MMA accumulate FP32 |
| `U` | bf16 | T_inv and downstream subtraction/MMA accumulate FP32 |
| `v_new` | bf16 | chunk output accumulates FP32 |
| `h_chunk` snapshot | bf16 | running state accumulator and `state_out` stay FP32 |

## Category Breakdown

| category | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear` | 93.3% | 88.1% | 78.0% | 63.4% | 46.7% | 31.1% |
| `gqa_attention` | 2.5% | 7.4% | 17.1% | 31.9% | 49.2% | 66.1% |
| `gdn` | 2.5% | 2.6% | 2.6% | 2.8% | 2.3% | 1.6% |
| `normalization` | 1.0% | 1.0% | 1.3% | 1.0% | 0.8% | 0.6% |
| `elementwise_conv_rope` | 0.7% | 0.9% | 1.0% | 0.9% | 0.9% | 0.7% |
| `io_sampling_bookkeeping` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## Per-Kernel Share Matrix

This table is from the original nsys profile above. The `gdn_cast_*` rows are
historical and do not exist in the current native-bf16 chunked path.

| kernel label | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gqa_attention_prefill` | 2.5% | 7.4% | 17.1% | 31.8% | 49.2% | 66.1% |
| `rowsplit_gemm_mma_q5` | 46.9% | 39.5% | 35.1% | 28.8% | 21.7% | 14.4% |
| `rowsplit_gemm_mma_q4` | 32.1% | 38.6% | 36.5% | 29.9% | 22.4% | 15.0% |
| `dense_gemm` | 13.9% | 9.5% | 6.1% | 4.6% | 2.7% | 1.6% |
| `gdn_state_passing` | 0.8% | 0.8% | 0.8% | 1.1% | 1.0% | 0.6% |
| `rmsnorm` | 0.6% | 0.7% | 1.0% | 0.8% | 0.6% | 0.5% |
| `gdn_prepare_wy_wu` | 0.8% | 0.8% | 0.8% | 0.7% | 0.5% | 0.3% |
| `gdn_chunk_output` | 0.6% | 0.6% | 0.5% | 0.6% | 0.5% | 0.4% |
| `silu_and_mul` | 0.3% | 0.4% | 0.5% | 0.5% | 0.6% | 0.4% |
| `lm_head_gemv_q6` | 0.4% | 0.5% | 0.2% | 0.1% | 0.0% | 0.0% |
| `gdn_cast_qkv_bf16_to_f32` | 0.2% | 0.3% | 0.3% | 0.3% | 0.2% | 0.2% |
| `rmsnorm_d5120` | 0.3% | 0.3% | 0.2% | 0.1% | 0.1% | 0.1% |
| `gdn_cast_f32_to_bf16` | 0.1% | 0.1% | 0.2% | 0.1% | 0.1% | 0.1% |
| `residual_add` | 0.2% | 0.2% | 0.2% | 0.2% | 0.2% | 0.1% |
| `causal_conv1d_pairs` | 0.1% | 0.2% | 0.2% | 0.2% | 0.1% | 0.1% |
| `l2norm` | 0.1% | 0.1% | 0.1% | 0.0% | 0.0% | 0.0% |
| `causal_conv1d_state` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `sigmoid_gate_mul` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `rope` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `gdn_gating` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `gqa_attention_kv_fill` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `embed_gather_q6` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `argmax` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `fill_positions` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| `set_pos` | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Full demangled names are preserved in
`profiles/nsys-prefill-length-sweep/prefill_kernel_breakdown.csv`.

## Top Kernels By Length

| pp | rank | kernel | launches | total ms | % kernel | avg us |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 1 | `rowsplit_gemm_mma_q5` | 256 | 67.313 | 46.9% | 262.941 |
| 128 | 2 | `rowsplit_gemm_mma_q4` | 256 | 45.991 | 32.1% | 179.654 |
| 128 | 3 | `dense_gemm` | 96 | 19.912 | 13.9% | 207.416 |
| 128 | 4 | `gqa_attention_prefill` | 16 | 3.553 | 2.5% | 222.039 |
| 128 | 5 | `gdn_prepare_wy_wu` | 48 | 1.190 | 0.8% | 24.782 |
| 256 | 1 | `rowsplit_gemm_mma_q5` | 256 | 66.876 | 39.5% | 261.236 |
| 256 | 2 | `rowsplit_gemm_mma_q4` | 256 | 65.414 | 38.6% | 255.523 |
| 256 | 3 | `dense_gemm` | 96 | 16.036 | 9.5% | 167.038 |
| 256 | 4 | `gqa_attention_prefill` | 16 | 12.500 | 7.4% | 781.280 |
| 256 | 5 | `gdn_state_passing` | 48 | 1.381 | 0.8% | 28.767 |
| 512 | 1 | `rowsplit_gemm_mma_q4` | 256 | 111.071 | 36.5% | 433.870 |
| 512 | 2 | `rowsplit_gemm_mma_q5` | 256 | 106.693 | 35.1% | 416.768 |
| 512 | 3 | `gqa_attention_prefill` | 16 | 51.953 | 17.1% | 3247.085 |
| 512 | 4 | `dense_gemm` | 96 | 18.598 | 6.1% | 193.727 |
| 512 | 5 | `rmsnorm` | 80 | 3.170 | 1.0% | 39.629 |
| 1024 | 1 | `gqa_attention_prefill` | 16 | 203.119 | 31.8% | 12694.934 |
| 1024 | 2 | `rowsplit_gemm_mma_q4` | 256 | 190.722 | 29.9% | 745.008 |
| 1024 | 3 | `rowsplit_gemm_mma_q5` | 256 | 183.742 | 28.8% | 717.741 |
| 1024 | 4 | `dense_gemm` | 96 | 29.516 | 4.6% | 307.460 |
| 1024 | 5 | `gdn_state_passing` | 48 | 6.883 | 1.1% | 143.386 |
| 2048 | 1 | `gqa_attention_prefill` | 16 | 812.769 | 49.2% | 50798.064 |
| 2048 | 2 | `rowsplit_gemm_mma_q4` | 256 | 368.995 | 22.4% | 1441.386 |
| 2048 | 3 | `rowsplit_gemm_mma_q5` | 256 | 357.977 | 21.7% | 1398.346 |
| 2048 | 4 | `dense_gemm` | 96 | 43.951 | 2.7% | 457.825 |
| 2048 | 5 | `gdn_state_passing` | 48 | 15.976 | 1.0% | 332.835 |
| 4096 | 1 | `gqa_attention_prefill` | 16 | 2996.758 | 66.1% | 187297.398 |
| 4096 | 2 | `rowsplit_gemm_mma_q4` | 256 | 681.728 | 15.0% | 2663.000 |
| 4096 | 3 | `rowsplit_gemm_mma_q5` | 256 | 653.630 | 14.4% | 2553.244 |
| 4096 | 4 | `dense_gemm` | 96 | 71.816 | 1.6% | 748.080 |
| 4096 | 5 | `gdn_state_passing` | 48 | 28.469 | 0.6% | 593.109 |

## Readout

- Long prefill optimization should start with
  `gqa_attention_prefill_kernel`. Its summed time grows from `3.553 ms`
  at `pp128` to `2996.758 ms` at `pp4096`, and its average launch time grows
  from `222.039 us` to `187297.398 us`.
- Linear work remains important through `pp1024`. The combined Q4/Q5
  `linear_rowsplit_gemm_mma_kernel` share is `79.0%` at `pp128`,
  `71.6%` at `pp512`, and `58.7%` at `pp1024`, then falls behind attention.
- The per-prefill launch count is stable at `1558` kernel launches across all
  lengths, so the length-dependent curve is dominated by kernel duration growth,
  not by a changing launch surface.
- `lm_head_gemv_q6` is essentially constant around `0.626 ms` because prefill
  returns one final token. Its share disappears at long lengths.

Nsys identifies where time goes, but it does not explain occupancy, memory
throughput, or stall reasons. The next Nsight Compute targets should be:

```bash
ncu --force-overwrite --target-processes all --replay-mode application \
  --set detailed \
  --kernel-name regex:'gqa_attention_prefill_kernel' \
  --launch-count 1 \
  -o profiles/ncu-prefill-length-sweep/pp4096_gqa_attention_prefill \
  ./build/bench/qus_bench \
    --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
    -p 4096 -r 1 --warmup 0 -o json \
    --output-file profiles/ncu-prefill-length-sweep/pp4096_gqa_attention_prefill.json

ncu --force-overwrite --target-processes all --replay-mode application \
  --set detailed \
  --kernel-name regex:'.*linear_rowsplit_gemm_mma_kernel.*Q4Codec.*' \
  --launch-count 1 \
  -o profiles/ncu-prefill-length-sweep/pp1024_rowsplit_gemm_mma_q4 \
  ./build/bench/qus_bench \
    --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
    -p 1024 -r 1 --warmup 0 -o json \
    --output-file profiles/ncu-prefill-length-sweep/pp1024_rowsplit_gemm_mma_q4.json
```
