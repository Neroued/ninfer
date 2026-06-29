# M3 Long Decode Attention Bottleneck Report

> Date: 2026-06-29
> Scope: Qwen3.6-27B q5090 long decode profiling on the current local build.

## Executive Summary

A full Nsight Systems trace was captured for the FEM prompt that generated 2863 tokens and showed
low end-to-end throughput. The slowdown is not caused by prefill and is no longer primarily caused
by lowbit GEMV.

The measured root cause is decode attention. In the long decode run, `gqa_attention_decode_kernel`
accounts for `192.092 s` of `232.841 s` decode GPU kernel summed time (`82.50%`). Its per-token cost
grows sharply with the current KV position:

| Decode step range | Kernel ms/token | Attention ms/token | Lowbit GEMV ms/token |
|---|---:|---:|---:|
| 1-128 | 17.933 | 3.827 | 12.560 |
| 129-512 | 29.633 | 15.472 | 12.611 |
| 513-1024 | 50.761 | 36.476 | 12.730 |
| 1025-2048 | 86.340 | 72.100 | 12.688 |
| 2049-2862 | 128.703 | 114.441 | 12.713 |

Lowbit GEMV remains nearly flat at about `12.6-12.7 ms/token`. Attention grows with sequence length
and dominates the final half of the run.

## Workload

Command under profiling:

```bash
nsys profile --force-overwrite=true \
  -o profiles/nsys/fem_m4096_full \
  --trace=cuda,nvtx \
  --sample=none --cpuctxsw=none \
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus \
    --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
    --prompt "怎么用fem方法求解heat equation。" \
    --max-context 8192 \
    --max-new 4096 \
  > profiles/nsys/fem_m4096_full.stdout.txt \
  2> profiles/nsys/fem_m4096_full.stderr.txt
```

Host-visible timing from `profiles/nsys/fem_m4096_full.stderr.txt`:

| Phase | Time |
|---|---:|
| tokenizer load | 0.419 s |
| engine load total | 13.458 s |
| prompt render/tokenize | 0.000 s |
| generate prefill | 0.259 s |
| generate decode | 251.582 s |
| generate total | 251.842 s |
| generated tokens | 2863 |
| model throughput | 11.37 tok/s |

The run reproduced the reported low-throughput behavior under Nsight Systems. The generated token
count includes the first token produced by prefill.

## Artifacts

Primary local artifacts:

| Artifact | Path |
|---|---|
| Nsight Systems report | `profiles/nsys/fem_m4096_full.nsys-rep` |
| Exported SQLite | `profiles/nsys/fem_m4096_full.sqlite` |
| Generated summary | `profiles/nsys/fem_m4096_full.nsys-summary.md` |
| Program stdout | `profiles/nsys/fem_m4096_full.stdout.txt` |
| Program stderr/timings | `profiles/nsys/fem_m4096_full.stderr.txt` |
| Prefill kernel stats | `profiles/nsys/fem_m4096_prefill_cuda_gpu_kern_sum_start=13911104503_end=14169234773.txt` |
| Decode kernel stats | `profiles/nsys/fem_m4096_decode_cuda_gpu_kern_sum_start=14169369174_end=265751349193.txt` |
| Decode CUDA API stats | `profiles/nsys/fem_m4096_decode_cuda_api_sum_start=14169369174_end=265751349193.txt` |
| Decode GPU memops stats | `profiles/nsys/fem_m4096_decode_cuda_gpu_mem_time_sum_start=14169369174_end=265751349193.txt` |

The `.nsys-rep` is `279 MiB`; the exported SQLite is `770 MiB`.

## Phase Boundary Method

The executable does not currently emit NVTX ranges for prefill and decode, so phase attribution is
derived from CUDA runtime events.

The reliable boundary signal is `cudaMemcpy_v3020`:

- `cudaMemcpy_v3020` count: `2864`
- first memcpy: prompt IDs H2D
- remaining `2863` memcpys: generated-token D2H readbacks
- first CUPTI memcpy payload: `80 B`, which corresponds to `20` prompt token IDs

Boundaries used:

| Window | Start ns | End ns | Meaning |
|---|---:|---:|---|
| Prefill kernels | 13911104503 | 14169234773 | after prompt H2D, before first token readback |
| Decode kernels/API | 14169369174 | 265751349193 | after prefill token readback, through final decode token readback |

Therefore:

- prompt tokens: `20`
- generated tokens: `2863`
- decode loop steps: `2862`
- final decode KV position is approximately `20 + 2862 = 2882`

## Top-Level Trace Totals

From `fem_m4096_full.nsys-summary.md`:

| Domain | Events | Sum ms | % capture span |
|---|---:|---:|---:|
| CUDA API | 8926246 | 250738.170 | 94.0% |
| kernels | 4045468 | 233089.820 | 87.4% |
| other event tables | 573415 | 231968.479 | 86.9% |
| memcpy | 828377 | 1755.015 | 0.7% |
| memset | 3055 | 1.067 | 0.0% |

Note: CUDA API duration and kernel duration overlap. `cudaStreamSynchronize` mostly represents host
waiting for GPU work; it must not be added to kernel time as extra wall time.

## Prefill Kernel Breakdown

Prefill is not the bottleneck. The whole prefill kernel window sums to `248.417 ms`.

| Kernel | Launches | Total ms | % prefill kernel |
|---|---:|---:|---:|
| `linear_generic_lowbit_gemm_kernel<Q4Codec>` | 256 | 123.594 | 49.8 |
| `linear_generic_lowbit_gemm_kernel<Q5Codec>` | 256 | 105.362 | 42.4 |
| `linear_generic_dense_gemm_kernel` | 96 | 16.669 | 6.7 |
| `linear_tile_lowbit_gemv_kernel<Q6Codec>` | 1 | 0.600 | 0.2 |
| `gated_delta_rule_recurrent_kernel<128>` | 48 | 0.556 | 0.2 |
| `rmsnorm_d5120_kernel` | 129 | 0.380 | 0.2 |
| `gqa_attention_prefill_kernel` | 16 | 0.300 | 0.1 |

Interpretation: for this short 20-token prompt, prefill is dominated by linear GEMM work. Attention
prefill is only `0.300 ms`.

## Decode Kernel Breakdown

Decode kernel summed time is `232841.403 ms` over `2862` decode loop steps.

| Kernel | Launches | Total ms | % decode kernel | ms/token |
|---|---:|---:|---:|---:|
| `gqa_attention_decode_kernel` | 45792 | 192092.327 | 82.50 | 67.118 |
| `linear_tile_lowbit_gemv_kernel<Q5Codec>` | 732672 | 17947.870 | 7.71 | 6.271 |
| `linear_tuned_q4_gemv_kernel` | 732672 | 16591.647 | 7.13 | 5.797 |
| `linear_tile_lowbit_gemv_kernel<Q6Codec>` | 2862 | 1769.494 | 0.76 | 0.618 |
| `rmsnorm_d5120_kernel` | 369198 | 1111.116 | 0.48 | 0.388 |
| `linear_generic_dense_gemv_kernel` | 274752 | 710.359 | 0.31 | 0.248 |
| `gated_delta_rule_recurrent_kernel<128>` | 137376 | 464.836 | 0.20 | 0.162 |
| `rmsnorm_kernel` | 228960 | 404.309 | 0.17 | 0.141 |
| `residual_add_kernel` | 366336 | 378.036 | 0.16 | 0.132 |
| `l2norm_kernel` | 274752 | 262.448 | 0.11 | 0.092 |
| `silu_and_mul_kernel` | 183168 | 235.366 | 0.10 | 0.082 |
| `causal_conv1d_decode_kernel` | 137376 | 181.856 | 0.08 | 0.064 |
| `rope_kernel` | 45792 | 158.213 | 0.07 | 0.055 |
| `gqa_attention_decode_append_kernel` | 45792 | 49.017 | 0.02 | 0.017 |

Grouped decode kernel cost:

| Bucket | Launches | Total ms | % decode kernel | ms/token |
|---|---:|---:|---:|---:|
| attention decode | 45792 | 192092.327 | 82.50 | 67.118 |
| lowbit GEMV | 1468206 | 36309.012 | 15.59 | 12.687 |
| normalization | 872910 | 1777.874 | 0.76 | 0.621 |
| GDN/MLP/misc | 915840 | 1351.155 | 0.58 | 0.472 |
| dense GEMV | 274752 | 710.359 | 0.31 | 0.248 |
| residual | 366336 | 378.036 | 0.16 | 0.132 |
| rotary | 45792 | 158.213 | 0.07 | 0.055 |
| attention append | 45792 | 49.017 | 0.02 | 0.017 |

## Decode Step Buckets

The following table splits the 2862 decode steps into ranges. `wall_ms` is based on readback
boundaries. Kernel and API times are summed durations inside each range; API wait time overlaps GPU
execution.

| Decode steps | Steps | Wall ms | Kernel ms | Kernel ms/token | Attention ms/token | Lowbit GEMV ms/token | Dense GEMV ms/token | Launch API ms/token | Sync wait ms/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-128 | 128 | 3054.854 | 2295.446 | 17.933 | 3.827 | 12.560 | 0.248 | 8.101 | 12.870 |
| 129-512 | 384 | 13796.178 | 11379.049 | 29.633 | 15.472 | 12.611 | 0.248 | 10.985 | 21.935 |
| 513-1024 | 512 | 29322.604 | 25989.724 | 50.761 | 36.476 | 12.730 | 0.248 | 18.746 | 35.355 |
| 1025-2048 | 1024 | 95239.856 | 88412.635 | 86.340 | 72.100 | 12.688 | 0.249 | 32.216 | 57.610 |
| 2049-2862 | 814 | 110168.489 | 104764.548 | 128.703 | 114.441 | 12.713 | 0.248 | 48.122 | 84.055 |

This is the central result. Lowbit GEMV cost is flat across the full decode. Attention grows with
sequence length and dominates once the generated sequence enters the long-context region.

## Per-Bucket Top Kernels

| Decode steps | Top kernel | Total ms | % bucket kernel | ms/token |
|---|---|---:|---:|---:|
| 1-128 | `linear_tile_lowbit_gemv_kernel<Q5Codec>` | 799.212 | 34.82 | 6.244 |
| 1-128 | `linear_tuned_q4_gemv_kernel` | 731.894 | 31.88 | 5.718 |
| 1-128 | `gqa_attention_decode_kernel` | 489.849 | 21.34 | 3.827 |
| 129-512 | `gqa_attention_decode_kernel` | 5941.103 | 52.21 | 15.472 |
| 129-512 | `linear_tile_lowbit_gemv_kernel<Q5Codec>` | 2398.545 | 21.08 | 6.246 |
| 129-512 | `linear_tuned_q4_gemv_kernel` | 2202.478 | 19.36 | 5.736 |
| 513-1024 | `gqa_attention_decode_kernel` | 18675.632 | 71.86 | 36.476 |
| 513-1024 | `linear_tile_lowbit_gemv_kernel<Q5Codec>` | 3230.248 | 12.43 | 6.309 |
| 513-1024 | `linear_tuned_q4_gemv_kernel` | 2976.238 | 11.45 | 5.813 |
| 1025-2048 | `gqa_attention_decode_kernel` | 73830.877 | 83.51 | 72.100 |
| 1025-2048 | `linear_tile_lowbit_gemv_kernel<Q5Codec>` | 6417.466 | 7.26 | 6.267 |
| 1025-2048 | `linear_tuned_q4_gemv_kernel` | 5939.433 | 6.72 | 5.800 |
| 2049-2862 | `gqa_attention_decode_kernel` | 93154.865 | 88.92 | 114.441 |
| 2049-2862 | `linear_tile_lowbit_gemv_kernel<Q5Codec>` | 5102.398 | 4.87 | 6.268 |
| 2049-2862 | `linear_tuned_q4_gemv_kernel` | 4741.604 | 4.53 | 5.825 |

## Decode CUDA API Breakdown

| API | Calls | Total ms | % CUDA API | Avg us | Max us |
|---|---:|---:|---:|---:|---:|
| `cudaStreamSynchronize_v3020` | 2862 | 155585.656 | 62.46 | 54362.563 | 96592.744 |
| `cudaLaunchKernel_v7000` | 4044006 | 87013.198 | 34.93 | 21.517 | 34868.717 |
| `cudaMemcpy2DAsync_v3020` | 824256 | 5776.925 | 2.32 | 7.009 | 6254.001 |
| `cudaMemcpy_v3020` | 2863 | 359.984 | 0.14 | 125.737 | 232.531 |
| `cuKernelGetName` | 4044006 | 311.172 | 0.12 | 0.077 | 5643.835 |
| `cudaMemsetAsync_v3020` | 2862 | 37.978 | 0.02 | 13.270 | 741.402 |

There are about `1413` kernel launches per decode token. The launch overhead is significant, but it
is not the primary reason for the collapse at long length: launch count per token is essentially
constant, while attention kernel cost per token grows from `3.827 ms` to `114.441 ms`.

## Interpretation

1. Prefill is not the issue for this prompt.
   - The prompt is short: `20` token IDs.
   - Prefill kernel sum is only `248.417 ms`.

2. Lowbit GEMV optimization is working in the sense that GEMV cost is flat with decode length.
   - Lowbit GEMV stays around `12.6-12.7 ms/token`.
   - It dominates only in the earliest decode bucket.

3. Long decode collapses because full-attention decode scans a growing KV prefix.
   - There are `16` full-attention decode launches per token.
   - `gqa_attention_decode_kernel` is `82.50%` of full decode kernel time.
   - In the final bucket, attention is `114.441 ms/token`, about `30x` its early-token cost.

4. Host-side token readback is not the root cause.
   - `cudaMemcpy_v3020` is `359.984 ms` across the whole decode.
   - That is about `0.126 ms/token`.

5. Kernel launch overhead remains a separate optimization target.
   - `cudaLaunchKernel` sums to `87.013 s` under Nsight.
   - This reflects about `4.04M` launches in decode.
   - However, launch overhead is relatively flat per token; it does not explain why late decode is
     much slower than early decode.

## Conclusion

The current long decode bottleneck is `gqa_attention_decode_kernel`, not prefill and not lowbit
GEMV. The kernel's cost grows with generated length, consistent with a per-step full-attention
implementation that scans the accumulated KV cache. By the final decode bucket, attention is almost
`89%` of kernel time and costs `114.441 ms/token`.

The next optimization phase should focus on the decode attention path:

- inspect `gqa_attention_decode_kernel` memory traffic and launch shape across long positions;
- reduce repeated work over the KV prefix where possible;
- consider a more efficient decode attention kernel/layout for long context;
- keep tracking launch overhead, but treat it as secondary to attention growth for this workload.

## Reproduction Notes

Generated split stats:

```bash
nsys stats --force-export=false --force-overwrite=true \
  --filter-time 13911104503/14169234773 \
  -r cuda_gpu_kern_sum -f column \
  -o profiles/nsys/fem_m4096_prefill \
  profiles/nsys/fem_m4096_full.nsys-rep

nsys stats --force-export=false --force-overwrite=true \
  --filter-time 14169369174/265751349193 \
  -r cuda_gpu_kern_sum -f column \
  -o profiles/nsys/fem_m4096_decode \
  profiles/nsys/fem_m4096_full.nsys-rep

nsys stats --force-export=false --force-overwrite=true \
  --filter-time 14169369174/265751349193 \
  -r cuda_api_sum -f column \
  -o profiles/nsys/fem_m4096_decode \
  profiles/nsys/fem_m4096_full.nsys-rep
```

Limitations:

- No NVTX prefill/decode ranges were present, so phase boundaries are inferred from CUDA runtime
  readback events.
- Nsight Systems overhead affects absolute wall time. The kernel composition and per-bucket
  growth pattern are the important evidence.
- Nsys cannot prove occupancy, bandwidth, or stall reasons. Those require a separate Nsight Compute
  run on `gqa_attention_decode_kernel`.
