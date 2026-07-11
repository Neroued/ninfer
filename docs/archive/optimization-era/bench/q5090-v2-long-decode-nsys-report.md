# Q5090 V2 Long Decode Nsys Report

> Date: 2026-06-30
> Scope: Qwen3.6-27B q5090 v2 runtime long-decode profile on one RTX 5090.
> Build: current local `master` with `qus.prefill` and `qus.decode` NVTX phase ranges.
> Nsight Systems: `2025.5.2.266-255236693005v0`

## Executive Summary

The real `qus` long-decode workload was re-profiled after adding phase-level NVTX ranges. Phase
splitting now uses the explicit NVTX ranges:

- `qus.prefill`
- `qus.decode`

The requested run used `--max-new 4096`, but generation stopped at `2858` tokens after hitting a
stop token. The generated-token count includes the first token produced by prefill, so the decode
loop executed `2857` steps.

Measured host timings:

| Phase | Time |
|---|---:|
| prefill | `1.189 s` |
| decode | `60.610 s` |
| total generation | `61.799 s` |
| generated tokens | `2858` |
| throughput | `46.25 tok/s` |

Decode is dominated by rowsplit low-bit GEMV. Inside the `qus.decode` NVTX range, rowsplit GEMV
kernels sum to `38.316 s`, or `87.34%` of decode GPU kernel time and `63.22%` of the decode wall
range. The two largest buckets are MLP `gate/up` Q4 and MLP `down` Q5, which together account for
`22.971 s` of kernel time.

GQA decode attention is not the dominant cost in this trace. The GQA decode attention kernels sum
to `1.103 s`, or `2.51%` of decode GPU kernel time and `1.82%` of the decode range.

## Workload

Command under profiling:

```bash
nsys profile --force-overwrite=true --stats=false \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  -o profiles/nsys-long-decode/heat_fem_max4096_nvtx \
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
    --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
    --prompt "怎么用fem方法求解heat equation。" \
    --max-context 8192 \
    --max-new 4096 \
  > profiles/nsys-long-decode/heat_fem_max4096_nvtx.stdout.txt \
  2> profiles/nsys-long-decode/heat_fem_max4096_nvtx.stderr.txt
```

Host-visible timing from `profiles/nsys-long-decode/heat_fem_max4096_nvtx.stderr.txt`:

| Phase | Time |
|---|---:|
| tokenizer load | `0.362 s` |
| engine load total | `12.376 s` |
| prompt render/tokenize | `0.000 s` |
| generate prefill | `1.189 s` |
| generate decode | `60.610 s` |
| generate total | `61.799 s` |
| generated tokens | `2858` |
| model elapsed | `61.799 s` |
| throughput | `46.25 tok/s` |

Engine load is excluded from the decode hotspot analysis.

## Artifacts

Primary local artifacts:

| Artifact | Path | Size |
|---|---|---:|
| Nsight Systems report | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.nsys-rep` | `285 MiB` |
| Exported SQLite | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.sqlite` | `805 MiB` |
| Generated summary | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.nsys-summary.md` | local artifact |
| NVTX summary | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.nvtx_sum_nvtx_sum.csv` | `280 B` |
| CUDA kernel summary | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv` | `6.7 KiB` |
| CUDA API summary | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.cuda_api_sum_cuda_api_sum.csv` | `1.2 KiB` |
| CUDA memop summary | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.cuda_gpu_mem_time_sum_cuda_gpu_mem_time_sum.csv` | `389 B` |
| Decode analysis | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.decode_analysis.txt` | local artifact |
| Program stdout | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.stdout.txt` | `13 KiB` |
| Program stderr/timings | `profiles/nsys-long-decode/heat_fem_max4096_nvtx.stderr.txt` | `3.3 KiB` |

The `.nsys-rep` and SQLite files are profiling artifacts and are not intended to be tracked in git.

## NVTX Phase Boundaries

`nsys stats --report nvtx_sum` reports one range for each phase:

| NVTX range | Instances | Duration |
|---|---:|---:|
| `:qus.prefill` | 1 | `1.188581860 s` |
| `:qus.decode` | 1 | `60.609885123 s` |

SQLite timestamps used for the phase split:

| Range | Start ns | End ns | Duration |
|---|---:|---:|---:|
| `qus.prefill` | `12757845430` | `13946427290` | `1.188581860 s` |
| `qus.decode` | `13946474750` | `74556359873` | `60.609885123 s` |

Decode count checks:

| Signal | Count |
|---|---:|
| generated tokens reported by CLI | `2858` |
| decode loop steps | `2857` |
| decode `advance_pos_kernel` launches | `2857` |
| decode argmax launches | `2857` |
| decode `lm_head` launches | `2857` |

## Top-Level Phase Totals

Durations below are summed event durations inside each NVTX range. CUDA API duration overlaps GPU
execution and should not be added to kernel duration as extra wall time.

| Phase | Range ms | Kernel events | Kernel ms | Memcpy events | Memcpy ms | Runtime API events | Runtime API ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill | `1188.582` | `1462` | `1175.706` | `289` | `0.240` | `3337` | `1187.209` |
| decode | `60609.885` | `4036941` | `43869.596` | `825673` | `673.044` | `8905282` | `58185.446` |

Decode kernel-active share of the NVTX range is `72.38%`.

## Decode Top Kernels

Top kernels inside `qus.decode`:

| Rank | Kernel | Launches | Total ms | % kernel time | % decode range | Avg us |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `linear_rowsplit_gemv_mlp_gate_up_q4_kernel` | 365696 | `12291.060` | `28.02` | `20.28` | `33.610` |
| 2 | `linear_rowsplit_gemv_mlp_down_q5_kernel` | 182848 | `10679.874` | `24.34` | `17.62` | `58.408` |
| 3 | `linear_rowsplit_gemv_proj_6144_q5_kernel` | 319984 | `6832.445` | `15.57` | `11.27` | `21.352` |
| 4 | `linear_rowsplit_gemv_out_6144_q5_kernel` | 182848 | `3940.277` | `8.98` | `6.50` | `21.549` |
| 5 | `linear_rowsplit_gemv_lm_head_q6_kernel` | 2857 | `2110.939` | `4.81` | `3.48` | `738.866` |
| 6 | `linear_rowsplit_gemv_gdn_qk_2048_q4_kernel` | 274272 | `1439.624` | `3.28` | `2.38` | `5.249` |
| 7 | `rmsnorm_d5120_kernel` | 368553 | `1108.741` | `2.53` | `1.83` | `3.008` |
| 8 | `gqa_attention_decode_partial_kernel<16,6,true>` | 45712 | `785.941` | `1.79` | `1.30` | `17.193` |
| 9 | `linear_generic_dense_gemv_kernel` | 274272 | `725.225` | `1.65` | `1.20` | `2.644` |
| 10 | `linear_rowsplit_gemv_proj_6144_q4_kernel` | 45712 | `582.011` | `1.33` | `0.96` | `12.732` |
| 11 | `gated_delta_rule_recurrent_kernel<128>` | 137136 | `461.393` | `1.05` | `0.76` | `3.364` |
| 12 | `rmsnorm_kernel` | 228560 | `414.144` | `0.94` | `0.68` | `1.812` |
| 13 | `residual_add_kernel` | 365696 | `379.683` | `0.87` | `0.63` | `1.038` |
| 14 | `gqa_attention_decode_reduce_output_kernel<32>` | 45712 | `316.915` | `0.72` | `0.52` | `6.933` |
| 15 | `linear_rowsplit_gemv_attn_kv_1024_q5_kernel` | 45712 | `283.693` | `0.65` | `0.47` | `6.206` |

Per decode step, the main contributors are:

| Kernel bucket | ms/step |
|---|---:|
| MLP `gate/up` Q4 rowsplit GEMV | `4.302` |
| MLP `down` Q5 rowsplit GEMV | `3.738` |
| projection rowsplit GEMV | `2.595` |
| attention out Q5 rowsplit GEMV | `1.379` |
| lm head Q6 rowsplit GEMV | `0.739` |
| normalization | `0.533` |
| GDN Q/K rowsplit GEMV | `0.504` |
| GQA decode attention | `0.386` |
| dense GEMV | `0.254` |

## Decode Grouped Breakdown

| Category | Launches | Total ms | % kernel time | % decode range |
|---|---:|---:|---:|---:|
| rowsplit GEMV: MLP gate/up | 365696 | `12291.060` | `28.02` | `20.28` |
| rowsplit GEMV: MLP down | 182848 | `10679.874` | `24.34` | `17.62` |
| rowsplit GEMV: full attention/GDN proj | 365696 | `7414.456` | `16.90` | `12.23` |
| rowsplit GEMV: attention out | 182848 | `3940.277` | `8.98` | `6.50` |
| rowsplit GEMV: lm head | 2857 | `2110.939` | `4.81` | `3.48` |
| normalization | 597113 | `1522.885` | `3.47` | `2.51` |
| rowsplit GEMV: GDN q/k | 274272 | `1439.624` | `3.28` | `2.38` |
| GDN recurrence/support | 822816 | `1146.569` | `2.61` | `1.89` |
| GQA decode attention | 91424 | `1102.856` | `2.51` | `1.82` |
| elementwise/rope/conv | 777104 | `1039.997` | `2.37` | `1.72` |
| dense GEMV: conv/GDN small | 274272 | `725.225` | `1.65` | `1.20` |
| rowsplit GEMV: attention k/v | 91424 | `439.378` | `1.00` | `0.72` |
| sampling argmax | 2857 | `9.460` | `0.02` | `0.02` |
| embedding | 2857 | `3.820` | `0.01` | `0.01` |
| position bookkeeping | 2857 | `3.176` | `0.01` | `0.01` |

Rowsplit GEMV total:

| Metric | Value |
|---|---:|
| summed rowsplit GEMV time | `38315.608 ms` |
| share of decode kernel time | `87.34%` |
| share of decode range | `63.22%` |

## Prefill Contrast

Prefill is dominated by generic low-bit GEMM and has a different shape from decode.

| Rank | Kernel | Launches | Total ms | % prefill kernel |
|---:|---|---:|---:|---:|
| 1 | `linear_generic_lowbit_gemm_kernel<Q4Codec>` | 256 | `1003.191` | `85.33` |
| 2 | `linear_generic_lowbit_gemm_kernel<Q5Codec>` | 256 | `154.336` | `13.13` |
| 3 | `linear_generic_dense_gemm_kernel` | 96 | `15.244` | `1.30` |
| 4 | `linear_rowsplit_gemv_lm_head_q6_kernel` | 1 | `0.751` | `0.06` |
| 5 | `gated_delta_rule_recurrent_kernel<128>` | 48 | `0.555` | `0.05` |
| 6 | `rmsnorm_d5120_kernel` | 129 | `0.381` | `0.03` |
| 7 | `gqa_attention_prefill_kernel` | 16 | `0.286` | `0.02` |

Grouped prefill:

| Category | Total ms | % prefill kernel |
|---|---:|---:|
| generic lowbit GEMM | `1157.527` | `98.45` |
| generic dense GEMM | `15.244` | `1.30` |
| all other kernels | `2.935` | `0.25` |

## Decode CUDA API And Memcpy

CUDA API duration overlaps GPU execution. It explains host-side wait and launch overhead but should
not be summed with kernel time as wall time.

Top CUDA runtime calls inside `qus.decode`:

| API | Calls | Total ms | % decode range | Avg us | Max ms |
|---|---:|---:|---:|---:|---:|
| `cudaStreamSynchronize_v3020` | 2857 | `29934.168` | `49.39` | `10477.483` | `14.017` |
| `cudaLaunchKernel_v7000` | 4036941 | `22414.291` | `36.98` | `5.552` | `22.736` |
| `cudaMemcpy2DAsync_v3020` | 822816 | `5140.580` | `8.48` | `6.248` | `8.458` |
| `cuKernelGetName` | 4036941 | `397.617` | `0.66` | `0.098` | `6.455` |
| `cudaMemcpy_v3020` | 2857 | `265.842` | `0.44` | `93.049` | `0.281` |
| `cudaMemsetAsync_v3020` | 2857 | `32.937` | `0.05` | `11.529` | `2.064` |

Decode memcpy activity:

| Kind | Meaning | Calls | Total ms | Bytes |
|---:|---|---:|---:|---:|
| 8 | Device-to-device | 822816 | `670.488` | `5617090560` |
| 2 | Device-to-host | 2857 | `2.556` | `11428` |

There is one host synchronization and generated-token readback per decode step. `cudaStreamSynchronize`
mostly represents host wait for queued GPU work. The largest kernel-side hotspot remains rowsplit
GEMV.

## Conclusion

With explicit NVTX phase ranges, the decode-stage evidence is now direct rather than inferred. The
current q5090 v2 long decode cost center is rowsplit GEMV:

1. `linear_rowsplit_gemv_mlp_gate_up_q4_kernel`
2. `linear_rowsplit_gemv_mlp_down_q5_kernel`
3. `linear_rowsplit_gemv_proj_6144_q5_kernel`
4. `linear_rowsplit_gemv_out_6144_q5_kernel`
5. `linear_rowsplit_gemv_lm_head_q6_kernel`

Prefill remains a separate generic-GEMM dominated phase. Future profiles should use the
`qus.prefill` and `qus.decode` NVTX ranges as phase boundaries.
