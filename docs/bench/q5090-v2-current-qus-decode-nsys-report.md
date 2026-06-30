# Q5090 V2 Current `qus` Decode Nsys Report

> Date: 2026-06-30
> Scope: current local `qus` actual text decode profile for Qwen3.6-27B q5090 v2 on one RTX 5090.
> Build: current local `master` at `b352d0f`; `cmake --build build -j --target qus`
> reported `ninja: no work to do`.
> Nsight Systems: `2025.5.2.266-255236693005v0`.

## Executive Summary

The current `qus` long-decode workload was profiled with the same FEM prompt used by the previous
long-decode reports, but with `--max-new 2916` instead of `4096`. This intentionally matches the
previous tuned-numeric trace's actual generated-token count:

- previous tuned numeric: requested `--max-new 4096`, stopped at `2916` generated tokens;
- current trace: requested `--max-new 2916`, produced exactly `2916` generated tokens.

The generated-token count includes the first token produced by prefill, so the measured decode loop
contains `2915` `decode_step()` calls in both traces.

Current host timings:

| Phase | Time |
| --- | ---: |
| prefill | `1.325 s` |
| decode | `57.409 s` |
| total generation | `58.736 s` |
| generated tokens | `2916` |
| generation throughput | `49.65 tok/s` |
| decode loop throughput | `50.78 step/s` |

Compared with `docs/bench/q5090-v2-tuned-numeric-long-decode-nsys-report.md`, current same-length
decode is slower in wall time even though summed GPU kernel time is lower:

| Metric | Previous tuned numeric | Current | Delta |
| --- | ---: | ---: | ---: |
| decode wall time | `54.750 s` | `57.409 s` | `+2.659 s` (`+4.86%`) |
| decode loop throughput | `53.24 step/s` | `50.78 step/s` | `-4.63%` |
| decode kernel summed time | `44.139 s` | `41.235 s` | `-2.904 s` (`-6.58%`) |
| decode kernel launches | `3,885,695` | `4,165,535` | `+279,840` (`+7.20%`) |
| kernel launches per decode step | `1333` | `1429` | `+96/step` |

The most visible regression is launch surface, not GPU math time. Current decode adds
`linear_rowsplit_gemv_proj_6144_q5_reduce_kernel` at `96` launches per token (`279,840` total). That
new reduce kernel consumes only `0.240 s` of GPU time, but the extra launches align with higher
CUDA runtime overhead:

| Runtime API | Previous tuned numeric | Current | Delta |
| --- | ---: | ---: | ---: |
| `cudaLaunchKernel` summed duration | `18.739 s` | `20.134 s` | `+1.395 s` |
| `cudaStreamSynchronize` summed duration | `33.278 s` | `34.446 s` | `+1.168 s` |

CUDA API durations overlap GPU execution and should not be added to kernel duration as extra wall
time. They do show that the current trace spends more host-side time launching and waiting while the
GPU kernel sum is lower.

## Workload

Command under profiling:

```bash
nsys profile --force-overwrite=true --stats=false \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  -o profiles/nsys-long-decode/heat_fem_max2916_current_nvtx \
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
    --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
    --prompt "怎么用fem方法求解heat equation。" \
    --max-context 8192 \
    --max-new 2916 \
  > profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.stdout.txt \
  2> profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.stderr.txt
```

Host-visible timing from `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.stderr.txt`:

| Item | Value |
| --- | ---: |
| tokenizer load | `0.395 s` |
| engine load total | `19.812 s` |
| prompt render/tokenize | `0.001 s` |
| generate prefill | `1.325 s` |
| generate decode | `57.409 s` |
| generate total | `58.736 s` |
| generated tokens | `2916` |
| model elapsed | `58.734 s` |
| throughput | `49.65 tok/s` |

Engine load is excluded from decode hotspot analysis.

## Artifacts

| Artifact | Path | Size |
| --- | --- | ---: |
| Nsight Systems report | `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.nsys-rep` | `261 MiB` |
| SQLite export | `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.sqlite` | `744 MiB` |
| Generated summary | `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.nsys-summary.md` | `8 KiB` |
| Decode comparison analysis | `profiles/nsys-long-decode/heat_fem_max2916_current_vs_tuned.decode_analysis.txt` | `12 KiB` |
| NVTX summary | `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx_nvtx_sum.csv` | `4 KiB` |
| CUDA kernel summary | `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx_cuda_gpu_kern_sum.csv` | `8 KiB` |
| CUDA API summary | `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx_cuda_api_sum.csv` | `4 KiB` |
| CUDA memop summary | `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx_cuda_gpu_mem_time_sum.csv` | `4 KiB` |
| Program stdout | `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.stdout.txt` | `16 KiB` |
| Program stderr/timings | `profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.stderr.txt` | `4 KiB` |

Export commands:

```bash
python3 /home/neroued/.codex/skills/nsys-inference-analysis/scripts/nsys_inference_summary.py \
  profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.nsys-rep \
  --out profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.nsys-summary.md

nsys stats --force-export=true --force-overwrite=true \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format csv \
  --output profiles/nsys-long-decode/heat_fem_max2916_current_nvtx \
  profiles/nsys-long-decode/heat_fem_max2916_current_nvtx.nsys-rep
```

## NVTX Phase Boundaries

| NVTX range | Start ns | End ns | Duration |
| --- | ---: | ---: | ---: |
| `qus.prefill` | `20249018001` | `21573623811` | `1.324605810 s` |
| `qus.decode` | `21573674861` | `78982631803` | `57.408956942 s` |

Decode count checks:

| Signal | Count |
| --- | ---: |
| generated tokens reported by CLI | `2916` |
| decode loop steps | `2915` |
| decode D2H token readbacks | `2915` |
| decode `lm_head` launches | `2915` |
| decode argmax launches | `2915` |

## Top-Level Phase Totals

Durations below are summed event durations inside each NVTX range. CUDA runtime duration overlaps
GPU execution.

| Phase | Range ms | Kernel events | Kernel ms | Memcpy events | Memcpy ms | Memset events | Memset ms | Runtime API events | Runtime API ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill | `1324.606` | `1462` | `1298.735` | `289` | `0.245` | `97` | `0.109` | `3337` | `1323.046` |
| decode | `57408.957` | `4165535` | `41235.438` | `2915` | `1.366` | `2915` | `0.923` | `8339829` | `55184.492` |

Current decode kernel-active share of the NVTX range is `71.83%`.

## Decode Top Kernels

Top kernels inside `qus.decode`:

| Rank | Kernel | Launches | Per step | Total ms | % kernel time | % decode range | Avg us |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel` | `186560` | `64` | `11791.824` | `28.60` | `20.54` | `63.207` |
| 2 | `linear_rowsplit_gemv_mlp_down_q5_kernel` | `186560` | `64` | `9678.152` | `23.47` | `16.86` | `51.877` |
| 3 | `linear_rowsplit_gemv_proj_6144_q5_kernel` | `279840` | `96` | `5014.913` | `12.16` | `8.74` | `17.921` |
| 4 | `linear_rowsplit_gemv_out_6144_q5_kernel` | `186560` | `64` | `3404.351` | `8.26` | `5.93` | `18.248` |
| 5 | `linear_rowsplit_gemv_lm_head_q6_kernel` | `2915` | `1` | `2210.449` | `5.36` | `3.85` | `758.301` |
| 6 | `linear_rowsplit_gemv_gdn_in_qk_4096_q4_kernel` | `139920` | `48` | `1265.247` | `3.07` | `2.20` | `9.043` |
| 7 | `rmsnorm_d5120_kernel` | `376035` | `129` | `1124.175` | `2.73` | `1.96` | `2.990` |
| 8 | `linear_rowsplit_gemv_attn_in_7168_q5_kernel` | `46640` | `16` | `1093.026` | `2.65` | `1.90` | `23.435` |
| 9 | `gqa_attention_decode_partial_kernel<16,6,true>` | `46640` | `16` | `818.311` | `1.98` | `1.43` | `17.545` |
| 10 | `linear_generic_dense_gemv_kernel` | `279840` | `96` | `740.080` | `1.79` | `1.29` | `2.645` |
| 11 | `linear_rowsplit_gemv_attn_in_7168_q4_kernel` | `46640` | `16` | `691.583` | `1.68` | `1.20` | `14.828` |
| 12 | `gated_delta_rule_recurrent_kernel<128>` | `139920` | `48` | `466.562` | `1.13` | `0.81` | `3.334` |
| 13 | `rmsnorm_kernel` | `233200` | `80` | `420.521` | `1.02` | `0.73` | `1.803` |
| 14 | `residual_add_kernel` | `373120` | `128` | `388.127` | `0.94` | `0.68` | `1.040` |
| 15 | `gqa_attention_decode_reduce_output_kernel<32>` | `46640` | `16` | `331.767` | `0.80` | `0.58` | `7.113` |
| 16 | `l2norm_kernel` | `279840` | `96` | `274.076` | `0.66` | `0.48` | `0.979` |
| 17 | `linear_rowsplit_gemv_proj_6144_q5_reduce_kernel` | `279840` | `96` | `240.401` | `0.58` | `0.42` | `0.859` |
| 18 | `silu_and_mul_kernel` | `186560` | `64` | `240.191` | `0.58` | `0.42` | `1.287` |
| 19 | `causal_conv1d_decode_kernel` | `139920` | `48` | `188.542` | `0.46` | `0.33` | `1.348` |
| 20 | `linear_rowsplit_gemv_mlp_down_q5_reduce_kernel` | `186560` | `64` | `173.038` | `0.42` | `0.30` | `0.928` |

## Decode Grouped Breakdown

| Category | Launches | Per step | Total ms | % kernel time | % decode range |
| --- | ---: | ---: | ---: | ---: | ---: |
| rowsplit GEMV: `mlp_gate_up` | `186560` | `64` | `11791.824` | `28.60` | `20.54` |
| rowsplit GEMV: `mlp_down` | `186560` | `64` | `9678.152` | `23.47` | `16.86` |
| rowsplit GEMV: `proj_6144` | `279840` | `96` | `5014.913` | `12.16` | `8.74` |
| rowsplit GEMV: `out_6144` | `186560` | `64` | `3404.351` | `8.26` | `5.93` |
| rowsplit GEMV: `lm_head` | `2915` | `1` | `2210.449` | `5.36` | `3.85` |
| normalization | `889075` | `305` | `1818.771` | `4.41` | `3.17` |
| rowsplit GEMV: `attn_in` | `93280` | `32` | `1784.609` | `4.33` | `3.11` |
| rowsplit GEMV: `gdn_in_qk` | `139920` | `48` | `1265.247` | `3.07` | `2.20` |
| GQA decode attention | `93280` | `32` | `1150.078` | `2.79` | `2.00` |
| elementwise/rope/conv | `792880` | `272` | `1049.310` | `2.54` | `1.83` |
| GDN recurrence/support | `559680` | `192` | `897.291` | `2.18` | `1.56` |
| dense GEMV: conv/GDN small | `279840` | `96` | `740.080` | `1.79` | `1.29` |
| rowsplit reduce: `proj_6144` | `279840` | `96` | `240.401` | `0.58` | `0.42` |
| rowsplit reduce: `mlp_down` | `186560` | `64` | `173.038` | `0.42` | `0.30` |

Rowsplit totals:

| Rowsplit subset | Launches | Per step | Total ms | % kernel time | % decode range |
| --- | ---: | ---: | ---: | ---: | ---: |
| base rowsplit GEMV, excluding reduce kernels | `1075635` | `369` | `35149.545` | `85.24` | `61.23` |
| rowsplit reduce kernels | `466400` | `160` | `413.439` | `1.00` | `0.72` |
| all rowsplit-related kernels | `1542035` | `529` | `35562.984` | `86.25` | `61.95` |

## Decode Step Buckets

The table below uses the `2915` decode D2H token readbacks as step boundaries. Kernel and API times
are summed durations inside each step bucket. API duration overlaps GPU work.

| Decode steps | Steps | Wall ms/step | Kernel ms/step | Rowsplit base ms/step | Rowsplit reduce ms/step | GQA ms/step | Dense GEMV ms/step | Launch API ms/step | Sync wait ms/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-128 | `128` | `19.929` | `13.977` | `11.986` | `0.141` | `0.306` | `0.252` | `7.210` | `11.632` |
| 129-512 | `384` | `19.695` | `13.975` | `11.968` | `0.142` | `0.318` | `0.254` | `7.059` | `11.573` |
| 513-1024 | `512` | `19.544` | `14.106` | `12.065` | `0.142` | `0.350` | `0.253` | `6.811` | `11.791` |
| 1025-2048 | `1024` | `19.654` | `14.162` | `12.074` | `0.142` | `0.397` | `0.254` | `6.881` | `11.820` |
| 2049-2915 | `867` | `19.795` | `14.251` | `12.086` | `0.142` | `0.465` | `0.255` | `6.883` | `11.963` |

The decode curve is mostly flat. GQA attention still grows with position, from `0.306 ms/step` to
`0.465 ms/step`, but it remains small relative to rowsplit GEMV. This is not the old long-context
attention-collapse profile.

## Decode CUDA API And Memcpy

Top CUDA Runtime/API calls inside `qus.decode`:

| API | Calls | Total ms | % decode range | Avg us | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize` | `2915` | `34445.524` | `60.00` | `11816.646` | `14.825` |
| `cudaLaunchKernel` | `4165535` | `20134.222` | `35.07` | `4.834` | `10.116` |
| `cudaMemcpy` | `2915` | `287.269` | `0.50` | `98.549` | `0.868` |
| `cuKernelGetName` | `4165535` | `285.993` | `0.50` | `0.069` | `0.213` |
| `cudaMemsetAsync` | `2915` | `31.473` | `0.05` | `10.797` | `0.068` |

Decode GPU memcpy activity:

| Kind | Meaning | Calls | Total ms | Bytes |
| ---: | --- | ---: | ---: | ---: |
| 2 | Device-to-host | `2915` | `1.366` | `11660` |

There are no steady-state decode H2D or D2D memcopies in this trace. The D2D copies present in the
older pre-fusion v2 report remain eliminated.

## Same-Length Comparison With Previous Tuned Numeric Trace

Baseline: `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.sqlite`, documented in
`docs/bench/q5090-v2-tuned-numeric-long-decode-nsys-report.md`.

| Metric | Previous tuned numeric | Current | Delta |
| --- | ---: | ---: | ---: |
| `qus.decode` NVTX | `54.749973 s` | `57.408957 s` | `+2.658984 s` |
| decode loop steps | `2915` | `2915` | same |
| decode loop throughput | `53.242 step/s` | `50.776 step/s` | `-4.63%` |
| kernel summed time | `44.139241 s` | `41.235438 s` | `-6.58%` |
| kernel events | `3,885,695` | `4,165,535` | `+279,840` |
| kernel events per step | `1333` | `1429` | `+96` |
| base rowsplit GEMV time | `38.116820 s` | `35.149545 s` | `-2.967275 s` |
| all rowsplit-related time | `38.302750 s` | `35.562984 s` | `-2.739766 s` |
| rowsplit-related launches per step | `433` | `529` | `+96` |
| `cudaLaunchKernel` summed duration | `18.739153 s` | `20.134222 s` | `+1.395069 s` |
| `cudaStreamSynchronize` summed duration | `33.277513 s` | `34.445524 s` | `+1.168011 s` |

Per-category kernel deltas:

| Category | Previous ms | Current ms | Delta |
| --- | ---: | ---: | ---: |
| rowsplit GEMV: `mlp_gate_up` | `12241.160` | `11791.824` | `-449.336` |
| rowsplit GEMV: `mlp_down` | `10045.379` | `9678.152` | `-367.228` |
| rowsplit GEMV: `proj_6144` | `6219.542` | `5014.913` | `-1204.629` |
| rowsplit reduce: `proj_6144` | `0.000` | `240.401` | `+240.401` |
| rowsplit GEMV: `out_6144` | `4091.690` | `3404.351` | `-687.339` |
| rowsplit GEMV: `lm_head` | `2392.203` | `2210.449` | `-181.754` |
| GQA decode attention | `1166.885` | `1150.078` | `-16.807` |
| normalization | `1876.117` | `1818.771` | `-57.346` |

The current kernel bodies are generally faster, especially `proj_6144` and `out_6144`. The wall-time
regression comes from the added launch surface and associated host-side wait, not from a single
large GPU kernel becoming slower.

## Bottleneck Readout

1. Current actual decode is still rowsplit-GEMV dominated. All rowsplit-related kernels sum to
   `35.563 s`, `86.25%` of decode kernel time and `61.95%` of the decode NVTX range.
2. The top two kernels remain MLP rowsplit GEMV:
   `linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel` and
   `linear_rowsplit_gemv_mlp_down_q5_kernel`, together `21.470 s` of GPU time.
3. GQA decode attention is not the main bottleneck in the current trace: `1.150 s`, or `2.00%` of
   the decode range.
4. The new `linear_rowsplit_gemv_proj_6144_q5_reduce_kernel` is individually small in GPU time but
   adds `96` launches per decode token. This is the clearest difference from the tuned numeric
   baseline and explains why lower kernel time does not translate into lower wall time.
5. Next profiling should use `ncu` on:
   - `linear_rowsplit_gemv_proj_6144_q5_kernel`
   - `linear_rowsplit_gemv_proj_6144_q5_reduce_kernel`
   - `linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel`
   - `linear_rowsplit_gemv_mlp_down_q5_kernel`

Nsys identifies where time goes. It does not prove occupancy, memory bandwidth, warp stalls, or
cache behavior; those require Nsight Compute.
