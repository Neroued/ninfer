# q5090 v2 tuned numeric long-decode nsys report

> Scope: Qwen3.6-27B q5090 v2 runtime long-decode profile after decode GEMV fusion
> and numeric-only Phase 2 tuning. Build commit: `ebbbfd7`
> (`bench(q5090): update numeric tuning ledger`).

This report summarizes the current local `master` trace captured as
`profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.nsys-rep`.
The analysis uses the `qus.prefill` / `qus.decode` NVTX ranges and treats
`qus.decode` as the steady-state decode window. Full-capture memcpy/API tables
include model load and weight upload, so decode claims below are computed from
the decode NVTX window unless explicitly stated otherwise.

## Workload

```bash
nsys profile --force-overwrite=true --stats=false \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx \
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
    --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
    --prompt "怎么用fem方法求解heat equation。" \
    --max-context 8192 --max-new 4096
```

Program stderr reported:

| item | value |
| --- | ---: |
| tokenizer load | `0.453 s` |
| weight file read | `1.536 s` for `15.93 GiB` |
| weight payload upload | `1.156 s` for `15.25 GiB` |
| engine load total | `12.594 s` |
| prefill | `1.192 s` |
| decode | `54.750 s` |
| generated tokens | `2916` |
| model elapsed | `55.942 s` |
| throughput | `52.13 tok/s` |

The generated-token count includes the prefill-produced token; the decode loop
step count inferred from `lm_head` launches inside `qus.decode` is `2915`.

## Artifacts

| artifact | path |
| --- | --- |
| Nsight Systems report | `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.nsys-rep` |
| SQLite export | `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.sqlite` |
| Generated summary | `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.nsys-summary.md` |
| NVTX summary | `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx_nvtx_sum.csv` |
| CUDA kernel summary | `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx_cuda_gpu_kern_sum.csv` |
| CUDA API summary | `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx_cuda_api_sum.csv` |
| CUDA memop summary | `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx_cuda_gpu_mem_time_sum.csv` |
| Program stdout | `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.stdout.txt` |
| Program stderr/timings | `profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.stderr.txt` |

Export commands:

```bash
nsys stats --force-export=true --force-overwrite=true \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format csv \
  --output profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx \
  profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.nsys-rep

python3 /home/neroued/.codex/skills/nsys-inference-analysis/scripts/nsys_inference_summary.py \
  profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.nsys-rep \
  --out profiles/nsys-long-decode/heat_fem_max4096_tuned_numeric_nvtx.nsys-summary.md
```

## Top-level timing

| metric | value |
| --- | ---: |
| capture event span | `69.188 s` |
| `qus.prefill` NVTX | `1.192154 s` |
| `qus.decode` NVTX | `54.749973 s` |
| decode CUDA kernel summed time | `44.139241 s` |
| decode CUDA API summed time | `52.621114 s` |
| decode kernel launches | `3,885,695` |
| decode kernels per step | `1333.000` |
| decode D2H memcpy | `2915` copies, `11,660 B`, `2.406547 ms` |
| decode H2D memcpy | `0` |
| decode D2D memcpy | `0` |
| decode memset | `2915`, `0.931911 ms` |

Whole-capture memops still include cold-start work:

| whole-capture operation | count | total time |
| --- | ---: | ---: |
| Host-to-Device memcpy | `826` | `1098.512 ms` |
| Device-to-Host memcpy | `2916` | `2.408 ms` |
| Device-to-Device memcpy | `288` | `0.242 ms` |
| memset | `3108` | `1.151 ms` |

The important steady-state decode result is that H2D and D2D memcopies are absent
inside `qus.decode`; only the 4-byte generated-token readback remains once per
decode step.

## Comparison with previous v2 long-decode report

Baseline values are from `docs/bench/q5090-v2-long-decode-nsys-report.md`.

| metric | previous v2 report | current tuned numeric | delta |
| --- | ---: | ---: | ---: |
| decode wall time | `60.610 s` | `54.750 s` | `-5.860 s` (`-9.67%`) |
| throughput | `46.25 tok/s` | `52.13 tok/s` | `+5.88 tok/s` (`+12.7%`) |
| decode D2D memcpys | `822,816` | `0` | eliminated in decode |
| rowsplit launch structure | about `1413/step` | `369/step` base rowsplit, `433/step` incl. split-K reduce | large launch-count drop |
| rowsplit-related kernel sum | `38.316 s` | `38.303 s` | roughly flat |
| rowsplit share of decode wall | `63.22%` | `69.96%` | higher share after other overheads dropped |

The rowsplit kernel sum is nearly unchanged, but the decode wall time improves
because the fusion work removed decode D2D copies and reduced the rowsplit launch
surface substantially. Phase 2 then shifted individual kernel costs; the remaining
top bottleneck is still rowsplit GEMV.

## Decode kernel hotspots

Top kernels inside `qus.decode`:

| rank | kernel | launches | per step | total ms | avg us | % decode | % kernel sum |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel` | `186,560` | `64.000` | `12,241.160` | `65.615` | `22.36%` | `27.73%` |
| 2 | `linear_rowsplit_gemv_mlp_down_q5_kernel` | `186,560` | `64.000` | `10,045.379` | `53.845` | `18.35%` | `22.76%` |
| 3 | `linear_rowsplit_gemv_proj_6144_q5_kernel` | `279,840` | `96.000` | `6,219.542` | `22.225` | `11.36%` | `14.09%` |
| 4 | `linear_rowsplit_gemv_out_6144_q5_kernel` | `186,560` | `64.000` | `4,091.690` | `21.932` | `7.47%` | `9.27%` |
| 5 | `linear_rowsplit_gemv_lm_head_q6_kernel` | `2,915` | `1.000` | `2,392.203` | `820.653` | `4.37%` | `5.42%` |
| 6 | `linear_rowsplit_gemv_gdn_in_qk_4096_q4_kernel` | `139,920` | `48.000` | `1,302.444` | `9.308` | `2.38%` | `2.95%` |
| 7 | `rmsnorm_d5120_kernel` | `376,035` | `129.000` | `1,158.144` | `3.080` | `2.12%` | `2.62%` |
| 8 | `linear_rowsplit_gemv_attn_in_7168_q5_kernel` | `46,640` | `16.000` | `1,132.594` | `24.284` | `2.07%` | `2.57%` |
| 9 | `gqa_attention_decode_partial_kernel<16,6,true>` | `46,640` | `16.000` | `829.000` | `17.774` | `1.51%` | `1.88%` |
| 10 | `linear_generic_dense_gemv_kernel` | `279,840` | `96.000` | `762.406` | `2.724` | `1.39%` | `1.73%` |
| 11 | `linear_rowsplit_gemv_attn_in_7168_q4_kernel` | `46,640` | `16.000` | `691.808` | `14.833` | `1.26%` | `1.57%` |
| 12 | `gated_delta_rule_recurrent_kernel<128>` | `139,920` | `48.000` | `476.988` | `3.409` | `0.87%` | `1.08%` |
| 13 | `rmsnorm_kernel` | `233,200` | `80.000` | `434.854` | `1.865` | `0.79%` | `0.99%` |
| 14 | `residual_add_kernel` | `373,120` | `128.000` | `399.152` | `1.070` | `0.73%` | `0.90%` |
| 15 | `gqa_attention_decode_reduce_output_kernel<32>` | `46,640` | `16.000` | `337.885` | `7.245` | `0.62%` | `0.77%` |

Grouped decode kernel time:

| bucket | launches | per step | total ms | % decode |
| --- | ---: | ---: | ---: | ---: |
| rowsplit GEMV: `mlp_gate_up_34816` | `186,560` | `64.000` | `12,241.160` | `22.36%` |
| rowsplit GEMV: `mlp_down` | `186,560` | `64.000` | `10,045.379` | `18.35%` |
| rowsplit GEMV: `proj_6144` | `279,840` | `96.000` | `6,219.542` | `11.36%` |
| rowsplit GEMV: `out_6144` | `186,560` | `64.000` | `4,091.690` | `7.47%` |
| rowsplit GEMV: `lm_head` | `2,915` | `1.000` | `2,392.203` | `4.37%` |
| norm kernels | `889,075` | `305.000` | `1,876.117` | `3.43%` |
| rowsplit GEMV: `attn_in_7168` | `93,280` | `32.000` | `1,824.402` | `3.33%` |
| other kernels | `985,270` | `338.000` | `1,627.981` | `2.97%` |
| rowsplit GEMV: `gdn_in_qk_4096` | `139,920` | `48.000` | `1,302.444` | `2.38%` |
| GQA decode attention | `93,280` | `32.000` | `1,166.885` | `2.13%` |
| state/MLP elementwise | `652,960` | `224.000` | `1,155.051` | `2.11%` |
| rowsplit GEMV: `mlp_down` reduce | `186,560` | `64.000` | `185.930` | `0.34%` |
| argmax | `2,915` | `1.000` | `10.457` | `0.02%` |

Rowsplit totals:

| rowsplit subset | launches | per step | total ms | % decode |
| --- | ---: | ---: | ---: | ---: |
| base rowsplit GEMV, excluding split-K reduce | `1,075,635` | `369.000` | `38,116.820` | `69.62%` |
| rowsplit-related, including `mlp_down` reduce | `1,262,195` | `433.000` | `38,302.750` | `69.96%` |

The `mlp_down` split-K reduce kernel adds `64` launches per step but only
`185.930 ms` summed kernel time across the whole decode window. The main
remaining cost is still the primary rowsplit GEMV body, especially MLP
gate/up and MLP down.

## Decode CUDA API

Top CUDA Runtime/API calls inside `qus.decode`:

| API | calls | total ms | avg us | max ms |
| --- | ---: | ---: | ---: | ---: |
| `cudaStreamSynchronize` | `2,915` | `33,277.513` | `11,415.956` | `26.067` |
| `cudaLaunchKernel` | `3,885,695` | `18,739.153` | `4.823` | `13.460` |
| `cudaMemcpy` | `2,915` | `296.386` | `101.676` | `0.368` |
| `cuKernelGetName` | `3,885,695` | `273.697` | `0.070` | `0.203` |
| `cudaMemsetAsync` | `2,915` | `34.355` | `11.786` | `2.040` |
| `cuLibraryGetKernel` | `13` | `0.010` | `0.754` | `0.002` |

`cudaStreamSynchronize` is the host wait for queued GPU work and the generated
token readback per decode step, not a separate GPU kernel bottleneck. The
large `cudaLaunchKernel` sum is nevertheless a visible remaining launch surface:
`1333` kernel launches per decode step in this trace. CUDA Graphs were out of
scope for the executed plan, but this is the main non-kernel overhead visible
from nsys.

## Current bottleneck readout

1. The current decode bottleneck is rowsplit GEMV, led by
   `linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel` at `12.241 s`
   (`22.36%` of decode range, `27.73%` of decode kernel sum).
2. `linear_rowsplit_gemv_mlp_down_q5_kernel` remains second at `10.045 s`
   (`18.35%` of decode range). Its split-K reduce kernel is small in GPU time
   (`0.186 s`) but contributes launch count.
3. Attention decode is not the dominant cost: the two GQA decode kernels sum to
   about `1.167 s`, `2.13%` of decode range.
4. The fusion target of eliminating steady-state D2D copies is met in this
   trace: decode-window D2D and H2D memcopies are zero.
5. The next measurements, if continuing optimization, should be ncu on:
   - `linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel`
   - `linear_rowsplit_gemv_mlp_down_q5_kernel`
   - `linear_rowsplit_gemv_proj_6144_q5_kernel`
   - `linear_rowsplit_gemv_out_6144_q5_kernel`

Nsys identifies where time goes; it does not prove occupancy, memory throughput,
or warp stalls. Use the existing per-kernel ncu artifacts in
`profiles/ncu-linear-v2/` for those claims.
