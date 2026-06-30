# Q5090 V2 Rowsplit GEMV NCU Report

> Date: 2026-06-30
> Scope: Nsight Compute profiling of the q5090 v2 `linear_rowsplit_gemv_*` decode kernels.
> GPU: NVIDIA GeForce RTX 5090
> Nsight Compute: `2025.4.1.0`

## Executive Summary

All model-used rowsplit GEMV specializations were profiled at their real Qwen3.6-27B dimensions and
qtypes. The measurement used `qus_linear_op_bench` single-target runs so each Nsight Compute replay
captured exactly one target kernel launch.

The strongest optimization candidates are:

1. `linear_rowsplit_gemv_mlp_down_q5_kernel` for `MlpDown5120x17408 Q5`.
   It is high end-to-end impact and underfilled: cold-cache bench reaches only `50.2%` of the local
   copy ceiling, NCU reports `0.63` waves/SM, `62.9%` achieved occupancy, `47.3%` no-eligible
   scheduler cycles, and L1TEX scoreboard stalls at `39.8%`.
2. `linear_rowsplit_gemv_proj_6144_q5_kernel` and
   `linear_rowsplit_gemv_out_6144_q5_kernel`.
   These are also high end-to-end impact. Both run below `41%` local cold-cache bandwidth and below
   one wave/SM, with Q5 divergence and L1TEX scoreboard stalls.
3. Small-N kernels, especially `AttnKV1024x5120` Q4/Q5 and `GdnQK2048x5120` Q4.
   These have the worst utilization, but lower aggregate decode share. Treat them as high headroom
   and lower impact than MLP/projection work.
4. `linear_rowsplit_gemv_mlp_gate_up_q4_kernel` is still expensive in end-to-end decode, but it is
   the healthiest non-lm-head rowsplit kernel: `59.8%` local cold-cache bandwidth, `66.5%` NCU SOL
   memory/DRAM throughput, `83.7%` achieved occupancy, and no branch divergence.
5. `linear_rowsplit_gemv_lm_head_q6_kernel` should not be first-priority tuning work. It is already
   near the measured memory/SOL ceiling: `86.5%` local cold-cache bandwidth, `91.6%` NCU memory
   throughput, `96.7%` achieved occupancy, and `30.43` waves/SM.

## Workload And Artifacts

Preflight:

```bash
/home/neroued/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
```

Result: `4 OK / 0 WARN / 0 FAIL`; ncu profiling permission probe is WSL2-host controlled.

Cold-cache per-op baseline:

```bash
./build/bench/qus_linear_op_bench \
  --all-targets \
  --repeat 10 \
  --warmup 3 \
  --copy-repeat 4 \
  --csv-out profiles/ncu-rowsplit-v2/pre_ncu_bench.csv
```

Detailed profile template:

```bash
ncu --force-overwrite \
  --set detailed \
  --replay-mode application \
  --kernel-name regex:<kernel> \
  --launch-skip 0 \
  --launch-count 1 \
  -o profiles/ncu-rowsplit-v2/<tag>_detailed \
  ./build/bench/qus_linear_op_bench \
    --shape <ShapeFamily> \
    --qtype <Q4|Q5|Q6> \
    --warmup 0 \
    --repeat 1 \
    --stream-ceiling-gbs 1512.892
```

Scheduler and warp-state template:

```bash
ncu --force-overwrite \
  --replay-mode application \
  --section SchedulerStats \
  --section WarpStateStats \
  --kernel-name regex:<kernel> \
  --launch-skip 0 \
  --launch-count 1 \
  -o profiles/ncu-rowsplit-v2/<tag>_sched \
  ./build/bench/qus_linear_op_bench \
    --shape <ShapeFamily> \
    --qtype <Q4|Q5|Q6> \
    --warmup 0 \
    --repeat 1 \
    --stream-ceiling-gbs 1512.892
```

Primary artifacts:

| Artifact | Path |
|---|---|
| cold-cache baseline CSV | `profiles/ncu-rowsplit-v2/pre_ncu_bench.csv` |
| detailed ncu reports | `profiles/ncu-rowsplit-v2/*_detailed.ncu.{rep,csv,txt}` |
| scheduler/warp ncu reports | `profiles/ncu-rowsplit-v2/*_sched.ncu.{rep,csv,txt}` |
| combined parsed summary | `profiles/ncu-rowsplit-v2/rowsplit_ncu_combined_summary.md` |
| run logs | `profiles/ncu-rowsplit-v2/logs/*.log` |

The benchmark's printed timings under NCU application replay are intentionally ignored because ncu
replay distorts CUDA event timing. The independent cold-cache CSV is the timing source; the ncu
reports are the metric source.

## Target Coverage

These are the model-used rowsplit GEMV targets in `bench/linear_op_bench.cu`:

| ShapeFamily | Qtype | N | K | Kernel |
|---|---:|---:|---:|---|
| `MlpGateUp17408x5120` | Q4 | 17408 | 5120 | `linear_rowsplit_gemv_mlp_gate_up_q4_kernel` |
| `MlpDown5120x17408` | Q5 | 5120 | 17408 | `linear_rowsplit_gemv_mlp_down_q5_kernel` |
| `LmHead248320x5120` | Q6 | 248320 | 5120 | `linear_rowsplit_gemv_lm_head_q6_kernel` |
| `Proj6144x5120` | Q5 | 6144 | 5120 | `linear_rowsplit_gemv_proj_6144_q5_kernel` |
| `Proj6144x5120` | Q4 | 6144 | 5120 | `linear_rowsplit_gemv_proj_6144_q4_kernel` |
| `Out5120x6144` | Q5 | 5120 | 6144 | `linear_rowsplit_gemv_out_6144_q5_kernel` |
| `GdnQK2048x5120` | Q4 | 2048 | 5120 | `linear_rowsplit_gemv_gdn_qk_2048_q4_kernel` |
| `AttnKV1024x5120` | Q5 | 1024 | 5120 | `linear_rowsplit_gemv_attn_kv_1024_q5_kernel` |
| `AttnKV1024x5120` | Q4 | 1024 | 5120 | `linear_rowsplit_gemv_attn_kv_1024_q4_kernel` |

## Summary Metrics

`bench DRAM %` is relative to the local cold-cache copy ceiling measured by the benchmark
(`1512.892 GB/s`). `SOL mem/DRAM %` is Nsight Compute's SpeedOfLight percentage. These are different
denominators and should not be numerically equated.

| tag | shape/qtype | bench us | bench DRAM % | SOL mem/DRAM % | occ % | waves/SM | eligible warp/sched | no eligible % | top stall | regs | priority |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| `q4_mlp_gate_up` | `MlpGateUp17408x5120 Q4` | `52.384` | `59.8` | `66.5/66.5` | `83.7` | `2.13` | `2.60` | `32.5` | L1TEX scoreboard `44.6%` | 40 | medium |
| `q5_mlp_down` | `MlpDown5120x17408 Q5` | `77.120` | `50.2` | `66.1/40.0` | `62.9` | `0.63` | `1.41` | `47.3` | L1TEX scoreboard `39.8%` | 38 | high |
| `q6_lm_head` | `LmHead248320x5120 Q6` | `759.104` | `86.5` | `91.6/62.8` | `96.7` | `30.43` | `2.81` | `23.3` | L1TEX scoreboard `38.4%` | 26 | low |
| `q5_proj_6144` | `Proj6144x5120 Q5` | `34.048` | `40.1` | `62.4/35.0` | `71.8` | `0.75` | `1.76` | `43.5` | L1TEX scoreboard `34.6%` | 38 | high |
| `q4_proj_6144` | `Proj6144x5120 Q4` | `23.776` | `46.5` | `55.6/55.6` | `64.9` | `0.75` | `1.84` | `37.2` | L1TEX scoreboard `48.6%` | 40 | high |
| `q5_out_6144` | `Out5120x6144 Q5` | `36.064` | `37.9` | `60.9/34.9` | `60.8` | `0.63` | `1.50` | `43.5` | L1TEX scoreboard `37.1%` | 32 | high |
| `q4_gdn_qk` | `GdnQK2048x5120 Q4` | `11.552` | `32.0` | `44.4/41.3` | `73.0` | `1.00` | `2.44` | `47.0` | n/a | 40 | high |
| `q5_attn_kv` | `AttnKV1024x5120 Q5` | `13.312` | `17.1` | `36.3/18.6` | `100.0` | `1.00` | `1.55` | `47.3` | L1TEX scoreboard `42.6%` | 40 | high |
| `q4_attn_kv` | `AttnKV1024x5120 Q4` | `7.392` | `25.0` | `30.9/28.7` | `45.0` | `0.50` | `1.01` | `55.0` | L1TEX scoreboard `49.6%` | 40 | high |

Spills were zero for all profiled targets.

## Kernel Assessments

### Q4 MLP Gate/Up `[17408,5120]`

This is the largest rowsplit decode cost in the long NVTX trace, but the kernel is not the weakest
per-op implementation. It reaches `59.8%` of the local cold-cache copy ceiling and `66.5%` NCU
memory/DRAM SOL. Occupancy is `83.7%` with `2.13` waves/SM. NCU calls compute and memory
"well-balanced", with the remaining visible stall still L1TEX scoreboard.

Assessment: important because it is large in aggregate, but the per-kernel headroom is moderate.

### Q5 MLP Down `[5120,17408]`

This is the strongest high-impact optimization target. It is the second largest rowsplit decode
kernel in the long trace and reaches only `50.2%` of the local cold-cache ceiling. NCU reports only
`0.63` waves/SM, `62.9%` occupancy, `1.41` eligible warps/scheduler, and `47.3%` no-eligible cycles.
The main stall rule is L1TEX scoreboard at `39.8%`; branch efficiency is only `45.4%`.

Assessment: high priority. The evidence points to a combination of underfilled launch geometry,
memory dependency latency, and Q5 unpack/control overhead.

### Q6 LM Head `[248320,5120]`

This kernel is already near the ceiling under the current layout. The cold-cache bench reaches
`86.5%` of the local copy ceiling, NCU reports `91.6%` memory throughput, occupancy is `96.7%`, and
there are `30.43` waves/SM. It still has L1TEX scoreboard stalls, but the SOL rule says the workload
is already using more than `80%` of available compute or memory performance.

Assessment: low priority for tuning unless the end-to-end profile changes and lm_head grows further.

### Q5 Projection `[6144,5120]`

This kernel has high aggregate impact through the decode projection family. The cold-cache bench
reaches `40.1%` of local ceiling, NCU SOL DRAM is `35.0%`, and there are only `0.75` waves/SM.
Scheduler data shows `43.5%` no-eligible cycles and L1TEX scoreboard at `34.6%`.

Assessment: high priority. This shape is underfilled and memory-dependency limited.

### Q4 Projection `[6144,5120]`

The Q4 projection is better than Q5 in cold-cache bandwidth (`46.5%`) but still underfilled at
`0.75` waves/SM. Unlike Q5, branch efficiency is `100%`; the main stall is L1TEX scoreboard
(`48.6%`). NCU also flags the grid as too small to fill the device.

Assessment: high headroom, lower aggregate impact than Q5 projection.

### Q5 Out `[5120,6144]`

This is another high-impact Q5 target. Cold-cache bandwidth is only `37.9%`, occupancy is `60.8%`,
and waves/SM is `0.63`. Scheduler no-eligible cycles are `43.5%`, with L1TEX scoreboard at `37.1%`
and branch efficiency at `45.5%`.

Assessment: high priority. It shares the same symptoms as Q5 projection and Q5 MLP down.

### Q4 GDN Q/K `[2048,5120]`

This shape is small enough that utilization is poor even though the kernel has one full wave/SM.
Cold-cache bandwidth is `32.0%`; NCU reports `44.4%` memory throughput and `47.0%` no-eligible
scheduler cycles. The scheduler profile did not emit a specific CPI stall rule for this kernel, but
the low utilization and one-wave launch still make it clearly shape limited.

Assessment: high headroom, moderate end-to-end impact.

### Q5 Attention KV `[1024,5120]`

This is the weakest Q5 target by bandwidth: `17.1%` cold-cache ceiling and only `18.6%` SOL DRAM.
It has `1.00` waves/SM and high occupancy, but only `1.55` eligible warps/scheduler, `47.3%`
no-eligible cycles, L1TEX scoreboard at `42.6%`, and branch efficiency at `55.3%`.

Assessment: high headroom but lower aggregate decode impact.

### Q4 Attention KV `[1024,5120]`

This is the most underfilled launch: `0.50` waves/SM. Cold-cache bandwidth is `25.0%`, occupancy is
`45.0%`, eligible warps/scheduler is `1.01`, and no-eligible cycles are `55.0%`. L1TEX scoreboard
accounts for `49.6%` of warp cycles per issued instruction. NCU explicitly flags the grid as too
small to fill the device.

Assessment: high headroom but low aggregate decode impact.

## Priority Order

Recommended priority by end-to-end impact times per-kernel headroom:

1. Q5 MLP down `[5120,17408]`.
2. Q5 projection/out family: `Proj6144x5120 Q5` and `Out5120x6144 Q5`.
3. Q4 MLP gate/up `[17408,5120]`, because aggregate decode time is large even though the kernel is
   already moderately healthy.
4. Q4 projection and GDN/attention small-N kernels, where headroom is large but decode share is
   smaller.
5. Q6 lm_head, which is already near memory/SOL ceiling.

## Caveats

- The benchmark uses synthetic row-split payloads and BF16 inputs, but dispatches through
  `qus::kernels::linear` and the same specialized kernel registry as the runtime.
- Cold-cache benchmark time and NCU replay metrics have different collection mechanics. Use the CSV
  for timing and ncu for architectural evidence.
- The local copy ceiling for this run was `1512.892 GB/s`, lower than some earlier reports on the
  same machine. The ranking still holds because all rows use the same ceiling.
- NCU alone does not prove a concrete optimization. It identifies the likely limiting signatures:
  partial-wave underfill, low eligible warp count, L1TEX scoreboard stalls, and Q5 branch/predicate
  overhead.
