# M3 Linear P2 Task 4 - Low-bit Decode GEMV Rounds

Date: 2026-06-29

## Round 1 - Q5 mlp.down [5120,17408]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q5 `mlp.down [5120,17408]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q5
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round1-q5-mlp-down/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-mlp-down/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-mlp-down/baseline_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-mlp-down/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-mlp-down/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-mlp-down/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-mlp-down/after_stalls.ncu.{rep,csv,txt}`

### Diagnosis

Baseline metrics showed the kernel was still L1TEX/load-latency limited: DRAM was only 18.97% while
Memory SOL was 89.56%, L1/TEX throughput was 98.09%, and the top stalls were long scoreboard on
L1TEX plus LG throttle. Source counters showed the Q5 packed path issuing ten uncoalesced 32-bit
loads per row group with 8x sector inflation per load instruction.

Single change: widen only the tuned Q5 packed-row load path from ten 32-bit loads to five aligned
64-bit loads, then feed the existing 32-bit decode path. No public API, registry, format, test, or
policy change.

### Metrics

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 200.61 us | 174.43 us |
| DRAM throughput | 18.97% | 21.58% |
| Memory SOL | 89.56% | 91.13% |
| Compute SOL | 11.40% | 12.99% |
| achieved occupancy | 62.50% | 62.65% |
| registers/thread | 34 | 35 |
| spills/local memory | 0 | 0 |
| long scoreboard | 46.5 cycles, 77.45% | 44.9 cycles, 83.41% |
| LG throttle | 15.35% | 8.62% |

Accepted: the absolute dominant limiter improved, LG throttle dropped, NCU duration improved by
13.05%, and correctness stayed clean. This does not meet the Task 4 floor yet; the shape remains far
below the calibrated ceiling `C = 82.76%`, with remaining L1TEX long-scoreboard stalls.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Q5 Cluster Round 1 - v/z [6144,5120] and out [5120,6144]

Subagent model/effort: gpt-5.5, xhigh.

Target: remaining Q5 cluster shapes from:

```bash
./build/bench/qus_linear_bench --decode --q5
```

Logical launch mapping:

- launch 1: Q5 `v/z [6144,5120]`
- launch 2: Q5 `out [5120,6144]`
- launch 3: Q5 `attn gate [6144,5120]`, same shape as launch 1

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-count 1
```

Capture note: literal `--launch-skip 1` and `--launch-skip 2` target repeated launches inside the
`mlp.down` bench loop, not the next logical decode shapes. Those sanity captures showed grid 1280
and mlp.down-sized global-load traffic, so they were discarded. The committed baseline artifacts use
the actual NCU skips that hit the intended logical shapes in this benchmark loop: skip 1700 for
`v/z` and skip 4500 for `out`.

Artifacts:

- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch1_vz_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch1_vz_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch1_vz_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch1_vz_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch1_vz_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch1_vz_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch1_vz_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch2_out_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch2_out_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch2_out_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch2_out_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch2_out_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch2_out_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/launch2_out_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/rejected_tw2_launch1_vz_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/rejected_tw2_launch1_vz_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/rejected_tw2_launch1_vz_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/rejected_tw2_launch2_out_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/rejected_tw2_launch2_out_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q5-cluster/rejected_tw2_launch2_out_stalls.ncu.{rep,csv,txt}`

### Baseline Metrics

| metric | launch 1 v/z [6144,5120] | launch 2 out [5120,6144] |
| --- | ---: | ---: |
| actual NCU launch skip | 1700 | 4500 |
| grid | 1536 | 1280 |
| waves per SM | 0.75 | 0.63 |
| NCU duration | 27.58 us | 28.54 us |
| DRAM throughput | 42.56% | 41.14% |
| Memory SOL | 64.41% | 63.05% |
| Compute SOL | 64.41% | 63.05% |
| L1/TEX throughput | 75.98% | 77.58% |
| achieved occupancy | 55.61% | 50.69% |
| registers/thread | 40 | 40 |
| spills/local memory | 0 | 0 |
| long scoreboard | 43.38% | 48.00% |
| LG throttle | 11.08% | 8.32% |
| MIO throttle | 11.11% | 8.05% |
| not selected | 12.07% | 9.64% |
| eligible warps/scheduler | 1.28 | 1.09 |
| issued warp/scheduler | 0.47 | 0.48 |
| global-load instructions | 2,457,600 | 2,457,600 |
| global-load L1TEX sectors | 4,915,200 | 4,915,200 |
| global-load data bytes | 101.25 MB | 101.25 MB |
| total SM instructions | 14,475,264 | 14,417,920 |
| source branch instructions | 159,744 | 153,600 |

### Diagnosis

Both distinct cluster shapes do the same total Q5 work (`6144 * 5120 == 5120 * 6144`) and show the
same global-load instruction and sector counts. The remaining limiter is short-K/short-wave
latency hiding, not spills or a new coalescing failure: `v/z` launches only 1536 CTAs (0.75 waves/SM)
and `out` launches only 1280 CTAs (0.63 waves/SM), with achieved occupancy at 55.61% and 50.69%.
The kernel is balanced on Memory and Compute SOL rather than DRAM-bound, and long-scoreboard plus
LG/MIO/not-selected stalls remain the binding issue.

Single experiment tried and rejected: add a Q5-only two-warps-per-row cluster path for `k <= 6144`.
This split each row's K64 groups across two warps in the same CTA and reduced the two partial sums in
shared memory. It left Q4/Q6, public APIs, q5090 ABI, tests, CMake, registry routing, and the
accepted Q5 `mlp.down [5120,17408]` path unchanged while it was under test.

| metric | v/z before | v/z rejected | out before | out rejected |
| --- | ---: | ---: | ---: | ---: |
| grid | 1536 | 3072 | 1280 | 2560 |
| waves per SM | 0.75 | 1.51 | 0.63 | 1.25 |
| NCU duration | 27.58 us | 29.98 us | 28.54 us | 34.66 us |
| DRAM throughput | 42.56% | 39.25% | 41.14% | 33.89% |
| Memory SOL | 64.41% | 61.11% | 63.05% | 52.43% |
| Compute SOL | 64.41% | 61.11% | 63.05% | 52.43% |
| achieved occupancy | 55.61% | 77.18% | 50.69% | 69.26% |
| registers/thread | 40 | 39 | 40 | 39 |
| spills/local memory | 0 | 0 | 0 | 0 |
| long scoreboard | 43.38% | 74.89% | 48.00% | 74.36% |
| LG throttle | 11.08% | 3.09% | 8.32% | 2.88% |
| MIO throttle | 11.11% | 3.38% | 8.05% | 3.28% |
| barrier stall | 0% | 1.14% | 0% | 1.01% |

Rejected: the experiment proved that simply increasing row-level K parallelism raises occupancy but
hurts the actual limiter. Long-scoreboard stall share jumped to about 75%, DRAM throughput fell on
both shapes, and duration regressed by 8.70% on `v/z` and 21.44% on `out`. The source change was
reverted and no code change is accepted for this round.

### Stop Condition

The Q5 cluster stops here under the plan's documented occupancy/short-K limiter rule. It does not
meet the `>=70%` DRAM target or approach `C = 82.76%`, and this is not claimed as target success.
The best accepted kernel remains the current Q5 four-K64 one-warp path: 42.56% DRAM for
`[6144,5120]` and 41.14% DRAM for `[5120,6144]`. The only justified Q5-specific cluster variable
regressed despite higher occupancy, so there is no accepted code patch for the cluster.

### Verification

The rejected experiment was built and passed the oracle plus sanitizer before it was reverted. After
reverting to the final no-code-change tree, the same verification was rerun:

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Round 2 - Q5 mlp.down [5120,17408]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q5 `mlp.down [5120,17408]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q5
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round2-q5-mlp-down/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q5-mlp-down/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q5-mlp-down/baseline_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q5-mlp-down/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q5-mlp-down/baseline_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q5-mlp-down/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q5-mlp-down/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q5-mlp-down/after_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q5-mlp-down/after_stalls.ncu.{rep,csv,txt}`

### Diagnosis

The round-2 baseline confirmed the remaining limiter was access shape, not spills or raw occupancy.
DRAM was still only 21.88% while L1/TEX throughput was 97.93%, long scoreboard was 44.8 cycles, and
Source Counters reported 46,909,440 excessive sectors (86% of 54,318,080 total sectors). This matched
the layout issue expected from the warp-per-row schedule: lanes in one warp touched different K64
groups for the same row, while `TILE_N64_K64` stores rows contiguously within a fixed K64 group.

Single change: specialize only the tuned Q5 GEMV kernel so a warp walks K64 groups serially for one
row while lanes split the 64 values inside each group. Each lane loads its needed Q5 words from the
same 40-byte row group and accumulates two positions per group. This keeps the one-warp-per-row
ownership, avoids CTA/shared-memory K partitioning, and leaves Q4/Q6, public APIs, q5090 ABI, tests,
CMake, and registry routing unchanged.

### Metrics

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 174.34 us | 169.82 us |
| DRAM throughput | 21.88% | 21.63% |
| Memory SOL | 90.95% | 39.01% |
| Compute SOL | 12.96% | 39.01% |
| L1/TEX throughput | 97.93% | 40.03% |
| achieved occupancy | 62.69% | 63.52% |
| registers/thread | 35 | 34 |
| spills/local memory | 0 | 0 |
| long scoreboard | 44.8 cycles, 83.33% | 15.2 cycles, 76.40% |
| second stall | LG throttle, 8.62% | wait, 9.77% |
| LG throttle | 8.62% | 1.68% |
| source coalescing warning | 46,909,440 excessive sectors | no uncoalesced-global-access warning |

Accepted: the diagnosed coalescing/long-scoreboard limiter improved materially, no spills were
introduced, and NCU duration improved by 2.59%. DRAM is effectively flat and still far below the
Task 4 target; against the accepted round-1 DRAM number this moved from 21.58% to 21.63%, so this is
not target success. The next limiter is still issue eligibility/L1TEX latency, with more instruction
work in the row/group-coalesced schedule.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Round 3 - Q5 mlp.down [5120,17408]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q5 `mlp.down [5120,17408]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q5
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/baseline_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/baseline_instr.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/after_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/after_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q5-mlp-down/after_instr.ncu.{rep,csv,txt}`

### Diagnosis

The round-3 baseline showed the remaining limiter as issue eligibility under L1TEX load latency and
unpack load pressure. DRAM was still only 21.44%, Memory SOL and Compute SOL were both 38.52%, and
the scheduler had only 0.76 eligible warps per scheduler. There were no spills, achieved occupancy
was already about 65.76%, and the top stalls were long scoreboard on L1TEX plus wait. The instruction
counter pass showed 9,748,480 global-load instructions and 19,496,960 global-load L1TEX sectors.

Single change: in the Q5-specialized row-group kernel, decode the two lane-owned Q5 values from one
10-bit packed window. This removes the second per-lane Q5 funnel/load window while preserving the
round-2 row-group schedule, one-warp-per-row ownership, public APIs, q5090 ABI, tests, CMake, and
registry routing.

### Metrics

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 172.22 us | 169.41 us |
| DRAM throughput | 21.44% | 22.29% |
| Memory SOL | 38.52% | 29.04% |
| Compute SOL | 38.52% | 29.96% |
| L1/TEX throughput | 40.91% | 30.21% |
| achieved occupancy | 65.76% | 64.41% |
| registers/thread | 34 | 30 |
| spills/local memory | 0 | 0 |
| long scoreboard | 16.0 cycles, 76.01% | 18.4 cycles, 78.11% |
| second stall | wait, 9.75% | wait, 9.71% |
| LG throttle | 1.70% | 0.05% |
| global-load instructions | 9,748,480 | 6,963,200 |
| global-load L1TEX sectors | 19,496,960 | 13,926,400 |
| global-load data bytes | 392.72 MB | 286.88 MB |
| total SM instructions | 76,943,360 | 69,268,480 |

Accepted with concern: the diagnosed Q5 load/unpack pressure improved materially, registers dropped,
there were still no spills, DRAM improved by 0.85 points, and NCU duration improved by 1.63%. This
does not meet the Task 4 target, and the top L1TEX long-scoreboard stall remains binding; its
per-issued-instruction stall cost rose after the instruction count dropped. The next round should
target latency hiding or `x`/payload staging rather than another local bit-extraction cleanup.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Round 4 - Q5 mlp.down [5120,17408]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q5 `mlp.down [5120,17408]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q5
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/baseline_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/baseline_instr.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/after_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/after_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q5-mlp-down/after_instr.ncu.{rep,csv,txt}`

### Diagnosis

The round-4 baseline confirmed the remaining limiter was latency hiding under L1TEX scoreboard
pressure, not spills or raw occupancy. DRAM was 21.76%, Memory SOL was 29.01%, Compute SOL was
29.93%, achieved occupancy was 64.52%, and there were zero local or shared memory spills. The
scheduler issued only 0.31 warp per scheduler with 0.70 eligible warps per scheduler, and the top
stalls were long scoreboard on L1TEX plus wait. The instruction counter pass still showed 6,963,200
global-load instructions and 13,926,400 global-load L1TEX sectors.

Single change: add a Q5-only two-K64-group full-group path in the row-group kernel. Each lane issues
the independent scale, packed payload, and `x` loads for two adjacent full K64 groups before consuming
them, while the existing tail-safe path handles odd or partial final groups. This keeps the round-2
row-group schedule, one-warp-per-row ownership, public APIs, q5090 ABI, tests, CMake, registry
routing, and the externally workspace-free contract unchanged.

### Metrics

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 169.57 us | 73.89 us |
| DRAM throughput | 21.76% | 50.12% |
| Memory SOL | 29.01% | 67.99% |
| Compute SOL | 29.93% | 67.99% |
| L1/TEX throughput | 30.22% | 77.77% |
| achieved occupancy | 64.52% | 64.40% |
| registers/thread | 30 | 38 |
| spills/local memory | 0 | 0 |
| long scoreboard | 18.6 cycles, 79.18% | 12.0 cycles, 73.95% |
| second stall | wait, 9.81% | wait, 6.91% |
| LG throttle | 0.05% | 5.70% |
| eligible warps/scheduler | 0.70 | 0.74 |
| issued warp/scheduler | 0.31 | 0.43 |
| global-load instructions | 6,963,200 | 6,963,200 |
| global-load L1TEX sectors | 13,926,400 | 13,926,400 |
| global-load data bytes | 286.88 MB | 286.88 MB |
| total SM instructions | 69,268,480 | 36,956,160 |
| source branch instructions | 4,904,960 | 389,120 |

Accepted: the diagnosed latency-hiding/issue limiter improved materially. Warp cycles per issued
instruction dropped from 24.20 to 16.60, issued warp per scheduler rose from 0.31 to 0.43, NCU
duration improved by 56.42%, and DRAM throughput rose by 28.36 points with no spills or correctness
regression. This still does not meet the Task 4 target (`>=70%` DRAM and near `C = 82.76%`), and the
kernel remains L1TEX-long-scoreboard limited with higher LG/MIO throttle after the two-group path.
The next round should continue Q5 `mlp.down`, likely targeting the unchanged global-load traffic or
the remaining L1TEX scoreboard pressure.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Round 5 - Q5 mlp.down [5120,17408]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q5 `mlp.down [5120,17408]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q5
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/baseline_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/baseline_instr.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/baseline_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/after_roofline_repeat.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/after_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/after_stalls.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/after_instr.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q5-mlp-down/after_sched.ncu.{rep,csv,txt}`

### Diagnosis

The round-5 baseline confirmed the remaining limiter was L1TEX long-scoreboard latency, not spills
or missing load coalescing. Global-load instructions, L1TEX sectors, and global-load data bytes were
unchanged from round 4, while long scoreboard was still 73.30% of active-warp issue stalls and the
scheduler had only 0.77 eligible warps per scheduler. The kernel still had zero local/shared memory
spilling, leaving some register headroom for more in-flight independent K64 groups.

Single change: replace the Q5-only two-K64 full-group path with a four-K64 full-group path. Each lane
issues scale, packed-payload, and `x` loads for four consecutive full K64 groups before consuming
them; the existing tail-safe per-group path remains responsible for non-multiple or partial final
groups. This keeps the row-group warp schedule, one-warp-per-row ownership, public APIs, q5090 ABI,
tests, CMake, registry routing, and externally workspace-free contract unchanged.

### Metrics

Headline DRAM/SOL values use the roofline report. The first after roofline report was a noisy outlier
against the basic and detailed passes, so the accepted headline uses the repeat roofline capture kept
as `after_roofline_repeat`.

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 71.14 us | 71.10 us |
| DRAM throughput | 49.08% | 53.27% |
| Memory SOL | 71.31% | 71.97% |
| Compute SOL | 71.31% | 71.97% |
| L1/TEX throughput | 78.18% | 81.95% |
| achieved occupancy | 65.71% | 56.35% |
| registers/thread | 38 | 40 |
| spills/local memory | 0 | 0 |
| long scoreboard | 73.30% | 60.69% |
| second stall | wait, 6.82% | not selected, 9.42% |
| LG throttle | 5.61% | 6.51% |
| MIO throttle | 3.52% | 7.29% |
| eligible warps/scheduler | 0.77 | 1.09 |
| issued warp/scheduler | 0.44 | 0.50 |
| global-load instructions | 6,963,200 | 6,963,200 |
| global-load L1TEX sectors | 13,926,400 | 13,926,400 |
| global-load data bytes | 286.88 MB | 286.88 MB |
| total SM instructions | 36,956,160 | 40,325,120 |
| source branch instructions | 389,120 | 378,880 |

Accepted with concern: the diagnosed limiter improved. Long-scoreboard stall share dropped by 12.61
points, eligible warps per scheduler rose from 0.77 to 1.09, issued warp per scheduler rose from
0.44 to 0.50, and roofline DRAM rose by 4.19 points with no spills or correctness regression. The
tradeoff is higher register pressure, lower achieved occupancy, more total SM instructions, and
higher MIO/LG throttle. This still does not meet the Task 4 target (`>=70%` DRAM and near `C =
82.76%`). The accepted duration delta was effectively flat, so Q5 required a final diminishing-returns
check rather than more open-ended tuning.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Q5 Stop Condition - Q5 mlp.down [5120,17408]

Q5 `mlp.down` stops here under the Task 4 diminishing-returns rule. Round 5 was accepted because its
diagnosed limiter improved, but the duration delta was effectively flat (`71.14 us -> 71.10 us`).
A follow-up Q5 round-6 experiment changed only the launch shape to a 64-thread CTA and was rejected
and reverted: duration regressed (`70.91 us -> 71.17 us`), DRAM regressed (`52.79% -> 52.01%`),
global-load traffic was unchanged, there were no spills, and verification passed. A prior shared-`x`
staging experiment was also rejected, but the stop decision is the round-5 flat duration plus the
round-6 regression. Best accepted Q5 remains round 5 at 53.27% DRAM with no spills and the documented
L1TEX/global-load limiter.

## Q4 Round 1 - Q4 mlp.gate/up [17408,5120]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q4 `mlp.gate/up [17408,5120]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q4
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/baseline_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/baseline_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/baseline_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/after_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/after_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/after_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round1-q4-mlp-gate-up/after_global_loads.ncu.{rep,csv,txt}`

### Diagnosis

The Q4 baseline confirmed the dominant limiter was uncoalesced global access through L1TEX, not
spills or missing occupancy. DRAM was only 15.85% while Memory SOL was 95.26%, L1/TEX throughput was
98.23%, achieved occupancy was 90.20%, and the kernel had zero local/shared spilling. Scheduler
metrics showed only 0.18 eligible warps per scheduler, with NCU's detailed rule reporting long
scoreboard at 48.0 cycles, 59.85% of cycles between issued instructions. Program-counter sampling
ranked long scoreboard first and LG throttle second. Source Counters reported 51,423,232 excessive
sectors, 88% of 58,508,288 total sectors, matching the old warp-per-row schedule where lanes touched
different K64 groups at a 2176-byte Q4 tile stride.

Single change: specialize only Q4 so one warp walks K64 groups serially for one output row while
lanes split the 64 values inside each group. Each lane loads one adjacent Q4 packed byte, decodes the
two nibbles it owns, loads the corresponding adjacent `x` pair, and participates in the same
warp-level reduction. This keeps Q5/Q6, public APIs, q5090 ABI, tests, CMake, registry routing, and
the externally workspace-free contract unchanged.

### Metrics

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 179.84 us | 133.82 us |
| DRAM throughput | 15.85% | 21.32% |
| Memory SOL | 95.26% | 31.07% |
| Compute SOL | 13.11% | 35.71% |
| L1/TEX throughput | 98.23% | 42.04% |
| achieved occupancy | 90.20% | 97.06% |
| registers/thread | 35 | 28 |
| spills/local memory | 0 | 0 |
| long scoreboard | 48.0 cycles, 59.85% | 16.9 cycles, 71.4% |
| top PC-sampled stall | long scoreboard, 76.54% | long scoreboard, 77.56% |
| second PC-sampled stall | LG throttle, 14.61% | wait, 10.74% |
| eligible warps/scheduler | 0.18 | 1.29 |
| issued warp/scheduler | 0.14 | 0.48 |
| global-load instructions | 2,193,408 | 5,570,560 |
| global-load L1TEX sectors | 58,490,880 | 9,748,480 |
| global-load requested bytes | 1.87 GB | 311.95 MB |
| global-load data bytes | 225.61 MB | 225.61 MB |
| total SM instructions | 32,204,800 | 65,140,736 |
| source coalescing warning | 51,423,232 excessive sectors | no warning |
| source branch instructions | 3,551,232 | 4,978,688 |

Accepted with concern: the diagnosed access-pattern limiter improved materially. L1TEX sector traffic
dropped by 83.33%, the uncoalesced global-access warning disappeared, long-scoreboard cycles dropped,
eligible warps per scheduler rose from 0.18 to 1.29, NCU duration improved by 25.59%, and DRAM
throughput rose by 5.47 points with no spills or correctness regression. This still does not meet the
Task 4 target (`>=70%` DRAM and near `C = 82.76%`). The tradeoff is higher total instruction count and
higher global-load instruction count; the next limiter is L1TEX long-scoreboard/wait latency in the
row-group schedule rather than sector inflation.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Q4 Round 2 - Q4 mlp.gate/up [17408,5120]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q4 `mlp.gate/up [17408,5120]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q4
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/baseline_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/baseline_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/baseline_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/baseline_stall_reasons.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/after_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/after_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/after_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/after_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round2-q4-mlp-gate-up/after_stall_reasons.ncu.{rep,csv,txt}`

### Diagnosis

The Q4 round-1 kernel no longer had the original sector-inflation problem, but the current baseline
was still load-latency limited. DRAM throughput was 21.58%, Memory SOL was 30.98%, Compute SOL was
35.60%, achieved occupancy was 95.88%, and local/shared spilling was zero. Scheduler metrics showed
only 1.30 eligible warps per scheduler and 0.49 issued warps per scheduler. NCU's warp-state rule
reported long scoreboard at 17.0 cycles, 71.5% of cycles between issued instructions; normalized
stall metrics ranked long scoreboard first at 72.01% and wait second at 10.72%.

Single change: specialize only Q4's full-K64 path to process four K64 groups per loop iteration.
The fast path loads scale bits, packed bytes, and `x` pairs for four full groups before issuing the
eight FMAs. The existing per-group helper remains the tail path for arbitrary `k`, so the generic
backend, public API, q5090 ABI, tests, CMake, registry routing, and externally workspace-free
contract are unchanged.

### Metrics

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 133.98 us | 51.30 us |
| DRAM throughput | 21.58% | 55.22% |
| Memory SOL | 30.98% | 82.64% |
| Compute SOL | 35.60% | 82.64% |
| L1/TEX throughput | 42.41% | 88.82% |
| achieved occupancy | 95.88% | 83.20% |
| registers/thread | 28 | 39 |
| spills/local memory | 0 | 0 |
| long scoreboard | 17.0 cycles, 71.5% | 7.3 cycles, 42.9% |
| top normalized stall | long scoreboard, 72.01% | long scoreboard, 45.65% |
| second normalized stall | wait, 10.72% | not selected, 15.16% |
| eligible warps/scheduler | 1.30 | 2.13 |
| issued warp/scheduler | 0.49 | 0.59 |
| global-load instructions | 5,570,560 | 5,570,560 |
| global-load L1TEX sectors | 9,748,480 | 9,748,480 |
| global-load requested bytes | 311.95 MB | 311.95 MB |
| global-load data bytes | 225.61 MB | 225.61 MB |
| total SM instructions | 65,140,736 | 37,183,488 |
| source branch instructions | 4,978,688 | 452,608 |

Accepted with concern: the diagnosed load-latency limiter improved materially. NCU duration improved
by 61.71%, DRAM throughput rose by 33.64 points, long-scoreboard cycles dropped from 17.0 to 7.3,
eligible warps per scheduler rose from 1.30 to 2.13, total SM instructions fell by 42.92%, and branch
instructions fell by 90.91%, with no spills or correctness regression. This still does not meet the
Task 4 target (`>=70%` DRAM and near `C = 82.76%`), and the kernel is now balanced between Memory SOL
and Compute SOL rather than clearly memory-bound. The tradeoff is higher register pressure
(28 -> 39 regs/thread) and lower achieved occupancy (95.88% -> 83.20%). The next limiter appears to
be mixed L1TEX/issue pressure, with long scoreboard still first and not-selected/MIO-throttle stalls
now prominent.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Q4 Round 3 - Q4 mlp.gate/up [17408,5120]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q4 `mlp.gate/up [17408,5120]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q4
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/baseline_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/baseline_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/baseline_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/baseline_stall_reasons.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/after_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/after_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/after_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/after_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round3-q4-mlp-gate-up/after_stall_reasons.ncu.{rep,csv,txt}`

### Diagnosis

The round-3 baseline showed the round-2 Q4 x4 path was no longer sector-inflation limited, but still
had x4-path load-instruction and issue pressure. DRAM was 55.16%, Memory SOL and Compute SOL were
both 82.05%, achieved occupancy was 83.14%, registers were 39/thread, and local/shared spilling was
zero. Scheduler metrics showed 2.15 eligible warps per scheduler and 0.59 issued warp per scheduler.
The top stall remained long scoreboard at 7.3 cycles, 42.7% of cycles between issued instructions;
normalized stall metrics also showed not-selected at 15.12%, MIO throttle at 12.25%, and LG throttle
at 8.84%. Global-load traffic matched round 2: 5,570,560 global-load instructions and 9,748,480
L1TEX sectors.

Single change: specialize only Q4's full-K64 x4 path so lanes 0-3 cooperatively issue the four scale
halfword loads, one lane per group, then broadcast them with shuffles after the packed-byte and `x`
loads are issued. This replaces four serial lane-0 byte-pair scale loads in the x4 fast path. The
tail path, Q5/Q6 paths, public API, q5090 ABI, tests, CMake, registry routing, and externally
workspace-free contract are unchanged.

### Metrics

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 51.90 us | 47.90 us |
| DRAM throughput | 55.16% | 60.00% |
| Memory SOL | 82.05% | 60.00% |
| Compute SOL | 82.05% | 58.85% |
| L1/TEX throughput | 89.08% | 69.00% |
| achieved occupancy | 83.14% | 82.75% |
| registers/thread | 39 | 36 |
| spills/local memory | 0 | 0 |
| long scoreboard | 7.3 cycles, 42.7% | 11.3 cycles, 66.9% |
| top normalized stall | long scoreboard, 46.15% | long scoreboard, 68.72% |
| second normalized stall | not selected, 15.12% | not selected, 10.66% |
| MIO throttle | 12.25% | 1.43% |
| LG throttle | 8.84% | 0.65% |
| eligible warps/scheduler | 2.15 | 1.65 |
| issued warp/scheduler | 0.59 | 0.59 |
| global-load instructions | 5,570,560 | 3,133,440 |
| global-load L1TEX sectors | 9,748,480 | 8,355,840 |
| global-load requested bytes | 311.95 MB | 267.39 MB |
| global-load data bytes | 225.61 MB | 225.61 MB |
| total SM instructions | 37,183,488 | 32,657,408 |
| source coalescing warning | no warning | 1,044,480 excessive sectors |
| source branch instructions | 452,608 | 452,608 |

Accepted with concern: the diagnosed x4 load-instruction pressure improved materially. Global-load
instructions dropped by 43.75%, L1TEX sectors dropped by 14.29%, total SM instructions dropped by
12.18%, MIO/LG throttling fell sharply, registers dropped, DRAM throughput rose by 4.84 points, and
NCU duration improved by 7.71% with no spills or correctness regression. This does not meet the Task
4 target (`>=70%` DRAM and near `C = 82.76%`). The concern is that the cooperative scale load creates
a small scattered-scale access warning and shifts the remaining limiter harder onto L1TEX long
scoreboard; eligible warps per scheduler also fell. The next round should target latency hiding or
cache behavior rather than another pure instruction-count cleanup.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Q4 Round 4 - Q4 mlp.gate/up [17408,5120]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q4 `mlp.gate/up [17408,5120]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q4
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/baseline_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/baseline_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/baseline_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/baseline_stall_reasons.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/after_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/after_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/after_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/after_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round4-q4-mlp-gate-up/after_stall_reasons.ncu.{rep,csv,txt}`

### Diagnosis

The round-4 baseline confirmed practical diminishing returns: the Q4 x4 cooperative-scale path was
still dominated by L1TEX long-scoreboard latency, but not by spills, gross coalescing failure, or
branch divergence. DRAM was 59.87%, Memory SOL was 59.87%, Compute SOL was 59.13%, achieved
occupancy was 82.82%, registers were 36/thread, and local/shared spilling was zero. Scheduler
metrics showed 1.71 eligible warps per scheduler and 0.60 issued warp per scheduler. The top stall
was long scoreboard at 11.1 cycles, 65.7% of cycles between issued instructions. The source
coalescing warning remained limited to 1,044,480 excessive sectors, 12% of 8,373,248 total sectors.

Single change: specialize only the Q4 kernel schedule so each output row is split across two warps in
the same CTA. Each paired warp owns alternating x4 K-group chunks, keeps the existing coalesced
packed-byte loads and cooperative scale loads inside its chunks, then the two lane-0 partial sums are
combined through a tiny shared-memory reduction. This targets long-scoreboard latency hiding and the
partial-wave effect without changing Q5/Q6 paths, public APIs, q5090 ABI, tests, CMake, registry
routing, or the externally workspace-free contract.

### Metrics

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 47.58 us | 46.27 us |
| DRAM throughput | 59.87% | 60.61% |
| Memory SOL | 59.87% | 62.40% |
| Compute SOL | 59.13% | 62.40% |
| L1/TEX throughput | 66.91% | 68.51% |
| achieved occupancy | 82.82% | 76.34% |
| registers/thread | 36 | 41 |
| spills/local memory | 0 | 0 |
| long scoreboard | 11.1 cycles, 65.7% | 9.1 cycles, 61.9% |
| top normalized stall | long scoreboard, 66.25% | long scoreboard, 67.75% |
| second normalized stall | not selected, 10.76% | not selected, 9.81% |
| barrier stall | 0% | 3.23% |
| eligible warps/scheduler | 1.71 | 1.49 |
| issued warp/scheduler | 0.60 | 0.62 |
| global-load instructions | 3,133,440 | 3,133,440 |
| global-load L1TEX sectors | 8,355,840 | 8,355,840 |
| global-load requested bytes | 267.39 MB | 267.39 MB |
| global-load data bytes | 225.61 MB | 225.61 MB |
| total SM instructions | 32,657,408 | 34,676,736 |
| source coalescing warning | 1,044,480 excessive sectors | 1,044,480 excessive sectors |
| source branch instructions | 452,608 | 957,440 |

Accepted with concern: the diagnosed long-scoreboard limiter improved on the warp-state metric, NCU
duration improved by 2.75%, DRAM throughput rose by 0.74 points, and correctness plus sanitizer
verification stayed clean. This does not meet the Task 4 target (`>=70%` DRAM and near
`C = 82.76%`). The concern is that the gain is small and comes with higher register pressure,
lower achieved occupancy, more branch/control instructions, and new barrier stalls; the residual
source coalescing warning is unchanged. This is likely a diminishing-returns area unless a later
round finds a lower-overhead way to hide the same L1TEX latency or remove the scale-sector warning.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.

## Q4 Round 5 - Q4 mlp.gate/up [17408,5120]

Subagent model/effort: gpt-5.5, xhigh.

Target: Q4 `mlp.gate/up [17408,5120]`, launch 0 of:

```bash
./build/bench/qus_linear_bench --decode --q4
```

NCU kernel filter:

```bash
--kernel-name regex:'linear_tuned_lowbit_gemv_kernel' --launch-skip 0 --launch-count 1
```

Artifacts:

- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/baseline_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/baseline_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/baseline_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/baseline_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/baseline_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/baseline_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/baseline_stall_reasons.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/after_basic.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/after_roofline.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/after_detailed.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/after_source.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/after_instr_sched.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/after_global_loads.ncu.{rep,csv,txt}`
- `profiles/m3-p2-gemv-task4/round5-q4-mlp-gate-up/after_stall_reasons.ncu.{rep,csv,txt}`

### Diagnosis

The round-5 baseline was still limited by L1TEX long-scoreboard latency and register-limited
occupancy, not spills or new memory traffic. DRAM throughput was 60.01%, Memory SOL and Compute SOL
were both 63.83%, L1/TEX throughput was 68.59%, achieved occupancy was 76.39%, and the compiler used
41 registers/thread. Scheduler metrics showed 1.48 eligible warps per scheduler and 0.61 issued warp
per scheduler. The warp-state rule reported long scoreboard at 9.1 cycles, 61.2% of cycles between
issued instructions, while the normalized stall capture reported long scoreboard at 64.81%. The
source coalescing warning stayed at 1,044,480 excessive sectors, 12% of 8,373,248 total sectors.

Single change: add a Q4-specialization-only `__launch_bounds__(128, 12)` hint. This tests whether
nvcc can trim the round-4 41-register allocation enough to raise occupancy and hide the remaining
L1TEX latency without changing the Q4 schedule, Q5/Q6 paths, public APIs, q5090 ABI, tests, CMake,
registry routing, or the externally workspace-free contract.

### Metrics

| metric | before | after |
| --- | ---: | ---: |
| NCU duration | 45.28 us | 45.31 us |
| DRAM throughput | 60.01% | 63.74% |
| Memory SOL | 63.83% | 70.40% |
| Compute SOL | 63.83% | 70.40% |
| L1/TEX throughput | 68.59% | 78.37% |
| achieved occupancy | 76.39% | 89.61% |
| theoretical occupancy | 83.33% | 100.00% |
| registers/thread | 41 | 40 |
| spills/local memory | 0 | 0 |
| long scoreboard | 9.1 cycles, 61.2% | 9.5 cycles, 58.2% |
| top normalized stall | long scoreboard, 64.81% | long scoreboard, 60.99% |
| second normalized stall | not selected, 9.80% | not selected, 11.46% |
| barrier stall | 3.08% | 3.05% |
| eligible warps/scheduler | 1.48 | 1.90 |
| issued warp/scheduler | 0.61 | 0.67 |
| global-load instructions | 3,133,440 | 3,133,440 |
| global-load L1TEX sectors | 8,355,840 | 8,355,840 |
| global-load requested bytes | 267.39 MB | 267.39 MB |
| global-load data bytes | 225.61 MB | 225.61 MB |
| total SM instructions | 34,676,736 | 35,825,664 |
| source coalescing warning | 1,044,480 excessive sectors | 1,044,480 excessive sectors |
| source branch instructions | 957,440 | 957,440 |

Accepted with concern: the intended limiter improved without spills or correctness regression. The
launch-bound hint reduced registers from 41 to 40, raised theoretical occupancy to 100%, raised
achieved occupancy by 13.22 points, increased eligible warps per scheduler from 1.48 to 1.90, reduced
the normalized long-scoreboard stall share, and raised DRAM throughput by 3.73 points. This still
does not meet the Task 4 target (`>=70%` DRAM and near `C = 82.76%`), and the basic-capture duration
is effectively flat at 45.28 us -> 45.31 us. The tradeoff is a higher executed-instruction count and
no reduction in the residual scale-sector coalescing warning.

Q4 stops here under the Task 4 diminishing-returns rule. Rounds 4 and 5 each yielded less than 5%
duration improvement, and round 5 shows that even a low-overhead occupancy/latency-hiding knob moves
occupancy and DRAM% more than wall time. The remaining limiter is the same Q4 scale-sector/L1TEX
scoreboard behavior, with balanced Memory and Compute SOL rather than clear memory-bound execution.

### Verification

```bash
cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All passed. `ctest` reported 1/1 tests passed. `compute-sanitizer` reported `ERROR SUMMARY: 0
errors`.
