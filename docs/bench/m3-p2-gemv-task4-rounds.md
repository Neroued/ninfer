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
