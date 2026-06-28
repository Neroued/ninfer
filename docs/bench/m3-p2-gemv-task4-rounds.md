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
