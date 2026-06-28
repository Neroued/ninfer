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
