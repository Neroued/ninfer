# M3 Linear P2 Task 3 - Tuned Low-bit Decode GEMV Round 0

Date: 2026-06-29

Code change: low-bit `T==1` linear plans now route to
`linear.gemv.lowbit.tuned.v1`. The tuned CUDA kernel is codec-templated for
Q4/Q5/Q6, warp-cooperative per output row, externally workspace-free, tail
checked on logical `N,K`, and uses direct decode-and-accumulate with vectorized
BF16 `x` loads.

Artifacts: `profiles/m3-p2-gemv-task3/`.

## Correctness

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
cmake --build build -j --target qus_linear_bench
```

All commands completed successfully. `ctest` reported 1/1 tests passed.
`compute-sanitizer` reported `ERROR SUMMARY: 0 errors`.

## Bench Delta

Command:

```bash
./build/bench/qus_linear_bench --decode
```

Output artifact: `profiles/m3-p2-gemv-task3/linear_decode_bench.txt`.

| shape | Task 1 median | Task 3 median | speedup vs Task 1 |
| --- | ---: | ---: | ---: |
| Q4 mlp.gate/up [17408,5120] | 170.08 us | 126.93 us | 1.34x |
| Q5 mlp.down [5120,17408] | 1730.53 us | 137.93 us | 12.55x |
| Q5 v/z [6144,5120] | 457.06 us | 52.94 us | 8.64x |
| Q6 lm_head [248320,5120] | 3599.84 us | 1874.56 us | 1.92x |

Task 1 medians are from `docs/bench/m3-p2-gemv-baseline.md`.

## NCU Round 0

Preflight:

```bash
~/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
```

Result: 4 OK / 0 WARN / 0 FAIL.

Roofline/stall command shape:

```bash
ncu --force-overwrite -o profiles/m3-p2-gemv-task3/<tag> \
  --set roofline \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --kernel-name regex:'linear_tuned_lowbit_gemv' \
  --launch-skip <skip> --launch-count 1 \
  ./build/bench/qus_linear_bench --decode --q5
```

The Q4 capture used `--q4`. Basic captures with the same kernel regex record
grid size and registers/thread.

| shape | launch skip | grid | NCU duration | DRAM % | Memory SOL % | Compute SOL % | achieved occ. | regs/thread | spills | top stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Q5 mlp.down [5120,17408] | 0 | 1280 | 201.60 us | 17.48 | 89.41 | 11.38 | 62.54% | 34 | 0 local, 0 shared | scoreboard dependency, 46.5 cycles |
| Q4 mlp.gate/up [17408,5120] | 0 | 4352 | 185.98 us | 15.47 | 92.38 | 12.69 | 90.40% | 35 | 0 local, 0 shared | scoreboard dependency, 48.2 cycles |
| Q5 v/z [6144,5120] | 5500 | 1536 | 76.29 us | 15.37 | 83.61 | 12.23 | 73.29% | 34 | 0 local, 0 shared | scoreboard dependency, 43.9 cycles |

Reports:

- `ncu_q5_mlp_down_roofline.ncu-rep`, `.ncu.txt`, `.ncu.csv`
- `ncu_q4_mlp_gate_up_roofline.ncu-rep`, `.ncu.txt`, `.ncu.csv`
- `ncu_q5_vz_6144x5120_roofline.ncu-rep`, `.ncu.txt`, `.ncu.csv`
- matching `*_basic` reports for launch statistics

Round-0 conclusion: Task 3 establishes the tuned registry route, passes the
oracle and sanitizer, and beats the Task 1 event-timed baseline on the required
dominant shapes. The kernel is still not near the calibrated ceiling
`C = 82.76%` DRAM; NCU shows high L1/TEX utilization and scoreboard stalls with
no spills. That is the starting point for Task 4 tuning.
