# M3 Linear P2 Task 2 - Branchless Q5/Q6 Codec Unpack

Date: 2026-06-29

Code change: `Q5Codec::load_group` and `Q6Codec::load_group` now use aligned
`uint32_t` word loads plus `__funnelshift_r` extraction on the canonical
`TILE_N64_K64` payload layout. Q4 was not changed.

Artifacts: `profiles/m3-p2-gemv-task2/`.

## Correctness

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure
compute-sanitizer ./build/tests/qus_linear_test
```

All three commands completed successfully. `compute-sanitizer` reported
`ERROR SUMMARY: 0 errors`.

## Bench Delta

Command:

```bash
./build/bench/qus_linear_bench --decode --q5 --q6
```

| shape | Task 1 median | Task 2 median | speedup |
| --- | ---: | ---: | ---: |
| Q5 mlp.down [5120,17408] | 1730.53 us | 556.61 us | 3.11x |
| Q5 v/z [6144,5120] | 457.06 us | 165.79 us | 2.76x |
| Q5 out [5120,6144] | 551.78 us | 198.30 us | 2.78x |
| Q5 attn gate [6144,5120] | 457.82 us | 166.34 us | 2.75x |
| Q6 lm_head [248320,5120] | 3599.84 us | 850.62 us | 4.23x |

Task 1 medians are from `docs/bench/m3-p2-gemv-baseline.md`. Task 2 output is
saved as `profiles/m3-p2-gemv-task2/linear_decode_q5_q6_bench.txt`.

## NCU Delta

Preflight:

```bash
~/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
```

Result: 4 OK / 0 WARN / 0 FAIL.

Roofline/stall command shape:

```bash
ncu --force-overwrite -o profiles/m3-p2-gemv-task2/<tag> \
  --set roofline \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --kernel-name regex:'lowbit_gemv' --launch-skip <skip> --launch-count 1 \
  ./build/bench/qus_linear_bench --decode --q5
```

Instruction-mix command shape:

```bash
ncu --force-overwrite -o profiles/m3-p2-gemv-task2/<tag> \
  --metrics smsp__sass_thread_inst_executed_op_conversion_pred_on.sum,smsp__sass_thread_inst_executed_op_integer_pred_on.sum,smsp__sass_thread_inst_executed_op_memory_pred_on.sum,smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.sum \
  --kernel-name regex:'lowbit_gemv' --launch-skip <skip> --launch-count 1 \
  ./build/bench/qus_linear_bench --decode --q5
```

| metric | Task 1 Q5 mlp.down | Task 2 Q5 mlp.down | Task 2 Q5 v/z |
| --- | ---: | ---: | ---: |
| launch skip | 0 | 0 | 1000 |
| grid | 40 | 40 | 48 |
| ncu duration | 4.86 ms | 1.96 ms | 301.60 us |
| DRAM throughput | 2.77% | 4.24% | 4.57% |
| compute SOL | 5.81% | 2.20% | 2.74% |
| achieved occupancy | 8.32% | 8.36% | 8.35% |
| registers/thread | 40 | 79 | 79 |
| issue slots busy | 4.82% | 2.28% | 2.75% |
| issued instructions | 171,735,840 | 32,123,840 | 11,342,976 |
| integer instructions | 3,672,453,120 | 557,096,960 | 196,657,152 |
| memory instructions | 608,624,640 | 107,294,720 | 37,920,768 |
| FP add/mul/ffma instructions | 178,257,920 | 178,257,920 | 62,914,560 |

The Q5 mlp.down integer instruction count dropped 84.8%, memory instruction count
dropped 82.4%, and issued instruction count dropped 81.3%. The previous fixed
latency stall entry on Q5 mlp.down was 35.27% / 1.7 cycles and is no longer a
top stall in the Task 2 roofline capture. The remaining top stall is L1TEX
scoreboard at 71.42% / 7.3 cycles for mlp.down and 71.5% / 7.3 cycles for v/z.

Bandwidth is still low because this is still the scalar generic GEMV skeleton.
No local or shared memory spilling was reported.
