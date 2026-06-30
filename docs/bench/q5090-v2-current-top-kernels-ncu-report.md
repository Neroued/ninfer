# q5090 v2 current top decode kernels NCU report

> Scope: NCU sampling for the current `master` top decode kernels identified by
> `docs/bench/q5090-v2-tuned-numeric-long-decode-nsys-report.md`. Build commit:
> `ebbbfd7` (`bench(q5090): update numeric tuning ledger`).

This report samples the current top decode kernels with Nsight Compute using
`qus_linear_op_bench`, not full-model replay. That keeps the measurement scoped
to one logical linear operator per capture. The target list is the top five
rowsplit GEMV kernels by `qus.decode` nsys time:

1. `linear_rowsplit_gemv_mlp_gate_up_34816_q4_kernel`
2. `linear_rowsplit_gemv_mlp_down_q5_kernel`
3. `linear_rowsplit_gemv_proj_6144_q5_kernel`
4. `linear_rowsplit_gemv_out_6144_q5_kernel`
5. `linear_rowsplit_gemv_lm_head_q6_kernel`

## Commands

Preflight and build:

```bash
/home/neroued/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
cmake --build build --target qus_linear_op_bench -j
```

For each target, the normal cold-cache bench was recorded first:

```bash
./build/bench/qus_linear_op_bench \
  --shape <ShapeFamily> --qtype <Q4|Q5|Q6> \
  --csv-out profiles/ncu-top-kernels-current/<tag>.bench.csv \
  > profiles/ncu-top-kernels-current/<tag>.bench.txt
```

Then one launch was captured with detailed NCU metrics:

```bash
ncu --force-overwrite --target-processes all --replay-mode application \
  --set detailed \
  --kernel-name regex:'<kernel_regex>' \
  --launch-skip 0 --launch-count 1 \
  -o profiles/ncu-top-kernels-current/<tag>.ncu-rep \
  ./build/bench/qus_linear_op_bench \
    --shape <ShapeFamily> --qtype <Q4|Q5|Q6> \
    --repeat 1 --warmup 0 \
    --csv-out profiles/ncu-top-kernels-current/<tag>.ncu.bench.csv

ncu --import profiles/ncu-top-kernels-current/<tag>.ncu-rep --csv \
  > profiles/ncu-top-kernels-current/<tag>.ncu.csv
ncu --import profiles/ncu-top-kernels-current/<tag>.ncu-rep --page details \
  > profiles/ncu-top-kernels-current/<tag>.ncu.txt
```

Because NCU 2025.4.1 did not include scheduler/warp-state sections in
`--set detailed` for these captures, each target also got a second one-launch
capture:

```bash
ncu --force-overwrite --target-processes all --replay-mode application \
  --section SchedulerStats --section WarpStateStats \
  --kernel-name regex:'<kernel_regex>' \
  --launch-skip 0 --launch-count 1 \
  -o profiles/ncu-top-kernels-current/<tag>.scheduler.ncu-rep \
  ./build/bench/qus_linear_op_bench \
    --shape <ShapeFamily> --qtype <Q4|Q5|Q6> \
    --repeat 1 --warmup 0 \
    --csv-out profiles/ncu-top-kernels-current/<tag>.scheduler.bench.csv
```

The `*.ncu.bench.csv` files run under NCU replay and are not used as operator
performance numbers below. The cold-cache `*.bench.csv` files are used for
operator latency/bandwidth.

## Artifacts

All raw captures and exports are under:

`profiles/ncu-top-kernels-current/`

Per target:

| tag | shape | qtype | detailed report | scheduler report |
| --- | --- | --- | --- | --- |
| `mlp_gate_up_34816_q4` | `MlpGateUp34816x5120` | Q4 | `mlp_gate_up_34816_q4.ncu-rep` | `mlp_gate_up_34816_q4.scheduler.ncu-rep` |
| `mlp_down_q5` | `MlpDown5120x17408` | Q5 | `mlp_down_q5.ncu-rep` | `mlp_down_q5.scheduler.ncu-rep` |
| `proj_6144_q5` | `Proj6144x5120` | Q5 | `proj_6144_q5.ncu-rep` | `proj_6144_q5.scheduler.ncu-rep` |
| `out_6144_q5` | `Out5120x6144` | Q5 | `out_6144_q5.ncu-rep` | `out_6144_q5.scheduler.ncu-rep` |
| `lm_head_q6` | `LmHead248320x5120` | Q6 | `lm_head_q6.ncu-rep` | `lm_head_q6.scheduler.ncu-rep` |

Each report also has `.ncu.csv`, `.ncu.txt`, `.ncu.stdout.txt`, and
`.ncu.stderr.txt` sidecars.

## Cold-cache operator bench

These are normal `qus_linear_op_bench` runs outside NCU replay.

| tag | cold median us | achieved GB/s | achieved DRAM % | roofline us | stream ceiling GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mlp_gate_up_34816_q4` | `100.288` | `945.072` | `62.485` | `62.665` | `1512.480` |
| `mlp_down_q5` | `66.880` | `875.238` | `57.915` | `38.733` | `1511.260` |
| `proj_6144_q5` | `35.776` | `577.660` | `38.263` | `13.689` | `1509.690` |
| `out_6144_q5` | `36.096` | `572.539` | `37.849` | `13.662` | `1512.690` |
| `lm_head_q6` | `759.136` | `1309.100` | `86.748` | `658.538` | `1509.080` |

## NCU SOL and memory metrics

One launch per target, `--set detailed`, application replay.

| tag | NCU duration us | SM throughput % | memory throughput % | DRAM throughput % | L1/TEX % | L2 % | memory workload throughput | mem pipes busy % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_gate_up_34816_q4` | `91.01` | `61.66` | `75.60` | `75.60` | `64.25` | `38.97` | `1.33 Tbyte/s` | `61.66` |
| `mlp_down_q5` | `73.31` | `84.29` | `84.29` | `45.37` | `86.60` | `28.66` | `800.34 Gbyte/s` | `84.29` |
| `proj_6144_q5` | `33.12` | `61.94` | `61.94` | `38.20` | `71.85` | `22.35` | `673.93 Gbyte/s` | `61.94` |
| `out_6144_q5` | `33.98` | `61.01` | `61.01` | `35.42` | `68.41` | `21.76` | `624.10 Gbyte/s` | `61.01` |
| `lm_head_q6` | `919.04` | `91.73` | `91.73` | `62.90` | `92.30` | `38.77` | `1.11 Tbyte/s` | `91.73` |

Notes tied to the raw numbers:

- `mlp_gate_up_34816_q4` is the only sampled top kernel where NCU reports DRAM
  throughput equal to the top memory metric (`75.60%`). It is the closest of the
  top decode kernels to a direct DRAM-throughput wall.
- `mlp_down_q5`, `proj_6144_q5`, `out_6144_q5`, and `lm_head_q6` report memory
  throughput/SM pressure higher than DRAM throughput. Their immediate limiter in
  this capture is not raw DRAM bandwidth alone.
- `lm_head_q6` has the highest SM and memory-pipe pressure among the sampled
  kernels (`91.73%`) but only `62.90%` DRAM throughput.

## Launch and occupancy

| tag | block size | grid size | registers/thread | waves/SM | theoretical occupancy % | achieved occupancy % | active warps/SM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_gate_up_34816_q4` | `256` | `4352` | `40` | `4.27` | `100` | `86.23` | `41.39` |
| `mlp_down_q5` | `128` | `10240` | `39` | `5.02` | `100` | `90.39` | `43.39` |
| `proj_6144_q5` | `128` | `1536` | `38` | `0.75` | `100` | `69.20` | `33.22` |
| `out_6144_q5` | `128` | `1280` | `32` | `0.63` | `100` | `60.12` | `28.86` |
| `lm_head_q6` | `128` | `62080` | `26` | `30.43` | `100` | `96.19` | `46.17` |

The small standalone `proj_6144_q5` and `out_6144_q5` captures still show low
waves/SM (`0.75` and `0.63`) even though theoretical occupancy is `100%`. That
matches the earlier non-bandwidth-wall evidence for `out_6144` and explains why
these kernels can have low DRAM percentages while remaining expensive in the
full decode trace through repetition.

## Scheduler and warp-state metrics

One launch per target, `SchedulerStats` + `WarpStateStats`, application replay.

| tag | issued warp / scheduler | no eligible % | active warps / scheduler | eligible warps / scheduler | warp cycles / issued instruction | active threads / warp | not-predicated-off threads / warp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_gate_up_34816_q4` | `0.62` | `37.51` | `10.50` | `2.29` | `16.80` | `31.90` | `30.41` |
| `mlp_down_q5` | `0.70` | `29.78` | `10.86` | `2.85` | `15.47` | `24.64` | `23.19` |
| `proj_6144_q5` | `0.56` | `44.08` | `8.39` | `1.68` | `15.00` | `25.20` | `23.71` |
| `out_6144_q5` | `0.56` | `43.75` | `7.38` | `1.51` | `13.13` | `24.22` | `22.84` |
| `lm_head_q6` | `0.76` | `23.63` | `11.53` | `2.76` | `15.09` | `25.90` | `24.60` |

Scheduler observations:

- `proj_6144_q5` and `out_6144_q5` have the weakest scheduler eligibility in
  this sample (`44.08%` and `43.75%` no-eligible cycles) and low waves/SM.
- `lm_head_q6` has the best issued-warp rate of this group (`0.76`) and highest
  achieved occupancy (`96.19%`), but it remains a large decode contributor
  because it streams a much larger matrix.
- `mlp_gate_up_34816_q4` keeps nearly all lanes active (`31.90` active
  threads/warp, `30.41` not-predicated-off), unlike Q5/Q6 kernels whose active
  thread counts are closer to `24-26`.

## Source counters

| tag | branch instruction ratio % | branch instructions | branch efficiency % | avg divergent branches |
| --- | ---: | ---: | ---: | ---: |
| `mlp_gate_up_34816_q4` | `0.00` | `313,344` | `100` | `0` |
| `mlp_down_q5` | `0.15` | `9,338,880` | `48.46` | `4035.76` |
| `proj_6144_q5` | `0.15` | `3,127,296` | `45.21` | `1445.65` |
| `out_6144_q5` | `0.14` | `3,128,320` | `45.45` | `1445.65` |
| `lm_head_q6` | `0.14` | `129,871,360` | `42.86` | `58428.24` |

The Q5/Q6 kernels show low branch efficiency in the source-counter section, but
the branch instruction ratio is only about `0.14-0.15%`. This is useful context:
branch divergence exists, but these samples do not by themselves prove branch
instructions dominate total runtime.

## Readout

- The current top decode kernel, `mlp_gate_up_34816_q4`, is the clearest
  DRAM-throughput-limited sample among the top five: `75.60%` DRAM throughput,
  `1.33 Tbyte/s` memory workload throughput, high lane utilization, and no
  branch divergence.
- `mlp_down_q5` is expensive but the NCU sample points to a mixed L1/TEX/SM or
  memory-pipe wall: `84.29%` memory/SM pressure, `86.60%` L1/TEX, but only
  `45.37%` DRAM throughput.
- `proj_6144_q5` and `out_6144_q5` remain underfilled standalone kernels with
  low waves/SM and high no-eligible scheduler percentages. Their DRAM throughput
  is low (`38.20%`, `35.42%`) while L1/TEX/SM pressure is around `61-72%`.
- `lm_head_q6` is not the top full-decode kernel by total time, but per-launch it
  is the heaviest sampled kernel. It reaches `91.73%` SM/memory pressure and
  `96.19%` achieved occupancy while DRAM is `62.90%`.

This report is descriptive. It intentionally does not prescribe the next
optimization direction; any next tuning loop should form its own hypothesis from
these NCU artifacts and any additional source-line or replay-specific data it
needs.
