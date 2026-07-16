# Qwen3.6-35B Linear Op Qualification

Date: 2026-07-17

Target: Qwen3.6-35B-A3B Linear rows L8-L13 on NVIDIA GeForce RTX 5090 (`sm_120a`,
CUDA 13.1, driver 591.86, NCU 2025.4.1).

This is retained standalone Op evidence. It does not register the 35B Engine target. Qualification
is being completed in Vision execution order; this revision establishes L8 and L9.

## L8: Q6 Vision patch projection `[1152,1536]`

The exact operation is Q6G64_F16S RowSplit `[1152,1536]` times BF16 `[1536,P]`, producing BF16
`[1152,P]`. The complete target domain is every admitted four-column patch count through maximum
video/image `P=49152/65536`.

### Route qualification

The existing kernels were numerically correct, but the retained production route was not the
cold-cache winner at two intervals:

- at `P=40`, production selected MMA64 at 40.16 us while the existing SIMT candidate measured
  21.76 us;
- at `P=772`, production selected MMA64 at a three-process median of 44.256 us while MMA128
  measured 43.008 us.

Matched Release-build candidate sweeps produced the final closed route:

```text
P=4..96       SIMT R8C4
P=100..704    MMA R64C64
P=708..828    MMA R64C128
P=832         MMA R64C64
P=836..896    MMA R64C128
P=900..960    MMA R64C64
P=964..1024   MMA R64C128
P=1028..1088  MMA R64C64
P>=1092       MMA R64C128
```

The first crossover was confirmed with three independent processes, eight warmups, and forty
cold samples per process:

| P | SIMT | MMA64 | Decision |
|---:|---:|---:|---|
| 92 | 38.144 us | 40.192 us | SIMT |
| 96 | 40.160 us | 40.128 us | practical tie; retain simpler SIMT route |
| 100 | 41.952 us | 40.160 us | MMA64 |

For `P=772..828`, MMA128 won every screened four-column point; `P=832` returned to MMA64.

### Correctness and dispatch

```bash
cmake --build build -j --target \
  ninfer_linear_op_bench ninfer_linear_test \
  ninfer_q6_linear_candidate_test ninfer_q6_linear_plan_test \
  ninfer_q6_linear_dispatch_test
ctest --test-dir build \
  -R '^ninfer_(linear_test|q6_linear_candidate_test|q6_linear_plan_test|q6_linear_dispatch_test)$' \
  --output-on-failure
```

All four tests passed. The candidate suite retains the independent FP64 oracle, the plan test scans
the complete admitted column set, and the dispatch test checks the revised route boundaries against
their fixed kernel identities.

### Release timing and roofline

The final production `auto` sweep used eight warmups, forty cold samples, a 256 MiB L2 flush before
each sample, and same-session probes of 1503.806 GB/s cold-copy bandwidth and 209.640 BF16
MMA TFLOP/s.

| P | Route | Cold median | Useful/executed TFLOP/s | MMA probe |
|---:|---|---:|---:|---:|
| 4 | SIMT R8C4 | 7.424 us | 1.91 useful | fixed launch/ownership floor |
| 96 | SIMT R8C4 | 40.192 us | 8.45 useful | crossover winner |
| 100 | MMA R64C64 predicated | 40.160 us | 11.28 executed | crossover winner |
| 704 | MMA R64C64 full | 42.208 us | 59.03 | 28.16% |
| 4096 | MMA R64C128 full | 87.296 us | 166.05 | 79.21% |
| 49152 | MMA R64C128 full | 881.984 us | 197.22 | 94.08% |
| 65536 | MMA R64C128 full | 1171.456 us | 197.98 | 94.44% |

The maximum video and image cases are compute/tensor-pipe bound and reach the same-session measured
MMA roofline. Smaller cases are judged by matched candidate latency and launch-wave geometry rather
than a false full-device percentage.

### NCU attribution

Basic-first captures matched the intended demangled kernel substrings with
`--launch-skip 0 --launch-count 1`. Detailed captures then checked memory behavior and spilling.
Profiler durations are diagnostic and are not substituted for the CUDA-event medians above.

| Topology | Representative | NCU duration | SM SOL | Memory SOL | Occupancy | Registers | Static shared | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIMT R8C4 | P=96 | 48.83 us | 53.75% | 32.92% | 32.07% | 84 | 12.80 KiB | 10.16 |
| MMA R64C64 | P=704 | 53.41 us | 27.16% | 23.85% | 9.73% | 102 | 30.98 KiB | 0.39 |
| MMA R64C128 | P=65536 | 1.62 ms | 87.55% | 54.13% | 16.59% | 154 | 47.62 KiB | 27.11 |

The MMA64 representative is explicitly partial-wave limited: its grid contains 198 CTAs, or only
0.39 full device waves. The saturated MMA128 representative is tensor-pipe limited. All three
detailed captures report zero local-memory and shared-memory spilling requests.

Reports are retained locally under:

```text
profiles/bench/qwen3_6_35b_linear_qualification/l8/
profiles/ncu/qwen3_6_35b_a3b/linear/l8/final/
```

## Retained result

L8 is supported for its complete Qwen3.6-35B-A3B target domain. The exact route selects the
lowest-latency existing topology around every measured crossover, reaches 94.44% of the measured
BF16 MMA ceiling at maximum image size, and has no profiler-visible spilling.

## L9: Q4 Vision QKV projection `[3456,1152]`

The exact operation is Q4G64_F16S RowSplit `[3456,1152]` times BF16 `[1152,P]`, producing the
packed BF16 Q/K/V parent `[3456,P]`. The complete target domain is every admitted four-column
patch count through maximum video/image `P=49152/65536`.

### Route qualification

Matched Release-build sweeps compared every physically relevant existing topology:

- SIMT R8C4 and R8C8 over the complete small-P crossover;
- MMA R64C64 and R64C128 at both route seams and representative full/tail waves;
- both MMA schedules at the maximum video and image sizes.

SIMT R8C8 never won. The retained production route was already closed around the measured winners:

```text
P=4..36     SIMT R8C4
P=40..320   MMA R64C64
P>=324      MMA R64C128
```

At `P=40`, SIMT R8C4 and MMA64 both measured 29.952 us; at `P=44`, MMA measured
29.984 us versus 32.000 us for SIMT. MMA64 and MMA128 remained within 0.6% over every screened
four-column point from `P=324..384`; the existing MMA128 route was retained because it becomes the
clear winner as the column wave grows. No production route change was required.

### Correctness and dispatch

```bash
ctest --test-dir build \
  -R '^ninfer_(linear_test|q4_linear_candidate_test|q4_linear_plan_test|q4_linear_dispatch_test)$' \
  --output-on-failure
```

All four tests passed. They retain the independent Q4 numerical oracle, exact support/route
closure, fixed-candidate legality, and public-to-fixed BF16 identity checks.

### Release timing and roofline

Route seams used three independent processes with eight warmups and forty cold samples. The maximum
video and image rows below are isolated one-point, three-process medians so one large launch cannot
alter the following point's boost state.

| P | Route | Cold median | Executed TFLOP/s | Interpretation |
|---:|---|---:|---:|---|
| 4 | SIMT R8C4 | 9.216 us | 3.46 useful | fixed launch/ownership floor |
| 36 | SIMT R8C4 | 27.904 us | 10.27 useful | small-P SIMT winner |
| 40 | MMA R64C64 predicated | 29.952 us | 17.01 | practical crossover tie |
| 320 | MMA R64C64 full | 30.880 us | 82.51 | partial-device wave winner |
| 324 | MMA R64C128 predicated | 33.760 us | 90.57 | practical MMA64/MMA128 tie |
| 4096 | MMA R64C128 full | 185.344 us | 175.97 | compute-bound |
| 49152 | MMA R64C128 full | 1854.750 us | 211.01 | compute-bound maximum video |
| 65536 | MMA R64C128 full | 2830.300 us | 184.38 | compute-bound maximum image |

The raw executed rate differs between the two maximum item sizes because the longer image launch
settles at a lower boost/power state. NCU normalizes the result against the active device state and
shows both cases at the same tensor-pipe roofline.

### NCU attribution

Basic captures matched the intended Q4 SIMT/MMA kernel substrings. Detailed captures for each
distinct topology then checked the memory path and spilling.

| Topology | Representative | NCU duration | SM SOL | Memory SOL | Occupancy | Registers | Static shared | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIMT R8C4 | P=36 | 34.30 us | 64.22% | 40.84% | 46.64% | 78 | 8.70 KiB | 7.62 |
| MMA R64C64 | P=320 | 40.67 us | 36.76% | 27.24% | 13.15% | 98 | 28.93 KiB | 0.53 |
| MMA R64C128 | P=49152 | 3.87 ms | 87.36% | 50.32% | 16.59% | 150 | 45.57 KiB | 60.99 |
| MMA R64C128 | P=65536 | 5.31 ms | 86.68% | 50.69% | 16.59% | 150 | 45.57 KiB | 81.32 |

MMA64 is explicitly limited by its 0.53-wave grid. Both maximum-item captures are tensor-pipe
limited; the detailed maximum-image capture reaches 91.01% Compute SOL. Every detailed topology
reports zero local-memory and shared-memory spilling requests.

Reports are retained locally under:

```text
profiles/bench/qwen3_6_35b_linear_qualification/l9/
profiles/ncu/qwen3_6_35b_a3b/linear/l9/final/
```

## L9 retained result

L9 is supported for its complete Qwen3.6-35B-A3B target domain. Its existing exact route is the
matched candidate winner, both maximum item sizes are tensor-pipe roofline kernels, and no captured
topology spills.
