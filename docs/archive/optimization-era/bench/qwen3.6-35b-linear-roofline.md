# Qwen3.6-35B Linear Op Qualification

Date: 2026-07-17

Target: Qwen3.6-35B-A3B exact Linear domains on NVIDIA GeForce RTX 5090 (`sm_120a`,
CUDA 13.1, driver 591.86, NCU 2025.4.1).

This is retained standalone Op evidence. It does not register the 35B Engine target. This revision
establishes L5 and L7-L14.

## L5: W8 MTP stem `[2048,4096]`

The exact operation is W8G32_F16S RowSplit `[2048,4096]` times BF16 `[4096,T]`, producing the
plain BF16 MTP-stem output `[2048,T]` for every Text chunk size `1<=T<=1024`.

### Route qualification

Matched Release-build sweeps screened all four existing W8 schedules. SIMT R8C8 never won.
Three independent processes around both crossovers established the following finite route:

```text
T=1..56      SIMT R8C4
T=57..895    MMA R32C128
T=896..1024  MMA R64C128
```

The first seam measured 62.496 us versus 62.720 us for SIMT/MMA32 at `T=56`, then
64.800 us versus 62.688 us at `T=57`. Around the second seam, MMA32 and MMA64 were practical
ties at `T=888/895`; at `T=896`, MMA64 measured 124.128 us, and at `T=897` it avoided the
MMA32 tail-wave jump from about 128 us to 145 us.

### Correctness and dispatch

```bash
cmake --build build -j --target \
  ninfer_linear_op_bench ninfer_linear_test \
  ninfer_w8_linear_plan_test ninfer_w8_linear_dispatch_test
ctest --test-dir build \
  -R '^ninfer_(linear_test|w8_linear_plan_test|w8_linear_dispatch_test)$' \
  --output-on-failure
```

All three tests passed. The independent FP64 oracle checks both sides of the SIMT/MMA seam at
`T=56/57`, while a distributed row/column sample checks the complete maximum `T=1024` problem.
The measured relative L2 errors were `1.654e-3`, `2.156e-3`, and `2.689e-3`, respectively.
Plan and dispatch tests close the complete admitted interval and verify every route boundary
against its fixed kernel identity.

### Release timing and launch-aware roofline

Production `auto` used eight warmups, forty L2-flushed samples, and isolated processes at the
reported points:

| T | Route | Cold median | Executed TFLOP/s |
|---:|---|---:|---:|
| 1 | SIMT R8C4 | 17.696 us | fixed launch/ownership floor |
| 56 | SIMT R8C4 | 62.464 us | 15.04 useful |
| 57 | MMA R32C128 | 62.720 us | crossover winner |
| 512 | MMA R32C128 | 71.392 us | 120.32 |
| 895 | MMA R32C128 | 128.768 us | 116.74 |
| 896 | MMA R64C128 | 124.128 us | 121.10 |
| 1024 | MMA R64C128 | 124.064 us | 138.48 |

At `T=1024`, the fixed 2048-row MMA64 grid contains only 256 CTAs, or 0.75 full device waves, so
a full-device dense-MMA percentage is not the relevant ceiling. The already qualified L14
`[2048,4608]` problem at the same `T=1024` uses the same W8 format, output rows, grid, schedule,
and resource envelope while executing 12.5% more K work. Three alternating independent processes
measured 124.064 us and 138.48 TFLOP/s for L5 versus 138.496 us and 139.55 TFLOP/s for that
control. L5 is therefore only 0.77% below the matched launch-aware throughput ceiling.

### NCU attribution

Basic-first captures matched the intended SIMT and MMA kernel substrings. Detailed follow-ups
reported:

| Topology | Representative | NCU duration | SM SOL | Memory SOL | Occupancy | Registers | Static shared | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIMT R8C4 | T=56 | 72.58 us | 63.51% | 54.88% | 62.82% | 64 | 17.41 KiB | 5.27 |
| MMA R32C128 | T=512 | 97.50 us | 52.40% | 57.47% | 25.10% | 83 | 39.42 KiB | 0.75 |
| MMA R64C128 | T=1024 | 182.37 us | 55.63% | 34.86% | 25.45% | 119 | 46.08 KiB | 0.75 |

The L14 control's basic capture has the same 256-CTA grid, 0.75 waves/SM, 119 registers/thread,
46.08 KiB static shared memory, and 25.44% achieved occupancy as the L5 MMA64 maximum. All three
detailed L5 captures report zero local-memory and shared-memory spilling requests. NCU durations
are diagnostic counter-replay measurements and are not substituted for the CUDA-event medians.

Reports are retained locally under:

```text
profiles/bench/qwen3_6_35b_linear_qualification/l5/
profiles/ncu/qwen3_6_35b_a3b/linear/l5/final/
```

## L5 retained result

L5 is supported for its complete Qwen3.6-35B-A3B target domain. Its finite route selects the
measured crossover winners, the maximum remains within 0.77% of a same-topology launch-aware
ceiling, and all production topologies are profiler-confirmed without spilling.

## L7: Q4 draft head `[131072,2048]`

The exact operation is Q4G64_F16S RowSplit `[131072,2048]` times BF16 `[2048,1]`, producing the
single BF16 shortlist-logit column `[131072,1]`.

### Route qualification

Every physically legal existing topology was screened in a Release build. The runtime-K
`GemvR4W1Direct` path was the clear winner; the static `GemvR1W8Direct` path is intentionally legal
only at K=5120.

| Candidate | Cold median |
|---|---:|
| GEMV R4W1 direct | 105.664 us |
| SIMT R8C4 predicated | 109.856 us |
| SIMT R8C8 predicated | 161.536 us |
| MMA R64C64 predicated | 288.032 us |
| MMA R64C128 predicated | 357.600 us |

The production registry therefore admits only `C=1` and maps it directly to GEMV R4W1.

### Correctness and dispatch

```bash
cmake --build build -j --target \
  ninfer_linear_op_bench ninfer_linear_test \
  ninfer_q4_linear_candidate_test ninfer_q4_linear_plan_test \
  ninfer_q4_linear_dispatch_test
ctest --test-dir build \
  -R '^ninfer_(linear_test|q4_linear_candidate_test|q4_linear_plan_test|q4_linear_dispatch_test)$' \
  --output-on-failure
```

All four tests passed. The main Linear suite includes an independent FP64 oracle for the exact
`[131072,2048]` problem; the plan and dispatch suites verify singleton admission, the fixed kernel
identity, and public-to-fixed BF16 equality.

### Release timing and roofline

Three independent production processes, each with eight warmups and forty L2-flushed samples,
measured a 105.760 us median and 1350.91 GB/s useful payload bandwidth. The corresponding
same-session cold-copy ceiling median was 1508.06 GB/s, so production reaches 89.58% of the
measured traffic roofline.

### NCU attribution

The basic capture matched the intended `q4_rowsplit_gemv_kernel` specialization. A detailed
follow-up reported:

| NCU duration | SM SOL | DRAM SOL | Occupancy | Registers | Static shared | Waves/SM |
|---:|---:|---:|---:|---:|---:|---:|
| 114.43 us | 56.41% | 73.88% | 75.09% | 44 | 2.18 KiB | 19.28 |

NCU classifies the kernel as memory-bound and reports 1.32 TB/s memory throughput, 5.19% L2 hit
rate, and zero local-memory or shared-memory spilling requests.

Reports are retained locally under:

```text
profiles/bench/qwen3_6_35b_linear_qualification/l7/
profiles/ncu/qwen3_6_35b_a3b/linear/l7/final/
```

## L7 retained result

L7 is supported for its complete Qwen3.6-35B-A3B target domain. The exact singleton route selects
the measured candidate winner, reaches 89.58% of the cold-copy roofline, and is profiler-confirmed
DRAM-bound without spilling.

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

## L10: Q5 Vision attention output `[1152,1152]`

The exact operation is Q5G64_F16S RowSplit `[1152,1152]` times BF16 `[1152,P]`, producing BF16
`[1152,P]`. The complete target domain is every admitted four-column patch count through maximum
video/image `P=49152/65536`.

### Route correction

The retained exact route was correct but substantially slower than existing legal candidates:

- `P=8`: production C8 measured 15.616 us versus 9.472 us for C4;
- `P=56`: production C8 measured 28.672 us versus 21.760 us for C4;
- `P=60`: production MMA128 measured 32.000 us versus 23.840 us for C4;
- full 64/128-column waves through `P=1088` alternated between MMA64 and MMA128 winners.

SIMT split2/split4 and the Q5 GEMV are not legal for this exact problem. Matched candidate screens
and three-process confirmations produced the final closed route:

```text
P=4..76       SIMT R8C4
P=80..636     MMA R64C64
P=640..700    MMA R64C128
P=704         MMA R64C64
P=708..828    MMA R64C128
P=832         MMA R64C64
P=836..896    MMA R64C128
P=900..960    MMA R64C64
P=964..1024   MMA R64C128
P=1028..1088  MMA R64C64
P>=1092       MMA R64C128
```

Representative three-process, eight-warmup, forty-cold-sample confirmations were:

| P | C4 | MMA64 | MMA128 | Decision |
|---:|---:|---:|---:|---|
| 76 | 27.904 us | 29.952 us | 32.000 us | C4 |
| 80 | 30.752 us | 29.952 us | 32.000 us | MMA64 |
| 640 | — | 31.968 us | 29.984 us | MMA128 |
| 704 | — | 32.000 us | 33.920 us | MMA64 |
| 768 | — | 32.000 us | 30.752 us | MMA128 |
| 832 | — | 32.000 us | 34.016 us | MMA64 |
| 896 | — | 33.568 us | 31.840 us | MMA128 |
| 960 | — | 32.032 us | 33.952 us | MMA64 |
| 1024 | — | 32.736 us | 32.000 us | MMA128 |
| 1088 | — | 33.376 us | 34.048 us | MMA64 |
| 1152 | — | 33.888 us | 32.000 us | MMA128 |

### Correctness and dispatch

```bash
cmake --build build -j --target \
  ninfer_linear_op_bench ninfer_linear_test \
  ninfer_q5_linear_candidate_test ninfer_q5_linear_plan_test
ctest --test-dir build \
  -R '^ninfer_(linear_test|q5_linear_candidate_test|q5_linear_plan_test)$' \
  --output-on-failure
```

All three tests passed. The Q5 plan test scans the complete admitted Vision column domain and now
checks every revised route boundary. The numerical suites retain independent fixed-candidate
oracles and public exact-shape coverage.

### Release timing and roofline

| P | Route | Cold median | Executed TFLOP/s | Interpretation |
|---:|---|---:|---:|---|
| 4 | SIMT R8C4 | 9.504 us | 1.12 useful | fixed launch/ownership floor |
| 76 | SIMT R8C4 | 27.904 us | 7.23 useful | small-P winner |
| 80 | MMA R64C64 predicated | 29.952 us | 11.34 | first MMA winner |
| 640 | MMA R64C128 full | 30.464 us | 55.76 | full-wave winner |
| 704 | MMA R64C64 full | 32.256 us | 57.93 | full-wave winner |
| 1024 | MMA R64C128 full | 31.936 us | 85.10 | full-wave winner |
| 1088 | MMA R64C64 full | 33.568 us | 86.03 | full-wave winner |
| 1280 | MMA R64C128 full | 34.048 us | 99.78 | stable large-P route |
| 4096 | MMA R64C128 full | 66.816 us | 162.71 | compute-bound |
| 49152 | MMA R64C128 full | 671.008 us | 194.42 | maximum video |
| 65536 | MMA R64C128 full | 892.928 us | 194.80 | maximum image |

The maximum rows are isolated one-point, three-process medians. Their median ratios to the
same-process BF16 MMA probe are 94.76% and 93.68%, respectively.

### NCU attribution

| Topology | Representative | NCU duration | SM SOL | Memory SOL | Occupancy | Registers | Static shared | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIMT R8C4 | P=76 | 32.70 us | 49.74% | 34.69% | 63.55% | 58 | 10.75 KiB | 4.02 |
| MMA R64C64 | P=960 | 42.75 us | 35.39% | 30.71% | 13.13% | 102 | 29.95 KiB | 0.53 |
| MMA R64C128 | P=49152 | 925.18 us | 86.59% | 54.38% | 16.56% | 154 | 46.59 KiB | 20.33 |
| MMA R64C128 | P=65536 | 1.23 ms | 87.00% | 52.21% | 16.58% | 154 | 46.59 KiB | 27.11 |

MMA64 is explicitly partial-wave limited. Both maximum-item captures are tensor-pipe limited.
Detailed captures for all three topologies report zero local-memory and shared-memory spilling
requests.

Reports are retained locally under:

```text
profiles/bench/qwen3_6_35b_linear_qualification/l10/
profiles/ncu/qwen3_6_35b_a3b/linear/l10/final/
```

## L10 retained result

L10 is supported for its complete Qwen3.6-35B-A3B target domain. The corrected exact route removes
large small-P regressions, selects the measured 64/128-column wave winner, and reaches at least
86.59% Compute SOL at both maximum item sizes without spilling.

## L11: Q4 Vision MLP fc1 `[4304,1152]`

The exact operation is Q4G64_F16S RowSplit `[4304,1152]` times BF16 `[1152,P]`, producing BF16
`[4304,P]`. Every MMA launch uses the predicated row variant because 4304 is not divisible by the
64-row tile. The complete target domain is every admitted four-column patch count through maximum
video/image `P=49152/65536`.

### Route qualification

Matched screens compared SIMT R8C4/R8C8 and MMA R64C64/R64C128 across every existing seam. The
retained non-monotonic route was already the stable candidate decision:

```text
P=4          SIMT R8C4
P=8          SIMT R8C8
P=12         SIMT R8C4
P=16..24     SIMT R8C8
P=28..320    MMA R64C64
P>=324       MMA R64C128
```

Three-process confirmation established:

| P | MMA64 | MMA128 | Decision |
|---:|---:|---:|---|
| 40 | 29.984 us | 30.016 us | practical tie; retain MMA64 |
| 68 | 29.696 us | 32.000 us | MMA64 |
| 192 | 32.768 us | 32.224 us | 1.7% isolated difference; retain the stable interval |
| 256 | 34.016 us | 33.952 us | practical tie |
| 320 | 34.048 us | 36.096 us | MMA64 |
| 324 | 36.096 us | 36.096 us | tie; begin the long-term MMA128 winner |

No route change was required.

### Correctness and dispatch

```bash
ctest --test-dir build \
  -R '^ninfer_(linear_test|q4_linear_candidate_test|q4_linear_plan_test|q4_linear_dispatch_test)$' \
  --output-on-failure
```

All four tests passed. The retained suites cover the independent Q4 oracle, the K=1152 checked
group stage, the Rows=4304 edge, exact plan closure, and public-to-fixed dispatch identity.

### Release timing and matched row-edge control

The maximum rows use a measurement-only `[4352,1152]` Full variant as a same-work control. Both
production and control execute the same padded 4352 rows, K, columns, and MMA tile count; the
remaining timing difference therefore isolates production's 4304-row predication.

| P | Production predicated | Full-row control | Executed TFLOP/s | Edge overhead |
|---:|---:|---:|---:|---:|
| 49152 | 2786.240 us | 2700.930 us | 176.89 | 3.16% |
| 65536 | 3690.460 us | 3561.470 us | 178.06 | 3.62% |

Each value is the median of three independent processes with eight warmups and forty cold samples.
The production route is therefore within 3.7% of the same-grid, same-executed-work Full ceiling.

Representative smaller production points were:

| P | Route | Cold median | Executed TFLOP/s |
|---:|---|---:|---:|
| 4 | SIMT R8C4 | 11.520 us | 3.44 useful |
| 8 | SIMT R8C8 | 13.568 us | 5.85 useful |
| 12 | SIMT R8C4 | 17.888 us | 6.65 useful |
| 24 | SIMT R8C8 | 27.904 us | 8.53 useful |
| 320 | MMA R64C64 predicated | 34.048 us | 94.24 |
| 324 | MMA R64C128 predicated | 35.968 us | 107.05 |
| 4096 | MMA R64C128 predicated | 228.512 us | 179.73 |

### NCU attribution

| Topology | Representative | NCU duration | SM SOL | Memory SOL | Occupancy | Registers | Static shared | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIMT R8C4 | P=12 | 19.74 us | 46.25% | 29.41% | 47.29% | 78 | 8.70 KiB | 3.16 |
| SIMT R8C8 | P=24 | 31.84 us | 45.86% | 28.83% | 32.24% | 96 | 8.70 KiB | 4.75 |
| MMA R64C64 | P=320 | 42.50 us | 45.30% | 33.65% | 16.49% | 101 | 28.93 KiB | 0.67 |
| MMA R64C128 | P=49152 | 6.18 ms | 81.67% | 48.68% | 16.60% | 152 | 45.57 KiB | 76.80 |
| MMA R64C128 | P=65536 | 6.64 ms | 82.89% | 49.04% | 16.60% | 152 | 45.57 KiB | 102.40 |

The detailed maximum-image capture reports 84.22% Compute SOL with the tensor pipeline limiting
execution. Its global/shared sector warnings are bounded by the direct Full-row control above:
their complete observable cost is at most 3.62%, not the profiler's heuristic estimate. Every
detailed topology reports zero local-memory and shared-memory spilling requests.

Reports are retained locally under:

```text
profiles/bench/qwen3_6_35b_linear_qualification/l11/
profiles/ncu/qwen3_6_35b_a3b/linear/l11/final/
```

## L11 retained result

L11 is supported for its complete Qwen3.6-35B-A3B target domain. Its existing route selects the
matched candidate winner, the maximum cases lie within 3.7% of the same-executed-work Full ceiling,
and the saturated predicated kernel is tensor-pipe limited without spilling.

## L12: Q5 Vision MLP fc2 `[1152,4304]`, Kpad=4352

The exact operation is Q5G64_F16S RowSplit `[1152,4304]` times BF16 `[4304,P]`, producing BF16
`[1152,P]`. Registered storage pads K to 4352, so every MMA route uses the predicated K-tail
variant. The complete target domain is every admitted four-column patch count through maximum
video/image `P=49152/65536`.

### Route correction

The previous exact route selected C8 for `P=8..84` and MMA128 from `P=88`. Existing legal
candidates were substantially faster:

- `P=8`: C8 measured 38.112 us versus 21.760 us for C4;
- `P=88`: MMA128 measured 111.904 us versus 83.232 us for C4;
- MMA64 remained faster than or tied with MMA128 through approximately `P=1152`.

Q5 split2/split4 and GEMV are not legal for this problem. Candidate sweeps and repeated crossover
confirmation produced a simple final route:

```text
P=4..120      SIMT R8C4
P=124..1148   MMA R64C64
P>=1152       MMA R64C128
```

At `P=116`, C4 measured 101.664 us versus 107.776 us for MMA64. At `P=120`, the routes were a
practical tie at 107.680/107.776 us, so the simpler C4 route was retained. At `P=124`, MMA64 was
already the clear winner. Across six independent `P=1152` processes, MMA128 measured a median
114.288 us versus 115.600 us for MMA64; from `P=1156` its advantage widened.

### Correctness and dispatch

```bash
cmake --build build -j --target \
  ninfer_linear_op_bench ninfer_linear_test \
  ninfer_q5_linear_candidate_test ninfer_q5_linear_plan_test
ctest --test-dir build \
  -R '^ninfer_(linear_test|q5_linear_candidate_test|q5_linear_plan_test)$' \
  --output-on-failure
```

All three tests passed. The complete Q5 plan scan covers the revised boundaries, while the
numerical suite retains K=4304/Kpad=4352 public coverage and independent fixed-candidate oracles.

### Release timing and matched K-tail control

Representative final production points were:

| P | Route | Cold median | Executed TFLOP/s |
|---:|---|---:|---:|
| 4 | SIMT R8C4 | 19.712 us | 2.01 useful |
| 116 | SIMT R8C4 | 103.680 us | 11.09 useful |
| 120 | SIMT R8C4 | 107.808 us | 11.04 useful |
| 124 | MMA R64C64 predicated | 109.824 us | 11.56 |
| 1024 | MMA R64C64 predicated | 113.856 us | 89.19 |
| 1148 | MMA R64C64 predicated | 115.488 us | 98.92 |
| 1152 | MMA R64C128 predicated | 114.816 us | 99.50 |
| 4096 | MMA R64C128 predicated | 251.168 us | 161.72 |

Maximum-item qualification uses measurement-only `[1152,4352]` Full-K as a matched control.
Production and control have the same padded K, row/column tile count, and executed MMA work; the
timing difference isolates the logical K=4304 tail.

| P | Production K-tail | Full-K control | Production TFLOP/s | Tail overhead |
|---:|---:|---:|---:|---:|
| 49152 | 2964.770 us | 2791.870 us | 164.40 | 6.19% |
| 65536 | 3965.380 us | 3668.960 us | 163.89 | 8.08% |

These are three-process medians with eight warmups and forty cold samples. The production route
remains within 8.1% of the same-topology Full-K ceiling across both complete maximum item sizes.

### NCU attribution

| Topology | Representative | NCU duration | SM SOL | Memory SOL | Occupancy | Registers | Static shared | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIMT R8C4 | P=120 | 126.08 us | 62.23% | 44.72% | 63.48% | 58 | 10.75 KiB | 6.35 |
| MMA R64C64 | P=1148 | 150.62 us | 45.62% | 51.79% | 15.80% | 103 | 29.95 KiB | 0.64 |
| MMA R64C128 production | P=49152 | 3.63 ms | 81.93% | 60.99% | 16.62% | 154 | 46.59 KiB | 20.33 |
| MMA R64C128 production | P=65536 | 5.74 ms | 80.60% | 60.25% | 16.63% | 154 | 46.59 KiB | 27.11 |
| MMA R64C128 Full-K control | P=65536 | 5.81 ms | 89.33% | 56.42% | 16.62% | 154 | 46.59 KiB | 27.11 |

Detailed production and control captures both identify the tensor pipeline as the limiter. The
K-tail increases shared-memory wavefront overhead, but its complete timing cost is bounded by the
matched 8.08% result above. Every captured topology reports zero local-memory and shared-memory
spilling requests.

Reports are retained locally under:

```text
profiles/bench/qwen3_6_35b_linear_qualification/l12/
profiles/ncu/qwen3_6_35b_a3b/linear/l12/final/
```

## L12 retained result

L12 is supported for its complete Qwen3.6-35B-A3B target domain. The corrected route removes large
small-P regressions, the maximum K-tail cases remain within 8.1% of the same-topology Full-K
ceiling, and all final topologies are spill-free.

## L13: W8 Vision merger fc1 `[4608,4608]`

The exact operation is W8G32_F16S RowSplit `[4608,4608]` times BF16 `[4608,V]`, producing BF16
`[4608,V]`. The complete target domain is every merger column count through maximum video/image
`V=12288/16384`.

### Route correction

The retained route selected SIMT R8C8 only at `V=5` and MMA R64C128 from `V=6`. Matched
fixed-candidate sweeps found that R8C8 never won, R64C128 paid too much fixed cost below 257
columns, and the small-domain winner changed at two non-monotonic seams. The final closed route is:

```text
V=1..8       SIMT R8C4
V=9..11      MMA R32C128
V=12         SIMT R8C4
V=13..256    MMA R32C128
V>=257       MMA R64C128
```

The seams were confirmed with three independent processes, eight warmups, and forty cold samples
per process:

| V | SIMT R8C4 | MMA R32C128 | MMA R64C128 | Decision |
|---:|---:|---:|---:|---|
| 8 | 50.464 us | 72.512 us | - | SIMT |
| 9 | 83.232 us | 72.800 us | - | MMA32 |
| 11 | 87.040 us | 72.896 us | - | MMA32 |
| 12 | 60.704 us | 72.928 us | - | SIMT |
| 13 | 101.664 us | 72.736 us | - | MMA32 |
| 256 | - | 87.328 us | 93.376 us | MMA32 |
| 257 | - | 156.960 us | 142.624 us | MMA64 |

The adjacent `[5120,4608]` registered route was kept unchanged.

### Correctness and dispatch

```bash
cmake --build build -j --target \
  ninfer_linear_op_bench ninfer_linear_test \
  ninfer_w8_linear_plan_test ninfer_w8_linear_dispatch_test
ctest --test-dir build \
  -R '^ninfer_(linear_test|w8_linear_plan_test|w8_linear_dispatch_test)$' \
  --output-on-failure
```

All three tests passed. The W8 dispatch suite retains an independent FP64 oracle at representative
points for every final topology and verifies public-to-fixed BF16 identity. The plan suite checks
all revised seams, the complete admitted upper bound, and that the `[5120,4608]` route did not
change.

### Release timing and matched W8 ceiling

Representative final production points were:

| V | Route | Cold median | Useful TFLOP/s |
|---:|---|---:|---:|
| 1 | SIMT R8C4 predicated | 34.048 us | 1.25 useful |
| 8 | SIMT R8C4 full | 50.272 us | 6.76 useful |
| 9 | MMA R32C128 predicated | 70.912 us | 5.39 |
| 12 | SIMT R8C4 full | 60.704 us | 8.39 useful |
| 13 | MMA R32C128 predicated | 70.912 us | 7.79 |
| 256 | MMA R32C128 full | 85.280 us | 127.48 |
| 257 | MMA R64C128 predicated | 142.624 us | 76.52 |
| 1024 | MMA R64C128 full | 275.712 us | 157.72 |
| 4096 | MMA R64C128 full | 933.888 us | 186.26 |

W8 performs online group-scale decode before each MMA tile, so a dense BF16-only probe is not the
semantic roofline for this fused kernel. Maximum-item qualification therefore uses the existing
`[5120,4608]` W8 route as a matched format ceiling: it has the same K, storage format, decode
cadence, MMA schedule, and two-CTA-per-SM resource envelope.

| V | L13 cold median | L13 TFLOP/s | Matched-control TFLOP/s | Difference |
|---:|---:|---:|---:|---:|
| 12288 | 3110.880 us | 167.75 | 168.07 | -0.19% |
| 16384 | 4141.860 us | 167.99 | 164.35 | +2.21% |

These are independent three-process medians with eight warmups and forty cold samples. L13 tracks
or exceeds the same-format ceiling at both complete maximum item sizes.

### NCU attribution

Basic-first captures matched the intended W8 SIMT/MMA kernel substrings. Detailed captures then
checked pipeline balance and spilling for every distinct topology.

| Topology | Representative | NCU duration | SM SOL | Memory SOL | Occupancy | Registers | Static shared | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIMT R8C4 | V=8 | 57.86 us | 37.35% | 32.25% | 61.86% | 64 | 17.41 KiB | 1.69 |
| MMA R32C128 | V=256 | 116.70 us | 56.12% | 60.76% | 27.99% | 83 | 39.42 KiB | 0.85 |
| MMA R64C128 | V=12288 | 5.09 ms | 78.17% | 50.52% | 32.94% | 119 | 46.08 KiB | 20.33 |
| MMA R64C128 | V=16384 | 7.19 ms | 76.89% | 49.74% | 33.02% | 119 | 46.08 KiB | 27.11 |
| `[5120,4608]` matched control | V=16384 | 9.34 ms | 75.69% | 49.05% | 33.03% | 119 | 46.08 KiB | 30.12 |

The detailed large-MMA capture identifies Tensor as the highest-utilized pipeline. Achieved
occupancy is within 0.4 percentage points of the 33.33% register/shared-memory-limited theoretical
occupancy, and its dominant warp stall is waiting for the oversubscribed math pipeline. The exact
L13 maximum reaches higher Compute SOL than the matched control. Every detailed topology reports
zero local-memory and shared-memory spilling requests.

Reports are retained locally under:

```text
profiles/bench/qwen3_6_35b_linear_qualification/l13/
profiles/ncu/qwen3_6_35b_a3b/linear/l13/final/
```

## L13 retained result

L13 is supported for its complete Qwen3.6-35B-A3B target domain. The corrected route selects the
measured small-domain winner, both maximum item sizes attain the matched W8 online-decode/MMA
roofline, and all final topologies are spill-free.

## L14: W8 Vision merger fc2 `[2048,4608]`

The exact operation is W8G32_F16S RowSplit `[2048,4608]` times BF16 `[4608,V]`, producing visual
BF16 `[2048,V]`. The complete target domain is every merger column count through maximum
video/image `V=12288/16384`.

### Route qualification

Release candidate sweeps covered every existing W8 topology and then every `V=1..1024` point.
Three-process confirmations resolved the non-monotonic SIMT full-tile wins and the MMA32/MMA64
crossover. The final closed route is:

```text
V=1..14      SIMT R8C4
V=15         MMA R32C128
V=16         SIMT R8C4
V=17..19     MMA R32C128
V=20         SIMT R8C4
V=21..23     MMA R32C128
V=24         SIMT R8C4
V=25..27     MMA R32C128
V=28         SIMT R8C4
V=29..31     MMA R32C128
V=32         SIMT R8C4
V=33..871    MMA R32C128
V>=872       MMA R64C128
```

Representative three-process seam medians were:

| V | SIMT R8C4 | MMA R32C128 | MMA R64C128 | Decision |
|---:|---:|---:|---:|---|
| 14 | 64.800 us | 68.896 us | - | SIMT |
| 15 | 70.656 us | 68.864 us | - | MMA32 |
| 16 | 46.304 us | 68.864 us | - | SIMT |
| 20 | 46.336 us | 68.896 us | - | SIMT |
| 24 | 56.288 us | 68.896 us | - | SIMT |
| 28 | 62.400 us | 68.896 us | - | SIMT |
| 32 | 64.800 us | 68.864 us | - | SIMT |
| 33 | 111.904 us | 68.896 us | - | MMA32 |
| 800 | - | 140.544 us | 143.360 us | MMA32 |
| 864 | - | 144.256 us | 144.640 us | practical tie; retain MMA32 |
| 872 | - | 144.640 us | 144.640 us | tie; switch to the forward MMA64 winner |

### Correctness and dispatch

```bash
cmake --build build -j --target \
  ninfer_linear_op_bench ninfer_linear_test \
  ninfer_w8_linear_plan_test ninfer_w8_linear_dispatch_test
ctest --test-dir build \
  -R '^ninfer_(linear_test|w8_linear_plan_test|w8_linear_dispatch_test)$' \
  --output-on-failure
```

All three tests passed. Exact FP64-oracle cases cover every small-route seam, and a sampled
independent oracle checks the maximum `V=16384` MMA64 result across distributed rows and columns.
The dispatch suite verifies every production route boundary against its fixed kernel identity.

### Release timing and roofline

Representative final production points were:

| V | Route | Cold median | Useful/executed TFLOP/s |
|---:|---|---:|---:|
| 1 | SIMT R8C4 predicated | 27.872 us | 0.68 useful |
| 14 | SIMT R8C4 predicated | 64.768 us | 4.08 useful |
| 15 | MMA R32C128 predicated | 68.832 us | 4.11 / 35.10 |
| 16 | SIMT R8C4 full | 45.952 us | 6.57 useful |
| 32 | SIMT R8C4 full | 64.800 us | 9.32 useful |
| 33 | MMA R32C128 predicated | 68.864 us | 9.04 / 35.08 |
| 512 | MMA R32C128 full | 79.104 us | 122.16 |
| 872 | MMA R64C128 predicated | 144.640 us | 113.79 / 116.92 |
| 4096 | MMA R64C128 full | 474.880 us | 162.80 |

Maximum-item rows are isolated three-process medians with eight warmups and forty cold samples:

| V | Cold median | Executed TFLOP/s | Same-session BF16 MMA probe |
|---:|---:|---:|---:|
| 12288 | 1256.736 us | 184.55 | 88.31% |
| 16384 | 1647.904 us | 187.66 | 91.42% |

Both complete maxima reach the compute roofline despite the fused W8 online decode work.

### NCU attribution

| Topology | Representative | NCU duration | SM SOL | Memory SOL | Occupancy | Registers | Static shared | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIMT R8C4 | V=16 | 46.05 us | 41.82% | 36.11% | 53.76% | 64 | 17.41 KiB | 1.51 |
| MMA R32C128 | V=512 | 111.78 us | 52.19% | 56.25% | 25.10% | 83 | 39.42 KiB | 0.75 |
| MMA R64C128 | V=16384 | 2.33 ms | 78.62% | 48.80% | 32.72% | 119 | 46.08 KiB | 12.05 |

The small topologies are explicitly launch-wave limited. At maximum image size, Tensor is the
highest-utilized pipeline and achieved occupancy is within 0.7 percentage points of the
register/shared-memory-limited 33.33% theoretical occupancy. Every detailed topology reports zero
local-memory and shared-memory spilling requests.

Reports are retained locally under:

```text
profiles/bench/qwen3_6_35b_linear_qualification/l14/
profiles/ncu/qwen3_6_35b_a3b/linear/l14/final/
```

## L14 retained result

L14 is supported for its complete Qwen3.6-35B-A3B target domain. The exact finite route preserves
the measured small-domain full-tile winners, both maximum item sizes reach the compute roofline,
and every final topology is spill-free.
