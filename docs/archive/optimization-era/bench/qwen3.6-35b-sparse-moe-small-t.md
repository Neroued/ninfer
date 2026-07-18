# Qwen3.6-35B SparseMoe Small-T Qualification

> Status: retained operator-level evidence for the repository-internal `SparseMoe` Small-T route.
> This does not register or qualify the complete 35B target.

## Scope and environment

The qualified Small-T domain is `T=2..8`, hidden size 2048, 256 routed experts, top-8
selection, one shared expert, intermediate size 512, and the `AddResidual` epilogue. The admitted
routed codec pairs are Q4+Q5, Q4+Q6, and W8+W8; the shared gate/up and down weights are W8. This
route deliberately uses CUDA Core/SIMT kernels only. It does not use tensor cores or grouped GEMM.

Measurements were collected on 2026-07-18 with an NVIDIA GeForce RTX 5090, driver 591.86, CUDA
13.1, Nsight Compute 2025.4.1, and the Release build. Each benchmark instance allocates the full
256-expert persistent address spans and flushes 256 MiB before every pipeline-cold sample. The
primary matrix uses 20 warmups, 200 measured repetitions, and three independent process runs while
no other GPU compute process is present.

The deterministic `trace-like` route generator is a realistic synthetic MTP control rather than a
captured model trace. Each token retains three experts from its predecessor, draws two from a
24-expert hot pool, and draws three from the full expert population. It produces 13 to 37 distinct
experts over `T=2..8`, with adjacent overlap three. `independent` and `same` controls bound the
sensitivity to expert locality. All tokens can select different experts; the benchmark never
assumes one route shared by the complete window.

Representative retained commands are:

```bash
./build/bench/ninfer_sparse_moe_bench --matrix --tokens 6 \
  --distribution trace-like --seed 20260718 --warmup 20 --repeat 200 \
  --csv-out profiles/bench/sparse-moe-small-t/final-run1-t6.csv

ncu --set basic --kernel-name regex:sparse_moe_small_t_s1_kernel \
  --launch-skip 20 --launch-count 1 \
  -o profiles/ncu/sparse-moe-small-t/partitioned-t8-s1-basic \
  ./build/bench/ninfer_sparse_moe_bench --scope d1 \
  --candidate small-t-token-loop --codec q4-q5 --tokens 8
```

## Accepted implementation

`T=1` retains the qualified four-launch decode route. `T=2..8` selects an exact-T template with
the following private stages:

1. S1 jointly projects all token columns. Each of 257 router rows is divided into four K
   partitions, giving 1028 128-thread CTAs. A router weight vector is loaded once and reused for
   every compile-time token accumulator. The output is FP32 `[T,257,4]` partial scores.
2. S2 uses one block with exactly T warps. It reduces the four partitions and independently
   performs lower-id-stable top-8 selection, selected-logit normalization, and shared sigmoid for
   every token.
3. S3 submits the qualified nine-path decode gate/up kernel once per token. Every token has its own
   FP32 `[9,512]` activation region, so no handoff aliases another token.
4. S4 submits the qualified nine-path down/merge/`AddResidual` kernel once per token.

The resulting topology has `2+2T` launches instead of the serial control's `4T`. This is a closed
Small-T route: token-local expert kernels are an intentional measured choice, not calls to public
sub-Ops. Workspace contains `8T` I32 ids, `8T` FP32 route weights, T FP32 shared scales, and one
FP32 scratch allocation sized as `[9T,512]`. S1 reuses the scratch prefix for its partial scores;
S3/S4 reuse it as disjoint token activation regions. There is no initialized workspace state,
hidden allocation, or graph-unstable address.

## Performance result

The table reports the median of three pipeline-cold medians in microseconds. `serial` is the exact
pre-Small-T implementation applied independently to each token on the same stream. Speedup is
`serial / Small-T`.

| Codec | T | Small-T | Serial | Speedup |
|---|---:|---:|---:|---:|
| Q4+Q5 | 2 | 63.456 | 69.600 | 1.097x |
| Q4+Q5 | 3 | 81.888 | 100.352 | 1.225x |
| Q4+Q5 | 4 | 100.320 | 131.040 | 1.306x |
| Q4+Q5 | 5 | 118.752 | 161.760 | 1.362x |
| Q4+Q5 | 6 | 138.752 | 199.904 | 1.441x |
| Q4+Q5 | 7 | 159.296 | 238.560 | 1.498x |
| Q4+Q5 | 8 | 179.744 | 276.448 | 1.538x |
| Q4+Q6 | 2 | 63.456 | 69.632 | 1.097x |
| Q4+Q6 | 3 | 81.888 | 101.504 | 1.240x |
| Q4+Q6 | 4 | 100.352 | 131.072 | 1.306x |
| Q4+Q6 | 5 | 118.784 | 162.208 | 1.366x |
| Q4+Q6 | 6 | 139.232 | 194.560 | 1.397x |
| Q4+Q6 | 7 | 159.520 | 233.472 | 1.464x |
| Q4+Q6 | 8 | 181.600 | 277.504 | 1.528x |
| W8+W8 | 2 | 75.776 | 81.920 | 1.081x |
| W8+W8 | 3 | 104.480 | 124.928 | 1.196x |
| W8+W8 | 4 | 137.216 | 165.792 | 1.208x |
| W8+W8 | 5 | 161.792 | 202.720 | 1.253x |
| W8+W8 | 6 | 194.560 | 243.712 | 1.253x |
| W8+W8 | 7 | 223.232 | 280.576 | 1.257x |
| W8+W8 | 8 | 249.408 | 315.392 | 1.265x |

Production's maximum range across the three primary runs was 1.72%; most points varied by less
than 1.3%. Q4+Q5 stage medians show that the joint front remains constant while expert work scales
with T:

| T | S1 | S2 | S3 | S4 | Complete |
|---:|---:|---:|---:|---:|---:|
| 2 | 5.344 | 6.176 | 34.816 | 18.432 | 63.456 |
| 6 | 5.344 | 6.176 | 75.776 | 50.944 | 138.752 |
| 8 | 5.344 | 6.176 | 98.304 | 67.584 | 179.744 |

### Expert-distribution sensitivity

The following Q4+Q5 values are medians across seeds 20260718, 20260719, and 20260720. The complete
route stays valid for every distribution; the timing variation is the expected effect of selected
expert weight reuse in L2 rather than a route-shape specialization.

| Distribution | T2 | T3 | T4 | T5 | T6 | T7 | T8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trace-like | 63.456 | 81.920 | 102.368 | 120.832 | 140.768 | 159.744 | 180.224 |
| independent | 65.504 | 85.984 | 108.512 | 133.088 | 157.664 | 178.176 | 204.640 |
| same | 55.296 | 73.696 | 94.176 | 112.608 | 131.040 | 149.472 | 167.936 |

### T=1 non-regression control

The extended benchmark preserves the decode report's fixed expert set
`{0,32,64,96,128,160,224,255}` at `T=1`. Three interleaved same-session runs compared the
unchanged base-worktree binary with the Small-T worktree binary:

| Codec | Base binary | Small-T worktree |
|---|---:|---:|
| Q4+Q5 | 40.928 | 40.928 |
| Q4+Q6 | 42.976 | 42.976 |
| W8+W8 | 49.120 | 48.960 |

An older 38.88 us Q4+Q5 observation was not used as the non-regression control because the current
base binary also measured 40.928 us in the same GPU session. The interleaved comparison isolates
the code change from session-level clock and system variation.

## Nsight Compute attribution

Basic-set reports were captured only after the benchmark identified a live candidate. Durations
below are profiler-instrumented and are used for attribution, not substituted for event timing.

| Kernel/candidate | Duration | Registers | Static shared | Waves/SM | Theoretical occupancy | Achieved occupancy | Memory SOL | DRAM SOL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial joint S1, T8 | 5.82 us | 43 | 256 B | 0.30 | 83.33% | 24.88% | 21.84% | 11.27% |
| Partitioned S1, T8 | 4.93 us | 32 | 128 B | 0.50 | 100.00% | 44.80% | 29.46% | 15.72% |
| S2, T8 | 9.70 us | 47 | 8.99 KiB | 0.00 | 83.33% | 16.65% | 3.45% | 1.96% |
| Accepted Q4 D3 token, T6 run | 17.34 us | 34 | 4.10 KiB | 0.60 | 93.75% | 54.07% | 50.39% | 50.39% |
| Accepted Q5 D4 token, T6 run | 13.31 us | 40 | 36 B | 2.41 | 93.75% | 78.75% | 37.52% | 33.94% |

Partitioning S1 converts a 257-CTA underfilled grid into 1028 CTAs, reduces the exact-T kernel's
register footprint, and improves measured occupancy. S2 is one tiny block and is launch-floor
limited; further arithmetic tuning cannot recover a separate launch, while the tested S1+S2
completion-counter fusion needs a memset and did not improve the complete route.

D3 and D4 dominate the remaining time. NCU also explains why the seemingly more parallel
alternatives lost:

- A token-batched T2 D3 used 1024 CTAs, 27 registers, and 4.10 KiB shared memory, but its 1.20 waves
  included a partial second wave and took 55.74 us, versus two accepted token kernels totaling
  about 34.8 us.
- An expert-grouped Q4 D3 used 56 registers. At T6 its 24.58 KiB shared memory allowed 75%
  theoretical occupancy and took 86.34 us. At T7, 28.67 KiB reduced theoretical occupancy to
  56.25%, left a tail wave, and raised duration to 176.26 us. Trace-like jobs are too small and
  imbalanced to repay that topology.
- Expert grouping helped only the artificial all-tokens-same-experts control at some larger T
  points. It lost on trace-like and independent routes, so it was removed rather than retained as
  an unprofitable runtime branch.

The retained `.ncu-rep`, text, and CSV views are under
`profiles/ncu/sparse-moe-small-t/`; the benchmark matrices are under
`profiles/bench/sparse-moe-small-t/`.

## Correctness and contract gate

One independent naive double-precision oracle decodes the exact packed formats and evaluates the
complete logical formula. Production routes are not compared only with one another. Retained
tests cover:

- Q4+Q5 at every exact template `T=2..8`, Q4+Q6 and W8+W8 at T6, and all codec profiles at T1;
- a dense common router background that exercises every S1 K partition while preserving controlled
  route ordering;
- different expert ids per token, a correlated six-token route trace, expert ids 0 and 255, and
  the lower-id top-8 boundary tie;
- nonzero residual input and the sole BF16 final write;
- exact and undersized workspace capacity, positive-T and shape/codec rejection;
- T8 and correlated-route CUDA Graph capture plus two graph replays; and
- T9 execution through the functionally complete serial fallback.

Every supported correctness case produced zero BF16 output difference against the oracle. The
workspace query returns enough capacity for the maximum exact-T route even when the caller asks
for a larger positive maximum, while T>8 continues to stream-order and reuse the decode workspace.

## Qualification result

`SparseMoe` now has three private dispatch regimes without changing its repository-internal API:

- T1: the existing qualified decode route, with contemporaneous non-regression evidence;
- T2..8: the qualified exact-T joint-router route described here; and
- T>8: the functionally complete column-serial fallback, without a performance claim.

The Small-T result is 1.081x-1.538x faster than the previous serial route across all measured
codec/T points (7.5%-35.0% lower latency). It is specifically qualified for MTP
verification-sized windows. Larger-T prefill and a future profitable grouped contraction remain
separate performance work; neither is needed to describe or preserve this Small-T implementation.
