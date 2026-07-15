# Qwen3.6-35B Causal Convolution Qualification

Date: 2026-07-15

Target: BF16 depthwise causal width-4 convolution with fused SiLU, runtime channel count, ordinary
in-place/distinct state, and snapshot state. The exact 35B domains are `C=8192`, ordinary
`1<=T<=1024`, and snapshot `1<=T<=6` with seven slots.

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1.115, `sm_120a`, Nsight Compute
2025.4.1, and a Release build. Work started from source revision `8072f60`. The benchmark repeats
the same device buffers after 20 warmups; its CUDA-event median is therefore a hot steady-state Op
latency. NCU uses one launch after ten skipped launches and application replay unless noted.

## Final dispatch

The implementation remains one operation family and does not encode either model width:

- in-place `T=1` retains the original single decode kernel;
- distinct-state `T=1` uses a separate single kernel so the compiler can exploit disjoint state
  storage without slowing the exact-alias decode route;
- ordinary `2<=T<=16` maps one 32-channel tile and all T columns to one CTA, caching the three
  history values and four weights in 448 bytes of shared memory;
- ordinary `17<=T<=64` uses one serial channel kernel, avoiding the second state-update launch;
- ordinary `T>64` retains the BF16x2 channel-pair prefill kernel and its compact final-state
  kernel;
- snapshot `T=1` has a loop-free single kernel, while `2<=T<=16` uses the same channel/token CTA
  organization and writes every token state directly to its slot in one launch;
- larger snapshot calls remain supported by the serial one-thread-per-channel path.

All paths take runtime C and handle a partial final channel tile. Ordinary state may be exactly
aliased or disjoint. Snapshot slot count and selected initial slot remain runtime values. The
width-4 mathematical contract is intentionally not generalized into an unrelated variable-width
convolution.

## Correctness

```bash
ctest --test-dir build-release \
  -R '^ninfer_causal_conv1d_silu_test$' --output-on-failure
```

The test passes the FP64 operation oracle and state-transition checks. Exact 35B coverage includes
`[8192,1]`, `[8192,6]`, `[8192,1024]`, distinct state, seven-slot snapshot state selected from slot
6, and bitwise snapshot equivalence to six sequential decode calls. Existing `C=10241` cases cover
the generalized partial-tile path, including a selected snapshot slot that is overwritten during
the call.

## Steady-state timing

```bash
build-release/bench/ninfer_causal_conv1d_silu_bench --decode \
  --channels 8192
build-release/bench/ninfer_causal_conv1d_silu_bench --distinct \
  --channels 8192 --tokens 6
build-release/bench/ninfer_causal_conv1d_silu_bench --snapshot \
  --channels 8192 --tokens 6 --slots 7 --initial-slot 6
build-release/bench/ninfer_causal_conv1d_silu_bench --prefill \
  --channels 8192 --tokens 1024
```

| Route | Final median |
|---|---:|
| in-place decode, T=1 | 9.47 us |
| distinct state, T=1 | 9.40 us |
| distinct state, T=6 | 9.29 us |
| snapshot, T=1 | 9.43 us |
| snapshot, T=6 | 9.34 us |
| prefill, T=1024 | 27.03 us |

The five short routes sit at the same batched launch-interval floor as the benchmark's 9.51 us
same-byte copy control. Their printed logical GB/s is not a roofline metric. NCU is used below to
separate actual kernel work from this interval floor.

## NCU qualification

Representative collection command:

```bash
ncu --set basic --kernel-name-base function \
  --kernel-name 'regex:causal_conv1d_smallt_kernel' \
  --launch-skip 10 --launch-count 1 --replay-mode application \
  -o profiles/ncu/qwen3_6_35b_a3b/causal_conv_smallt_distinct_t6 \
  build-release/bench/ninfer_causal_conv1d_silu_bench \
  --distinct --channels 8192 --tokens 6
```

| Exact route and selected kernel | Grid x block | NCU duration | Compute SOL | Memory SOL | Achieved occupancy | Registers/thread |
|---|---:|---:|---:|---:|---:|---:|
| in-place T=1 decode | `32 x 256` | 2.94 us | 0.95% | 11.57% | 16.26% | 62 |
| distinct-state T=1 decode | `32 x 256` | 3.01 us | 0.94% | 11.21% | 16.12% | 72 |
| ordinary T=6 channel/token | `256 x 192` | 3.65 us | 3.80% | 9.31% | 17.78% | 38 |
| snapshot T=1 decode | `32 x 256` | 3.36 us | 0.78% | 10.06% | 16.38% | 40 |
| snapshot T=6 channel/token | `256 x 192` | 4.35 us | 4.64% | 9.85% | 18.40% | 38 |
| prefill output T=1024 BF16x2 | `16384 x 256` | 30.78 us | 71.63% | 61.75% | 86.88% | 38 |

The T=1 and T=6 problems are fixed-work/launch limited. They contain only 8192 channels, the final
kernels use one launch with no intermediate tensor, and NCU reports only 0.05 wave/SM for T=1 and
0.19 wave/SM for T=6 on 170 SMs. Low aggregate SOL is consequently the expected signature of the
exact problem size, not evidence for a missing bandwidth or compute path. A 32-thread decode CTA
raised the grid from 32 to 256 blocks but measured 2.98 us versus the retained 2.94 us route, so it
was rejected. Detailed replay of the ordinary T=6 kernel reports zero local/shared spilling.

The long prefill kernel has 16.06 waves/SM, 86.88% achieved occupancy, and 38 registers per thread;
NCU classifies compute and memory as well balanced. It therefore remains the applicable
compute/L2-throughput path rather than being replaced by the Small-T organization.

The new parallel snapshot path reduces its T=6 kernel from 6.66 us to 4.35 us, a 34.7% reduction.
The loop-free T=1 snapshot path reduces 4.26 us to 3.36 us, and the disjoint T=1 path reduces the
runtime-loop kernel from 4.48 us to 3.01 us. An attempted shared in-place/distinct decode kernel
raised registers from 62 to 70 and regressed the in-place kernel to 4.13 us; it was rejected and
the original in-place path was retained.

Retained reports:

- `profiles/ncu/qwen3_6_35b_a3b/causal_conv_baseline_decode.ncu-rep`
- `profiles/ncu/qwen3_6_35b_a3b/causal_conv_distinct_t1_final.ncu-rep`
- `profiles/ncu/qwen3_6_35b_a3b/causal_conv_smallt_distinct_t6.ncu-rep`
- `profiles/ncu/qwen3_6_35b_a3b/causal_conv_smallt_distinct_t6_detailed.ncu-rep`
- `profiles/ncu/qwen3_6_35b_a3b/causal_conv_snapshot_t1_final.ncu-rep`
- `profiles/ncu/qwen3_6_35b_a3b/causal_conv_smallt_snapshot_t6.ncu-rep`
- `profiles/ncu/qwen3_6_35b_a3b/causal_conv_baseline_prefill_t1024.ncu-rep`

These measurements qualify both S1 and S2 at their exact 35B decode, verification, and maximum
prefill domains. The wider runtime-C and longer Small-T support is implementation coverage; it does
not claim qualification for unregistered target shapes.
