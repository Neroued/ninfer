# Qwen3.6-35B Token-Decision Qualification

Date: 2026-07-16

Target: Qwen3.6-35B-A3B G1-G3 on NVIDIA GeForce RTX 5090 (`sm_120a`, CUDA 13.1,
driver 591.86, NCU 2025.4.1):

- G1 full-vocabulary argmax `[248320,C]`, 248077 valid rows, `C=1..6`, and shortlist
  `[131072,1]`;
- G2 greedy and top-20 stochastic sampling over 248077 valid rows, with optional occurrence
  counts;
- G3 greedy and stochastic MTP acceptance for every `K=1..5`.

This is standalone Op qualification for the future target. It does not register a 35B Engine
target or qualify the logits-producing Linear Ops.

## Implementation

G1 uses exact BF16-to-FP32 comparison with lower-token-id tie breaking. Its full and shortlist
domains use 512-thread, one-item tiled reductions followed by an atomic winner per column. A local
geometry sweep rejected 128-, 256-, and 1024-thread alternatives; 512 threads gives 485 CTAs for
the full valid vocabulary, or 0.95 waves/SM on this 170-SM GPU.

G2 and stochastic G3 share one finite two-launch implementation:

- 256-thread partial CTAs consume two vocabulary rows per thread and retain exact top-20 keys;
- groups merge 25 partials at a time, then the last group produces the truncated distribution;
- greedy G2 uses the same graph-stable launches but replaces both sorts with 64-bit maximum
  reductions;
- G3 builds `K+1` column distributions in parallel and performs the sequential accept/resample
  and complete documented state transition only after every column is ready.

All global staging is caller-owned. The public sizing query and launcher binding share one checked
layout definition; the exact workspace is 80.2 KiB for `C=1` and 475.2 KiB for `C=6`. There is no
hidden `__device__` scratch or counter state. Initial counters are written by the first partial CTA
of each column, avoiding an extra memset launch while preserving CUDA-stream and graph ordering.

Sampling order canonicalizes `+0` and `-0` before encoding its sortable key, so numeric equality
reaches the lower-token-id tie break. MTP penalties always include the round-local accepted-draft
overlay, even when the optional committed-count pointer is null.

## Correctness

The Release build and exact tests are:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j --target \
  ninfer_argmax_test ninfer_sampling_test ninfer_mtp_round_test \
  ninfer_argmax_bench ninfer_sampling_select_bench
ctest --test-dir build \
  -R '^ninfer_(argmax|sampling|mtp_round)_test$' --output-on-failure
```

All three tests pass. Retained cases include:

- exact G1 physical-stride/valid-row checks for each `C=1..6`, shortlist ties, and independent
  BF16-rounded CPU argmax oracles;
- G2 full-domain greedy/argmax parity, physical padding exclusion, top-k/top-p/min-p behavior,
  stochastic distribution checks, occurrence-count reproducibility, top-k clamping, and a
  cross-partial signed-zero tie;
- G3 exact greedy and stochastic state/output checks for every `K=1..5`, physical padding
  exclusion, distribution/reproducibility checks, count mutation, and presence/frequency overlay
  checks with and without committed counts.

## Release benchmark

The G1 payload control reads the same rotating BF16 payload, uses the same grid/block geometry and
same output memset, but replaces argmax comparison with a minimal XOR reduction. CUDA-event
medians are microseconds:

| G1 domain | Op | Same-grid control | Interpretation |
|---|---:|---:|---|
| full C=1 | 16.80 | 19.31 | fixed launch/grid floor |
| full C=2 | 16.76 | 18.88 | fixed launch/grid floor |
| full C=3 | 16.39 | 17.28 | fixed launch/grid floor |
| full C=4 | 16.82 | 16.66 | within 1.0% of control |
| full C=5 | 16.69 | 16.34 | within 2.1% of control |
| full C=6 | 17.27 | 16.46 | within 4.9% of control |
| shortlist C=1 | 17.04 | 19.21 | fixed launch/grid floor |

G2/G3 repeatedly consume the same already-produced logits and therefore represent the warm-cache
condition of token decision immediately after the output head. The benchmark covers both optional
count states and every MTP window:

| Route | Median us |
|---|---:|
| G2 greedy | 19.40 |
| G2 stochastic, no counts | 19.50 |
| G2 stochastic, counts active | 19.51 |
| G3 greedy K=1 / 2 / 3 / 4 / 5 | 19.69 / 19.57 / 19.41 / 19.53 / 19.51 |
| G3 stochastic counts K=1 / 2 / 3 / 4 / 5 | 24.02 / 27.96 / 32.23 / 35.78 / 40.03 |

The greedy routes are dominated by the fixed two-launch graph-stable envelope. Stochastic G3
scales with `K+1` columns without adding a launch per column.

## NCU attribution

Basic captures were taken before detailed sections and use narrow demangled kernel regexes. G3
uses application replay because it mutates counters and published round state. Representative
commands are:

```bash
ncu --force-overwrite --set basic \
  -o profiles/ncu/qwen3_6_35b_a3b/token_decision/final/g1_full_c1_basic.ncu-rep \
  --kernel-name regex:argmax_tiled_atomic_kernel --launch-skip 0 --launch-count 1 \
  ./build/bench/ninfer_argmax_bench --shape full --cols 1

ncu --force-overwrite --set basic --replay-mode application \
  -o profiles/ncu/qwen3_6_35b_a3b/token_decision/final/g2_stochastic_counts_basic.ncu-rep \
  --kernel-name 'regex:sampling_(partial_topk|group_finalize_sample)_kernel' \
  --launch-skip 0 --launch-count 2 \
  ./build/bench/ninfer_sampling_select_bench --sample --mode stochastic --top-k 20
```

NCU multi-pass duration is diagnostic and is not summed in place of the CUDA-event Op latency.
The important resource result is finite-grid attribution:

| Kernel | Grid | NCU us | SM SOL | Occupancy | Registers | Static shared |
|---|---:|---:|---:|---:|---:|---:|
| G1 full C=1 | 485 | 4.03 | 19.06% | 64.42% | 18 | 128 B |
| G1 full C=6 | 2910 | 13.38 | 34.40% | 62.61% | 18 | 128 B |
| G1 shortlist | 256 | 3.68 | 11.03% | 34.08% | 18 | 128 B |
| G2 stochastic partial | 485 | 10.30 | 35.28% | 45.21% | 30 | 4.17 KiB |
| G2 stochastic group/final | 20 | 15.68 | 1.08% | 16.54% | 56 | 4.42 KiB |
| G3 K=5 stochastic partial | 2910 | 35.74 | 64.84% | 88.68% | 40 | 4.10 KiB |
| G3 K=5 stochastic group/final | 120 | 23.58 | 4.34% | 16.40% | 64 | 4.35 KiB |

The partial stage is the only broadly resident stage. Group/final work is intentionally bounded to
20 CTAs per column and ends with a completion barrier; its low whole-GPU utilization is a finite
work property, not evidence for a larger generic kernel. A 512-thread group experiment did not
improve duration. No route reports local/shared spilling.

## Result

G1, G2, and G3 are qualified for exactly the inventory domains above. The implementation keeps
device-dynamic greedy/stochastic selection and stable workspace addresses, so it does not require
duplicating full-model CUDA Graphs. Complete 35B generation and MTP execution remain unsupported
until the other inventory gaps are closed.
