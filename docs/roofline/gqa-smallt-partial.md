# GQA small-T partial roofline

Date: 2026-07-05

Target kernel:
`gqa_attention_small_t_tc_partial_kernel<4,4>`

Fixed bench profile:

```bash
./build/bench/qus_gqa_attention_bench --append-small-t --tokens 4 --context 2048 --profile-once
```

NCU command shape:

```bash
ncu --force-overwrite --target-processes all --kernel-name-base demangled --set basic \
  --launch-skip 0 --launch-count 1 \
  --kernel-name 'regex:gqa_attention_small_t_tc_partial_kernel' \
  -o profiles/gqa-smallt-partial/t4_ctx2048 \
  ./build/bench/qus_gqa_attention_bench --append-small-t --tokens 4 --context 2048 --profile-once
```

## Baseline

Report: `profiles/gqa-smallt-partial/t4_ctx2048.ncu-rep`

Bench metadata:

```text
PROFILE_ONCE gqa_attention append-small-T T=4 context=2048 splits=5 useful_kv_bytes=8404992 scratch_bytes=510720 total_model_bytes=9030400 redundancy=1.074409
```

NCU basic metrics:

| Metric | Value |
| --- | ---: |
| Duration | 65.31 us |
| Memory throughput | 7.82% |
| DRAM throughput | 7.82% |
| Compute (SM) throughput | 4.84% |
| Achieved occupancy | 9.16% |
| Registers/thread | 254 |
| Static shared memory/block | 36.86 KiB |
| Block size | 128 |
| Grid | `(4, 192, 1)` = 768 CTAs |
| Waves/SM | 2.26 |

The measured low SOL starts with launch geometry. The bench window is 2052 keys, and the old
small-T policy selected only 5 active splits. The launcher still emitted 192 split columns, so only
20 CTAs performed real attention work while 748 CTAs returned at the inactive-split guard. With 170
SMs, 20 useful CTAs cannot fill the GPU.

## Attempts

All attempts used the fixed bench profile and the NCU command shape above, changing only report
names.

| Attempt | Change | Report | Splits | Grid | Duration | Memory | SM | Achieved occ. | Result |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | old policy | `t4_ctx2048` | 5 | `(4,192,1)` | 65.31 us | 7.82% | 4.84% | 9.16% | starting point |
| 1 | small-window target 12 keys/split | `t4_ctx2048_attempt1` | 171 | `(4,192,1)` | 39.55 us | 21.72% | 25.44% | 15.90% | better SOL, too much redundant split work |
| 2 | small-window target 32 keys/split | `t4_ctx2048_attempt2` | 65 | `(4,192,1)` | 23.84 us | 20.61% | 16.73% | 13.21% | best split policy before grid cap |
| 3 | small-window target 24 keys/split | `t4_ctx2048_attempt3` | 86 | `(4,192,1)` | 30.91 us | 15.51% | 16.94% | 16.12% | slower than 32 keys/split |
| 4 | T=4 route to `<4,2>` plus 32 keys/split | `t4_ctx2048_attempt4` | 65 | `(4,192,1)` | 24.61 us | 20.10% | 8.78% | 6.50% | slower and no longer the target `<4,4>` kernel |
| 5 | 32 keys/split plus host split upper-bound grid cap | `t4_ctx2048_attempt5` | 65 | `(4,65,1)` | 23.71 us | 20.46% | 16.60% | 12.88% | kept |

Final bench metadata:

```text
PROFILE_ONCE gqa_attention append-small-T T=4 context=2048 splits=65 useful_kv_bytes=8404992 scratch_bytes=6639360 total_model_bytes=15159040 redundancy=1.803576
```

## Conclusion

The retained change improves the target partial kernel duration from 65.31 us to 23.71 us, a 2.75x
speedup on the fixed NCU profile, while preserving the `<4,4>` kernel route. The fastest policy is
32 keys/split because it exposes 65 split columns without increasing the number of 32-key K/V tiles
loaded by the partial kernel for this 2052-key window. More aggressive split counts raise SOL
slightly but add redundant split and reduce work, making the standalone partial kernel slower.

The standalone kernel cannot physically reach 85% SOL on this profile with the current algorithmic
shape. Final NCU still reports only 260 CTAs, 0.76 waves/SM, 254 registers/thread, 36.86 KiB static
shared memory/block, and 16.67% theoretical occupancy limited by registers and shared memory.
Attempt 1 supplied enough active split work to raise achieved occupancy to 15.90% and SM throughput
to 25.44%, but duration regressed to 39.55 us. The evidence points to a resource- and
small-grid-limited kernel, not a missed scheduling flag that can reach 85% SOL while keeping this
standalone workload and math path.

## Verification

```bash
cmake --build build --target qus_gqa_attention_bench qus_gqa_attention_test -j
./build/tests/qus_gqa_attention_test
```

Result: `OK gqa_attention correctness`.

## Full small-T retune

Date: 2026-07-05

The first pass above only measured `T=4, context=2048`. The full retune swept
`T in {1..6}` and contexts `{2048, 8192, 16384, 32768}`. Candidate policies used
uniform keys-per-split targets for `T=2..6`: 32, 64, 96, 128, 160, 192, 256,
320, 384, 480, 512, plus 640 and 768 for the 32768-context fallback. T=1 keeps
the existing max-split path because it is structurally a decode-like case.

Retained policy:

```text
T=1: max splits
T=2..6:
  window <= 4096  ->  64 keys/split
  window <= 8198  -> 128 keys/split
  window <= 16390 -> 256 keys/split
  otherwise       -> 480 keys/split
```

The 8198 and 16390 limits are window limits, not nominal context labels. They
cover the measured 8192 and 16384 context regimes with the maximum small-T
append length of 6 tokens while keeping the host upper-bound grid cap from
inflating above the measured 32k fallback cap.

Decision summary:

- The `tokens <= 5` boundary had no measured justification. T=6 follows the
  same policy as T=2..5.
- The old T=6 512-key branch under-split the 2048 context case: final T=6 at
  context 2048 uses 33 splits and measures 22.47 us, versus the old 5-split
  branch at 49.65 us.
- 64 keys/split was the best 2048-context target in the combined append bench.
  The previous 32-key target exposed more CTAs but paid more scratch/reduce
  overhead.
- 128 keys/split was best at 8192, 256 keys/split was best at 16384, and 480
  keys/split was best or tied at 32768. The 512-key T=6 exception did not win
  enough to justify a separate branch.

Final NCU command shape:

```bash
ncu --force-overwrite --target-processes all --kernel-name-base demangled --set basic \
  --launch-skip 0 --launch-count 1 \
  --kernel-name 'regex:gqa_attention_small_t_tc_partial_kernel' \
  -o profiles/gqa-smallt-partial/final_t<T>_ctx<ctx> \
  ./build/bench/qus_gqa_attention_bench --append-small-t --tokens <T> --context <ctx> --profile-once
```

Generated artifacts:

- `profiles/gqa-smallt-partial/final_bench.txt`
- `profiles/gqa-smallt-partial/final_copy_ceiling.txt`
- `profiles/gqa-smallt-partial/final_t<T>_ctx<ctx>.ncu-rep`
- `profiles/gqa-smallt-partial/final_t<T>_ctx<ctx>.ncu.csv`
- `profiles/gqa-smallt-partial/final_t<T>_ctx<ctx>.ncu.txt`

Final bench and NCU basic metrics:

| T | ctx | splits | bench us | useful GB/s | redundancy | NCU partial us | DRAM % | SM % | occ % | grid blocks | waves/SM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 192 | 36.19 | 231.9 | 1.59 | 31.52 | 15.35 | 17.40 | 7.89 | 768 | 2.26 |
| 1 | 8192 | 192 | 46.03 | 729.1 | 1.15 | 52.80 | 35.78 | 17.64 | 8.07 | 768 | 2.26 |
| 1 | 16384 | 192 | 54.92 | 1222.0 | 1.07 | 76.64 | 50.23 | 18.16 | 8.10 | 768 | 2.26 |
| 1 | 32768 | 192 | 119.99 | 1118.6 | 1.04 | 136.03 | 57.57 | 20.54 | 7.99 | 768 | 2.26 |
| 2 | 2048 | 33 | 21.74 | 386.2 | 1.21 | 21.79 | 22.27 | 14.80 | 8.33 | 132 | 0.39 |
| 2 | 8192 | 65 | 33.37 | 1005.8 | 1.10 | 43.23 | 44.10 | 29.56 | 12.77 | 260 | 0.76 |
| 2 | 16384 | 65 | 48.66 | 1379.3 | 1.05 | 74.82 | 50.87 | 34.11 | 12.69 | 260 | 0.76 |
| 2 | 32768 | 69 | 110.67 | 1212.9 | 1.03 | 133.92 | 63.76 | 38.26 | 13.46 | 276 | 0.81 |
| 3 | 2048 | 33 | 21.28 | 394.7 | 1.31 | 22.78 | 21.44 | 14.09 | 8.31 | 132 | 0.39 |
| 3 | 8192 | 65 | 34.35 | 977.1 | 1.15 | 44.93 | 42.39 | 28.48 | 12.72 | 260 | 0.76 |
| 3 | 16384 | 65 | 48.61 | 1380.8 | 1.08 | 77.57 | 49.50 | 33.09 | 12.69 | 260 | 0.76 |
| 3 | 32768 | 69 | 111.53 | 1203.6 | 1.04 | 131.62 | 61.21 | 38.77 | 13.44 | 276 | 0.81 |
| 4 | 2048 | 33 | 21.37 | 393.4 | 1.41 | 24.00 | 20.31 | 13.46 | 8.33 | 132 | 0.39 |
| 4 | 8192 | 65 | 34.22 | 981.0 | 1.20 | 46.18 | 41.71 | 27.66 | 12.73 | 260 | 0.76 |
| 4 | 16384 | 65 | 49.01 | 1369.6 | 1.10 | 76.45 | 50.54 | 33.57 | 12.72 | 260 | 0.76 |
| 4 | 32768 | 69 | 112.91 | 1188.9 | 1.05 | 133.79 | 60.37 | 38.21 | 13.46 | 276 | 0.81 |
| 5 | 2048 | 33 | 21.58 | 389.7 | 1.52 | 24.90 | 19.63 | 12.91 | 8.33 | 132 | 0.39 |
| 5 | 8192 | 65 | 34.45 | 974.5 | 1.25 | 47.52 | 43.07 | 27.11 | 12.71 | 260 | 0.76 |
| 5 | 16384 | 65 | 50.50 | 1329.2 | 1.13 | 79.58 | 50.43 | 32.08 | 12.70 | 260 | 0.76 |
| 5 | 32768 | 69 | 115.00 | 1167.3 | 1.07 | 135.65 | 58.78 | 37.88 | 13.46 | 276 | 0.81 |
| 6 | 2048 | 33 | 22.47 | 374.3 | 1.62 | 26.50 | 19.90 | 12.11 | 8.33 | 132 | 0.39 |
| 6 | 8192 | 65 | 35.14 | 955.5 | 1.30 | 49.25 | 40.36 | 26.16 | 12.77 | 260 | 0.76 |
| 6 | 16384 | 65 | 50.79 | 1321.7 | 1.15 | 80.42 | 47.97 | 31.83 | 12.70 | 260 | 0.76 |
| 6 | 32768 | 69 | 117.41 | 1143.3 | 1.08 | 136.99 | 63.58 | 37.28 | 13.48 | 276 | 0.81 |

Copy ceiling:

| T | ctx | cold C_copy GB/s | cold payload GB/s | hot C_copy GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 1766.1 | 883.1 | 1660.4 |
| 1 | 8192 | 1726.3 | 863.1 | 4730.8 |
| 1 | 16384 | 1538.2 | 769.1 | 1814.6 |
| 1 | 32768 | 1493.7 | 746.9 | 1472.1 |
| 2 | 2048 | 1767.0 | 883.5 | 1806.6 |
| 2 | 8192 | 1758.3 | 879.2 | 4772.2 |
| 2 | 16384 | 1537.7 | 768.8 | 1814.8 |
| 2 | 32768 | 1495.7 | 747.8 | 1472.2 |
| 3 | 2048 | 1773.8 | 886.9 | 1742.8 |
| 3 | 8192 | 1738.1 | 869.1 | 4794.6 |
| 3 | 16384 | 1537.8 | 768.9 | 1821.7 |
| 3 | 32768 | 1496.0 | 748.0 | 1469.2 |
| 4 | 2048 | 1774.7 | 887.4 | 1825.3 |
| 4 | 8192 | 1749.9 | 875.0 | 4763.0 |
| 4 | 16384 | 1537.9 | 768.9 | 1822.9 |
| 4 | 32768 | 1486.5 | 743.2 | 1471.8 |
| 5 | 2048 | 1775.6 | 887.8 | 1743.9 |
| 5 | 8192 | 1700.5 | 850.3 | 4712.9 |
| 5 | 16384 | 1537.4 | 768.7 | 1825.3 |
| 5 | 32768 | 1484.9 | 742.5 | 1466.8 |
| 6 | 2048 | 1776.4 | 888.2 | 1756.4 |
| 6 | 8192 | 1735.9 | 867.9 | 4751.2 |
| 6 | 16384 | 1538.1 | 769.0 | 1816.6 |
| 6 | 32768 | 1482.1 | 741.0 | 1472.4 |
