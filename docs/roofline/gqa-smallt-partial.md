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
