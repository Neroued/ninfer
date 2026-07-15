# Qwen3.6-35B A2 KV Append Qualification

Date: 2026-07-15

Target: standalone KV append with BF16 K/V `[256,2,T]`, sequential I32 absolute cache positions,
`1<=T<=1024`, and either BF16 or INT8-G64 KV storage. The operation overwrites the addressed K/V
slots and INT8 scale slots without computing attention or advancing the host cache cursor.

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1, `sm_120a`, Nsight Compute
2025.4.1, and a Release build. Timings are CUDA-event medians after ten warmups. NCU uses one
selected launch and application replay.

## Qualified implementation

- BF16 launches 256-thread CTAs and copies two 16-byte vectors per thread, one from each input
  plane. The 35B grid is `ceil(T/4)` CTAs.
- INT8-G64 assigns one warp to each `(token,kv_head,group64)` tuple. The warp obtains the group
  absolute maximum, writes 64 signed codes and one FP16 scale for each K/V plane, and uses eight
  warps per CTA. The 35B grid is exactly T CTAs.
- Both routes use the supplied absolute position for the cache address and contain no dependency
  on current context length. The benchmark nevertheless sweeps positions from 0 through 261120
  to cover both ends of the registered address domain.

For one 35B token, BF16 reads 2 KiB and writes 2 KiB. INT8-G64 reads the same 2 KiB and writes
1.03125 KiB, including its K/V FP16 scale planes.

## Correctness

```bash
cmake --build build-release --target \
  ninfer_gqa_attention_test ninfer_gqa_attention_bench -j8
build-release/tests/ninfer_gqa_attention_test
```

The independent GQA operation test covers standalone append for both 24Q/4KV and 16Q/2KV
geometries, both cache formats, prompt-sized chunks, and split-API parity. It completes with
`OK gqa_attention correctness`.

## Canonical benchmark matrix

```bash
build-release/bench/ninfer_gqa_attention_bench \
  --kv-append --geometry 35b --tokens 1,2,3,4,5,6,1024 \
  --context 0,128,512,2048,8192,32768,131072,261120 \
  --kv-dtype bf16 --warmup 10 --repeat 100 \
  --csv-out profiles/bench/qwen3_6_35b_gqa_a2/bf16.csv \
  --json-out profiles/bench/qwen3_6_35b_gqa_a2/bf16.json
build-release/bench/ninfer_gqa_attention_bench \
  --kv-append --geometry 35b --tokens 1,2,3,4,5,6,1024 \
  --context 0,128,512,2048,8192,32768,131072,261120 \
  --kv-dtype int8 --warmup 10 --repeat 100 \
  --csv-out profiles/bench/qwen3_6_35b_gqa_a2/int8.csv \
  --json-out profiles/bench/qwen3_6_35b_gqa_a2/int8.json
```

The control has the same grid and moves the same input/output payload. BF16 is a direct two-plane
vector copy. The INT8 control packs the input bytes and writes the code and scale payloads but
omits absmax, scaling, and rounding. It is therefore a fixed-resource and transfer lower control,
not a numerical alternative implementation.

At the longest tested cache address, the retained batched launch intervals are:

| T | BF16 append | BF16 control | INT8 append | INT8 control |
|---:|---:|---:|---:|---:|
| 1 | 9.61 us | 9.60 us | 9.27 us | 9.37 us |
| 2 | 9.54 us | 9.49 us | 9.38 us | 9.42 us |
| 3 | 9.53 us | 9.42 us | 9.46 us | 9.52 us |
| 4 | 9.36 us | 9.63 us | 9.16 us | 9.52 us |
| 5 | 9.58 us | 9.50 us | 9.45 us | 9.54 us |
| 6 | 9.55 us | 9.30 us | 9.40 us | 9.36 us |
| 1024 | 9.33 us | 9.17 us | 9.23 us | 9.32 us |

Across all eight addresses, append and control intervals overlap at every T. Context changes only
the destination address and show no systematic latency trend. These batched intervals establish
the launch-limited small-T behavior; NCU separates actual device work for the largest chunk.

## NCU fixed-resource qualification

Representative collection uses the benchmark's `--profile-once` route and selects either the
production or control regex:

```bash
ncu --set basic --replay-mode application --launch-count 1 \
  --kernel-name regex:'gqa_attention_prefill_fill_bf16_kernel' ...
ncu --set basic --replay-mode application --launch-count 1 \
  --kernel-name regex:'bench_kv_append_bf16_control_kernel' ...
ncu --set basic --replay-mode application --launch-count 1 \
  --kernel-name regex:'gqa_attention_prefill_fill_i8_kernel' ...
ncu --set basic --replay-mode application --launch-count 1 \
  --kernel-name regex:'bench_kv_append_i8_payload_control_kernel' ...
```

T=1024 at cache address 261120:

| Route | Grid x block | NCU duration | Control fraction | DRAM SOL | Compute SOL | Occupancy | Registers/thread |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 append | `256 x 256` | 4.51 us | 92.2% | 34.32% | 1.71% | 21.29% | 20 |
| BF16 same-payload control | `256 x 256` | 4.16 us | 100% | 32.57% | 1.47% | 22.25% | 18 |
| INT8-G64 append | `1024 x 256` | 4.77 us | 89.3% | 36.75% | 22.48% | 69.85% | 33 |
| INT8 same-payload control | `1024 x 256` | 4.26 us | 100% | 36.43% | 8.05% | 73.55% | 16 |

The BF16 grid is only 0.25 wave/SM and the INT8 grid exactly one wave/SM. Neither exact problem is
a long streaming workload capable of approaching the card-wide DRAM number. The applicable bound
is the same-grid, same-payload control: production retains 92.2% for BF16 and 89.3% for INT8 while
also performing the required address mapping and, for INT8, the full group quantization. No
generic fallback or extra launch is present.

Retained reports:

- `profiles/bench/qwen3_6_35b_gqa_a2/bf16.csv`
- `profiles/bench/qwen3_6_35b_gqa_a2/bf16.json`
- `profiles/bench/qwen3_6_35b_gqa_a2/int8.csv`
- `profiles/bench/qwen3_6_35b_gqa_a2/int8.json`
- `profiles/ncu/qwen3_6_35b_gqa_a2_t1024_c261120_bf16__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_gqa_a2_t1024_c261120_bf16_control__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_gqa_a2_t1024_c261120_int8__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_gqa_a2_t1024_c261120_int8_payload_control__basic.ncu-rep`

This evidence qualifies A2 for the complete 35B target domain. It does not qualify cached-only
attention A3, whose small-T high-performance path remains separate work.
