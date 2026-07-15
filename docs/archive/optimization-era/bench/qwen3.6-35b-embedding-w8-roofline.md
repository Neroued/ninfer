# Qwen3.6-35B W8 Embedding Gather Qualification

Date: 2026-07-16

Target: the Qwen3.6-35B-A3B token embedding table in registered
`row-split-k128-v1` W8G32_F16S storage. The logical table is `[248320,2048]`; input ids are I32
`[T]` and output is BF16 `[2048,T]`. The complete target execution domain is decode/verification
`T=1..6` and one canonical prefill chunk `T=1024`.

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1.115, `sm_120a`, Nsight Compute
2025.4.1, and a Release build. CUDA-event values are medians after warmup. NCU preflight reported
`4 OK / 0 WARN / 0 FAIL`.

## Implementation

The W8 table stores one signed int8 code per element and one FP16 scale per 32-value group. For
token t and feature k:

```text
row = ids[t]
group = floor(k / 32)
out[k,t] = BF16(int8(code[row,group,k mod 32]) * FP16(scale[row,group]))
```

Two fixed routes cover the exact D=2048 domain:

- `T<=6`: one 32-thread CTA owns one 32-value group. The grid has `64*T` CTAs, one scale load is
  broadcast within the warp, and each lane decodes and writes one BF16 value.
- `T>6`: one 256-thread CTA owns one token row. Each thread vector-loads eight consecutive int8
  codes and writes four BF16x2 pairs. Four neighboring threads address the same FP16 scale; the
  coalescer services those reads directly, so the final kernel has no shared memory or block
  barrier.

The public Op also has a dimension-driven functional W8 fallback. That fallback is not evidence
for any domain other than the fixed `[248320,2048]` route qualified here. The existing 27B
Q6/D=5120 dispatch and kernels are unchanged.

## Correctness

```bash
cmake --build build --target ninfer_embedding_test ninfer_qwen3_6_27b_rtx5090 -j8
build/tests/ninfer_embedding_test
```

The independent test constructs row-split W8 payloads from signed int8 codes and FP16 scales, then
compares device output against direct CPU dequantization from those encoded values. It covers exact
`D=2048, T=1..6/1024`, a generic W8 fallback case, and the retained dense/Q6 domains. The test
completes with `OK embedding correctness`; the reported W8 relative-L2 error is approximately
0.16%, representing the final BF16 output rounding from the encoded-value oracle.

## Fixed-resource qualification

```bash
build/bench/ninfer_embedding_bench --w8
build/bench/ninfer_embedding_bench --w8 --control
```

The benchmark control uses the same grid and block geometry as production, reads the same ids,
int8 codes, and FP16 scales, and writes the same BF16-sized output payload. It retains every input
through the output while omitting W8-to-BF16 multiplication. It is a fixed-resource lower control,
not an alternate implementation.

CUDA-event medians are microseconds:

| T | Production | Payload control | Production / control |
|---:|---:|---:|---:|
| 1 | 10.16 | 9.78 | 1.039x |
| 2 | 10.00 | 9.72 | 1.029x |
| 3 | 10.15 | 9.86 | 1.029x |
| 4 | 10.07 | 9.94 | 1.013x |
| 5 | 9.96 | 9.81 | 1.015x |
| 6 | 9.87 | 9.74 | 1.013x |
| 1024 | 9.62 | 9.59 | 1.003x |

Every decode/verification point is within 3.9% of its fixed-work control. The T=1024 route moves
6,426,624 useful bytes and measures 668.1 GB/s in the hot repeated-launch benchmark, but this
finite one-wave problem is qualified against its payload control rather than the full-device DRAM
roofline.

## NCU confirmation

The final production and control captures use the first matching launch:

```bash
ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/embedding/w8_d2048_t1024_final__basic \
  --set basic --kernel-name regex:'embed_gather_w8_row_2048_kernel' \
  --launch-skip 0 --launch-count 1 \
  build/bench/ninfer_embedding_bench --w8 --prefill

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/embedding/w8_d2048_t1024_control__basic \
  --set basic --kernel-name regex:'w8_row_payload_control_kernel' \
  --launch-skip 0 --launch-count 1 \
  build/bench/ninfer_embedding_bench --w8 --prefill --control
```

| Route | Grid x block | Duration | Memory SOL | DRAM SOL | Compute SOL | Achieved occupancy | Registers/thread | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Production | `1024 x 256` | 5.34 us | 47.75% | 26.29% | 8.14% | 69.82% | 23 | 1.00 |
| Payload control | `1024 x 256` | 5.15 us | 33.16% | 27.14% | 6.14% | 68.41% | 18 | 1.00 |

Production is within 3.7% of the raw-kernel payload floor. Both launches use zero static or dynamic
shared memory, and NCU confirms that the intended fixed kernels were captured. Retained reports:

- `profiles/ncu/qwen3_6_35b_a3b/embedding/w8_d2048_t1024_final__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_a3b/embedding/w8_d2048_t1024_control__basic.ncu-rep`

This qualifies I2 for its complete W8/D=2048 Text and MTP domain.
