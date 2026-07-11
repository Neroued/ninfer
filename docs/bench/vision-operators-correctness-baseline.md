# Vision Operator Correctness Baseline

Date: 2026-07-11

GPU: NVIDIA GeForce RTX 5090, driver 591.86

Build: Release, `sm_120a`, CUDA `-lineinfo`

This is the pre-optimization baseline for the native Qwen3.6 Vision operator set. The first phase
prioritizes the exact model semantics and fixed-shape numerical oracles. The per-op binaries below
are the stable NCU targets for the serial optimization phase.

## Correctness

```bash
ctest --test-dir build \
  -R '^(qus_rope_test|qus_vision_elementwise_test|qus_layer_norm_test|qus_vision_pos_embed_test|qus_vision_attention_test|qus_vision_support_test)$' \
  --output-on-failure
```

All six tests pass. Coverage includes Text 1-D RoPE, three-axis MRoPE, packed-stride Vision 2-D
RoPE, affine LayerNorm at `[1152,4096]`, both GELU definitions, position interpolation with the
required intermediate BF16 rounding, placeholder scatter, F32-to-BF16 patch conversion, packed
segmented attention, and a real 256-patch Vision attention segment.

The packed attention test also passes:

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 \
  build/tests/qus_vision_attention_test
```

with `ERROR SUMMARY: 0 errors`.

## Informational timing baseline

These numbers come from the benchmark harness and are not the optimization acceptance gate. Reused
buffers can be served partly from cache; the second phase uses NCU SpeedOfLight, Occupancy, memory,
roofline, and stall metrics.

| Operator and shape | Median |
| --- | ---: |
| `add_bias [3456,4096]` | 41.54 us |
| `gelu tanh [4304,4096]` | 43.35 us |
| `gelu exact [4304,4096]` | 45.66 us |
| `layer_norm [1152,4096]` | 16.72 us |
| `vision_pos_embed_add [1152,4096]` | 22.45 us |
| `scatter [5120,2040]` | 37.75 us |
| Text MRoPE `[3,4096]` | 23.68 us |
| packed Vision RoPE `P=4096` | 29.48 us |
| `vision_attention L=256` correctness kernel | 574.94 us |

The attention implementation is deliberately an online-softmax correctness kernel: it never
materializes an `L×L` score tensor, but it synchronizes for each key. It is expected to be the
largest optimization task and is not presented as a performance implementation.
