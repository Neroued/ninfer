# Qwen3.6 Vision Engine Integration

Date: 2026-07-12

## Result

The native C++ engine now owns the complete Qwen3.6-27B multimodal path:

```text
structured text/image/video messages
  -> native Processor
  -> fixed 27-layer Qwen3_6_Vision tower
  -> exact patch merger [V,5120]
  -> per-prefill-chunk placeholder scatter
  -> 64-layer text model with three-axis MRoPE
  -> decode with persisted rope_delta
```

There is no Hugging Face runtime dependency in C++. The Python implementation remains the numerical
reference and diagnostic oracle.

## Static Vision model card

`Qwen3_6_Vision` binds the 333 tensors in `VISION_ENCODER` by `ModuleKind`, `SourceKind`, and source
layer. Its fixed schedule is:

1. Q6 patch projection `[1536 -> 1152]`, bias, and four-corner position interpolation;
2. 27 blocks of affine LayerNorm, packed QKV, 2-D RoPE, segmented noncausal attention, projection,
   tanh GELU MLP, and two residual additions;
3. affine LayerNorm, zero-copy `[P,1152] -> [P/4,4608]` merge view, W8G32 FC1, exact GELU, and W8G32
   FC2 to `[V,5120]`.

Q/K/V remain strided views of the packed QKV projection. The tower does not materialize an `L x L`
score tensor.

## Workspace and lifetime

The Vision card exposes an exact request-derived workspace bound. The engine keeps one grow-only
Vision `WorkspaceArena` and reuses it across requests:

- capacity is derived from actual raw-patch count;
- available GPU memory is checked before growth;
- text prefill semantically consumes only the merger output, while the grow-only arena keeps all
  request allocations reserved until that synchronous prefill completes;
- the arena is rewound immediately after text prefill, including exception paths;
- no kernel owns or wraps an Arena.

The normal text workspace remains unchanged. `EngineMemoryStats.workspace` accounts for both arenas.

## Text integration

Physical cache positions and rotary positions are deliberately separate:

- KV append/attention always receives contiguous physical token positions;
- RoPE receives `[T]` for text-only, `[T,3]` for multimodal prefill, and
  `physical_position + rope_delta` for subsequent decode;
- MTP prompt, proposal, and verify paths use the same separation;
- each prefill chunk packs a contiguous `[len,3]` position tensor and scatters only the visual
  columns belonging to that chunk.

Multimodal residents are never reused by the token-only prefix cache. Identical placeholder token
IDs do not prove that two requests contain identical media.

## Runtime surfaces

- `Engine::prefill(const ProcessedInput&)` performs Vision plus mixed text prefill.
- `TextGenerationRunner` selects the multimodal path when messages contain media.
- `qus --messages` accepts image/video messages directly.
- OpenAI and Anthropic serving paths admit at most one memory-heavy media preparation at a time,
  run Vision under the existing single-engine execution lock, and release CPU patches immediately
  after Vision prefill.
- multimodal token counting uses the fully expanded processor output.
- malformed media returns 400, media/request budgets return 413, remote fetch failures return
  502/504, and internal failures remain 500.

## Numerical evidence

One real screenshot produced `P=1428`, `V=357`, grid `[1,34,42]`. C++ activations were compared with
the q5090 Python reference using `qus-vision-dump` and the structured activation comparator:

| Activation | RMSE | Cosine |
| --- | ---: | ---: |
| patch embedding | 0.0002739 | 0.999999833 |
| block 0 | 0.0016921 | 0.999996467 |
| block 13 | 0.0163964 | 0.999848062 |
| block 26 | 1.059311 | 0.999924972 |
| merger | 0.0051239 | 0.999935148 |

The same Python q5090 path was independently compared with the source BF16 Hugging Face tower. Its
merger cosine was `0.9956069`; differences are expected from quantized weights and different GEMM/
attention accumulation paths.

## Observable E2E evidence

- Canonical 64x64 red/blue PPM images produce distinct `红色` and `蓝色` answers in the permanent
  real-weight E2E test.
- A real screenshot was correctly described as a browser/application bookmark menu.
- A real video (`P=8448`, `V=2112`, prompt 2198 tokens) was correctly identified as a video-editing
  application interface; this also exercises visual scatter and MRoPE across multiple 1024-token
  prefill chunks.
- Vision with MTP draft window 2 ran successfully and reported accepted speculative tokens.
- The text-only greedy smoke prompt still returned `你好。`.
- A local OpenAI `image_url` data-URI request returned `红色` with expanded usage
  `prompt_tokens=26`; the thinking-enabled blue-image request returned a nonempty
  `reasoning_content` and final answer `蓝色`.

## Diagnostics

```bash
build/src/qus-vision-dump MODEL.qus messages.json /tmp/cpp-vision --no-thinking

python -m tools.q5090.diagnostics.vision \
  --weights MODEL.qus --model-dir HF_MODEL --messages messages.json \
  --q5090-dump /tmp/python-vision --no-thinking

python -m tools.q5090.diagnostics.activations \
  /tmp/python-vision /tmp/cpp-vision
```

The dump contract captures patch embedding, blocks 0/13/26, and merger output as FP32 files with a
`qus_activation_dump_v1` manifest.
