# qwen3.6-ultraspeed

A from-scratch C++/CUDA inference engine built to run **Qwen3.6-27B** as fast as physically
possible for a **single user on a single RTX 5090**.

No dynamic compute graph, no general-purpose model zoo — the computation for this one model
is hand-written and specialized to the hardware. Weights are pre-quantized to a static 4-bit
format offline; the runtime just loads one fixed file and runs.

## Current status

M2 correctness baseline is implemented. M2.5 hardening/documentation sync is mostly landed, with
graph-readiness fixes, EOS handling, FileTap parity dumps, and hardening cleanup structure already
present. The M2.8 benchmark/I/O/memory observability gate is complete; M3 planning may begin from
[`docs/m3-readiness.md`](docs/m3-readiness.md).

Current code includes L0 infrastructure, the q5090 `WeightStore`/loader and unified `Weight` handle,
the public L1 operator surface and implementations, the L2 text/Vision model cards, the `Engine`,
and q5090 Python reference/diagnostic tooling. No performance numbers are claimed
here; performance claims require the M2.8 e2e benchmark/report standard.

## Goal

- **Primary metric:** single-stream decode throughput (tokens/sec, batch = 1).
- **Secondary:** prefill / time-to-first-token for long prompts.
- **Anchor:** the weight-bandwidth roofline (~120–130 tok/s) — the project is about how close
  we can get.

## The model (frozen target)

`Qwen3.6-27B` (internally the `qwen3_5` architecture): a **hybrid-attention** dense model —
64 layers in a 3:1 pattern of **Gated-DeltaNet linear attention** (48 layers) and **GQA full
attention** (16 layers), SwiGLU MLP, vocab 248320, plus the fixed 27-layer Vision tower and patch
merger. The runtime supports the target decoder, MTP speculative decoding, and native image/video
prefill.

## Scope (v1)

- Text, image, and video chat input. Vision runs only during prefill; decode uses the resulting
  language-model KV/GDN state.
- Greedy or sampled decoding, BF16 or INT8 KV, and current official **max_ctx = 8192**.
- **W4A16**: 4-bit weights, bf16 activations.
- Single sequence (batch = 1), single GPU.

128K context is a later target, not part of the current M2.8 gate. Deferred (in order):
fp8/fp4 prefill → 128K/256K context and fp8-KV → multi-GPU/batching.

## Architecture (3 layers)

| Layer | Role |
|---|---|
| **Model card** (`src/model/qwen3_6_27b.cpp`) | the static compute graph: hand-written forward schedule |
| **Kernels** | generic API ⟂ specialized CUDA impls + a dispatcher that routes by dims/phase |
| **Infra** | memory pool, weight loader, KV/state allocators, workspace arena, streams |

Freedom at the API surface; 固化 (frozen) in the implementations, fusion, and the model card.

## How it fits together

```
bf16 safetensors + tokenizer ──(Python, offline)──> quantize + relayout/pack ──> ONE v4.2 file
                                                                          │
                                                (C++/CUDA runtime) selective staged load + run
                                                                          │
 text/image/video -> native processor -> Vision + mixed embeddings -> text forward -> sampler -> text
```

## Build (intended)

Requires CUDA 13.1+ (sm_120), gcc 13+, CMake 3.28+, FFmpeg 6 development libraries
(`libavformat`, `libavcodec`, `libavutil`, `libswscale`) and libcurl 7.85+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Usage

```bash
./build/src/qus /path/to/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --prompt "用三句话解释 prefill 和 decode 的区别。" \
  --max-new 128
```

The v4.2 artifact embeds `tokenizer.json`, `merges.txt`, and `generation_config.json`; runtime
commands do not accept a tokenizer directory. The loader validates the 4 KiB header before CUDA
initialization, reads only the bounded catalog/tokenizer prefix, and uploads only requested modules.

For multimodal inference, pass a structured messages file directly to `qus`:

```bash
./build/src/qus MODEL.qus --messages messages.json --no-thinking --max-new 256
```

The same native multimodal preprocessor can also be exercised independently:

```bash
./build/src/qus-preprocess MODEL.qus messages.json preprocess.json patches.f32 --no-thinking
```

It reads the tokenizer from the q5090 catalog, supports structured text/image/video messages, and
emits expanded token IDs, token types, three-axis positions, `rope_delta`, grids, timestamps and the
row-major `[P,1536]` patch buffer. `qus` and `qus-serve` feed those outputs through the native Vision
tower, scatter the merged `[V,5120]` embeddings into the text prompt, and continue decode with the
correct MRoPE offset.

`bench/qus_bench` is the real-weight throughput tool (llama-bench-style prefill/decode rates); see
[`bench/README.md`](bench/README.md).

The OpenAI-compatible server includes best-effort function tool calling; see
[`docs/non-strict-tool-calling.md`](docs/non-strict-tool-calling.md).
Its HTTP request body is capped at 384 MiB by default; use `--max-request-mib N` to set a tighter
deployment limit. Memory-heavy media preprocessing admits one in-flight request at a time.

## Toolchain target

NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB) · CUDA 13.1 · gcc 13.3 · CMake 3.28 · Linux/WSL2.

## Documentation

- [`docs/design.md`](docs/design.md) — master design & goal document (scope, boundaries,
  architecture, data flow, memory, numerics, roadmap).
- [`docs/m3-readiness.md`](docs/m3-readiness.md) — completed M2.8 gate evidence and starting point
  for M3 planning.
- [`docs/qwen3.6-27b-architecture.md`](docs/qwen3.6-27b-architecture.md) — exact model
  architecture reference: per-layer parameters, computation flow, Gated-DeltaNet math,
  operator inventory, and runtime tensor-transform ownership.
- [`docs/2026-07-12-qwen3-6-vision-engine-integration.md`](docs/2026-07-12-qwen3-6-vision-engine-integration.md)
  — native Vision tower, text/MRoPE integration, serving lifecycle, and numerical evidence.
- [`docs/q5090_packed_file_format_v4.md`](docs/q5090_packed_file_format_v4.md) — canonical
  packed-weight ABI consumed by the C++ runtime.
- [`docs/archive/pre-optimization/`](docs/archive/pre-optimization/) — completed pre-optimization
  plans, specs, and historical M2.8 standard material.
- [`tools/q5090_convert`](tools/q5090_convert) — canonical safetensors-to-q5090 converter.
- [`tools/q5090`](tools/q5090) — budget-aware Python reference model and q5090 diagnostics.
