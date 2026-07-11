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
the 13 public L1 operator APIs and implementations, the L2 `Qwen3_6_27B` model card, the `Engine`,
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
attention** (16 layers), SwiGLU MLP, vocab 248320. v1 freezes to the **text decoder only**
(vision tower and the MTP layer ship in the checkpoint but are deferred).

## Scope (v1)

- Text-only; the primary `qus` binary accepts Qwen3.6 chat text and prints decoded text.
  The runtime engine and benchmark/parity tools still use token ids internally.
- **Greedy** decoding, **bf16 KV**, current M2.8 official **max_ctx = 8192**.
- **W4A16**: 4-bit weights, bf16 activations.
- Single sequence (batch = 1), single GPU.

128K context is a later target, not part of the current M2.8 gate. Deferred (in order):
MTP speculative decode → fp8/fp4 prefill → 128K/256K context and fp8-KV → full sampler →
vision → multi-GPU/batching.

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
           text/messages -> C++ Qwen tokenizer/chat template -> token ids -> forward -> greedy -> ids -> text
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

The native multimodal preprocessor can be exercised independently while the C++ Vision forward is
being implemented:

```bash
./build/src/qus-preprocess MODEL.qus messages.json preprocess.json patches.f32 --no-thinking
```

It reads the tokenizer from the q5090 catalog, supports structured text/image/video messages, and
emits expanded token IDs, token types, three-axis positions, `rope_delta`, grids, timestamps and the
row-major `[P,1536]` patch buffer. The main inference binary rejects media until those outputs are
connected to the Vision tower, preventing a single unexpanded placeholder from being executed as a
text-only prompt.

`bench/qus_bench` is the real-weight throughput tool (llama-bench-style prefill/decode rates); see
[`bench/README.md`](bench/README.md).

The OpenAI-compatible server includes best-effort function tool calling; see
[`docs/non-strict-tool-calling.md`](docs/non-strict-tool-calling.md).

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
- [`docs/q5090_packed_file_format_v4.md`](docs/q5090_packed_file_format_v4.md) — canonical
  packed-weight ABI consumed by the C++ runtime.
- [`docs/archive/pre-optimization/`](docs/archive/pre-optimization/) — completed pre-optimization
  plans, specs, and historical M2.8 standard material.
- [`tools/q5090_convert`](tools/q5090_convert) — canonical safetensors-to-q5090 converter.
- [`tools/q5090`](tools/q5090) — budget-aware Python reference model and q5090 diagnostics.
