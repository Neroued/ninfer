# qwen3.6-ultraspeed

A from-scratch C++/CUDA inference engine built to run **Qwen3.6-27B** as fast as physically
possible for a **single user on a single RTX 5090**.

No dynamic compute graph, no general-purpose model zoo — the computation for this one model
is hand-written and specialized to the hardware. Weights are pre-quantized to a static 4-bit
format offline; the runtime just loads one fixed file and runs.

## Current status

M2 correctness baseline is implemented. M2.5 hardening/documentation sync is in progress, with
graph-readiness fixes, EOS handling, FileTap parity dumps, and hardening cleanup structure already
landed. M3 per-kernel performance optimization is the next main milestone.

Current code includes L0 infrastructure, the q5090 `WeightStore`/loader and unified `Weight` handle,
the 13 public L1 operator APIs and implementations, the L2 `Qwen3_6_27B` model card, the `Engine`,
and parity tooling for block parity and greedy token matching. No performance numbers are claimed
here; performance gating remains an M3+ activity.

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

- Text-only; **token-ids in → token-ids out** (tokenizer runs in Python, outside the engine).
- **Greedy** decoding, **128K** context, **bf16 KV**.
- **W4A16**: 4-bit weights, bf16 activations.
- Single sequence (batch = 1), single GPU.

Deferred (in order): MTP speculative decode → fp8/fp4 prefill → 256K/fp8-KV → full sampler →
C++ tokenizer → vision → multi-GPU/batching.

## Architecture (3 layers)

| Layer | Role |
|---|---|
| **Model card** (`src/model/qwen3_6_27b.cpp`) | the static compute graph: hand-written forward schedule |
| **Kernels** | generic API ⟂ specialized CUDA impls + a dispatcher that routes by dims/phase |
| **Infra** | memory pool, weight loader, KV/state allocators, workspace arena, streams |

Freedom at the API surface; 固化 (frozen) in the implementations, fusion, and the model card.

## How it fits together

```
bf16 safetensors ──(Python, offline)──> quantize + relayout/pack ──> ONE fixed weight file
                                                                          │
                                                          (C++/CUDA runtime) mmap + load + run
                                                                          │
                                              token-ids in ──> forward ──> greedy ──> token-ids out
```

## Build (intended)

Requires CUDA 13.1+ (sm_120), gcc 13+, CMake 3.28+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Toolchain target

NVIDIA RTX 5090 (Blackwell, sm_120, 32 GB) · CUDA 13.1 · gcc 13.3 · CMake 3.28 · Linux/WSL2.

## Documentation

- [`docs/design.md`](docs/design.md) — master design & goal document (scope, boundaries,
  architecture, data flow, memory, numerics, roadmap).
- [`docs/qwen3.6-27b-architecture.md`](docs/qwen3.6-27b-architecture.md) — exact model
  architecture reference: per-layer parameters, computation flow, Gated-DeltaNet math,
  operator inventory, and runtime tensor-transform ownership.
- [`docs/q5090_packed_file_format_v1.md`](docs/q5090_packed_file_format_v1.md) — canonical
  packed-weight ABI consumed by the C++ runtime.
- [`tools/q5090_convert`](tools/q5090_convert) — canonical safetensors-to-q5090 converter.
