# System Design

> Status: current system architecture.
>
> This document defines repository boundaries, ownership, runtime flows, and supported scope. The
> implemented model mathematics belong in
> [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md); the exact 35B-A3B source profile is
> described separately in
> [`qwen3.6-35b-a3b-architecture.md`](qwen3.6-35b-a3b-architecture.md). The packed artifact contract
> belongs in
> [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md).

## 1. Purpose

The currently implemented NInfer system is a from-scratch C++/CUDA inference engine for one fixed
deployment:

- Qwen3.6-27B;
- one RTX 5090 (`sm_120a`, 32 GB);
- one user and one active sequence;
- BF16 activations with offline-packed low-bit weights;
- maximum single-stream decode throughput, with prefill/TTFT as the secondary target.

For this implementation, the fixed target is an architectural constraint, not a generic-runtime
starting point. It deliberately uses compile-time dimensions, hand-written model schedules,
model-shape CUDA kernels, a single artifact ABI, and direct frontends. The broader project mission
and target-selection policy are governed by
[`ninfer-project-positioning.md`](ninfer-project-positioning.md).

## 2. Scope

### Supported by the current implementation

- text, image, and video chat input;
- native tokenization, chat templates, media decoding/preprocessing, Vision inference, and Text
  inference;
- chunked prefill and autoregressive decode;
- eager and CUDA Graph decode;
- optional one-layer MTP speculative decoding with up to five draft tokens per round;
- optional shortlisted Q4 draft `lm_head` for proposal sites;
- greedy and sampled generation;
- BF16 or INT8 KV cache;
- command-line generation, OpenAI Chat Completions, and Anthropic Messages;
- offline q5090 conversion plus Python reference/diagnostic tooling;
- real-weight and per-operator benchmarking.

### Deliberately unsupported by the current implementation

- models other than the fixed Qwen3.6-27B checkpoint architecture;
- batching or concurrent GPU execution;
- tensor/pipeline parallelism and multi-GPU inference;
- dynamic computation graphs or runtime model discovery;
- runtime weight repacking or alternate weight loaders;
- compatibility parsing for retired q5090 formats;
- server-side tool execution or constrained JSON decoding;
- a qualified 128K/256K operating point without explicit memory, numerical, and performance
  evidence.

## 3. Component ownership

| Component | Paths | Owns |
|---|---|---|
| L0 infrastructure | `include/ninfer/core`, `src/core` | device/stream setup, tensors, arenas, q5090 parsing/loading, KV cache, GDN state |
| L1 operators | `include/ninfer/kernels`, `src/kernels` | mathematical operator APIs, dispatch, launchers, specialized CUDA kernels |
| L2 model | `include/ninfer/model`, `src/model` | frozen dimensions, weight binding, Text/MTP/Vision schedules, model-private CUDA helpers |
| Runtime | `include/ninfer/runtime`, `src/runtime` | resource lifetime, Engine API, prefix reuse, eager/graph execution, MTP rounds, sampling state |
| Text frontend | `include/ninfer/text`, `src/text` | tokenizer, chat template, CLI translation, output decoding |
| Media frontend | `include/ninfer/media`, `src/media`, model processor | local/remote media acquisition, decode, resize, normalization, patch construction, budgets |
| Serving | `include/ninfer/serve`, `src/serve` | HTTP transport, OpenAI/Anthropic schemas, request translation, streaming, tool-call parsing |
| Offline tools | `tools/q5090_convert`, `tools/q5090`, `tools/parity` | artifact production, Python reference, structural/numerical diagnostics |
| Measurement | `bench`, `tools/bench` | real-weight throughput, per-operator benchmarks, corpus tooling, report schemas |

The layer names L0/L1/L2 describe the inference core. Runtime and frontends are explicit peers above
that core, not hidden responsibilities of the model card.

## 4. Fixed artifact boundary

The offline converter is the only path from the source BF16 checkpoint to runtime weights:

```text
HF safetensors + tokenizer assets
              │
              ▼
tools.q5090_convert: quantize + slice + fuse + pack + verify
              │
              ▼
one q5090 v4.2 .qus artifact + diagnostic sidecar manifest
```

The artifact contains four logical module kinds:

- `TEXT_CORE` — embeddings, 64 decoder layers, final norm, and full `lm_head`;
- `MTP_DRAFT` — the one-layer MTP module;
- `VISION_ENCODER` — patch embedding, 27 Vision blocks, and merger;
- `LM_HEAD_DRAFT` — optional Q4 shortlisted proposal head and vocab-id map.

It also embeds the three CPU tokenizer assets. `WeightStore` validates the 4 KiB header before CUDA
initialization, reads the bounded metadata/tokenizer prefix, and selectively uploads requested
module payloads. Quantized weights remain packed on device; CUDA kernels decode them while computing.

The runtime accepts only the active v4.2 contract. Retired format documents are historical and do
not imply parser or converter compatibility.

## 5. Core layering

### 5.1 L0: storage and lifetime

L0 types expose bytes, shapes, and state without knowing model semantics:

- `DeviceContext` owns the CUDA device and stream;
- `DeviceArena` owns long-lived device allocations;
- `WorkspaceArena` is a mark/rewind bump allocator for transient tensors;
- `WeightStore` owns validated catalog metadata and resident module payloads;
- `KVCache` owns GQA K/V planes and logical position;
- `GdnState` owns convolution and FP32 recurrent state slots;
- `Tensor` and `Weight` describe dense/control storage and packed quantized views.

No hot-path operator performs hidden device allocation, filesystem access, or runtime repacking.

### 5.2 L1: operators and CUDA specialization

Public headers under `include/ninfer/kernels/` define mathematical contracts. Implementations follow
the normal path:

```text
public wrapper → private launcher → CUDA kernel
```

The linear family has a larger private subtree because codec, shape policy, GEMV, GEMM, and dense
reference paths are independently specialized. Dispatch keys are limited to facts that change the
implementation: qtype/layout, real `(N,K)` shape, token-count regime, state form, and KV dtype.

Fused public operators are used only when fusion changes the useful contract, such as shared-input
projections, projection-plus-SwiGLU, or projection-plus-residual. Model-only bookkeeping kernels stay
under `src/model` rather than becoming reusable L1 APIs.

See [`kernel-development.md`](kernel-development.md) for the source layout and verification rules.

### 5.3 L2: fixed schedules

`Qwen3_6_27B` binds catalog entries to fixed per-layer structs once during construction. Its hot
path then issues straight-line operator calls for the 64-layer Text decoder and MTP module without
name lookup or virtual dispatch.

`Qwen3_6_Vision` similarly binds the fixed 27-layer Vision tower. It runs before multimodal text
prefill and returns merged `[5120,V]` BF16 embeddings that remain valid in the Vision workspace until
the text prompt consumes them.

The model layer owns computation order and model-specific state semantics. It does not own physical
arenas, HTTP behavior, or kernel selection policy.

## 6. Load and resource lifecycle

`Engine::load()` performs one bounded setup sequence:

1. validate options, including prefill-chunk alignment, KV dtype, and MTP draft count;
2. parse the q5090 header/catalog/tokenizer assets;
3. allocate persistent weight, cache/state, workspace, and Vision resources;
4. upload `TEXT_CORE` and `VISION_ENCODER`, plus MTP/draft modules when requested;
5. construct and bind Text/MTP and Vision model objects;
6. initialize stable device-resident step and sampling buffers.

Default workspace capacity is a function of the configured prefill chunk, not total prompt length.
The prompt is processed in aligned chunks, and block-scoped workspace marks are rewound between
mixer and MLP phases. KV/state capacity is a function of configured context, KV dtype, and whether
MTP state is enabled.

## 7. Text request flow

```text
prompt/messages
    │
    ▼
chat template → embedded tokenizer → token ids
    │
    ▼
Engine prefill (chunked) → first sampled token
    │
    ▼
decode step/round → token stream decoder → stdout or HTTP stream
```

Text prefill resets the active sequence unless the runtime selects an explicit prefix-reuse path.
The final prompt column is normalized and projected by the full `lm_head`; the first generated token
is selected by the configured sampler. Subsequent calls continue from resident KV/GDN state.

The server can reuse a stable text prefix across turns. `prefill_cached()` compares the requested
tokens with the resident logical-token mirror and chooses one of three behaviors:

- append an exact resident prefix;
- restore the saved assistant-content boundary and append from there;
- perform a full reset prefill when reuse is unsafe.

Multimodal resident prefixes are not reused through this text-only mechanism.

## 8. Multimodal request flow

```text
structured messages
    │
    ├─ chat template/tokenizer ──────────────► expanded text ids and placeholders
    │
    └─ media fetch/decode/resize/normalize ─► [P,1536] FP32 patches
                                                │
                                                ▼
                                    27-layer Vision + merger
                                                │
                                                ▼
                                    [5120,V] visual embeddings
                                                │
                                                ▼
                         scatter into text embeddings + three-axis MRoPE
                                                │
                                                ▼
                                      ordinary Text prefill/decode
```

The processor produces token types, axis-major temporal/height/width positions, `rope_delta`, Vision
grids, timestamps, scatter spans, and the patch buffer. It enforces byte, decoded-pixel, patch,
Vision-token, attention-work, duration, and item-count budgets before expensive execution.

Vision runs only for prefill. Decode continues exclusively through Text state using the resulting
MRoPE offset.

## 9. Decode paths

### 9.1 Ordinary decode

With MTP disabled, one logical decode step consumes the current token, advances all Text layers,
projects the new hidden state through the full `lm_head`, samples the next token, and updates the
position/state buffers.

CUDA Graph execution is enabled by default. Stable device addresses and fixed step shapes allow the
runtime to warm the eager path, capture the record path, and replay it on later steps. Disabling
graphs changes execution mechanics, not model semantics.

### 9.2 MTP speculative decode

With `mtp_draft_tokens = k`, a round:

1. prepares up to `k` proposals with the one-layer MTP model;
2. evaluates the target model over the candidate window;
3. samples/compares the target distribution at each position;
4. accepts the valid prefix and commits the matching KV/GDN snapshot slot;
5. falls back to a normal target token when the proposal prefix ends or is rejected.

The target model always decides emitted tokens. The optional shortlisted `lm_head` is used only at
MTP proposal sites, so it can change acceptance and speed but not the target distribution. Both the
ordinary one-token path and the complete MTP round have eager and captured execution forms.

## 10. Sampling and output

Sampling configuration lives in stable device-resident storage so graph replay can read new request
values without changing captured addresses. Temperature zero is an exact greedy bypass. Nonzero
temperature uses the supported top-k/top-p/min-p and penalty semantics described in
[`serving.md`](serving.md).

Token selection happens on device. The runtime synchronizes and returns caller-visible tokens at the
Engine boundary, where CLI/server code applies stop-token handling and incremental UTF-8 decoding.

## 11. Precision and state

The main precision policy is:

- Q4/Q5/Q6 or W8G32 packed weights according to the active q5090 tensor assignment;
- BF16 activations and most persistent dense parameters;
- FP32 GDN recurrent state and control tensors where required by the model;
- BF16 or group-quantized INT8 GQA/MTP KV storage;
- FP32 accumulation where operator numerics require it.

The `Weight` handle is the precision seam: qtype/layout are data, while the model schedule invokes a
stable operator contract. Structural model flexibility is intentionally not a seam.

Persistent state is divided by lifetime:

- weights: process lifetime;
- KV/GDN/MTP state: active sequence lifetime;
- step/sampling buffers: Engine lifetime with stable addresses;
- Vision workspace: one multimodal prefill;
- Text workspace: block/chunk scoped with arena rewind;
- frontend request/media buffers: request lifetime.

## 12. Performance method

The primary objective is caller-visible single-stream decode throughput. The weight-bandwidth
roofline is a design anchor, not a published performance result. Optimizations are accepted using the
smallest evidence that covers their risk:

- numerical or behavior checks for the affected operator;
- per-operator benchmark and NCU evidence for kernel changes;
- NSYS for full inference, launch gaps, CPU/GPU overlap, and phase breakdown;
- before/after `ninfer_bench` reports for relevant prefill/decode matrices;
- `compute-sanitizer` when memory access or lifetime is at risk.

Historical profiles live in the archive. They are not silently promoted to current performance
claims after the implementation, artifact, or measurement contract changes.

## 13. Source map

| Concern | Primary implementation |
|---|---|
| fixed dimensions | `include/ninfer/model/config.h` |
| Text/MTP schedule and binding | `include/ninfer/model/model.h`, `src/model/qwen3_6_27b.cpp` |
| Vision schedule | `include/ninfer/model/vision.h`, `src/model/qwen3_6_vision.cpp` |
| multimodal processor | `include/ninfer/model/processor.h`, `src/model/processor.cpp`, `src/media` |
| Engine and prefix reuse | `include/ninfer/runtime/engine.h`, `src/runtime/engine.cpp` |
| graph capture/replay | `include/ninfer/runtime/decode_graph.h`, `src/runtime/decode_graph.cpp` |
| q5090 parser/loader | `include/ninfer/core/weight_store*.h`, `src/core/weight_store*.cpp` |
| public operators | `include/ninfer/kernels` |
| CUDA implementations | `src/kernels` |
| CLI and text frontend | `include/ninfer/text`, `src/text`, `src/main.cpp` |
| HTTP serving | `include/ninfer/serve`, `src/serve` |
| converter/reference | `tools/q5090_convert`, `tools/q5090` |
| measurement | `bench`, `tools/bench` |
