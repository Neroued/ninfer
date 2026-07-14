# NInfer

> Selected checkpoints. Selected GPUs. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for users who want the highest practical local
inference performance from one GPU. It supports a small, explicitly registered set of exact model
checkpoints and GPU targets. Each pair is implemented as a concrete compiled target; NInfer is not
a generic model runtime, compatibility layer, or model zoo.

The currently registered target is exactly **Qwen3.6-27B on NVIDIA RTX 5090**. The current runtime
owns one resident sequence and executes one active request at a time. Decode efficiency is the
primary goal, followed by prefill throughput and time to first token. Limited continuous batching
is a future direction, not a current capability or a large-scale-serving goal.

## Capabilities

The delivered target includes:

- the complete 64-layer hybrid Text decoder: 48 Gated-DeltaNet layers and 16 GQA layers;
- the one-layer MTP draft model, full and optimized proposal heads, and eager or CUDA Graph decode;
- the 27-layer Vision tower, patch merger, image/video preprocessing, embedding injection, and
  three-axis MRoPE;
- chunked prefill, single-token decode, and target-shaped CUDA kernels;
- BF16 and INT8 group-64 KV-cache storage;
- greedy decoding and configurable temperature/top-k/top-p/min-p/penalty sampling;
- prefix reuse for compatible text requests;
- text and structured multimodal CLI input;
- OpenAI Chat Completions and Anthropic Messages endpoints, including streaming, token counting,
  usage reporting, reasoning/content channels, and prompt-and-parse tool calls;
- a public opaque C++ `Engine` API used by the CLI, server, and real-weight benchmark;
- the native `.ninfer` converter, inspector, verifier, Python Text/Vision/MTP reference, and
  target-private parity diagnostics.

NInfer does not currently support another checkpoint, another GPU, concurrent request execution,
multi-GPU execution, or distributed serving. Context capacity is selected when the Engine is
constructed and is bounded by the configured KV format and available GPU memory.

## Registered target

| Item | Current value |
|---|---|
| Model | Qwen3.6-27B (`qwen3_5` checkpoint architecture) |
| GPU | RTX 5090, Blackwell, `sm_120a`, 32 GB |
| Runtime target key | `qwen3_6_27b_rtx5090` |
| Artifact | `qwen3_6_27b_rtx5090.ninfer` |
| Workload | one resident sequence, one active request |
| Activations | BF16 |
| KV cache | BF16 or INT8 group-64 |
| Primary metric | single-stream decode tokens/second |

The Qwen3.6-35B-A3B architecture document is a model reference. It is not a registered runtime
target.

## Build

The supported toolchain is CUDA 13.1, GCC 13, and CMake 3.28 or newer. The build also requires
PkgConfig, FFmpeg 6 development libraries (`libavformat`, `libavcodec`, `libavutil`, `libswscale`),
and libcurl 7.85 or newer.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j
```

The main products are:

| Product | Purpose |
|---|---|
| `build/apps/ninfer` | text and multimodal generation CLI |
| `build/apps/ninfer-serve` | OpenAI/Anthropic-compatible HTTP server |
| `build/bench/ninfer_bench` | public-Engine prefill/decode benchmark |
| `build/bench/ninfer_*_bench` | CUDA operator microbenchmarks |
| `build/tools/ninfer-qwen3_6_27b-dump` | target-private activation diagnostic |

Executable `--help` output is the exact option reference.

## Create the `.ninfer` artifact

Conversion is offline and reads the original BF16 checkpoint directly:

```bash
python -m tools.convert.qwen3_6_27b_rtx5090.convert \
  --model /path/to/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b_rtx5090.ninfer

python -m tools.artifact.inspect \
  out/qwen3_6_27b_rtx5090.ninfer --objects

python -m tools.convert.qwen3_6_27b_rtx5090.verify \
  out/qwen3_6_27b_rtx5090.ninfer \
  --model /path/to/Qwen3.6-27B/base-hf-bf16
```

The artifact contains the complete registered Text, MTP, Vision, draft-head, tokenizer, template,
and generation-resource inventory. The C++ Engine and Python reference both consume this artifact;
the source checkpoint is not accessed during inference.

The complete Python correctness reference is available independently:

```bash
python -m tools.reference.qwen3_6_27b_rtx5090 \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  --prompt "用三句话解释 prefill 和 decode 的区别。" \
  --decode 128 --mtp-draft-tokens 3
```

## Run the CLI

Text prompt:

```bash
./build/apps/ninfer out/qwen3_6_27b_rtx5090.ninfer \
  --prompt "用三句话解释 prefill 和 decode 的区别。" \
  --max-new 128
```

Structured text/image/video messages:

```bash
./build/apps/ninfer out/qwen3_6_27b_rtx5090.ninfer \
  --messages messages.json --no-thinking --max-new 256
```

CUDA Graph decode is enabled by default. Enable MTP with `--mtp-draft-tokens N`, where `N` is in
`1..5`; add `--lm-head-draft` to select the optimized proposal head. Use `--kv-dtype int8` for INT8
KV storage, `--greedy` for argmax decoding, and `--max-context` to choose the Engine capacity.

Media message parts accept local paths, HTTP(S) URLs, and base64 data URIs. The CLI acquires the
bytes before the target Frontend performs checkpoint-specific image/video preparation.

## Run the server

```bash
./build/apps/ninfer-serve out/qwen3_6_27b_rtx5090.ninfer \
  --host 127.0.0.1 --port 8080 --max-context 8192
```

The server exposes:

- `GET /health`;
- `GET /v1/models` and `GET /v1/models/{id}`;
- `POST /v1/chat/completions`;
- `POST /v1/messages`;
- `POST /v1/messages/count_tokens`.

See [`docs/serving.md`](docs/serving.md) for protocol, streaming, sampling, multimodal, and tool-call
behavior.

## Benchmark

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  -p 512,2048 -n 128 -pg '2048,128' -r 5 --warmup 1
```

`ninfer_bench` drives the same public Engine route as the CLI and server. It reports prefill and
decode separately, supports BF16/INT8 KV and ordinary/MTP graph or eager modes, and emits table,
JSON, or CSV. See [`bench/README.md`](bench/README.md).

## Public Engine

The installed product surface is [`include/ninfer/engine.h`](include/ninfer/engine.h) plus owning
host values in [`include/ninfer/types.h`](include/ninfer/types.h). `Engine` constructs one selected
target from an `EngineOptions` value and `.ninfer` path. Requests follow one route:

```text
Engine::prepare(...) or prepare_tokens(...)
  -> opaque PreparedPrompt
  -> Engine::generate(...)
  -> GenerationResult + optional OutputSink deltas
```

Target selection, artifact binding, CUDA state, `LoadedModel`, `Frontend`, and `Program` remain
internal. The public API exposes load, memory, timing, and speculative summaries, not target phase
methods or mutable model state.

## Architecture

```text
BF16 checkpoint
  -> target converter
  -> qwen3_6_27b_rtx5090.ninfer
  -> ArtifactReader / ArtifactBinder / Materializer
  -> closed qwen3_6_27b_rtx5090 target package
       immutable LoadedModel + Frontend + one mutable Program
  -> common generated-token controller
  -> public Engine
       CLI / server / benchmark
```

The source and build boundaries are explicit:

- `src/core`, `src/artifact`, and `src/kernels` own reusable mechanisms;
- `src/text` and `src/media/decode` own checkpoint-neutral Unicode and media decoding;
- `src/product/media_acquire` owns path, URL, and data-URI acquisition for product entry points;
- `src/product/prompt_input` owns the shared CLI/diagnostic message-input adapter;
- `src/targets/qwen3_6_27b_rtx5090` owns exact checkpoint/GPU load, frontend, state, schedules, and
  target-only kernels/graph policy;
- `src/runtime` owns common request contracts, generation policy, and the public
  Engine implementation;
- `src/serve` owns HTTP schemas, translation, streaming, and transport;
- `apps/cli` and `apps/serve` are thin product entry points;
- `tools/convert`, `tools/reference`, and `tools/parity` are target-keyed offline tools.

## Documentation

Start at [`docs/README.md`](docs/README.md). Important active documents include:

- [`docs/ninfer-project-positioning.md`](docs/ninfer-project-positioning.md) — mission, target
  selection, workload, priorities, and non-goals;
- [`docs/design.md`](docs/design.md) — implemented ownership and runtime flow;
- [`docs/ninfer-engine-architecture.md`](docs/ninfer-engine-architecture.md) — detailed Engine,
  target-package, lifetime, and source-boundary decisions;
- [`docs/ninfer-container-format.md`](docs/ninfer-container-format.md),
  [`docs/ninfer-storage-layouts.md`](docs/ninfer-storage-layouts.md), and
  [`docs/ninfer-tensor-formats.md`](docs/ninfer-tensor-formats.md) — native artifact contracts;
- [`docs/qwen3.6-27b-ninfer-artifact.md`](docs/qwen3.6-27b-ninfer-artifact.md) — registered object
  inventory, source transforms, and binding obligations;
- [`docs/qwen3.6-27b-architecture.md`](docs/qwen3.6-27b-architecture.md) — Text/MTP/Vision model math;
- [`docs/kernel-development.md`](docs/kernel-development.md) — operator ownership and performance
  workflow;
- [`docs/serving.md`](docs/serving.md) — CLI and HTTP behavior.

Completed plans, retired implementations, and dated evidence belong under
[`docs/archive/`](docs/archive/); they do not define the current product.
