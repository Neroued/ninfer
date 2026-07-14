# NInfer

> Selected checkpoints. Selected GPUs. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for users who want the highest practical local
inference performance from one GPU. It deliberately supports a small, explicitly selected set of
exact model checkpoints and GPU targets. Each supported pair is implemented and optimized as a
concrete target; NInfer is not a generic model runtime, compatibility layer, or model zoo.

The current delivered target is exactly **Qwen3.6-27B on one NVIDIA RTX 5090**. It serves one active
request on one GPU and prioritizes decode efficiency, then prefill throughput and time to first
token, while using single-machine KV-cache policy to push useful context capacity toward the
checkpoint's native limit. Limited continuous batching is a future direction, not a current
capability or a large-scale-serving goal.

For the current target, the model schedule is hand-written, CUDA kernels are specialized for fixed
shapes, and the offline converter produces one self-contained q5090 v4.2 `.qus` artifact. The
current C++ Engine loads that artifact directly without runtime repacking. In parallel, the native
`.ninfer` converter, Python and narrow C++ readers, verifier, and complete Python Text/Vision/MTP
reference are implemented. `.ninfer` is not yet a C++ Engine input: the new multi-target C++ engine
architecture remains future work. The Qwen3.6-35B-A3B document is a model reference, not a claim of
runtime support.

## Current capabilities

The current implementation includes:

- the complete 64-layer hybrid text decoder: 48 Gated-DeltaNet layers and 16 GQA layers;
- the one-layer MTP draft model with eager and CUDA Graph speculative decode rounds;
- the 27-layer Vision tower, patch merger, native image/video preprocessing, embedding injection,
  and three-axis MRoPE;
- chunked prefill and single-token decode with shape-specialized CUDA kernels;
- BF16 and INT8 KV-cache storage;
- greedy decoding and Qwen thinking-oriented sampling;
- an optional frequency-shortlisted Q4 draft `lm_head` for faster MTP proposals;
- text and structured multimodal CLI input;
- OpenAI Chat Completions and Anthropic Messages HTTP endpoints, including streaming and
  best-effort function tool calling;
- the current q5090 converter, legacy Python reader/codec, parity diagnostics, and
  real-weight/per-operator benchmarks used by the `.qus` Engine route;
- a native `.ninfer` converter, generic Python reader/inspector, narrow C++ reader, target verifier,
  and complete Python Text/Vision/MTP reference with text, image, video, and speculative generation;
- a configurable capability-evaluation coordinator for local or online OpenAI-compatible targets,
  with EvalScope adapters, progress, persistent logs, resume, and normalized reports.

The current implementation does not support other checkpoints, batching, multi-GPU inference, or
old q5090 formats. Context capacity is configured at process start and bounded by GPU memory. The
CLI defaults to 2,048 tokens and the server defaults to 8,192; larger contexts require an explicit
setting and are not an implied 128K/256K qualification claim.

## Fixed target

| Item | Target |
|---|---|
| Model | Qwen3.6-27B (`qwen3_5` checkpoint architecture) |
| GPU | RTX 5090, Blackwell, `sm_120a`, 32 GB |
| Workload | one sequence, batch size 1 |
| Primary metric | single-stream decode tokens/second |
| Secondary metric | prefill throughput and time to first token |
| Current C++ Engine artifact | `q5090_w4g64_mixed_v4_2` (`.qus`) |
| Native reference artifact | `qwen3_6_27b_rtx5090.ninfer` |
| Activations | BF16 |
| KV cache | BF16 or INT8 |

## Build

The supported toolchain is CUDA 13.1, GCC 13, and CMake 3.28 or newer. The build also requires
PkgConfig, FFmpeg 6 development libraries (`libavformat`, `libavcodec`, `libavutil`, `libswscale`),
and libcurl 7.85 or newer.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j
```

The main binaries are:

| Binary | Purpose |
|---|---|
| `build/src/ninfer` | text or multimodal generation CLI |
| `build/src/ninfer-serve` | OpenAI/Anthropic-compatible HTTP server |
| `build/src/ninfer-preprocess` | native multimodal preprocessing diagnostic |
| `build/src/ninfer-vision-dump` | Vision activation dump |
| `build/bench/ninfer_bench` | real-weight prefill/decode benchmark |
| `build/bench/ninfer_*_bench` | per-operator CUDA benchmarks |

Use `ninfer`, `ninfer-serve`, and benchmark `--help` output as the authoritative option reference.
The two diagnostic executables print their usage when required positional arguments are missing.

## Build artifacts

### Current C++ Engine artifact

Conversion is offline and requires the original BF16 checkpoint and tokenizer assets:

```bash
python -m tools.q5090_convert.convert \
  --model /path/to/Qwen3.6-27B/base-hf-bf16 \
  --tokenizer /path/to/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus

python -m tools.q5090_convert.verify \
  out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus --quick
```

The artifact contains packed Text, MTP, Vision, optional draft-head weights, plus
`tokenizer.json`, `merges.txt`, and `generation_config.json`. See
[`tools/q5090_convert/README.md`](tools/q5090_convert/README.md) for conversion and verification
details.

### Native `.ninfer` artifact and Python reference

The native converter reads the same BF16 checkpoint directly and emits the complete registered
Text, draft-head, MTP, Vision, and frontend object inventory:

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

The complete Python reference consumes only the resulting `.ninfer` artifact at inference time. It
uses the embedded frontend resources through the existing Hugging Face libraries and supports Text,
image/video Vision, full and shortlisted MTP proposal heads, and speculative generation:

```bash
python -m tools.reference.qwen3_6_27b_rtx5090.cli \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  --prompt "用三句话解释 prefill 和 decode 的区别。" \
  --decode 128 --mtp-draft-tokens 3
```

Use `--messages messages.json` for structured text/image/video input. This Python path is the
correctness reference for the native artifact route; it is not the new C++ Engine implementation.

## Run the current C++ CLI

Text prompt:

```bash
./build/src/ninfer out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --prompt "用三句话解释 prefill 和 decode 的区别。" \
  --max-new 128
```

Structured text/image/video messages:

```bash
./build/src/ninfer out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --messages messages.json --no-thinking --max-new 256
```

CUDA Graph decode is enabled by default. Enable MTP with `--mtp-draft-tokens N`, where `N` is in
`1..5`; add `--lm-head-draft` to use the embedded shortlisted proposal head. Use
`--kv-dtype int8` for INT8 KV storage and `--greedy` for deterministic argmax diagnostics.

## Run the server

```bash
./build/src/ninfer-serve out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --host 127.0.0.1 --port 8080 --max-context 8192
```

The server exposes:

- `GET /health`;
- `GET /v1/models` and `GET /v1/models/{id}`;
- `POST /v1/chat/completions`;
- `POST /v1/messages`;
- `POST /v1/messages/count_tokens`.

The request body limit defaults to 384 MiB and is enforced before JSON parsing. Media-heavy
preprocessing is admitted one request at a time. Function tool calling is prompt-and-parse based;
the server does not execute tools or provide constrained JSON decoding. See
[`docs/serving.md`](docs/serving.md) for request and generation semantics.

## Benchmark

```bash
./build/bench/ninfer_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  -p 512,2048 -n 128 -pg 2048,128 -r 5 --warmup 1
```

`ninfer_bench` reports prefill and decode separately and can emit table, JSON, or CSV output.
Performance reports record the command, q5090 identity, relevant git state, profiler evidence for
the changed kernels, and before/after real-weight results so the measurements can be interpreted.
Those records are descriptive evidence, not a fixed-worktree or byte-reproducibility gate. See
[`bench/README.md`](bench/README.md).

## Architecture

```text
BF16 checkpoint + tokenizer
          │
          ├── q5090 converter ──► q5090 v4.2 .qus ──► current C++ Engine
          │                                              │
          │                              CUDA Text/MTP/Vision + CLI/server
          │
          └── native converter ─► .ninfer v1 ─────────► complete Python reference
                                     │                    Text/Vision/MTP
                                     ├── Python reader/inspector/verifier
                                     └── narrow C++ reader (not Engine integration)
```

The accepted multi-target C++ architecture will eventually replace the first route's q5090 loader
with compiled exact-target packages over `.ninfer`; that Engine migration is not implemented yet.

- **L0** owns devices, arenas, tensors, q5090 loading, KV cache, and recurrent state.
- **L1** owns public operator contracts, dispatch, CUDA launchers, and specialized kernels.
- **L2** owns the fixed Qwen3.6 Text, MTP, and Vision schedules and weight bindings.
- **Runtime** owns resource lifetimes, prefix reuse, sampling state, eager/CUDA Graph execution,
  and MTP rounds.
- **Frontends and serving** own tokenization, chat templates, media processing, request translation,
  streaming, and protocol behavior.

## Documentation

Start at [`docs/README.md`](docs/README.md). The active project documents are:

- [`docs/design.md`](docs/design.md) — current system design and ownership boundaries;
- [`docs/qwen3.6-27b-architecture.md`](docs/qwen3.6-27b-architecture.md) — fixed model, MTP, and
  Vision computation reference;
- [`docs/qwen3.6-35b-a3b-architecture.md`](docs/qwen3.6-35b-a3b-architecture.md) — exact 35B-A3B
  source-checkpoint Text/MoE/MTP/Vision reference (not current runtime-support status);
- [`docs/q5090_packed_file_format_v4.md`](docs/q5090_packed_file_format_v4.md) — canonical v4.2
  artifact ABI;
- [`docs/kernel-development.md`](docs/kernel-development.md) — kernel layering and verification;
- [`docs/serving.md`](docs/serving.md) — CLI, sampling, multimodal, and HTTP behavior;
- [`docs/ninfer-naming.md`](docs/ninfer-naming.md) — canonical project name, `.ninfer` extension,
  and naming-cutover status;
- [`docs/ninfer-project-positioning.md`](docs/ninfer-project-positioning.md) — project mission,
  target-selection policy, performance priorities, and non-goals;
- [`docs/ninfer-tensor-formats.md`](docs/ninfer-tensor-formats.md),
  [`docs/ninfer-storage-layouts.md`](docs/ninfer-storage-layouts.md),
  [`docs/ninfer-container-format.md`](docs/ninfer-container-format.md),
  and [`docs/qwen3.6-27b-ninfer-artifact.md`](docs/qwen3.6-27b-ninfer-artifact.md) — the implemented
  native artifact contracts used by the converter, readers, verifier, and Python reference;
- [`docs/ninfer-engine-architecture.md`](docs/ninfer-engine-architecture.md) — the accepted
  multi-target C++ Engine design, still pending implementation.

Completed plans, retired formats, implementation reports, and profiler evidence live under
[`docs/archive/`](docs/archive/). Archived documents are historical records, not current design
entrypoints.
