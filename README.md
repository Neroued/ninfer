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
runtime loads it directly without runtime repacking. The accepted `.ninfer` container and the new
multi-target engine architecture are not implemented yet. The Qwen3.6-35B-A3B document is a model
reference, not a claim of runtime support.

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
- a Python q5090 reference model, converter, structural verifier, numerical diagnostics, and
  real-weight/per-operator benchmarks;
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
| Artifact | `q5090_w4g64_mixed_v4_2` |
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

## Build a q5090 artifact

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

## Run the CLI

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
Performance claims require a reproducible command, q5090 identity, git state, profiler evidence for
the changed kernels, and before/after real-weight benchmark reports. See
[`bench/README.md`](bench/README.md).

## Architecture

```text
BF16 checkpoint + tokenizer
          │
          ▼
offline converter ───────────────► q5090 v4.2 artifact
                                          │
                                          ▼
L0 storage/state ─► L1 CUDA operators ─► L2 Text/MTP/Vision schedules
                                                   │
                                                   ▼
                                      Engine + CUDA Graph runtime
                                                   │
                          ┌────────────────────────┴──────────────────────┐
                          ▼                                               ▼
                 text/media frontend                              CLI / HTTP serving
```

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
- [`docs/ninfer-naming.md`](docs/ninfer-naming.md) — canonical project name, reserved
  not-yet-implemented `.ninfer` extension, and naming-cutover status;
- [`docs/ninfer-project-positioning.md`](docs/ninfer-project-positioning.md) — project mission,
  target-selection policy, performance priorities, and non-goals;
- [`docs/ninfer-tensor-formats.md`](docs/ninfer-tensor-formats.md),
  [`docs/ninfer-container-format.md`](docs/ninfer-container-format.md), and
  [`docs/ninfer-engine-architecture.md`](docs/ninfer-engine-architecture.md) — accepted designs
  pending implementation.

Completed plans, retired formats, implementation reports, and profiler evidence live under
[`docs/archive/`](docs/archive/). Archived documents are historical records, not current design
entrypoints.
