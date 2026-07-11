# qwen3.6-ultraspeed

A from-scratch C++/CUDA inference engine specialized for **Qwen3.6-27B** on one
**NVIDIA RTX 5090**. The runtime is built for one user, one sequence, and one GPU; it is not a
general-purpose model runtime or compatibility layer.

The model schedule is hand-written, the CUDA kernels are specialized for the fixed model shapes,
and the offline converter produces one self-contained q5090 v4.2 artifact. The C++ runtime loads
that artifact directly without runtime repacking or a second weight format.

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
  real-weight/per-operator benchmarks.

The project does not support other models, batching, multi-GPU inference, or old q5090 formats.
Context capacity is configured at process start and bounded by GPU memory. The CLI defaults to
2,048 tokens and the server defaults to 8,192; larger contexts require an explicit setting and are
not an implied 128K/256K qualification claim.

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
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The main binaries are:

| Binary | Purpose |
|---|---|
| `build/src/qus` | text or multimodal generation CLI |
| `build/src/qus-serve` | OpenAI/Anthropic-compatible HTTP server |
| `build/src/qus-preprocess` | native multimodal preprocessing diagnostic |
| `build/src/qus-vision-dump` | Vision activation dump |
| `build/bench/qus_bench` | real-weight prefill/decode benchmark |
| `build/bench/qus_*_bench` | per-operator CUDA benchmarks |

Use each binary's `--help` output as the authoritative option reference.

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
./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --prompt "用三句话解释 prefill 和 decode 的区别。" \
  --max-new 128
```

Structured text/image/video messages:

```bash
./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  --messages messages.json --no-thinking --max-new 256
```

CUDA Graph decode is enabled by default. Enable MTP with `--mtp-draft-tokens N`, where `N` is in
`1..5`; add `--lm-head-draft` to use the embedded shortlisted proposal head. Use
`--kv-dtype int8` for INT8 KV storage and `--greedy` for deterministic argmax diagnostics.

## Run the server

```bash
./build/src/qus-serve out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
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
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus \
  -p 512,2048 -n 128 -pg 2048,128 -r 5 --warmup 1
```

`qus_bench` reports prefill and decode separately and can emit table, JSON, or CSV output.
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
- [`docs/q5090_packed_file_format_v4.md`](docs/q5090_packed_file_format_v4.md) — canonical v4.2
  artifact ABI;
- [`docs/kernel-development.md`](docs/kernel-development.md) — kernel layering and verification;
- [`docs/serving.md`](docs/serving.md) — CLI, sampling, multimodal, and HTTP behavior.

Completed plans, retired formats, implementation reports, and profiler evidence live under
[`docs/archive/`](docs/archive/). Archived documents are historical records, not current design
entrypoints.
