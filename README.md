# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for two exact Qwen3.6 checkpoints on a single
NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs.

NInfer deliberately supports a closed set of model artifacts instead of acting as a general model
runtime:

| Model | NInfer artifact | Size | SHA-256 |
|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 bytes (21.22 GiB) | `5194407dd6d3092b8c2f81ce41e014b50ca0d6f1ba4e5d8c1492b8652bfa267f` |

## Performance

Serving performance was measured on an RTX 5090 with INT8 group-64 KV cache, CUDA Graphs, a 1,024-
token prefill chunk, and a maximum context of 262,144 tokens. Each reported fixture uses five fixed
seeds after one warm-up. The two registered targets are reported independently and are not
cross-target comparisons.

**Qwen3.6-35B-A3B**

- MTP0 at a 7,680-token prompt: **15,544.3 prefill tok/s** and **271.1 decode tok/s**.
- MTP0 at a 260,096-token prompt: **5,157.1 prefill tok/s** and **188.2 decode tok/s**.
- MTP3 long reasoning: **542.8–634.3 decode tok/s** with **73.0–82.7% acceptance**.
- MTP3 structured output: **661.2 decode tok/s**, **87.2% acceptance**, and **3.62 tokens/round**.

**Qwen3.6-27B**

- MTP0 at a 7,680-token prompt: **3,218.1 prefill tok/s** and **77.6 decode tok/s**.
- MTP0 at a 260,096-token prompt: **1,614.8 prefill tok/s** and **54.8 decode tok/s**.
- MTP3 long reasoning: **158.7–174.2 decode tok/s** with **73.3–79.9% acceptance**.
- MTP3 structured output: **189.1 decode tok/s**, **88.9% acceptance**, and **3.67 tokens/round**.

See [Performance](docs/performance.md) for the full methodology, variability, reproduction command,
and per-fixture results.

## Evaluation

Capability scores from the published model cards, measured through NInfer's OpenAI-compatible
serving route with thinking enabled, MTP=3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample
per problem):

| Model | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| [Qwen3.6-27B](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% |
| [Qwen3.6-35B-A3B](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% |

These are single-sample results under that NInfer evaluation profile, not pass@k. See each model
card for correct/total counts and the full evaluation notes.

## Requirements

NInfer currently requires:

- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- NVIDIA driver support for CUDA 13.1 and the CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config`;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja, when using the commands below.

The build rejects CUDA architectures other than `120a`. There is no install target or packaged
binary distribution; NInfer is run from its source build tree.

## Build

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The default configuration builds:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

## Download a model

Use the Hugging Face CLI to download either registered artifact:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

The `.ninfer` file contains the weights and frontend resources needed by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

The artifact is complete, while GPU residency is fixed at process startup. Speculative decoding is
disabled by default, so MTP/DFlash state and the optimized proposal head are not uploaded.
Vision is also disabled by default, so its weights and workspace are omitted. Add `--vision` to the
CLI or server process that must accept image or video input. Disabled capabilities cannot be
enabled by a later request. DFlash is available only for the 35B-A3B target and is text-only.

## Run the CLI

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --model-id qwen3.6-27b \
  --max-context 16384 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements Anthropic Messages, streaming, token counting, multimodal input, and
function-tool request/response translation. See [HTTP serving](docs/serving.md).

## Capabilities

Both registered artifacts support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- greedy, temperature, top-k, top-p, min-p, and presence/frequency-penalty sampling;
- compatible-prefix reuse;
- OpenAI Chat Completions and Anthropic Messages, including streaming and usage accounting;
- prompt-rendered function tools and parsed tool calls.

## Current limits

- Only the two artifacts listed above are accepted product targets.
- Execution is specialized for one RTX 5090 and one CUDA device.
- One Engine owns one resident sequence and runs one active request at a time.
- Continuous batching, multi-GPU execution, CPU/GPU offload, and distributed serving are not
  implemented.
- Context capacity is configurable up to the registered models' native 262,144-token limit, subject
  to GPU memory and KV-cache configuration.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [CLI examples](examples/cli/)

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), which are also distributed
under Apache-2.0. Vendored dependencies retain their own license files under `third_party/`.
