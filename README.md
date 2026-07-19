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
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `qwen3_6_35b_a3b.ninfer` | 22,373,184,256 bytes (20.84 GiB) | `9e8378398d2b789a77224b5110c7590adbbc6fd4accd139b918157b2b9da7163` |

## Performance

The primary MTP path was measured on an RTX 5090 using the same long-context prompt source (7,678
tokens after NInfer prompt preparation), 512 generated tokens, BF16 KV cache, greedy decoding, and
a draft window of three. Each value is the mean of five measured runs after one warm-up.

| Model | Engine | Prefill | Decode |
|---|---|---:|---:|
| Qwen3.6-27B | **NInfer** | **3,260.67 tok/s** | **188.30 tok/s** |
| Qwen3.6-27B | llama.cpp | 2,646.64 tok/s | 140.38 tok/s |
| Qwen3.6-35B-A3B | **NInfer** | **15,496.77 tok/s** | **593.60 tok/s** |
| Qwen3.6-35B-A3B | llama.cpp | 5,476.82 tok/s | 285.88 tok/s |

For these configurations, NInfer reaches 1.23x/1.34x the llama.cpp prefill/decode rates on 27B and
2.83x/2.08x on 35B-A3B. Within NInfer, enabling the measured MTP path raises decode throughput by
2.45x on 27B and 2.16x on 35B-A3B.

These are complete-engine comparisons using each engine's corresponding quantized artifact, not
kernel measurements or identical weight formats. See [Performance](docs/performance.md) for the
full MTP-on/off results, variability, exact commands, and comparison boundaries.

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

## Run the CLI

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
MTP statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --model-id qwen3.6-27b \
  --max-context 16384 \
  --mtp-draft-tokens 3 \
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
