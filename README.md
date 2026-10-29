# NInfer

> Native C++/CUDA inference for selected models on RTX 5090 and DGX Spark.

NInfer runs text and image/video generation through a local CLI or OpenAI- and
Anthropic-compatible HTTP APIs. It supports single-device execution and model-parallel execution
across two DGX Sparks, with CUDA Graph decode, speculative decoding, and shared-prefix caching.
One Engine serves one model with up to eight active requests.

## Supported models and deployments

| Model | Architecture | Deployment | Speculative backends |
|---|---|---|---|
| Qwen3.6-27B / Qwen3.8-27B | Qwen3.5 Dense | RTX 5090 | MTP; DFlash2 for Qwen3.8-27B with companion weights |
| Qwen3.6-35B-A3B | Qwen3.5 MoE | RTX 5090, resident or CPU expert offload | MTP or DFlash with companion weights |
| Qwen3.8-Flash-Next | Qwen4Exp | RTX 5090 with CPU expert offload; one or two Sparks | MTP |
| GLM-5.3-Flash | GLM5Next | Two Sparks | MTP |

All rows support text, Vision, tools, constrained output, and offline perplexity scoring.
Vision and speculative decoding require the corresponding components in the artifact and must be
enabled at startup. Scoring loads Text only. CPU expert offload applies to the RTX 5090 target;
Spark execution keeps selected weights on the GPU. Flash Next offload also needs substantial system
RAM for expert weights; see the [CLI guide](docs/cli.md#cpu-expert-offload).

Dual-Spark Flash Next uses TP2, optionally with EP2. GLM uses TP2 plus attention DP2, optionally
with EP2. The [deployment guide](docs/startup-configuration.md#distributed-deployment) explains
these settings and worker startup. [Example configurations](examples/configs/) cover the common
deployments, including Vision, FP8 KV, MTP3 and the optimized draft head.

## Build

Build natively on Linux: x86-64 / `sm_120a` for RTX 5090, AArch64 / `sm_121a` for DGX Spark.
Dependencies are a CUDA toolkit supporting the target, CMake 3.28+, a C++20 host compiler,
Ninja, `pkg-config`, libjpeg development headers, FFmpeg development libraries
(`libavformat`, `libavcodec`, `libavutil`, `libswscale`), `hwloc >= 2.5`, `liburing >= 2.5`,
and `libcurl >= 7.85`. CUDA 13.4 is the development toolkit; CMake checks the requested architecture.

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer
cmake --preset rtx5090-release
cmake --build build/rtx5090-release -j
```

For a single Spark, use `dgx-spark-release` and `build/dgx-spark-release`.
For two Sparks, install NCCL development headers and libraries and build on **both nodes**:

```bash
cmake --preset dgx-spark-release -DNINFER_ENABLE_DISTRIBUTED=ON
cmake --build build/dgx-spark-release -j
```

Product binaries are `apps/ninfer`, `apps/ninfer-serve`, `apps/ninfer-perplexity`, and, with
distributed execution enabled, `apps/ninfer-worker`. Run them from the build tree; keep its `lib/`
directory alongside `apps/` when moving outputs. There is no installed C++ SDK.

Product presets omit tests and benchmarks. The `rtx5090-dev` / `dgx-spark-dev` presets enable
both; the corresponding `-ops` presets build only the lower-level libraries and their checks.
See [build organization](docs/maintainer/build-system.md) for development options.

## Get weights

The Engine reads v3 `.ninfer` artifacts containing weights, configuration, tokenizer, chat template,
and component resources. These official downloads are available:

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |
| Qwen3.8-Flash-Next | Mixed NVFP4/Q8/NF4 | `qwen3_8_flash_next.ninfer` + 3 parts | [Qwen3.8-Flash-Next](https://huggingface.co/neroued/Qwen3.8-Flash-Next-NInfer) |
| GLM-5.3-Flash | Mixed NVFP4/Q8 | `glm5_3_flash.ninfer` + 5 parts | [GLM-5.3-Flash](https://huggingface.co/neroued/GLM-5.3-Flash-NInfer) |

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir models
```

Download Flash Next or GLM with all their storage parts:

```bash
hf download neroued/Qwen3.8-Flash-Next-NInfer \
  --include 'qwen3_8_flash_next.ninfer*' \
  --local-dir models/Qwen3.8-Flash-Next-NInfer

hf download neroued/GLM-5.3-Flash-NInfer \
  --include 'glm5_3_flash.ninfer*' \
  --local-dir models/GLM-5.3-Flash-NInfer
```

Flash Next uses 101.00 GiB across four files; GLM uses 178.33 GiB across six. Keep every part beside
the entry file and pass only the entry path to NInfer. For two Sparks, place the complete set on each
node; rank-specific slicing happens during loading. The model cards describe their mixed precision,
components and deployment options.

For custom weights, follow [weight conversion](docs/weight-conversion.md). Older v2 files can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact).

## Run

Run one request on RTX 5090:

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode briefly." \
  --max-context 8192 --max-tokens 1024 --kv-dtype fp8
```

Answer content goes to stdout; reasoning, startup information and the result report go to stderr.
Use `--messages FILE` and `--vision` for image/video input. See the [CLI guide](docs/cli.md).

For a persistent service, edit a deployment example and launch it:

```bash
cp examples/configs/rtx5090-qwen3_8-27b-nvfp4.toml ninfer.local.toml
# Edit model.path for your artifact.
./build/rtx5090-release/apps/ninfer-serve --config ninfer.local.toml
```

The example enables Vision, FP8 KV, MTP3 and the optimized draft head. It sets a per-request
context limit of 240,000 tokens; the actual shared KV/State capacity depends on available memory.
The startup log reports the Context pool size and an estimated token capacity.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

The server defaults to loopback. For LAN access, set `server.host = "0.0.0.0"` and configure
`server.api_key` as needed. The [Serve guide](docs/serving.md) covers endpoints, tools,
structured output, streaming, metrics, and request logs.

Both programs accept `--config FILE`; explicit command-line values override it. The root
[ninfer.example.toml](ninfer.example.toml) lists every field, while
[deployment examples](examples/configs/) provide focused starting points.
[Startup configuration](docs/startup-configuration.md) explains defaults, resource controls,
and distributed launch. Personal TOML files are ignored by Git.

## Context and prefix reuse

`max_context` limits a request's input plus generated tokens. `context_capacity` controls a separate,
startup-sized byte pool on each rank, shared by growing KV histories and continuation state.
With `auto`, the pool uses the available budget after weights and execution resources are planned.
It is not multiplied by concurrency, and a logical context limit does not guarantee that every
active request can simultaneously reach that length.

Serve retains exact text and multimodal prefixes, including the state needed to continue them.
Memory pressure can reclaim cached prefixes or pause active requests for later restoration/replay.
RTX 5090 can also retain context in pinned Host memory; Spark uses Device context storage.
See [resource controls](docs/startup-configuration.md#context-and-memory) for tuning.

## Performance

These recorded campaigns used an RTX 5090 and the engine version identified by each result page. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results. Those campaigns used FP8 E4M3 row-256 KV for Qwen3.8
and INT8 group-64 KV for Qwen3.6.
The measurements retain their original cache formats and resource settings; they are not measurements
of the current Spark or distributed paths.

### Concurrent MTP3 decode

Saturated decode used CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept |
|---|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `groupwise-int` | 136.5 / 44.4% | 253.3 / 45.2% | 398.1 / 46.1% | 582.4 / 46.4% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 147.7 / 46.2% | 291.0 / 48.7% | 522.2 / 45.8% | 922.4 / 46.1% |

### Single-request serving

The serial serving corpus used CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,331.9 tok/s | 2,139.4 tok/s | 214.7 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 12,819.1 tok/s | 4,016.4 tok/s | 231.7 tok/s |

## Evaluation

The 27B/35B capability scores below were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

[Flash Next](model-cards/Qwen3.8-Flash-Next-NInfer/README.md#evaluation) was evaluated separately on
two Sparks with TP2/EP2, BF16 KV, Vision, MTP3, optimized draft head and EvalScope 1.10.0:

| Benchmark | Accuracy | Correct / scored samples | Repeats per question |
|---|---:|---:|---:|
| GPQA-Diamond | 91.79% | 727 / 792 | 4 |
| AIME 2026 | 97.78% | 88 / 90 | 3 |
| ERQA | 68.00% | 272 / 400 | 1 |
| RealWorldQA | 85.75% | 656 / 765 | 1 |
| MMMU-Pro, standard 10 options | 80.81% | 1,398 / 1,730 | 1 |

These are mean sample accuracies, not pass@k. The card records the generation settings and includes
the truncated/unparsed samples in the totals. Vision scores retain the original campaign's decoder
and scoring behavior.

## Docker

The included Dockerfile builds the RTX 5090 CLI and server. It requires the NVIDIA Container Toolkit
on the host:

```bash
docker build --tag ninfer:local .
docker run --rm --gpus '"device=0"' --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --max-context 8192 --kv-dtype fp8
```

## Documentation

- [Deployment configurations](examples/configs/) and [startup parameters](docs/startup-configuration.md)
- [CLI](docs/cli.md), [HTTP serving](docs/serving.md), and [perplexity](docs/perplexity.md)
- [Weight conversion](docs/weight-conversion.md)
- [Recorded performance](docs/performance.md), [benchmarks](bench/README.md), and [evaluation](eval/README.md)
- [Documentation index](docs/README.md) and [contributing](CONTRIBUTING.md)

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The 27B and 35B-A3B artifact sources use Apache-2.0, as recorded in their model cards.
Flash Next combines the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
source with [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
weights. GLM's source weights use [MIT](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE).
Model licenses are separate from NInfer's software license. Vendored dependencies retain their own
license files under `third_party/`.
