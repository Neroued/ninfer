# NInfer for NVIDIA V100

> Blackwell gets NVFP4 in silicon. Volta gets it from software.

This is the NVIDIA V100 (`sm_70`) port of
[NInfer](https://github.com/Neroued/ninfer), a from-scratch C++/CUDA inference engine for a closed
set of Qwen checkpoints. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The product model is deliberately narrow: one registered
artifact, one NVIDIA GPU, and kernels tuned for the exact shapes the model uses.

Upstream targets the GeForce RTX 5090 (`sm_120a`). This fork keeps that path intact and adds a
separate compile-time Volta implementation. It is not a compatibility shim around an existing
framework.

NInfer deliberately supports a closed set of model artifacts instead of acting as a general model
runtime:

| Model | Weights | NInfer artifact | Size | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 bytes (17.07 GiB) | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 18,210,531,328 bytes (16.96 GiB) | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 bytes (20.02 GiB) | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 bytes (21.22 GiB) | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |

Qwen3.6-27B and Qwen3.8-27B each expose two registered weight profiles. The version-2 artifact
identity selects the profile without a separate runtime flag; Qwen3.8 uses target key
`qwen3_8_27b` while sharing the 27B execution package. The Qwen3.6 `nvfp4` profile uses W4A4 Tensor
Core MMA for prefill and A16 NVFP4 kernels for decode. The Qwen3.8 `nvfp4` profile preserves its
source's mixed allocation: NVFP4 MLP weights in Text layers 0–55 and row-scaled FP8 for the token
embedding, attention input/output projections, GDN Q/K/V/Z and output projections, output head, and
remaining MLP weights. All four 27B artifacts retain the same Text, Vision, MTP, prefix-reuse, CLI,
and serving routes.

## The V100 port

The port replaces architecture-specific Blackwell mechanisms while leaving the artifact format,
model packages, scheduler, frontend, CLI, and HTTP APIs intact. Selection happens at build time:
`sm_120a` keeps the upstream kernels and `sm_70` compiles the Volta routes.

| Area | V100 implementation |
|---|---|
| Groupwise linear layers | Shape-specific Q4, Q5, Q6, and W8 decode kernels; row-split SIMT and Volta `m8n8k4` MMA schedules for narrow batches; CUTLASS SM70 tensor-core GEMMs for wide prefill |
| NVFP4 and FP8 | Register-level software decode for narrow work; load-time NVFP4 prepack; transient FP16 reconstruction and CUTLASS GEMM for wide work |
| Attention | Paged BF16 and INT8 group-64 KV decode kernels, plus a vendored and pinned llama.cpp Volta flash-attention kernel for prefill |
| Gated Delta Net | Volta projection, convolution, recurrent-state, record/replay, and gating routes for the hybrid Text layers |
| Speculative decoding | CUDA Graph decode, MTP windows from one to five, optimized proposal head, sampling, acceptance, and commit on the existing product path |
| Serving | The upstream CLI, OpenAI Chat/Responses, Anthropic Messages, streaming, tools, multimodal input, prefix reuse, and bounded concurrent batching |

Groupwise support is the base of the port, not a fallback. Its Q4/Q5/W8 kernels were tuned by
registered model shape and token width, including separate GEMV, row-split, tensor-core, and
CUTLASS crossovers. The final decode profile remains dominated by those specialized linear
kernels; attention and general launch overhead are a small part of an MTP3 round. At full context,
the groupwise artifact reaches 352.55 prefill tok/s with INT8 KV.

The port also covers the less visible product work needed to run the model correctly on Volta:
BF16 instruction substitutions, workspace and graph sizing, paged KV behavior, long-context
attention, vision attention, GDN state transitions, artifact binding, and architecture-specific
route registration. Unsupported Blackwell-only kernels remain isolated from the V100 build rather
than being selected and failing at runtime.

## Software NVFP4 on Volta

V100 has no FP4 instructions. The NVFP4 path in this repository is a software implementation of
the published NVFP4 artifact, not a conversion to groupwise integers and not a persistent FP16
copy of the model.

At model load, NVFP4 MLP gate/up weights are permuted in place into the fragment order required by
Volta's `mma.sync.m8n8k4` instruction. The temporary prepack buffer is released immediately, so
weight residency remains the artifact's packed 4-bit codes plus E4M3 K16 scales. During decode and
speculative verification, QPN2 streams those packed bytes from HBM, decodes E2M1 values and E4M3
scales in registers, and feeds FP16 tensor-core operands without materializing an FP16 weight
matrix. The prepack removes the nibble permutation from the inner loop; this is the change that
takes the five-draft verification round below the 70 tok/s target budget.

Wide prefill uses a different route. It reconstructs one packed matrix into transient FP16
workspace, runs a stock CUTLASS SM70 tensor-core GEMM, then reuses that workspace for the next
operation. The reconstruction uses the same shift decoder as QPN2. Nothing expands persistently,
which is why the 20.02 GiB Qwen3.8 NVFP4 artifact can still prefill at the 262,144-token model
limit on a 32GB card with INT8 KV.

## Performance

Measurements below use a Tesla V100-SXM2-32GB at a locked 1,530 MHz graphics clock, CUDA 12.8,
one request, and Qwen3.8-27B. Decode figures are from the target-round benchmark with the optimized
proposal head. Round latency is independent of speculative acceptance; committed tok/s is not.

### Decode

The short target-round benchmark also gives the measured groupwise comparison. These rates use
each profile's best draft width and the licensed-token mean observed in its ten-round sample;
acceptance varies with the continuation.

| Weight profile | Drafts | Mean round | Mean licensed | Derived rate |
|---|---:|---:|---:|---:|
| `groupwise-int` | 3 | **53.79 ms** | 3.2 | 59.5 tok/s |
| software `nvfp4` | 5 | 60.80 ms | **5.0** | **82.2 tok/s** |

Groupwise MTP5 was slower: 70.21 ms per round, 3.9 licensed tokens, and 55.5 tok/s in the same
short sample. Its measured optimum remains MTP3. NVFP4's wider useful verification window is what
lets it produce the higher committed rate despite the longer round.

#### Software NVFP4 kernel delta

| Five-draft target verification | Mean round | Rate at 4.5 licensed tokens/round |
|---|---:|---:|
| Checkpoint-native QPN2 | 68.95 ms | 65.3 tok/s |
| Load-time-prepacked QPN2 | **60.80 ms** | **74.0 tok/s** |

The prepack reduces the complete MTP5 round by **11.8%** and clears the theorized 70 tok/s point
when the workload licenses 4.5 tokens per round. A short ten-round sample licensed 5.0 tokens per
round and therefore computes to 82.2 tok/s; a 100-round synthetic continuation licensed only 3.87
and would compute to 63.7 tok/s at the final round latency. Those different rates are acceptance
behavior, not different kernel speed. For a workload with mean licensed width `L`, use
`L / 0.06080` to estimate the short-context kernel-bound rate.

MTP6 and MTP7 were also tested. Acceptance did not increase: both averaged 4.8 licensed tokens in
the test continuation while round time rose to 78.50 and 80.61 ms. The product limit remains five.

### Prefill

Both profiles use a 2,048-token chunk. The 260,096-token case allocates the full 262,144-token
context with INT8 group-64 KV and no warm-up; both routes peak at 1.75 GiB of workspace.

| Weight profile | 2,048 tokens | 260,096 tokens |
|---|---:|---:|
| `groupwise-int` | **1,235.1 tok/s** | **352.55 tok/s** |
| software `nvfp4` | 1,214.3 tok/s | 351.72 tok/s |

The long-context comparison is the useful one: attention and KV work dominate there, while the
fixed per-chunk weight reconstruction cost is most visible in the 2K case. At 260,096 tokens the
software NVFP4 path is within 0.24% of groupwise prefill throughput.

### Difference from the RTX 5090 build

The upstream README reports 143.8 tok/s for Qwen3.8-27B NVFP4 at C=1/MTP3 and 2,203.1 prefill
tok/s at a 260,096-token prompt. The V100 numbers above use MTP5 for the target-round measurement
and a 2,048-token prefill chunk, so a raw ratio mixes protocol, acceptance, memory bandwidth, and
native FP4 hardware. The architectural delta is simpler: the 5090 consumes NVFP4 in silicon;
V100 runs a load-time fragment prepack, software E2M1/E4M3 decode, and FP16 `m8n8k4` tensor-core
MMA. The artifact and public serving surface stay the same.

## Evaluation

These are upstream artifact scores and were not rerun for this V100 release. Capability scores
were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% |

Both Qwen3.8-27B profiles are supported but have not yet been added to this published evaluation
campaign.

These are single-sample results under that NInfer evaluation profile, not pass@k. See the model
cards and [full performance document](docs/performance.md) for correct/total counts and evaluation
notes.

## Requirements

The V100 build requires:

- 64-bit Linux;
- NVIDIA Tesla V100 (`sm_70`), with 32GB recommended for the published 27B artifacts;
- CUDA Toolkit 12.8 and a compatible NVIDIA driver;
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config`;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja, when using the commands below.

CUDA 12.8 is the final toolkit release that can compile `sm_70`; CUDA 13 removes offline Volta
compilation. The default upstream/Blackwell build still requires CUDA 13.1 or newer. There is no
install target or packaged binary distribution; NInfer is run from its source build tree.

## Build

```bash
git clone https://github.com/geoffwatts/ninfer-v100.git
cd ninfer-v100

cmake -S . -B build-v100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 --parallel
```

The default configuration builds:

```text
build-v100/apps/ninfer
build-v100/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

## Docker

The inherited Dockerfile follows upstream's CUDA 13.1/RTX 5090 build and is not the V100 release
path. Build natively with CUDA 12.8 as shown above. A Volta runtime image must use a CUDA 12.8
development base and pass `-DCMAKE_CUDA_ARCHITECTURES=70` during configuration.

## Download a model

Use the Hugging Face CLI to download one of the registered artifacts:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# Or the 27B NVFP4 weight variant:
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

# Or Qwen3.8-27B:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

# Or Qwen3.8-27B NVFP4:
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

Current NInfer builds accept only the version-2 artifact container, and all five downloads above
are version 2. Migration applies only to Qwen3.6 artifacts downloaded before their version-2
publication; both Qwen3.8-27B profiles were published directly as version 2. Migrate an older exact
local file in place:

```bash
python3 -m tools.artifact.migrate_v1_to_v2 models/qwen3_6_27b.ninfer
```

Use the same command with `qwen3_6_27b_nvfp4.ninfer` or `qwen3_6_35b_a3b.ninfer` for those
artifacts. The migration updates only container metadata; it does not rewrite the weight payload.
Alternatively, download the current version-2 file again from its Hugging Face repository.

Each `.ninfer` file contains the weights and frontend resources needed by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

Each artifact is complete, while GPU residency is fixed at process startup. Speculative decoding is
disabled by default, so MTP/DFlash state and the optimized proposal head are not uploaded.
Vision is also disabled by default, so its weights, Vision scratch phase, and frozen
request-transient allocation are omitted. Add `--vision` to the CLI or server process that must
accept image or video input. Disabled capabilities cannot be enabled by a later request. DFlash is
available only for the 35B-A3B target and is text-only.

## Run the CLI

```bash
./build-v100/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 5 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build-v100/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
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
./build-v100/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 5 \
  --lm-head-draft
```

For one full-context request on a 32GB V100, use INT8 group-64 KV and the measured 2,048-token
prefill chunk:

```bash
./build-v100/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 262144 \
  --kv-capacity auto \
  --kv-dtype int8 \
  --prefill-chunk 2048 \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 5 \
  --lm-head-draft
```

`--kv-capacity auto` sizes one shared KV pool from the memory left after weight upload. Raising
`--max-concurrency` does not create another 262,144-token pool per request; active and retained
requests divide the same token capacity.

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages, token counting, and multimodal input. See
[HTTP serving](docs/serving.md).

## Capabilities

All three registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- startup-bounded small-scale concurrent serving with true batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- model- and thinking-mode-aware official sampling defaults, with explicit greedy, temperature,
  top-k, top-p, min-p, and presence/frequency-penalty overrides;
- compatible-prefix reuse;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming and
  usage accounting;
- prompt-rendered function tools and parsed tool calls.

The 35B-A3B target additionally supports text-only DFlash speculative decoding with draft windows
from one to fifteen.

## Current limits

- Only the five `(model_id, weights_id)` artifact identities listed above are accepted product
  identities.
- The Volta build is specialized for one Tesla V100 and one CUDA device.
- One Engine owns one resident model and supports a startup-fixed capacity of 1–8 active requests.
  Decode-ready requests are compacted at round boundaries and executed in one batched model
  traversal.
- NInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- `--max-context` is the logical ceiling of each sequence and is configurable up to the registered
  models' native 262,144-token limit. `--kv-capacity N` explicitly sizes the shared Main Text KV
  pool for all active and retained sequences, while `--kv-capacity auto` selects the largest usable
  capacity from the memory remaining after weights are loaded while preserving 1 GiB of sizing
  headroom. Omission defaults to one `--max-context` worth of pages. The resolved pool is fixed at
  startup and is not divided statically among request lanes.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [CLI examples](examples/cli/)

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
