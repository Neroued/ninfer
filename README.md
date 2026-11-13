
<h1 align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/capy-flash-lockup-dark.svg">
  <img alt="capy-flash" src="docs/logo/capy-flash-lockup.svg" width="560">
</picture></h1>

**Qwen3.8-Flash-Next (125B MoE) and Qwen3.8-27B on one or two 16 GB RTX 50-series GPUs, built for agents.**

capy-flash is an inference server. It runs two models on consumer GPUs with 16 GB of VRAM and exposes an OpenAI- and
Anthropic-compatible HTTP API, so agent tools such as Claude Code, Codex CLI and OpenCode can use it like a hosted model.

- **Qwen3.8-Flash-Next** is a mixture-of-experts model too large for the GPUs. capy-flash keeps the experts that do
  not fit in VRAM in host RAM and computes them on the CPU; everything else runs on the GPU (split over two GPUs when
  there are two). That is why Flash-Next needs a lot of RAM and a CPU with AVX2 or AVX-512.
- **Qwen3.8-27B** is a dense model that runs entirely on the GPUs.
- For agents: a prefix cache across turns, idle conversations moved to RAM (and optionally to disk) instead of being
  recomputed, and the expert cache follows the conversation. For long contexts Flash-Next can also keep the KV cache
  of the conversations being generated in RAM (`--kv-ram-active`), with one request or several at once.

capy-flash **is ninfer-tp2, renamed at v0.5.0**: the two-GPU tensor-parallel fork of
[Neroued/ninfer](https://github.com/Neroued/ninfer) (v0.1-v0.4.9 under the old name); [NOTICE](NOTICE) lists every piece
of prior work it builds on. Status: **v0.11.1**. Built, tested and measured on one machine ([below](#benchmarks));
users report other setups. Not affiliated with Qwen or NVIDIA.

## Pick a model

**16 GB of VRAM per GPU is the minimum.** RAM matters for Flash-Next only.

| what you want to run | GPUs (RTX 50 series, sm_120) | RAM | `install.sh` |
|---|---|---|---|
| Qwen3.8-27B, 3-bit | **one** with 16 GB of VRAM (the minimum) | nothing special | `--model 27b-16gb` |
| Qwen3.8-27B, NVFP4 | two with 16 GB | nothing special (8 GiB of context cache) | `--model 27b` |
| Flash-Next, 64 GB edition | **one** or two with 16 GB | a **64 GB** machine (~56 GiB free) | `--model flash-next-64gb` |
| Flash-Next, NVIDIA NVFP4 (default) | two with 16 GB | a **96 GB** machine (~75 GiB free) | (default) |

- **One GPU, little RAM:** the 3-bit 27B. It costs quality: GSM8K on par with the NVFP4 27B, MMLU-Pro about 0.72
  against 0.78 ([benchmarks](#benchmarks)).
- **One GPU and 64 GB of RAM:** the 64 GB edition of Flash-Next (ISTA-DASLab's GSQ-RCO experts, 48 GiB in RAM instead
  of 65; quality on par with NVFP4 on GSM8K and MMLU-Pro).
- **Two GPUs:** the 64 GB edition is the fastest Flash-Next here; the NVFP4 edition uses NVIDIA's experts as published.
  For the 27B on two cards, the NVFP4 27B is the better choice than the 3-bit one.

## What you need

- Linux x86_64 (tested on Ubuntu). Users run the 27B on WSL2 and on Windows with Docker; a native Windows build exists
  as a community port ([OmarxGx/capy-flash](https://github.com/OmarxGx/capy-flash)), not in the releases yet.
- An NVIDIA driver with CUDA 13.1 support (R580 or newer). The release needs no CUDA toolkit; building does (CUDA 13.1).
- `sm_120` GPUs with 16 GB: RTX 5060 Ti 16 GB, 5070 Ti, 5080 and 5090 are reported to work; a mixed pair runs at the
  pace of the slower card. Peer-to-peer is not needed.
- **Flash-Next only:** a CPU with AVX-512 (F/BW/VL, VNNI) or AVX2 + FMA + F16C (Zen 2/3, Intel client cores), and the
  free RAM of the table: the routed experts are one page-locked host allocation (64.6 GiB for NVFP4, 48 GiB for the
  64 GB edition), and `--profile flash-next` also keeps the token embedding and up to 4 GiB of idle conversations in
  RAM. The server checks this at startup and refuses with the numbers when it does not fit (`NINFER_RAM_GUARD=0`
  starts anyway). On Windows/WSL2, which limit page-locked memory to about half the RAM, the layers that do not fit
  stay in pageable RAM (same tokens, slower).
- Disk: about 135 GB (Flash-Next NVFP4), 110 GB (64 GB edition), 20 GB (27B NVFP4), 15 GB (27B 3-bit).

## Quick start

Default model (Flash-Next NVFP4, two 16 GB cards); for another row of the table add its `--model`:
```bash
curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash -s -- --check   # can this machine run it?
curl -fsSL https://raw.githubusercontent.com/ValerioDolci/capy-flash/main/install.sh | bash               # release + model into ~/capy-flash
~/capy-flash/current/serve-flash-next.sh ~/capy-flash/models/flash-next/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"capy-flash","messages":[{"role":"user","content":"What is 17*23?"}]}'                # from another shell
```
- The installer checks GPUs, driver, AVX2/AVX-512, RAM and disk and says what to fix, downloads the latest release and
  **one** model repository from Hugging Face, verifies both, and prints the command that serves it and a test request.
  `--dir` picks another place, `--no-model` downloads the release only, `--verify-model` also checks the model's digests.
- The first start takes a minute or two (a cold start of Flash-Next reads its experts from disk).
- **Launchers.** `serve-flash-next.sh` (Flash-Next; `NINFER_DEVICES=0` for one GPU), `serve-tp2.sh` (27B, two GPUs) and
  `serve-27b-16gb.sh` (27B 3-bit, one GPU) start `ninfer-serve` with the tested profile, the model id (`capy-flash` or
  `qwen3.8-27b`, the `model` name requests must use) and the chat template. `NINFER_PORT=9000` changes the port;
  extra `ninfer-serve` flags go after the model path and win over the profile.
- **Vision.** The two 27B artifacts include the vision tower and their launchers load it (it takes VRAM from the
  context); `NINFER_VISION=0` before the launcher, or `--no-vision` after the model path, turns it off. The Flash-Next
  downloads are text only: images on Flash-Next need the separate
  [Vision artifact](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash), which `install.sh` does
  not fetch, and `--vision`.
- **Keep it running.** `ninfer-serve` exits with status 2 when the Engine fails, so run it under a supervisor that
  restarts it: a systemd unit and a llama-swap entry are in [examples/serving](examples/serving).
- By hand: download a [release](https://github.com/ValerioDolci/capy-flash/releases) tarball
  ([contents](tools/release/README-release.md)) and `hf download Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash --local-dir <dir>`.

## Connect a client

The server listens on `http://127.0.0.1:8080` (`NINFER_HOST` / `NINFER_PORT` change it): OpenAI Chat Completions and
Responses under `/v1`, Anthropic Messages at `/v1/messages`. `--api-key KEY` makes it require
`Authorization: Bearer KEY` or `x-api-key: KEY`; `--api-key-file FILE` or the `NINFER_API_KEY` environment variable
set the key without putting it on the command line.

- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` (any model name is accepted on Messages).
- **Codex CLI:** a custom provider with base URL `http://127.0.0.1:8080/v1`, the `responses` wire API and the server's
  model id. It needs the chat template the launchers pass (`qwen3_8.jinja`), because Codex always sends `instructions`.
- **OpenCode and other OpenAI clients:** base URL `http://127.0.0.1:8080/v1`, model `capy-flash` (or `qwen3.8-27b`).

Details, including authentication and what each client sends: [docs/serving.md](docs/serving.md#connect-a-client).

**Structured output.** `response_format` (`json_object`, `json_schema`) on Chat Completions, `text.format` on
Responses, `output_config.format` on Messages, and the `structured_outputs` extension (`grammar`, `regex`, `choice`)
constrain the answer: the reply is valid JSON for the schema (or matches the grammar, regex or choice) whenever the
model ends it. With thinking on, only the answer after `</think>` is constrained.

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "capy-flash", "messages": [{"role": "user", "content": "A cat, as JSON."}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "cat", "schema": {"type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}}}'
```

Schemas and grammars come from the client, so they are bounded: size, nesting and item limits before compilation, a
compilation timeout, and a time and memory budget while generating (`--constraint-*` flags). A request over a limit
fails alone with a 400 that names the flag; the other requests keep their speed. Fields, errors, limits and cost:
[docs/serving.md](docs/serving.md#structured-output).

## Run it other ways

### Build
```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=OFF -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
ninja -C build apps/ninfer-serve
build/apps/ninfer-serve <dir>/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer --profile flash-next --model-id capy-flash \
  --chat-template tools/chat_templates/qwen3_8.jinja
```
Or take the prebuilt tarball of a [release](https://github.com/ValerioDolci/capy-flash/releases) (Ubuntu, `sm_120a`),
which has the three launchers ([tools/release](tools/release/README-release.md)).

### Profiles
`--profile NAME` applies the tested configuration of a model; flags given on the command line override it, in any
order. `ninfer-serve --help` and [docs/serving.md](docs/serving.md#profiles) list the full expansions.

| profile | for | in short |
|---|---|---|
| `flash-next` | Flash-Next on two GPUs (the launcher's default) | `--tp 2`, INT8 KV, elastic KV / expert slots, 262,144 tokens, MTP 3, 4 GiB of idle KV and 8 conversation states in RAM, embedding in RAM |
| `flash-next-1gpu` | the 64 GB edition on one GPU (`NINFER_DEVICES=0`) | the same on one device, 1 GiB of idle KV and 8 states in RAM |
| `27b` | Qwen3.8-27B NVFP4 on two GPUs | `--tp 2`, INT8 KV, 196,608 tokens, MTP 3, Vision, 8 GiB of context cache in RAM |
| `27b-16gb` | Qwen3.8-27B 3-bit on one GPU | INT8 KV, 65,536 tokens, MTP 3, Vision, embedding in RAM (13.9 GB of VRAM) |

For the 64 GB edition on two GPUs add `--host-kv-mib 1024 --host-state-slots 2` to `--profile flash-next` (smaller RAM
caches); `install.sh --model flash-next-64gb` prints that command.

For long contexts on Flash-Next, add `--kv-ram-active` to either Flash-Next profile: the KV cache stays in pinned RAM,
taken as the conversations grow (about 2.5 GiB per GPU for 262k tokens, 4.2 GiB on one GPU), and the VRAM it frees
goes to expert slots. Same tokens; within a few per cent of the default speed
([costs](docs/maintainer/active-kv-in-ram.md#cost)). For several long conversations at once add
`--max-concurrency N --kv-capacity T` (T tokens for all of them together; the RAM follows what they use). Prompts are
prefilled one at a time: a request waits at most `--pending-timeout-ms` (default 10 minutes) for the others.

### Docker
[docker/Dockerfile.release](docker/Dockerfile.release) packs a release tarball (builds in about a minute; the root
[Dockerfile](Dockerfile) compiles from source instead). It needs the NVIDIA Container Toolkit:
```
docker build -f docker/Dockerfile.release -t capy-flash .
docker run --gpus all -p 8080:8080 -v ~/capy-flash/models:/models capy-flash \
  /models/flash-next/qwen3_8_flash_next_nvfp4_idxgdnq8.ninfer --profile flash-next --host 0.0.0.0 \
  --model-id capy-flash --chat-template /opt/capy-flash/chat_templates/qwen3_8.jinja
```
Tested with rootless Podman and the two GPUs passed through (Flash-Next loads its 64.6 GiB of page-locked experts
without extra limits).

### Model weights
Ready to serve on Hugging Face (`install.sh --model` downloads one of these; each has its `SHA256SUMS`):
- [`Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash)
  (126.6 GB, the default; NVIDIA Open Model License together with the Qwen Community License of Qwen3.8-Flash-Next).
  Or convert `nvidia/Qwen3.8-Flash-Next-NVFP4` yourself in one command, same result:
  `python tools/capy_flash/convert.py --model <checkpoint dir> --out <dir>`. It runs `python -m tools.convert --recipe
  qwen3_8_flash_next_nvfp4_idxbf16_gdnq8 --components text,mtp` (experts NVFP4 as published, indexer BF16, GDN Q8,
  about 5 minutes on a GPU, 118 GB) and `tools/qwen4_draft_shortlist.py`. The previous text artifact, with NVIDIA's
  MTP head, stays available as revision `v0.5.4` (`hf download ... --revision v0.5.4`).
- [`Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash)
  (the same plus the Q8 vision tower; download by hand, serve with `--vision`; or `convert.py ... --vision`).
- [`Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash)
  (102 GB, the 64 GB edition: ISTA-DASLab's GSQ-RCO experts, the rest as above).
- [`Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash`](https://huggingface.co/Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash) (12.5 GB,
  ISTA-DASLab's 3-bit GSQ 27B plus the official MTP layer; recipe `qwen3_8_27b_gsq3`).
- [`Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer`](https://huggingface.co/Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer)
  (17.4 GB, the 27B for two GPUs; [model card](model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/README.md)).

How the engine runs these models, every server flag, the RAM tiers, the chat template and the CPU threads:
[docs/serving.md](docs/serving.md#capy-flash-serving). Environment variables and diagnostics:
[docs/capy-flash/diagnostics.md](docs/capy-flash/diagnostics.md). All documentation: [docs/README.md](docs/README.md).

### Optional speed switches (off by default)

Two environment variables make Flash-Next prefill faster at the price of a small change in the numbers. Both are
**off by default**. Set them in the server's environment, e.g. `NINFER_TP_AR_PAYLOAD=int8 ninfer-serve ...`. A third
one, lossless, is on by default since v0.11.1 on pairs without P2P (`NINFER_TP_XIN_KERNEL=0` turns it off).

| Variable | Applies to | What it changes | Prefill gain on Flash-Next (2x RTX 5070 Ti, fresh 30k-100k prompts) | Loss, our measurements |
|---|---|---|---|---|
| `NINFER_TP_XIN_KERNEL` (on by default without P2P since v0.11.1) | tp 2 without P2P | nothing in the numbers: the exchange between the GPUs is received by a kernel instead of queueing on the copy engine | 0 to +3 % on our setup; +6-7 % with `--kv-ram-active` | none, same tokens |
| `NINFER_TP_AR_PAYLOAD=int8` | Flash-Next (no effect on the 27B) | the prefill exchange between the GPUs carries int8 codes with an FP16 scale per 32 rows instead of FP16 | +7-8 % | perplexity −0.22 % (NVFP4), −0.26 % (GSQ-RCO): within noise |
| `NINFER_Q8_A8=128` (or `32`) | Flash-Next | the dense Q8 projections of the prefill run with int8 activations on the int8 tensor cores | +7-8 % | perplexity +0.21 % (NVFP4), +0.004 % (GSQ-RCO): within noise |

The two lossy switches change individual tokens, so they stay opt-in, but in our tests the loss is negligible. All
three together on Flash-Next NVFP4: prefill +28-29 % over the default; GSM8K (1,319 questions) 0.9788, against 0.9811
for vLLM on a GB300 (McNemar p 0.38); MMLU-Pro (504 questions) 0.8472, against 0.8492 for the default under the same
conditions (paired, p 1.0). Details in [docs/capy-flash/diagnostics.md](docs/capy-flash/diagnostics.md).

## Benchmarks

Our machine: AMD Ryzen 9 9900X (12 cores), 96 GB DDR5-5600, 2x RTX 5070 Ti 16 GB on PCIe 5.0 x8 + x8 (no P2P),
Ubuntu, CUDA 13.1, full clocks. Decode is the server-side rate. v0.8.0 runs at the speed of v0.7.7 (the
[release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.8.0) have the comparison). The agentic
replay is [bench/capy_flash](bench/capy_flash): 3 synthetic coding-agent sessions, 53 turns, prompts growing to ~81k tokens with
~90 % prefix reuse, 256 output tokens per turn, temperature 0. LocalMaxxing is the reasoning-v1 benchmark (short
prompt, 512 tokens).

**Ours** (one GPU = one RTX 5070 Ti):

| model, edition | GPUs | version, date | measured | result |
|---|---|---|---|---|
| Flash-Next NVFP4 | 2 | v0.6.0, 2026-10-18 | agentic replay: decode median / total time / time to first token, median (p90) | 155 tok/s / 177 s / 1.48 s (2.62 s) |
| Flash-Next NVFP4 | 2 | v0.6.0, 2026-10-18 | prefill, cold, ~3.5k / ~15k / ~55k-token prompts | 2,422 / 4,615 / 4,828 tok/s |
| Flash-Next NVFP4 | 2 | v0.6.0, 2026-10-18 | LocalMaxxing reasoning-v1 | 178 tok/s |
| Flash-Next NVFP4 | 2 | v0.7.0, 2026-10-22 | agentic replay decode median | 155 tok/s (as v0.6) |
| Flash-Next 64 GB | 2 | v0.7.0, 2026-10-22 | agentic replay decode median | 182 tok/s |
| Flash-Next 64 GB | 2 | v0.7.0, 2026-10-22 | LocalMaxxing reasoning-v1, [verified run](https://www.localmaxxing.com/runs/cmvjb7d5d03o3rf01ex9g9wiw) | 234 tok/s |
| Flash-Next 64 GB | 2 | v0.7.2, 2026-10-23 | prefill 4k / 16k / 64k | 2,168 / 3,924 / 3,994 tok/s |
| Flash-Next 64 GB | 1 | v0.7.0, 2026-10-22 | LocalMaxxing reasoning-v1, [verified run](https://www.localmaxxing.com/runs/cmvjb7dm603o7rf01i54inoup) | 121 tok/s |
| Flash-Next 64 GB | 1 | v0.7.1, 2026-10-22 | agentic replay: decode median / total time / time to first token, median / prompt tokens from cache | 104 tok/s / 276 s / 1.9 s / 83 % |
| Flash-Next 64 GB | 1 | v0.7.2, 2026-10-23 | prefill 4k / 16k / 64k | 1,671 / 2,838 / 2,921 tok/s |
| Qwen3.8-27B 3-bit | 1 | v0.7.0, 2026-10-22 | decode, 8k-token real-text prompt, MTP | 139 tok/s |
| Qwen3.8-27B 3-bit | 1 | v0.7.0 | prefill 4k / 16k | 3,401 / 3,211 tok/s |
| Qwen3.8-27B 3-bit | 1 | v0.7.0, 2026-10-22 | LocalMaxxing reasoning-v1, [verified run](https://www.localmaxxing.com/runs/cmvjb7ebb03ofrf01mkfqs948) | 174 tok/s |
| Qwen3.8-27B 3-bit | 2 | v0.7.0, 2026-10-22 | decode, 8k-token real-text prompt, MTP | 230 tok/s |
| Qwen3.8-27B 3-bit | 2 | v0.7.0 | prefill 4k / 16k / 64k | 5,787 / 5,567 / 4,304 tok/s |
| Qwen3.8-27B 3-bit | 2 | v0.7.0, 2026-10-22 | LocalMaxxing reasoning-v1, [verified run](https://www.localmaxxing.com/runs/cmvjb7emi03oirf01vyxuhulr) | 288 tok/s |
| Qwen3.8-27B NVFP4 (QUASAR) | 2 | v0.6.0 | prefill 4k / 16k / 64k, `--profile 27b` | 9,343 / 9,234 / 6,287 tok/s |

More on the 27B NVFP4 (`--profile 27b`, MTP; 280 prompts in three languages, v0.6): the same tokens as v0.5.4 on 280 of
280 answers (and with DFlash2), decode +5 % (Italian 168 → 176 tok/s), prefill at 4k / 16k / 64k 5,777 / 5,585 / 4,352
→ 9,343 / 9,234 / 6,287 tok/s with MTP and 5,719 / 5,530 / 4,316 → 9,205 / 9,052 / 6,198 with DFlash2; concurrent decode (NInfer's `decode-saturation`
suite, MTP3, 8,192 tokens per request) 175 / 326 / 560 / 927 tok/s aggregate at 1 / 2 / 4 / 8 requests. On an agentic
replay of the 27B (3 interleaved sessions and 11 subagents, 4 lanes), the v0.5.4 context cache took the median time to
first token from 2.14 to 0.98 s and the prompt tokens served from cache from 67 to 78 %, decode unchanged.
Flash-Next decode stays flat with context: 168 / 165 / 171 / 165 tok/s with 0 / 14k / 61k / 118k tokens of prompt
(v0.6), and a cold 382-token prompt answers in 1.35 s. A cold start of the Flash-Next server takes 26-29 s here
(v0.5.3), most of it reading the 64.6 GiB of experts. The 64 GB edition peaks at 55.2 GiB of RSS inside a 64 GiB
memory cgroup without swap (57 GiB used with one GPU); its speeds were measured on the 96 GB machine.

**Reported by users** (not reproduced by us):

| setup | model, edition | version | measured | result | source |
|---|---|---|---|---|---|
| AMD EPYC 7F52 16 cores (Zen 2, AVX2), 2x RTX 5080 (overclocked) | Flash-Next NVFP4 | v0.7.2 built with CUDA 13.4 | agentic replay (53 turns, greedy): decode median | 141.3 tok/s with automatic threads; 144.4 with `--expert-cpu-threads 15` and `taskset -c 0-15`; release tarball (CUDA 13.1) 137.6; v0.7.1 tarball 121.8 | [issue #11](https://github.com/ValerioDolci/capy-flash/issues/11) |
| same | same | same | own measurement, not greedy | prefill at 131k 5,541 tok/s; time to first token 22.1 s at 131k, 1.15 s at 1k | [issue #11](https://github.com/ValerioDolci/capy-flash/issues/11) |
| AMD Threadripper PRO 5975WX (AVX2), 2x RTX PRO 4000 Blackwell 24 GB (145 W), Proxmox LXC | Flash-Next 64 GB (GSQ-RCO) | v0.7.1 built from source, CUDA 13.3 | decode / prefill / local aider coding benchmark | decode ~100-114 tok/s on prose, ~140-155 on reasoning; prefill at 40k ~3,300 tok/s; aider 145-146 of 148 | [issue #22](https://github.com/ValerioDolci/capy-flash/issues/22) |

**Against Strata.** [Strata](https://github.com/Niko1221/Strata) 0.1.45 on our machine, same day (2026-10-18), two GPUs,
capy-flash v0.6.0:

| | **capy-flash**, NVFP4 | Strata, same NVFP4 weights | Strata, 3-bit IQ3_S (tuned) |
|---|---:|---:|---:|
| agentic replay, 53 turns up to 81k tokens: total time | **177 s** | 268 s | 197 s |
| agentic replay: decode, server side (median) | **155 tok/s** | 98 tok/s | 151 tok/s |
| agentic replay: time to first token, median (p90) | **1.48 s (2.62)** | 2.35 s (4.17) | 1.60 s (3.69) |
| prefill, ~3.5k-token prompt | **2,422 tok/s** | 1,519 tok/s | 2,216 tok/s |
| prefill, ~15k-token prompt | **4,615 tok/s** | 3,513 tok/s | 3,811 tok/s |
| prefill, ~55k-token prompt | 4,828 tok/s | 4,849 tok/s | **5,233 tok/s** |
| LocalMaxxing reasoning-v1 (short prompt, 512 tokens) | 178 tok/s | 119 tok/s | **195 tok/s** |

Strata's IQ3_S experts are 26 % smaller than NVIDIA's NVFP4 ones (lower quality: KL 0.144 vs 0.070 for NVFP4, as
Strata reports), which is why it leads on the short benchmark. v0.6 against the build before the v0.6 work
(2026-10-17): replay total 212 → 177 s, decode 118 → 155 tok/s.

**Comparing decode with other engines.** capy-flash computes every routed expert the model selects; an expert that
misses the VRAM cache is computed on the CPU, never skipped. Strata's "route tail skip" skips low-ranked experts that
miss the cache, which changes the output slightly and saves CPU work. A decode-speed gap against Strata, especially on
a slower CPU, most likely comes from that. Measured here with the same GSQ-RCO IQ3_S experts on both (one RTX 5070 Ti,
LocalMaxxing decode, 2026-10-23): Strata 138 tok/s with the skip, 116.6 with `STRATA_ROUTE_TAIL_SKIP=0`, capy-flash
129.8. On two GPUs (capy-flash v0.7.0 against Strata 0.2.2, 2026-10-22) the agentic replay decode median was 200.9
against 171.3 tok/s; Strata finished that replay sooner overall (189 against 200 s) thanks to its faster prefill.

**Quality** against the GB300 reference (NVIDIA's NVFP4 checkpoint served by vLLM on a GB300, same prompts, paired):

| model, version | GSM8K (reference) | MMLU-Pro (reference) |
|---|---:|---:|
| Flash-Next NVFP4, v0.6.0 | 0.9750 (0.9810), 1,319 questions; 9 / 1 discordant, McNemar p 0.02 | 0.8413 (0.8452), 1,008 questions: non-inferior (difference -0.40 pt, lower bound -1.85 pt, margin -2.5 pt) |
| Flash-Next NVFP4, v0.7.0 | 0.985 (0.985), 330 questions | 0.829 (0.849), 252 questions; paired with v0.6: no difference |
| Flash-Next 64 GB, v0.7.0 | 0.982 (0.985), 330 questions | 0.837 (0.849), 252 questions |
| Qwen3.8-27B 3-bit, v0.7.0 | 0.968 (0.968), 1,319 questions | 0.69-0.71, about 0.72 with the empty-reply guards (0.78), 252 questions |

The v0.6.0 release binary was checked on the first 330 / 252 questions (GSM8K 0.979, MMLU-Pro 0.841). Output is not
bit-reproducible run to run (which experts sit in VRAM depends on timing), so two runs of the same build differ by a
few questions. The 3-bit 27B loses MMLU-Pro points mostly because more answers run out of their 8,192
tokens while reasoning; the v0.7 empty-reply guards bring back the ones that stopped inside their reasoning (9 of 9,
all correct: about +3.6 points, estimated on those questions). Agent tasks (2026-10-14 build): 5 multi-step coding
tasks of a private task bank, 4 runs each, OpenCode with the model's sampling: 20/20 solved, 42.9k generated tokens per
run (Strata 0.1.43 IQ3_S on the same tasks: 52.7k, 20/20; the GB300 reference: 26.5k), no malformed tool call in 1,124.

## Known limits
- Flash-Next's profiles serve one request at a time (`--max-concurrency 1`) and only the machine above is tested. With
  `--max-concurrency 2` to `4` (with or without `--kv-ram-active` and `--elastic`) the decode of the requests is
  batched: two requests of 32k-100k tokens decode 1.2-1.4x the tokens per second of one, four only 1.1-1.2x (four
  MTP windows exceed the 8-token fast path of the expert kernels; with `--draft-tokens 1` four requests decode
  1.3x the tokens of one request with MTP 3, at the cost of 10 % for a request alone). RAM is the limit: with Flash-Next loaded about
  10 GB of the 96 GB stay free, about 400k tokens of int8 KV in RAM at tp 2. The server accepts up to 8
  (`--max-concurrency`); the 27B NVFP4 was measured at 1 to 8.
- With several requests decoding, an answer that outgrows its KV window when the GPUs are full first moves idle cached
  conversations to RAM (or drops them without a RAM tier); only if that cannot free a page does it end early with
  `finish_reason: length` (the request log counts it). `--kv-lease-window 0` restores the full reservation of v0.5.3.
- Moving a conversation between RAM and the GPUs shares PCIe with the two GPUs' exchange: decode of other requests
  is slower while a move is in flight (no difference otherwise).
- Generated tokens per agent task are about 1.6x the GB300 reference (above): the remaining gap with the datacenter
  stack is length of reasoning, not correctness.
- Output is not bit-reproducible run to run: which experts sit in VRAM (GPU path) or not (CPU path) depends on timing,
  and long streamed prefills differ in the last bits; quality is measured with that in place.
- Decode of Flash-Next is bound by host-RAM bandwidth for the experts computed on the CPU (about 270-300 per MTP
  window), so a slower CPU or RAM decodes slower.

## Tests
`ctest --test-dir build` runs the unit tests. The Engine integration tests run when their artifacts are set:
`NINFER_TEST_QWEN4_ARTIFACT` (a small synthetic Flash-Next artifact with the real layer geometry, made by
`tools/capy_flash/make_test_artifact.sh <checkpoint dir> <out dir>`, CPU only) and `NINFER_TEST_ARTIFACT` (a
Qwen3.8-27B artifact, for the tests inherited from ninfer-tp2). `ctest -j4` is supported; tests that need a model too
large for one 16 GB board skip with the sizes. For the 27B, `tools/tp2/gate.sh` is the reference run: it sets the
artifacts, runs the integration tests (`GATE_CTEST_EXCLUDE` excludes more), then checks golden ids, perplexity and
greedy texts against a recorded reference ([tools/tp2/README.md](tools/tp2/README.md)).

## What's new

**v0.11.1** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.11.1))
- The lossless receive of the tp-2 exchange by kernel ([PR #66](https://github.com/ValerioDolci/capy-flash/pull/66)) is on
  by default on pairs without P2P; `NINFER_TP_XIN_KERNEL=0` restores the copy engine. Same tokens as v0.11.0.

**v0.11.0** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.11.0))
- Faster prefill, same bits at the default: Flash-Next +13 % at 30k-100k (QSA attention at two CTAs per SM, fused
  hyper-connection Up, wider GDN tiles), 27B +2-5 % prefill and −1 % per decode round. Eleven PRs by @Mici81.
- Opt-in: `NINFER_TP_XIN_KERNEL=1` (same bits, tp-2 exchange without P2P off the copy engine),
  `NINFER_TP_AR_PAYLOAD=int8` and `NINFER_Q8_A8=128` (Flash-Next, change numerics): all three together +28 % on
  Flash-Next prefill, GSM8K and MMLU-Pro at the level of v0.10.0. See
  [Optional speed switches](#optional-speed-switches-off-by-default).
- Fixed: tool-call arguments that quote the tool-call markup no longer fall back to text
  ([#59](https://github.com/ValerioDolci/capy-flash/issues/59)).

**v0.10.0** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.10.0))
- Structured output ([#17](https://github.com/ValerioDolci/capy-flash/issues/17)): JSON mode, JSON Schema, GBNF
  grammars, regex and choice lists on Chat Completions (`response_format`, `structured_outputs`), Responses
  (`text.format`) and Anthropic Messages (`output_config.format`), streaming included, with MTP and next to other
  requests. Objects are strict by default as in vLLM (`--constraint-json-strict off` for literal JSON Schema); every
  client input is bounded (`--constraint-*`). Cost on the 27B: +0.3 % per round. See
  [docs/serving.md](docs/serving.md#structured-output). Unconstrained requests are served as in v0.9.0.
- `--pending-timeout-ms` defaults to 10 minutes (was 30 s): queued requests behind long prompts are no longer refused.
- Fixed: a request body nested very deeply (for example a tool schema) crashed the server; beyond
  `--max-request-json-depth` (default 1000) it now gets a 400.
- Grouped MoE metadata in one scan ([PR #52](https://github.com/ValerioDolci/capy-flash/pull/52) by @lupodevelop).

**v0.9.0** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.9.0))
- Opt-in, nothing changed at the default (greedy output identical to v0.8.2).
- `--kv-ram-active` (Flash-Next) now serves several requests at once, with or without `--elastic`; its RAM grows
  with the conversations and is given back when they end; the MTP layer's KV is in RAM too; beyond 262k tokens the
  VRAM region no longer grows. Three coding agents at once: 171 s with it against 274 s with the KV in VRAM
  (`--max-concurrency 3 --draft-tokens 1`) and 180 s one at a time, first token after 1.8 s instead of 8.1 s.
- `--decode-share F`: the requests already decoding keep a share of the time while a new prompt runs.
- Fixed: `--elastic` with a `--kv-capacity` above `--max-context` could run out of device memory; a failed elastic
  growth no longer leaves unzeroed memory mapped, and host-memory copies work inside CUDA Graph capture
  ([PR #49](https://github.com/ValerioDolci/capy-flash/pull/49), [PR #50](https://github.com/ValerioDolci/capy-flash/pull/50)
  by @lupodevelop).

**v0.8.2** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.8.2))
- Two opt-in community contributions, nothing changed at the default (greedy output identical to v0.8.1).
- Long context with YaRN: `--rope-yarn-factor F` (1-4) raises the accepted `--max-context` to native × F,
  up to 1,048,576 tokens on Flash-Next (`--kv-dtype nvfp4` on two 16 GB cards). Experimental; 4× costs
  perplexity. @splickz's work, ported and measured by @Hazelhof
  ([#28](https://github.com/ValerioDolci/capy-flash/issues/28), [PR #29](https://github.com/ValerioDolci/capy-flash/pull/29)).
- A portable benchmark, `bench/capy_flash/ladder.py`: one command, a fixed prefill/decode ladder with the
  machine and the server's P2P state recorded, so rows compare across setups
  ([#30](https://github.com/ValerioDolci/capy-flash/issues/30), by @Zeppe79).

**v0.8.1** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.8.1))
- Fixes and hardening, same tokens and speed as v0.8.0. Recommended for everyone.
- Flash-Next no longer hangs at «loading weights» on kernels that keep part of the RAM as CMA (Ubuntu 26.04):
  the page-locked experts stay outside it and the rest goes to pageable RAM
  ([#40](https://github.com/ValerioDolci/capy-flash/issues/40)). Two GPUs that advertise peer access but cannot
  map it (a VM with a 256 MiB BAR1) now start with `p2p off` instead of aborting
  ([#41](https://github.com/ValerioDolci/capy-flash/issues/41)).
- `--host-pinned-max-gib` is validated and always enforced, and `install.sh` stops on a failed checksum
  ([PR #44](https://github.com/ValerioDolci/capy-flash/pull/44), [PR #45](https://github.com/ValerioDolci/capy-flash/pull/45),
  [PR #46](https://github.com/ValerioDolci/capy-flash/pull/46) by @lupodevelop).
- Security: on a loopback address the server refuses cross-site and DNS-rebinding requests from web pages (`--cors`
  turns that off); the disk tier is private to the server's user; the API key is compared in constant time; response
  IDs are no longer predictable.

**v0.8.0** ([release notes](https://github.com/ValerioDolci/capy-flash/releases/tag/v0.8.0))
- Before updating: Qwen3.5 MoE (Qwen3.6-35B-A3B) and DFlash v1 (`--spec dflash`) are no longer supported; stay on
  v0.7.7 for them. Qwen3.8-27B (the dense Qwen3.5 architecture) and DFlash2 are not affected. As after every update,
  conversations saved by the disk tier start cold; the request log is schema 28.
- Tool calls the model gets almost right (undeclared or catalog-spelled names, missing closing tags, prose after the
  calls) come back as calls instead of plain text ([PR #35](https://github.com/ValerioDolci/capy-flash/pull/35),
  [PR #39](https://github.com/ValerioDolci/capy-flash/pull/39) by @omni326); correct output is unchanged.
- `--kv-ram-active` for Flash-Next: the KV cache in RAM, its VRAM to expert slots, for long contexts
  ([Profiles](#profiles)).
- Preparation for Qwen4: a dense variant of Flash-Next's architecture runs, checked on synthetic checkpoints. The
  released Qwen4 architecture is not known yet.
- Fixed: a probable fix for replies cut with `output_limit`
  ([#27](https://github.com/ValerioDolci/capy-flash/issues/27)); a two-GPU exchange that hangs while serving no
  longer stops the server ([#20](https://github.com/ValerioDolci/capy-flash/issues/20)); Flash-Next MTP drafts no
  longer depend on a race; a request near the maximum context no longer takes Flash-Next down.
- The 27B now runs the same Text forward as Flash-Next, with the same tokens and speed as v0.7.7.

**v0.7.7**
- Two engine-wide failures fixed. Both could follow a reasoning-close retry (`--reasoning-close-retry`, on by default) when a later request repeated the prompt or restored from its turn anchor. The fixes are a continuation that no longer registers the chat prompt's capture points ([PR #36](https://github.com/ValerioDolci/capy-flash/pull/36) by @omni326), and a state fork closed before a shared prefix is promoted at the reuse point. Recommended for everyone.

**v0.7.6**
- `/v1/messages` with four `cache_control` breakpoints (the Anthropic maximum, which OpenCode sends) no longer
  returns HTTP 400: the server's own turn anchor gives way when the request already uses every marker slot
  ([PR #34](https://github.com/ValerioDolci/capy-flash/pull/34) by @omni326).
- `--spec off` (or `none`) after a `--profile` runs the profile without speculative decoding, dropping its
  `--draft-tokens` and `--lm-head-draft` ([issue #33](https://github.com/ValerioDolci/capy-flash/issues/33)).
- Builds with GCC 16 ([issue #31](https://github.com/ValerioDolci/capy-flash/issues/31)); an out-of-range write in
  the context cache's KV lease bookkeeping is fixed.

**v0.7.5**
- `--miss-split calibrate` (Flash-Next NVFP4, opt-in; asked for in
  [issue #11](https://github.com/ValerioDolci/capy-flash/issues/11)): the decode miss split with costs measured on
  your machine instead of the fixed ones fitted on our Ryzen 9 9900X. It times the GPU's copy of an expert over PCIe at
  startup and the CPU's time per expert over the first 4,096 decode calls, logs both, then splits with them. On our
  box it lands where the fixed costs already were, and splitting does not beat `off` there (agentic replay, two runs:
  one GPU 74.8 / 73.5 tok/s against 72.9 / 73.0 with `off`; two GPUs 155.5 / 152.5 against 157.6 / 154.1), so the
  default stays `off`. It is there to check whether the split pays on other CPUs and links
  ([docs/serving.md](docs/serving.md#capy-flash-serving)).

**v0.7.4**
- Disk tier: a long conversation saved on disk now resumes even when a shorter match (a system prompt another
  conversation left) is in memory, and resumes work on one GPU (27B: 7.63 → 0.46 s on two GPUs, 15.18 → 0.60 s on
  one; [issue #6](https://github.com/ValerioDolci/capy-flash/issues/6)).
- `NINFER_HOST_MOE_CPUS` takes ranges (`0-15,32-47`); `--api-key-file FILE` and `NINFER_API_KEY` keep the key off
  the command line; a second server on the same port is refused.

**v0.7.3**
- CPU expert threads: when the allowed CPUs have no free SMT sibling (for example `taskset -c 0-15` on a 16-core /
  32-thread CPU, or SMT off) and the automatic count would take every core, one core is left to the main thread, which
  spin-waits on the GPU. Reported in [issue #11](https://github.com/ValerioDolci/capy-flash/issues/11): on an EPYC 7F52,
  16 threads on `taskset -c 0-15` dropped decode to 80-90 tok/s, against about 144 with 15. `--expert-cpu-threads N`
  still overrides ([details](docs/serving.md#cpu-expert-threads)).
- 64 GB edition: the startup line `Qwen4Exp offload experts: 49 of 49 expert layers in pageable host memory` was a
  false alarm (the check read the NVFP4 bank pointers, which ggml block banks do not use); fixed, no speed change
  ([issue #22](https://github.com/ValerioDolci/capy-flash/issues/22)).
- Documentation: every server flag and the four profiles in [docs/serving.md](docs/serving.md#capy-flash-serving),
  client setup, the chat template, the vision opt-out.

**v0.7.2: pick up where you left off.**
- Disk tier of the context cache (opt-in, `--disk-cache-dir DIR --disk-cache-gib N`; idea by @KouShuo): conversations
  survive a restart and resume without a new prefill (27B, two GPUs: 140k tokens in 1.08 s instead of 48 s); see
  [docs/serving.md](docs/serving.md#disk-tier-of-the-context-cache).
- Faster long prompts on the 64 GB edition of Flash-Next: prefill 64k +19 % on two GPUs (3,994 tok/s) and +35 % on
  one (2,921 tok/s), same tokens.
- CPU expert threads chosen from the machine (physical cores, affinity, L3 domains) instead of fixed counts;
  `--expert-cpu-threads N` sets them by hand.

**v0.7.1:** `--profile flash-next-1gpu` keeps 8 conversation states in RAM (+0.66 GiB): on the agentic replay with one
GPU 83 % of the prompt tokens come from the cache, 1.9 s to the first token (median) and 276 s for the whole replay
(v0.7.0: 34 %, 13.3 s, 858 s).

**v0.7.0: start small, grow later, same engine.**
- Flash-Next on 64 GB of RAM, with one or two GPUs (`--model flash-next-64gb`, GGUF i-quant experts).
- Qwen3.8-27B on one 16 GB card (`--model 27b-16gb`, 3-bit GSQ weights, 64k context, vision).
- AVX2 CPUs (Zen 2/3, Intel client) and Windows/WSL2 with its page-locked memory limit start Flash-Next too.
- Empty-reply guards on by default for every model; boards with P2P overlap the prefill exchanges on the direct copy
  (+29-45 % measured by @tzolkincz); a context-cache accounting error fails one request instead of the server.

Older releases: [GitHub releases](https://github.com/ValerioDolci/capy-flash/releases) and the
[development log](docs/capy-flash/development-log.md), which records the work behind the `Tnn` tags in the sources.

## License
Apache License 2.0 ([LICENSE](LICENSE)); attribution in [NOTICE](NOTICE). The original ninfer-tp2 README is the one of
[v0.4.9](https://github.com/ValerioDolci/capy-flash/blob/v0.4.9/README.md) (historical).
