# NInfer tools

`tools/` contains artifact conversion and inspection, benchmark orchestration, and serving smoke
checks. To download and run an existing artifact, start with the [project README](../README.md).
To build your own weights, use the [weight conversion guide](../docs/weight-conversion.md).

Run commands from the repository root with a Python environment containing the dependencies
for the selected tool. The maintained environment uses Python 3.11.

Python tools are independent of CMake; there is no `NINFER_BUILD_TOOLS` option.

## Task index

| Task | Location |
|---|---|
| Convert weights with an official or custom recipe | [`convert/`](convert/); [user guide](../docs/weight-conversion.md) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Replace the chat template embedded in an artifact (weights unchanged; writes a new artifact directory) | [`replace_chat_template.py`](replace_chat_template.py), with positional `ENTRY.ninfer TEMPLATE.jinja OUT_DIR`; maintained templates in [`chat_templates/`](chat_templates/) |
| Check that every `ninfer-serve` flag, `NINFER_*` variable, conversion recipe and `Tnn` tag of the sources is in the user docs (ctest `ninfer_docs_guard_test`) | [`check_docs.py`](check_docs.py) |
| One-time upgrade of official v2 artifacts | [`upgrade_ninfer_v2_to_v3.py`](upgrade_ninfer_v2_to_v3.py), with positional `INPUT OUTPUT` paths |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Measure external Serve TTFT | [`bench/ttft/`](bench/ttft/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Check structured output (JSON object, JSON Schema, regex, choice, GBNF) end to end on a running server, validating replies with an independent JSON Schema validator | [`smoke/serve_structured_output.py`](smoke/serve_structured_output.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |
| Measure the physical HBM read/copy ceiling | [`hbm_bandwidth_probe.cu`](hbm_bandwidth_probe.cu); [build command](#standalone-hbm-probe) |
| Check whether the two-GPU mailbox transport works on a machine | [`tp2/mailbox_probe.cu`](tp2/mailbox_probe.cu); [build command](#standalone-tp2-mailbox-probe) |
| Validate the Qwen4 forward end to end on synthetic checkpoints: generate (dense FFN, PLE or not, hidden 2560 and 5120, Q8 hyper-connections, NVFP4 snapped), convert, FP64 oracle and floor, then the engine against it on every route | [`ref_q4/synthetic.sh`](ref_q4/synthetic.sh); [Qwen4Exp model](../docs/maintainer/qwen4-exp-model.md) |
| Compare one synthetic Qwen4 artifact's engine forward with the FP64 oracle (scoring, chunked prefill, MTP teacher forcing, forced ordinary and MTP decode, tp 1 and 2) | [`ref_q4/engine_check.sh`](ref_q4/engine_check.sh) |
| Snap a synthetic dense Qwen4 checkpoint's projections onto an exact NVFP4 grid, so that the FP64 oracle sees the weights a `qwen4_nvfp4` artifact decodes | [`ref_q4/snap_nvfp4.py`](ref_q4/snap_nvfp4.py); [fused routes](../docs/maintainer/qwen4-exp-model.md#the-text-forwards-options) |
| Run a Qwen4 reference on a token sequence and write its teacher-forcing dump: the FP64 NumPy oracle (`--impl fp64`, [`ref_q4/qwen4_reference.py`](ref_q4/qwen4_reference.py)) or the layer-streaming transformers reference (`--impl hf`, [`ref_q4/flashnext_ref.py`](ref_q4/flashnext_ref.py)) | [`ref_q4/ref_dump.py`](ref_q4/ref_dump.py) |
| Compare an engine dump with a reference dump (KL, top-1, NLL, per-layer hidden error) against a threshold profile; exit 0 pass, 1 a threshold failed, 2 bad or incomparable input | [`ref_q4/tf_compare.py`](ref_q4/tf_compare.py) |
| Read and write the teacher-forcing dump format (`tfdump/1`) shared by the references and the engine | [`ref_q4/tfdump.py`](ref_q4/tfdump.py) |
| Greedy token identity of two `ninfer` runs: a build against a release binary, or one binary under two environments, with flags for one arm only | [`greedy_ab.sh`](greedy_ab.sh) |

## Python dependencies of `ref_q4/`

The validation scripts run on Python 3.11 or newer with: NumPy (every script); safetensors
(checkpoints, and `.safetensors` dumps); PyTorch and transformers with the `qwen4_exp` models
(`flashnext_ref.py`, `ref_dump.py --impl hf`, `snap_nvfp4.py`, and the synthetic checkpoints of
`synthetic.sh`); tokenizers (`ref_dump.py --text-file`). They were validated with NumPy 2.5.3,
safetensors 0.8.0, PyTorch 2.14.0, transformers 5.16.1 and tokenizers 0.23.2. `synthetic.sh` and
`engine_check.sh` take the interpreter from `PYTHON`.

## Standalone HBM probe

This maintainer probe has an explicit standalone CUDA build, independent of the CMake benchmark
targets. Build it with the project's CUDA toolkit and run it from the repository root:

```bash
mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_120a tools/hbm_bandwidth_probe.cu \
  -o build/hbm_bandwidth_probe
./build/hbm_bandwidth_probe
```

## Standalone TP2 mailbox probe

`--tp 2` exchanges its all-reduce operands through a pinned-host mailbox (one kernel per GPU) when
the GPUs have no peer access, and falls back to driver-staged copies when the startup probe finds
the mailbox timing out or slow (WSL2 does that, [issue #1](https://github.com/ValerioDolci/capy-flash/issues/1)).
`tools/tp2/mailbox_probe.cu` runs the same check without the engine or a model — nvcc and the CUDA
runtime only; it includes the production exchange kernels from `src/ops/kernel/peer_exchange.cuh`,
so build it from a checkout — and prints driver, devices, peer access, the startup exchange, the
per-exchange cost of both transports (both mailbox kernels) and a verdict:

```bash
nvcc -O2 -std=c++20 -arch=sm_120a -o build/mailbox_probe tools/tp2/mailbox_probe.cu
./build/mailbox_probe            # devices 0 1; add `--simulate-hang` to see the timeout report
./build/mailbox_probe --payload 40960 --legacy   # the original exchange kernel
./build/mailbox_probe --sweep    # maintainer: every kernel variant at 4, 10 and 40 KiB
./build/mailbox_probe --timed --payload 40960    # maintainer: per-phase clock64 anatomy
```

Exit status 0 means the mailbox is usable, 2 that the engine will run on copies (`--no-tp-mailbox`
on builds without the startup probe), 1 a CUDA error or wrong sums (every exchange is checked bit
for bit against a CPU reference). On 2× RTX 5070 Ti without P2P, PCIe 5.0 x8 each (driver 595.91,
CUDA 13.1), per exchange in a graph of 128, two alternating slots:

| payload | pipelined kernel (default) | original kernel (`--legacy`) | copies |
|--:|--:|--:|--:|
| 4 KiB | 3.5 µs | 7.8 µs | 15.9 µs |
| 10 KiB (decode token) | 3.7 µs | 8.8 µs | 17.2 µs |
| 40 KiB (MTP-3 verify) | 6.2 µs | 19.6 µs | 20.0 µs |

The startup exchange takes about 0.03 ms and a missing peer is reported after about 0.8 s with
either kernel.

## Artifact workflow

The common converter reads selected local sources and writes a `.ninfer` artifact plus its
`.conversion.json` report. These examples include the optional weights used by the official
artifacts. The input paths are placeholders for local checkpoint checkouts:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b --components text,vision,mtp --proposal \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --name qwen3.6-27b \
  --out out/qwen3_6_27b.ninfer

python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b --components text,vision,mtp,dflash2 --proposal \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --name qwen3.8-27b \
  --out out/qwen3_8_27b.ninfer
```

Inspect a result:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

Recipes, mixed sources, custom methods, resources and sharding are described in the
[conversion guide](../docs/weight-conversion.md). Numeric formats, layouts and framing are defined
by the references linked from the [documentation map](../docs/README.md).

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

For request-arrival latency, use the managed Qwen3.8-27B NVFP4/FP8 TTFT campaign. Its measurement
runner remains an external-only HTTP client; the separate controller owns Serve lifecycle and
artifacts. See [`tools/bench/ttft/README.md`](bench/ttft/README.md).

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.

For typed rewrite-checkpoint and thinking-history behavior, the managed smoke script launches a
real server and consumes the repository fixture:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```
