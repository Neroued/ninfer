# NInfer maintainer tools

`tools/` contains the project-owner workflows for artifact conversion and inspection, independent
Python references, numerical parity diagnostics, benchmark orchestration, and serving smoke checks.
These tools are not part of the public download-and-run path; normal users should start with the
[project README](../README.md).

Run commands from the repository root with a Python 3.11 environment containing the dependencies
for the selected tool.

## Task index

| Task | Location |
|---|---|
| Build the 27B artifact | [`convert/qwen3_6_27b/`](convert/qwen3_6_27b/) |
| Build the 35B-A3B artifact | [`convert/qwen3_6_35b_a3b/`](convert/qwen3_6_35b_a3b/) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| Run the 27B Python reference | [`reference/qwen3_6_27b/`](reference/qwen3_6_27b/README.md) |
| Run the 35B-A3B Python reference | [`reference/qwen3_6_35b_a3b/`](reference/qwen3_6_35b_a3b/README.md) |
| Compare 27B reference/C++/source activations | [`parity/qwen3_6_27b/`](parity/qwen3_6_27b/README.md) |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |

## Artifact workflow

The converters consume an official local BF16 checkpoint and write one complete `.ninfer`
artifact. The paths below are placeholders for the maintainer's local checkpoint checkouts:

```bash
python3 -m tools.convert.qwen3_6_27b.convert \
  --model /path/to/Qwen3.6-27B \
  --out out/qwen3_6_27b.ninfer

python3 -m tools.convert.qwen3_6_35b_a3b.convert \
  --model /path/to/Qwen3.6-35B-A3B-base \
  --dflash-model /path/to/Qwen3.6-35B-A3B-DFlash \
  --out out/qwen3_6_35b_a3b.ninfer
```

Inspect either result:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

The exact source revisions, inventories, formats, and conversion recipes are recorded in
[`docs/maintainer/`](../docs/maintainer/). Published users download the completed artifacts from
Hugging Face instead of running these workflows.

## Python references and parity

```bash
python3 -m tools.reference.qwen3_6_27b \
  --weights out/qwen3_6_27b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 128

python3 -m tools.reference.qwen3_6_35b_a3b \
  --weights out/qwen3_6_35b_a3b.ninfer \
  --prompt "请简短介绍一下你自己。" --decode 128
```

The Python implementations are independent diagnostic references, not alternate public inference
products or generated-token goldens for the C++ engine.

Build the target-private C++ diagnostic only when needed:

```bash
cmake -S . -B build -DNINFER_BUILD_TOOLS=ON
cmake --build build --parallel --target ninfer-qwen3_6_27b-dump
```

See the parity README for activation and multimodal/MTP comparison commands.

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --preset core
```

See [`tools/bench/README.md`](bench/README.md) and [`bench/README.md`](../bench/README.md) for the
orchestrator and executable contracts.

## Serving smoke

After starting `ninfer-serve` in another terminal:

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 \
  --model qwen3.6-27b
```

The client exercises OpenAI, Anthropic, streaming, usage, multimodal, and tool-call response
surfaces against the resident process.
