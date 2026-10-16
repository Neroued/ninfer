# NInfer tools

`tools/` contains artifact conversion and inspection, benchmark orchestration, and serving smoke
checks. To download and run an existing artifact, start with the [project README](../README.md).
To build your own weights, use the [weight conversion guide](../docs/weight-conversion.md).

Run commands from the repository root with a Python environment containing the dependencies
for the selected tool. The maintained environment uses Python 3.11.

Python tools are independent of CMake; there is no `NINFER_BUILD_TOOLS` option.
Tools that launch native binaries require an explicit executable or build-directory argument.
The stored-scale CUDA quantization extension uses PyTorch's visible-device architecture selection;
`TORCH_CUDA_ARCH_LIST` can set an explicit architecture list.

## Task index

| Task | Location |
|---|---|
| Convert weights with an official or custom recipe | [`convert/`](convert/); [user guide](../docs/weight-conversion.md) |
| Inspect artifact metadata and objects | [`artifact/inspect.py`](artifact/inspect.py) |
| One-time upgrade of official v2 artifacts | [`upgrade_ninfer_v2_to_v3.py`](upgrade_ninfer_v2_to_v3.py), with positional `INPUT OUTPUT` paths |
| Run benchmark matrices | [`bench/`](bench/README.md) |
| Measure external Serve TTFT | [`bench/ttft/`](bench/ttft/README.md) |
| Exercise a resident HTTP server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Exercise thinking preservation through a managed server | [`smoke/serve_thinking_preservation.py`](smoke/serve_thinking_preservation.py) |
| Measure sustained device-memory bandwidth | [`hbm_bandwidth_probe.cu`](hbm_bandwidth_probe.cu); [build command](#memory-bandwidth-probe) |

## Memory bandwidth probe

This probe is a CMake benchmark target, using the selected native CUDA architecture and benchmark
reference table. With a configured development tree (substitute `dgx-spark-dev` on Spark):

```bash
cmake --build build/rtx5090-dev -j --target ninfer_memory_bandwidth_bench
./build/rtx5090-dev/bench/ninfer_memory_bandwidth_bench
```

## Artifact workflow

The [conversion guide](../docs/weight-conversion.md) owns commands for official/custom recipes,
mixed sources, optional components, resources and sharding. The converter writes a `.ninfer`
artifact plus its `.conversion.json` report. Inspect a result with:

```bash
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
```

Numeric formats, layouts and framing are defined by the references in the
[documentation map](../docs/README.md).

## Benchmark orchestration

`tools/bench/run_ninfer_bench_matrix.py` builds and runs the public-Engine benchmark matrix and
writes ignored local reports below `profiles/bench/`:

```bash
python3 tools/bench/run_ninfer_bench_matrix.py --build-dir build/rtx5090-dev --preset core --dry-run
python3 tools/bench/run_ninfer_bench_matrix.py --build-dir build/rtx5090-dev --preset core
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
  --server-bin build/rtx5090-dev/apps/ninfer-serve \
  --artifact out/qwen3_6_27b.ninfer --backend mtp
```
