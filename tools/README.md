# NInfer tools

This directory contains offline artifact tooling, the independent Python reference, numerical
diagnostics, benchmark helpers, and the serving smoke client. Run commands from the repository
root. The normal C++ products live under `apps/` and `bench/`; most Python tools here are invoked
manually with `python -m`.

The currently registered target is `qwen3_6_27b_rtx5090`. Target-specific tools use that exact key
in their directory name. Shared artifact mechanisms and checkpoint-reading helpers stay outside a
target directory. The accepted future `qwen3_6_35b_a3b_rtx5090` converter is available for artifact
bring-up, but it does not register a runtime target or advertise 35B product support.

## Start by task

| Task | Start here |
|---|---|
| Build a `.ninfer` artifact | [`convert/qwen3_6_27b_rtx5090/`](convert/qwen3_6_27b_rtx5090/) |
| Bring up the future 35B-A3B artifact | [`convert/qwen3_6_35b_a3b_rtx5090/`](convert/qwen3_6_35b_a3b_rtx5090/) |
| Inspect an artifact directory | [`artifact/inspect.py`](artifact/inspect.py) |
| Verify an artifact against the BF16 checkpoint | [`convert/qwen3_6_27b_rtx5090/verify.py`](convert/qwen3_6_27b_rtx5090/verify.py) |
| Run independent Python inference | [`reference/qwen3_6_27b_rtx5090/README.md`](reference/qwen3_6_27b_rtx5090/README.md) |
| Compare Python, C++, and source-model activations | [`parity/qwen3_6_27b_rtx5090/README.md`](parity/qwen3_6_27b_rtx5090/README.md) |
| Run the end-to-end performance matrix | [`bench/README.md`](bench/README.md) |
| Exercise a resident OpenAI/Anthropic server | [`smoke/serve_contract.py`](smoke/serve_contract.py) |

## Artifact workflow

Create, inspect, and verify the artifact:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.convert.qwen3_6_27b_rtx5090.convert \
  --model /path/to/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b_rtx5090.ninfer

/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.artifact.inspect out/qwen3_6_27b_rtx5090.ninfer --objects

/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.convert.qwen3_6_27b_rtx5090.verify \
  out/qwen3_6_27b_rtx5090.ninfer \
  --model /path/to/Qwen3.6-27B/base-hf-bf16
```

`artifact/` owns the generic Python `.ninfer` codec and registered numeric/layout formats.
`convert/common/` owns architecture-independent safetensors and quantization helpers.
`convert/qwen3_6/common/` owns narrow family-invariant recipe, shortlist, Vision, resource-name,
and writer mechanics. Each target converter still owns its exact config, complete inventory,
source mapping, draft ranking policy, and verification. Targets never import sibling targets.

The 35B-A3B converter deliberately uses the measured 27B ranking because the two checkpoints have
the same semantic token-id vocabulary:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.convert.qwen3_6_35b_a3b_rtx5090.convert \
  --model /home/neroued/models/llm/qwen/Qwen3.6-35B-A3B/base-hf-bf16 \
  --out out/qwen3_6_35b_a3b_rtx5090.ninfer
```

The ranking path is fixed by the exact target converter. The sidecar records that its source
measurement target was `qwen3_6_27b_rtx5090`; the selected rows themselves are always gathered from
the 35B BF16 output head before Q4 quantization.

## Python reference and parity

Run artifact-native Text, Vision, and MTP inference:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  -m tools.reference.qwen3_6_27b_rtx5090 \
  --weights out/qwen3_6_27b_rtx5090.ninfer \
  --prompt "Explain prefill and decode briefly." \
  --decode 128 --mtp-draft-tokens 3
```

The reference is an independent correctness implementation, not the C++ runtime wrapped in Python.
Install its dependencies from
[`reference/qwen3_6_27b_rtx5090/requirements.txt`](reference/qwen3_6_27b_rtx5090/requirements.txt).

For activation comparisons, build `ninfer-qwen3_6_27b-dump` and follow the target parity guide:

```bash
cmake --build build -j --target ninfer-qwen3_6_27b-dump
```

The C++ diagnostic is implemented in [`qwen3_6_27b_dump/main.cpp`](qwen3_6_27b_dump/main.cpp). It is
target-private and is not part of the public `ninfer::Engine` API.

## Performance tools

[`bench/run_ninfer_bench_matrix.py`](bench/run_ninfer_bench_matrix.py) drives the public-Engine
`ninfer_bench` executable and writes local reports under `profiles/bench/`. Inspect its command
matrix without running the model:

```bash
/home/neroued/miniconda3/envs/py311/bin/python \
  tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run
```

[`bench/make_bench_corpus.py`](bench/make_bench_corpus.py) regenerates or checks the committed token
corpus. [`bench/flash_attn_gqa_bench.py`](bench/flash_attn_gqa_bench.py) is an optional
FlashAttention baseline for the target GQA kernel; it is not a runtime dependency.

CUDA operator microbenchmarks themselves live under the repository-level `bench/` directory.

## Serving smoke

After starting `build/apps/ninfer-serve` in another terminal, exercise the advertised OpenAI,
Anthropic, streaming, and multimodal routes:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This is a manual real-server check, not a CTest.

## Directory map

```text
tools/
├── artifact/                         generic .ninfer codec, layouts, formats, and inspector
├── convert/common/                   shared checkpoint-reading and quantization helpers
├── convert/qwen3_6/common/            narrow Qwen3.6-family conversion leaves
├── convert/qwen3_6_27b_rtx5090/      exact-target converter, inventory, recipe, and verifier
├── convert/qwen3_6_35b_a3b_rtx5090/  future-target converter, inventory, and source recipe
├── reference/qwen3_6_27b_rtx5090/    independent artifact-native Python reference
├── parity/qwen3_6_27b_rtx5090/       target numerical comparison tools
├── qwen3_6_27b_dump/                 C++ target activation diagnostic
├── bench/                            corpus generation and performance orchestration
├── smoke/                            resident-server product smoke client
└── freq_corpus/                      tracked draft-head frequency input and experiment records
```

Generated artifacts, profile reports, Python bytecode caches, and model weights are local outputs
and are ignored by Git. The tracked training-frequency file under `freq_corpus/` is different: it
is a registered input to the current target's draft-head conversion recipe.
