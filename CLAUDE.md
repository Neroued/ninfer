# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Governing rules

[`AGENTS.md`](AGENTS.md) is the mandatory engineering contract for this repository: product scope,
ownership boundaries, scope control, evidence/provenance, numerical-correctness rules, performance
methodology, and test-admission rules. Read the sections relevant to the task before substantial
work; this file does not restate them. [`CONTRIBUTING.md`](CONTRIBUTING.md) adds the pull-request
and review requirements.

Consequences that most often surprise a fresh agent:

- Project-owned C++ APIs, CLIs, Python tools, fixtures, formats, and active docs keep **no**
  backward compatibility. When a change supersedes a project-owned path, delete it — do not add an
  alias, fallback, or second lane. The OpenAI and Anthropic protocol surfaces are the exception:
  they are real external contracts.
- Small diffs, few changed files, and short-term simplicity are not quality criteria; correctness,
  architecture, ownership, and performance are.
- Create a commit only when the user asks. Conventional Commit subjects, lowercase types
  (`feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, `chore`).
- Large artifacts, source checkpoints, and profiler outputs are local prerequisites. Do not
  download or regenerate them unless that is in scope, and never select an artifact by glob,
  mtime, or an unqualified "latest".

## Platform constraint

Builds target **only** `sm_120a` (RTX 5090) and require CUDA ≥ 13.1, CMake ≥ 3.28, C++20, and 64-bit
Linux. `CMakeLists.txt` fatal-errors on any other `CMAKE_CUDA_ARCHITECTURES`. A build failure is
never a reason to widen the architecture list. Also required: `pkg-config`, FFmpeg dev libs
(`libavformat/libavcodec/libavutil/libswscale`), and `libcurl >= 7.85`.

## Commands

Build (apps only; tests and benchmarks are off by default). Use `-j` with no numeric limit:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# -> build/apps/ninfer, build/apps/ninfer-serve
```

Tests:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

# one test
cmake --build build -j --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure

# per-comparison floating-point error records when setting or reviewing an Op criterion
NINFER_OP_REPORT_STATS=1 ctest --test-dir build -V -R '^ninfer_(rmsnorm|gqa_attention)_test$'
```

Python suites and the evaluation coordinator (separate venv):

```bash
python3 -m pytest tests/artifact tests/targets/qwen3_6_27b tests/targets/qwen3_6_35b_a3b \
  tests/test_bench_matrix.py tests/test_serve_corpus.py

PYTHONPATH=eval eval/.venv/bin/python -m unittest discover -s eval/tests -p 'test_*.py'
```

**Artifact-gated tests.** Anything needing a real `.ninfer` skips without its environment variable:
Python binding tests use `NINFER_QWEN3_6_27B_ARTIFACT` / `NINFER_QWEN3_6_35B_A3B_ARTIFACT` (falling
back to `out/qwen3_6_27b.ninfer` and `out/qwen3_6_35b_a3b.ninfer`); the opt-in real-Engine CTest
routes use `NINFER_QWEN3_6_27B_WEIGHTS` / `NINFER_QWEN3_6_35B_A3B_WEIGHTS`:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_prefix_real_test --output-on-failure
```

Benchmarks (`bench/README.md` for the matrix and Op benchmark contracts):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build -j --target ninfer_bench
./build/bench/ninfer_bench --weights out/qwen3_6_27b.ninfer -p 512,2048 -n 128 -r 5 --warmup 1
```

Maintainer workflows — conversion, artifact inspection, Python references, parity, benchmark
orchestration, serving smoke — are in [`tools/README.md`](tools/README.md), e.g.:

```bash
python3 -m tools.convert.qwen3_6_27b.convert --model /path/to/Qwen3.6-27B --out out/qwen3_6_27b.ninfer
python3 -m tools.artifact.inspect out/qwen3_6_27b.ninfer --objects
python3 -m tools.smoke.serve_contract --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

C++/CUDA sources follow `.clang-format` (LLVM base, 100-column limit, aligned consecutive
assignments).

## Architecture

NInfer runs a closed set of registered `.ninfer` checkpoint artifacts on one GPU with one resident
model instance and a startup-fixed 1–8 active requests. Every route — CLI, serving, benchmark —
reaches inference only through the public `ninfer::Engine`.

Public surface and dispatch:

- `include/ninfer/engine.h`, `include/ninfer/types.h` — the opaque Engine interface; `src/runtime`
  holds its PIMPL, generated-token transaction/publication policy, admission, KV capacity, and
  request-memory resolution.
- `src/targets/registry.{h,cpp}` — `ActiveTarget`, a `std::variant` over the per-package loaded
  model + Program instances. Target selection happens at load, from the artifact identity.

The family/variant scheme is the part that takes several files to see:

- `src/targets/qwen3_6` is an **identity-free** Qwen3.6-family runtime. It owns the shared
  `SequencePlan<Variant>` / `RequestPlan<Variant>` / `Program<Variant>` algorithms, tokenizer/chat
  template and output semantics, media preprocessing and MRoPE prompt construction,
  Text/Vision/speculative schedules, state transactions, workspace composition, and CUDA Graph
  capture/replay. Its `export/` headers are the family contract; `impl/` holds the templates.
- `src/targets/qwen3_6_27b` and `src/targets/qwen3_6_35b_a3b` are **peer compile-time Variants**,
  not deltas of each other. Each supplies `impl/config.h` (dimensions, storage facts),
  `impl/variant.{h,cpp}` (compile-time constants plus exactly three closed execution-leaf families:
  attention projection, GDN projection/control, post-mixer), `impl/load/bindings.*`, and its
  registered identities. Neither owns a copied Program, schedule, workspace recipe, or graph
  algorithm. There is no runtime family selection or target-dependent branch inside family
  scheduling.
- `qwen3.8-27b` is a **registered identity inside the 27B package** (model id `qwen3.8-27b`, target
  key `qwen3_8_27b` in `src/targets/qwen3_6_27b/export/.../package.h`), not a separate package. The
  version-2 artifact identity also selects the `groupwise-int` vs `nvfp4` weight profile — there is
  no runtime flag for it.
- Leaf Op implementations always live in `src/ops` (`linear`, `linear_add`, `linear_pair`,
  `linear_swiglu`, `attn_input_proj`, `gdn_*`, `linear_attention`, `sparse_moe`, plus `kernel/`,
  `launcher/`, `wrapper/`). Op ownership follows the mathematical or state-transition contract, not
  the first model that called it.

Supporting layers: `src/core` (device primitives, tensors/views, layouts, arenas, graph RAII,
paged/cyclic KV containers), `src/artifact` (generic `.ninfer` framing, descriptors, binding,
materialization — no checkpoint execution semantics), `src/serve` (OpenAI Responses/Chat, Anthropic
Messages, tool-call parsing, transport), `src/product` (prompt-input adapter, media acquisition,
load progress), `src/media/decode` (consumes already-owned bytes only), `apps/cli`, `apps/serve`.

The full ownership boundary list is the "Product and ownership boundaries" section of `AGENTS.md`;
the deeper contracts (concurrent inference architecture, paged KV cache, artifact container, tensor
formats, per-target model and artifact semantics, Op development and qualification) are the
`docs/maintainer/` routing map in the "Sources of truth" section. Read only what a live decision
needs.
