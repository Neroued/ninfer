# Build system

NInfer builds from its source tree with CMake 3.28+ and C++/CUDA 20.
CUDA 13.4 is the validated development toolkit. Product commands and prerequisites are in the
[README](../../README.md#quick-start); test and measurement workflows live in
[tests](../../tests/README.md) and [benchmarks](../../bench/README.md).

## Execution targets

`NINFER_TARGET` selects one execution target for the entire build directory:

| Target | System | CUDA architecture |
|---|---|---|
| `rtx5090` (default) | Linux x86-64 | `120a` |
| `dgx_spark` | Linux AArch64 | `121a` |

[`cmake/NinferPlatform.cmake`](../../cmake/NinferPlatform.cmake) resolves the target and validates
`CMAKE_CUDA_ARCHITECTURES` before `project()` detects CUDA. After toolchain initialization it checks
`CMAKE_SYSTEM_NAME` and `CMAKE_SYSTEM_PROCESSOR`, so a cross toolchain is checked against its target
system. Unknown targets, conflicting CUDA architectures and target changes within an existing
build directory are configuration errors. Compiler changes also require a fresh build directory.
Libraries and executables retain their logical names across targets; their build trees are separate.

The same selection generates `<build-dir>/generated/core/platform.h` from
[`src/core/platform.h.in`](../../src/core/platform.h.in). Internal consumers explicitly include
`core/platform.h`; `ninfer_internal_includes()` supplies its private include root. Both
`NINFER_TARGET_RTX5090` and `NINFER_TARGET_SPARK` are defined as 0 or 1, so test them with `#if`.
These definitions do not propagate into public headers or third-party targets.

Target identity and capabilities are distinct. The platform header maps each target to
`NINFER_SHARED_PHYSICAL_MEMORY` and `NINFER_EXPERT_OFFLOAD` independently. Memory planning uses the
former; model offload entry points use the latter. Shared physical memory is not a Spark identity
test, and lack of shared memory does not itself imply CPU expert support. CMake still owns
target-private translation units and dependencies; macros handle small differences within shared
sources, not whole implementation catalogs. Benchmark reference data belongs to
[`bench/common/roofline.h`](../../bench/common/roofline.h), selected by target identity.

Target selection owns source composition and compilation, not model mathematics or artifact
capability admission. Ops retain the same public interfaces and report unsupported implementations
at their own capacity/execution entries. Models do not receive a target key or reproduce Op shape
catalogs. Runtime input-dependent dispatch and worker ISA selection remain inside the owning Op.

## Configuration

The root `CMakeLists.txt` owns language/toolchain constraints, build options and top-level
composition. The default configuration is Release. Ninja links and archives share the single-slot
`ninfer_link` pool; compilation uses the build command's parallelism.

| Cache option | Default | Scope |
|---|---|---|
| `NINFER_TARGET` | `rtx5090` | Execution target and native architecture |
| `NINFER_BUILD_ENGINE` | ON | Frontend, media decode, models and Engine |
| `NINFER_BUILD_APPS` | ON | CLI, HTTP server and perplexity evaluator; requires Engine |
| `BUILD_TESTING` | OFF | Tests for the selected components |
| `NINFER_BUILD_BENCHMARKS` | OFF | Benchmarks for the selected components |

Core, Artifact and Ops are always present. An Ops development build disables both Engine and Apps;
it does not discover FFmpeg, curl, Jinja, XGrammar or spdlog. Engine adds FFmpeg and the frontend
libraries. Apps or Engine tests add product input, logging and serving support, including curl and
spdlog. A Python 3 interpreter is found when tests are enabled. Standalone Python suites and
conversion/evaluation scripts run separately from CMake and CTest.

Shared CPU topology, affinity, worker startup and `TaskPool` belong to the ordinary C++
`ninfer_cpu` library, linked with hwloc >= 2.5 and Threads. The x86 collective executor and CPUID
implementation belong to `ninfer_cpu_executor`, compiled only for `rtx5090`. CPU MoE, the model runtime
when built with offload, and executor tests/benchmarks link it explicitly. Spark has no CPU expert
executor or AVX objects; its CPU MoE Op entry reports unsupported. Artifact Host page population uses shared
worker startup/affinity facilities directly, without depending on the arithmetic executor.
Core's direct-file I/O uses liburing >= 2.5.
Its registered files and buffers are caller-owned; the queue creates no CUDA resources or threads.

CUDA 13.4 and Python 3.11 describe the maintained environment, not configuration version gates.
FFmpeg is discovered without a version floor. The libcurl 7.85 minimum follows the use of
`CURLOPT_PROTOCOLS_STR` and `CURLOPT_REDIR_PROTOCOLS_STR`. Actual API support is checked by compilation.

The checked-in presets use independent `build/<preset-name>/` directories:

- `rtx5090-release`: product binaries;
- `rtx5090-dev`: products, tests and benchmarks;
- `rtx5090-ops`: Core, Artifact and Ops with their tests and benchmarks;
- `dgx-spark-release`: AArch64/SM121a product binaries;
- `dgx-spark-dev`: AArch64/SM121a products, tests and benchmarks;
- `dgx-spark-ops`: AArch64/SM121a Core, Artifact and Ops development.

Each has a matching build preset; development presets also have test presets. For example:

```bash
cmake --preset rtx5090-dev
cmake --build build/rtx5090-dev -j
ctest --preset rtx5090-dev
```

Keep compiler, Python, pkg-config and dependency paths in ignored `CMakeUserPresets.json`.
A local preset can inherit any checked-in preset:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local-spark",
      "inherits": "dgx-spark-ops",
      "binaryDir": "${sourceDir}/build/local-spark",
      "cacheVariables": {
        "CMAKE_CUDA_COMPILER": "/path/to/cuda/bin/nvcc",
        "Python3_EXECUTABLE": "/path/to/python3"
      },
      "environment": {
        "PKG_CONFIG_PATH": "/path/to/deps/lib/pkgconfig:$penv{PKG_CONFIG_PATH}"
      }
    }
  ]
}
```

There is no installed SDK, package export or configure-time dependency download.

## Targets and dependencies

[`cmake/Dependencies.cmake`](../../cmake/Dependencies.cmake) owns system dependency discovery,
repository-pinned third-party targets and their product-support conditions. External include
requirements follow their consuming targets.
[`cmake/NinferTargets.cmake`](../../cmake/NinferTargets.cmake) provides private project includes
and the RDC library and non-RDC archive policies.

[`src/CMakeLists.txt`](../../src/CMakeLists.txt) explicitly enters the component directories.
Each component's `CMakeLists.txt` owns its targets, dependency visibility and compile properties;
those files provide the current target and dependency inventory.

Component declaration order is not the execution dependency graph: CMake resolves named target
links during generation. Model runtime can refer to the runtime support target declared later;
support links core, while Engine links model runtime. The ownership and dependency contracts are
defined by [Engine architecture](engine-architecture.md) and [Op development](op-development.md).

### Shared-library boundary

Products, tests and benchmarks link the same internal shared libraries:

| Target | Contents |
|---|---|
| `ninfer_cpu` | Portable CPU topology and task pool |
| `ninfer_core` | Storage, transfers, CUDA execution and resource primitives |
| `ninfer_ops` | GPU Ops and their host dispatch |
| `ninfer_cpu_executor` | RTX 5090 collective CPU executor |
| `ninfer_moe_cpu` | RTX 5090 CPU MoE dispatch and ISA implementations |

Shared libraries are written to `<build-dir>/lib/`. Apps, tests and benchmarks retain their own
output directories. Build RPATH uses `$ORIGIN` for in-tree dependencies: moving a build's binaries
requires preserving their relative layout with `lib/`. External dependencies retain their discovered
search paths. Running from the build tree needs no `LD_LIBRARY_PATH` setup.

Artifact, model, runtime and test-support targets remain static libraries. The non-RDC CUDA archive
and CPU ISA object libraries are position-independent inputs to their owning shared libraries;
consumers do not embed another copy of them. CPU arithmetic qualification archives remain private
to the tests that select restricted ISA routes.

Project CUDA targets use the shared CUDA host runtime. Each CUDA shared library owns its device
link and registration; host functions connect library boundaries. Device functions and variables do
not form a cross-library interface. Host symbols retain normal C++ linkage, including internal
entries used by existing qualification tests. These are build-tree libraries, not an exported SDK
or a versioned ABI.

`LINK_DEPENDS_NO_SHARED` avoids relinking consumers for shared-library implementation changes.
The build still brings their dependency targets up to date. Header dependencies and link-command
changes rebuild affected consumers normally. Library types are explicit; `BUILD_SHARED_LIBS` does
not select a second NInfer linkage mode. Project defaults are set after third-party declarations.

## Explicit source ownership

Small components list their sources directly. Larger components include local `sources.cmake`
manifests which use `target_sources(<owner> PRIVATE ...)`. Included manifests anchor file paths
with `CMAKE_CURRENT_LIST_DIR`; `include()` does not change `CMAKE_CURRENT_SOURCE_DIR`.

- Every implementation under `src/` has one target owner and one explicit registration.
- Ops register implementations through their owning family manifests. Shared weight-input
  preparation and compile policy are owned by `src/ops/CMakeLists.txt`.
- Linear separates shared kernel and projection manifests. Format manifests list explicit sources
  and shape filenames with their actual `.cpp`/`.cu` extensions.
- Models retain distinct immutable-loading and mutable-runtime targets. Native loading, execution
  and Program manifests own architecture code; `common/qwen/sources.cmake` owns shared Frontend
  and Vision sources. Shared bindings and context/MoE mechanisms have explicit target entries.
  `common/moe/execution.*` owns the shared GPU calls; `schedule.*` bounds model-facing offload
  planning and traversal. `common/moe/offload/` is compiled only for `rtx5090`.
  Runtime startup orchestration links model runtime; native startup planners only depend on
  runtime support/contracts. Checkpoints/recipes do not define additional compile owners.
- New files require an explicit manifest edit. New families require an explicit parent include.
  There is no recursive glob, directory discovery or generated shape/dispatch registry.

Adding a source manifest does not require another library. Add a target only for a meaningful
dependency, reuse or compilation boundary. Shape semantics, supported domains and finite
dispatch remain in C++; CMake only selects translation units. A family may explicitly include
`${NINFER_TARGET}/sources.cmake` where target-private implementations exist. Shared sources are
registered once in the parent; a target must not include another target's private route catalog.
No automatic directory scan, missing-file fallback or global shape registry selects implementations.

## CUDA compilation boundaries

`ninfer_core` and `ninfer_ops` enable separable compilation (RDC) and resolve device symbols in
their own shared-library builds. Keep these device-link boundaries explicit; CUDA consumers do
not repeat the Ops device link or embed its fatbins.

`ninfer_ops_non_rdc` disables separable compilation and device-symbol resolution. It contains
the Linear, fused projection, attention and MoE sources that depend on
`setmaxnreg` register transfer. Their owning family manifests register these sources into the
non-RDC target; host launchers connect them to the normal Ops. All three CUDA targets
retain `-lineinfo`.

Source-list maintenance must preserve language, architecture, RDC mode, device-link ownership
and numerical compiler options. Splitting a manifest does not reduce kernel instantiation work
or justify combining shape translation units or enabling unity builds.

## Consumers and verification

`apps/CMakeLists.txt` explicitly defines the three product executables. Tests and benchmarks
include their domain registrations without creating new CMake subdirectories, preserving
`<build-dir>/tests/`, `<build-dir>/bench/` and CTest working directories. The helpers are local to those
consumers. Tests explicitly name their linked libraries; public-header and pure-host checks keep
their isolated include/link requirements.

Base builds register Core, Artifact and Op tests/benchmarks. Engine builds add frontend, model,
runtime and product checks. CPU executor/MoE and mixed CPU/GPU checks are registered only for
`rtx5090`; GPU MoE qualification has a separate executable. Target-specific Op cases are registered
only where the implementation is present.
Independent mathematical oracles and shared fixtures do not split with target tuning.

Tests can reuse compiled support libraries. Oracle options apply to the libraries containing
oracle code as well as the test executables. Test variants, skip return codes,
interop commands and special linker options belong with the affected registrations.
Failure-injection executables export only their replacement symbols through
`ninfer_test_interpose()`. Their replacements call the next shared-library definition through
the test-only `symbol_interpose.h` helper. This exercises calls made inside shared libraries without
adding production fault hooks; link-time `--wrap` does not intercept those calls.

For build-organization changes, compare configured target/source ownership, compile flags,
device links and CTest commands/properties; check the relevant option combinations, build the
affected targets and run their behavioral tests. A shape-source edit should rebuild its own
translation unit plus necessary downstream links. Build-time or inference-speed claims require
measurements at that scope.
