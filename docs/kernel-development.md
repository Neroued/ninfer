# Kernel Development

> Status: current L1 source-layout and verification guide.
>
> Repository-wide testing policy remains in [`../AGENTS.md`](../AGENTS.md). Public headers under
> `include/ninfer/kernels/` are the authoritative operator catalog.

## 1. L1 responsibility

L1 translates mathematical operator contracts into shape-specialized CUDA execution. It owns:

- the public operator surface;
- validation and dispatch at that surface;
- private host launchers;
- reusable CUDA primitives and kernels;
- the linear codec/GEMV/GEMM policy;
- numerical operator tests and per-operator benchmarks.

L1 does not own model layer order, q5090 file parsing, persistent state allocation, HTTP behavior,
or model-specific accept/commit bookkeeping.

## 2. Normal source path

Most operators use three layers:

```text
include/ninfer/kernels/<op>.h       public mathematical contract
            │
            ▼
src/kernels/wrapper/<op>.cpp     validation, views, dispatch, public symbol
            │
            ▼
src/kernels/launcher/<op>.*      private launch policy and CUDA entry
            │
            ▼
src/kernels/kernel/<op>.cuh      device implementation
```

Shared low-level primitives live under `src/kernels/common/`. They may provide memory, math, MMA,
or warp helpers, but must not grow a second public operator API.

Keep the wrapper thin. It may validate dtype/shape/layout, choose a launcher, construct zero-copy
views, and sequence a documented fallback. It must not allocate persistent storage, parse weights,
or contain a hidden model schedule.

## 3. Public contracts

Public APIs describe mathematical work, not CUDA mechanics. Names such as block size, split count,
codec implementation, shared-memory staging, or MMA policy stay private unless callers must provide
different semantic inputs.

Public fused operators are justified when they expose a useful semantic contract, for example:

- two projections sharing an input;
- projection plus residual accumulation;
- gate/up projection plus SwiGLU;
- grouped attention or GDN input projections;
- state snapshot variants required by MTP verification.

Do not add a public operator merely to expose a benchmark-only kernel experiment. Benchmark private
launchers through an existing boundary or promote the behavior only after it becomes a real runtime
contract.

## 4. Linear subtree

Low-bit linear work has its own private structure:

```text
src/kernels/linear/
├── codec/       Q4/Q5/Q6/W8G32 device decode
├── gemv/        T=1 and small-T shape-specialized kernels
├── gemm/        large-T tensor-core and fused projection kernels
├── plan/        finite dispatch classification
├── reference/   correct generic/dense fallbacks
└── linear.cpp   public `linear` dispatch integration
```

The dispatch plan may use only implementation-relevant facts already present at runtime:

- qtype and layout;
- exact `(N,K)` shape or a finite shape family;
- T regime;
- documented fusion group;
- dense/control versus packed representation.

Do not add model names, source tensor names, arbitrary string registries, or hypothetical future
hardware axes. A new specialized path must retain a correct fallback for supported current shapes,
not for retired formats or unrelated models.

Quantized kernels consume the canonical q5090 planes directly. Runtime repacking, decoded
long-lived BF16 weight copies, and hidden device allocation are not allowed.

## 5. State and workspace

Operators fall into three lifetime classes:

- stateless output operators write caller-provided tensors;
- workspace operators receive `WorkspaceArena&` explicitly and release allocations at the caller's
  scope boundary;
- stateful operators receive explicit KV/GDN state objects or tensors.

No operator owns sequence lifetime. Snapshot-capable GDN/conv operators receive their slot selector
explicitly because snapshot semantics are part of the call contract.

Workspace requirements must depend on real operator shape or configured prefill chunk, not silently
on the total context. Callers must be able to size arenas before execution.

## 6. Numerical implementation rules

The oracle is defined from model semantics after the documented input rounding. Tests must not copy
the CUDA algorithm and call that an oracle.

Important invariants include:

- BF16 public activations with explicitly chosen accumulation precision;
- zero-centered `(1+w)` RMSNorm where the model requires it;
- plain gated RMSNorm for the GDN internal norm;
- exact partial/MRoPE indexing;
- causal attention and correct KV append order;
- FP32 GDN gate and recurrent-state behavior;
- dequantization from q5090 codes/scales without a second layout;
- deterministic lowest-index tie handling for exact argmax;
- target-distribution correctness for sampled MTP verification.

Fast math, approximate activation formulas, reduced accumulation, or fused rounding changes are
allowed only when the operator's numerical contract and real downstream tolerance justify them.

## 7. Correctness tests

Tests are added only for risks allowed by `AGENTS.md`. For kernel work, the normal acceptable forms
are:

- numerical comparison to an FP64/FP32 mathematical oracle using BF16-rounded inputs;
- exact integer/code/metadata checks for artifact and KV quantization boundaries;
- real model shapes, plus stress/edge cases that can expose the targeted failure;
- repeated execution or sanitizer-backed coverage for memory/lifetime risks;
- a reproduced observable bug regression.

Avoid source scanning, call-order assertions, symbol-name tests, trivial enum/getter coverage, and
tests that preserve retired behavior.

The common numerical helpers under `tests/kernels/` provide established error summaries and
tolerance presets. A new test should reuse them when their contract matches, not invent a looser
one-off threshold.

## 8. Per-operator benchmarks

`build/bench/ninfer_<op>_bench` binaries run real Qwen3.6 shapes and isolate one kernel family. They are
profiling entrypoints, not correctness tests and not sufficient evidence for end-to-end speed.

A useful microbenchmark must state:

- exact shape, dtype, layout, and T regime;
- warmup and measurement method;
- whether weights/data are cold or cache-resident;
- bytes and FLOPs used for derived rates;
- the kernel/launcher path actually exercised.

Convenience GB/s printed by a benchmark is not an optimization gate by itself. Cache effects,
instruction pressure, occupancy, launch overhead, and tensor-core utilization must be interpreted
with profiler evidence.

## 9. NCU workflow

Use Nsight Compute after NSYS or benchmark evidence identifies a specific hot kernel. Collect only
the metrics needed to answer the current question, commonly:

- achieved occupancy and active warps;
- DRAM/L2 throughput and sector behavior;
- SM and tensor-pipe throughput;
- instruction mix;
- warp stall reasons;
- source/SASS correlation;
- roofline position.

Compare identical shapes and cache conditions before and after. An isolated kernel improvement is
not accepted if it changes numerical behavior outside tolerance or regresses the real inference
path.

## 10. NSYS workflow

Use Nsight Systems for full-inference questions:

- prefill versus decode phase time;
- CUDA API and kernel launch gaps;
- CPU/GPU synchronization;
- graph capture/replay behavior;
- MTP proposal/verify/accept round cost;
- memcpy and media/host overhead;
- whether a microbenchmark win matters in `ninfer_bench`.

NVTX ranges should identify stable user-meaningful phases. Do not add ranges solely to encode a
temporary implementation layout.

## 11. End-to-end acceptance

Kernel optimization completion normally requires:

1. build the affected targets;
2. run existing numerical/behavior tests covering the changed risk;
3. run the relevant per-operator benchmark;
4. collect NCU when the claim concerns kernel efficiency;
5. collect NSYS when the claim concerns full inference or launch behavior;
6. compare before/after `ninfer_bench` on the relevant prompt/decode matrix;
7. run `compute-sanitizer` when indexing, shared memory, state slots, or arena lifetime changed.

Published results must record the command, git commit, worktree state, GPU/toolchain, artifact
identity, and benchmark report. Raw reports belong under ignored `profiles/`; historical written
summaries belong in the documentation archive after the work is complete.

## 12. Review checklist

- Does the public API express semantics rather than one kernel strategy?
- Is dispatch limited to real current shapes and representations?
- Does the implementation consume q5090 directly without hidden allocation/repack?
- Are workspace and persistent-state lifetimes explicit?
- Is the numerical oracle independent of the CUDA algorithm?
- Does verification cover the real risk and real model shapes?
- Is profiler evidence collected under matching conditions?
- Does the change improve or preserve the relevant end-to-end path?
- Have completed tuning notes and plans been archived instead of becoming active design documents?
