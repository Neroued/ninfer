# Kernel Development

This guide defines the implemented shared-operator and target-private CUDA boundary. Repository-wide
engineering and verification principles remain in [`../AGENTS.md`](../AGENTS.md).

## 1. Ownership

Shared kernel code translates a complete mathematical operator contract into efficient CUDA
execution. It owns:

- host-side dtype/shape/layout validation;
- finite dispatch over supported representations and real shape regimes;
- private launch policy;
- reusable CUDA primitives and kernels;
- linear codec/GEMV/GEMM policy;
- numerical operator tests and per-operator benchmarks.

Shared kernels do not own artifact parsing, model layer order, sequence lifetime, prefix state,
CUDA Graph lifetime, MTP commit policy, user-visible stops, or serving behavior.

A fused or fixed-shape operation whose full invariant is true only for one checkpoint/GPU remains
under that target's `impl/`. Similar names are not enough to promote code into shared kernels.

## 2. Source boundary

Current shared operators are built explicitly from `src/kernels`. Host contracts and dispatch are
kept next to their CUDA implementation by complete operator:

```text
src/kernels/
├── <shared-operator>/
│   ├── <operator>.h           mathematical contract and host entry
│   ├── <operator>.cpp         validation and finite implementation selection
│   ├── launch.h               private launcher declaration
│   ├── launch.cu              CUDA launch and device entry
│   └── detail.cuh             optional device-only implementation material
├── linear/                    low-bit codec, plans, GEMV/GEMM, and fallbacks
└── common/                    narrow math, memory, MMA, and warp primitives
```

The exact file count may be smaller, but the language and ownership rules are stable:

- host-visible declarations describe mathematical semantics, not block/warp/tiling strategy;
- `.cpp` files own host validation and finite dispatch;
- `.cu` files own CUDA launches and `__global__` definitions;
- `.cuh` files contain device-only implementation material;
- callers provide outputs, workspaces, state views, and streams;
- a kernel never allocates sequence state or controls a generated-token transaction.

Kernel headers are internal development contracts. The installed product API is only the opaque
Engine and owning host values.

## 3. Shared versus target-private

An operation belongs in shared kernels only when its complete contract is independent of a target's
layer schedule and state machine. Current examples include RMSNorm, RoPE, attention primitives,
GDN mathematical primitives, sampling, and low-bit linear operations.

The Qwen3.6-27B target retains:

- fixed Text/Vision/MTP call order and fusion decisions;
- target-specific graph shapes and stable I/O buffers;
- KV/GDN/MTP frontier alignment and prefix repair;
- exact-checkpoint proposal/verification composition;
- any helper whose dimensions or semantics are only valid for this target.

A shared operator may receive target-selected dimensions and explicit state views. It must not
import target headers merely to discover those values.

## 4. Linear storage and dispatch

Low-bit linear work consumes registered `.ninfer` tensor views directly:

```text
src/kernels/linear/
├── codec/       Q4/Q5/Q6/W8G32 device decode
├── gemv/        T=1 and small-T specialized kernels
├── gemm/        larger-T tensor-core and fused projection kernels
├── plan/        finite dispatch classification
├── reference/   correct generic/dense fallbacks
└── linear.cpp   integration
```

Dispatch may use only facts needed by the implementation:

- numeric format and registered storage layout;
- exact `(N,K)` or a finite real shape family;
- T regime;
- semantic fusion group;
- dense/control versus grouped-quantized representation.

It does not dispatch on source checkpoint tensor names or arbitrary runtime strings. Persistent
runtime repacking, hidden long-lived decoded weight copies, and hidden device allocation are not
part of the inference path.

## 5. State and workspace

Operators fall into three forms:

- stateless operators write caller-provided outputs;
- workspace operators receive explicit arena/tensor storage scoped by the caller;
- stateful operators receive explicit KV/GDN state or tensor views.

Program owns sequence and graph lifetime. Snapshot-capable recurrent operations receive their slot
or boundary inputs explicitly. Workspace requirements derive from the real operator shape or
configured prefill chunk so the target can plan memory before execution.

## 6. Numerical contracts

The oracle comes from model semantics after the documented input-rounding boundary, not from a copy
of the CUDA algorithm.

Important current invariants include:

- BF16 public activations with explicitly selected accumulation precision;
- zero-centered `(1+w)` RMSNorm where required;
- plain gated RMSNorm for the GDN internal normalization;
- exact partial/MRoPE indexing;
- causal attention and correct KV append order;
- FP32 GDN gate and recurrent-state behavior;
- direct dequantization from registered codes/scales;
- lowest-index tie behavior for exact argmax;
- correct sampling and MTP verification distributions.

Fast math, approximate activations, reduced accumulation, or changed fused rounding are acceptable
only when the operator contract and end-to-end numerical evidence support them.

## 7. Correctness verification

Add a permanent kernel test when it protects a concrete supported risk, commonly:

- comparison to an independent FP64/FP32 mathematical oracle with BF16-rounded inputs;
- exact code/scale/metadata checks at a registered storage or KV boundary;
- the real model shapes plus an edge case that can expose the changed indexing or dispatch;
- a reproduced numerical or state bug.

Use the established helpers in `tests/kernels` when their tolerance contract matches. Do not add
source scans, call-order tests, symbol-name checks, trivial configuration tests, or checks for
retired behavior.

## 8. Per-operator benchmarks

`build/bench/ninfer_<op>_bench` binaries isolate one operator at real Qwen3.6 shapes. They
are profiling entry points, not correctness tests and not sufficient end-to-end evidence.

A useful result states:

- exact shape, format/layout, and T regime;
- warmup and measurement method;
- relevant cache conditions;
- bytes/FLOPs used for derived rates;
- the actual dispatch/launcher path.

Convenience GB/s is descriptive. Cache behavior, occupancy, instruction pressure, launch overhead,
and tensor-core utilization require the profiler evidence relevant to the change.

## 9. NCU and NSYS

Use Nsight Compute after a benchmark or Nsight Systems trace identifies a specific kernel question.
Collect the metrics needed to answer that question, such as occupancy, DRAM/L2 traffic, SM/tensor
throughput, instruction mix, warp stalls, source/SASS correlation, or roofline position.

Use Nsight Systems for full-request questions:

- prefill/decode/Vision phase time;
- CPU launch gaps and synchronization;
- CUDA Graph capture/replay;
- MTP proposal/verify/accept cost;
- copies and host/media overhead;
- whether a microbenchmark improvement affects `ninfer_bench`.

NVTX ranges identify stable inference phases rather than temporary file/function layout.

## 10. End-to-end acceptance

Verification should match the change:

1. build the affected component and exact target;
2. run the numerical or behavior checks covering the changed risk;
3. measure the relevant operator when a kernel path changed;
4. collect NCU/NSYS when the performance claim requires those observations;
5. compare the public-Engine `ninfer_bench` path on the relevant ordinary/MTP and prompt/decode
   cases;
6. use a memory checker when changed indexing, shared memory, or lifetime makes it useful.

Record the command, target/artifact, GPU/toolchain, relevant settings, and emitted reports needed to
interpret a local result. Raw profiler and benchmark outputs stay under `profiles/`; completed
written investigations belong in the documentation archive.

## 11. Review checklist

- Does the interface express mathematical semantics rather than one CUDA strategy?
- Is the code genuinely shared, or should it remain target-private?
- Is dispatch limited to current registered representations and real shapes?
- Does the kernel consume `.ninfer` views directly without hidden allocation or repacking?
- Are workspace, state, and stream lifetimes explicit?
- Is the oracle independent of the CUDA implementation?
- Does verification cover the actual changed risk?
- Does the change improve or preserve the relevant product route?
