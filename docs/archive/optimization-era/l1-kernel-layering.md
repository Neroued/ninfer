# L1 Kernel Layering & File Layout

> Status: implemented design. Synchronized 2026-07-11.

## 1. Boundary

L1 owns CUDA operators called by the fixed Qwen3.6 model schedule. It is specialized for one model
and one GPU target; it is not a generic runtime or compatibility layer.

The call chain is:

```text
public API -> host wrapper -> CUDA launcher -> CUDA kernel
```

| Layer | Typical location | Responsibility |
|---|---|---|
| Public API | `include/qus/kernels/<op>.h` | Stable semantic operation declaration |
| Wrapper | `src/kernels/wrapper/<op>.cpp` | Validation, workspace lifetime, qtype/shape/T dispatch |
| Launcher | `src/kernels/launcher/*.h,*.cu` | Grid/block/shared-memory setup and kernel launch |
| Common primitive | `src/kernels/common/*` | Zero-cost host/device math, memory, warp, and MMA leaves |
| Kernel | `src/kernels/kernel/*.cuh` | Device computation |

Only the public layer is an operator contract. Launcher/kernel names may describe phase, qtype,
shape, tiling, or other backend policy.

## 2. Public naming

Public names describe semantics:

- `linear`, not GEMV/GEMM or a quantization-specific name;
- `gqa_attention`, not separate decode/prefill names;
- `causal_conv1d_silu` and `gated_delta_rule`, with T-regime dispatch in the wrapper;
- explicit fused names such as `linear_add`, `linear_swiglu`, and `gated_rmsnorm` when fusion changes
  the operation contract;
- snapshot suffixes only when state storage semantics actually differ.

Backend-policy words such as `decode`, `prefill`, `recurrent`, `chunked`, `small_t`, `grouped`,
`W8G32`, and `mma` belong below the public API.

## 3. Wrapper contract

A wrapper performs these steps in order:

1. Validate dtype, shape, layout, contiguity, data pointers, and operator invariants.
2. Return for supported empty inputs where the operator contract allows them.
3. Open a `WorkspaceArena::scope()` and allocate temporary tensors.
4. Classify qtype, registered shape, T regime, cache dtype, or state form.
5. Call the selected private launcher.
6. Let the lexical scope restore the workspace cursor on normal or exceptional exit.

There is no public phase enum and no dispatcher object. The wrapper derives execution policy from
the tensors and state-form overload selected by the caller.

Examples:

- `linear` classifies weight format, `(N,K)`, and T, then selects a tuned GEMV/GEMM or reference
  backend;
- `gqa_attention` selects the small-T or prompt launcher and BF16/I8 KV-cache implementation;
- the in-place `causal_conv1d_silu` overload selects decode at T=1 and prefill otherwise;
- `gated_delta_rule` keeps chunk size 64 private and selects recurrent/chunked execution internally.

## 4. Fused operators

A fused entry is public only when its contract differs from the primitive it contains. Current fused
entries are:

- paired projections: `linear_pair`;
- projection plus residual update: `linear_add`;
- projection plus SwiGLU: `linear_swiglu`;
- attention/GDN projection groups: `attn_input_proj`, `gdn_input_proj`, `gdn_gating_proj`;
- normalization plus SiLU gate: `gated_rmsnorm`;
- causal convolution plus SiLU: `causal_conv1d_silu`.

Fallback composition belongs inside these wrappers. For example, unsupported fusion regimes fall
back to ordinary `linear`, `residual_add`, or `silu_mul` calls without changing the public verb.

## 5. Private model helpers

Fixed-schedule helpers are not public L1 operators:

```text
src/model/gqa_prompt_ops.h  # MTP-only prompt attention decomposition
src/model/mtp_ops.h         # MTP packing, accept/commit, counters, snapshot-slot updates
```

Their CUDA implementation may still live under `src/kernels/`, but consumers outside the model
schedule do not include them through `include/qus/kernels/`.

## 6. Linear subtree

`linear` and its fused projection backends are large enough to use a dedicated private subtree:

```text
src/kernels/linear/
  linear.cpp
  plan/
  codec/
  reference/
  gemv/
  gemm/
```

`linear.cpp` is the public wrapper implementation and owns plan classification. Fused wrappers may
call private launch declarations from the subtree. Backend filenames are allowed to encode shape,
qtype, regime, and fusion policy.

## 7. Common kernel primitives

Reusable CUDA leaves live under `src/kernels/common/`:

- `math.h`: host/device integer division, rounding, alignment, and sign extension;
- `math.cuh`: device activation math, approximate exp2, and packed FP16/BF16 conversions;
- `memory.cuh`: typed vector access, shared addresses, raw `cp.async`, and CUDA pipeline leaves;
- `warp.cuh`: sum/max warp collectives and the lane-0 block sum primitive;
- `mma.cuh`: `ldmatrix` and the exact MMA forms used by the fixed SM120 kernels.

These are private L1 headers, not operator APIs. Template parameters carry instruction immediates
such as copy width, cache policy, subgroup width, and wait depth. Device leaves are force-inlined;
they own no storage, launch policy, pipeline state, fragment class, or compatibility fallback.

Layout and algorithm policy remain local to the consuming kernel. In particular, shared-memory
swizzles, tile views, quantization codecs, online softmax, pipeline stage ownership, index mapping,
and compound reductions such as argmax are not common primitives.

Kernels include the leaf header they use. A linear or attention implementation must never include a
GDN header merely to obtain a CUDA intrinsic wrapper.

## 8. Host/device seam

Wrappers are ordinary host C++ and do not include device kernel headers. A private launcher header
declares a host-callable function in `qus::kernels::detail`; its `.cu` definition includes the
corresponding `.cuh` and performs the CUDA launch.

```text
wrapper.cpp -> launcher.h -> launcher.cu -> kernel.cuh
```

The launcher checks `cudaGetLastError()` after launch. Kernels assume wrapper validation has already
succeeded.

## 9. Build integration

`src/CMakeLists.txt` recursively collects `.cpp` and `.cu` files below `src/kernels/`. Public include
directories expose only `include/`; `src/` is a private include directory for the core library.
Tests or benchmarks that intentionally exercise model-private helpers add `src/` explicitly to their
own private include path.

## 10. Change policy

When the operator contract changes:

- replace old project-owned names directly;
- update L2, tests, and benchmarks in the same change;
- delete old declarations and headers;
- do not add aliases, forwarding wrappers, deprecated overloads, or transition headers;
- verify numerical behavior through the existing operator and model tests.
