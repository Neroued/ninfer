# NInfer Op Architecture and Atomic Migration Plan

> Status: reviewed plan; pending user approval on 2026-07-15.
>
> Scope: define the NInfer execution-layer Op boundary, classify the current Qwen3.6-27B
> implementation by that definition, and plan one atomic migration from the temporary
> kernel/model-owned organization to the final Op organization. This document does not authorize
> implementation until it is reviewed and accepted.

## 1. Definition of an Op

An Op is a host-callable, semantically closed computation contract in the engine execution layer.
Given all explicit inputs, weights, state, and semantic parameters, it completely defines its
outputs and state changes without depending on the identity of the model or the schedule that calls
it.

Formally, an Op defines:

```text
(outputs, new_state) = F(inputs, weights, old_state, semantic_parameters)
```

Execution resources and device facts are supplied separately:

```text
workspace, CUDA stream, device facts
```

The wrapper selects an implementation from those facts. They affect how the contract is executed,
not what the contract means.

For a stochastic Op, every semantic random input is also explicit. This includes the seed or RNG
state, logical position, purpose/domain separation, and draw index required by the algorithm. The
contract defines the probability process and state transition; it does not automatically promise a
particular backend hash function or bitstream.

### 1.1 Required properties

A component is an Op only when all of the following are true.

#### 1. Logical effect

It computes or changes a logical tensor value or an explicit local state value. Its effect can be
written as a formula, an index mapping, an algorithm, a probability process, or an
`old_state -> new_state` transition.

#### 2. Semantic closure

The same contract states every observable output and mutation, including:

- output values;
- in-place updates;
- cache writes;
- counters and statistics written by the call;
- aliasing restrictions;
- valid and invalid regions;
- intermediate cast/rounding boundaries when observable.

An effect is not allowed to remain implicit merely because the current caller already knows it.

#### 3. Explicit dependencies

Everything that affects the result is available through:

- input `Tensor` views;
- read-only `Weight` views;
- explicit state/container references;
- scalar or configuration parameters;
- dtype, numeric format, layout, and logical shape metadata;
- explicit stochastic inputs when applicable.

An Op must not read a target key, checkpoint tensor name, layer topology, Program phase, prefix
ledger, request object, or artifact provenance to discover its operands or semantics.

#### 4. Schedule independence

An Op performs one complete local transformation. It does not decide:

- whether a model feature is enabled;
- which model layer runs next;
- where a prompt chunk starts or ends;
- whether to enter an MTP proposal/verification round;
- how many accepted tokens are committed to the sequence;
- when cache/recurrent state is advanced, rewound, committed, or discarded;
- how product statistics are published or interpreted.

The schedule may choose views, indices, scalar values, and the call time. Once called, the Op owns
the complete local transformation specified by its contract.

#### 5. Implementation independence

The contract does not expose:

- grid, block, warp, CTA, tile, split, or shared-memory policy;
- MMA instruction selection;
- a `__global__` symbol;
- the number of kernels launched;
- a CUDA implementation filename;
- a model-labelled backend path.

Those facts select or describe an implementation of the Op.

#### 6. Independent invocation and verification

The operation has a meaningful host-callable boundary and can be tested from its explicit
inputs/outputs/state without executing a complete model schedule. A device helper that only forms
one step inside a larger implementation is not promoted into an Op merely because it has a formula.

### 1.2 Facts that do not disqualify an Op

An Op may currently have:

- one caller;
- one supported checkpoint shape;
- one registered quantization format;
- one token-count regime;
- one SM120 implementation;
- a fused formula;
- mutable explicit state;
- stochastic behavior.

Reuse count, shape breadth, and hardware breadth do not determine ownership. They only define the
supported domain and available implementation set.

Therefore an exact-shape RTX 5090 kernel remains an implementation of an Op; it does not belong to
`qwen3_6_27b_rtx5090` merely because that target is its first caller.

### 1.3 Fusion and granularity

A fused computation is an Op when the complete fused result and all state effects form one closed
contract. Examples include `linear_swiglu`, `linear_add`, GQA with KV append, and MTP acceptance.

Fusion is not treated as a private implementation detail when it changes an observable rounding
boundary, output layout, state transition, or host-callable contract. Conversely, a tile decoder,
partial reduction, epilogue helper, or warp primitive remains implementation detail when it cannot
be invoked as the complete transformation.

An Op may use other Ops as a fallback implementation. This does not turn the calling wrapper into a
model schedule when its own complete formula remains independent of model call order.

### 1.4 Logical assignment versus raw transfer

Classification follows the logical contract rather than the CUDA API used underneath:

- `assign_i32_scalar(source, destination)` is an Op when `source` and `destination` are distinct
  logical state variables and the contract is `destination' = source`;
- `cudaMemcpyAsync(dst, src, bytes, kind, stream)` is a core/host transfer mechanism when it is
  described only as moving bytes or storage.

The semantic scalar assignment API accepts typed I32 scalar views rather than addresses, byte
counts, or transfer directions; source and destination must not alias. Its implementation may use a
D2D copy without changing its Op identity. An API expressed in addresses/bytes/transfer kind remains
a core transfer regardless of which CUDA primitive implements it.

Likewise, a copy that performs cast, transpose, concat, scatter, remap, or another logical index
transformation belongs to the corresponding Op. A raw H2D upload does not become an Op.

### 1.5 Op admission test

Every proposed Op is classified with this sequence:

1. Does it change a logical tensor or explicit state value?
   - No: it is infrastructure, planning, transfer, validation, or implementation detail.
2. Can its full value/state transformation be defined only from explicit arguments and metadata?
   - No: it belongs to target schedule or product logic.
3. Does it decide call order, lifecycle, frontier, commit, rollback, or request policy?
   - Yes: it belongs to Program/schedule/runtime.
4. Is it only a partial implementation step beneath another complete contract?
   - Yes: it is an Op-private launcher/kernel/common/linear detail.
5. Otherwise it is an Op and receives one repository-internal semantic contract.

A useful review question is:

> If the target name and call site are removed, can the transformation still be described and
> verified completely?

If yes, it is normally an Op. If the description still requires “during this checkpoint's MTP
fallback”, “at this Program frontier”, or similar schedule context, it is not.

## 2. What is not an Op

The following categories remain outside the Op catalog.

### 2.1 Program and target schedule

Program/schedule owns:

- Text, Vision, and MTP call order;
- exact layer topology and weight roles;
- prompt chunking and multimodal span interpretation;
- proposal/verification round orchestration;
- prefix/frontier/commit/rollback policy;
- cache and recurrent-state instance lifecycle;
- CUDA Graph variants and stable graph addresses;
- interpretation and publication of product statistics.

A schedule helper that only derives operands and composes existing Ops remains target-owned.

### 2.2 L0 storage and execution mechanisms

Core owns non-Op mechanisms such as:

- `Tensor` and `Weight` views;
- device and dtype facts;
- layout builders and checked spans;
- `WorkspaceArena` and device arenas;
- KV-cache physical containers and views;
- CUDA Graph RAII;
- raw H2D/D2D transfers when exposed as storage movement.

`KVCache::advance`, `rewind`, and `reset` are container cursor mechanics, not Scalar Ops. Core
implements them; Program owns the policy and every invocation.

An Op may consume these types without owning their lifetime policy.

### 2.3 Op implementation details

The following are parts of an Op implementation, not separate Ops:

- wrappers' validation and dispatch logic;
- launchers;
- `__global__` and `__device__` functions;
- linear format codecs;
- plan/selector code;
- partial reductions and staging kernels;
- common math, memory, warp, and MMA primitives;
- workspace layout helpers and sizing calculations.

### 2.4 Artifact, frontend, product, and tooling

Artifact framing/binding/materialization, conversion recipes, tokenizer/template handling, media
acquisition/decoding, request-schema translation, and serving transport are not Ops. They prepare or
move product data rather than execute the registered model's device computation contracts.

## 3. Consequences for ownership

The definition produces one clear rule:

> Every Op contract and every implementation of that contract are owned by the central Op layer.
> Target packages own schedules and state policy, not mathematical implementations.

There is no target-facing private Op category. A target directly invokes only:

- repository-internal Op contracts;
- L0/core storage and transfer mechanisms;
- its own host-side schedule composition.

An implementation-private helper may exist below an Op, but the target cannot include or invoke it.
If a target must directly invoke a device transformation, that transformation must either satisfy
the Op definition and receive a contract, or be expressed by composing existing Ops/core
mechanisms.

### 3.1 `mtp_accept_tokens` is an Op

The current fused MTP acceptance kernel remains intact and becomes a normal stateful Op. Its
contract must include:

- greedy and sampling acceptance paths;
- draft-prefix acceptance conditions;
- rejection residual sampling;
- all-accepted bonus sampling;
- semantic RNG inputs;
- `sampled_out` initialization and valid output interval;
- new values of `num_sampled`, `accepted`, `token`, `length`, and `ar_pos`;
- the exact increment of every written statistics slot;
- token-count updates through `SamplingConfig` when enabled.

Program still decides whether and when to call this Op, how many returned tokens are committed, how
KV/GDN/MTP state follows that commit, and how statistics are exposed. Making the complete existing
side effects part of the Op contract avoids both a private gray layer and an unrelated kernel split.

### 3.2 Bookkeeping names do not create model Ops

The current model-labelled helpers are re-expressed as generic transformations:

| Current helper | Final classification |
|---|---|
| `mtp_increment_i32` | `increment_i32_scalar` Op |
| `mtp_count_fallback_step` | target selects the `stats[3]` view, then calls `increment_i64_scalar` |
| `mtp_reset_gdn_initial_slot` | target selects the state view, then calls `set_i32_scalar(..., 0)` |
| `mtp_set_gdn_initial_slot_from_accepted` | target selects source/destination, then calls `assign_i32_scalar` |

The target owns the meaning of `stats[3]` and the GDN slot. The scalar Ops own only their typed,
fully explicit state transitions.

## 4. Final terminology and repository names

The top-level execution abstraction is renamed from `kernels` to `ops` everywhere it denotes the
semantic layer.

The final call structure is:

```text
include/ninfer/ops/<family>.h      repository-internal Op contracts
                |
                v
src/ops/wrapper/<family>.cpp      validation, workspace scope, finite dispatch
                |
                v
src/ops/launcher/<family>.*       private CUDA launch policy
                |
                v
src/ops/kernel/<family>*.cuh      __global__/__device__ implementation
```

The established implementation subtrees remain:

```text
src/ops/common/                   narrow CUDA math/memory/warp/MMA primitives
src/ops/linear/                   codec, plan, reference, GEMV, GEMM, fused variants
```

This is a responsibility chain, not a requirement for four files per header. Related Ops may share
a launcher family, a wrapper-only composition may reuse other Ops, and header overloads do not
require empty kernel files.

### 4.1 Complete semantic rename

The migration performs all of these changes together:

| Temporary name | Final name |
|---|---|
| `include/ninfer/kernels/`, current `src/kernels/<op>/<op>.h`, or target-local semantic headers | `include/ninfer/ops/` |
| `src/kernels/` plus target mathematical implementations | `src/ops/` |
| `ninfer::kernels` | `ninfer::ops` |
| `ninfer::kernels::detail` | `ninfer::ops::detail` |
| `ninfer_kernels` | `ninfer_ops` |
| `tests/kernels/` | `tests/ops/` |
| `bench/kernels/` | `bench/ops/` |
| `docs/kernel-development.md` | `docs/op-development.md` |

No alias, forwarding header, namespace alias, symlink, compatibility CMake target, or dual source
tree preserves the temporary name.

### 4.2 The word kernel remains valid below the Op boundary

The migration must not blindly replace every occurrence of “kernel”. The term remains correct for:

- `src/ops/kernel/`;
- CUDA `__global__` functions;
- device-kernel implementation comments;
- kernel launch, launch bounds, occupancy, register use, and shared memory;
- kernel-specific benchmarks and profiler output when they measure an implementation rather than
  the Op contract.

The distinction is:

```text
Op      = semantic host-callable computation contract
kernel  = one CUDA implementation or implementation stage of an Op
```

### 4.3 Product API boundary

`include/ninfer/ops/` is a repository-internal execution catalog, not installed product ABI. The
product C++ API remains:

```text
include/ninfer/engine.h
include/ninfer/types.h
```

Engine clients, CLI, server, and product benchmark do not call Op headers directly.

## 5. Op contract standard

Each semantic Op or closely related overload group in `include/ninfer/ops/` has one authoritative
contract comment. The comment describes the common observable semantics of every implementation.

The applicable fields are:

```cpp
/**
 * Op: <semantic operation name>
 *
 * Math / indexing:
 *   <complete formula, algorithm, probability process, or index mapping>
 *
 * Logical shapes:
 *   <symbols, axes, valid ranges, and layout interpretation>
 *
 * Supported domain:
 *   <dtype, numeric format, storage layout, shape, and semantic variants>
 *
 * Numeric:
 *   <decode rule, accumulation, epsilon, casts, rounding, masking, and ties>
 *
 * Effects:
 *   <outputs, state before/after, in-place writes, valid regions, and aliases>
 *
 * Workspace:
 *   <caller-owned transient scratch or none>
 */
```

Workspace sizing functions and other helpers covered by the contract do not repeat the formula.

### 5.1 Observable numeric contract

The header states only numerical behavior shared by all valid implementations. It must not freeze:

- reduction association/order;
- warp/CTA decomposition;
- the number of launches;
- a backend approximation instruction;
- bitwise equality unless explicitly required by the semantic format.

The comment may cite the authoritative registered tensor-format document instead of copying an
entire Q4/Q5/Q6/W8 layout definition.

### 5.2 Stateful Ops

A stateful Op states both transformations:

```text
output    = G(old_state, input)
new_state = F(old_state, input)
```

For GQA this includes:

- query-head to KV-head mapping;
- score scale;
- causal/segment mask and visible range;
- whether newly appended KV is visible to the current call;
- BF16/INT8 KV encode/decode semantics;
- exact cache locations written by the call.

It does not include Program's later cursor advance, prefix restore, or commit policy.

For MTP acceptance the contract includes every scalar/statistics/token-count mutation listed in
Section 3.1.

### 5.3 Fused Ops

“linear + SwiGLU” or “linear + residual” is not a sufficient formula. A fused contract states the
full composition and every observable intermediate cast/rounding boundary. If the fused path has a
different observable rounding seam from separately calling its component Ops, that seam belongs to
the fused Op contract.

### 5.4 Stochastic Ops

Sampling and MTP acceptance specify penalty order, temperature, top-k/top-p/min-p truncation,
normalization, acceptance/resampling, RNG semantic inputs, and all state effects. They do not make a
permanent promise about a particular hash implementation or sampled text sequence unless that is a
deliberate contract.

### 5.5 Implementation comments

Launcher/kernel files reference the semantic contract rather than copying it:

```text
Implements: include/ninfer/ops/linear.h
Match: SM120, BF16 x/out, Q5G64_F16S, RowSplit, N=7168, K=5120, T=1
Algorithm assumptions: warp-per-row, direct row-split decode, alignment/padding requirements
```

Op-private codecs/plans/helpers document their implementation invariant and enclosing Op, not a
synthetic Op formula.

## 6. Responsibilities inside `src/ops`

### 6.1 Wrapper

`src/ops/wrapper/<family>.cpp` owns:

1. semantic dtype/rank/shape/layout/alignment validation;
2. explicit scalar/config/state validation;
3. transient workspace scope creation;
4. finite implementation selection from format/layout/shape/T/device facts;
5. invocation of the selected launcher or composed fallback.

It may dispatch on semantic variant, format, layout, exact shape, token regime, state dtype, and
device capability. It must not dispatch on target key, artifact tensor name, source layer role,
Program phase string, or arbitrary registry key.

The wrapper does not own persistent state, allocate hidden device memory, capture graphs, or choose
model call order.

### 6.2 Launcher

`src/ops/launcher/<family>.h/.cu` owns private launch declarations, grid/block/shared-memory policy,
template instantiation, and launch-error checking. Targets, product code, and semantic headers never
include launcher headers.

### 6.3 Kernel

`src/ops/kernel/<family>*.cuh` owns `__global__` and Op-local reusable `__device__` computation.
Kernels may encode exact shape, format, SM, tiling, and launch assumptions. Those facts must match a
wrapper/launcher predicate and must not be inferred from checkpoint identity.

### 6.4 Common

`src/ops/common/` contains narrow zero-cost arithmetic, memory, warp, and MMA primitives. It is not a
second semantic catalog, and targets never include it directly.

### 6.5 Linear implementation subtree

Linear retains the QUS special subtree because its format and implementation matrix is larger than
the ordinary flat buckets:

```text
src/ops/linear/
├── linear.cpp
├── codec/
├── plan/
├── reference/
├── gemv/
└── gemm/
```

Responsibilities remain:

- `linear.cpp`: `linear`/`linear_pair` validation and dispatch integration;
- `codec/`: registered Q4G64/Q5G64/Q6G64/W8G32 device decode;
- `plan/`: finite format/shape/T classification;
- `reference/`: supported generic/dense CUDA paths;
- `gemv/`: T=1 and small-T implementations;
- `gemm/`: larger-T tensor-core and fused implementations.

No Op base class, runtime registry, universal plan interface, backend plugin system, string dispatch,
or graph IR is introduced. Existing private backend names such as `MlpDown`, `LmHead`, and `AttnIn`
are not renamed as part of this ownership migration. They remain implementation-internal historical
symbols only: selectors still match dtype, format, layout, numerical shape, token regime, and device
facts, and no target passes a model-role label into Op dispatch.

## 7. L0 state and workspace boundary

### 7.1 Tensor and Weight

Ops receive non-owning execution views. They may read only facts required for numerical execution:

- dtype/qtype;
- registered numeric format and quantization group;
- storage layout;
- logical and padded shape;
- data/scale/high-bit planes;
- payload geometry and required alignment.

Ops must not receive artifact object names/handles, converter source fields, model weight roles,
`ModuleKind`, `SourceKind`, recipe information, or provenance.

### 7.2 KV cache

The current parameterized `KVCacheLayout`, `plan_kv_cache`, `KVCache`, and `KVHeadSlot` move from the
exact target to L0 core:

```text
src/core/kv_cache.h
src/core/kv_cache.cpp
namespace ninfer
```

They define a contiguous physical container from explicit layer count, context capacity, KV heads,
head dimension, BF16/INT8 storage, and quantization group.

Core owns layout, binding, views, and physical cursor mechanics. Program owns every cache instance,
which instance is Text/MTP, when cursor methods are invoked, prefix restore, commit/rollback, and
terminal invalidation.

GQA may accept the core cache type or its explicit views. That dependency is valid because the cache
is a target-neutral L0 container, not a model schedule. Leaving it in the target would make a public
GQA Op depend backwards on a target type.

This move does not introduce a universal cache hierarchy. A future materially different cache may
use another core type or remain target-private.

### 7.3 Workspace

Caller-owned `WorkspaceArena&` remains the execution contract. An Op may suballocate transient
scratch inside a scope, but it does not call `cudaMalloc`, retain a workspace pointer, or own the
arena.

Existing sizing queries move with their Ops. This migration adds sizing queries only for the three
known nested fallback gaps:

```cpp
std::size_t gdn_input_proj_workspace_bytes(
    std::int32_t qk_rows, std::int32_t value_rows, std::int32_t tokens);
std::size_t linear_add_workspace_bytes(
    std::int32_t output_rows, std::int32_t input_rows, std::int32_t tokens);
std::size_t linear_swiglu_workspace_bytes(
    std::int32_t gate_up_rows, std::int32_t tokens);
```

They cover the current scratch layouts:

- GDN input projection small-T fallback: BF16 `[4096,T]` plus `[6144,T]`;
- linear-add generic fallback: BF16 `[N,T]`;
- linear-SwiGLU small-T fallback: BF16 `[34816,T]`.

Each affected wrapper and sizing query share one private allocation definition. Program replaces
only the corresponding duplicated/implicit sizing formulas. Existing GQA, GDN gating, gated delta
rule, and Vision sizing authorities remain. This is not a whole-engine workspace redesign.

## 8. Closed Op catalog

The final repository-internal catalog for this migration contains 32 headers:

```text
include/ninfer/ops/
├── add_bias.h
├── argmax.h
├── attn_input_proj.h
├── causal_conv1d_silu.h
├── cast.h
├── embedding.h
├── gated_delta_rule.h
├── gated_rmsnorm.h
├── gdn_gating.h
├── gdn_gating_proj.h
├── gdn_input_proj.h
├── gelu.h
├── gqa_attention.h
├── l2norm.h
├── layer_norm.h
├── linear.h
├── linear_add.h
├── linear_pair.h
├── linear_swiglu.h
├── mtp_pack.h
├── mtp_round.h
├── position.h
├── residual_add.h
├── rmsnorm.h
├── rope.h
├── sampling.h
├── scalar.h
├── scatter.h
├── sigmoid_mul.h
├── silu_mul.h
├── vision_attention.h
└── vision_pos_embed.h
```

Twenty-seven headers preserve the historical QUS semantic catalog under the new `ops` name. Five
headers are intentional NInfer additions:

| Header | Contracts |
|---|---|
| `cast.h` | registered elementwise casts, initially `cast_fp32_to_bf16` |
| `mtp_pack.h` | MTP FC-input packing and attention-field splitting |
| `mtp_round.h` | verification-input construction, acceptance/resampling, shifted ids, hidden gather, token remap |
| `position.h` | `fill_i32_positions` and `offset_i32_positions` |
| `scalar.h` | finite typed scalar set/assignment/increment state transformations |

A header may group closely related Ops; the catalog is not one header per C++ function. New names do
not imply a dynamic registration system.

The scalar catalog is deliberately finite:

```cpp
void set_i32_scalar(Tensor& destination, std::int32_t value, cudaStream_t stream);
void assign_i32_scalar(const Tensor& source, Tensor& destination, cudaStream_t stream);
void increment_i32_scalar(Tensor& scalar, cudaStream_t stream);
void increment_i64_scalar(Tensor& scalar, cudaStream_t stream);
```

It is not a tensor expression framework or general copy API.

The new cast and position contracts have these exact entry points:

```cpp
void cast_fp32_to_bf16(
    const Tensor& source, Tensor& destination, cudaStream_t stream);
void fill_i32_positions(
    Tensor& positions, std::int32_t start, cudaStream_t stream);
void offset_i32_positions(
    const Tensor& source, const Tensor& delta, Tensor& destination, cudaStream_t stream);
```

`scatter.h` additionally owns the existing strided channel-block extraction as a logical slicing
Op rather than a raw transfer:

```cpp
void extract_bf16_columns(
    const Tensor& source, std::int32_t source_column,
    Tensor& destination, cudaStream_t stream);
```

For rank-2 logical matrices, it defines
`destination[row, column] = source[row, source_column + column]` over the full destination extent.
Its implementation may use `cudaMemcpy2DAsync`, but callers do not supply byte pitches or a copy
kind.

## 9. Classification of the current implementation

### 9.1 Current central vertical directories

The temporary vertical directories under `src/kernels/` move directly to the final QUS-style
buckets under `src/ops/`:

| Current directory | Final wrapper | Final launcher | Final kernel |
|---|---|---|---|
| `add_bias/` | `wrapper/add_bias.cpp` | `launcher/add_bias.{h,cu}` | `kernel/add_bias.cuh` |
| `argmax/` | `wrapper/argmax.cpp` | `launcher/argmax.{h,cu}` | `kernel/argmax.cuh` |
| `causal_conv1d/` | `wrapper/causal_conv1d_silu.cpp` | `launcher/causal_conv1d.{h,cu}` | `kernel/causal_conv1d.cuh` |
| `embedding/` | `wrapper/embedding.cpp` | `launcher/embed_gather.{h,cu}` | `kernel/embed_gather.cuh` |
| `gated_delta_rule/` | `wrapper/gated_delta_rule.cpp` | established GDN launcher TUs | established GDN `.cuh` files |
| `gelu/` | `wrapper/gelu.cpp` | `launcher/gelu.{h,cu}` | `kernel/gelu.cuh` |
| `l2norm/` | `wrapper/l2norm.cpp` | `launcher/l2norm.{h,cu}` | `kernel/l2norm.cuh` |
| `layer_norm/` | `wrapper/layer_norm.cpp` | `launcher/layer_norm.{h,cu}` | `kernel/layer_norm.cuh` |
| `residual_add/` | `wrapper/residual_add.cpp` | `launcher/residual_add.{h,cu}` | `kernel/residual_add.cuh` |
| `rmsnorm/` | `wrapper/rmsnorm.cpp` | `launcher/rmsnorm.{h,cu}` | `kernel/rmsnorm.cuh` |
| `scatter/` | `wrapper/scatter.cpp` | `launcher/scatter.{h,cu}` | `kernel/scatter.cuh` |
| `sigmoid_mul/` | `wrapper/sigmoid_mul.cpp` | `launcher/sigmoid_gate_mul.{h,cu}` | `kernel/sigmoid_gate_mul.cuh` |
| `silu_mul/` | `wrapper/silu_mul.cpp` | `launcher/silu_and_mul.{h,cu}` | `kernel/silu_and_mul.cuh` |

`src/kernels/common/` moves to `src/ops/common/`. Exact private filenames may retain their QUS names.

### 9.2 Current target-owned mathematical directories

Every device computation under
`src/targets/qwen3_6_27b_rtx5090/impl/kernels/` is classified as an Op or an Op implementation and
moves to the central owner:

| Current target directory | Final ownership |
|---|---|
| `gdn_gating/` | `gdn_gating.h`/`gdn_gating_proj.h`; wrappers plus central launcher/kernel |
| `gqa_attention/` | `gqa_attention.h`; BF16/INT8 prefill/decode/append wrapper/launcher/kernel |
| `input_projection/` | `attn_input_proj.h`/`gdn_input_proj.h`; wrapper; grouped GEMM under `src/ops/linear/gemm` |
| `linear/` | complete `src/ops/linear/{codec,plan,reference,gemv,gemm}` implementation family |
| `linear_add/` | `linear_add.h` plus wrapper/fused implementations |
| `linear_swiglu/` | `linear_swiglu.h`; wrapper; fused GEMM implementation under linear |
| `mtp/` | `mtp_pack.h` and `mtp_round.h`; wrapper/launcher/kernel, including public `mtp_accept_tokens` |
| `rope/` | `rope.h` plus wrapper/launcher/kernel |
| `sampling/` | `sampling.h` plus wrapper/device primitives/launcher/kernel |
| `vision_attention/` | `vision_attention.h` plus wrapper/launcher/kernel |
| `vision_pos_embed/` | `vision_pos_embed.h` plus wrapper/launcher/kernel |

After the migration, the target has no `impl/kernels/` directory.

### 9.3 CUDA hidden in target schedules

| Current code | Final classification |
|---|---|
| `position_ops.cu`: position fill/iota | `position.h` Op and central implementation |
| `position_ops.cu`: position vector offset | `position.h` Op and central implementation |
| `position_ops.cu`: set/advance scalar | `scalar.h` Ops and central implementation |
| `position_ops.cu`: host pointer -> device I32 upload | local helper in `text_context.cpp`; no Op |
| `vision_convert.cu`: FP32 -> BF16 | generic `cast.h` Op and central implementation |
| `schedule/mtp.cpp`: Tensor -> Tensor I32 scalar assignment | `assign_i32_scalar` Op |
| `text_context.cpp`: strided BF16 channel block extraction | `extract_bf16_columns` in `scatter.h` |
| `text_context.cpp`: `scatter_shifted_visual_embeddings` | target schedule composition over `scatter` Op |
| `vision_control.cpp` | target host/model composition |

The target finishes with no `__global__` implementation. It may retain CUDA runtime transfers in
host schedule code.

`impl/schedule/ops.h` is deleted rather than retained as a second Op facade. The shifted-visual
composition declaration moves to `impl/schedule/visual_scatter.h`. After their device operations
move centrally, `position_ops.cu` and `vision_convert.cu` are deleted; the sole H2D I32 upload helper
is kept local to `text_context.cpp`.

### 9.4 MTP function ledger

| Current symbol | Final symbol/owner |
|---|---|
| `mtp_pack_fc_input` | `ops::mtp_pack_fc_input`, `mtp_pack.h` |
| `mtp_split_attn_in` | `ops::mtp_split_attn_in`, `mtp_pack.h` |
| `mtp_prepare_verify_inputs` | `ops::mtp_prepare_verify_inputs`, `mtp_round.h` |
| `mtp_accept_tokens` | `ops::mtp_accept_tokens`, `mtp_round.h`; fused behavior unchanged |
| `mtp_prepare_shifted_ids` | `ops::mtp_prepare_shifted_ids`, `mtp_round.h` |
| `mtp_gather_hidden_row` | `ops::mtp_gather_hidden_row`, `mtp_round.h` |
| `mtp_remap_draft_token` | `ops::mtp_remap_draft_token`, `mtp_round.h` |
| `mtp_increment_i32` | `ops::increment_i32_scalar`, `scalar.h` |
| `mtp_count_fallback_step` | target view selection + `ops::increment_i64_scalar` |
| `mtp_reset_gdn_initial_slot` | target view selection + `ops::set_i32_scalar` |
| `mtp_set_gdn_initial_slot_from_accepted` | target view selection + `ops::assign_i32_scalar` |

No MTP function is declared through a target-facing private Op header.

### 9.5 Non-Op code that remains target-owned

The target retains:

- `impl/config.h` and exact checkpoint constants;
- `impl/load` tensor role binding;
- `impl/frontend` tokenizer/template/media preparation;
- `impl/program` capacity, graph, state-instance, and request planning;
- GDN state types whose physical representation is still target-specific;
- `impl/schedule` Text/Vision/MTP composition and host transfers;
- `impl/diagnostic` taps and product diagnostics.

## 10. Include and dependency rules

The final link direction is:

```text
ninfer_core
    ^
ninfer_ops
    ^
ninfer_qwen3_6_27b_rtx5090
    ^
ninfer_engine
```

Additional dependencies remain:

```text
ninfer_artifact -> ninfer_core
exact target    -> ninfer_artifact + ninfer_ops + ninfer_text + ninfer_media_decode
```

Rules:

- Op contract headers include only required L0 types and CUDA host types;
- `src/ops/**` cannot include target, Program, schedule, product, or artifact-provenance headers;
- target schedules include semantic Op headers, never launcher, kernel `.cuh`, common, codec, or
  plan headers;
- `ninfer_ops` does not link the exact target;
- core and artifact do not link Ops or targets;
- Op tests/microbenchmarks link `ninfer_ops`, not the target;
- target integration tests link target/Engine only for real target composition;
- all source lists remain explicit; the old recursive QUS GLOB is not restored.

The existing broad internal `src/` include root is not redesigned in this task. The dependency rule
is enforced by actual includes, link ownership, and review, not a new include sandbox or permanent
source-string test.

## 11. CMake atomic ownership switch

`src/CMakeLists.txt` replaces `ninfer_kernels` with one `ninfer_ops` static library. Its explicit
source list contains:

```text
src/ops/wrapper/*.cpp             explicit entries
src/ops/launcher/*.cu             explicit entries
src/ops/linear/**/*.cpp|*.cu      explicit entries
```

Device `.cuh` files are included from their CUDA translation units. The exact target source list
simultaneously removes:

- every file below `impl/kernels/`;
- target `impl/state/kv_cache.cpp` after the core move;
- device implementations moved out of `position_ops.cu` and `vision_convert.cu`.

The explicit `ninfer_core` source list simultaneously adds `core/kv_cache.cpp`; moving the file
without adding its new link owner is not a complete cutover.

Target host schedule, load, frontend, Program, diagnostics, and retained state files remain.

The switch also updates:

- `target_link_libraries` from `ninfer_kernels` to `ninfer_ops`;
- helper/variable/comment names that classify semantic tests or benchmarks;
- all includes from `kernels/...` semantic paths to the final Op paths;
- all `ninfer::kernels` and `kernels::` call sites to `ninfer::ops` and `ops::`.

There is exactly one link owner for every moved symbol. No `ninfer_kernels` alias or compatibility
library remains.

## 12. Tests and benchmarks

### 12.1 Op tests

`tests/kernels/` moves atomically to `tests/ops/`. Existing independent numerical tests remain; no
test is added merely to assert a path, namespace, or source filename.

Current target tests are split by the new definition:

| Current test | Final ownership |
|---|---|
| `test_gqa_attention.cpp` | `tests/ops/test_gqa_attention.cpp`, core KV + `ninfer_ops` |
| `test_linear_kernels.cpp` | `tests/ops/test_linear.cpp`, semantic Op names and `ninfer_ops` |
| `test_mtp_pack.cpp` | `tests/ops/test_mtp_pack.cpp` |
| `test_mtp_round.cpp` | `tests/ops/test_mtp_round.cpp`, including acceptance side effects |
| `test_vision_elementwise.cpp` | add-bias/GELU/scatter cases to Ops; shifted-media composition remains target test |
| `tests/test_kv_cache.cpp` | core test using `ninfer::KVCache`, links `ninfer_core` only |

Add `tests/ops/test_cast.cpp` with one minimal numerical case for `cast_fp32_to_bf16`, and
`tests/ops/test_position.cpp` with cases for both position entry points. These are required because
the current suite does not independently cover the schedule-local device transformations being
promoted into Op contracts. Scalar transitions remain covered through the decoupled MTP round
cases.

The MTP round test removes dependencies on target `TextConfig` and target statistics constants by
using semantic values local to the Op test or deriving them from tensor shapes. This proves the Op
contract rather than the target package.

Existing GDN gating, RoPE, sampling, Vision attention, and Vision position tests already located
under the temporary semantic test directory relink only to `ninfer_ops`/core.

### 12.2 Benchmarks

`bench/kernels/` moves to `bench/ops/`. The GQA implementation-stage profiler and pure
sampling-selection microbenchmark currently under the target benchmark directory also move to
`bench/ops/` and link `ninfer_ops`.

The GQA benchmark intentionally remains an implementation-level benchmark: it may invoke private
central launchers to isolate append/prefill/decode stages. Its dependencies are cleaned at the
ownership boundary:

- KV cache uses the core type;
- the launcher and its implementation constants move together under `src/ops`;
- target config/KV headers are removed.

The public Engine end-to-end benchmark and target MTP-round benchmark remain under the target
benchmark directory because they exercise Program/model composition.

The CMake helper becomes `ninfer_add_op_bench`; there is no target-kernel-benchmark helper for Ops
after centralization.

## 13. Active documentation switch

The implementation updates active authorities in the same final state.

### `AGENTS.md`

Replace the rule that only proven shared operations live centrally. Define Op by Section 1, assign
all Op implementations to `src/ops`, and distinguish internal Op headers from product API. Preserve
the exact-target Program/schedule ownership rules.

### `docs/op-development.md`

Rename `docs/kernel-development.md` and make it the active authority for:

- Op definition and admission test;
- contract/wrapper/launcher/kernel layering;
- formula comments;
- correctness and performance workflow;
- the distinction between semantic Op and CUDA kernel.

### Other active documents

Update only affected ownership/path/terminology sections in:

- `README.md`;
- `docs/README.md`;
- `docs/design.md`;
- `docs/ninfer-engine-architecture.md`;
- `docs/qwen3.6-27b-architecture.md` where it refers to the semantic execution layer;
- `tests/README.md`;
- `bench/README.md`;
- CMake comments and executable help only if they expose the old semantic term.

Historical files under `docs/archive/` remain unchanged. Real CUDA kernel terminology, C++
`operator`, and unrelated prose are not changed by a blind repository-wide replacement. No
forwarding `docs/kernel-development.md` stub remains.

## 14. Atomic implementation sequence

This is one architectural cutover with one accepted final state. The numbered list is an internal
execution order, not a sequence of supported intermediate designs.

1. **Lock the classification ledger.** Use Sections 8 and 9 as the closed catalog and source move
   map. Record current source lists for the move; no repository/artifact hash pinning is required.
2. **Move KV cache to core.** Relocate the existing parameterized implementation and update Program
   type/namespace references without changing cache policy.
3. **Create final Op contracts.** Land all 32 headers with complete formula/effect comments,
   including the full current `mtp_accept_tokens` effects.
4. **Move directly to final source buckets.** Move current central vertical directories and all
   target mathematical implementations into `src/ops/{wrapper,launcher,kernel,common,linear}`. Do
   not create a restored `src/kernels` intermediate state.
5. **Move schedule device transformations.** Centralize position/scalar/cast Ops, preserve the H2D
   host upload, centralize strided BF16 extraction, move the shifted-media declaration to its
   schedule-specific header, and delete the emptied schedule CUDA/facade files.
6. **Switch namespace and includes.** Change the semantic namespace to `ninfer::ops` and all call
   sites/includes in one source state. Preserve genuine kernel implementation names.
7. **Close workspace planning gaps.** Move existing sizing queries and add only the three queries in
   Section 7.3 from shared allocation definitions.
8. **Switch CMake ownership.** Replace `ninfer_kernels` with `ninfer_ops`, add every central source
   once, and remove target mathematical sources in the same configure state.
9. **Move/relink tests and benchmarks.** Apply Section 12, including target-dependency removal from
   MTP/GQA/linear Op tests and GQA benchmarks.
10. **Update active authorities.** Rename the development guide and update the documents in Section
    13 without rewriting archives.
11. **Delete the temporary architecture.** Remove old paths, target `impl/kernels`, obsolete
    declarations, CMake helpers, and all compatibility aliases.
12. **Run Section 15 acceptance.** Fix only actual classification, migration, build, numerical,
    product, or performance regressions.

No intermediate state with both `ninfer_kernels` and `ninfer_ops`, both namespaces, duplicate
implementations, or forwarding headers is accepted as complete.

## 15. Acceptance

Verification is limited to the observable risks changed by this migration.

### 15.1 Structural acceptance

- perform one fresh configure/build;
- build `ninfer_core`, `ninfer_ops`, the exact target, Engine, CLI, server, benchmarks, and affected
  tools;
- confirm every target device computation enters through an Op contract;
- confirm target schedule includes no launcher/kernel/common/linear-detail header;
- confirm every moved symbol has one link owner;
- confirm target has no `impl/kernels/` directory and no `__global__` implementation;
- confirm current source/build/include/test/benchmark files and active architecture guides have no
  old semantic paths/names:
  `include/ninfer/kernels`, `src/kernels`, `tests/kernels`, `bench/kernels`, `ninfer_kernels`,
  `ninfer::kernels`, or semantic `kernels::` call sites.

The last item is a one-time migration review, not a permanent source-scanning test. The active tree
may and should still contain the word `kernel` for real CUDA implementation concepts. Historical
archives and this plan's explicit old-to-new/deletion ledger are excluded from the stale-reference
check.

### 15.2 Numerical Op acceptance

Run the existing meaningful tests for moved families:

- registered linear formats, exact shapes, and fused variants;
- normalization, activation, and elementwise Ops;
- causal convolution and gated delta rule state transitions;
- GDN gating and projections;
- GQA BF16/INT8 KV paths;
- RoPE/MRoPE and Vision position variants;
- sampling and all MTP round effects;
- Vision segmented attention and interpolated position embedding;
- new finite scalar and cast contracts using the existing affected tests.

No new oracle is required for a mechanical move when an existing independent oracle already covers
the same contract. Add or change a test only when decoupling exposes an actual missing semantic
case.

### 15.3 Product acceptance

Run representative real `.ninfer` requests for:

- ordinary BF16 Text;
- INT8-KV Text;
- Text MTP with proposals/acceptance statistics;
- one representative Vision request.

Additional video, mixed-media, or Vision+MTP runs are required only if their independent schedule
branches changed. Sampled output text need not match exactly; the run must satisfy the supported
behavioral contract.

### 15.4 Performance acceptance

The migration preserves algorithms and launch policy. Run focused before/after measurements only
for wrappers or paths whose dispatch/workspace boundary materially changed, plus a quick
end-to-end decode/prefill sanity check. There is no fixed percentage gate. Use NSYS/NCU only when a
specific observed regression needs attribution.

### 15.5 Documentation acceptance

- `docs/README.md` points to `docs/op-development.md` as the single active authority;
- active ownership/path/namespace examples match the final tree;
- archived QUS/old-engine documents remain historical;
- `git diff --check` passes.

## 16. Deletion ledger

Completion deletes:

- `src/kernels/`;
- `src/targets/qwen3_6_27b_rtx5090/impl/kernels/`;
- temporary target declarations of central Ops;
- target CMake entries for moved mathematical sources;
- `ninfer_kernels` and every alias/reference to it;
- old `ninfer::kernels` semantic namespace/call sites;
- `tests/kernels/` and `bench/kernels/`;
- target-only Op-test/Op-benchmark CMake helpers;
- target `impl/schedule/ops.h`, `position_ops.cu`, and `vision_convert.cu` after their final owners
  are established;
- `docs/kernel-development.md` after its direct rename;
- forwarding headers, namespace aliases, CMake aliases, symlinks, or compatibility includes for
  any deleted path;
- obsolete model-labelled scalar bookkeeping wrappers.

The target retains its load, frontend, Program, schedule, retained state, and diagnostic sources.

## 17. Explicit non-goals

This migration does not include:

- a generic Op base class, registry, plugin/backend system, graph IR, or string dispatch;
- a new model graph or family abstraction;
- CUDA algorithm rewrites or tuning unrelated to a migration regression;
- quantization/format/layout changes;
- persistent weight repacking;
- private backend role-name cleanup;
- splitting or redesigning MTP acceptance/statistics behavior;
- a universal KV-cache abstraction;
- converter, reference-model, parity, or `.ninfer` container changes;
- Program/runtime/concurrency redesign;
- include sandboxing or repository-wide visibility refactoring;
- compatibility layers;
- fixed hashes, clean-worktree requirements, byte-identical reproduction, or exact sampled-text
  comparison;
- tests for directory names, namespace strings, or mechanical coverage.

## 18. Completion criteria

The migration is complete only when one final source state satisfies all of the following:

1. Op is defined and applied consistently by Sections 1-3.
2. The 32-header catalog is the only target-callable mathematical contract layer.
3. The final semantic namespace and library are `ninfer::ops` and `ninfer_ops`.
4. All Op implementations, including fixed-shape/fused/SM120 paths, live under `src/ops`.
5. `kernel` refers only to implementation concepts below the Op boundary.
6. Target owns schedules/state policy but no mathematical CUDA implementation.
7. No target-facing private Op/helper contract exists.
8. MTP acceptance is a complete public internal Op with unchanged fused behavior and documented
   side effects.
9. KV cache is a target-neutral core physical container while Program owns each instance/policy.
10. Affected workspace queries and execution share their allocation definitions.
11. Op tests/benchmarks link central `ninfer_ops`; target/product tests use their proper owners.
12. Active documentation expresses the same definition, terminology, and source tree.
13. No old semantic path, alias, forwarding layer, or duplicate implementation remains.
14. The scoped build, numerical, product, performance, and documentation acceptance passes.

## 19. Review checklist

Review the actual final diff with these questions:

- Is each target-callable computation classified by the Op definition rather than current caller?
- Can every Op formula and state effect be understood without target/schedule context?
- Does any Op read target config, Program, artifact role, or provenance?
- Does any non-Op helper accidentally expose a target-callable device computation?
- Are MTP acceptance statistics/token-count side effects fully documented?
- Does Program still own call order, instance lifetime, frontier, commit, and rollback?
- Are exact-shape and SM120 implementations still directly dispatched without framework overhead?
- Do formula comments avoid freezing backend reduction/tile/kernel details?
- Are workspace changes limited to the known planning gaps?
- Are `ops` and `kernel` used consistently at their two different abstraction levels?
- Did CMake preserve one implementation/link owner and explicit source lists?
- Were only necessary tests and product/performance checks used?

## 20. Final repository boundary

```text
include/ninfer/
├── engine.h                         installed product API
├── types.h                          installed product values
└── ops/                             repository-internal semantic Op catalog
    └── <family>.h

src/
├── core/                            L0 device/tensor/layout/arena/graph/KV/transfer
├── artifact/                        generic .ninfer mechanisms
├── ops/
│   ├── wrapper/                     Op validation, workspace, finite dispatch
│   ├── launcher/                    private CUDA launch policy
│   ├── kernel/                      __global__/__device__ implementations
│   ├── common/                      narrow CUDA implementation primitives
│   └── linear/                      codec/plan/reference/GEMV/GEMM family
├── targets/qwen3_6_27b_rtx5090/
│   ├── export/                      scoped package facade
│   └── impl/
│       ├── config.h
│       ├── load/
│       ├── frontend/
│       ├── program/
│       ├── state/
│       ├── schedule/
│       └── diagnostic/
└── runtime/                         common Engine/controller policy

tests/
├── ops/                             Op numerical/behavioral contracts
└── targets/                         schedule/product integration

bench/
├── ops/                             Op and implementation microbenchmarks
└── targets/                         Program/end-to-end benchmarks
```

The resulting architecture preserves the useful QUS implementation discipline while naming the
layers accurately:

- Ops own mathematical and explicit local-state semantics;
- kernels implement Ops;
- targets compose Ops into exact checkpoint programs.

## 21. Review record

This plan was reviewed from three independent angles before being presented for approval:

1. **Op boundary and semantic closure.** Checked the definition against stateful GQA, fused linear,
   stochastic sampling, MTP acceptance, scalar state updates, raw transfers, KV cursor operations,
   and target schedule composition. No blocker, high, or medium issue remained.
2. **QUS discipline and migration scope.** Compared the final wrapper/launcher/kernel/common/linear
   organization with the former QUS structure and checked that the plan introduces no private Op
   layer, runtime registry, backend framework, compatibility tree, or unrelated redesign. No
   blocker, high, or medium issue remained.
3. **Current-source inventory.** Cross-checked the central and target mathematical directories,
   schedule-local CUDA transformations, public symbols, MTP helpers, tests, benchmarks, workspace
   gaps, KV ownership, and CMake link owners against Sections 8-13.

Review corrections already incorporated include:

- making device facts wrapper inputs rather than caller-selected implementations;
- including current vertical semantic headers in the old-to-new path map;
- distinguishing typed scalar assignment from raw byte transfer and forbidding aliased assignment;
- stating explicitly that KV cursor methods are core mechanisms, not scalar Ops;
- preventing historical private linear backend names from becoming model-role dispatch inputs.
- classifying strided BF16 block extraction as a typed slicing Op, and fixing the final APIs,
  schedule-file deletion ledger, KV link owner, new Op tests, and GQA implementation benchmark
  treatment found by the implementation-level source audit.

The review establishes that this is an executable migration plan for the current tree. It does not
authorize implementation; user approval remains the gate.
