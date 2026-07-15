# GQA Contract and Execution Refactor

Status: implemented and archived on 2026-07-15.

This plan governed the completed GQA refactor across `core`, `ops`, the registered 27B target,
CUDA Graph integration, and the future 35B-A3B target. It replaced an A3-local fix with one
coherent design for all three causal grouped-query attention operations. Sections that describe
the "current" cursor-based implementation record the pre-refactor state.

The stable ownership, execution-envelope, cache-view, Program-frontier, and graph-tier decisions
now live in [`op-development.md`](../../op-development.md),
[`ninfer-engine-architecture.md`](../../ninfer-engine-architecture.md), and the repository-internal
GQA contract. Retained qualification evidence is in the
[`35B A3 roofline report`](../optimization-era/bench/qwen3.6-35b-gqa-a3-roofline.md) and the
[`27B graph-frontier report`](../optimization-era/bench/qwen3.6-27b-gqa-graph-frontier-qualification.md).

## 1. Scope and outcome

The refactor covers these semantic operations:

| ID | Operation | Cache effect | Attention result |
|---|---|---|---|
| A1 | append-and-attend GQA | writes the supplied K/V columns | yes |
| A2 | KV append | writes the supplied K/V columns | no |
| A3 | cached-only GQA | read-only | yes |

It also covers the mechanisms needed to execute those operations correctly and efficiently:

- the physical per-layer KV view exposed by `core`;
- ownership of Text and MTP cache frontiers;
- eager small-T launch selection;
- graph-stable launch selection and replay;
- BF16 and INT8-G64 cache implementations;
- split-KV producer/reducer agreement;
- prompt, ordinary decode, target verification, MTP alignment, MTP proposal building, prefix reuse,
  diagnostics, and the future 35B target.

The intended result is:

```text
physical cache storage
    != Program logical/transaction frontier
    != actual device attention position
    != host execution envelope captured by a CUDA Graph
```

Each concept has one owner and one purpose. No GQA launcher reads a target lifecycle cursor, and no
target chooses a CUDA grid or kernel symbol.

This plan does not introduce a generic attention framework, graph IR, backend registry, family
runtime layer, or compatibility API. It directly supports the registered Qwen3.6 geometries and the
RTX 5090 implementations.

## 2. Why the current boundary is wrong

### 2.1 Hidden host state changes launch behavior

The mathematical GQA inputs include device `positions`. For a small-T call, the last position
defines the largest causal domain:

```text
actual_visible_keys = positions[T - 1] + 1
```

The current launcher does not use that explicit input to choose its host launch. It computes:

```text
host_window = kv.pos + T
```

inside `src/ops/launcher/gqa_attention_decode.cu`. Consequently, two calls with identical Q/K/V,
device positions, cache contents, output, and workspace can launch differently because of a host
field that is neither a mathematical input nor an explicit execution resource.

This violates semantic closure. `KVCache::pos` currently has three incompatible interpretations:

1. the target's reusable or materialized prefix;
2. the next host append slot used by legacy core helpers;
3. the attention window used to choose split count and INT8 implementation.

Only the first two are lifecycle concepts, and neither belongs in an Op launcher. The third must be
derived from an explicit execution envelope, while the exact causal domain remains device data.

### 2.2 CUDA Graph freezes the wrong launch

`Program::prepare_graphs()` resets the cache cursors to zero before capturing the ordinary,
ordinary-aligned, and MTP graphs. Stream capture records the kernel function, grid, block, dynamic
shared-memory size, and by-value kernel arguments. Replay calls `cudaGraphLaunch`; it does not rerun
the GQA wrapper or launcher.

For the ordinary T=1 graph, capture therefore records the short-context minimum:

| Geometry | Captured splits | Captured producer CTAs | Long-context split capacity | Long-context producer CTAs |
|---|---:|---:|---:|---:|
| 27B: 24Q/4KV, group 6 | 4 | 16 | 85 | 340 |
| 35B: 16Q/2KV, group 8 | 8 | 16 | 170 | 340 |

The current device code clamps its active split count to `gridDim.y`, then repartitions the full
causal domain across the available CTAs. This normally preserves the mathematical result, but a
long-context replay remains stuck at the tiny captured grid and makes each producer loop over far
too many keys. The eager A1 kernel can be roofline-qualified while the default graph-integrated
engine route still fails to use that implementation at its qualified launch geometry.

INT8 is more strongly coupled to capture-time state. The host window also selects:

- warps per CTA;
- minimum blocks per SM;
- 32- or 64-key block size;
- static or dynamic shared-memory arena;
- the concrete templated kernel function.

A short-context capture therefore fixes not only too few splits but also a short-context INT8
kernel variant for every later replay.

### 2.3 MTP demonstrates why a cursor cannot be the execution window

During prompt preparation, MTP first performs A2 over a chunk, then A3 for the last query, then may
run several A1 autoregressive proposal steps. Those physical writes occur before Program publishes
the persistent MTP prefix. The same pattern appears inside a speculative graph, where positions can
depend on device acceptance results.

Temporarily advancing `kv.pos`, restoring it later, or treating it as “furthest byte written” would
corrupt transaction meaning and prefix-reuse policy. Physical rows may be useful within the current
stream-ordered schedule without being published as a reusable Program prefix. The Op must use
explicit positions plus an explicit execution envelope instead.

### 2.4 Current support status is not erased

The existing A1 eager kernels and A2 append kernels retain their standalone numerical and
performance evidence. This refactor does not reclassify those kernel results. It fixes the semantic
boundary and makes the graph-integrated product route select qualified launches. A3 remains
unsupported for small T until its cached-only split-KV path is implemented and qualified.

## 3. Required invariants

The implementation must preserve all of these invariants.

1. **Device positions define mathematics.** For query column `t`, only cache slots
   `0..positions[t]` participate in attention.
2. **Execution envelopes do not define mathematics.** Changing to a larger valid envelope cannot
   change the logical output or cache effect.
3. **Physical cache has no lifecycle cursor.** It stores planes, shapes, formats, and capacity only.
4. **Program owns persistent frontiers.** Prefix validity, transaction publication, rewind, reset,
   commit, and invalidation are target Program policy.
5. **Ops receive one layer.** GQA does not receive a whole multi-layer cache plus a layer index.
6. **No hidden host window.** Wrapper and launcher inspect only explicit tensors, the per-layer
   physical view, explicit execution resources, and device facts.
7. **Graph replay is launch-valid across its declared interval.** Capture-time cursor values do not
   select replay geometry.
8. **Producer and reducer use one split policy.** They cannot infer different active split counts.
9. **Workspace is caller-owned and graph-stable.** No hidden allocation or recapture is introduced.
10. **Target code never names a private kernel path.** BF16/INT8, geometry, T, envelope, and device
    facts are sufficient for central Op dispatch.

## 4. GQA mathematical contracts

### 4.1 Common symbols and cache representation

For all registered paths:

```text
D    = 256                         head dimension
Hq   = 24 or 16                    query heads
Hkv  = 4 or 2                      KV heads
G    = Hq / Hkv                    6 or 8
T    = number of query columns
p[t] = positions[t]                absolute cache position
kvh  = floor(h / G)                KV head mapped from query head h
```

`positions` is contiguous I32 `[T]` and contains sequential absolute cache slots:

```text
p[t] = p[0] + t
0 <= p[0]
p[T - 1] < cache.max_context
```

Every cache row read before the call, and every row made visible by an earlier operation in the same
stream-ordered schedule, must be populated. Cache positions are unrelated to scalar or three-axis
RoPE coordinates.

One logical layer has K/V shape `[D, C, Hkv]`. Its physical view is either:

- BF16 K and V `[D, C_pad, Hkv]`; or
- INT8-G64 K/V code planes `[D, C_pad, Hkv]` with FP16 scale planes
  `[D/64, C_pad, Hkv]`.

INT8-G64 is a runtime-state codec, not a persistent format from
[`ninfer-tensor-formats.md`](../../ninfer-tensor-formats.md). Its observable cache encoding is part of
the GQA contract. For each 64-element BF16 source group `x` in one token and KV head:

```text
a          = max_i abs(FP32(x[i]))
scale_bits = FP16_RNE(a / 127)
s          = FP32(scale_bits)

if s == 0:
    code[i] = 0
else:
    inv     = FP32(1 / s)
    code[i] = I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))

decode(code[i], scale_bits) = FP32(code[i]) * s
```

The scale is stored once per group. A1 and A2 must produce the same code and FP16 scale bits for
the same BF16 input. The ideal logical cache value used by the mathematical attention oracle is
`decode(code, scale_bits)`, including for rows appended by the current A1 call.

Optimized INT8 attention may quantize Q internally, use integer MMA, or round decoded K/V while
staging into a narrower compute type. Those are implementation approximations, not extra semantic
inputs and not part of the cache representation. BF16 Q and the FP32-decoded cache define one
common higher-precision attention oracle for prompt, small-T, A1, and A3; every implementation must
meet one contract tolerance against that oracle. The oracle promotes BF16 Q and FP32-decoded cache
values to FP64 and performs dot products, softmax, and value reduction in FP64. Kernel BF16 output
is promoted to FP64 for comparison. NaN is always a failure. Otherwise the result passes when either
every element satisfies `abs(error) <= 2e-3 + 1.6e-2*abs(reference)`, or all three tail conditions
hold: violating fraction `<=2e-3`, worst violation divided by its elementwise bound `<=5`, and
relative L2 residual `<=8e-3`. This criterion applies equally to BF16 and INT8 cache paths; there is
no path-specific relaxation.

A prompt path and a small-T path may have different internal rounding trees, but neither path's
current tree becomes the definition of the Op. A higher-precision implementation is valid when it
meets the same oracle; Q quantization and staging casts are tolerated only as performance-motivated
approximations within the criterion above.

For every query head `h`, column `t`, and visible key `s`:

```text
score[h,t,s] = scale * dot(Q[:,h,t], Kcache[:,s,kvh])
P[h,t,:]     = softmax(score[h,t,0..p[t]])
O[:,h,t]     = BF16(sum_s P[h,t,s] * Vcache[:,s,kvh])
scale        = 1 / sqrt(256)
```

### 4.2 A1: append-and-attend

Inputs are BF16 Q `[D,Hq,T]`, BF16 new K/V `[D,Hkv,T]`, I32 positions `[T]`, scale,
a mutable physical layer view, an execution envelope, and caller workspace. Output is BF16
`[D,Hq,T]`.

For each `t`, A1 first defines the logical cache row at `p[t]` from new K/V using the selected cache
format, then computes the causal formula above. Each column sees its own newly defined row and all
earlier rows. The implementation may fuse writes and reads and may source current rows directly from
new K/V only when the result is equivalent to the registered cache representation.

Effects:

- overwrites every addressed K/V code and, for INT8, every addressed scale;
- writes all output elements;
- does not change a Program frontier or any host cache state.

Q, new K, new V, positions, output, and every cache plane are pairwise non-overlapping. Transient
suballocations obtained from caller workspace must not overlap any live operand, output, or cache
plane; the arena backing may also own the live operand allocations as long as its scoped allocation
discipline preserves that property.

### 4.3 A2: KV append

Inputs are BF16 K/V `[D,Hkv,T]`, I32 positions `[T]`, and a mutable physical layer view. A2 applies
the same registered cache encoding used by A1 and overwrites every addressed code and scale. It
does not read unrelated cache rows, compute attention, use an execution envelope, or update a
frontier.

K, V, positions, and every cache plane are pairwise non-overlapping.

### 4.4 A3: cached-only attention

Inputs are BF16 Q `[D,Hq,T]`, I32 positions `[T]`, scale, a read-only physical layer view, an
execution envelope, and caller workspace. Output is BF16 `[D,Hq,T]` using the common formula.

A3 writes all output elements and does not mutate any cache plane or Program frontier. Small-T A3
must use a true cached-only split-KV specialization: it must not accept dummy new K/V tensors,
rewrite existing rows, or compose prompt-scale small kernels as a performance fallback.

Q, positions, output, and every cache plane are pairwise non-overlapping. Its workspace
suballocations follow the same non-overlap rule as A1.

### 4.5 Domain and strict support boundary

The semantic interfaces accept positive T representable by I32 when the sequential positions and
all tensor dimensions fit the physical cache. The wrapper dispatches `T=1..6` to small-T and larger
T to prompt attention; this refactor does not silently narrow the registered 27B product's existing
positive-multiple-of-128 `prefill_chunk` option.

That broad semantic domain is not a performance-support claim. The strict target domains carried
through this refactor are:

| Operation | 27B target requirement | 35B target requirement |
|---|---|---|
| A1 | small-T `1..6`; prompt at every runtime-reachable `T=7..prefill_chunk`, including a shortened prompt tail or checkpoint-boundary chunk | small-T `1..6`; every runtime-reachable prompt `T=7..1024`, including shortened tails |
| A2 | every runtime-reachable MTP prompt `T=1..prefill_chunk`, including shortened tails and checkpoint-boundary chunks | every runtime-reachable `T=1..1024`, including shortened tails |
| A3 | `T=1` is the current target performance requirement; `T=2..6` remains a low-cost small-T implementation domain without a current caller | `T=1` is the target performance requirement; `T=2..6` remains an implementation domain without a current caller |

The existing prompt-scale A3 implementation may remain in the semantic dispatch for larger T, but
it is not classified as Supported without a target requirement and retained high-performance
qualification. Before completion, either the 27B prompt/A2 matrix must be qualified over its
advertised configurable regime or that product option must be narrowed through an explicit product
decision outside this plan. Narrowing the configured maximum does not remove arbitrary shortened
tail and checkpoint-boundary shapes below that maximum. Correctness covers the complete reachable T
domain. Performance qualification need not enumerate every integer T; it covers kernel/dispatch
transitions, shortened-tail and boundary-chunk shapes, and representative base/context values. A
generic functional route alone does not satisfy the strict support definition.

## 5. Core physical cache interface

### 5.1 Per-layer view

`core` will expose a target-neutral non-owning layer view with only physical execution facts:

```cpp
struct KVCacheLayerView {
    Tensor k;
    Tensor v;
    Tensor k_scale;
    Tensor v_scale;
    std::uint32_t max_context;
    std::uint32_t padded_context;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    DType dtype;
    std::int32_t quant_group;
};
```

BF16 views have empty scale tensors and `quant_group == 0`. INT8 views have FP16 scale tensors and
the registered group size. `KVCache` remains the multi-layer physical owner/binder and returns a
checked `KVCacheLayerView` for a selected layer.

The view contains no model identity, layer role, target phase, graph key, prefix identity, or
frontier.

### 5.2 Removed lifecycle surface

The following members are removed from `KVCache`:

- `pos`;
- `advance()`;
- `rewind()`;
- `reset()`;
- `append_slot()`;
- cursor-gated `slot()` and `KVHeadSlot` if no production caller remains.

They currently serve only target lifecycle policy or legacy core tests. Keeping them would preserve
two sources of truth after Program gains explicit frontiers. Reset and rewind become integer state
changes in Program; physical cache bytes are not cleared.

No compatibility aliases are retained. Tests of the physical container inspect layer views and
layout binding rather than recreating a deleted cursor abstraction.

## 6. Explicit execution envelope

### 6.1 Type and meaning

The GQA contract adds an Op-specific host execution resource:

```cpp
struct GqaExecutionEnvelope {
    std::uint32_t min_visible_keys;
    std::uint32_t max_visible_keys;
};
```

For a valid call:

```text
1 <= min_visible_keys <= actual_visible_keys <= max_visible_keys <= cache.max_context
actual_visible_keys = positions[T - 1] + 1
```

The interval is a promise by the caller about a device-derived numerical shape. It is not a causal
mask, cache-validity frontier, split count, tile size, graph ID, or target phase. The lower bound is
necessary because a graph tier should select an INT8 implementation qualified for its actual
interval instead of requiring one max-bound implementation to be optimal for every shorter context.

Usage is direct:

- eager calls with a host-known position pass the exact interval `[W,W]`;
- eager calls with device-dependent positions pass the tightest provable interval;
- a graph-captured call passes the interval covered by that target-private graph variant.

The wrapper validates the host interval and physical capacity. It does not copy device positions to
the host. The caller owns the position-within-envelope precondition, which is checked by meaningful
Op/graph integration tests rather than a synchronization or defensive device status path.

### 6.2 Contract API after the refactor

The intended contract shape is:

```cpp
std::size_t gqa_attention_workspace_bytes(std::int32_t q_heads,
                                          std::int32_t tokens);

void gqa_attention(const Tensor& q,
                   const Tensor& k,
                   const Tensor& v,
                   const Tensor& positions,
                   float scale,
                   KVCacheLayerView cache,
                   GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace,
                   Tensor& out,
                   cudaStream_t stream);

void gqa_kv_append(const Tensor& k,
                   const Tensor& v,
                   const Tensor& positions,
                   KVCacheLayerView cache,
                   cudaStream_t stream);

void gqa_attention_cached(const Tensor& q,
                          const Tensor& positions,
                          float scale,
                          const KVCacheLayerView& cache,
                          GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace,
                          Tensor& out,
                          cudaStream_t stream);
```

The exact declaration may use references where appropriate, but these semantic facts are fixed:

- no whole-cache-plus-layer-index argument;
- no cursor argument;
- A1 and A3 both receive the execution envelope and workspace;
- A2 receives neither;
- no target or graph type crosses the Op boundary.

### 6.3 Workspace rule

The small-T workspace retains capacity for the registered geometry's maximum split count. It is not
resized per replay or per graph tier. The producer grid may use fewer splits, and the reducer reads
only active partials. Prompt-scale routes continue to require no split-KV partial workspace.

A3 reuses the same partial accumulator/statistics layout as A1. The existing MTP workspace already
reserves the small-T GQA tail region; layout planning must use the common sizing query rather than a
second cached-only formula.

## 7. Split-KV launch and device policy

### 7.1 One actual-window policy

For geometry, cache format, and compile-time T, define one private split policy:

```text
desired_splits(W) = measured split policy for actual visible-key count W
```

The policy is shared by host capacity calculation, the producer, and the reducer. In practice it
should be one compile-time-specialized host/device helper or one set of constants used by both host
and device helpers. The reducer must be specialized by cache format when BF16 and INT8 split rules
differ; it cannot use a format-blind approximation.

For envelope `[L,U]`, the host computes:

```text
launch_capacity = max(desired_splits(W) for L <= W <= U)
```

The private implementation computes this maximum analytically over its finite policy breakpoints;
it does not loop over every context position. Taking the interval maximum is required because the
measured INT8 policy is not monotonic at every specialization boundary.

At replay, producer and reducer read device positions and compute:

```text
W             = positions[T - 1] + 1
active_splits = desired_splits(W)
```

`launch_capacity` must be at least `active_splits` for every valid W in the envelope. Producer CTAs
with `split >= active_splits` return without writing. Active CTAs partition the complete `[0,W)`
domain using `active_splits`, not the captured grid size. The reducer combines exactly those same
active partials. No inactive partial is initialized merely to make an oversized grid safe.

### 7.2 Graph-fixed INT8 implementation

CUDA Graph also fixes the concrete INT8 kernel function, block size, and dynamic shared memory. For
each `(geometry, T, envelope, cache format, device)` match, the launcher chooses one finite kernel
variant whose correctness and performance are valid over the whole envelope. The target provides
only the envelope; it does not choose warps, key blocks, or kernel names.

If one interval is too wide for one high-performance variant, the target's graph frontier tiers are
split. This is a performance-driven finite target decision, not an Op registry. Tier boundaries are
accepted only after graph-replay benchmarks show the selected kernels remain qualified at the
interval boundaries and representative interior points.

### 7.3 A1 and A3 implementation sharing

Small-T A1 and A3 should share indexing, Q staging, cache decode, QK/PV, online-softmax, partial
layout, active-split policy, and reducer code. Cache effect is a compile-time implementation choice:

- A1 accepts new K/V, encodes/writes the addressed rows, and makes those rows visible to the current
  causal computation;
- A3 has no new K/V operands, performs no quantization/write/synchronization for append, and reads
  every visible row from cache.

This can be expressed by separate kernels using shared primitives or a compile-time cache-effect
template. A runtime branch, dummy input, or small-kernel composition is not acceptable for the
performance-critical path.

Prompt-scale A1 remains append plus prompt attention. Prompt-scale A3 remains read-only prompt
attention. A2 keeps its standalone BF16 vector-copy and INT8-G64 quantizing append kernels.

## 8. Program frontier and transaction model

### 8.1 Persistent target state

The lifecycle and graph sections use these target-level symbols:

```text
C = logical cache capacity
P = token count of the new prompt
B = reused prompt-prefix length, 0 <= B <= P
K = configured MTP draft window (`mtp_k`)
```

The registered target replaces cache-owned cursors with explicit Program members. The final names
may follow target conventions, but their meanings are:

```text
E                   resolved execution frontier from the generated-token contract
S                   resolved sampled frontier, with S = E + 1 in Active/Resident state
text_kv_valid        published Text KV prefix [0, text_kv_valid)
mtp_kv_valid         published, aligned MTP KV prefix [0, mtp_kv_valid)
proposal_ready       validity of the complete MTP proposal state anchored at mtp_kv_valid
```

“Published” here is target-internal: the prefix belongs to the current Program identity and may be
used by a later schedule or prefix-reuse decision. It is distinct from the public generated-token
commit decision.

Physical rows beyond a published prefix may contain stale values or schedule-local provisional
values. They become readable only through explicit positions when the current stream-ordered
schedule has established them. They are never made reusable merely because bytes were written.

### 8.2 Required state relationships

| Program condition | Required relationship |
|---|---|
| Empty/reset | `text_kv_valid=0`, `mtp_kv_valid=0`, no physical memset required |
| Active or Resident | `text_kv_valid=E` |
| `proposal_ready` in Active or Resident | MTP storage exists and `mtp_kv_valid=E` |
| Begin pending after a P-token prompt | `text_kv_valid=P`; `mtp_kv_valid=P` only if the aligned proposal state was published |
| Ordinary round pending from base E | `text_kv_valid=E+1`; resolved `E` remains the old value until round resolution |
| Aligned ordinary round pending | both published prefixes are `E+1`, but `proposal_ready=false` because no complete draft proposal was built |
| MTP round pending with r licensed tokens | both published candidate prefixes are `E+r`; resolved `E` remains the old value |
| MTP pending `(base E, produced r)`, terminal accepts `m<r` | transition directly to Invalid; neither cache prefix is reusable and no rewind or publication of `E+m` is attempted |
| Text prefix rewind to nonzero B | `text_kv_valid:=B`; physical rows at `B..` are left untouched |
| MTP rebuild for nonzero B | require the old aligned prefix through `B-1`, set `mtp_kv_valid:=B-1`, bridge row `B-1`, then publish `mtp_kv_valid:=B` |
| Full reset / B=0 | both published prefixes become zero; no MTP bridge row exists |
| Invalid Program | no published prefix is usable regardless of physical bytes |

MTP proposal construction can write farther than `mtp_kv_valid`. Those rows are part of the current
proposal computation, not a persistent aligned prefix. On the next schedule they are either used
under the same explicit device dependency or overwritten. This is why renaming the current
`mtp_materialized` concept to a “valid/aligned prefix” is preferable.

`proposal_ready` is always anchored at `mtp_kv_valid`, not directly at the resolved E. In Active or
Resident state a ready proposal therefore has `mtp_kv_valid=E`. In Begin Pending it may be anchored
at prompt frontier P, and in MTP Pending it is anchored at candidate frontier `base_E+r`, while
resolved E still names the old frontier. Ordinary Pending, including the aligned-cache form, has no
complete next proposal and must clear `proposal_ready`.

### 8.3 Schedule interfaces

Program is the sole persistent owner of these frontier fields. Target schedule functions receive
the starting prefix or absolute positions they need and return the resulting publishable prefix and
proposal state. `TextContext` must not recover a starting position from the physical cache.

Concrete call-site rules are:

- Text prefill derives chunk positions from the Program-supplied base and passes exact envelope
  `[base+t0+len, base+t0+len]` to each attention call.
- MTP prompt A2 uses explicit prompt positions; its final A3 uses the exact final prompt window.
- Prompt proposal A1 steps use their known exact host position where possible and a conservative
  explicit interval where the position is device-derived.
- Prefix reuse with a nonempty suffix (`0<B<P`) first reduces the MTP aligned prefix to `B-1`.
  The bridge consumes the new request's `prompt[B]` and target hidden state at `B-1`, writes MTP
  cache row `B-1`, uses envelope `[B,B]`, and publishes the aligned MTP prefix as B only after the
  bridge succeeds. It does not build the final proposal; suffix prefill does that afterward.
- Zero-suffix reuse (`0<B=P`) has no `prompt[B]`. It first samples the frontier token from target
  hidden state `h[P-1]`, then bridges that sampled token with the same hidden state at cache row
  `P-1`, envelope `[P,P]`, and builds the proposal immediately. Only a successful bridge/proposal
  publishes `mtp_kv_valid=P` and `proposal_ready=true`.
- Ordinary eager decode uses `[E+1,E+1]`.
- Target verify and MTP proposal schedules use the tightest host-provable interval derived from E,
  K, and the selected graph frontier tier.
- Diagnostic execution uses the same explicit rules; disabling graphs does not reintroduce a
  cursor-derived launcher path.

On schedule failure the existing target invalidation policy remains authoritative. This refactor
does not add rollback copying or cache clearing.

## 9. CUDA Graph policy

### 9.1 Target-private frontier tiers

Ordinary and MTP rounds have different legal frontier domains:

```text
ordinary: 0 <= E < C
MTP:      0 <= E <= C - 2K
```

They therefore use independent, finite, target-private coverage tables. A conceptual 27B
representation is:

```cpp
struct OrdinaryGraphVariant {
    std::uint32_t min_execution_frontier;
    std::uint32_t max_execution_frontier;
    DecodeGraph ordinary;
    DecodeGraph ordinary_aligned;
};

struct MtpGraphVariant {
    std::uint32_t min_execution_frontier;
    std::uint32_t max_execution_frontier;
    DecodeGraph mtp;
};
```

Each table consists of sorted, non-overlapping, gap-free intervals over its own legal domain. The
ordinary table covers every E in `[0,C-1]`. When `K>0`, the MTP table covers every E in
`[0,C-2K]`; it has no entry above that boundary. The target may use arrays or another direct finite
container, but this must not become a common graph planner.

Before a round, Program selects the unique variant containing the current E from the table for the
chosen round kind. The schedule derives a tight `GqaExecutionEnvelope` for each captured GQA call
from that variant and the fixed round shape. For an ordinary interval `[E_min,E_max]` and an MTP
interval `[M_min,M_max]`:

```text
ordinary T=1:        [E_min + 1,       E_max + 1]
target verify T=K+1: [M_min + K + 1,   M_max + K + 1]
MTP batch T=K+1:     [M_min + K + 1,   M_max + K + 1]
MTP AR iteration i:  [M_min + i + 1,   M_max + K + i + 1], 1 <= i < K
```

For the AR interval, device accepted-draft count `A` is in `[0,K]` and the actual visible window is
`W=E+A+i+1`. The final `i=K-1` upper bound is `M_max+2K`, so the table domain guarantees every
envelope remains within C. These per-call intervals are used instead of one unnecessarily broad
full-round envelope, allowing `min_visible_keys` to participate in INT8 selection.

At `E=C-2K`, MTP remains graph-covered when its other scheduling gates are satisfied. At
`E=C-2K+1`, the MTP capacity gate is false and only ordinary graph coverage exists. When graph
execution is enabled, a gap in either legal table or an envelope beyond C is a target construction
error, not an implicit eager fallback. Diagnostics remain deliberately eager.

### 9.2 Capture preparation

Capture no longer derives launch policy from reset cache state. For each variant:

1. Before variant warmup starts, Program initializes the allocated physical K/V code and scale
   planes once to the valid all-zero cache representation. This makes every cache row that a
   long-frontier warm body may read defined without publishing a logical prefix.
2. Program supplies the variant's explicit GQA envelopes to the schedule.
3. Graph-stable device controls and target recurrent state are initialized to a representative
   valid frontier/state in the variant before the eager warm body, again before stream capture, and
   again before the newly instantiated graph's first replay. Warmup must not leave the capture body
   starting from an advanced `io.pos` or mutated recurrent slot.
4. Capture records grids and concrete INT8 functions selected from the explicit envelopes.
5. Program resets logical frontiers, workspace position, and mutable recurrent/control state
   between variants.
6. Physical cache bytes written by warmup are simply outside any published prefix afterward.

The representative control state is important: warm execution must satisfy the same position
interval and populated-cache preconditions promised to the Op. One-time zero initialization is a
capture-preparation transfer, not a lifecycle reset and not a published cache prefix. Warm execution
need not publish a cache prefix or preserve generated output.

The current `warm_capture` helper therefore needs an explicit target setup callback or equivalent
per-stage reset; calling the same mutating body three times without restoring representative state
is not sufficient for tiered capture.

### 9.3 Tier selection evidence

Exact tier boundaries are implementation constants chosen from measurement, not guessed in this
contract. The retained benchmark must cover the union of relevant split-policy and INT8 kernel
specialization transitions for T=1..6, both cache formats, and both head geometries. A coarser tier
is valid only if one captured implementation remains high-performance throughout it.

The initial implementation uses finite tier graphs. These alternatives are not the default design:

- **one max-context graph:** semantically simple but launches a long-context grid and one fixed INT8
  variant at short contexts; use only if measurement proves it meets the same performance bar;
- **graph kernel-node parameter updates:** can avoid whole-graph duplication but couples graph
  management to private Op nodes/functions and adds per-round host update work;
- **capture/recapture per request:** adds latency and lifecycle complexity and is outside the current
  stable-at-load graph policy;
- **cursor-derived capture:** is the bug being removed.

If measured graph memory makes finite variants unacceptable, node updates require a separate design
review rather than being hidden in this refactor.

### 9.4 Graph memory

Every graph executable is real driver memory. The target must measure total graph consumption for
the final variant set, update its explicit graph allowance, include that allowance in the sequence
memory plan, and retain the existing post-capture consumption check. The future 35B 32-GB/256K
budget includes this measured graph cost; graph variants are not free unreported capacity.

## 10. Source ownership after the refactor

| Source area | Responsibility |
|---|---|
| `src/core/kv_cache.*` | physical multi-layer allocation/binding and checked per-layer views |
| `include/ninfer/ops/gqa_attention.h` | A1/A2/A3 math, effects, formats, envelope, alias, workspace contracts |
| `src/ops/wrapper/gqa_attention.cpp` | shape/format/envelope validation, workspace scope, finite dispatch |
| `src/ops/launcher/gqa_attention*` | interval capacity, concrete kernel selection, grid/block/smem |
| `src/ops/kernel/gqa_attention*` | append/cached device implementations, actual split policy, reduction |
| `src/targets/<target>/impl/program` | published cache prefixes, transactions, graph tiers, memory allowance |
| `src/targets/<target>/impl/schedule` | explicit positions/envelopes and composition of A1/A2/A3 |

The future 35B target consumes the same central contracts and exact 16Q/2KV implementations. It does
not copy the 27B Program, include a sibling target, or create a Qwen-family runtime base.

## 11. Implementation sequence

The cutover is direct; no compatibility overloads or dual cursor path remain.

### Phase 1: physical view and contracts

1. Add `KVCacheLayerView` and checked layer-view construction.
2. Add `GqaExecutionEnvelope` and rewrite the A1/A2/A3 contract comments.
3. Change all GQA entry points to consume a layer view; give A3 caller workspace.
4. Update wrappers and benchmarks to the new explicit interface.

### Phase 2: split policy and A3

1. Replace host `kv.pos + T` with envelope-derived launch capacity.
2. Unify host/device split policy by geometry, T, and cache format.
3. Make the reducer compute the same actual active split count as the producer.
4. Select INT8 kernel configuration from the complete envelope.
5. Add the cached-only small-T implementation using the shared split-KV machinery.
6. Preserve and recheck A1/A2 eager performance before target integration changes.

### Phase 3: target frontiers and eager schedules

1. Add explicit Text and MTP valid-prefix fields to Program.
2. Pass starting positions/frontiers into schedule composition and return publishable results.
3. Convert prefill, prefix reuse, ordinary decode, MTP bridge, MTP prompt, verification, proposal,
   diagnostics, reset, invalidation, and round resolution.
4. Delete `KVCache` cursor/slot lifecycle APIs and their obsolete tests.
5. Reconcile active architecture wording that currently calls cache cursors a core mechanism.

### Phase 4: graph tiers

1. Add independent finite target-private ordinary and MTP frontier tables and selection.
2. Capture each variant with explicit per-call envelopes and representative stable controls.
3. Measure graph memory and set the target allowance.
4. Benchmark replay across tier boundaries, then freeze the smallest variant set that meets the
   performance target.

### Phase 5: 35B consumption

The 35B target is implemented only against the final interface. Its 16Q/2KV A1/A2 evidence is
retained; A3, graph replay, Program frontiers, and 256K graph memory are qualified through the new
design rather than inheriting the old cursor behavior.

## 12. Verification and acceptance

Verification is limited to observable semantic, integration, and performance risks introduced by
this refactor.

### 12.1 Op correctness

- A1 and A2 produce exact INT8 code and FP16 scale bits from the Section 4.1 codec oracle; A1 and A3
  match the common higher-precision decoded-cache attention oracle for 24Q/4KV and 16Q/2KV, BF16
  and INT8-G64, at the supported T regimes.
- Internal INT8 Q round-trip, decoded K/V staging casts, and prompt/small-T reduction trees are
  evaluated as implementation approximation against that common oracle, not copied into separate
  path-specific semantic references.
- A1 and A3 produce contract-equivalent results for the same device positions under an exact
  envelope and a larger valid envelope.
- A1/A2 correctness covers every runtime-reachable prompt T below the selected target
  `prefill_chunk`, including short final tails and checkpoint-boundary truncation; performance
  evidence covers the finite dispatch/kernel transitions and representative shapes identified in
  Section 4.5.
- A3 leaves all cache code and scale planes unchanged.
- The split producer and reducer cover policy breakpoints and non-monotonic INT8 ranges without
  reading inactive partials.
- Prompt A1/A3 and small-T A1/A3 agree at an overlapping supported shape.

### 12.2 Target lifecycle

- reset and prefix rewind change published prefixes without clearing physical storage;
- prompt append, zero-suffix reuse, boundary reuse, ordinary pending/resolution, MTP
  pending/resolution, and invalidation preserve the Section 8 relationships;
- terminal strict-prefix acceptance of an MTP pending round invalidates the Program without
  publishing or rewinding either cache prefix to the accepted sub-prefix;
- MTP prompt A2 -> A3 -> proposal steps use the newly written rows before the persistent MTP prefix
  is published;
- graph-disabled diagnostics and graph-enabled execution use identical frontier semantics.

### 12.3 Graph correctness and performance

One graph is captured per retained variant and replayed without recapture at representative and
boundary frontiers. The checks include:

- device positions far from the representative capture position;
- every T=1..6 call shape present in ordinary and MTP graphs;
- BF16 and INT8-G64;
- final long-context tiers for 27B and the 35B 256K target;
- MTP coverage at `E=C-2K`, ordinary-only coverage at `E=C-2K+1`, no MTP legal-domain gaps, and
  every captured GQA envelope bounded by C;
- equality to the eager semantic result within the Op contract;
- producer CTA capacity and concrete INT8 kernel matching the declared envelope;
- graph replay latency versus the qualified eager kernel and whole `ninfer_bench` decode behavior.

An isolated eager microbenchmark is not sufficient evidence for graph-integrated support. The final
variant set must retain the command matrix, hardware/toolchain, cache format, interval, selected
kernel, and graph memory consumption needed to interpret the result.

### 12.4 Completion gate

The refactor is complete only when:

- no `src/ops` GQA code reads a cache lifecycle cursor;
- `KVCache` contains no lifecycle cursor or cursor mutation API;
- every GQA target call supplies a layer view and, for A1/A3, an explicit envelope;
- A1/A3 graph replay selects qualified capacity/configuration across all retained tiers;
- A3 `T=1`, the current target-required cached-only shape, is roofline-qualified rather than routed
  through prompt attention; `T=2..6` becomes a support claim only with a concrete caller and the
  corresponding retained evidence;
- 27B product routes pass meaningful Text, multimodal, MTP, prefix-reuse, and graph execution checks;
- the 35B operator inventory is updated only for paths actually qualified;
- stable decisions are integrated into active authorities and this plan is archived.

## 13. Explicitly rejected shortcuts

- Reading `KVCache::pos` or another target frontier in the wrapper/launcher.
- Temporarily advancing and restoring a frontier around an Op call.
- Treating the last physically written byte as a reusable cache prefix.
- Passing a host “exact cache end” without a graph-stable interval design.
- Using `cache.max_context` as the default launch window without performance qualification.
- Implementing A3 with dummy K/V, cache rewrites, or a collection of under-parallel small kernels.
- Letting Program select split counts, CTA geometry, INT8 tiles, or CUDA symbols.
- Introducing a generic attention planner, runtime graph model, or cross-target schedule base.
- Preserving old cursor APIs for compatibility after all production callers have moved.
