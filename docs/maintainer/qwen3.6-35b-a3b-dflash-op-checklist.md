# Qwen3.6-35B-A3B DFlash Op checklist

This checklist is the implementation authority for the Op work needed by the optional
Qwen3.6-35B-A3B DFlash proposal path after its weights have been represented in the `.ninfer`
artifact. It replaces the deleted historical checklist, which qualified only the target
verification pass before the companion's actual tensor and execution contracts were available.

The DFlash product route is Text-only. Image and video requests continue through the existing
non-DFlash inference routes; this checklist does not define, implement, or qualify a multimodal
DFlash path.

The governing model and storage contracts are:

- [`qwen3.6-35b-a3b-model.md`](qwen3.6-35b-a3b-model.md), Sections 9, 10, 15, and 16;
- [`qwen3.6-35b-a3b-artifact.md`](qwen3.6-35b-a3b-artifact.md), Sections 8, 9, and 13.3;
- [`tensor-formats.md`](tensor-formats.md) for `W8G32_F16S`; and
- [`op-development.md`](op-development.md) for Op boundaries and qualification.

The artifact binder currently makes the companion weights available conditionally. It does not
yet provide proposal scheduling, persistent context, workspace, CUDA Graphs, or a public execution
route. This document inventories those execution needs without naming any Op after DFlash.
Model-specific call order and state publication remain Program responsibilities; reusable
transformations remain Ops under `src/ops`.

## 1. Contract and workload

### 1.1 Attention-mask contract

All six companion layers are non-causal. Layers `0..4` use symmetric local attention and layer
`5` uses full attention. The artifact intentionally does not encode attention class.

Let `L` be the committed context length before a proposal round, let the query block contain `B`
rows, and let query row `i` have absolute position `p_i=L+i`. The two logical key segments occupy:

```text
context keys:  p_j = 0 .. L-1
query keys:    p_j = L .. L+B-1
```

For the configured sliding-window parameter `W=4096`, the exact local predicate is:

```text
allowed_swa(p_i, p_j) = (0 <= p_j < L+B) and (abs(p_j - p_i) < W)
                       = (p_i-4095 <= p_j <= p_i+4095), clipped to populated keys
```

Both endpoints at distance 4095 are inclusive; a key at distance 4096 is excluded. The mask is
symmetric rather than left-only, so its maximum extent with keys populated on both sides is 8191
positions. For the product range `B=2..16`, every query row therefore sees all `B` temporary query
keys. Once the left boundary is fully populated, query row `i` sees context positions
`L+i-4095 .. L-1`, namely `4095-i` context keys, in addition to the complete query block.

This contract follows the checkpoint reference's FlashAttention route. Transformers maps scalar
`sliding_window=4096` to FlashAttention `window_size=(4095,4095)` because FlashAttention's edges
are inclusive and bottom-right-aligns a shorter Q sequence with K. Omitting the local mask when
`L+B<=4096` is equivalent for this geometry. The reference's unmasked SDPA fallback does not
enforce the sliding window once the sequence is longer and is not a numerical oracle or an
alternate supported semantic profile.

A cyclic context cache may retain 4096 physical rows, but the extra context row at distance 4096
must never influence an output. Boundary and ring-wrap tests must poison that row independently
from the admitted row at distance 4095.

### 1.2 Notation

| Symbol | Meaning | Required domain |
|---|---|---:|
| `K` | proposal count | `1..15` |
| `B=K+1` | query block: one anchor plus `K` masks | `2..16` |
| `A` | accepted proposal prefix | `0..K` |
| `C=A+1` | target input/context columns committed by the round | `1..16` |
| `L` | committed context length before the round | `0..262144` subject to capacity |
| `T_p` | one target-prefill chunk | `1..prefill_chunk`, default maximum 1024 |
| `W` | local-context capacity | `4096` |

Every `B=2..16` is a separate semantic case. In a non-causal layer, calculating `B=16` and
discarding trailing columns does not reproduce a shorter block because the retained queries would
have seen different temporary keys.

For an existing Op whose contract names a token extent `T`, that public axis remains any positive
extent; `B`, `C`, and `T_p` identify this caller's correctness and performance points rather than
new admission limits. Qualification of each adapted small-T route therefore includes every exact
`T=1..16`, even when the proposal schedule itself calls only `B=2..16`. The new attention Ops
follow the same positive-query-extent rule; their first product-critical range is `B=2..16`.
`prepare_masked_block` is different because its block size is an explicit finite semantic input.

The registered companion geometry is fixed:

| Property | Value |
|---|---:|
| layers | 6 |
| hidden / intermediate | 2048 / 6144 |
| query / KV heads | 32 / 8 |
| head dimension / GQA group | 128 / 4 |
| attention scale | `1/sqrt(128)` |
| plain RMSNorm epsilon | `1e-6` |
| 1-D RoPE | split-half, rotary dimension 128, theta `1e7` |
| stored vocabulary rows / valid token rows | 248320 / 248077 |
| mask embedding row | 248077 |
| persistent cache dtype | BF16 |

All companion matrices use `W8G32_F16S` RowSplit storage with FP16 scales. A numerical oracle
decodes the represented signed code with its exact stored scale for every group of 32 input
elements. Source BF16 weights and another production route are not oracles.

### 1.3 Round dataflow

The complete path that this checklist must eventually support is:

1. At target Text blocks `1,6,11,16,22,27,32,37`, capture the complete post-MLP residual
   `[2048,T]` in that order.
2. Project their logical concatenation `[16384,T]` with W8 weight `[2048,16384]`, then apply plain
   RMSNorm to produce target context features `c [2048,T]`.
3. For each of six companion layers, project only the K/V rows of the stored fused QKV weight,
   normalize and rotate K, and append the committed prefix to that layer's context cache.
4. Build `ids=[anchor,MASK,...,MASK]` and absolute scalar positions `[L,...,L+B-1]`, then gather
   the shared W8 token embeddings into `u_0 [2048,B]`.
5. Run six plain-norm, QKV, bidirectional attention, output-residual, dense-SwiGLU, and
   down-residual layers.
6. Plain-normalize only the `K` mask columns, use the shared Q6 full output head, and greedily
   select one same-position proposal from each column.
7. Run the already-qualified target causal verification pass over `[anchor, proposals...]`.
8. Publish the accepted target result and make only input columns `0..A`, namely `C=A+1`
   positions, durable in target state and companion context. The correction or bonus token is the
   next unprocessed anchor.

Temporary query K/V is never persistent context. Rejected target-feature and context-KV bytes may
exist only where no logical frontier can reach them.

## 2. Inventory summary

### 2.1 New semantic Ops

These transformations do not have a compatible public semantic contract today. Their names
describe the computation rather than its first caller.

| ID | Op | Priority | Purpose |
|---|---|---:|---|
| `OP-N01` | `prepare_masked_block` | P1 | Build anchor/mask ids and absolute positions on device |
| `OP-N02` | `swa` | P0 | Read-only non-causal sliding-window GQA over context plus a query block |
| `OP-N03` | `bidirectional_gqa_attention` | P0 | Read-only non-causal full GQA over context plus a query block |
| `OP-N04` | `kv_cache_append_prefix` | P0 | Transactionally write only a device-selected K/V prefix |

### 2.2 Existing Ops requiring adaptation

The first implementation choice is to extend the listed existing Op and its nearest W8 or
attention kernel. A parallel model-labelled implementation is not wanted.

| ID | Existing Op | Required extension | Priority |
|---|---|---|---:|
| `OP-A01` | `linear` | W8 feature projection `[2048,16384]` | P0 |
| `OP-A02` | `rmsnorm` | plain D=2048 and per-head D=128 qualification | P0 |
| `OP-A03` | `linear_pair` | W8 K/V row views `[1024,2048]` | P0 |
| `OP-A04` | `rope` | 1-D full-head D=128, Q32/KV8 | P0 |
| `OP-A05` | `attn_input_proj` | W8 `[q,k,v]` direct split from `[6144,2048]` | P0 |
| `OP-A06` | `linear_swiglu` | W8 `[12288,2048] -> [6144,T]` | P0 |
| `OP-A07` | `linear_add` | W8 MLP down `[2048,6144]` | P0 |

### 2.3 Existing support to reuse

| ID | Existing support | Current conclusion | Remaining DFlash-specific evidence |
|---|---|---|---|
| `OP-R01` | W8 `embedding [248320,2048]` | direct reuse | mask row 248077, repeated masks, every `B=2..16`, Graph replay |
| `OP-R02` | W8 `linear_add [2048,4096]` | direct reuse for attention output plus residual | containing companion-layer measurement |
| `OP-R03` | W8 `linear [12288,2048]` plus `silu_mul` | correct composed control | production route is `OP-A06` |
| `OP-R04` | Q6 `linear [248320,2048]` | direct reuse for the full shared head | run only `K=1..15` mask columns |
| `OP-R05` | `argmax(valid_rows=248077)` | direct reuse | same-position proposal integration |
| `OP-R06` | target verify and greedy speculative transaction | target side supports `K=1..15`, `A=0..15` | add companion-context commit/resume checks |

The old 35B `gqa_attention` qualification is target-only: causal, head dimension 256, and
16 query/2 KV heads. It is not evidence for `OP-N02` or `OP-N03`.

### 2.4 Deferred fusion notes

The extra fusions below are outside the current implementation phase. `FUT-F01..F07` are retained
only as design notes for a later profiling-driven optimization phase; they are not current
checklist rows and require no implementation, public contract, test, benchmark, or commit now.

The current phase uses the explicit composed routes described in Sections 3 through 5. This
deferral does not remove adaptations of already-established semantic Ops such as
`attn_input_proj`, `linear_swiglu`, and `linear_add`.

| Future ID | Candidate Op | Composition that a later phase may remove |
|---|---|---|
| `FUT-F01` | `head_rmsnorm_rope` | separate per-head norm and RoPE launches |
| `FUT-F02` | `context_kv_append` | K/V projection, K norm/RoPE, and accepted-prefix append |
| `FUT-F03` | `linear_argmax` | full proposal-logit materialization and reread |
| `FUT-F04` | `linear_add_rmsnorm` | residual publication followed by a separate plain norm launch |
| `FUT-F05` | `multi_input_linear` | materialized eight-way target-feature concatenation |
| `FUT-F06` | `masked_block_embedding` | masked-block preparation plus repeated-row embedding gather |
| `FUT-F07` | `linear_rmsnorm` | feature-projection output materialization before context norm |

## 3. New Op definitions

### OP-N01: `prepare_masked_block`

**Status:** [x] definition [x] correctness [x] benchmark [x] route [ ] integration

**Definition**

For device I32 scalars `anchor` and `length`, host semantic parameter `mask_id`, and finite output
extent `B`:

```text
ids[0]       = anchor
ids[i]       = mask_id,                 1 <= i < B
positions[i] = length + i,              0 <= i < B
```

`ids` and `positions` are distinct contiguous I32 `[B]` outputs. Inputs remain unchanged. The Op
has no workspace or persistent state. The current registration is `B=2..16`, nonnegative
positions representable by I32, and `mask_id=248077` at the companion call site.

**Closest implementation**

Reuse the one-block structure of `speculative_prepare_verify_inputs` and the arithmetic of
`offset_i32_positions`. This should be one small exact-transform kernel, not a DFlash helper in a
target directory.

**Qualification**

- Compare every output element exactly for every `B=2..16`, including near-maximum valid
  positions.
- Poison both outputs and check full overwrite, non-aliasing, and preserved inputs.
- Capture and replay `B=2` and `B=16`.
- Benchmark only as part of block preparation unless the launch becomes visible in the round.

**Qualification result (2026-07-24)**

The repository now owns one exact-transform Op, wrapper, compact launcher, one-block kernel,
independent test, and candidate/production benchmark. The implementation reads the device anchor
and length directly, writes both complete I32 outputs, and owns no workspace or state. It does not
consume draft tokens and does not fuse the deferred embedding gather.

`ninfer_prepare_masked_block_test` passed exact comparison at every `B=2..16`, both ordinary and
maximum-valid I32 positions. Both outputs are poisoned before execution and surrounded by guards;
the test also proves preserved inputs, rejects output/output and input/output aliasing, qualifies
all 32/64/128/256-thread candidate launches at every B, and captures/replays the public route at
`B=2` and `B=16` with changing device inputs.

Production uses one 32-thread CTA for every `B=2..16`. On RTX 5090, CUDA 13.1, `sm_120a`, all four
candidate block sizes were tied at the timer resolution: the clock-conditioned hot CUDA Graph
median was `2.048 us` at representative `B=2,8,16`, while the 256 MiB cold-cache medians stayed
within `3.328..4.128 us` across repeated processes. Same-process candidate sweeps did not expose
a B-dependent winner. Larger blocks therefore provide no measured benefit and only add idle warps.

The targeted Nsight Compute command used `--set basic`, kernel regex
`prepare_masked_block_kernel`, and `--launch-count 1`; its report is
`profiles/ncu/prepare_masked_block_b16_basic.ncu-rep`. At `B=16`, the selected one-CTA/one-warp
kernel used 16 registers/thread and zero static or dynamic shared memory. It executed for only
11.95 active SM cycles inside a `2.43 us` profiled launch, with 1.26% DRAM throughput and 0.00%
reported compute throughput. The logical traffic is only `(2+2B)*4 = 24..136` bytes, so bandwidth
roofline percentage is not a meaningful acceptance target: the qualified terminal condition is
the minimum legal launch topology at the fixed launch floor.

The remaining unchecked integration gate belongs to masked-block embedding and the complete
proposal graph, which do not exist yet.

### OP-N02: `swa`

**Status:** [x] mask contract [x] definition [x] correctness [x] benchmark [x] route [ ] integration

**Definition**

The Op consumes:

- BF16 `q [128,32,B]`;
- temporary BF16 `query_k` and `query_v [128,8,B]`;
- a read-only BF16 cyclic context K/V view with logical absolute positions and capacity `W=4096`;
- I32 query positions `[B]`; and
- scale `1/sqrt(128)`.

For query head `h`, `kvh=floor(h/4)`, using the exact `allowed_swa` predicate from Section 1.1:

```text
score(i,j,h) = scale * dot(q[:,h,i], key[:,kvh,j])
probability  = softmax over every admitted context or query key j
out[:,h,i]   = BF16(sum_j probability(i,j,h) * value[:,kvh,j])
```

Context keys and temporary query keys are two logical sequence segments. The implementation must
not concatenate them physically. All admitted query-block rows are non-causal under the active
model contract. The context and query K/V are read-only; `out [128,32,B]` is the only mutation.
In particular, query K/V is not appended to persistent state.

**Closest implementation**

Reuse grouped-head mapping, online softmax, cache decoding/load structure, long-context
split/reduce, and execution-envelope ideas from `gqa_attention`. Reuse a cyclic cache view rather
than adding position arithmetic to the target schedule. A fixed Q32/KV8/D128 `sm_120a` route is
preferred over a generic runtime attention framework.

**Qualification**

- Use one independent FP64 attention oracle over represented BF16 Q/K/V.
- Cover every `T=1..16`, including every product block `B=2..16`; `L=0,1`; total K lengths
  immediately below, at, and above 4096; a full window; at least two ring wraps; and positions near
  maximum context.
- At the inclusive edge, prove that context distance 4095 contributes and distance 4096 does not.
  Verify the symmetric `+4095`/`+4096` predicate separately, and prove all-to-all query-block
  visibility for every product `B`.
- Include adversarial equal logits, a dominant key, cancellation in V, and non-finite rejection.
- Check every output, unchanged query/context inputs, untouched cache bytes, and ring position
  tags or equivalent validity metadata.
- Capture/replay the shortest and native blocks at empty, boundary, and wrapped contexts.
- Benchmark hot- and cold-cache latency across `B=2..16`, with boundary and steady-state full
  windows. Record the selected route and split count.
- Close integration only after a six-layer proposal replay shows the local-attention route and no
  query-K/V cache writes.

**Qualification result (2026-07-24)**

The repository now owns the public semantic contract, cyclic BF16 cache view, wrapper, workspace
calculation, compact `T=1..16` launcher, tensor-core kernel, split reducer, independent test, and
candidate/production benchmark. Context and temporary query K/V remain separate read-only
segments. The kernel maps each absolute context position into the 4096-slot ring, globally stages
only the union needed by the query block, and then applies the per-query
`abs(context_position-query_position)<4096` mask.

`ninfer_swa_test` passed every `T=1..16` against an independent FP64 oracle. The cases include
`L=0,1`, both sides of the measured route boundary, `L=4095,4096`, two ring wraps, the
`L=262144,T=16` maximum model position, forced direct and 64-key controls, input/cache
immutability, output guards, and repeated CUDA Graph replay. The largest observed relative L2
error was `2.715e-3`. Analytic equal-logit cases separately prove that distance 4095 is included,
distance 4096 is excluded, wrapped slots preserve the same rule, and every query row can see the
entire query block. A dominant-key/cancelling-V case had `1.403e-3` relative L2 error; non-finite
scales are rejected before launch.

Production dispatch is the measured winner for all `T=1..16`:

| `max_context` | Route | Key tile | Split capacity |
|---:|---|---:|---:|
| `0..96` | direct | 32 | 1 |
| `97..1024` | split-KV | 32 | `ceil(max_context/32)` |
| `1025+` | split-KV | 32 | 32 |

The route is selected from the host execution envelope, while the device position controls the
actual live context on every replay. Cold-cache measurements put the direct/split crossover
between 96 and 128 rows for every `T`; raising a full-window split count from 32 to 40, 43, 48, or
64 regressed both hot and cold latency, so those occupancy-oriented candidates remain
benchmark-only.

On RTX 5090, CUDA 13.1, `sm_120a`, a stable full-window CUDA Graph sweep at `L=4096` measured
`14.973..18.994 us` hot and `21.632..23.808 us` after a 256 MiB cache flush across `T=1..16`.
That is `49.9..62.7%` hot and `39.9..43.5%` cold against the conservative useful-data roofline,
which counts the cyclic context once and excludes implementation scratch. At `T=16`, including
the unavoidable selected-route partial write/read raises the traffic floor from `9.543 us` to
`14.370 us`; the `18.649 us` hot result is `77.0%` of that implementation-traffic roof.

Nsight Compute on the selected `T=16`, 32-key, 32-split partial kernel reports 256 CTAs,
0.75 waves/SM, 200 registers/thread, 16.38 KiB dynamic shared memory, 12.50% achieved occupancy,
53.61% DRAM throughput, and 28.18% SM throughput. This confirms a memory-bound fixed-window
route whose remaining gap is dominated by its sub-wave launch and required split/reduce
materialization; profiler replay duration is not substituted for the graph benchmark.

The remaining unchecked integration gate belongs to the six-layer proposal path, which does not
exist yet.

### OP-N03: `bidirectional_gqa_attention`

**Status:** [x] definition [x] correctness [x] benchmark [x] route [ ] integration

**Definition**

Inputs and output use the same Q32/KV8/D128 BF16 geometry as `OP-N02`. The persistent context is a
read-only linear cache over `L` committed positions, up to the configured 262144-token capacity.
For every query row, the admitted key set is all populated persistent-context rows plus all `B`
temporary query rows:

```text
score(i,j,h) = scale * dot(q[:,h,i], key[:,floor(h/4),j])
probability  = softmax over j in {context, complete query block}
out[:,h,i]   = BF16(sum_j probability(i,j,h) * value[:,floor(h/4),j])
```

There is no causal triangle and no cache mutation.

**Closest implementation**

Adapt the current `gqa_attention` kernel family for D128/group-4 and a two-segment, read-only key
source. The non-causal Vision attention implementation may supply useful tiling patterns, but its
head geometry and segmented-image semantics are not the public contract.

**Qualification**

- Compare directly with an FP64 full-attention oracle at every `T=1..16`, including every product
  block `B=2..16`.
- Cover `L=0,1`, short contexts around kernel route changes, 4096, a representative long context,
  and the maximum execution envelope.
- Verify that all query rows can affect all other query rows, including a case whose result differs
  from causal attention.
- Check no cache/query mutation and full output writes.
- Capture/replay `B=2` and `B=16` at short and long context envelopes.
- Benchmark hot and flushed-cache conditions; select routes by `B` and visible keys without
  changing the mathematical domain.
- Close integration in the complete sixth companion layer and complete proposal graph.

**Qualification result (2026-07-24)**

The repository now owns the public semantic contract, wrapper, workspace calculation, compact
`T=1..16` launcher, tensor-core kernel, split reducer, independent test, and candidate/production
benchmark. The implementation keeps context and query K/V as separate read-only sources. A direct
single-kernel route handles `L=0`; non-empty contexts use online-softmax split-KV partials followed
by one reducer.

`ninfer_bidirectional_gqa_attention_test` passed every `T=1..16` against an independent FP64
full-attention oracle, with `L=0,1`, short route shapes, `L=4096`, both 32- and 64-key tiles, input
immutability, output guards, and all-to-all query visibility. Across the random cases the largest
observed relative L2 error was `2.754e-3`. Separate analytic equal-logit cases cover the exact
`L=262144,T=2/16` maximum and replay each CUDA Graph twice; the T=16 maximum absolute error was
`3.725e-9`.

Production dispatch is the measured winner for these host execution envelopes:

| T | `max_context` | Key tile | Split capacity |
|---:|---:|---:|---:|
| `1..8` | `1..131072` | 32 | `min(ceil(max_context/32),32)` |
| `1..8` | `131073..196608` | 32 | 48 |
| `1..8` | `196609..262144` | 32 | 64 |
| `9..16` | `1..65536` | 32 | `min(ceil(max_context/32),32)` |
| `9..16` | `65537..131072` | 64 | 32 |
| `9..16` | `131073..196608` | 64 | 38 |
| `9..16` | `196609..262144` | 64 | 40 |

Every capacity is also capped by the number of context tiles, so a short exact envelope does not
launch inactive split CTAs. Candidate controls remain benchmark-only.

On RTX 5090, CUDA 13.1, `sm_120a`, a stable cold-cache CUDA Graph sweep at `L=262144` measured
`663.3..670.7 us` median across `T=1..16`. This is
`1.595..1.619 TB/s` of useful traffic and `89.0..90.3%` of the
`max(bytes/1792 GB/s, FLOPs/209.5 TFLOP/s)` roofline. The corresponding hot-cache sweep measured
`649.0..655.2 us`, or `91.5..92.3%` of the same conservative DRAM roofline. Nsight Compute on
`T=16` confirmed the selected 64-key/40-split launch: 320 CTAs, 0.94 waves/SM, 242
registers/thread, 32.77 KiB dynamic shared memory, and 7.50 achieved active warps/SM. Nsight replay
timing is not substituted for the graph benchmark.

The remaining unchecked integration gate belongs to the complete sixth companion layer and
proposal graph, which do not exist yet.

### OP-N04: `kv_cache_append_prefix`

**Status:** [x] definition [x] correctness [x] benchmark [x] route [ ] integration

**Definition**

The Op consumes candidate BF16 `k,v [128,8,T]`, sequential absolute I32 `positions [T]`, a device
I32 scalar `commit_count`, an explicit cache layout, and a host execution envelope
`0 <= commit_count <= T`. It writes exactly:

```text
for i in [0, commit_count):
    slot = positions[i]                       # linear layout
    slot = positions[i] mod capacity          # cyclic layout
    cache_k[:,slot,:] = k[:, :, i]
    cache_v[:,slot,:] = v[:, :, i]
```

No byte for `i >= commit_count` is written. BF16 bits are copied exactly. The Op does not decide
the count, publish the sequence frontier, or own cache lifetime. The Program publishes its
frontier only after the append is ordered on the same stream.

The caller supplies the next sequential absolute positions beginning at its current frontier. For
a cyclic view, its declared live interval ends immediately before `positions[0]`; advancing that
interval by `commit_count` makes every overwritten old slot dead. These are explicit state
preconditions rather than a frontier inferred by the Op.

The positive candidate extent `T` is not capped by the Op. The proposal round uses `T=B=2..16` and
`commit_count=C=1..B`; the general registered domain permits `commit_count=0`. Initial prefill may
use a host-known full count over `T_p`.

**Why the prefix is required**

For a linear full cache, speculative tail writes could remain unreachable. The same shortcut is
incorrect for a `W=4096` ring: writing all `B` candidates when only a small prefix commits can
overwrite old rows that remain inside the new live window. A device-count prefix append avoids a
host synchronization and preserves Graph replay.

**Closest implementation**

Reuse BF16 stores, position handling, and validation from `gqa_kv_append`. Extend the physical
cache/view layer with the fixed D128/KV8 linear and cyclic layouts; do not put request-frontier
policy into the cache container.

**Qualification**

- Use an exact state-transition oracle for every `T=1..16` and every
  `commit_count=0..T`.
- Seed all cache bytes and verify written rows, untouched rejected rows, untouched unrelated
  planes, wrap behavior, and input preservation bit-for-bit.
- Exercise consecutive appends, multiple wraps, `A=0` (`C=1`), `A=K` (`C=B`), and resume after
  every intermediate accepted prefix.
- Capture one `T=16` graph and replay it with every device count without recapture or host reads.
- Benchmark linear and cyclic layouts. Integration evidence must show cache/frontier agreement
  after all `A=0..15` and on the following proposal round.

**Qualification result (2026-07-24)**

The repository now owns two overloads of one semantic Op for the existing linear cache view and
the fixed 4096-slot cyclic view. Both consume contiguous BF16 `[128,8,T]` K/V, sequential device
positions, a device I32 `commit_count`, and a host count envelope. The selected kernel copies 32
bytes per thread for both K and V and launches `ceil(max_count/4)` 256-thread CTAs. It reads the
device count inside the captured launch, writes only tokens below that count, and leaves the cache
unchanged if the count lies outside the declared envelope. The Op owns neither count selection nor
frontier publication.

`ninfer_kv_cache_append_prefix_test` passed an independent bit-exact state-transition oracle for
both layouts at every `T=1..16` and every `commit_count=0..T`. Every cache bit is seeded and
checked, including padded/unrelated rows, rejected candidate rows, two absolute ring wraps, K/V
separation, input preservation, and cache guards. The test also qualifies all private benchmark
candidates, proves invalid device counts perform no write, captures one cyclic `T=16` graph, and
replays that graph for every count `0..16` without recapture. From every resulting prefix it then
appends the following two-token proposal and checks the complete state again, covering resume
after every intermediate accepted prefix.

Production uses the same measured 32-byte flat route for linear and cyclic layouts:

| Host `max_count` | Block | Grid | Device behavior |
|---:|---:|---:|---|
| `0` | — | — | no launch |
| `1+` | 256 | `ceil(max_count/4)` | copy exactly `commit_count` rows |

On RTX 5090, CUDA 13.1, `sm_120a`, cold-destination CUDA Graph replay measured approximately
`3.36 us` for both layouts throughout `T=1..16`; the full `T=C=16` case moved 131072 useful bytes
at `39.0 GB/s` for both layouts. This small product extent is launch-bound: its raw
DRAM floor is only `0.073 us`. The one-CTA persistent candidate regressed to `5.408 us` once
`T>=8`; 16-byte flat and per-token candidates did not beat 32-byte flat in the product domain.
For the host-known full-prefix use case, 32-byte flat reached `908.1 GB/s` linear and
`922.8 GB/s` cyclic at `T=C=1024` while using half as many CTAs as 16-byte flat and one quarter as
many as the per-token route.

Nsight Compute on cyclic `T=C=16` confirms four CTAs, 30 registers/thread, no dynamic shared
memory, `3.52 us` duration, and only 99.6 average SM-active cycles inside 6981 elapsed cycles.
The resulting 2.08% DRAM and 0.06% SM throughput identify fixed launch time—not copy work—as the
small-round limit. Fusion with a producer could remove that launch, but fusion is explicitly
outside this phase.

The remaining integration gate requires the future proposal Program to prove cache/frontier
agreement for `A=0..15` and on the following round.

## 4. Existing Op adaptations

### OP-A01: W8 `linear [2048,16384]`

**Status:** [x] admission [x] correctness [x] benchmark [x] route [ ] integration

**Definition and shape**

The ordinary `linear` contract computes:

```text
projected[:,t] = BF16(W_fc @ concat(r1,r6,r11,r16,r22,r27,r32,r37)[:,t])
```

The W8 weight is `[2048,16384]`; input and output are contiguous BF16 `[16384,T]` and `[2048,T]`.
The logical concatenation order is fixed by the caller, not by `linear`.

Required qualification includes every exact `T=1..16` and prefill
`T=T_p in [1,prefill_chunk]`. The product round measures every `T=B=2..16`. Prefill performance
evidence must include 128, the default 1024, and final-chunk tails around any selected route
boundary.

**Adaptation path**

Add the shape to the W8 closed catalog and reuse the row-split SIMT/MMA and exact-T split-K cores.
The first correct route captures the eight target residuals into fixed row slices of one
`[16384,T]` buffer. `FUT-F05` records a possible later alternative but is not implemented in this
phase.

**Qualification**

- Exact-decode W8 codes/scales and evaluate the full GEMM independently in FP64.
- Cover every `T=1..16`, representative prefill extents, all eight input-slice boundaries, full
  output writes, guards, and cancellation.
- Capture/replay native block and prefill routes.
- Benchmark production candidates at both round and prefill scopes and verify the chosen route in
  the target-feature conditioning stage.

### OP-A02: plain `rmsnorm`

**Status:** [x] shape coverage [x] correctness [x] benchmark [x] route [ ] integration

The existing formula is already correct with `unit_offset=false`:

```text
out[d,r] = BF16(x[d,r] * rsqrt(mean_d(x[d,r]^2) + 1e-6) * weight[d])
```

Required real shapes are:

| Role | Logical shape |
|---|---|
| context, layer input, post-attention, final | `[2048,T]` |
| proposal Q head norm | `[128,32,B]` |
| proposal/query K head norm | `[128,8,B]` |
| persistent-context K head norm | `[128,8,T_p]` and `[128,8,B]` |

The D=128 and D=2048 kernel families already exist. Extend direct qualification to plain norms,
Q32/KV8 batching, every `T=1..16` including all product blocks, representative prefill `T_p`,
near-zero norms, large/small values, and Graph replay. Record separate D128 and D2048 timings;
integrate the selected composed routes without adding `FUT-F01` or `FUT-F04` in this phase.

**Op evidence (RTX 5090, CUDA 13.1, `sm_120a`)**

- `ninfer_rmsnorm_test` qualifies plain `[2048,T]`, `[128,32,T]`, and `[128,8,T]` directly
  against the FP64 oracle for every `T=1..16`, prefill `T=128,1024`, near-zero and stress inputs,
  output guards, preserved inputs, and CUDA Graph replay.
- `ninfer_rmsnorm_bench --decode --prefill` measures every `T=1..16` and both prefill points.
  The selected routes are one fixed-D128 two-pair-per-lane warp kernel with 128-thread CTAs and
  one D2048 512-thread CTA kernel. The complete small-T sweep has no dispatch boundary or
  repeatable latency cliff.
- Narrow NCU captures use one matched launch with `--set basic`. At `T=1024`, D2048 takes
  6.752 us versus 6.208 us for the same-topology payload control; Q32 D128 takes 8.736 us and KV8
  D128 takes 4.160 us, both faster than their 10.272 us and 4.192 us payload controls. The
  production benchmark reports 1720 GB/s for Q32 D128, 96.0% of the measured 1792 GB/s device
  bandwidth ceiling. Small product extents remain launch/parallelism limited rather than
  bandwidth limited.

### OP-A03: W8 `linear_pair [1024,2048]`

**Status:** [ ] admission [ ] correctness [ ] benchmark [ ] route [ ] integration

For context feature `c [2048,T]`, create zero-copy row views of one stored QKV parent:

```text
W_k = parent rows [4096,5120)     # [1024,2048]
W_v = parent rows [5120,6144)     # [1024,2048]
k_raw = BF16(W_k @ c)             # [1024,T] == [128,8,T]
v     = BF16(W_v @ c)             # [1024,T] == [128,8,T]
```

The current `linear_pair` admits only paired `[1024,5120]` weights. Extend it to the
`[1024,2048]` W8 row views and add an exact small-T route. This is preferred to projecting the
full `[6144,2048]` parent and discarding Q, and it can share each activation tile between the two
weights.

Qualification covers every exact `T=1..16`, every product block `B=2..16`, representative prefill
extents, exact row-view offsets and group-scale addressing, direct distinct K/V outputs, Graph
replay, and comparison against two independently decoded FP64 projections. The benchmark includes
two ordinary `linear` calls as the composed control.

### OP-A04: D128 `rope`

**Status:** [x] admission [x] correctness [x] benchmark [x] route [ ] integration

Extend the existing one-dimensional split-half Op to:

```text
head_dim = rotary_dim = 128
theta    = 1e7
positions: scalar absolute I32 [T]
Q: [128,32,B]
K: [128,8,B] or [128,8,T_p]
```

The generic launcher is a starting point, but the production route should be fixed and measured
for Q32/KV8/D128. Qualification uses the independent trigonometric oracle, covers every
`T=1..16` including all product blocks, representative prefill extents, position 0, large phases
near maximum context, head and token stride boundaries, full mutation of dimensions `0..127`,
preserved positions, and Graph replay.

This is one-dimensional RoPE. Target three-axis MRoPE and its `rope_delta` are not inputs.

**Op evidence (RTX 5090, CUDA 13.1, `sm_120a`)**

- `ninfer_rope_test` qualifies D128/R128 Q32/KV8 directly against the independent FP64
  trigonometric oracle for every `T=1..16`, dispatch boundaries `16/17` and `399/400/401`,
  prefill `T=128,1024`, position zero, positions near 262144, padded token strides, unaligned
  fallback, full observable mutation of dimensions `0..127`, output guards, preserved positions,
  pair/single parity, and CUDA Graph replay. Existing D256 Text/MRoPE and D72 Vision coverage
  remains passing.
- `ninfer_rope_bench --text --geometry dflash` measures the public pair route and its
  same-topology payload control. Candidate sweeps cover every `T=1..16`, representative prefill,
  64--1024-thread one-token CTAs, and 1/2/4/5/8/10/20-head split CTAs. The selected dispatch is
  five heads per CTA through `T=16`, eight heads per CTA through `T=400`, and a 160-thread
  one-token CTA above 400. NCU measures 3.39 us at `T=16`; the two route boundaries are smooth
  at 3.39/3.46 us for `T=16/17` and 7.62/7.46 us for `T=400/401`.
- Narrow NCU `--set basic` captures measure 3.39 us for production versus 3.62 us for the
  same-topology phase-and-payload control at `T=16`. At `T=1024`, production and control both
  take 12.19 us. The 20,971,520 bytes of compulsory in-place traffic therefore reach 1720 GB/s,
  96.0% of the 1792 GB/s logical memory-traffic ceiling. Small extents reach the empirical
  launch/parallelism ceiling, while prefill matches the measured phase-and-payload roofline.

### OP-A05: W8 `attn_input_proj [6144,2048]`

**Status:** [ ] contract [ ] correctness [ ] benchmark [ ] route [ ] integration

Add a Q/K/V overload to the existing attention-input projection family:

```text
q[:,t] = BF16(W[0:4096,:]    @ x[:,t])    # [4096,B] == [128,32,B]
k[:,t] = BF16(W[4096:5120,:] @ x[:,t])    # [1024,B] == [128,8,B]
v[:,t] = BF16(W[5120:6144,:] @ x[:,t])    # [1024,B] == [128,8,B]
```

The stored parent is W8 `[6144,2048]`; `x [2048,B]` and all three outputs are BF16. Outputs must
be independent final allocations. Do not materialize a padded parent output and then copy-split
it in the selected route.

Adapt the existing 35B W8 `[q,k,gate,v]` direct-output implementation and exact-T K=2048 core.
A three-way output policy with row extents 4096/1024/1024 is the closest implementation change.
Qualification covers every `T=1..16`, including every product block `B=2..16`, row boundaries
4095/4096 and 5119/5120, W8 group boundaries, full direct writes, guards, Graph replay, and one
complete FP64 exact-decode oracle for all three outputs.

The Op continues to output raw Q/K. A later kernel may not silently normalize or rotate them under
this contract; that composition requires a fused semantic Op.

### OP-A06: W8 `linear_swiglu [12288,2048]`

**Status:** [ ] admission [ ] correctness [ ] benchmark [ ] route [ ] integration

Extend `linear_swiglu` from its current Q4 geometry to:

```text
gate = W[0:6144,:] @ x
up   = W[6144:12288,:] @ x
out  = BF16(SiLU(gate) * up)

x   [2048,B] BF16
W   [12288,2048] W8G32_F16S
out [6144,B] BF16
```

Reuse the existing Op plan/fallback organization and the W8 K=2048 exact-T projection core.
`linear [12288,2048]` followed by `silu_mul` is a composed correctness and performance control,
not the preferred production route.

Compare the complete expression directly with an exact-decode FP64 oracle at every `T=1..16`,
including every product block `B=2..16`. Cover the gate/up row seam, negative and near-zero gates,
saturation, full writes, Graph replay, and fused-versus-composed timing inside a dense MLP stage.

### OP-A07: W8 `linear_add [2048,6144]`

**Status:** [ ] admission [ ] correctness [ ] benchmark [ ] route [ ] integration

Extend the existing fused residual Op:

```text
residual[:,t]' = BF16(residual[:,t] + W_down @ x[:,t])

x        [6144,B] BF16
W_down   [2048,6144] W8G32_F16S
residual [2048,B] BF16, updated in place
```

Reuse the W8 residual epilogue and exact-T split-K structure already used by
`linear_add [2048,4096]`. Qualification covers every `T=1..16`, including every product block
`B=2..16`, cancellation with the incoming residual, exact W8 decode, input preservation, full
in-place output, Graph replay, and dense-layer timing against `linear` plus an add.

## 5. Reuse checks

These checks should not create new implementations unless measurement exposes a real deficit.

### OP-R01: masked-row embedding

- Use the shared W8 table `[248320,2048]`.
- Input ids are `[anchor,248077,...,248077] [B]`; output is `[2048,B]`.
- Extend the existing exact-decode test with row 248077, a distant anchor, repeated ids, and every
  `B=2..16`.
- Verify one native-block Graph replay. Benchmark the complete block preparation plus embedding
  while retaining the explicit preparation and embedding Ops; `FUT-F06` is deferred.

### OP-R02 and OP-R03: attention and MLP controls

- Reuse `linear_add [2048,4096]` for `residual += W_o @ attention_output`.
- Reuse ordinary W8 `linear [12288,2048]` plus `silu_mul` as the dense-SwiGLU bring-up route and
  control.
- The former already has exact `T=1..16` target evidence; add a companion-layer measurement
  rather than duplicating its oracle campaign.
- The latter needs full-formula evidence only through `OP-A06`; do not claim the composition is the
  optimized result.

### OP-R04 and OP-R05: proposal head

After final plain norm, pass only mask columns `1..B-1`, producing hidden `[2048,K]`.

```text
logits = Q6_linear(shared_output_head, hidden)   # [248320,K]
drafts = argmax(logits, valid_rows=248077)       # [K]
```

The prediction is same-position and has no language-model shift. The MTP shortlist head and token
remap are not used. Existing Q6 and argmax qualifications cover `K=1..15`; add a composed
same-position proposal check and complete-stage benchmark. The current phase materializes BF16
logits and calls `argmax`; `FUT-F03` is only a later greedy-mode optimization note.

### OP-R06: target verification and acceptance

Reuse the existing target causal verification path for `B=K+1` and the existing greedy-draft
transaction:

- proposal `i` is compared with target logit column `i`;
- `A` is the longest accepted proposal prefix;
- target column `A` supplies the correction or bonus;
- target state commits input columns `0..A`;
- correction/bonus remains the next unprocessed anchor.

The current transaction already covers `K=15` and `A=0..15` for target state. Extend integration
with the six context caches, exact `C=A+1` publication, the following proposal round, eager and
Graph routes, and prefix reuse.

The current companion reference makes proposals greedily. Target sampling can therefore use the
existing one-hot draft-distribution semantics in `speculative_accept_greedy_drafts`. A future
non-greedy proposal policy would require proposal probabilities and a general stochastic
accept/correction Op; it is not part of this checklist. `sample` is not a core proposal Op here.

## 6. Deferred fusion design notes

Nothing in this section is implemented or qualified in the current phase. These definitions only
preserve plausible semantic boundaries for a future optimization pass after the explicit complete
round is correct and measured. Starting any item requires a new scope decision backed by
containing-stage evidence.

### FUT-F01: `head_rmsnorm_rope`

For each Q or K head, compute plain RMSNorm over 128 values followed by full-D128 split-half RoPE.
Inputs are raw BF16 Q/K, BF16 norm weights, absolute positions, epsilon, and theta; output is the
final rotated BF16 tensor. The contract must explicitly state whether the normalized value has an
observable BF16 seam.

Use `rmsnorm` plus `rope` as the oracle-bearing composed control. Implement only after the six-layer
launch trace shows that removing these launches matters.

### FUT-F02: `context_kv_append`

A useful closed stateful composition is:

```text
k_raw, v = linear_pair(c, W_k, W_v)
k        = rope(plain_head_rmsnorm(k_raw), positions)
kv_cache_append_prefix(k, v, positions, commit_count, cache)
```

Its only observable result is the exact updated cache prefix. It can reuse the qualified
projection, norm/RoPE, and prefix-store internals while avoiding intermediate K/V materialization.
Keep `c` materialized once and fan it out to six layers; fusing `W_fc` into this Op would recompute
the same expensive feature projection six times.

A future implementation must qualify the complete fused state transition directly, including
every accepted count and cyclic wrap. Pairwise parity with the composed route would be
supplementary only.

### FUT-F03: `linear_argmax`

For each input column, exact-decode the registered quantized matrix and define:

```text
logical_logits[v,t] = BF16(W[v,:] @ x[:,t])
out[t] = lowest v attaining max(logical_logits[0:valid_rows,t])
```

The BF16 logit boundary is semantic because this Op replaces `argmax(BF16(linear(...)))`.
Comparing private FP32 accumulators could change a near-tied token and is not equivalent. The first
registered shape is the Q6 `[248320,2048]` head with `K=1..15`.

This avoids writing and rereading up to `248320*K` BF16 logits. Preserve the independent full-head
oracle and tie rule. Do not use this Op for stochastic proposals, target sampling, or any caller
that requires logits.

### FUT-F04: `linear_add_rmsnorm`

The closed formula is:

```text
residual' = BF16(residual + W @ x)
normalized = plain_rmsnorm(residual', norm_weight)
```

Both `residual'` and `normalized` are outputs, preserving the BF16 residual boundary. Registered
shapes would be `[2048,4096]` for attention output followed by post-attention norm and
`[2048,6144]` for MLP down followed by the next input or final norm.

The norm reduction is not a trivial per-element GEMM epilogue. Retain separate Ops until
stage-level profiling shows enough launch or memory cost to justify the additional reduction
structure.

### FUT-F05: `multi_input_linear`

The general formula is:

```text
out = W @ concat(x_0, x_1, ..., x_(S-1))
```

Inputs explicitly provide ordered BF16 matrices with a common token extent, and W partitions its
K dimension accordingly. The first registered case has eight `[2048,T]` inputs and W
`[2048,16384]`.

An implementation may use a segmented input accessor or K-slice accumulation, but partial sums
are private. The target schedule still owns when the eight snapshots are captured. Introduce this
Op only if the contiguous feature buffer or its copies are material in a target-plus-context
measurement.

### FUT-F06: `masked_block_embedding`

This exact fused Op would gather the anchor row once and decode/broadcast one repeated mask row:

```text
out[:,0] = embedding_table[anchor,:]
out[:,i] = embedding_table[mask_id,:], 1 <= i < B
positions[i] = length + i
```

It is a candidate only if `prepare_masked_block` plus ordinary `embedding` is visible. The name and
contract describe a reusable masked block, not a model.

### FUT-F07: `linear_rmsnorm`

The complete formula is:

```text
out = plain_rmsnorm(W @ x, norm_weight)
```

The first registered case is W8 `[2048,16384]`, input `[16384,T]`, BF16 norm weight `[2048]`, and
BF16 output `[2048,T]`. The unnormalized projection is not an output, so a fused implementation
may keep its private representation until the row reduction is complete.

This is a natural semantic boundary for target-feature conditioning, especially for prefill where
the composed route materializes up to 4 MiB at `T=1024`. It is not necessarily a natural single
GEMM epilogue because RMSNorm requires a reduction across 2048 output rows. A future optimization
phase may reconsider it only after `OP-A01` plus `OP-A02` and the composed context-construction
stage are measured.

### 6.1 Boundaries preserved by the current composed route

- Do not create a monolithic companion-layer Op. Layer order, weight selection, and state lifetime
  are schedule concerns.
- Do not append temporary query K/V to persistent context.
- Do not fuse acceptance with context construction. Acceptance publishes a count; a separate
  state transition consumes it.
- Do not fuse feature projection into six context-K/V projections; that repeats `[2048,16384]`
  work six times.
- Do not hide Q/K norm or RoPE inside an `attn_input_proj` kernel while retaining a contract that
  claims to output raw Q/K.
- Do not fuse attention with `W_o`, or input RMSNorm with a large W8 projection, without
  containing-stage evidence. They cross reductions or head mixing and are not natural
  elementwise epilogues.
- `swa` and `bidirectional_gqa_attention` may share private tiling and softmax templates, but their
  public masks and cache contracts remain distinct.

## 7. Program and state integration requirements

These items are necessary for the proposal route but are not additional Ops.

### INT-01: target feature capture

- Capture exact BF16 post-MLP residuals after target layers `1,6,11,16,22,27,32,37`.
- Replace the current diagnostic-only tap use with graph-safe production storage.
- Store in fixed ordered slices of one materialized `[16384,T]` buffer.
- The complete snapshot payload is 512 KiB at `B=16` and 32 MiB at `T_p=1024`; include it in
  workspace planning.
- Cover target prefill chunks and target verification `B=2..16`.
- Do not capture mixer output, next-layer normalized input, or final normalized hidden.

### INT-02: context construction and caches

- Materialize one normalized feature `c [2048,T]`, then fan it out to six independent K/V
  projections.
- Layers `0..4` own bounded BF16 cyclic K/V context; layer `5` owns BF16 full-context K/V.
- Each cyclic layer stores K and V `[128,8,4096]`, 16 MiB total; five layers consume 80 MiB.
  The full layer stores K and V `[128,8,max_context]`, reaching 1 GiB at 262144 positions.
- The current companion contract is BF16-only. Target Text's INT8-G64 cache qualification does not
  implicitly register an INT8 companion cache.
- Temporary proposal-query K/V uses separate workspace.
- Initial prefill appends the complete prompt context. Round updates append exactly the selected
  target-feature prefix.
- Prefix reuse restores all six context caches and their logical positions/frontiers; target KV and
  GDN state cannot reconstruct these values.

### INT-03: accepted-prefix transaction

For every `K=1..15` and `A=0..K`, verify after the round:

```text
C = A + 1
new_target_frontier  = old_target_frontier  + C
new_context_frontier = old_context_frontier + C
pending_anchor       = correction_or_bonus
```

Check all target state families, all six companion caches, cyclic live rows, full-cache rows,
untouched rejected rows, statistics, published tokens, and the immediately following round.
Stale physical bytes are acceptable only outside every published frontier.

### INT-04: workspace and CUDA Graphs

- Keep stable addresses for block tensors, temporary query K/V, feature snapshots, context-update
  candidates, attention reductions, and output-head workspace.
- Provide graph variants or finite dispatch for every supported `K`; no graph may assume
  `K<=5`.
- `A` and `C` remain device values inside replay. The graph must not synchronize them to the host
  to choose an append length.
- Capture/replay the complete proposal, target verify, accept, and context-update round for every
  `K=1..15`; at minimum, retain detailed node evidence for `K=1`, route boundaries, and `K=15`.

### INT-05: containing benchmarks

Extend the existing target-round benchmark into a complete proposal-round benchmark with:

- proposal `K=1..15`;
- forced acceptance `A=0..K` for state and cost isolation;
- real greedy acceptance for throughput measurement;
- short, full-local-window, and representative long contexts;
- eager and CUDA Graph execution;
- proposal, target verify, accept, and context-update stage timings; and
- complete-round latency and accepted/published tokens per second.

An Op microbenchmark establishes only an Op claim. A selected route is complete after its
improvement is visible in its containing stage or the complete round.

### INT-06: Text-only admission

DFlash is admitted only for requests with Text input. Any request containing image or video media
uses the existing supported non-DFlash path without allocating companion state or launching
companion Ops. No Vision activation, multimodal placeholder span, target MRoPE position, or
`rope_delta` enters the companion route.

Add one focused product integration check that a Text request can select DFlash while equivalent
image and video requests cannot. Multimodal DFlash numerical or performance qualification is
outside scope.

## 8. Qualification and execution order

### 8.1 Common completion gates

Every row is closed only after:

- **C — correctness:** every registered shape, route boundary, output, and state effect is checked
  directly against one independent exact or FP32/FP64 oracle;
- **B — benchmark:** RTX 5090, CUDA 13.1, `sm_120a` measurements cover every exact round extent and
  the stated context/prefill regimes;
- **O — optimization:** production dispatch selects the measured winner without a cliff inside
  `B=2..16`, and the composed control remains available in the benchmark rather than the product
  route; and
- **E — enclosing evidence:** the chosen implementation improves or at least does not materially
  regress its containing layer, context-update stage, or complete speculative round.

Exact Ops compare every observable bit. Floating-point Ops use named route-appropriate criteria,
reject unexpected non-finite outputs, and combine a normwise bound with a finite gross pointwise
cap where reductions are involved. Graph capture/replay is required wherever the production route
is captured.

### 8.2 Ordered implementation checklist

Work one row at a time. Keep each Op's contract, implementation, tests, benchmark, result, and
commit together.

| Order | ID | Deliverable | C | B | O | E |
|---:|---|---|:---:|:---:|:---:|:---:|
| 0 | mask contract | Freeze non-causal symmetry and the inclusive `abs(delta)<4096` predicate | [x] | — | — | — |
| 1 | `OP-A02` | Plain D2048/D128 RMSNorm domains | [x] | [x] | [x] | [ ] |
| 2 | `OP-A04` | D128 full-head 1-D RoPE | [x] | [x] | [x] | [ ] |
| 3 | `OP-A05` | W8 Q/K/V direct projection | [ ] | [ ] | [ ] | [ ] |
| 4 | `OP-N03` | Bidirectional full GQA | [x] | [x] | [x] | [ ] |
| 5 | `OP-N02` | Symmetric non-causal SWA | [x] | [x] | [x] | [ ] |
| 6 | `OP-A06` | W8 fused dense SwiGLU | [ ] | [ ] | [ ] | [ ] |
| 7 | `OP-A07` | W8 MLP down plus residual | [ ] | [ ] | [ ] | [ ] |
| 8 | `OP-A01` | W8 target-feature projection | [ ] | [ ] | [ ] | [ ] |
| 9 | `OP-A03` | W8 context K/V pair projection | [ ] | [ ] | [ ] | [ ] |
| 10 | `OP-N04` | Device-count linear/ring prefix append | [x] | [x] | [x] | [ ] |
| 11 | `OP-N01` | Masked-block preparation | [x] | [x] | [x] | [ ] |
| 12 | `OP-R01..R06` | Focused reuse and composed-route checks | [ ] | [ ] | [ ] | [ ] |
| 13 | `INT-01..06` | Complete Text-only proposal/context round | [ ] | [ ] | [ ] | [ ] |

The order deliberately establishes the independent mathematical pieces before the new attention
and state transitions, then closes the complete composed round. Section 6 is not part of this
execution order or the current completion criteria.
