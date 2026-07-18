# SparseMoe Design Log

> Status: completed design log retained for the `SparseMoe` Op used by the registered
> `qwen3_6_35b_a3b_rtx5090` target.

This file records decisions reached while designing the closed `SparseMoe` Op. Stable conclusions
are reflected in the active Op and target authorities.

The accepted decode-only execution work is tracked in the
completed [`SparseMoe Decode Implementation Plan`](2026-07-17-sparse-moe-decode-implementation.md).

## 2026-07-17: top-level API

Decision:

- The Op identity and sole execution entry point are `SparseMoe` and `sparse_moe(...)`.
- Router projection, top-k selection, assignment grouping, grouped expert contractions, inverse
  scatter, and reduction remain implementation-private stages rather than target-callable Ops.
- A companion `sparse_moe_workspace_bytes(...)` capacity query is expected; it is a resource helper,
  not a second Op.
- The five physical weight roles are passed as one repository-internal `SparseMoeWeights` aggregate:
  `router_shared_gate`, `routed_gate_up`, `routed_down`, `shared_gate_up`, and `shared_down`.
- Final writeback is an explicit semantic epilogue of `SparseMoe`. The registered 35B call requires
  `AddResidual`, where the destination is read as the decoder residual and updated in place.
- The wrapper maps the semantic epilogue to an implementation route; a kernel may implement the
  final writeback through a template-selected epilogue. `AddResidual` does not become part of the
  Op name.
- Do not add or qualify a `Store` production mode until a supported caller actually requires it.
- The API does not expose decode/prefill, `small_t`, model role, layer id, router logits, selected
  ids, route weights, grouped jobs, or other private intermediates. Token count and registered
  weight formats select the execution route.

Current interface direction:

```cpp
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights,
                SparseMoeEpilogue epilogue, Tensor& destination,
                WorkspaceArena& workspace, cudaStream_t stream);
```

The exact workspace query signature and the remaining prefill execution/dataflow design remain to
be discussed.

## 2026-07-17: source organization

Decision:

- Keep the semantic contract in `include/ninfer/ops/` and one thin validation/workspace/dispatch
  wrapper in `src/ops/wrapper/`.
- Keep the multi-stage implementation together in a dedicated `src/ops/sparse_moe/` subtree rather
  than scattering its private launch policy, descriptors, and kernels across the global launcher
  and kernel directories.
- Organize substantial implementation work primarily by private workload-tuned routes
  (single-column, verification-sized, and throughput-oriented), because one complete route mixes
  router BF16, routed
  Q4/Q5/Q6/W8, and shared W8 work. Do not split the Op primarily into format-owned directories.
- Plan data, workspace views, assignments, inverse maps, expert jobs, and other intermediate types
  remain private to the SparseMoe subtree. Move a narrow primitive to `src/ops/common/` only after
  another Op actually shares it.
- The exact Variant owns the sparse-MoE post-mixer leaf and its leaf-local workspace contribution;
  the family runtime owns layer order/workspace lifetime and calls `sparse_moe(...)`. Neither owns
  MoE CUDA mathematics.
- Keep SparseMoe in the existing `ninfer_ops` library with explicit build source registration; do
  not create another library or runtime registry.
- Correctness evidence is centered on the complete Op against one independent oracle. Private
  stages do not receive separate mathematical contracts or stage goldens. Benchmarking covers the
  real single-column, verification-sized, and prefill workloads without turning those measurement
  points into Op domains.

The exact directory depth, source filenames, translation-unit boundaries, and compilation grouping
are intentionally deferred until implementation makes those boundaries concrete.

## 2026-07-17: decode and Small-T kernel semantics

Decision:

- Lock the private decode route for `T=1` and the private Small-T route for `2<=T<=6` before
  choosing CUDA geometry, filenames, or translation-unit boundaries.
- Both routes have four logical kernel responsibilities. The router projection and route-selection
  responsibilities may later share one launch if that implementation qualifies; this does not
  change either responsibility. Do not add a separate grouping or final-reduction launch to the
  Small-T route.
- Keep prefill routing, grouping, expert execution, and launch topology undecided. Nothing in the
  decode or Small-T decisions below is a prefill contract.

### Meaning of a private kernel semantic

The definitions below assign mathematical work and data dependencies inside the closed
`SparseMoe` Op. They do not create public sub-Ops or independently observable values. A named
logical input or output does not by itself require a tensor materialization. When two baseline
launches require a handoff, the handoff is Op-private and does not prescribe its dtype, layout,
rounding point, or longer lifetime than its consumer requires.

In particular, an implementation may keep a producer's natural result precision for its consumer,
may apply a route or shared scale before or after a linear map when the transformation is
mathematically equivalent, and may eliminate a private value through fusion. It must still produce
the complete `SparseMoe` result from the represented public input and registered weights. The
complete Op, not these private responsibilities, is checked against the independent naive
FP32/FP64 oracle.

The common represented domain is:

```text
X             BF16 [2048,T] normalized MoE input
R             BF16 [2048,T] destination containing the decoder residual
router_shared BF16 [257,2048], router rows 0..255 and independent shared-score row 256
routed_gu     256 expert banks, logical [1024,2048] per expert, gate rows then up rows
routed_down   256 expert banks, logical [2048,512] per expert
shared_gu     logical [1024,2048], gate rows then up rows
shared_down   logical [2048,512]
```

Main Text and MTP select different registered routed codecs, but execute the same formulas. Expert
id is the router row id and addresses the stored expert bank directly; no kernel gathers selected
weights into another weight buffer.

### Decode route (`T=1`)

The decode baseline is four graph-stable launches. `D1` and `D2` may become one qualified launch;
`D3` and `D4` remain separate baseline launches because they decompose the work along different
contraction axes.

#### D1. Router and shared-score projection

For input column `x=X[:,0]`, compute the 257 logical dot products

```text
router_logit[e] = dot(router_shared[e,:], x), e=0..255
shared_score    = dot(router_shared[256,:], x)
```

There is no bias. D1 owns only this projection. Its logical result is one set of 256 router logits
and one independent shared score; it need not be stored as a 257-element tensor. If D1 remains a
separate launch, its private output representation is chosen together with D2 rather than exposed
as an Op contract.

#### D2. Route selection and shared scaling

D2 consumes D1's logical scores and determines exactly eight routed assignments. At an exact
selection-boundary tie, the lower expert id is selected. For the selected set `I`, it computes

```text
alpha[e]    = exp(router_logit[e]) / sum(j in I, exp(router_logit[j])), e in I
shared_scale = sigmoid(shared_score)
```

This is the same mathematical result as softmax over all 256 logits followed by selected-value
renormalization. D2 therefore does not have to materialize all 256 softmax probabilities. A
numerically stable shifted exponential evaluation is an implementation choice, not a distinct
semantic route.

D2's logical routed result is eight associated `(expert_id, alpha)` pairs. Route-slot order is
private: the implementation may retain score order or sort the pairs by expert id, but it must
preserve the selected set and keep every weight attached to its expert. With one token, assignment
grouping degenerates to this optional reordering and does not receive another kernel.

#### D3. Selected routed and shared gate/up SwiGLU

For every routed pair `r=0..7`, with `e=expert_id[r]`, and every intermediate coordinate
`j=0..511`, D3 computes the logical activation

```text
gate[r,j] = dot(routed_gu[e][j,:],     x)
up[r,j]   = dot(routed_gu[e][512+j,:], x)
act[r,j]  = SiLU(gate[r,j]) * up[r,j]
```

It also computes the always-on shared activation

```text
shared_gate[j] = dot(shared_gu[j,:],     x)
shared_up[j]   = dot(shared_gu[512+j,:], x)
shared_act[j]  = SiLU(shared_gate[j]) * shared_up[j]
```

D3 is one selected-routed-plus-shared launch. It must not launch once per expert, scan all 256
experts, materialize gate and up as separate public tensors, or copy selected expert weights. Its
logical producer/consumer boundary is the nine 512-element SwiGLU activations. Their physical
layout and representation are private. Applying `alpha` to a routed activation at this point is
allowed because the following down map is linear; it does not change the logical formula or make
the scaled activation observable.

The previously proposed one-intermediate-coordinate CTA and nine-warp mapping remains a concrete
implementation candidate, not part of D3's semantic definition.

#### D4. Down projections, route/shared merge, and residual epilogue

For each hidden coordinate `k=0..2047`, D4 completes

```text
routed[k] = sum(r=0..7,
                alpha[r] * dot(routed_down[expert_id[r]][k,:], act[r,:]))
shared[k] = shared_scale * dot(shared_down[k,:], shared_act[:])
R[k,0]    = R[k,0] + routed[k] + shared[k]
```

D4 owns all routed down projections, the shared down projection, route weighting, the eight-route
reduction, shared scaling, and the required `AddResidual` writeback. It produces one observable
BF16 destination value per hidden coordinate. It does not expose or materialize eight
2048-element expert outputs, a standalone shared output, the merged `Y`, or a separate
residual-add result.

The points at which `alpha` and `shared_scale` are multiplied, and the association and precision
of the routed/shared/residual additions, remain implementation choices. The previously proposed
one-hidden-coordinate CTA and nine-warp mapping is a candidate rather than part of D4's semantic
definition.

#### Decode fusion boundary

The only planned fusion experiment is D1+D2. It may avoid the private score handoff and one launch,
but must be qualified against the separate projection/selection baseline because concentrating all
257 dot products into a selection-capable launch may reduce projection parallelism. D2+D3 would
require a grid-wide handoff from one global selection to the selected-expert work, and D3+D4 would
either retain a global activation handoff or recompute the 2048-wide gate/up contractions for the
512-to-2048 down map. Neither is a planned baseline fusion. D4 is already the final natural fusion
boundary.

### Small-T route (`2<=T<=6`)

Small-T covers target verification and MTP rebuild windows. It is one multi-column route, not `T`
serial invocations of decode. It has four graph-stable launches, with selection and assignment
grouping deliberately combined in `S2` and final reduction deliberately contained in `S4`.

#### S1. Multi-column router and shared-score projection

For every token column `t=0..T-1`, S1 computes

```text
router_logit[e,t] = dot(router_shared[e,:], X[:,t]), e=0..255
shared_score[t]   = dot(router_shared[256,:], X[:,t])
```

S1 owns one `[257,T]` logical projection and must exploit the multi-column domain rather than
serially launch a singleton projection for each token. As in decode, the logical scores have no
public representation or prescribed private dtype/layout.

#### S2. Per-token selection and complete Small-T grouping

For every token independently, S2 applies the same top-8 set, tie, selected-weight normalization,
and shared-sigmoid semantics as D2. It then forms all `A=8T` logical assignments

```text
(expert_id, token_id, route_slot, alpha)
```

and partitions them into at most 48 expert jobs. A job for expert `e` contains all and only its
`M_e` selected token/route pairs, with

```text
0 <= M_e <= T
sum(e=0..255, M_e) = 8T
```

No assignment is dropped, duplicated, or redirected. The mapping back to `(token_id, route_slot)`
is retained with its route weight. Job order and assignment order inside a job are private; logical
expert ids continue to address the persistent banks directly.

Selection and grouping are one Small-T launch. With at most 48 assignments, do not add separate
histogram, prefix-scan, assignment-scatter, host grouping, or one-launch-per-active-expert stages.
S2 may use registers and shared memory while constructing the jobs; the assignments and descriptors
needed by the separate S3/S4 launches are handed off through Op-private caller-provided workspace.
Their concrete encoding and numeric representation remain implementation choices.

#### S3. Multi-column grouped gate/up SwiGLU

For every expert job `e` and each of its assignments referring to token `t`, S3 computes

```text
act[e,t,:] = SwiGLU(routed_gu[e] * X[:,t])    # logical [512]
```

and, for every token, it computes

```text
shared_act[t,:] = SwiGLU(shared_gu * X[:,t])  # logical [512]
```

S3 is one launch covering every active routed expert and the always-on shared expert. Within an
expert job, one routed weight stream serves all `M_e` token columns; it must not fall back to one
decode-style weight stream or launch per assignment. The kernel consumes X through the assignment
token mapping and need not construct a duplicated grouped-X tensor. It reads registered routed and
shared weights in place and does not gather them.

Its logical output is one 512-element activation per routed assignment plus one per shared token.
Assignment-major, expert-major, token-major, and tiled physical representations are all permitted,
as are natural private precision and staging choices. As in D3, route scaling may be folded into
these activations when useful.

#### S4. Multi-column down projections and direct final merge

S4 consumes the grouped assignments and logical activations and completes, for every token `t` and
hidden coordinate `k`,

```text
routed[k,t] = sum(r selected for t,
                  alpha[r,t] *
                  dot(routed_down[expert_id[r,t]][k,:], act[r,t,:]))
shared[k,t] = shared_scale[t] * dot(shared_down[k,:], shared_act[t,:])
R[k,t]      = R[k,t] + routed[k,t] + shared[k,t]
```

S4 is one launch. Its work decomposition must allow a routed down row for expert `e` to serve all
`M_e` token columns in that job before advancing the weight stream. It uses the retained token
mapping to accumulate directly into the corresponding token result and finishes the shared merge
and `AddResidual` epilogue. It does not create a separate inverse-scatter kernel, route-reduction
kernel, shared-merge kernel, residual-add kernel, or persistent `[2048,8,T]` expert-output tensor.

The logical accumulation may be implemented with per-token registers, shared-memory partials, or
another natural private organization. Weighting may occur before or after a down dot product;
reduction order, intermediate precision, and final conversion to the observable BF16 destination
are not fixed by this responsibility.

#### Small-T fusion boundary

S1+S2 is the only planned launch-fusion experiment and must be compared with the separate
multi-column projection baseline. S2 already contains the tiny-domain grouping work. S2+S3 has a
global route/job dependency, while S3+S4 crosses the 2048-to-512 and 512-to-2048 contraction axes;
neither is a planned baseline fusion. S4 already contains inverse mapping, routed reduction,
shared merge, and residual writeback.

The exact CTA tiling, warp allocation, job encoding, workspace offsets, materialized activation
format, and D1+D2/S1+S2 fusion choice remain implementation decisions. They are to be selected by
complete-Op correctness against the common oracle and by isolated plus end-to-end measurements for
the registered decode and Small-T workloads.

## 2026-07-17: decode implementation outcome

> Historical implementation outcome. The later functionally complete column-serial decision below
> supersedes its `T=1` admission restriction.

Decision:

- The repository-internal interface is
  `sparse_moe(x, weights, SparseMoeEpilogue::AddResidual, destination, workspace, stream)`, with
  one `sparse_moe_workspace_bytes(max_tokens)` capacity query. At decode qualification, the route
  admitted exactly `max_tokens=1`; Small-T and prefill were rejected rather than falling back to
  repeated decode.
- Decode uses four launches: an 8-warp row-CTA router projection, one-warp register top-8 and
  selected normalization, one nine-path-warp gate/up SwiGLU launch, and one nine-path-warp
  one-output-row down/merge/`AddResidual` launch.
- D1 hands FP32 scores to D2. D2 hands I32 ids, FP32 route weights, and one FP32 shared scale to the
  expert launches. D3 hands slot-major FP32 `[9,512]` SwiGLU results directly to D4. Only D4 rounds
  the observable destination to BF16. These are selected implementation profiles, not new
  semantic rounding requirements.
- Main Text directly specializes Q4 routed gate/up and Q5/Q6 routed down; MTP directly specializes
  W8/W8. Shared gate/up and down are W8. Every selected expert id addresses its persistent bank
  span directly, with no selected-weight gather, repack, or per-expert launch.
- The D1+D2 draining-fusion experiment was rejected. Although it saved about 2 us for the Q4
  route, its completion counter required initialized workspace state before the first call. Adding
  a memset would reintroduce the launch/API work being removed, and W8+W8 had no complete-route
  gain. Production workspace therefore carries no counter or cross-call state.
- The complete Op is checked against one independent naive double-precision formula with exact
  test-only packed-weight decode. Permanent cases cover Q4+Q5, Q4+Q6, W8+W8, expert-bank endpoints,
  the lower-id boundary tie, nonzero residual, exact workspace capacity, and CUDA Graph replay;
  there are no private-stage goldens or artificial sensitivity tests.
- The accepted stage and complete-route measurements, rejected candidates, and profiler
  attribution are retained in the
  [`SparseMoe decode qualification report`](../archive/optimization-era/bench/qwen3.6-35b-sparse-moe-decode-roofline.md).

This closed ordinary decode implementation only at that phase. The previously agreed Small-T
semantics remained design input for a later implementation phase, and prefill
routing/grouping/topology remained undecided.

## 2026-07-17: functionally complete column-serial implementation

Decision:

- Before grouped kernels are optimized, `SparseMoe` accepts every positive T. The current
  implementation slices contiguous `[2048,T]` input and destination tensors by column and submits
  the same closed single-column D1-D4 route for every column on the caller's stream. There is no
  T-specific dispatch or semantic distinction inside the Op.
- Target schedule sizes such as the current prefill chunk are caller workload choices, not
  `SparseMoe` limits. They must not appear in the Op's tensor validation or workspace contract.
- Every column still evaluates the complete logical formula independently. The fallback does not
  expose or add a sub-Op, gather selected weights, launch once per expert, change routing, or add a
  numeric boundary. In particular, it preserves the decode route's natural FP32 private handoffs
  and sole final BF16 destination write.
- The calls are stream ordered and reuse one column's private workspace, so
  `sparse_moe_workspace_bytes(max_tokens)` accepts every positive `max_tokens` and currently
  returns the same capacity for all of them. The launch sequence is deterministic under CUDA Graph
  capture for a fixed input shape.
- This decision deliberately supersedes the earlier rejection of repeated decode as an interim
  *functional* route. It does not qualify that topology for Small-T or prefill performance. The
  S1-S4 grouped Small-T design and the still-undecided grouped prefill design remain the required
  optimization work; replacing the fallback must not change the public Op or oracle.
- Weight codec and token count are independent dimensions. Permanent complete-Op coverage keeps
  the three admitted codec profiles separate from routing behavior, adds an independent lower-id
  boundary-tie case, and adds one multi-column case with distinct columns and Graph replay. Every
  result is checked against the same independent naive FP64 formula, never against another
  production route.

## 2026-07-18: Small-T implementation outcome

> Current Small-T outcome. This supersedes the planned grouped `2<=T<=6` topology above and the
> column-serial route for `T=2..8`. The column-serial decision remains the functional fallback for
> `T>8`.

Decision:

- `SparseMoe` keeps the same single repository-internal API and exact mathematical oracle. `T=1`
  retains the qualified decode path, `T=2..8` selects an exact-T CUDA Core/SIMT route, and `T>8`
  continues to reuse the complete decode route one column at a time. Token count and codec profile
  remain independent dispatch dimensions.
- Small-T S1 computes all token router columns jointly. Four K partitions per router row produce
  1028 128-thread CTAs and FP32 `[T,257,4]` partial scores. Exact-T templates keep token
  accumulators compile-time sized and reuse every loaded router vector across the T columns.
- S2 uses one warp per token in one block. It reduces the four partitions and performs the same
  lower-id-stable top-8, selected-logit normalization, and shared sigmoid as decode. There is no
  cross-token routing assumption: every token owns independent ids and scales.
- S3 and S4 deliberately reuse the qualified nine-path decode kernels once per token, with a
  disjoint FP32 `[9,512]` activation region for every token. The complete route therefore uses
  `2+2T` launches rather than the fallback's `4T`. This private token loop is the measured winner
  for the registered Small-T workload; it does not create public sub-Ops or a semantic rounding
  boundary.
- Expert-grouped S3/S4, token-batched CTAs, and S1+S2 completion-counter fusion were implemented
  and profiled as candidates. Trace-like routes made grouped jobs too small and imbalanced; shared
  memory and tail-wave thresholds caused sharp T6/T7 regressions. Token batching added a partial
  CTA wave, and fusion needed workspace initialization. These candidates lost the complete-route
  comparison and were removed rather than retained as runtime branches.
- Workspace now scales through T8: I32 ids `[8T]`, FP32 route weights `[8T]`, FP32 shared scales
  `[T]`, and one FP32 scratch allocation `[9T,512]`. S1 reuses the scratch prefix for partial
  scores. `sparse_moe_workspace_bytes(max_tokens)` returns at least the T8 requirement for every
  larger positive maximum, so the graph-stable caller allocation covers both exact-T and serial
  fallback dispatch.
- Permanent correctness covers every exact template T2-T8, all three codec profiles, distinct and
  correlated per-token routes, boundary ties, nonzero residual, workspace capacity, CUDA Graph
  replay, and the T9 fallback against the independent complete oracle. A same-session three-run
  binary control confirms no T1 decode regression.
- The accepted timings, distribution sensitivity, rejected candidates, and Nsight Compute
  attribution are retained in the
  [`SparseMoe Small-T qualification report`](../archive/optimization-era/bench/qwen3.6-35b-sparse-moe-small-t.md).

This outcome qualifies MTP verification-sized `T=2..8` without tensor cores or grouped GEMM.
Larger-T prefill remains functionally supported but has no performance claim; any later grouped
implementation must beat this route at the complete-Op level rather than being admitted by
topology alone.
