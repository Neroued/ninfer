# MTP GQA Unified Small-T Attention Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this
> plan as one atomic refactor. Steps use checkbox (`- [ ]`) syntax for tracking. Do not use
> subagent-driven development for implementation.

**Goal:** Replace the current split public GQA attention API with one general `gqa_attention(...)`
entry point, adapt the underlying GQA implementation to MTP's device-position append semantics, and
add a dedicated MTP-required small-T coverage path that preserves T=1 decode performance before any
T>1 routing is optimized or treated as permanent.

## Execution Mode

Implementation mode: **single main-agent refactor**.

Reason: this work intentionally crosses the public L1 kernel API, launcher/kernel layering,
runtime/model call sites, tests, benches, and MTP graphability contracts. There will be a period
where old API names have been removed but all call sites and tests are not yet updated. Subagent-
driven implementation is intentionally not used.

Final review mode: after the atomic refactor builds and the targeted tests/benchmarks have run,
perform one comprehensive self-review. Do not perform intermediate subagent reviews.

## User Constraints

- No subagent-driven implementation.
- MTP GQA and CUDA Graph must be compatible: all step/position semantics needed inside a round must
  be device-owned.
- Do not keep two public GQA attention entry points.
- Use a general API name. The public API name is `gqa_attention`.
- T=1 performance is the first gate. T>1 work cannot land until the unified T=1 path has no
  measurable regression versus the current decode path.
- Small-T must be driven to the hardware memory-bandwidth limit **on useful KV bytes**, not on raw
  DRAM throughput. A kernel that re-streams the KV cache once per query token can saturate
  `dram__throughput` while doing several times the necessary work; that is a failure, not a pass.
  The gate is `ncu`-measured useful-KV bandwidth plus a redundancy check (`total_dram / useful_kv`
  close to 1), not printed bench GB/s and not `dram__throughput` alone.
- Prefer one bottom kernel family, controlled by template parameters/policies, over independent
  duplicate kernels. Do not add template parameters or policy branches for cases that no call site
  exercises (per `AGENTS.md`: no abstractions for hypothetical reuse).
- Do not hard-code a small-T/large-T performance crossover without measured evidence. The current
  `T=1..6` range is only the MTP functional range implied by `kMaxMtpDraftTokens == 5`, not a
  measured attention regime split.
- The route may only key on host-known inputs. Under the CUDA Graph / device-position contract the
  wrapper does not read `positions` (context length) back to host, so the route selected at launch
  can depend on host-known `T` and static capacity bounds only, never on device-resident context
  length. Any context-dependent behavior must be an internal device-side branch, not a host route
  choice. This is why the measured crossover resolves to a per-`T` route (see Regime Model).

## Non-Goals

- No round-level MTP controller rewrite in this plan.
- No CUDA Graph capture harness changes beyond making the GQA op graph-compatible.
- No logits/top1 or MTP hidden-output contract cleanup.
- No broad model refactor outside the full-attention call sites.
- No compatibility wrappers named `gqa_attention_prefill` or `gqa_attention_decode` in the public
  header after migration.

## Final Public Contract

Replace:

```cpp
void gqa_attention_prefill(..., std::uint32_t cache_offset, ...);
void gqa_attention_decode(..., const Tensor& pos, ...);
```

with one public entry:

```cpp
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& positions, float scale,
                   KVCache& kv, int layer, WorkspaceArena& ws,
                   Tensor& out, cudaStream_t stream);
```

Shape contract:

```text
q:         BF16 [256,24,T]
k:         BF16 [256, 4,T]
v:         BF16 [256, 4,T]
positions: I32  [T]
out:       BF16 [256,24,T]
T >= 1
```

Append semantics:

```text
for token i in 0..T-1:
    p_i = positions[i]
    write k[:,:,i], v[:,:,i] to cache position p_i
    out[:,:,i] = attention(q[:,:,i], cache positions 0..p_i)
```

Additional MTP/contiguous append requirement:

```text
For T > 1, positions must be contiguous:
    positions[i] == positions[0] + i
```

The wrapper validates shape, dtype, contiguity, cache layout, scale, and host-known `T <= max_ctx`
capacity. It does not read position values back to host. The caller owns the device contract that
every `positions[i]` is in range and contiguous for `T > 1`; kernel code reads `positions` from
device memory.

`KVCache::pos` is not read or advanced by this API. The caller owns logical commit/mirror state.

## Regime Model (what actually determines the crossover)

The crossover is not a free tuning knob. It is a consequence of how each kernel family maps to the
RTX 5090 (≈1792 GB/s DRAM peak, ≈209.5 dense-BF16-TC TFLOP/s per this repo's bench constants) at a
given `(T, context)`. Any threshold must be justified against this model, not asserted.

Two kernel families with opposite scaling:

- Split-K streaming (the current decode family). It splits the key/context axis into many CTAs so a
  batch-1, few-query workload still fills the GPU. Each CTA streams its key slice from DRAM once with
  CUDA-core FMA (no tensor cores) and runs online softmax. It is the right family when the query set
  is small relative to context: parallelism comes from splitting K, and the work is dominated by
  streaming the KV cache once. Its costs grow with the number of splits (partial/reduce scratch
  traffic) and, if generalized naively, with `T` (see below).
- Flash Q/K tiling (the current prefill family). It tiles Q into `Br=64`-row blocks and reuses each
  shared-memory K/V tile across all 64 rows via tensor cores, so arithmetic intensity rises with the
  number of live query rows and the kernel becomes compute-bound. It is the right family when there
  are many query rows to amortize each K/V tile load.

Why small-`T` is the problem the current code has: MTP verify/shifted append (`T = k+1 <= 6`)
currently runs on the flash prefill kernel. At long context that kernel launches only
`ceil(T/64)*24 ≈ 24` CTAs (one Q-block, all heads), so it cannot fill the SMs and is DRAM-starved;
worse, a 64-row Q tile carries only `<=6` valid rows, so ~90% of its tensor-core work is masked
waste. It still streams the whole KV cache once, but at low occupancy with wasted compute. That is
the concrete inefficiency this plan removes.

The correct small-`T` shape follows directly from the decode family: keep the split-K geometry (so
occupancy stays high at long context), but inside each CTA stream each key slice **once** and apply
it to **all `T` query tokens** of the KV group (independent online-softmax state per token). This
reads useful KV bytes once regardless of `T` while doing `T`x the (cheap, memory-hidden) FMA work.
It is the memory-bound optimum until per-token register/state pressure forces occupancy down.

Crossover consequences that the routing must respect:

- The physically-correct route depends on **both** `T` and context length. At tiny context the flash
  family can win (KV fits in cache, split-K scratch dominates); at long context split-K-with-token-
  tile wins decisively for the MTP `T` range.
- But context length is device-resident under the graph contract, so the launch-time route may key
  on host-known `T` only. Therefore the measured crossover must be resolved into a **per-`T` route**
  that is chosen for the operating regime that matters (long-context decode/verify), and validated to
  not lose across the context sweep. It must not be a route that only wins at contexts the model
  rarely hits.

## Internal Architecture

Keep one public API. Internal routing keys on host-known `T` only, and every route entry is backed by
measured evidence, never by an unjustified constant:

```text
gqa_attention(...)
    T == 1:
        detail::gqa_attention_small_t_launch(... TokenTile=1 ...)   // decode-derived, no-regression
    T > 1:
        detail::gqa_attention_launch(... per-T route from the measured route table ...)
```

The route table is internal only; it is not visible as separate public names, and it is indexed by
host-known `T` (plus static capacity), never by device context. The only route fixed at plan start is
that `T==1` uses the decode-derived no-regression path. Every `T>1` entry — including whether a given
`T` uses the split-K token-tile family or the flash prompt-scale family — is added only after
correctness and `ncu` evidence per the Regime Model.

Initial T coverage:

```text
T=1:
    mandatory no-regression decode replacement (split-K, TokenTile=1).

T=2..6:
    mandatory functional coverage for current MTP because max draft k is 5 and verify/shifted
    append uses T=k+1. This is a required correctness range, not a performance threshold. The
    working hypothesis (to be proven, not assumed) is that this whole range routes to the split-K
    token-tile family because MTP verify matters at long context where flash is DRAM-starved.

T>6:
    future larger-MTP / prompt-scale coverage. Route is profiling-derived per T, with no assumption
    that "T>6 == flash".
```

The eventual crossover is named `gqa_attention_policy_threshold` in benchmark reports, not in the
public API. It is accepted only with saved `ncu` artifacts and a comparison table over at least
`T=1,2,3,4,5,6,8,16,32` **crossed with a context sweep** (see Small-T DRAM Roofline Gate), so that
the per-`T` route is justified in the regime the model actually runs.

The bottom small-T kernel family starts decode-derived:

```cpp
template <int QHeadsPerCta, int TokenTile>
__global__ void gqa_attention_small_t_partial_kernel(...);

template <int DChunk>
__global__ void gqa_attention_small_t_reduce_kernel(...);
```

Signature notes (correcting the earlier draft):

- There is no `WriteKv` template parameter. Append semantics always write `k/v` into the cache, and
  no call site attends without also writing. The KV write is a **runtime once-guard** exactly as the
  decode kernel already does today (the split/warp that owns the position writes; everyone else only
  reads). A compile-time `WriteKv` would be dead genericity and is forbidden by `AGENTS.md`.
- The reduce kernel is **not** templated on `TokenTile`. It reduces `T * splits` partials per query
  head, so `T` is a runtime loop/grid bound, matching the current `<int DChunk>`-only decode reducer.
  Templating it on `T` only multiplies instantiations for no benefit.

`TokenTile == 1` is the current decode computation generalized with the same launch geometry, same
split-K decomposition, same runtime K/V write guard, same partial/reduce layout for one token, and
same online-softmax math. It must compile/emit to the same work as today's decode kernel; this is the
T=1 no-regression baseline.

`TokenTile > 1` is the memory-bound target of this plan, not merely one candidate among equals: it is
the only shape that reads useful KV bytes once regardless of `T` (Regime Model). It is accepted when
`ncu` confirms it reaches the useful-KV roofline for the MTP `T`/context regime without harming T=1.
The token-independent generalization below (Candidate A) exists only as a correctness scaffold, since
it re-reads KV per token and therefore cannot meet the useful-KV gate at long context.

## Small-T Design Space

This is a measured search, but the candidates are **not** peers. Candidate A is a correctness
scaffold. Candidate B is the intended memory-bound design. Candidates C–E are constrained tuning
axes, several with a physically-predictable sign that the plan states up front so the search does not
waste `ncu` runs re-discovering it.

Candidate A: token-independent decode generalization (correctness scaffold only).

```text
grid.x = kv_head * q_subgroup
grid.y = split
grid.z = token_i          // one independent decode-style solve per token
```

Properties:

- Closest to the current decode path; lowest integration risk; good first step to lock in the CPU
  oracle and append/masking semantics for `T>1`.
- **Re-reads the KV cache once per token (`T`x useful-KV traffic).** At long context (KV cache >
  last-level cache) these re-reads hit DRAM, so this candidate is `~T`x slower than the optimum and
  **cannot pass the useful-KV gate**. It is explicitly not a performance baseline and must not be the
  shipped kernel for any `T>1` route. Keep it only until Candidate B is correct, then it may be
  deleted (no back-compat, per `AGENTS.md`).

Candidate B: token-tile, single-KV-pass (the intended design).

Grid (same split-K geometry as decode, so occupancy at long context is preserved):

```text
grid.x = kv_head * q_subgroup
grid.y = split
grid.z = ceil(T / TokenTile)   // with TokenTile == T this is 1 => grid == decode grid
```

For each CTA:

```text
token range = [tile_start, min(tile_start + TokenTile, T))
kv split    = [split_start, split_end)   // sized from the largest window in the tile (see below)
q heads     = one KV group's query-head subgroup
```

Inner loop: for each key position in the CTA's split slice, load `k`/`v` **once**, then update the
independent online-softmax state of every (query head in group x token in tile) it is causal for.
This is the decode inner loop with the per-key work multiplied over the token tile.

Properties:

- Reads useful KV bytes once regardless of `T`; extra cost is `T`x cheap FMA that the memory latency
  hides until register/state pressure lowers occupancy.
- Keeps independent online-softmax state per (query head, token).
- Register/shared pressure grows with `TokenTile` (and with `QHeadsPerCta`); the interaction with
  occupancy is the real thing to measure.
- Split window sizing: the CTA's key range must cover the **largest** causal window in the tile,
  i.e. up to `positions[tile_end-1] + 1`; per-token masking then trims shorter windows. Do not size
  the split from `positions[0]` alone.

Candidate C: q-head grouping (`QHeadsPerCta`), with a predicted sign.

```text
QHeadsPerCta = 6 (baseline)   ; 3 or 2 only under proven register pressure
```

`QHeadsPerCta == 6` means one CTA owns the whole KV group, so each KV head's `k`/`v` is streamed once
and shared across all 6 query heads. Dropping to `3` or `2` creates 2x/3x KV re-reads of the same KV
head across subgroups — the opposite of the bandwidth goal. Therefore `6` is the default and the only
value expected to win in the memory-bound regime. `3`/`2` are considered **only if** Candidate B at
`QHeadsPerCta=6` is register/occupancy limited *and* `ncu` shows the occupancy recovery outweighs the
added KV traffic. Treat a smaller group as a fallback, not a co-equal candidate.

Candidate D: partial/reduce layout.

Logical shape and physical stride order (d fastest), chosen so the `T=1` case aliases decode exactly:

```text
partial_acc BF16 [256, 24, T, splits]   // index: d + 256*(q_head + 24*(token + T*split))
partial_m   FP32 [     24, T, splits]
partial_l   FP32 [     24, T, splits]
```

With `T=1` the `token` axis collapses and the index reduces to `d + 256*(q_head + 24*split)`, which
is byte-identical to the current decode `gqa_partial_acc_index`. Token-major vs split-major variants
may be measured, but any variant must (a) keep `d` innermost for coalesced partial writes and (b)
preserve the exact `T=1` decode aliasing. Note that scratch traffic scales with `splits * T` and can
dominate useful KV at MTP contexts — this is a Candidate F concern, not just a layout detail.

Candidate E: new-row source and cache-write guard (a required generalization, not an option).

The current decode kernel **already** reads the newest token from the input `k`/`v` (not the cache)
and writes that token to the cache exactly once from the split that owns its position
(`split_start <= p < split_end`, `q_subgroup == 0 && local_q == 0`). For `T>1` this must be
generalized, not reinvented, and the earlier draft's `if split == 0` write guard is wrong: split 0
owns the *first* keys of the context, not the newly appended positions.

Cache write (generalize the decode guard; the owner of each new position writes it):

```text
for each token in the tile:
    p_tok = positions[token]                       // = positions[0] + token (contiguous)
    if q_subgroup == 0 and local_q == 0 and split owns p_tok (split_start <= p_tok < split_end):
        cache_k/v[p_tok] = k/v[:,:,token]
```

Attention read rule (per query token `query_token_i`, per key position `key_p` in the split):

```text
if key_p < positions[0]:
    read cache key/value
else:
    new_i = key_p - positions[0]
    if 0 <= new_i <= query_token_i:
        read k/v input at new_i        // resident this launch; no cross-CTA ordering dependence
    else:
        mask as future / invalid
```

This preserves the decode invariant (attend to freshly-appended rows from the input, write them to
cache once for later ops) and extends it to the token tile without relying on inter-CTA ordering.

Candidate F: split count vs scratch vs parallelism (the axis the earlier draft omitted).

The decode kernel uses a fixed `kGqaDecodeSplits = 192`, chosen so a batch-1 decode still fills the
SMs at long context. That choice is not free: the partial/reduce scratch write+read is
`~2 * splits * 24 * 256 * 2` bytes per token (`>= ~4.7 MB` at 192 splits), and it scales with `T`.
The bench already models this as `scratch` bytes separately from `useful_kv`. Consequences to
measure:

- At short/moderate MTP context, `splits=192` makes each split own only a few keys, so launch and
  scratch traffic — not KV — dominate. Fewer splits may be faster there.
- At long context, more splits buy occupancy and are worth their scratch.
- Because context is device-resident (not a host route input), the split count must be a host-known
  choice: either a single value validated to win across the context sweep for each `T`, or a
  device-side adaptive scheme internal to the kernel. A host branch on device context is not allowed.
- If scratch dominates, evaluate a lower split count and/or whether the separate reduce pass can be
  avoided for the small-context regime; do not assume the 192-split two-pass structure is optimal
  outside the long-context T=1 decode it was tuned for.

Partial buffer allocation follows Candidate D: one rank-4 view `[256,24,T,splits]` for `acc` and
`[24,T,splits]` for `m`/`l`, whose `T=1` slice is byte-identical to today's decode
`[256,24,splits]` / `[24,splits]`. The T=1 path must not grow the workspace byte count or add inner-
loop index math versus decode after compiler optimization; verify with SASS/`ncu` before enabling
`T>1`.

## Large-T Policy

The existing prefill flash-style kernel becomes the internal prompt-scale policy under the unified
API. It must stop consuming host `cache_offset`.

Required changes:

- Launcher passes `positions.data` instead of `cache_offset`.
- Fill kernel writes cache at `positions[token]`.
- Attention kernel computes query absolute position from `positions[qrow]`.
- For the existing contiguous large-T prefill call sites, `positions` is already created on device.
- The wrapper does not read positions to host.

This policy can retain its current two-launch fill + attention structure. This plan is not a
prompt-scale performance rewrite. It is **not** the default for any particular `T`: it is selected
per-`T` only where the profile sweep shows it beats the split-K token-tile family in the operating
context regime (Regime Model). In particular, "`T>6` uses flash" is a hypothesis to test, not a
given — at long context the flash family is DRAM-starved for small query sets.

## T=1 No-Regression Gate

This gate is mandatory before extending or routing T>1.

- [ ] Introduce `gqa_attention(...)` with `T==1` routed to a mechanically generalized copy of the
      current decode partial/reduce path.
- [ ] Preserve T=1 launch count: partial + reduce only.
- [ ] Preserve T=1 grid geometry, split count, `QHeadsPerCta`, block size, and workspace byte count.
- [ ] Preserve decode math and tie behavior bitwise within existing test tolerances.
- [ ] Update model decode call sites to use `gqa_attention(...)`.
- [ ] Run `qus_gqa_attention_test`.
- [ ] Run `qus_gqa_attention_bench --decode` or the updated equivalent.
- [ ] Capture `ncu` for the T=1 partial kernel on a long-context cold-cache or 16-layer round-robin
      case.
- [ ] Compare against the current decode path. If T=1 regresses beyond measurement noise, stop and
      fix before implementing T>1.

The intended T=1 implementation is not "new attention"; it is the current decode kernel factored
into the unified kernel family with `TokenTile=1`.

## T>1 Correctness Gate

After T=1 passes:

- [ ] Bring up `T=2,3,4,5,6` correctness on Candidate A (token-independent scaffold) first, since it
      is the least risky place to pin down append + causal-masking semantics against the oracle.
- [ ] Add CPU oracle tests for:
      `T=1,2,3,4,5,6`, bases `0,1,17,128`, and at least one near-capacity base.
- [ ] Verify cache writes exactly at `positions[i]`.
- [ ] Verify row `i` attends to `0..positions[i]`.
- [ ] Verify future-token isolation: changing `k/v` for rows `j > i` does not change output row `i`.
- [ ] Verify wrapper does not read or mutate `KVCache::pos`.
- [ ] Verify the same captured graph can be relaunched with changed device `positions`.

Only after this correctness gate is green should Candidate B (the token-tile single-KV-pass kernel)
be brought up and optimized. Candidate A is a scaffold: once Candidate B is correct for a route, the
scaffold is deleted rather than kept as a fallback (no back-compat, per `AGENTS.md`).

## Small-T DRAM Roofline Gate

Small-T optimization is a measured search over the design space above. The hypothesis — that reading
streamed KV **once** and applying it to all `T` query tokens reaches the useful-KV roofline for `T>1`
— must be proven or rejected with `ncu`, and the gate must measure *useful* work, not raw DRAM
throughput (a `T`x-re-reading kernel can hit 85% `dram__throughput` while being `T`x too slow).

Implementation rounds:

- [ ] Use Candidate A only to lock the CPU oracle and append/masking semantics for `T=2..6`. Do not
      treat it as a performance baseline or ship it for any route.
- [ ] Implement Candidate B (`TokenTile` single-KV-pass), starting with `TokenTile == T` for the MTP
      range (`T=2..6`) so each split reads its KV slice once for the whole tile.
- [ ] Add Candidate C (`QHeadsPerCta < 6`) only if Candidate B at `6` is proven register/occupancy
      limited by `ncu`; measure whether occupancy recovery beats the added KV re-reads.
- [ ] Sweep the split count (Candidate F) per `T`; find the value that wins across the context sweep
      or justify a device-side adaptive scheme.
- [ ] Profile the route candidates over `T = 1,2,3,4,5,6,8,16,32` **crossed with** context lengths
      that match the model's real decode range (e.g. `2048, 8192, 32768`, matching the decode bench
      positions), because the winning family depends on context. Do not set a crossover before these
      profiles exist.
- [ ] Record the chosen per-`T` route table in the benchmark report; keep the public API independent
      of that table.
- [ ] For each variant/(T,context), report: register count, achieved occupancy, split count, partial
      scratch bytes, useful-KV bytes, total DRAM bytes, `total_dram/useful_kv` redundancy ratio,
      useful-KV GB/s, and correctness. Reuse the bench's existing `useful_kv`/`scratch`/`total` model.
- [ ] Use cache K/V vectorized loads shared across all tokens in the tile; keep per-token softmax
      state independent; never re-read appended `k/v` from global when it is resident in input.
- [ ] Tune with `ncu` Memory Workload / Speed-of-Light sections; do not steer by stdout GB/s.

Acceptance (per selected memory-bound route, at the MTP operating context regime):

- [ ] `ncu` evidence saved under `profiles/ncu-gqa-smallt/` (`.ncu-rep` per (T, context)).
- [ ] **Useful-KV gate:** effective useful-KV bandwidth `>= 85%` of the copy-kernel ceiling recorded
      per `docs/l1-op-test-standard.md` §2.3 (equivalently `>= 85%` of DRAM peak when the copy-kernel
      baseline is near peak). `dram__throughput` alone is reported but is not the pass condition.
- [ ] **Redundancy gate:** `total_dram / useful_kv <= 1.10` (allow a small margin for `q`/`out`/
      scratch). This is what actually rejects Candidate A: a per-token re-reading kernel has a
      redundancy ratio near `T`, so it fails here even if it saturates `dram__throughput`.
- [ ] **Scratch gate:** partial/reduce scratch bytes do not exceed useful-KV bytes at the profiled
      context; if they do, the split count / two-pass structure must be revisited (Candidate F).
- [ ] **Regression gate vs the path being replaced:** wall-clock median for the MTP `T=2..6` verify
      shape must beat the current flash-prefill verify path at the same (T, long context). Replacing
      a correct kernel with a slower one is not acceptable even if it passes the roofline.
- [ ] Useful-KV and total DRAM bandwidth are reported separately in the bench and the report.
- [ ] Hot-cache single-layer timing is diagnostic only; HBM claims require cold-cache or 16-layer
      round-robin evidence.
- [ ] If the kernel stops being DRAM-bound before the useful-KV gate, document the new limiter with
      `ncu` evidence and keep tuning until either useful-KV is saturated or the remaining limiter
      needs a separate architecture plan.

## Call-Site Migration

- [ ] Replace `kernels::gqa_attention_decode(...)` calls in model decode/MTP AR with
      `kernels::gqa_attention(...)`.
- [ ] Replace `kernels::gqa_attention_prefill(...)` calls in prefill/verify/MTP shifted paths with
      `kernels::gqa_attention(...)`.
- [ ] Thread `positions` through full-attention call sites instead of host `cache_offset`.
- [ ] Remove host `cache_offset` from GQA-facing model internals where GQA was the last consumer.
- [ ] Keep any non-GQA host-offset cleanup that belongs to round-level MTP in the later round
      contract plan.

## File-Level Work Plan

- [ ] Modify `include/qus/kernels/gqa_attention.h`: expose only `gqa_attention(...)`.
- [ ] Modify `src/kernels/launcher/gqa_attention.h`: declare the internal small-T launch
      (`gqa_attention_small_t_launch`), the prompt-scale/flash launch, and the per-`T` route
      dispatcher (`gqa_attention_launch`) that keys on host-known `T` only.
- [ ] Refactor `src/kernels/kernel/gqa_attention_decode.cuh` into the small-T kernel family.
- [ ] Refactor `src/kernels/launcher/gqa_attention_decode.cu` into `gqa_attention_small_t` launch
      code or rename the file if the build system expects source-level clarity.
- [ ] Refactor `src/kernels/kernel/gqa_attention_prefill.cuh` to consume device `positions` for the
      large-T policy.
- [ ] Refactor `src/kernels/launcher/gqa_attention_prefill.cu` into prompt-scale launch code or
      rename the file if needed.
- [ ] Update `src/kernels/wrapper/gqa_attention.cpp` to validate the single API and dispatch by T.
- [ ] Update `src/model/qwen3_6_27b.cpp` full-attention call sites.
- [ ] Update `tests/kernels/test_gqa_attention.cpp`.
- [ ] Update `bench/gqa_attention_bench.cu` to drive the unified API with
      `T=1,2,3,4,5,6,8,16,32` and prompt-scale T. The small-T/append mode must take a **context
      length** argument (analogous to `--decode-pos`) so the (T, context) sweep required by the
      roofline gate is possible, and must extend the existing `useful_kv`/`scratch`/`total` byte
      model (and print the `total_dram/useful_kv` redundancy ratio) to `T>1`.
- [ ] Update docs that name the removed public APIs.

## Verification Commands

Build:

```bash
cmake --build build --target qus_gqa_attention_test qus_gqa_attention_bench qus -j
```

Correctness:

```bash
./build/tests/qus_gqa_attention_test
```

Bench smoke (small-T append mode takes both `--tokens T` and a context length):

```bash
./build/bench/qus_gqa_attention_bench --decode
# sweep T x context so the routing decision has data (context flag name TBD by the bench update):
for T in 1 2 3 4 5 6 8 16; do
  for CTX in 2048 8192 32768; do
    ./build/bench/qus_gqa_attention_bench --append-small-t --tokens "$T" --context "$CTX"
  done
done
```

Profile gate examples (capture each (T, context) cold-cache and read useful-KV, not just DRAM %):

```bash
mkdir -p profiles/ncu-gqa-smallt
# T=1, long context, cold cache -> confirms the no-regression decode-derived path
ncu --force-overwrite --set roofline \
    -o profiles/ncu-gqa-smallt/t1_ctx32768_cold \
    ./build/bench/qus_gqa_attention_bench --append-small-t --tokens 1 --context 32768 \
      --profile-once --cold-cache

# T=6, long context, cold cache -> the token-tile single-KV-pass kernel under the useful-KV gate
ncu --force-overwrite --set roofline \
    -o profiles/ncu-gqa-smallt/t6_ctx32768_cold \
    ./build/bench/qus_gqa_attention_bench --append-small-t --tokens 6 --context 32768 \
      --profile-once --cold-cache

# read the metric that actually gates, alongside dram throughput for context:
ncu --import profiles/ncu-gqa-smallt/t6_ctx32768_cold.ncu-rep --csv \
  | grep -iE 'dram__bytes.sum|dram__throughput.avg.pct_of_peak_sustained_elapsed'
```

Exact bench flags may change during the bench update, but final commands must target isolated
single-kernel captures per (T, context), save `.ncu-rep` under `profiles/ncu-gqa-smallt/`, and the
accepted evidence is the useful-KV bandwidth and the `total_dram/useful_kv` redundancy ratio, with
`dram__throughput` recorded for context only.

## Completion Criteria

- One public GQA API remains: `gqa_attention(...)`.
- Existing public `gqa_attention_prefill` and `gqa_attention_decode` declarations are removed.
- T=1 unified path has no measurable performance regression and no extra launch/workspace overhead.
- `T=2,3,4,5,6` pass append semantic tests and future-token isolation tests.
- MTP/verify full-attention call sites no longer pass host `cache_offset` into GQA.
- GQA kernels read positions from device memory.
- The per-`T` internal route table is backed by `ncu` evidence over a (T, context) sweep; no route is
  justified by a single `T` value or a single context, and the route keys only on host-known `T`.
- Small-T tuning has `ncu` evidence and, for each selected memory-bound route, meets the useful-KV
  gate with a `total_dram/useful_kv` redundancy ratio near 1 (so a per-token re-reading kernel cannot
  pass), or documents a new non-DRAM limiter with enough evidence to justify the next architecture
  plan.
- The selected `T=2..6` route beats the current flash-prefill verify path in wall-clock at long
  context; the refactor does not regress the operation it replaces.
