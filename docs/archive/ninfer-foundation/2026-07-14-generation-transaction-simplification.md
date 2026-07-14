# Generation transaction simplification

> Status: implemented on 2026-07-14.

## 1. Purpose

Simplify the synchronous single-request generation transaction without changing model math,
artifact loading, product features, or the exact-target architecture.

The current implementation expresses one required invariant through several overlapping protocols:

```text
Program PendingRound with epoch/callback dispatch
  + decoder StagedText -> OutputResolution -> DecoderCommitPlan with another epoch
  + request GenerationGuard and publication-failure invalidation
```

Only the invariant is required:

> For each generated round, Program and output decoding commit the same accepted token prefix before
> any output is published.

This cutover replaces the overlapping protocols with one pending Program state, one decoder preview,
one small decision value, and one request-level abort guard.

## 2. Scope

This is one atomic internal replacement. It changes:

- the common generated-round contract;
- the exact target Program facade and its pending-round resolution;
- the exact target OutputSession preview/commit API;
- the common generation controller;
- the target-private activation dump driver;
- focused tests and the active engine documentation.

It does not change:

- `.ninfer`, Reader/Binder/Materializer, target selection, or target load ownership;
- Qwen3.6-27B Text, Vision, MTP, sampling, KV, GDN, prefix, or CUDA Graph math;
- tokenizer, chat template, stop-string, UTF-8, reasoning/content, or streaming behavior;
- the public `Engine`, `PreparedPrompt`, `GenerationResult`, serving, CLI, or benchmark contracts;
- the current one-active-request execution model;
- future continuous batching design.

There is no compatibility layer. Old internal transaction types and their tests are deleted in the
same change.

## 3. Required semantics

The replacement must preserve these facts:

1. `begin` and `decode_round` may provisionally mutate target-private GPU and host state and expose
   one non-empty batch of licensed token IDs.
2. A continuing round accepts the complete returned batch.
3. A terminal round accepts an exact non-empty prefix. A target may make the resulting sequence
   resident or non-reusable; the controller does not need that distinction.
4. An unresolved or failed request makes its current Program sequence non-reusable.
5. The output decoder previews without modifying committed decoder state.
6. UTF-8 tails, reasoning transitions, stop-string tails, stop-token policy, and output channels are
   resolved against the exact accepted token prefix.
7. Program resolution occurs before decoder preview commit, budget charge, and output publication.
8. Output publication is not rollbackable. A synchronous sink failure propagates and the request
   guard makes the resulting Program sequence non-reusable.
9. Cancellation before model work performs no execution. Cancellation between rounds may retain the
   coherent committed prefix; cancellation after provisional work discards that provisional round.

## 4. Replacement contract

### 4.1 Common values

The common runtime keeps only small values needed by the controller:

```cpp
struct GeneratedRound {
    std::span<const TokenId> tokens;
};

struct BeginResult {
    BeginSummary summary;
    GeneratedRound round;
};

struct OutputDecision {
    std::uint32_t accepted_tokens;
    FinishReason finish_reason; // None means continue

    bool finished() const noexcept { return finish_reason != FinishReason::None; }
};
```

`GeneratedRound` is a synchronous view into Program-owned stable host storage. It never escapes the
controller iteration. It has no owner pointer, epoch, callback table, destructor action, or dynamic
allocation.

Remove from the common contract:

- `PendingRound` and its callback table;
- `StagedChoiceId`;
- `OutputResolution` and `RoundContinuation`;
- `FinishDisposition`;
- `SequenceSummary` and common `ProgramState`;
- unused fields in `GenerationSummary`.

`RequestPlanSummary`, `BeginSummary`, `RoundBudget`, and generation timing/result values remain.

### 4.2 Program

The target facade becomes conceptually:

```cpp
RequestPlan plan_request(const PreparedPrompt&, const ExecutionOptions&) const;
BeginResult begin(PreparedPrompt&&, RequestPlan&&, TransientRegion);
GeneratedRound decode_round(RoundBudget);
void resolve_pending(std::uint32_t accepted_tokens, bool terminal);
void finish_active();
void abort_request() noexcept;
```

The Program keeps its target-private lifecycle and provisional data. `resolve_pending` enforces:

- Program must currently be pending;
- `1 <= accepted_tokens <= produced`;
- Continue requires `accepted_tokens == produced` and establishes Active;
- terminal full acceptance establishes Resident;
- the current Qwen MTP partial-terminal policy accepts the logical prefix and makes the sequence
  non-reusable, exactly as before.

The following machinery is removed:

- Program and pending-candidate epochs;
- `RequestPlan::expected_epoch`;
- `is_live`, callback thunks, and stale-handle checks;
- `commit_all`, `commit_prefix_and_finish`, and `discard` facade variants;
- unused pending fields `accepted_drafts` and `resulting_gdn_slot`;
- unused `state()` and `clear_resident()` facade methods.

The diagnostic tool receives a narrow `materialized_tokens()` query instead of the common sequence
summary.

### 4.3 OutputSession

The target output facade becomes conceptually:

```cpp
OutputDecision preview(std::span<const TokenId> tokens,
                       std::uint32_t budget_remaining,
                       FinishReason limit_reason);
OutputDecision preview_terminal(FinishReason reason);
PublishedOutput commit_preview() noexcept;
```

OutputSession owns reusable request-local scratch:

- one working/next `DecoderState`;
- one selected `PublishedOutput`;
- selected stop metadata;
- a boolean indicating that a preview is ready.

`preview` scans the returned tokens once and selects the one real outcome:

1. earliest token boundary wins;
2. at the same boundary, stop string wins over stop token and the budget limit;
3. among stop strings, earliest decoded byte cut wins, then declaration order;
4. stop token wins over the budget limit at the same boundary;
5. the budget limit is considered only at the complete returned batch and uses the single
   `RequestPlan`-selected reason;
6. otherwise the complete batch continues.

The decoder retains only the selected next state and output. It does not construct a lattice of
Continue/OutputLimit/ContextCapacity/stop candidates for every token.

Remove:

- `StagedCandidateSummary` and `StagedText`;
- `choices` and `summaries` vectors;
- `StagedChoiceId` and `choice_for`;
- `DecoderCommitPlan`;
- output epochs and stale-preview checks;
- both per-round PIMPL allocations.

### 4.4 Controller

The steady-state loop is:

```text
Program returns GeneratedRound
  -> OutputSession previews one OutputDecision
  -> Program resolves the same accepted token count
  -> OutputSession commits its preview with a no-fail swap
  -> budget is charged
  -> output is published
```

One request-level `GenerationGuard` remains. It is armed after `begin` establishes provisional
state, disarmed only after normal completion, and calls `abort_request()` on any escaping exception.

Cancellation keeps three meaningful boundaries only:

- before `begin`: return Cancelled without Program mutation;
- after a provisional round: abort that Program request, terminalize the already committed decoder
  tail, and return Cancelled with a non-reusable Program sequence;
- between committed rounds: call `finish_active`, terminalize decoder state, and retain the coherent
  resident prefix.

### 4.5 PreparedPrompt identity

Remove the process-wide never-reused `ProductCookie` allocator and exhaustion policy. The prepared
value is owning, and the closed variant already enforces target type. A prepared value for the same
exact target may be consumed by another Engine instance; the receiving Program still validates
token domain, metadata shape, and its own capacity. Different target types remain rejected by the
variant.

## 5. File changes

Expected removals:

- `src/runtime/contract/pending_round.h`;
- `src/runtime/generation/output_resolution.h`;
- `src/runtime/engine/product_cookie.h/.cpp`;
- `tests/test_runtime_pending_round.cpp`;
- `tests/test_output_resolution.cpp`.

Expected modifications:

- `src/runtime/contract/types.h`;
- `src/runtime/generation/generation_controller.h` and `generation_guard.h`;
- `src/runtime/engine/engine.cpp`;
- `src/targets/registry.h/.cpp`;
- target package export, package facade, Program, request plan, Frontend, and activation dump;
- focused frontend/controller tests and CMake source lists;
- `docs/design.md`, `docs/ninfer-engine-architecture.md`, and test documentation.

## 6. Verification

Verification protects current behavior rather than the removed implementation shape.

### 6.1 Focused host tests

- Adapt target frontend tests to `preview/commit_preview` and retain cases for ordinary continuation,
  output/context limit, stop token publication policy, stop-string priority, cross-token UTF-8,
  reasoning/content transition, and terminal flush.
- Add one small controller test with a fake Program and OutputSession covering:
  - a continuing round followed by a terminal budget round;
  - terminal acceptance of a strict prefix of a multi-token round;
  - request cleanup when output resolution or publication throws;
  - cancellation before work, after provisional work, and between rounds only if those paths are not
    already exercised through the same compact fixture.
- Delete stale-handle/epoch and candidate-ID tests. They protect removed internal misuse scenarios.

### 6.2 Build and real product checks

- Build the affected runtime, target, CLI, benchmark, dump tool, and tests.
- Run the focused controller and frontend tests plus the existing real prefix test.
- Run one short ordinary Text request and one short k=3 MTP request through the public Engine/CLI.
- Run one streaming request if the existing local smoke tool makes that direct and inexpensive.

No numerical parity or performance-regression claim is made because model math and kernels are not
changed. The final review confirms that the new round/output path contains no `StagedText`,
`DecoderCommitPlan`, callback thunk, transaction epoch, or explicit per-round PIMPL allocation.

## 7. Completion conditions

The cutover is complete when:

- the old transaction headers, implementations, tests, and CMake entries are deleted;
- all product and diagnostic callers use the new contract;
- Program and decoder still commit the same exact token prefix;
- current stop/UTF-8/reasoning/streaming behavior passes focused tests;
- ordinary and MTP public product requests run successfully;
- active architecture documentation describes only the implemented simplified contract;
- no compatibility branch or duplicate transaction route remains.
