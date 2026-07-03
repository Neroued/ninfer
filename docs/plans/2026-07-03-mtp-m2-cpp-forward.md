# MTP M2 C++ Forward Implementation Plan

**Goal:** Land MTP roadmap milestone M2 by adding MTP draft weight binding, a one-layer MTP KV
namespace, the required BF16 data-movement operators, correctness-first C++ `mtp_forward` batch and
AR-step entry points, and MTP workspace/cache accounting.

**Requirements:** `docs/2026-07-03-mtp-roadmap.md` M2 and coordination table;
`docs/2026-07-03-mtp-implementation-requirements.md` R-L0-2, R-L1-2, R-L2-1, R-L2-4, and the MTP
part of R-L2-5; `docs/2026-07-03-mtp-round-algorithm.md` §1.2, §2.3, §2.4, and §7;
`docs/2026-07-03-mtp-state-management.md` §3 and §6; Part 1 §3/§7 and Part 2 §6/§7/§8.

**Execution Mode:** Main agent direct implementation. Subagent-driven implementation is
intentionally not used because the user explicitly required direct implementation, M2 spans shared
model-card and engine files that must avoid M3 scheduling churn, and the MTP forward path is a
tightly ordered numerical pipeline. Subagents are used only for two reviews: one plan audit before
implementation and one code review after self-test. Every review subagent must use the current
strongest available model exposed by the multi-agent tool; weaker, cheaper, legacy, or
compatibility-oriented models are not allowed.

**Architecture:** Add MTP as an optional model-card capability. `Engine` allocates a second
`KVCache` only when `mtp_draft_tokens > 0`, resets it with target KV at prefill start, and passes a
nullable MTP cache pointer to `Qwen3_6_27B`. The model card binds MTP weights only when the MTP cache
is present. The batch MTP entry consumes caller-provided `ids [T]`, target-final hidden
`[5120,T]`, absolute `positions [T]`, and a host `cache_offset`, then writes caller-owned
`mtp_hidden [5120,T]` and optionally computes one logits column plus greedy draft token. The AR-step
entry consumes one previous draft token, one previous MTP hidden column, and a device position,
writes one new hidden column, and always computes logits plus the next greedy draft. Both entries
reuse existing `linear`, `embed_gather`, `rmsnorm`, `rope`, `gqa_attention_prefill`,
`gqa_attention_decode`, `sigmoid_gate_mul`, `silu_and_mul`, `residual_add`, and `argmax`.

## Preflight Evidence

The predecessor gate was run before writing this plan:

```bash
ctest --test-dir build --output-on-failure -R 'qus_(q5090_parser|weight_store|weight_store_real_file|engine_real_file|linear)_test'
```

Observed result:

```text
100% tests passed, 0 tests failed out of 5
```

This covers M0 fixture/parser/store and real MTP-enabled engine loading, plus the M1 W8G32 linear
operator family test.

## Non-Goals

- No engine round loop, accept/commit logic, draft state machine, MTP-on generation path, or
  prefill-phase integration into `Engine::generate`; these are M4.
- No target verify scheduling, target KV rewind, GDN snapshot state, target chunk-hidden handoff, or
  `run_layers` refactor; these are M3.
- No device-scalar row gather, fixed-shape round assembly kernels, or graph-safe dynamic column
  selection; these are M4/M5.
- No W8 fused epilogues, direct W8 compact `attn_in` output, attention tuning, or launch-count
  optimization; these are M5.
- No compatibility aliases, legacy flags, dual old/new schemas, or tests that preserve old behavior.

## Final MTP C++ API

`include/qus/model/model.h` will add:

```cpp
struct MtpW {
    const Weight* fc = nullptr;
    const Tensor* pre_fc_norm_embedding = nullptr;
    const Tensor* pre_fc_norm_hidden = nullptr;
    const Tensor* input_norm = nullptr;
    const Weight* attn_in = nullptr;
    const Tensor* q_norm = nullptr;
    const Tensor* k_norm = nullptr;
    const Weight* o_proj = nullptr;
    const Tensor* post_attn_norm = nullptr;
    const Weight* gate_up = nullptr;
    const Weight* down = nullptr;
    const Tensor* norm = nullptr;
};

bool mtp_enabled() const noexcept;
const MtpW& mtp_weights() const;
void mtp_set_cache_position(std::uint32_t position);

void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                       std::uint32_t cache_offset, Tensor& mtp_hidden,
                       int logits_column, Tensor* logits, Tensor* draft_token);

void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                         const Tensor& position, Tensor& mtp_hidden,
                         Tensor& logits, Tensor& draft_token);
```

API contracts:

- `mtp_enabled()` is true only when `Qwen3_6_27B` was constructed with an MTP cache. Calling either
  MTP forward entry while disabled throws `std::runtime_error`.
- `mtp_forward_batch` accepts `1 <= T <= prefill_chunk`. `ids` and `positions` are `I32 [T]`.
  `hidden` and `mtp_hidden` are `BF16 [5120,T]`. `cache_offset + T <= mtp_kv.max_context`.
- `logits_column < 0` skips logits and draft. Otherwise `0 <= logits_column < T`, `logits` is
  `BF16 [248320,1]`, and `draft_token` is `I32 [1]`.
- `mtp_forward_ar_step` accepts `token I32 [1]`, `previous_hidden BF16 [5120,1]`,
  `position I32 [1]`, `mtp_hidden BF16 [5120,1]`, `logits BF16 [248320,1]`, and
  `draft_token I32 [1]`.
- Batch attention uses `gqa_attention_prefill(..., *mtp_kv_, layer=0, cache_offset, ...)`.
  AR attention uses `gqa_attention_decode(..., *mtp_kv_, layer=0, ...)`.
- `mtp_set_cache_position(position)` sets the host logical end of the MTP KV namespace after
  validating `position <= max_context`. It exists for M4 to roll back from shifted-pass junk rows to
  `L + a + 1` before AR steps without adding the target-KV rewind API owned by M3.
- `mtp_forward_batch` sets the MTP cursor to `cache_offset + T` only for fully valid windows such as
  prefill-style full-valid batches. If a shifted pass intentionally writes junk rows `a+1..k`, the caller must set
  the logical end to `L + a + 1` before any AR step or subsequent round.
- `mtp_forward_ar_step` does not infer a host cursor from the device position. The caller must ensure
  the MTP cache has no logical gap before the requested position; after the step, either the caller
  advances the logical end with `mtp_set_cache_position(host_position + 1)` or the future M4 wrapper
  does it as part of its host/device scalar synchronization.

## Scope And File Ownership

- `include/qus/model/config.h`: add MTP constants: one MTP layer, `mtp_fc_in = 10240`,
  `mtp_attn_in = 14336`, `mtp_attn_q_rows = 6144`, `mtp_attn_gate_rows = 6144`,
  `mtp_mlp_gateup = 34816`, and reuse existing hidden/head/vocab constants.
- `include/qus/model/model.h` and `src/model/qwen3_6_27b.cpp`: add `MtpW`, optional MTP binding,
  MTP forward helpers, and public MTP entry points. Only additive MTP regions are changed; existing
  `run_layers`, target prefill, decode, and GDN/full-attention code remain structurally intact.
- `include/qus/runtime/engine.h` and `src/runtime/engine.cpp`: add an optional `KVCache mtp_kv_`,
  allocate it when `mtp_draft_tokens > 0`, pass it into the model card, reset it on prefill, and add
  MTP terms to the MTP-enabled cache formula and `default_work_bytes`.
- `include/qus/kernels/mtp_pack.h`,
  `src/kernels/wrapper/mtp_pack.cpp`, `src/kernels/launcher/mtp_pack.h`,
  `src/kernels/launcher/mtp_pack.cu`, and `src/kernels/kernel/mtp_pack.cuh`: implement
  `mtp_pack_fc_input` and `mtp_split_attn_in` using the L1 api/wrapper/launcher/kernel layout.
- `tests/kernels/test_mtp_pack.cpp`: exact BF16 bit-equality tests for pack and split across
  `T = 1,2,6,17,1024`.
- `tests/test_model_bind.cpp`: extend fixture model bind coverage to MTP when `load_mtp = true`.
- `tests/test_engine_memory_stats.cpp`: update cache/workspace formula expectations for
  `mtp_draft_tokens > 0` and assert `mtp_draft_tokens == 0` capacity and behavior are unchanged.
- `tests/CMakeLists.txt`: register only the new behavior tests/tools that protect M2 risks.

## Coordination Points

- `src/model/qwen3_6_27b.cpp` / `include/qus/model/model.h`: M2 only adds MTP functions and data
  structures. It does not refactor target `run_layers`, prefill, decode, or tap behavior.
- `src/runtime/engine.cpp`: M2 adds independent MTP cache/workspace terms. It does not add M3
  verify/GDN snapshot terms.
- `StepState`: M2 keeps the current single-column `StepState` unchanged. MTP public APIs accept
  caller-owned `logits`, `draft_token`, `ids`, `positions`, and hidden tensors so M3 can later
  extend `StepState` without M2 blocking on it.
- `WeightStore`: lookup support is already present. M2 only consumes `tensor/qweight/qfused` APIs;
  it does not change parser or store lookup semantics.

## Task Breakdown

### Task 1: Plan Review

**Reading List:**
- This plan.
- `docs/2026-07-03-mtp-roadmap.md` M2 and coordination table.
- `docs/2026-07-03-mtp-implementation-requirements.md` §2, §3, and §4 M2 rows.
- `docs/2026-07-03-mtp-round-algorithm.md` §1.2, §2.3, §2.4, and §7.
- `docs/2026-07-02-mtp-foundation-part1-design.md` §3 and §7.
- `docs/2026-07-02-mtp-foundation-part2-operators.md` §6, §7, and §8.
- `docs/2026-07-02-mtp-foundation-part1-design.md` §7 reference forward specification.

**Implementation:**
- Dispatch one review subagent using the current strongest available model.
- Require the reviewer to audit MTP math order, API shape contracts, weight binding, KV cursor
  semantics, plan/test-policy compliance, and M2/M3/M4 boundary discipline.
- Revise this plan for every valid Critical or Important finding before changing implementation
  files.

**Definition Of Done:**
- Plan review has no unresolved Critical or Important findings.
- The final API above remains explicit after review or is revised in this file before code starts.

**Verification Commands:**
- No build command is required for this task.
- Record the review summary and any plan revisions in the working notes and final evidence.

### Task 2: MTP Constants And Weight Binding

**Reading List:**
- `include/qus/model/config.h`.
- `include/qus/model/model.h`.
- `src/model/qwen3_6_27b.cpp` bind helpers.
- `include/qus/core/tensor.h` `SourceKind` and `ModuleKind`.
- `include/qus/core/weight_store.h`.
- `docs/2026-07-02-mtp-foundation-part1-design.md` §3.
- `tools/q5090_convert/qtypes.py`.

**Implementation:**
- Add the MTP constants to `ModelConfig`.
- Add `MtpW` and `std::optional`-free nullable binding fields in `Qwen3_6_27B`.
- Add MTP-specific `require_mtp_weight`, `require_mtp_weight_fused`, and `require_mtp_tensor`
  helpers using `ModuleKind::MtpDraft`.
- Bind all required MTP tensors and weights:
  - `mtp.fc.weight`: `SourceKind::MtpFc`, `NO_LAYER`, W8G32 `[5120,10240]`.
  - `mtp.pre_fc_norm_embedding.weight`: `SourceKind::MtpPreFcNormEmb`, `NO_LAYER`, BF16 `[5120]`.
  - `mtp.pre_fc_norm_hidden.weight`: `SourceKind::MtpPreFcNormHid`, `NO_LAYER`, BF16 `[5120]`.
  - `mtp.layers.0.input_layernorm.weight`: `SourceKind::InputLayernorm`, layer 0, BF16 `[5120]`.
  - `mtp.layers.0.attn_in.w8`: fused group `ATTN_IN=1`, index 0, layer 0, W8G32 `[14336,5120]`.
  - `mtp.layers.0.self_attn.q_norm.weight`: `SourceKind::AttnQNorm`, layer 0, BF16 `[256]`.
  - `mtp.layers.0.self_attn.k_norm.weight`: `SourceKind::AttnKNorm`, layer 0, BF16 `[256]`.
  - `mtp.layers.0.self_attn.o_proj.weight`: `SourceKind::AttnO`, layer 0, W8G32 `[5120,6144]`.
  - `mtp.layers.0.post_attention_layernorm.weight`: `SourceKind::PostAttnLayernorm`, layer 0,
    BF16 `[5120]`.
  - `mtp.layers.0.mlp.gateup.w8`: fused group `MLP_GATEUP=3`, index 0, layer 0, W8G32
    `[34816,5120]`.
  - `mtp.layers.0.mlp.down_proj.weight`: `SourceKind::MlpDown`, layer 0, W8G32 `[5120,17408]`.
  - `mtp.norm.weight`: `SourceKind::MtpNorm`, `NO_LAYER`, BF16 `[5120]`.
- Expose `mtp_enabled()`, `mtp_weights()`, and `mtp_set_cache_position()` for future M4 callers.
  Tests should validate them only through observable binding and cache-position behavior, not as
  standalone getter coverage.
- Keep shared embedding and lm_head bound from `TEXT_CORE`; do not look for MTP copies.

**Definition Of Done:**
- Text-only construction with `mtp_draft_tokens == 0` still succeeds without loading/binding MTP.
- MTP-enabled construction binds all 12 MTP-owned logical entries through `MtpW`; shared embed and
  lm_head remain the existing `TEXT_CORE` bindings.
- Fixture and real artifact binding validate source kind, layer, qtype, layout, and shapes.

**Verification Commands:**
```bash
cmake --build build --target qus_model_bind_test qus_engine_real_file_test -j
ctest --test-dir build --output-on-failure -R 'qus_model_bind_test|qus_engine_real_file_test'
```

### Task 3: L1 MTP Pack And Split Operators

**Reading List:**
- `docs/l1-kernel-layering.md`.
- `docs/l1-operator-catalog.md`.
- `docs/2026-07-02-mtp-foundation-part2-operators.md` §6 and §7.
- `include/qus/kernels/silu_and_mul.h` and its wrapper/launcher/kernel files.
- `tests/kernels/test_silu_and_mul.cpp` and `tests/kernels/op_tester.h`.

**Implementation:**
- Add `include/qus/kernels/mtp_pack.h` with:
  - `void mtp_pack_fc_input(const Tensor& embedding_norm, const Tensor& hidden_norm, Tensor& out,
    cudaStream_t stream);`
  - `void mtp_split_attn_in(const Tensor& attn_in, Tensor& q, Tensor& k, Tensor& gate, Tensor& v,
    cudaStream_t stream);`
- Validate all tensors as BF16, contiguous, non-null, and exact MTP shapes:
  - pack: `embedding_norm [5120,T]`, `hidden_norm [5120,T]`, `out [10240,T]`, `T > 0`.
  - split: `attn_in [14336,T]`, `q [256,24,T]`, `k [256,4,T]`, `gate [256,24,T]`,
    `v [256,4,T]`, `T > 0`.
- Implement a simple copy kernel:
  - pack rows `0..5119` from embedding norm and rows `5120..10239` from hidden norm.
  - split row ranges `q[0,6144)`, `k[6144,7168)`, `gate[7168,13312)`, `v[13312,14336)` into
    compact output tensors.
- Add exact BF16 bit-equality tests for `T = 1,2,6,17,1024`, using deterministic host patterns that
  expose row and token indexing mistakes.
- Run `compute-sanitizer` on the new operator test when available.

**Definition Of Done:**
- Pack and split pass exact bit-equality checks for both aligned and non-aligned token counts.
- Invalid shape/dtype/contiguity cases throw `std::invalid_argument` from the wrapper.
- The split output is compact for `T > 1`; no downstream op consumes a strided slice view.

**Verification Commands:**
```bash
cmake --build build --target qus_mtp_pack_test -j
./build/tests/qus_mtp_pack_test
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_pack_test
```

### Task 4: Engine MTP KV Allocation And Memory Accounting

**Reading List:**
- `include/qus/runtime/engine.h`.
- `src/runtime/engine.cpp`.
- `include/qus/core/kv_cache.h` and `src/core/kv_cache.cpp`.
- `docs/2026-07-03-mtp-state-management.md` §3 and §6.
- `tests/test_engine_memory_stats.cpp`.

**Implementation:**
- Add `std::optional<KVCache> mtp_kv_` to `Engine`.
- In `Engine::load`, allocate `mtp_kv_` after target `kv_` only when `mtp_draft_tokens > 0`:
  - `full_layers = 1`;
  - same `max_ctx`, `num_kv_heads`, `head_dim`, and BF16 dtype as target KV.
- Pass `mtp_kv_ ? &*mtp_kv_ : nullptr` into `Qwen3_6_27B`.
- Reset `mtp_kv_` in `Engine::prefill` when present, next to target `kv_->reset()`.
- Split cache sizing into a shared helper that takes `include_mtp_kv` and the existing private
  `default_cache_bytes(max_ctx)` MTP-off wrapper. In `Engine::load`, call the helper with
  `include_mtp_kv = options_.mtp_draft_tokens > 0` when `options_.cache_bytes == 0`.
- Keep `mtp_draft_tokens == 0` default cache capacity unchanged. MTP-enabled default cache capacity
  grows by exactly one BF16 K/V layer pair plus arena alignment slack.
- Update `default_work_bytes(prefill_chunk)` with a separate MTP phase peak from persistent base:
  - `ids [T]` and positions `[T]` are already in existing persistent prefill work.
  - MTP-specific live tensors include embedding norm `[5120,T]`, hidden norm `[5120,T]`,
    fc input `[10240,T]`, fc output/residual `[5120,T]`, layer norm `[5120,T]`,
    attn_in `[14336,T]`, compact q/gate `[6144,T]`, compact k/v `[1024,T]`, q/k norm,
    attention output/gated flat `[6144,T]`, o-proj output `[5120,T]`, post norm `[5120,T]`,
    gateup `[34816,T]`, SwiGLU act `[17408,T]`, down output `[5120,T]`, final norm
    `[5120,T]`, and one logits column `[248320,1]`.
  - Use `T = prefill_chunk` for the peak.
- Update `qus_engine_memory_stats_test` to check the new formula and MTP load path without adding
  source-structure assertions.

**Definition Of Done:**
- Engine loads text-only and MTP-enabled fixtures with correctly sized default arenas.
- MTP KV is present only for `mtp_draft_tokens > 0` and has one layer.
- Prefill resets MTP KV when present.
- MTP cache position can be set explicitly for shifted-pass rollback without changing target KV.
- Memory stats test covers loaded MTP arena behavior and confirms text-only generation behavior is
  not changed by M2.

**Verification Commands:**
```bash
cmake --build build --target qus_engine_memory_stats_test qus_engine_real_file_test -j
ctest --test-dir build --output-on-failure -R 'qus_engine_memory_stats_test|qus_engine_real_file_test'
```

### Task 5: MTP Batch Forward Entry

**Reading List:**
- `src/model/qwen3_6_27b.cpp` `attn_mix`, `mlp_tail`, `prefill_impl`, and `decode_step_impl`.
- `include/qus/kernels/{embed_gather,rmsnorm,linear,rope,gqa_attention,sigmoid_gate_mul,silu_and_mul,residual_add,argmax}.h`.
- `docs/2026-07-03-mtp-round-algorithm.md` §1.2, §2.3, and §7.
- `docs/2026-07-02-mtp-foundation-part1-design.md` §7 MTP forward specification.

**Implementation:**
- Add private validation helpers for MTP tensor shapes and MTP-enabled state.
- Implement `mtp_forward_batch` in this order:
  1. gather shared embedding from `ids` into `[5120,T]`;
  2. RMSNorm embedding with `pre_fc_norm_embedding` and `unit_offset=true`;
  3. RMSNorm supplied hidden with `pre_fc_norm_hidden` and `unit_offset=true`;
  4. `mtp_pack_fc_input` into `[10240,T]`;
  5. W8G32 `linear` through `mtp.fc` into working residual `[5120,T]`;
  6. input RMSNorm, W8G32 `attn_in`, `mtp_split_attn_in`;
  7. q/k RMSNorm with `unit_offset=true`, RoPE using absolute `positions`;
  8. `gqa_attention_prefill` against `*mtp_kv_`, layer 0, and `cache_offset`;
  9. sigmoid gate multiply, W8G32 `o_proj`, and residual add;
  10. post-attention RMSNorm, W8G32 `gateup`, `silu_and_mul`, W8G32 `down`, and residual add;
  11. final MTP RMSNorm into caller-owned `mtp_hidden`;
  12. if requested, slice the host-specified column, run shared `lm_head`, and run `argmax`.
- Rewind workspace marks around the MTP layer so output tensors remain caller-owned.
- Set `mtp_kv_->pos = cache_offset + T` after the batch call only for fully valid batch windows.
  Round shifted callers that wrote junk rows must call `mtp_set_cache_position(L + a + 1)` before AR.

**Definition Of Done:**
- Batch entry supports `1 <= T <= prefill_chunk`.
- Non-logits batch mode fills MTP KV and hidden without requiring a logits buffer.
- Logits mode computes only one requested column and uses lowest-index argmax tie behavior through
  the existing `argmax` kernel.
- T=1 batch and AR semantics can be compared on identical one-token inputs after matching cache
  setup.

**Verification Commands:**
```bash
cmake --build build --target qus_core -j
```

### Task 6: MTP AR-Step Entry

**Reading List:**
- `docs/2026-07-03-mtp-round-algorithm.md` §1.3 and §2.4.
- `src/model/qwen3_6_27b.cpp` decode branch of `attn_mix`.
- `docs/2026-07-03-mtp-round-algorithm.md` §2.4 AR-step construction.

**Implementation:**
- Implement `mtp_forward_ar_step` using the same MTP graph as batch with `T = 1`.
- Use `gqa_attention_decode` with the provided device `position` and `*mtp_kv_`, layer 0.
- Always compute one logits column and one draft token.
- Require the caller to set the host logical cache end before and after AR when the device position
  comes from runtime scalars. In M2 tests with host-known positions, call `mtp_set_cache_position`
  before the AR chain and after each step to keep `KVCache::pos` consistent with the written prefix.

**Definition Of Done:**
- AR step writes one MTP KV row at the device position and attends over `0..position`.
- A five-step AR chain can feed each `mtp_hidden` and draft token into the next call with explicit
  host logical-end updates and no cache gaps.
- The implementation shares the same helper path as batch wherever possible without adding a generic
  runtime abstraction for future models.

**Verification Commands:**
```bash
cmake --build build --target qus_core -j
```

### Task 7: Non-Parity MTP Verification

**Reading List:**
- `docs/l1-op-test-standard.md`.
- `tests/kernels/test_mtp_pack.cpp`.
- `tests/test_model_bind.cpp`.
- `tests/test_engine_memory_stats.cpp`.
- `src/model/qwen3_6_27b.cpp` MTP forward entry points.

**Implementation:**
- Do not add or keep a cross-implementation MTP forward parity harness. The project owner explicitly
  waived that requirement because Python and C++ use different math paths.
- Verify the new L1 MTP pack/split operators with exact BF16 bit-copy tests and sanitizer.
- Verify MTP binding against fixture and real q5090 artifact loading.
- Verify MTP cache/workspace accounting through engine memory stats and real-file load tests.
- Verify the MTP forward entries by compilation, shape/contract review, and the post-implementation
  code-review phase rather than Python-vs-C++ numerical comparison.

**Definition Of Done:**
- No `qus_mtp_forward_test`, MTP parity runner, or parity-only source file remains in this change.
- `qus_mtp_pack_test` passes and is clean under `compute-sanitizer --tool memcheck`.
- Binding, real-file load, and engine memory tests pass with MTP enabled and disabled.
- Full build succeeds.

**Verification Commands:**
```bash
cmake --build build --target qus_core qus_mtp_pack_test qus_model_bind_test qus_engine_memory_stats_test qus_engine_real_file_test -j
ctest --test-dir build --output-on-failure -R 'qus_mtp_pack_test|qus_model_bind_test|qus_engine_memory_stats_test|qus_engine_real_file_test'
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_pack_test
```

### Task 8: Full M2 Gate, Review, And Commits

**Reading List:**
- `docs/2026-07-03-mtp-roadmap.md` M2 gate.
- This plan's task definitions.
- `git diff --stat` and changed files.

**Implementation:**
- Run focused verification as tasks land.
- Run the full M2 gate after implementation self-test.
- Dispatch one code-review subagent using the current strongest available model. The reviewer must
  focus on row-range split/compactness, norm/RoPE parameters, KV bounds and fill-then-attend order,
  residual chain ordering, workspace/cache peak math, and scope boundaries.
- Fix every valid Critical and Important finding and rerun affected verification.
- Commit in conventional chunks, for example:
  - `docs(plan): add mtp m2 cpp forward plan`
  - `feat(model): bind mtp draft weights`
  - `feat(kernels): add mtp pack and split ops`
  - `feat(model): add mtp forward entries`
  - `feat(runtime): account for mtp kv and workspace`

**Definition Of Done:**
- All M2 gate items have fresh command evidence.
- Plan and verification evidence are in the repository.
- Code review has no unresolved Critical or Important findings.
- Git status is clean after conventional commits.

**Verification Commands:**
```bash
cmake --build build
ctest --test-dir build --output-on-failure
ctest --test-dir build --output-on-failure -R 'qus_model_bind_test|qus_engine_memory_stats_test|qus_mtp_pack_test|qus_engine_real_file_test'
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_pack_test
```

## Review Phase

### Plan Review

Dispatch one review subagent before implementation. The prompt must include this plan and ask the
reviewer to audit:

- M2 boundary and coordination discipline against the roadmap;
- MTP C++ API suitability for M4 without implementing M4;
- exact MTP weight binding, source kinds, fused groups, layers, qtypes, and shapes;
- shifted-pass and AR-step math order against the Python ref model;
- fill-then-attend and MTP KV cursor semantics;
- allowed tests under AGENTS.md.

### Code Review

Dispatch one review subagent after self-test. The prompt must include the changed diff, this plan,
and focused instructions to inspect:

- `attn_in` split row ranges and compact layout for `T > 1`;
- q/k norm axis and `(1+w)` parameters for every MTP norm;
- RoPE position use and cache-offset semantics;
- MTP KV write/read bounds and stale/junk row behavior;
- residual after attention and residual after MLP ordering;
- workspace peak formula and cache arena formula;
- absence of M3/M4 scope creep.

Valid Critical and Important findings are fixed before final verification and commits. If a reviewer
finding is technically incorrect, record the reason and the command or code evidence that disproves it.

## Testing Policy Notes

- MTP pack/split tests are allowed under AGENTS.md hard whitelist 1/2 because they protect real
  tensor layout and exact data movement contracts that would corrupt numerical behavior.
- MTP binding tests are allowed under binary/file-format contract coverage because they validate
  consumed q5090 source kinds, fused groups, layers, qtypes, and shapes.
- No cross-implementation MTP forward numerical comparison is added in M2 per owner instruction;
  the C++ path is covered by constituent operator tests, binding/file-format tests, real artifact
  load checks, build verification, sanitizer on the new L1 operator, and review.
- Engine memory stats tests are allowed for GPU memory/lifetime and downstream-consumed arena
  accounting. They must not assert source structure.
- No tests will scan source files, lock private call order, preserve deprecated behavior, or merely
  increase coverage.

## Verification Evidence

This section is append-only after implementation begins. Each entry must include the exact command,
exit status, and key output lines from the fresh run.

### 2026-07-03 M2 Verification

```bash
cmake --build build --target qus_core qus_mtp_pack_test qus_model_bind_test qus_engine_memory_stats_test qus_engine_real_file_test -j
```

Exit status: 0. Key result: focused M2 targets rebuilt successfully.

```bash
ctest --test-dir build --output-on-failure -R 'qus_mtp_pack_test|qus_model_bind_test|qus_engine_memory_stats_test|qus_engine_real_file_test'
```

Exit status: 0. Key result: `100% tests passed, 0 tests failed out of 4`.

```bash
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_pack_test
```

Exit status: 0. Key result: `OK mtp_pack correctness`; `ERROR SUMMARY: 0 errors`.

```bash
cmake --build build -j
```

Exit status: 0. Key result: full build completed successfully.

```bash
ctest --test-dir build --output-on-failure
```

Exit status: 0. Key result: `100% tests passed, 0 tests failed out of 36`.

```bash
cmake --build build --target qus_model_bind_test -j && ./build/tests/qus_model_bind_test
```

Exit status: 0. Key result: binding test rebuilt and passed after removing the redundant
`mtp_enabled()` getter assertion flagged during review.

### 2026-07-03 Code Review

Reviewer: strongest available subagent model (`gpt-5.5`, `xhigh`, priority).

Result: no Critical or Important findings. Minor low-value getter assertion in
`tests/test_model_bind.cpp` was removed. Reviewer confirmed no parity-driven production diffs
remained in GQA decode, W8G32 codec, linear dispatch, or `tools/parity/ref_model.py`.
