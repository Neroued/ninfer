# MTP Decode Round CUDA-Graph Redesign — Implementation Document

**Goal:** Make the k≥1 MTP speculative-decode round fully CUDA-Graph capturable by removing every
mid-round host↔device synchronization and host-side control flow that depends on device data, so
one whole round (verify → accept → commit → propose → sample) replays as a single captured
`cudaGraph`, with the host doing exactly one readback at the round boundary.

**Non-goals:**

- No stochastic sampling. Acceptance stays greedy (argmax target, longest matching draft prefix),
  identical to today. No RNG state, no probability buffers.
- No cross-round pipelining / speculative double-buffering. One round graph, then one boundary
  readback, then the next round. (vLLM-style overlap is explicitly out of scope.)
- No change to any CUDA kernel. GQA append, GDN snapshot, and all MTP round kernels are already
  device-scalar driven; this is a host-orchestration refactor only.
- No new model behavior, no numerical change. Graphed and eager MTP must produce identical tokens.

---

## Execution Mode

- **Mode:** Linear, single-agent execution (intended for Codex). This is a cohesive refactor across
  tightly coupled host orchestration in `Engine` and a small set of L2 signatures; the tasks share
  the same files (`engine.cpp`, `engine.h`, `qwen3_6_27b.cpp`, `model.h`) and cannot be cleanly
  partitioned across independent subagents without churn. **Subagent-driven development is
  intentionally NOT used.**
- **Verification gating:** There are **no per-task verification gates**. Implement all tasks in
  order, then run the single **Verification** phase at the end. Do not build/test between tasks.
- **Commits:** One commit per task is fine, but do not treat inter-task points as validation
  checkpoints. Final correctness is proven only by the Verification phase.
- **Backward compatibility:** Per repo policy, delete the old eager orchestration outright. Do not
  keep legacy paths, flags, or the removed method signatures behind shims.

---

## Background: what blocks capture today (for context only)

The current k≥1 path (`Engine::decode_round`, `src/runtime/engine.cpp:676`) is explicitly excluded
from graph capture (`engine.cpp:629`: `use_graph = use_cuda_graph && mtp_draft_tokens == 0`) and,
even if enabled, would break capture because of:

1. A **mid-round** blocking readback of the device length into the host:
   `read_i32_scalar(&StepState::pos)` at `engine.cpp:694` (`ctx_->synchronize()` +
   `cudaMemcpy` D2H), used to update `kv_->pos` / `mtp_kv_->pos` and to drive the propose loop.
2. A **host-driven propose loop** that computes `host_pos` from that readback and calls
   `mtp_set_cache_position()` (host writes to `mtp_kv_->pos`): `engine.cpp:654,662,672`.

Verified facts that make the redesign safe and small:

- **GQA append is device-driven.** The decode/verify attention writes new K/V at
  `cache[positions[token]]`, where `positions` is a device tensor; it never reads `kv.pos` on the
  host (`src/kernels/launcher/gqa_attention_decode.cu:64-113`, kernel reads `positions[...]`).
- **GDN snapshot is device-driven.** Conv/SSM snapshot selects its slot from `*initial_slot`, a
  device I32 scalar (`src/kernels/kernel/causal_conv1d.cuh:154`).
- **`io_.pos` is the device logical length `L`.** `mtp_prepare_verify_inputs` reads it to build
  `window_base`/`positions`; `mtp_accept_tokens` advances it (`*length += a+1; *ar_pos = *length`)
  — see `src/kernels/kernel/mtp_round.cuh:12-52`. Propose AR steps use `io_.ar_pos` (device),
  incremented by `mtp_increment_i32` (device).
- **Host `kv_->pos` / `mtp_kv_->pos` are used only for host bounds checks and bookkeeping**, never
  to compute a kernel argument (`grep` across `qwen3_6_27b.cpp` + `engine.cpp`). `mtp_kv_->pos` is
  written at `qwen3_6_27b.cpp:460,556` and `engine.cpp:696` and read nowhere except the two host
  bounds checks that this refactor deletes.
- **KV padding already covers a full round.** `KVCache` sets `padded_context = align_up(max_ctx,128)`
  (`src/core/kv_cache.cpp:79`), and the existing guard `kv_->pos + 2k > max_ctx → fallback`
  bounds the maximum device write index in a round to `< max_ctx ≤ padded_context`. **No
  over-allocation is required.**

---

## Target Architecture

### One device-owned position, one host mirror

`io_.pos` (device I32) is the single source of truth for the logical KV length `L`. It is advanced
only by device kernels (`mtp_accept_tokens` for rounds, `advance_pos` for single steps). The host
keeps a mirror in `kv_->pos`, advanced **only at the round boundary** by the accepted-token count
`n` read back once. Invariant, checked implicitly by tests:

> At every round boundary, host `kv_->pos == device io_.pos`. It starts equal after prefill
> (`qwen3_6_27b.cpp:988` sets `kv_.pos = t0+len`; `:991` sets device `io_.pos = T`), and each round
> advances both by the same `n = num_sampled = a+1`.

`mtp_kv_->pos` is removed from the round entirely (it was never read by kernels).

### The captured round body

`Engine::record_decode_round()` is a **device-only** function (no host reads of device data, no host
`pos` writes, no host branch on device values). It is captured once (after a one-round warmup) into
a dedicated `DecodeGraph round_graph_` and replayed every round. The propose loop has a **fixed**
trip count `k` (a load-time constant), so its `k-1` iterations are recorded during capture and
replayed verbatim. Body, in order:

```
mtp_prepare_verify_inputs   // io_.pos -> io_.window_base, io_.verify_ids, io_.positions
target_verify(T=k+1)        // 64 layers; GQA append via io_.positions; GDN snapshot via gdn_initial_slot
mtp_accept_tokens           // greedy prefix; writes io_.token/sampled_out/num_sampled/accepted/ar_pos; L += a+1
mtp_set_gdn_initial_slot_from_accepted
mtp_prepare_shifted_ids
mtp_forward_batch(T=k+1)    // MTP shifted pass; append via io_.positions
mtp_sample_from_hidden_row  // draft[0]
repeat (i = 1 .. k-1):      // unrolled at capture
    mtp_forward_ar_step     // append via io_.ar_pos (device)
    cudaMemcpyAsync D2D (io_.mtp_ar_hidden <- next_hidden)   // async, capture-legal
    mtp_increment_i32(io_.ar_pos)
```

All scratch comes from the pre-sized `WorkspaceArena` via deterministic `mark`/`rewind`/`reset`
(pure host pointer math on a fixed base) — identical to how the existing single-token decode graph
is already captured, so pointers are stable across replays. There is **no** `cudaMalloc`/`cudaFree`,
no host sync, and no host branch on device data anywhere in the body.

### Host orchestration and boundary readback

`Engine::decode_round()` keeps host responsibilities only:

```
if k <= 0 or MTP disabled:                       -> return {decode_step_one()}
if kv_->pos + 2k > max_ctx (host mirror guard):  -> return {decode_step_one()}   // context tail
if use_cuda_graph and round_warmed_:
    if !round_graph_.ready(): round_graph_.capture(stream, record_decode_round)
    round_graph_.launch(stream)
else:
    record_decode_round(); round_warmed_ = true   // one eager warmup round
return read_round_output()                         // the single boundary readback
```

`read_round_output()` performs the one sync per round (outside the graph, exactly like the existing
decode graph's `read_token()`), reads `num_sampled` and `sampled_out[0..n)`, advances the host
mirror `kv_->pos += n`, applies host-side stop-token truncation, and returns the tokens.

### Edge paths (unchanged behavior, now graphable)

- **k = 0 (MTP off):** unchanged — `decode_step_one()` captures `decode_step_record()` into
  `decode_graph_`.
- **Context tail (k≥1):** when the host mirror shows a full round would overflow, `decode_round()`
  falls back to `decode_step_one()`. This refactor lets `decode_step_one()` also use the captured
  single-token `decode_graph_` when `mtp>0` (dropping the `&& mtp==0` gate), with the MTP
  bookkeeping kernels (`mtp_count_fallback_step`, `mtp_reset_gdn_initial_slot`) run after launch as
  today. The host chooses which pre-captured graph to launch **before** launching — there is no
  in-graph branch.

---

## Invariants and Assumptions (do not violate)

1. `record_decode_round()` must contain **no** `cudaMemcpy` (non-async), `cudaStreamSynchronize`,
   `cudaDeviceSynchronize`, `cudaMalloc`, `cudaFree`, and **no** host read of any device tensor.
   Every enqueue is on `ctx_->stream`.
2. The propose loop bound is `options_.mtp_draft_tokens` (constant for the engine's lifetime). Never
   derive the loop bound or any position from a device value on the host inside the body.
3. The workspace arena is sized for the MTP+verify peak at load
   (`default_work_bytes_for(prefill_chunk, mtp_draft_tokens)`, `engine.cpp:462-464`). Do not add
   allocations that could exceed it, or capture may allocate and fail.
4. Host mirror rule: `kv_->pos` is advanced by `n` only in `read_round_output()` (rounds) and by
   `kv_->advance()` in `decode_step_one()` (single steps). No other site writes `kv_->pos` during
   decode.

---

## File / Ownership Change Map

| File | Change |
|------|--------|
| `include/qus/model/model.h` | Drop `mtp_set_cache_position`; drop `cache_offset` param from `target_verify` and `mtp_forward_batch`. |
| `src/model/qwen3_6_27b.cpp` | Delete `mtp_set_cache_position`; remove `cache_offset` use + bounds checks + `mtp_kv_->pos` writes in `target_verify`/`mtp_forward_batch`; update prefill call sites. |
| `include/qus/runtime/engine.h` | Remove `read_i32_scalar`, `read_sampled_tokens`, `propose_mtp_after_accept`; add `record_decode_round`, `record_propose`, `read_round_output`; add members `round_graph_`, `round_warmed_`. |
| `src/runtime/engine.cpp` | Rewrite `decode_round`; add `record_decode_round`/`record_propose`/`read_round_output`; delete `read_i32_scalar`/`read_sampled_tokens`/`propose_mtp_after_accept`; enable graph in `decode_step_one`; reset `round_graph_`/`round_warmed_` in `load`. |
| `bench/qus_bench_support.cpp` | Update `decode_path_name` so graphed MTP reports `mtp_cuda_graph`. |
| `bench/target_verify_bench.cpp` | Update the `target_verify` call to the new 2-arg signature. |
| `tests/test_engine_mtp_e2e.cpp` | Add a graph-vs-eager parity scenario (Verification). |

---

## Implementation Tasks

> Do all tasks in order. No build/test between tasks. Full code is given for every change.

### Task 1 — L2 signature cleanup (`model.h`)

**File:** `include/qus/model/model.h`

Delete the `mtp_set_cache_position` declaration (currently line 156) and change the two signatures.

Current:

```cpp
    void mtp_set_cache_position(std::uint32_t position);

    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           std::uint32_t cache_offset, Tensor& mtp_hidden, int logits_column,
                           Tensor* logits, Tensor* draft_token);
```

Replace with (drop `mtp_set_cache_position` entirely, drop `cache_offset`):

```cpp
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           Tensor& mtp_hidden, int logits_column, Tensor* logits,
                           Tensor* draft_token);
```

Current (line 169):

```cpp
    void target_verify(const Tensor& ids, const Tensor& positions, std::uint32_t cache_offset);
```

Replace with:

```cpp
    void target_verify(const Tensor& ids, const Tensor& positions);
```

### Task 2 — L2 body cleanup (`qwen3_6_27b.cpp`)

**File:** `src/model/qwen3_6_27b.cpp`

**2a. Delete `mtp_set_cache_position` definition** (currently lines 455-461):

```cpp
void Qwen3_6_27B::mtp_set_cache_position(std::uint32_t position) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP KV cache is not enabled"); }
    if (position > mtp_kv_->max_context) {
        throw std::out_of_range("MTP KV cache position exceeds max_context");
    }
    mtp_kv_->pos = position;
}
```

Delete the whole function (and its blank line).

**2b. `mtp_forward_batch`** — drop `cache_offset` param, its bounds check, and the `mtp_kv_->pos`
write. Current signature + head (lines 529-556):

```cpp
void Qwen3_6_27B::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, std::uint32_t cache_offset,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    const auto token_count = static_cast<std::uint32_t>(T);
    if (cache_offset > mtp_kv_->max_context || token_count > mtp_kv_->max_context - cache_offset) {
        throw std::out_of_range("MTP batch cache range exceeds max_context");
    }
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    mtp_forward_core(ids, hidden, positions, mtp_hidden);
    mtp_kv_->pos = cache_offset + token_count;

    if (logits_column >= 0) {
```

Replace the head (down to and including the `mtp_forward_core(...)` call) with:

```cpp
void Qwen3_6_27B::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions, Tensor& mtp_hidden, int logits_column,
                                    Tensor* logits, Tensor* draft_token) {
    if (mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    mtp_forward_core(ids, hidden, positions, mtp_hidden);

    if (logits_column >= 0) {
```

(Everything from `if (logits_column >= 0) {` onward is unchanged: the `linear`/`argmax` block and
`work_.rewind`.)

**2c. `target_verify`** — drop `cache_offset` param and its bounds check. Current head (lines
606-621):

```cpp
void Qwen3_6_27B::target_verify(const Tensor& ids, const Tensor& positions,
                                std::uint32_t cache_offset) {
    const int T = ids.ne[0];
    if (T <= 0) { throw std::invalid_argument("target_verify T must be positive"); }
    require_tensor_shape(ids, DType::I32, {T}, "target_verify ids");
    require_tensor_shape(positions, DType::I32, {T}, "target_verify positions");
    require_tensor_window(io_.verify_hidden, DType::BF16, kCfg.hidden, T, "target_verify hidden");
    require_tensor_window(io_.logits, DType::BF16, kCfg.vocab, T, "target_verify logits");
    require_vector_window(io_.target_tokens, DType::I32, T, "target_verify target_tokens");
    const auto token_count = static_cast<std::uint32_t>(T);
    if (cache_offset > kv_.max_context || token_count > kv_.max_context - cache_offset) {
        throw std::out_of_range("target_verify cache range exceeds max_context");
    }

    cudaStream_t s = ctx_.stream;
    work_.reset();
```

Replace with:

```cpp
void Qwen3_6_27B::target_verify(const Tensor& ids, const Tensor& positions) {
    const int T = ids.ne[0];
    if (T <= 0) { throw std::invalid_argument("target_verify T must be positive"); }
    require_tensor_shape(ids, DType::I32, {T}, "target_verify ids");
    require_tensor_shape(positions, DType::I32, {T}, "target_verify positions");
    require_tensor_window(io_.verify_hidden, DType::BF16, kCfg.hidden, T, "target_verify hidden");
    require_tensor_window(io_.logits, DType::BF16, kCfg.vocab, T, "target_verify logits");
    require_vector_window(io_.target_tokens, DType::I32, T, "target_verify target_tokens");

    cudaStream_t s = ctx_.stream;
    work_.reset();
```

(The verify body, lines 623-638, is unchanged.)

**2d. Prefill call sites** — update `mtp_forward_batch` calls and delete `mtp_set_cache_position`
calls in the final-chunk MTP block.

Current (lines 953-984), the `is_last` branch and its `else`:

```cpp
                if (is_last) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.drafts.slice(0, 0, 1);
                    mtp_forward_batch(mtp_ids, xf, positions, static_cast<std::uint32_t>(t0),
                                      mtp_hidden, len - 1, &logits, &draft0);
                    const auto* src = static_cast<const unsigned char*>(mtp_hidden.data) +
                                      static_cast<std::size_t>(len - 1) *
                                          static_cast<std::size_t>(kCfg.hidden) *
                                          dtype_size(DType::BF16);
                    CUDA_CHECK(cudaMemcpyAsync(io_.mtp_ar_hidden.data, src,
                                               io_.mtp_ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                               s));

                    detail::set_pos(io_.ar_pos, T, s);
                    for (int i = 1; i < io_.drafts.ne[0]; ++i) {
                        const int host_pos = T + i - 1;
                        mtp_set_cache_position(static_cast<std::uint32_t>(host_pos));
                        Tensor prev_token  = io_.drafts.slice(0, i - 1, 1);
                        Tensor next_token  = io_.drafts.slice(0, i, 1);
                        Tensor next_hidden = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        mtp_forward_ar_step(prev_token, io_.mtp_ar_hidden, io_.ar_pos, next_hidden,
                                            logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp_ar_hidden.data, next_hidden.data,
                                                   io_.mtp_ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        kernels::mtp_increment_i32(io_.ar_pos, s);
                        mtp_set_cache_position(static_cast<std::uint32_t>(host_pos + 1));
                    }
                } else {
                    mtp_forward_batch(mtp_ids, xf, positions, static_cast<std::uint32_t>(t0),
                                      mtp_hidden, -1, nullptr, nullptr);
                }
```

Replace with (drop the `t0` argument in both calls; delete both `mtp_set_cache_position` lines):

```cpp
                if (is_last) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.drafts.slice(0, 0, 1);
                    mtp_forward_batch(mtp_ids, xf, positions, mtp_hidden, len - 1, &logits, &draft0);
                    const auto* src = static_cast<const unsigned char*>(mtp_hidden.data) +
                                      static_cast<std::size_t>(len - 1) *
                                          static_cast<std::size_t>(kCfg.hidden) *
                                          dtype_size(DType::BF16);
                    CUDA_CHECK(cudaMemcpyAsync(io_.mtp_ar_hidden.data, src,
                                               io_.mtp_ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                               s));

                    detail::set_pos(io_.ar_pos, T, s);
                    for (int i = 1; i < io_.drafts.ne[0]; ++i) {
                        Tensor prev_token  = io_.drafts.slice(0, i - 1, 1);
                        Tensor next_token  = io_.drafts.slice(0, i, 1);
                        Tensor next_hidden = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        mtp_forward_ar_step(prev_token, io_.mtp_ar_hidden, io_.ar_pos, next_hidden,
                                            logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp_ar_hidden.data, next_hidden.data,
                                                   io_.mtp_ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        kernels::mtp_increment_i32(io_.ar_pos, s);
                    }
                } else {
                    mtp_forward_batch(mtp_ids, xf, positions, mtp_hidden, -1, nullptr, nullptr);
                }
```

> Note: `mtp_kv_->pos` is now written nowhere. It stays `0` after `reset()`; no kernel reads it.
> This is intentional (positions are device-driven via `io_.positions` / `io_.ar_pos`).

**2e. Bench call site** — the target-verify microbench calls the old 3-arg signature.

**File:** `bench/target_verify_bench.cpp`

Current (line 212):

```cpp
            card.target_verify(ids_t, positions_t, 0);
```

Replace with:

```cpp
            card.target_verify(ids_t, positions_t);
```

### Task 3 — Engine header (`engine.h`)

**File:** `include/qus/runtime/engine.h`

**3a. Private methods.** Current (lines 92-99):

```cpp
    void require_loaded() const;
    [[nodiscard]] int read_token();
    [[nodiscard]] int read_i32_scalar(const Tensor model::StepState::*field);
    [[nodiscard]] std::vector<int> read_sampled_tokens();
    [[nodiscard]] bool is_stop_token(int token) const noexcept;
    [[nodiscard]] int decode_step_one();
    [[nodiscard]] std::vector<int> decode_round();
    void propose_mtp_after_accept(std::uint32_t host_window_base, int host_length, int k);
```

Replace with:

```cpp
    void require_loaded() const;
    [[nodiscard]] int read_token();
    [[nodiscard]] bool is_stop_token(int token) const noexcept;
    [[nodiscard]] int decode_step_one();
    [[nodiscard]] std::vector<int> decode_round();
    void record_decode_round();
    void record_propose(int k);
    [[nodiscard]] std::vector<int> read_round_output();
```

**3b. Members.** Current (lines 112-114):

```cpp
    DecodeGraph decode_graph_;
    bool decode_warmed_ = false;
    std::vector<int> pending_sampled_;
```

Replace with:

```cpp
    DecodeGraph decode_graph_;
    DecodeGraph round_graph_;
    bool decode_warmed_ = false;
    bool round_warmed_  = false;
    std::vector<int> pending_sampled_;
```

### Task 4 — Engine load reset (`engine.cpp`)

**File:** `src/runtime/engine.cpp`

Current (lines 444-446):

```cpp
    decode_graph_.reset();
    decode_warmed_ = false;
    pending_sampled_.clear();
```

Replace with:

```cpp
    decode_graph_.reset();
    round_graph_.reset();
    decode_warmed_ = false;
    round_warmed_  = false;
    pending_sampled_.clear();
```

### Task 5 — Delete old readback helpers (`engine.cpp`)

**File:** `src/runtime/engine.cpp`

Delete `read_i32_scalar` (lines 582-588) and `read_sampled_tokens` (lines 590-603) in full:

```cpp
int Engine::read_i32_scalar(const Tensor model::StepState::*field) {
    int value = 0;
    ctx_->synchronize();
    const model::StepState& io = io_;
    CUDA_CHECK(cudaMemcpy(&value, (io.*field).data, sizeof(value), cudaMemcpyDeviceToHost));
    return value;
}

std::vector<int> Engine::read_sampled_tokens() {
    const int n = read_i32_scalar(&model::StepState::num_sampled);
    if (n <= 0 || n > io_.sampled_out.ne[0]) {
        throw std::runtime_error("Engine MTP sampled token count is invalid");
    }
    std::vector<int> out(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(out.data(), io_.sampled_out.data, out.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    const auto stop = std::find_if(out.begin(), out.end(), [&](int token) {
        return is_stop_token(token);
    });
    if (stop != out.end()) { out.erase(stop + 1, out.end()); }
    return out;
}
```

Add the replacement `read_round_output()` in their place (keep `read_token` above it unchanged):

```cpp
std::vector<int> Engine::read_round_output() {
    ctx_->synchronize();
    int n = 0;
    CUDA_CHECK(cudaMemcpy(&n, io_.num_sampled.data, sizeof(n), cudaMemcpyDeviceToHost));
    if (n <= 0 || n > io_.sampled_out.ne[0]) {
        throw std::runtime_error("Engine MTP sampled token count is invalid");
    }
    std::vector<int> out(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(out.data(), io_.sampled_out.data, out.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    kv_->pos += static_cast<std::uint32_t>(n); // host mirror; equals device io_.pos at boundary
    const auto stop = std::find_if(out.begin(), out.end(),
                                   [&](int token) { return is_stop_token(token); });
    if (stop != out.end()) { out.erase(stop + 1, out.end()); }
    return out;
}
```

### Task 6 — Enable the single-token graph on the fallback path (`engine.cpp`)

**File:** `src/runtime/engine.cpp`

`decode_step_one` currently gates the graph off whenever MTP is enabled. Current (lines 625-645):

```cpp
int Engine::decode_step_one() {
    if (kv_->pos >= kv_->max_context) {
        throw std::out_of_range("Engine::decode_step exceeds max_ctx");
    }
    const bool use_graph = options_.use_cuda_graph && options_.mtp_draft_tokens == 0;
    if (use_graph && decode_warmed_) {
        if (!decode_graph_.ready()) {
            decode_graph_.capture(ctx_->stream, [this] { card_->decode_step_record(); });
        }
        decode_graph_.launch(ctx_->stream);
    } else {
        card_->decode_step_record();
        decode_warmed_ = true;
    }
    kv_->advance();
    if (options_.mtp_draft_tokens > 0) {
        kernels::mtp_count_fallback_step(io_.stats, ctx_->stream);
        kernels::mtp_reset_gdn_initial_slot(io_.gdn_initial_slot, ctx_->stream);
    }
    return read_token();
}
```

Replace only the `use_graph` line (remove `&& options_.mtp_draft_tokens == 0`):

```cpp
    const bool use_graph = options_.use_cuda_graph;
```

Everything else in `decode_step_one` stays. The MTP bookkeeping kernels remain after `launch`, so
they run every fallback step (graphed or eager).

### Task 7 — Rewrite the round orchestration and add the capture body (`engine.cpp`)

**File:** `src/runtime/engine.cpp`

Delete `propose_mtp_after_accept` (lines 647-674) and the current `decode_round` (lines 676-700):

```cpp
void Engine::propose_mtp_after_accept(std::uint32_t host_window_base, int host_length, int k) {
    // ... entire function ...
}

std::vector<int> Engine::decode_round() {
    const int k = options_.mtp_draft_tokens;
    if (k <= 0 || !mtp_kv_ || !card_->mtp_enabled()) { return {decode_step_one()}; }
    if (kv_->pos + static_cast<std::uint32_t>(2 * k) > kv_->max_context) {
        return {decode_step_one()};
    }

    const auto host_window_base = kv_->pos;
    const int T = k + 1;
    kernels::mtp_prepare_verify_inputs(io_.token, io_.drafts, io_.pos, io_.window_base,
                                       io_.verify_ids, io_.positions, ctx_->stream);
    card_->target_verify(io_.verify_ids, io_.positions, host_window_base);
    kernels::mtp_accept_tokens(io_.target_tokens, io_.drafts, io_.pos, io_.token,
                               io_.sampled_out, io_.num_sampled, io_.accepted, io_.ar_pos,
                               io_.stats, ctx_->stream);
    kernels::mtp_set_gdn_initial_slot_from_accepted(io_.accepted, io_.gdn_initial_slot,
                                                    ctx_->stream);

    const int host_length = read_i32_scalar(&model::StepState::pos);
    kv_->pos             = static_cast<std::uint32_t>(host_length);
    mtp_kv_->pos         = static_cast<std::uint32_t>(host_length);

    propose_mtp_after_accept(host_window_base, host_length, k);
    return read_sampled_tokens();
}
```

Replace both with the following three functions (device-only body + host orchestration):

```cpp
void Engine::record_propose(int k) {
    const int T = k + 1;
    kernels::mtp_prepare_shifted_ids(io_.verify_ids, io_.token, io_.accepted, io_.shifted_ids,
                                     ctx_->stream);
    Tensor mtp_hidden = io_.prefill_hidden.slice(1, 0, T);
    card_->mtp_forward_batch(io_.shifted_ids, io_.verify_hidden, io_.positions, mtp_hidden, -1,
                             nullptr, nullptr);

    Tensor logits = io_.logits.slice(1, 0, 1);
    Tensor draft0 = io_.drafts.slice(0, 0, 1);
    card_->mtp_sample_from_hidden_row(mtp_hidden, io_.accepted, io_.mtp_ar_hidden, logits, draft0);

    for (int i = 1; i < k; ++i) {
        Tensor prev_token  = io_.drafts.slice(0, i - 1, 1);
        Tensor next_token  = io_.drafts.slice(0, i, 1);
        Tensor next_hidden = io_.prefill_hidden.slice(1, i, 1);
        card_->mtp_forward_ar_step(prev_token, io_.mtp_ar_hidden, io_.ar_pos, next_hidden, logits,
                                   next_token);
        CUDA_CHECK(cudaMemcpyAsync(io_.mtp_ar_hidden.data, next_hidden.data,
                                   io_.mtp_ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                   ctx_->stream));
        kernels::mtp_increment_i32(io_.ar_pos, ctx_->stream);
    }
}

void Engine::record_decode_round() {
    const int k = options_.mtp_draft_tokens;
    kernels::mtp_prepare_verify_inputs(io_.token, io_.drafts, io_.pos, io_.window_base,
                                       io_.verify_ids, io_.positions, ctx_->stream);
    card_->target_verify(io_.verify_ids, io_.positions);
    kernels::mtp_accept_tokens(io_.target_tokens, io_.drafts, io_.pos, io_.token, io_.sampled_out,
                               io_.num_sampled, io_.accepted, io_.ar_pos, io_.stats, ctx_->stream);
    kernels::mtp_set_gdn_initial_slot_from_accepted(io_.accepted, io_.gdn_initial_slot,
                                                    ctx_->stream);
    record_propose(k);
}

std::vector<int> Engine::decode_round() {
    const int k = options_.mtp_draft_tokens;
    if (k <= 0 || !mtp_kv_ || !card_->mtp_enabled()) { return {decode_step_one()}; }
    if (kv_->pos + static_cast<std::uint32_t>(2 * k) > kv_->max_context) {
        return {decode_step_one()};
    }

    const bool use_graph = options_.use_cuda_graph;
    if (use_graph && round_warmed_) {
        if (!round_graph_.ready()) {
            round_graph_.capture(ctx_->stream, [this] { record_decode_round(); });
        }
        round_graph_.launch(ctx_->stream);
    } else {
        record_decode_round();
        round_warmed_ = true;
    }

    return read_round_output();
}
```

> `decode_step()` (currently lines 702-714) is unchanged: it drains `pending_sampled_`, calls
> `decode_round()`, returns the first token and queues the rest.

### Task 8 — Bench path label (`qus_bench_support.cpp`)

**File:** `bench/qus_bench_support.cpp`

Current (lines 397-400):

```cpp
std::string decode_path_name(bool use_cuda_graph, int mtp_draft_tokens) {
    if (mtp_draft_tokens > 0) { return "mtp_eager"; }
    return use_cuda_graph ? "cuda_graph" : "eager";
}
```

Replace with:

```cpp
std::string decode_path_name(bool use_cuda_graph, int mtp_draft_tokens) {
    if (mtp_draft_tokens > 0) { return use_cuda_graph ? "mtp_cuda_graph" : "mtp_eager"; }
    return use_cuda_graph ? "cuda_graph" : "eager";
}
```

---

## Verification (single final phase — run only after all tasks are done)

Run these in order. This is the only place correctness is asserted.

### V1 — Build

```bash
cmake --build build -j
```

Build everything (not just `qus`) so every call site of the changed signatures is compiled,
including `qus_target_verify_bench`. Expected: clean build. If the build still references
`mtp_set_cache_position`, `read_i32_scalar`, `read_sampled_tokens`, `propose_mtp_after_accept`, or a
3-arg `target_verify` / `cache_offset` `mtp_forward_batch` anywhere, fix that call site — those
symbols no longer exist.

### V2 — Graph vs eager parity (the correctness oracle)

Add a parity scenario to `tests/test_engine_mtp_e2e.cpp`. First make the `generate` helper honor the
caller's flag instead of forcing eager. Current (lines 70-80):

```cpp
Run generate(const std::filesystem::path& weights, qus::EngineOptions options,
             const std::vector<int>& prompt, int max_new_tokens) {
    options.device         = 0;
    options.use_cuda_graph = false;
    qus::Engine engine(options);
    engine.load(weights.string());
    Run out;
    out.tokens = engine.generate(prompt, max_new_tokens);
    out.mtp    = engine.mtp_stats();
    return out;
}
```

Replace with:

```cpp
Run generate(const std::filesystem::path& weights, qus::EngineOptions options,
             const std::vector<int>& prompt, int max_new_tokens) {
    options.device = 0;
    qus::Engine engine(options);
    engine.load(weights.string());
    Run out;
    out.tokens = engine.generate(prompt, max_new_tokens);
    out.mtp    = engine.mtp_stats();
    return out;
}
```

Set `options.use_cuda_graph = false;` explicitly inside each existing scenario helper
(`scenario_batched`, `scenario_capacity_fallback`, `scenario_fallback_after_accept`, and any other
that calls `generate`) so their existing expectations are unchanged. Then add a new scenario and
call it from `main`:

```cpp
int scenario_graph_parity(const std::filesystem::path& weights) {
    const std::vector<int> prompt = foundation_prompt_ids();

    qus::EngineOptions eager;
    eager.max_ctx          = 256;
    eager.mtp_draft_tokens = 5;
    eager.use_cuda_graph   = false;
    const Run a = generate(weights, eager, prompt, 24);

    qus::EngineOptions graph;
    graph.max_ctx          = 256;
    graph.mtp_draft_tokens = 5;
    graph.use_cuda_graph   = true;
    const Run b = generate(weights, graph, prompt, 24);

    int failures = 0;
    failures += a.tokens == b.tokens ? 0 : fail("graph/eager MTP token streams differ");
    failures += a.mtp.accepted_tokens == b.mtp.accepted_tokens
                    ? 0
                    : fail("graph/eager accepted-token counts differ");
    return failures;
}
```

```bash
./build/qus_test_engine_mtp_e2e
```

Expected: all scenarios pass, including `scenario_graph_parity` (identical token vectors). This is
the key gate: greedy decoding is deterministic, so graphed and eager MTP must match exactly.

### V3 — Memory/lifetime safety across replays

```bash
compute-sanitizer --tool memcheck ./build/qus_test_engine_mtp_e2e
```

Expected: `0 errors`. This exercises repeated `round_graph_` replays plus the tail fallback, catching
any KV/workspace out-of-bounds or use-after-rewind introduced by capturing the round.

### V4 — Confirm the graph collapses launches

Profile a short MTP decode and confirm one graph launch per round instead of hundreds of kernel
launches (mirror the existing decode-graph nsys evidence in
`docs/bench/q5090-decode-cuda-graph-nsys-report.md`):

```bash
nsys profile -o /tmp/mtp_graph --force-overwrite true \
  ./build/qus --model out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  --mtp-draft-tokens 5 --max-ctx 2048 --prompt "..." --max-new-tokens 64
```

Expected in the report: `cudaGraphLaunch` count ≈ number of MTP rounds; per-round kernel-launch API
overhead effectively gone.

### V5 — Bench label sanity

```bash
./build/qus_bench --mtp-draft-tokens 5 --decode --json | grep decode_path
```

Expected: `"decode_path": "mtp_cuda_graph"` when cuda graph is on; `"mtp_eager"` with
`--no-cuda-graph`.

---

## Risks and Rollback

- **A stray host read/allocation inside the body aborts capture.** `capture()` already uses
  `cudaStreamCaptureModeThreadLocal` and rethrows on failure. If capture throws, the culprit is a
  host sync / `cudaMalloc` / non-async memcpy inside `record_decode_round` → `target_verify` /
  `mtp_forward_*`. Re-audit against Invariant 1; the only legal memcpy in the body is the D2D
  `cudaMemcpyAsync` on `ctx_->stream`.
- **Host mirror drift.** If V2 tokens match but positions look wrong later, verify the mirror rule
  (Invariant 4): `kv_->pos` advances by `n` only in `read_round_output`, by `+1` only in
  `decode_step_one`.
- **Rollback:** the change is contained to the listed files. Reverting Tasks 6-7 (restoring the old
  `decode_round` + `read_i32_scalar` + `mtp==0` gate) returns to the eager MTP path without touching
  kernels.
