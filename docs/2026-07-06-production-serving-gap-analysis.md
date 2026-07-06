# Production Serving Gap Analysis (single-user endpoint)

> Status: analysis / design note. Date: 2026-07-06.
> This is a gap inventory, **not** a formal implementation plan. It records the distance between
> the current CLI-only engine and a production OpenAI-compatible endpoint for **one user's stream
> at a time**, and classifies each gap by the depth of change required.
> Scope decisions captured here: (1) production target is a **single-user** endpoint; (2) **MTP
> speculative decode stays on under every sampling mode** (it is a primary speed lever, not
> optional). These two decisions determine the entire classification below.

---

## 1. Framing and scope decisions

The engine was built on a frozen assumption: *"single user, single sequence, batch = 1"*
(`docs/design.md` §1). Every core structure encodes it — tensors are `[rows, T]` with `T` the token
count of **one** sequence (there is no batch dimension), `KVCache` is one contiguous buffer for one
stream, `GdnState` is one running recurrent state, `StepState` (`io_`) is one scratch set, and the
decode step (and the MTP round) are captured into single CUDA graphs.

Because the production target is **single-user**, the core architecture does **not** need to change.
Everything below is either a serving shell around the existing engine or new kernels + narrow engine
hooks that keep the batch=1 design intact.

**Explicitly out of scope for single-user (would have been the only architectural rewrites):**

- Continuous batching / concurrent in-flight sequences.
- Paged KV / global cross-user prefix cache.

**Hardened by decision (2):** because MTP must remain on even when the user samples, making MTP
**distribution-correct under sampling** is a mandatory requirement, not a tradeoff. It is the single
deepest and highest-risk item in the whole effort.

---

## 2. What exists today (grounding)

| Capability | State |
|---|---|
| Entry point | Single blocking CLI (`src/main.cpp`) — one process, one request, then exit |
| Concurrency | None; `Engine` holds mutable state (`io_`, `kv_`, `state_`, `work_`, `pending_sampled_`), not thread-safe |
| Prefill | Always **resets** KV + GDN state, positions start at 0 (`engine.cpp:624-628`, `qwen3_6_27b.cpp:897-907`); no "continue/append" path |
| Token selection | **Greedy argmax only** (`qwen3_6_27b.cpp:1022`); no temperature/top-k/top-p/penalties/RNG anywhere in the tree |
| Logits | Fully materialized in `io_.logits` (BF16 `[vocab, window]`) — a sampler has the raw material it needs |
| Decode graph | Argmax is captured **inside** the decode CUDA graph; the MTP verify+propose round is captured in a second graph (`round_graph_`) |
| KV cache | One contiguous per-full-attn-layer buffer sized to `max_ctx`; `rewind()` exists (built for MTP) |
| GDN state | Fixed-size fp32 recurrent + conv state; **snapshot slots already exist** (built for MTP verify) |
| Spec decode | MTP in progress; **greedy-verify only** (roadmap M5: "概率采样仍在范围外") |
| Text frontend | C++ Qwen tokenizer + chat template + `TokenStreamDecoder` streaming — solid foundation |
| Context | `max_ctx` fixed at load; official gate 8192; model capacity 256K |
| Error model | Exceptions bubble to `main` and terminate the process |

Key consequence: the 27B model load is expensive, so the engine **must be resident** in the serving
process. A "shell out to the CLI per request" design is a non-starter (model load would dominate
every request).

---

## 3. Gap inventory

### A. Serving/runtime infrastructure (all single-user-compatible)

1. **Request serialization + lifecycle.** No queue and no guard around the non-thread-safe `Engine`.
   Even a single user's client can issue overlapping requests; requests must be serialized, not
   allowed to corrupt in-flight state.
2. **Cancellation / client disconnect.** The decode loop runs to `max_new_tokens`/stop with no abort
   path; a disconnected SSE stream keeps burning GPU time.
3. **Crash isolation.** A bad request or a CUDA error currently kills the process; a server must
   catch per-request, return HTTP 4xx/5xx, and keep the GPU process alive. OOM on an oversized prompt
   must be a rejected request, not a dead server.
4. **Warmup.** CUDA graph capture is lazy on the first decode/round; the first request pays capture
   latency. Needs an explicit startup warmup.
5. **Observability/ops.** Health/readiness endpoints, metrics (tok/s, TTFT/ITL, GPU memory), structured
   logging, graceful shutdown, config/model-path management.

### B. API-surface completeness (what apps actually send)

6. **Full sampling params**, not just top-k + temperature: top-p (nucleus), min-p,
   `frequency_penalty`/`presence_penalty`/repetition penalty, `logit_bias`, and a **seed** for
   reproducibility.
7. **`logprobs` / `top_logprobs`** — top-k logit readback from `io_.logits`.
8. **Stop *strings*** — the engine currently stops on token ids only; clients send arbitrary strings,
   which require detokenized-suffix matching. Plus `n > 1` completions.
9. **Tool/function calling + `response_format` (JSON mode)** — this is constrained decoding: per-step
   logit masking driven by a grammar/FSM. A subsystem, not a flag.
10. **Reasoning-content mapping** — `enable_thinking` already exists; map it to the API's reasoning
    field.

### C. Correctness interactions (the traps — harder because MTP is always on)

11. **Sampling × MTP (mandatory, deepest item).** MTP verify is greedy-only. With MTP forced on under
    sampling, a naive sampler yields the **wrong output distribution**. The acceptance test in the
    MTP round must become **distribution-correct** (rejection sampling / typical acceptance), and the
    sampler must live inside **both** the single-step decode path **and** the k-token verify/propose
    round — it cannot be a post-argmax replacement. Top correctness risk.
12. **Sampling × CUDA graph.** Argmax is captured inside the decode graph (and the round graph). To
    vary temperature/top-k per request (or per-step penalties) without re-capturing, sampling
    parameters must be **device-side buffers** the kernel reads (the graph replays the same pointers
    while the values change), or the sampling stage must be structured so it can be updated inside the
    captured round. Given (11) this must be solved within the graphed round.
13. **Per-request parameter isolation.** `max_ctx`, stop tokens, and MTP are load-time-fixed in
    `EngineOptions`. Sampling params, stop, seed, and `max_tokens` must become **per-request** and be
    plumbed into the decode/round step.

### D. Prefix caching for TTFT (single-session form)

14. In the single-user world, "prefix caching" means **multi-turn conversation reuse**: keep KV + GDN
    state resident between turns and prefill only the *new* turn. Two engine facts make this a real
    capability gap, not a wrapper:
    - Full-attention layers (16) reuse KV normally (absolute positions, RoPE-applied).
    - **GDN linear-attention layers (48) hold a *running* recurrent+conv state that cannot be indexed
      into.** Continuation requires **restoring the fp32 SSM+conv snapshot at the turn boundary** —
      the snapshot machinery already exists (built for MTP), so the foundation is present.
    - Prefill currently hardcodes position 0 and resets state; a **`prefill_append`** that starts at
      `kv_->pos` without resetting is the missing hook.

### E. Context & memory

15. **Context-length policy.** `max_ctx` is fixed at startup and the KV pool is sized once; need a
    policy for prompts exceeding the window and a memory budget for longer contexts (128K KV ≈ 8 GB).

---

## 4. Classification by modification depth

### Tier 1 — Wrapping / additive (no engine internals touched)

Reuses `Engine::generate` / `TextGenerationRunner` and its existing `stream_callback`.

| Item | Notes |
|---|---|
| OpenAI HTTP server + schema translation + SSE | `messages` → existing `render_qwen_chat`; `nlohmann/json` already vendored |
| Request queue / serialization guard + lifecycle + crash isolation | Makes the endpoint safe without touching engine internals |
| Warmup, health/metrics/logging, graceful shutdown | Ops layer |
| `max_tokens`, stop, `n=1`, `logprobs` readback, error mapping | Runner already supports max_new + stop ids; logprobs = small device→host read |

### Tier 2 — New kernels + narrow engine hooks (batch=1 preserved)

| Item | Depth |
|---|---|
| **Sampler** (temperature, top-k, top-p, min-p, penalties, RNG/seed) over `io_.logits` | Must integrate with device-side param buffers for the graph; see Tier 3 for the MTP coupling |
| **`prefill_append`** (continue without resetting KV/GDN; positions from `kv_->pos`) → single-session multi-turn prefix caching | Generalize `prefill_impl` to a nonzero start offset (today `t0` starts at 0); restores the existing GDN snapshot |
| Stop-**string** matching in the decode loop | Extend `TokenStreamDecoder` logic into the stop path |
| Per-request sampling/stop/seed plumbing into decode/round | Move params out of load-time `EngineOptions` |
| Configurable/longer `max_ctx` + KV memory budgeting | Config + pool sizing; verify attention kernels at larger T |

### Tier 3 — Deep engine change (still single-sequence, but hard)

| Item | Why it's deep |
|---|---|
| **Sampling-correct MTP** (rejection-sampling acceptance inside the captured verify/propose round) | Mandatory given "MTP always on"; rewrites the greedy MTP accept kernels and couples the sampler to the graphed round. **Top risk.** |
| **Grammar/JSON-constrained decoding** (tool calling, `response_format`) | New FSM + per-step logit-mask subsystem — only if these API features are in scope |

**Not needed for single-user (removed from scope):** paged KV, continuous batching / concurrent
sequences, cross-user global prefix cache. These were the only items that would have forced an
architectural rewrite.

---

## 5. Minimal viable server achievable *today* (greedy-only, no sampler, no prefix caching)

Everything required for a working OpenAI-compatible endpoint that streams already exists: a resident
engine, greedy generation, MTP (greedy, correct), the Qwen chat template, the streaming
`TokenStreamDecoder`, and stop-token handling. A minimal endpoint is therefore **Tier-1 only**:

- Hold one **resident** `Engine` in the serving process (never reload per request).
- Translate `POST /v1/chat/completions` `messages` → `ChatMessage[]` → `TextGenerationRunner::generate`.
- Stream via SSE using the existing `stream_callback`; support non-streaming by concatenating.
- **Serialize** requests (one at a time) with a mutex/queue.
- **Accept-and-ignore** unsupported sampling params (`temperature`, `top_p`, …) and advertise
  greedy-only, so standard OpenAI clients (which always send these) still work. Honor `max_tokens`
  and `stop`.
- Keep **MTP on** — greedy MTP is already correct; it only affects speed.
- Reject prompts that exceed `max_ctx` with a clean HTTP error.

Correctness note: because prefill already resets KV/GDN state, each request is independent — there is
no state bleed between requests without any extra work. The only functional cost of skipping prefix
caching is TTFT: each turn re-prefills the whole conversation the client sends.

This path defers Tier-2 (sampler, `prefill_append`) and Tier-3 (sampling-correct MTP,
constrained decoding) without blocking them later.

---

## 6. Bottom line

For a single-user endpoint, **no architectural change is required.** The full distance is: a Tier-1
serving shell (protocol, serialization, lifecycle, ops), a Tier-2 sampler + `prefill_append` for
sampling and multi-turn TTFT, and one genuinely hard Tier-3 item — **making MTP speculative decode
distribution-correct under sampling**, which the "MTP always on" decision promotes from optional to
mandatory and makes the largest correctness risk. Grammar-constrained decoding is the only other
deep piece, and only if JSON/tool-calling is in the API scope.

A **greedy-only OpenAI server is achievable immediately** with Tier-1 work alone.
