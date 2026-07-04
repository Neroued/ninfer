# Chunked Prefill (ubatch) — methodology / design (goal)

> Status: design goal (not yet implemented). Date: 2026-07-02.
> Scope: the **target design** for splitting the Qwen3.6-27B **prefill** path into fixed-size
> sequence chunks ("ubatches"), each pushed through the whole network while carrying KV + GDN
> state, so that peak workspace is bounded by the chunk size instead of the prompt length. This
> document fixes the goal, the locked constraints, the execution model, the subsystems/operators to
> adapt, and the correctness/bench bar. It deliberately leaves task breakdown and sequencing to a
> later implementation plan. **Decode is out of scope and left byte-for-byte unchanged.**

## 1. Context and motivation

Prefill today runs **"wide" / single-shot**: `Qwen3_6_27B::prefill_impl`
(`src/model/qwen3_6_27b.cpp:546-579`) allocates one `x[hidden=5120, T]` residual stream for the
entire prompt and runs all 64 layers once over it, with a per-layer `mark`/`rewind` on the `work_`
bump arena (`run_layers`, `qwen3_6_27b.cpp:511-538`). Every intermediate buffer therefore scales
with the full prompt length `T`.

The kernel-quality walls have already been addressed by earlier phases — the low-bit prefill GEMM
now runs on tensor cores
([`2026-07-01-prefill-linear-foundation-design.md`](2026-07-01-prefill-linear-foundation-design.md))
and GQA prefill attention is a tuned flash kernel
([`2026-07-02-gqa-prefill-flash-attention-design.md`](2026-07-02-gqa-prefill-flash-attention-design.md)).
What remains is a **scheduling / memory** problem: workspace grows linearly with `T`.

Measured per-token workspace draw (from the current schedule):

| Consumer | Bytes / token | Note |
|---|---:|---|
| GDN mixer (peak layer type) | ~288 KiB | dominated by GDN in/out projections + `gated_delta_rule_chunked` stage workspace |
| MLP tail | ~122 KiB | `gate_up[34816,T]`, `silu` out `[17408,T]`, `down` |
| Persistent residual stream `x[5120,T]` (+ `ids`, `positions`, final `xf`) | ~10 KiB (×2 with `xf`) | lives for the whole forward |

Because layers rewind the mixer before the MLP, the per-forward peak is
`persistent + max(mixer, mlp)` ≈ **~300 KiB/token**. That is ~610 MiB at `T=2048` and ~1.22 GiB at
`T=4096`, against a **fixed 4 GiB** `work_` reservation (`EngineOptions::work_bytes`,
`include/qus/runtime/engine.h`). Long prompts either waste a large fixed reservation or eventually
`std::bad_alloc` out of the arena (`src/core/arena.cu`). The GDN stage workspace alone is
`≈ 60 KiB/token` (`compute_workspace_layout`, `src/kernels/kernel/gdn_chunked_common.cuh:50-82`).

The fix is a **sequence-level chunk loop**: process the prompt in fixed-size ubatches, each carried
through all layers, so no workspace buffer is ever wider than one chunk.

### Terminology (disambiguation)

Two unrelated "chunk" concepts must not be conflated:

- **Prefill chunk / ubatch (this document).** A *sequence-level* unit: `chunk_size` consecutive
  prompt tokens run through the whole 64-layer network in one round, KV/GDN state carried to the
  next round. Analogous to llama.cpp's `ubatch`. Startup-fixed, runtime-constant.
- **GDN kernel tile (`kChunkSize = 64`).** An *intra-op* tiling inside
  `gated_delta_rule_chunked` (`gdn_chunked_common.cuh`). It is an implementation detail of one
  operator and is unchanged by this design. Where both appear, this document says "GDN kernel tile"
  for the 64.

## 2. Goal and non-goals

**Goal.** Make prefill run as an ordered sequence of fixed-size ubatches so that:

1. **Peak `work_` usage is `O(chunk_size)`, independent of prompt length `T`.**
2. The freed VRAM is **reclaimed**: the `work_` reservation is *derived from `chunk_size`* at load
   time (a few hundred MiB, not a fixed 4 GiB), leaving headroom to grow the cache arena / raise
   `max_ctx`.

**Non-goals.**

- Decode (`Engine::decode_step`, the split-KV decode kernel, CUDA-graph capture) — unchanged.
- The GQA flash kernel's tuning envelope and the low-bit linear GEMM — unchanged except an additive
  offset parameter on the attention path (§6).
- Multi-request / cross-sequence batching. Batch is 1; ubatches are within one prompt.
- Bit-exact parity with the old single-shot path (see §3, constraint 4).
- A second weight or KV layout. Prefill continues to adapt to the existing storage.

## 3. Locked constraints

1. **Replace, do not branch (AGENTS.md — no backward compatibility).** Prefill is *always* chunked.
   A prompt of `T ≤ chunk_size` is simply a one-chunk run; there is no separate single-shot code
   path kept beside the chunked one. The current wide `prefill_impl` body is rewritten, not
   duplicated.
2. **Chunk-outer / layer-inner (the ubatch model).** For chunk `k` covering absolute tokens
   `[t0, t0+len)` (`t0 = k·chunk_size`), run all 64 layers, writing KV at absolute `[t0, t0+len)`
   and updating GDN state in place, then discard the chunk's activations and advance. The residual
   stream is **never** needed across chunks — all cross-chunk information lives in the KV cache and
   GDN state.
3. **`chunk_size` is a power-of-two-friendly multiple of 128, startup-fixed.** Only multiples of 128
   are supported (default **1024**). This is deliberate: 128 is a multiple of the GDN kernel tile
   (64), so every non-final chunk is GDN-fast-path aligned; and `chunk_size ≥ 128 > τ=16` keeps
   every prefill linear in the tensor-core MMA regime (`classify_regime`,
   `src/kernels/linear/plan/linear_plan.cpp`). The **final** chunk (`T mod chunk_size`) may be
   unaligned and reuses GDN's existing chunked-plus-recurrent-tail wrapper
   (`src/kernels/wrapper/gated_delta_rule.cpp`).
4. **Parity is numerical, not bitwise.** Splitting the sequence changes reduction/accumulation order
   — chiefly GDN's cross-chunk `ssm_state` passing and the shifted boundary between GDN's chunked
   fast path and its recurrent tail. Correctness is therefore judged by a **normwise closeness**
   criterion against each op's own oracle (§8), consistent with the existing `linear_tc` / flash-attention
   `rel_l2` oracles in [`l1-op-test-standard.md`](l1-op-test-standard.md).
5. **Single weight copy; single KV/state copy; decode owns its layouts.** Unchanged from the prior
   prefill phases: prefill adapts to on-chip storage, storage never adapts to prefill.

## 4. Why chunk-outer bounds workspace (memory model)

The key property: in chunk-outer scheduling, a chunk's residual stream `x[5120, len]` is fully
consumed by the time the chunk finishes all 64 layers. For an attention layer, the chunk's
contribution is captured when its K/V are written to the cache; for a GDN layer, when `ssm_state` /
`conv_state` are updated. Nothing downstream reads chunk `k`'s hidden activations after chunk `k`
completes. Therefore:

- **Persistent workspace** shrinks from `O(T)` to `O(chunk_size)` — only the current chunk's `x`,
  `ids`, `positions` are live (`~10 KiB/token × chunk_size`). The final `xf` and `lm_head` touch
  only the **last chunk's last token**.
- **Per-op scratch** (GDN mixer, MLP, GDN stage workspace) is allocated at width `len ≤ chunk_size`,
  so its peak is `max(mixer, mlp) × chunk_size ≈ 288 KiB/token × chunk_size`.

Per-chunk peak ≈ `~300 KiB/token × chunk_size`:

| `chunk_size` | approx `work_` peak |
|---:|---:|
| 128 | ~38 MiB |
| 256 | ~75 MiB |
| 512 | ~150 MiB |
| 1024 | ~300 MiB |

This is **flat across prompt length** — a 128k-token prompt at `chunk_size=512` peaks at the same
~150 MiB as a 512-token prompt. The KV cache and GDN state (the separate `cache_arena`, sized by
`max_ctx` in `Engine::default_cache_bytes`) are unchanged; the point of reclaiming the work arena is
that raising `max_ctx` no longer also inflates the (previously `O(T)`) workspace.

**Total compute is unchanged.** Causal attention over the whole prompt is `T(T+1)/2` key-dots
whether tiled as one launch or `⌈T/chunk_size⌉` launches; the linear GEMM work is `N·K·T` summed
over chunks either way. Chunking changes *how* the work is tiled into launches, not its asymptotic
amount (§9 covers the finite-size efficiency cost).

## 5. Execution design

### 5.1 The chunk loop (lives in the L2 model card)

`Qwen3_6_27B::prefill_impl` is rewritten from a single wide pass into a loop over ubatches. The
model card owns the loop because it already owns the static schedule, position filling, `kv.pos`,
and the last-token `lm_head`. `Engine::prefill` (`src/runtime/engine.cpp:206-216`) keeps its
**one-time** KV + GDN reset and passes `chunk_size` in; it does **not** slice the prompt.

Per chunk `k` (tokens `[t0, t0+len)`):

1. `work_.reset()` — the arena returns to zero at each chunk boundary (bounding the peak).
2. Copy `ids[t0:t0+len)` to device; fill `positions = [t0 … t0+len-1]` (§6, RoPE offset).
3. `embed_gather` → `x[5120, len]`.
4. `run_layers(x, Phase::Prefill)` — identical per-layer schedule, now at width `len`; attention
   layers pass `cache_offset = t0` (§6); GDN/conv layers read+update carried state.
5. If **not** the last chunk: advance `kv.pos += len` and continue (no final norm / no `lm_head`).
6. If the **last** chunk: final `rmsnorm` + `lm_head` + `argmax` on the last token only, then set
   `kv.pos = T` and `io_.pos = T`.

The `Phase::Prefill` branch inside `attn_mix` / `gdn_mix` / `mlp_tail` is unchanged in structure; it
simply runs on `len`-wide activations and threads `t0` into attention.

### 5.2 State carried across chunks (not reset mid-prompt)

- **KV cache** — layout `[head_dim, padded_context, n_kv]`, indexed by absolute position
  (`src/core/kv_cache.cpp`). Already correct for offset writes; only `reset()` (which just zeroes
  `pos`) must be confined to prompt start.
- **GDN `ssm_state`** — in-out FP32 `[128,128,48]` per GDN layer; `gated_delta_rule_chunked` already
  loads and stores it (`src/kernels/launcher/gated_delta_rule_chunked.cu`). Carries automatically if
  not reset.
- **GDN `conv_state`** — in-out `[conv_dim, 3]` per GDN layer; `causal_conv1d_prefill` seeds from it
  when a chunk's first tokens need history and writes the trailing samples back
  (`src/kernels/kernel/causal_conv1d.cuh`). Carries automatically if not reset.

The single behavioral change is **where** reset happens: `Engine::prefill` resets KV + GDN once
before the loop; nothing resets between chunks.

## 6. Subsystems & operators to adapt

Classified by how each operator meets chunking: **(A)** trivially per-token (slice only), **(B)**
chunkable with carried state / absolute positions (no kernel math change), **(C)** needs a kernel
signature/logic change.

### (C) GQA prefill attention — the one real kernel change

> Superseded API note (2026-07-04): the public GQA surface is now the unified
> `gqa_attention(..., positions, ...)`; the host `cache_offset` path below is historical context.

`gqa_attention_prefill` (kernel `src/kernels/kernel/gqa_attention_prefill.cuh`, launcher
`.../launcher/gqa_attention_prefill.cu`, wrapper `.../wrapper/gqa_attention.cpp`) today assumes the
chunk *is* the whole prompt: its fill writes cache slots `[0, T)` and its causal mask compares
**local** query/key indices. Add a `cache_offset (t0)` parameter threaded public API → wrapper →
launcher → kernel:

- **Fill:** write cache slot `t0 + token` instead of `token`.
- **Attention:** each local query row `r` (absolute `t0+r`) iterates key blocks from absolute 0 and
  masks `key_abs ≤ t0 + r`; the key-range upper bound becomes `t0 + len`. Prior chunks' K/V are
  already resident in the cache.

This is a **parameter extension of the tuned flash kernel** (~10–15 index/mask sites), not a new
kernel and not a decode-style rewrite. Validation `cache_offset + len ≤ max_context`. The decode
kernel is the right *semantic* analogue (query attends all cached history) but not a usable
implementation (wrong parallelism, per-launch single query).

### (B) Positions / RoPE, GDN, conv — carried state, no math change

- **`fill_positions`** (`src/model/position.cu`) must emit `[t0 … t0+len-1]`; the RoPE kernel is
  already per-position.
- **`gated_delta_rule_chunked` + recurrent tail** — no change; correctness depends only on
  `ssm_state` carry and on `chunk_size` being a multiple of 128 so non-final chunks stay on the
  fast path.
- **`causal_conv1d_prefill`** — no change; depends only on `conv_state` carry.

### (A) Per-token operators — slice only

`embed_gather`, `rmsnorm` (input / q-k / post-attn / final / GDN-gated), `linear`
(GEMV/multistep/MMA), `l2norm`, `silu_and_mul`, `sigmoid_gate_mul`, `residual_add`,
`gdn_in_ab_gated_prefill`, and the host pack/split memcpy helpers. Each operates on the chunk's
`[·, len]` slice with no cross-token coupling.

### Orchestration / lifecycle

- Reset KV + GDN **once** at prompt start (`Engine::prefill`), never between chunks.
- Advance `kv.pos` by `len` per chunk; set `io_.pos = T` only after the last chunk.
- Run final norm / `lm_head` / `argmax` only on the last chunk's last token.

### Configuration threading

- `EngineOptions::prefill_chunk` (`include/qus/runtime/engine.h`), validated `> 0` and a multiple of
  128; default 1024.
- `--prefill-chunk N` on the CLI (`include/qus/text/cli.h`, `src/text/cli.cpp`, `src/main.cpp`) and
  on `qus_bench` (`bench/qus_bench_support.*`).

### Workspace reclamation (formula-derived)

At load, size the `work_` reservation from a **closed-form function of `chunk_size`**:
`work_bytes(chunk) ≈ persistent(chunk) + max(mixer(chunk), mlp(chunk))` with the GDN stage term from
`gdn_chunked_common.cuh` evaluated at `L = chunk_size`, plus a small safety margin. This is
deterministic and documented (denominators from §4). The default is formula-derived;
`--work-bytes` may still override it for experiments.

## 7. Implementation sequencing (bottom-up; op-first)

Adapt and **prove the operators before the engine**, so chunking is validated at the lowest layer
first and the integration step inherits already-trusted kernels:

1. **Ops + op tests (the gate).** Add `cache_offset` to GQA prefill (fill + flash kernel + launcher
   + wrapper + public header) and extend its parity test; add the GDN and conv **state-carry
   equivalence** tests (§8). No engine code changes in this step. Exit only when every adapted op
   passes its numerical oracle and `compute-sanitizer`.
2. **Engine / orchestration.** Only after step 1: rewrite `prefill_impl` into the chunk loop, move
   the KV+GDN reset to once-per-prompt in `Engine::prefill`, offset `fill_positions`, and confine
   final-norm / `lm_head` / `argmax` to the last chunk.
3. **Config + workspace + bench.** Thread `--prefill-chunk`, switch `work_bytes` to the
   formula-derived size, and extend the bench report schema.

Rationale: the only real math change lives in the ops (chiefly GQA `cache_offset`, plus the
state-carry paths). Proving those in isolation means an integration failure can only be wiring, not
kernel math — which is why no end-to-end reference capture is needed (§8).

## 8. Correctness & verification bar

Verification is **bottom-up and op-local**. Each adapted operator is proven against its **own
oracle**; there is **no end-to-end dump/reference comparison**. Whole-model correctness is confirmed
simply by running inference, and analysis is escalated **only if** that inference misbehaves. This is
an AGENTS.md numerical + GPU-lifetime hard-whitelist case.

**Op-level numerical verification (the correctness gate).**

- **GQA prefill attention with `cache_offset`.** Against the fp32 causal-GQA oracle on `(qn, cached
  K, V)`: with prior keys `[0, t0)` pre-filled, queries `[t0, t0+len)` must match the oracle
  attending `[0, t0+r]`. Offset/shape matrix over `t0 ∈ {0, 128, …}`, `len ∈ {128, 512, …, non-128
  tail}`, full 24/4 heads; pass by the normwise `rel_l2` band from
  [`l1-op-test-standard.md`](l1-op-test-standard.md) (`rel_l2 ≲ 4e-3`) plus max-abs. Extends the
  `tests/kernels/` harness that today hardcodes `t0=0`.
- **GDN state-carry equivalence.** `gated_delta_rule_chunked` run once over `[0, T)` vs. run as
  consecutive calls `[0, t0)` then `[t0, T)` with `ssm_state` carried (never reset between calls)
  must agree within the same normwise band — including a split that is **not** a multiple of the GDN
  kernel tile (exercising the recurrent-tail boundary). This is the op-level proof that the SSM
  recurrence resumes correctly — the core "can this op support ubatching" question (non-bitwise,
  constraint 4).
- **conv state-carry equivalence.** The same split-vs-one-shot check for `causal_conv1d_prefill`
  with `conv_state` carried; the FIR filter should agree tightly.

**Sanitizer (mandatory).** `compute-sanitizer` `memcheck` + `racecheck` + `initcheck` on the new
`cache_offset` fill/attention path and the not-reset-between-chunks state lifetime — the
offset/aliasing/lifetime surface AGENTS.md calls out.

**Engine integration — confirm, don't dump-compare.** Once the ops pass, the chunked engine is
validated by **running inference**: greedy decode on canonical prompts produces coherent, correct
output, stable across `chunk_size ∈ {128, 512, whole-prompt}`. No captured-reference `rel_l2` is
required. **Only if** integrated inference is wrong do we escalate to targeted comparison/bisection
to localize the break (e.g. the existing per-layer taps), and even then as a debugging tool, not a
standing gate.

**Bench evidence (success metric, not a correctness gate).** `workspace_peak_bytes` collapses to the
`O(chunk_size)` bound and is flat across prompt length; prefill tok/s at a fixed total length stays
within a stated tolerance of the current baseline (the finite-size efficiency cost in §9 is the
tunable trade).

## 9. Performance considerations (the `chunk_size` trade)

Chunking preserves total compute but changes per-launch shapes, so `chunk_size` trades workspace
against kernel efficiency:

- **Attention.** Early chunks are short-context (`K ≈ chunk_size`), where the flash kernel's
  tensor-core efficiency is lower (the achieved sweep shows ~20% at `T=512` rising to ~82% at
  `T=8192`); later chunks have long `K` (good arithmetic intensity) but only `chunk_size` query rows
  (`24 × ⌈chunk_size/64⌉` CTAs). `chunk_size=512` → 192 attention CTAs, enough to fill the SMs;
  `chunk_size=128` → 48 CTAs, likely underfilled.
- **Linear GEMM.** Efficiency at `T=chunk_size` columns; `chunk_size ≥ 128` keeps every matrix on
  the MMA path, larger chunks amortize weight reuse better.
- **Launch overhead.** `⌈T/chunk_size⌉` × more launches; negligible at `chunk_size ≥ 512`.

Hence the default **1024**: it favors attention/GEMM throughput and fewer chunk launches while
keeping workspace bounded at O(chunk). `chunk_size` is the single knob exposing this trade; the bench
length-sweep should report tok/s and `workspace_peak_bytes` per `chunk_size` so the default can be
confirmed against evidence.

## 10. Benchmark & report schema (whitelisted contract)

- Add `config.prefill_chunk` to the `qus_bench` JSON/CSV/table
  (`bench/qus_bench_support.*`, `schema_version` bump if the contract test requires); it is a run
  config, so it lives under `config.*`, not per-test.
- `workspace_peak_bytes` (already reported per test) now reflects the bounded peak — the primary
  success signal.
- Fix `bench/README.md`'s CSV column list, which currently omits `workspace_peak_bytes`, and
  document `--prefill-chunk`.
- The bench-support schema contract test (`tests/test_qus_bench_support.cpp`) is updated to include
  the new field (AGENTS.md whitelists CLI/report schema).

## 11. Integration surface (context, not a task list)

Ordered to match the bottom-up sequencing (§7): ops first, then engine, then config/bench.

- **Attention op (step 1):** `src/kernels/kernel/gqa_attention_prefill.cuh`,
  `src/kernels/launcher/gqa_attention_prefill.cu`, `src/kernels/wrapper/gqa_attention.cpp`,
  `include/qus/kernels/gqa_attention.h` (add `cache_offset`).
- **Op tests (step 1):** the attention parity harness in `tests/kernels/` (offset/shape matrix) plus
  the GDN and conv state-carry equivalence tests.
- **Model card / orchestration (step 2):** `src/model/qwen3_6_27b.cpp` (`prefill_impl` → chunk loop;
  thread `t0`) and `src/model/position.cu` (`fill_positions` offset). Public `prefill(...)` signature
  unchanged.
- **Runtime / config (steps 2–3):** `include/qus/runtime/engine.h` (`EngineOptions::prefill_chunk`,
  formula-derived `work_bytes`), `src/runtime/engine.cpp` (reset once; pass chunk size).
- **CLI (step 3):** `include/qus/text/cli.h`, `src/text/cli.cpp`, `src/main.cpp` (`--prefill-chunk`).
- **Bench / report (step 3):** `bench/qus_bench.cpp`, `bench/qus_bench_support.*`, `bench/README.md`,
  `tests/test_qus_bench_support.cpp`.
- **Unchanged:** KV cache and GDN state layouts (`src/core/kv_cache.cpp`, `src/core/state_store.cpp`);
  GDN and conv kernels; all class-(A) element-wise/linear kernels; the entire decode path.

## 12. Preserved freedom / deferred to the plan

The exact `work_bytes(chunk)` safety margin, the attention key-block loop structure for `t0>0`
(iterate-from-0 vs. base-block skip), and the final `chunk_size` default are open to
evidence-driven tuning under the bench sweep and `compute-sanitizer`; none changes the contract
above. This document fixes *what* to build, *why it bounds workspace*, and the *bottom-up order*
(§7); the detailed per-task breakdown and final constants belong to a later implementation plan.
