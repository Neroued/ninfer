# q5090 — Decode CUDA Graph (fixed-split attention + full-step capture/replay)

> **For agentic workers:** REQUIRED SUB-SKILLS: `superpowers:subagent-driven-development` (dispatch one
> fresh subagent per task, two-stage review), `superpowers:systematic-debugging` (when a kernel
> regresses or `compute-sanitizer`/parity trips). Performance/graph subagents MUST read first:
> `/home/neroued/.cursor/skills/profile-cuda/SKILL.md`,
> `/home/neroued/.codex/skills/ncu-kernel-profile/SKILL.md`,
> `/home/neroued/.codex/skills/nsys-inference-analysis/SKILL.md`. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reclaim the ~28% GPU-idle, host-launch-bound decode wall (see
[../bench/q5090-v2-current-qus-decode-nsys-report.md](../bench/q5090-v2-current-qus-decode-nsys-report.md):
`1429` launches/step, `cudaLaunchKernel` 20.1 s, kernel-active share 71.83%) by capturing the entire
single-token decode step as **one** CUDA graph and replaying it per token.

**Architecture:** The decode step is already shape-invariant and address-stable across steps (bump-arena
workspace is deterministic; `io_.token`/`io_.pos` are device-resident and flow device→device) **except**
the GQA full-attention launch, whose grid/scratch are derived from the host `kv.pos`
(`src/kernels/wrapper/gqa_attention.cpp` `decode_tile_n`/`tile_count`). We first make decode attention
launch-invariant by switching its KV partition from *fixed keys-per-tile, variable tile-count* to a
**fixed number of KV splits** (flash-decoding style; this is how vLLM keeps decode graph-capturable —
graphs keyed by batch size, sequence length read on-device, split count fixed). With the attention grid
constant, the whole step is one static graph captured once and replayed.

**Tech Stack:** CUDA 13.x, RTX 5090 (sm_120), C++20, CUDA stream-capture graphs (`cudaStreamBeginCapture`
/ `cudaGraphInstantiate` / `cudaGraphLaunch`), the existing per-op fp64 oracle + cold-cache bench + nsys.

---

## Why one graph is enough for all context lengths (design rationale)

`TileN` in the current decode kernel is **not** an inner vectorization width — it is the number of tokens
one CTA processes (one token per loop iteration), and `tile_count = ceil((pos+1)/TileN)` is the CTA count
(`grid.y`). So the launch grows with position. The fix inverts the partition: **fix the CTA count
(`NUM_KV_SPLITS = S`) and let keys-per-split (`kps = ceil((pos+1)/S)`) vary**, read from the device `pos`.
`S` does not depend on position or `max_context`, so a single captured graph is valid for every token and
every context length — adding context never adds graphs. This mirrors vLLM (`csrc/.../attention_kernels.cuh`
fixed `PARTITION_SIZE` + grid sized once, early-return empty partitions, reduce bounded by device
`seq_len`; `vllm/v1/.../triton_decode_attention.py` fixed `grid = (batch, head, NUM_KV_SPLITS)`).

The current kernel is already ~90% ready: it has a **neutral-partial path** for empty tiles
(`src/kernels/kernel/gqa_attention_decode.cuh:189-198`), the **reduce already has a fixed grid**
`(24, 8)` and tolerates neutral partials, and the **current token is read from `k_new`/`v_new`, not
cache** (so the cache append at `pos` has no intra-step read-after-write hazard). This is a partition
relabeling plus a capture harness, not a rewrite.

## Goal / non-goals

**Goal (restated):** decode wall drops toward the kernel-sum floor (~41 s in the cited report);
`cudaLaunchKernel`/step collapses ~1429 → ~1; greedy output is token-for-token identical to the eager path.

- **Non-goal — prefill.** Prefill stays eager (variable `T`, already ~1.3 s). Only the `T==1` decode step
  is refactored and graphed. `gqa_attention_prefill` and its kernel are untouched.
- **Non-goal — model schedule / weights / KV layout / formats.** No change to `qwen3_6_27b.cpp` block
  math, the weight store, the q5090 byte contract, or any `qtype`. The only `qwen3_6_27b.cpp` edit is the
  host-vs-device step split in Task 3.
- **Non-goal — sampling change.** Decode is device-side greedy `argmax` (already graph-friendly); no host
  RNG sampling is added.
- **Non-goal — multi-sequence batching.** qus is single-sequence on one 5090, so the decode graph count is
  exactly **one** (no batch-size capture set like vLLM).
- **Correctness gate is the fp64/tolerance oracle + greedy token parity**, not bit-identity. The
  fixed-split partition reorders the flash reduction vs the current tile grouping, so values differ at ULP
  level by construction; the existing `qus_gqa_attention_test` tolerance (`Tolerance::attention_bf16()`)
  and end-to-end greedy parity are the gates.

## Execution mode — subagent-driven, four sequential tasks, linear mainline

A **coordinator** dispatches one fresh strongest-model subagent per task, integrates onto `master` as one
conforming commit, and verifies. Per the user's split (Task 4 added to honor the e2e report contract that
graph-default decode changes — see Task 4):

```
Task 1  fixed-split GQA decode refactor (correctness)      -> refactor(q5090): commit
   │  (one graph becomes possible; attention launch is now position-invariant)
Task 2  GQA decode perf tune (single kernel, ON MAINLINE)  -> perf(q5090): commit
   │  (no worktree — only one kernel; reuse the push-down per-kernel loop in place)
Task 3  CUDA graph capture/replay harness (DecodeGraph)    -> feat(q5090): commit
   │
Task 4  e2e report contract: reflect graph decode          -> feat/test(q5090): commit
   │
Review  independent strict review subagent (kernels + numerics + GPU memory + runtime lifetime + report schema)
```

- **Dependencies:** Task 2 and Task 3 both depend only on **Task 1** (the fixed grid). Task 4 depends on
  **Task 3** (it reflects the now-graph-default decode in the report). They touch disjoint files (Task 2:
  attention `.cu`/`.cuh` + the shared constant; Task 3: engine + model + a new `DecodeGraph`; Task 4: the
  e2e bench/report/tooling). Default to sequential 1 → 2 → 3 → 4 so `S` is finalized before the graph
  captures the kernel (re-tuning `S` after Task 3 only needs a rebuild + re-capture, which the harness does
  on load).
- **Task 2 runs directly on `master`** (the user's instruction): there is exactly one kernel to tune, so
  the push-down protocol's *per-kernel tuning loop* is reused **in place** (no detached worktree, no
  parallel kernels). Round 0 baseline → discovery rounds → best round, gated by the fp64 oracle.
- **Shared split constant.** `kGqaDecodeSplits` (`S`) lives in **one** header both C++ (`.cpp` wrapper) and
  CUDA (`.cu`/`.cuh`) include — the public `include/qus/kernels/gqa_attention.h` — so the wrapper, launcher,
  and bench all read the same value and cannot diverge. The decode kernel itself derives the split count
  from `gridDim.y` (like vLLM's `max_num_partitions = gridDim.z`), so the `.cuh` needs no constant and the
  `.cpp` wrapper never includes a kernel `.cuh`. Task 1 introduces it; Task 2 owns its value.

## Reference facts (verified in tree)

- RTX 5090 (sm_120), single **non-blocking** compute stream `ctx.stream`
  (`src/core/device.cu:64-81` — `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`), so it is
  stream-capturable. Decode = 64 layers (16 full-attention + 48 GDN), `head_dim=256`, `n_q=24`, `n_kv=4`
  (`include/qus/model/config.h`).
- Decode step body: `src/model/qwen3_6_27b.cpp` `decode_step_impl` (611-631) — `work_.reset()` →
  `embed_gather(io_.token)` → `run_layers(Decode)` → final `rmsnorm` → `linear(lm_head)` →
  `argmax(io_.logits, io_.token)` → `kv_.advance()` (host) → `detail::advance_pos(io_.pos)` (device).
- Per-step host sync is one `cudaStreamSynchronize` + token D2H in `Engine::read_token`
  (`src/runtime/engine.cpp:192-197`); this stays (it overlaps GPU work) — graphs remove the **launch**
  overhead, not the token read.
- Attention position dependence (the only obstruction): `src/kernels/wrapper/gqa_attention.cpp:157-170`
  (`decode_tile_n(kv.pos)`, `tile_count = ceil((pos+1)/tile_n)`, partial buffers sized by `tile_count`),
  launched at `src/kernels/launcher/gqa_attention_decode.cu:21-101` (`grid.y = tile_count`).
- **Launcher prototype:** `src/kernels/launcher/gqa_attention.h:15-20` declares
  `gqa_attention_decode_launch(..., std::int32_t tile_n, std::int32_t tile_count,
  std::int32_t q_heads_per_cta, ...)` — this signature is changed by Task 1 and **must** be in its file
  list + commit.
- **Shared constant home:** `include/qus/kernels/gqa_attention.h` is plain C++ (included by the `.cpp`
  wrapper and the `.cu` bench); it is the home for `kGqaDecodeSplits`. The wrapper currently redefines
  `kHeadDim/kQHeads/kKVHeads` locally (`gqa_attention.cpp:15-18`) and does **not** include any kernel
  `.cuh` — keep it that way.
- Decode kernel + reduce: `src/kernels/kernel/gqa_attention_decode.cuh`
  (partial `170-364`, reduce `366-432`). Neutral-partial helpers `126-168`; append `296-300` / `212-221`;
  current-token-from-`k_new` `239-256` / `325-328`.
- **e2e report contract (Task 4):** `bench/e2e_bench.cpp:300` drives the real `Engine::decode_step()` loop;
  the report names the decode path **eager** — `decode_eager_tok_s`(`_valid`/`_median`) in
  `bench/e2e_bench_support.{h,cpp}` (`RepeatReport` methods + JSON writers `e2e_bench_support.cpp:240,283`)
  and `"decode_metric": "decode_eager_tok_s"` in the engine block (`e2e_bench_support.cpp:742`, alongside
  `sampling_location`/`token_readback`/`includes_token_readback`). Consumers: `docs/bench/e2e-report-schema.md`,
  `tools/bench/e2e_report_common.py`, `tools/bench/compare_e2e_reports.py`, `tools/bench/make_baseline_summary.py`,
  `tests/test_e2e_bench_support.cpp`, `tests/test_bench_report_tools.py`, and baselines under
  `docs/bench/baselines/`.
- Test (fp64 oracle): `tests/kernels/test_gqa_attention.cpp` → target `qus_gqa_attention_test`
  (`tests/CMakeLists.txt:71`). Decode cases at `pos = 1, 17, 2048, 2882, 8191` (+ `32768` with
  `--long-decode`) + stress; asserts output tolerance, **K/V append bit-equality**, and **decode must not
  advance host `kv.pos`**.
- Bench: `bench/gqa_attention_bench.cu` → target `qus_gqa_attention_bench` (`bench/CMakeLists.txt:24`).
  `--decode --decode-pos N --profile-once --cold-cache` (ncu target) and `--round-robin-layers 16`
  (models the 16-layer rotation, avoids hot-cache bias). It currently computes its byte model + workspace
  from `decode_tile_n`/`decode_tile_count` and prints `tile_n`/`tile_count` — these must move to the split
  model in Task 1 so it keeps building and measuring the real kernel.
- End-to-end: `./build/src/qus <…>.qus --tokenizer … --prompt … --max-context … --max-new …`
  (`src/main.cpp`); greedy decode is the parity oracle.

---

## Task 1 — Fixed-split GQA decode refactor (coordinator dispatches one subagent)

Make the decode attention launch position-invariant by replacing fixed-`TileN` / variable-`tile_count`
with a fixed `NUM_KV_SPLITS = 128` (compile-time constexpr; Task 2 tunes the value). After this task the
attention grid and all decode launch configs are byte-identical across steps, but **no graph is captured
yet** — the kernel is verified eagerly against the existing oracle.

**Files:**
- Modify: `include/qus/kernels/gqa_attention.h` (define the shared `kGqaDecodeSplits` constant; plain C++, includable by the `.cpp` wrapper and the `.cu` bench/launcher)
- Modify: `src/kernels/launcher/gqa_attention.h` (change the `gqa_attention_decode_launch` prototype — drop `tile_n`/`tile_count`/`q_heads_per_cta`; **Issue 1**)
- Modify: `src/kernels/kernel/gqa_attention_decode.cuh` (partial kernel partition by `gridDim.y`; reduce loop bound = passed split count)
- Modify: `src/kernels/launcher/gqa_attention_decode.cu` (fixed `grid.y = kGqaDecodeSplits`; drop the dead `tile_n`/`q_heads_per_cta` switch — decode only uses warp-per-query-head, `q_heads_per_cta = 6`)
- Modify: `src/kernels/wrapper/gqa_attention.cpp` (drop `decode_tile_n`/`window`/`tile_count`; size partials by `kGqaDecodeSplits` from the shared header; keep all validation + the `kv.pos < max_context` guard)
- Modify: `bench/gqa_attention_bench.cu` (byte model + workspace sizing + printed fields → split model, **reading the shared `kGqaDecodeSplits`** — no local literal; keep CLI flags)
- No new test: the existing `qus_gqa_attention_test` is the oracle (its `pos=1` case already exercises `window < S` → `kps=1` + `S-2` neutral splits).

**Seed design (the round-1 starting direction; the subagent may refine indexing details but not the contract):**

- Add in the shared public header `include/qus/kernels/gqa_attention.h`:
  `inline constexpr int kGqaDecodeSplits = 128;` (the wrapper `.cpp`, launcher `.cu`, and bench `.cu` all
  include this header; **do not** put it in the kernel `.cuh`, because the `.cpp` wrapper must not include a
  CUDA kernel header — **Issue 2**). The kernel itself reads the split count from `gridDim.y`, so it needs
  no constant.
- **Partial kernel** `gqa_attention_decode_partial_kernel<QHeadsPerCta=6, WarpPerQueryHead=true>` (drop the
  `TileN` template param):
  - `p = pos[0]; window = p + 1; S = gridDim.y; split = blockIdx.y; kps = (window + S - 1) / S; split_start = split*kps; split_end = min(split_start + kps, window)`.
  - **Neutral:** `if (p < 0 || p >= max_context || split_start >= window) { write_neutral_partial(split); return; }` (reuse `gqa_write_neutral_partial_warp`, indexed by `split`).
  - **Append (exactly one split per kv-head):** `if (q_subgroup == 0 && split_start <= p && p < split_end)` write `k_new`/`v_new` → `cache_{k,v}[gqa_cache_index(kv_head, d, p, padded_context)]`. (`split_end == window` only for the split owning `p`; provable single writer.)
  - **Inner loop:** `for (token = split_start; token < split_end; ++token)` — the existing online-softmax body verbatim (current token `== p` reads `k_new`/`v_new`; else reads cache). Accumulate `m/l/acc` across the whole `[split_start, split_end)` range (a CTA now owns `kps` tokens, not `TileN`).
  - **Store** `partial_{acc,m,l}` indexed by `split`.
- **Reduce kernel** `gqa_attention_decode_reduce_output_kernel<DChunk>`: body unchanged; it already loops
  `tile < tile_count` and skips neutral partials — pass the split count (`kGqaDecodeSplits`) as that arg.
  Grid stays `(24, 8)`.
- **Launcher** `gqa_attention_decode_launch`: drop the `tile_n`/`tile_count`/`q_heads_per_cta` parameters
  from the prototype (`src/kernels/launcher/gqa_attention.h`) **and** the definition; set
  `grid = (kGqaKVHeads * q_subgroups, kGqaDecodeSplits)`; instantiate only `<6, true>`; pass
  `kGqaDecodeSplits` as the reduce split count; delete the `tile_n`/`q_heads_per_cta` switches.
- **Wrapper** `gqa_attention_decode`: delete `decode_tile_n`/`window`/`tile_count`; allocate
  `partial_acc = ws.alloc(BF16, {kHeadDim, kQHeads, kGqaDecodeSplits})`, `partial_m`/`partial_l =
  ws.alloc(FP32, {kQHeads, kGqaDecodeSplits})`; call the new launcher signature; keep `ArenaScope`, all
  shape/dtype validation, and the `kv.pos >= kv.max_context` guard.
- **Bench:** replace `decode_tile_n`/`decode_tile_count` with `kGqaDecodeSplits` (from the shared header) +
  `kps`; size the scratch byte model and `decode_workspace_bytes_for_pos` by `kGqaDecodeSplits`; print
  `splits=%d kps=%d` instead of `tile_n`/`tile_count`. Keep `--profile-once`/`--cold-cache`/`--round-robin-layers`.

**Steps:**
- [ ] **Step 1 — implement the seed design** across the files above (shared header + launcher header first, then kernel/launcher/wrapper/bench).
- [ ] **Step 2 — build:** `cmake --build build -j` (lib + `qus_gqa_attention_test` + `qus_gqa_attention_bench` + `qus`). Expected: clean build.
- [ ] **Step 3 — fp64 oracle (decode correctness + append + no-host-advance):**
  `ctest --test-dir build -R gqa --output-on-failure` and the long case:
  `./build/tests/qus_gqa_attention_test --long-decode`. Expected: `OK gqa_attention_decode correctness`.
- [ ] **Step 4 — memory safety (new split/append/neutral index math):**
  `compute-sanitizer --tool memcheck ./build/tests/qus_gqa_attention_test --long-decode`. Expected: `0 errors`.
- [ ] **Step 5 — end-to-end greedy parity (eager; not yet graphed):**
  `./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 --prompt "怎么用fem方法求解heat equation。" --max-context 8192 --max-new 64`
  and confirm the generated tokens match the pre-Task-1 `master` output for the same command (token-for-token; tolerance differences must not change greedy argmax). Expected: identical text.
- [ ] **Step 6 — full ctest:** `ctest --test-dir build`. Expected: green.
- [ ] **Step 7 — commit:** `git add include/qus/kernels/gqa_attention.h src/kernels/launcher/gqa_attention.h src/kernels/kernel/gqa_attention_decode.cuh src/kernels/launcher/gqa_attention_decode.cu src/kernels/wrapper/gqa_attention.cpp bench/gqa_attention_bench.cu && git commit -m "refactor(q5090): fixed-split GQA decode attention (position-invariant launch)"`

**Task 1 DoD:** `qus_gqa_attention_test` (incl. `--long-decode`) green; `compute-sanitizer` clean; e2e
greedy output identical to pre-task `master`; the decode attention grid is `(4, kGqaDecodeSplits)` for every
position (confirm by inspecting the launcher); the launcher prototype no longer carries
`tile_n`/`tile_count`/`q_heads_per_cta`; the `.cpp` wrapper includes no kernel `.cuh`; no change to prefill
or any non-attention file.

---

## Task 2 — GQA decode perf tune (single kernel, on mainline; reuse the push-down loop in place)

Reuse the per-kernel tuning loop from
[2026-06-30-q5090-v2-decode-gemv-roofline-pushdown.md](2026-06-30-q5090-v2-decode-gemv-roofline-pushdown.md)
("Per-kernel tuning loop") **but on `master` directly** — there is one kernel, so no detached worktree and
no parallelism. The coordinator drives round 0 (baseline) then fresh strongest-model discovery rounds
until the stop condition, keeping the fp64 oracle green every round.

**Files (owned this task):** `include/qus/kernels/gqa_attention.h` (the `kGqaDecodeSplits` **value**),
`src/kernels/kernel/gqa_attention_decode.cuh`, `src/kernels/launcher/gqa_attention_decode.cu`. Because
`bench/gqa_attention_bench.cu` reads `kGqaDecodeSplits` from the shared header (Task 1), its byte model and
prints **auto-follow** the tuned value — the bench needs no Task-2 edit unless a print/metric is refined
(**Issue 3** — no divergence by construction). No other file.

**What to tune:**
- **`kGqaDecodeSplits` (`S`)** — the one structural knob. It is fixed at capture, so it stays a
  compile-time constant; pick the single value that best balances (a) occupancy at long context (4 kv-heads
  × `S` CTAs run while the 16 attention layers are serial — want enough resident CTAs to fill ~170 SMs) vs
  (b) neutral-split + reduce overhead at short context. Sweep `S ∈ {32, 64, 96, 128, 192, 256}`.
- **Inner-loop micro-opts** the `ncu` evidence reveals (e.g., small `#pragma unroll` over the per-split
  token loop for ILP, vectorized cache reads), within the same partition contract.

**Metric & gate (attention is ~2% of decode, so the bar is "don't regress, prefer better"):**
- Authoritative per-shape metric: cold-cache **median µs** + **useful_kv GB/s** from
  `qus_gqa_attention_bench --decode --decode-pos N --profile-once --cold-cache` at `N ∈ {2048, 2882, 8192, 32768}`,
  cross-checked with `ncu` (occupancy, mem throughput, warp stalls) per the ncu-kernel-profile skill. Use
  `--round-robin-layers 16` for the steady-state-style timing pass.
- **Correctness gate every round:** `ctest --test-dir build -R gqa` (fp64 oracle) +
  `compute-sanitizer --tool memcheck ./build/tests/qus_gqa_attention_test --long-decode`.
- **Stop:** best `S` chosen with no regression vs the Task-1 baseline at any tested position (and improved
  where `ncu` shows headroom), **or** `ncu` documents a non-tunable wall — record it.

**Ledger:** `profiles/ncu-decode-cuda-graph/gqa_tune/` — `round0_baseline.*`, per-round `ncu`/bench
artifacts, and `summary.md` (before→after table + per-round limiter→change→result; the chosen `S` and why).

**Steps:**
- [ ] **Step 1 — round 0 baseline:** build `qus_gqa_attention_bench`; record cold-cache `--profile-once --cold-cache` + `--round-robin-layers 16` at the four positions; `ncu` the partial + reduce kernels; write `round0_baseline.*` + the ledger.
- [ ] **Step 2..N — discovery rounds:** per the push-down loop, one `ncu`-revealed limiter per round; sweep `S` / inner-loop; fp64 + sanitizer green each round; append `roundN_*`.
- [ ] **Step 3 — finalize:** set the chosen `kGqaDecodeSplits`; re-run `ctest --test-dir build -R gqa` + `compute-sanitizer`; record the best round in `summary.md`.
- [ ] **Step 4 — commit kernel:** `git add src/kernels/kernel/gqa_attention_decode.cuh src/kernels/launcher/gqa_attention_decode.cu && git commit -m "perf(q5090): tune fixed-split GQA decode (S=<value>)"`
- [ ] **Step 5 — commit evidence:** `git add profiles/ncu-decode-cuda-graph/gqa_tune && git commit -m "bench(q5090): record GQA decode fixed-split tuning"`

**Task 2 DoD:** fp64 oracle green; `compute-sanitizer` clean; cold-cache bench shows no regression at any
tested position (improvement where `ncu` had headroom); `S` finalized as a constexpr; `summary.md` records
the limiter and chosen value.

---

## Task 3 — CUDA graph capture/replay harness (coordinator dispatches one subagent)

With the decode step now fully static, capture it once and replay per token. The graph state is
**encapsulated** in a small RAII `DecodeGraph` type so `Engine` holds no raw CUDA graph handles; the device
work is split from the host bookkeeping so the captured body contains only replayable device ops.

**Files:**
- Create: `include/qus/runtime/decode_graph.h` + `src/runtime/decode_graph.cpp` — the `DecodeGraph` RAII
  owner (auto-registered by `src/CMakeLists.txt`'s `GLOB_RECURSE runtime/*.cpp` — no CMake edit).
- Modify: `include/qus/model/model.h` + `src/model/qwen3_6_27b.cpp` — add `void decode_step_record();`
  and move host `kv_.advance()` out of `decode_step_impl` into the public entry points.
- Modify: `include/qus/runtime/engine.h` + `src/runtime/engine.cpp` — add `EngineOptions::use_cuda_graph`
  (default `true`); hold a `DecodeGraph` + `bool decode_warmed_`; drive warmup → capture → replay in
  `Engine::decode_step()`; reset the graph on `load()`.
- Modify: `include/qus/text/cli.h` + `src/text/cli.cpp` + `src/main.cpp` — add a `--no-cuda-graph` flag
  (sets `CliOptions::use_cuda_graph=false` → `engine_options.use_cuda_graph`) for the parity gate and
  debugging.

**Seed design:**
- **`DecodeGraph` (encapsulation)** — a minimal RAII type, no templates leaking CUDA into `Engine`:
  ```cpp
  // include/qus/runtime/decode_graph.h
  #pragma once
  #include <cuda_runtime.h>
  #include <functional>
  namespace qus {
  class DecodeGraph {
  public:
      DecodeGraph() = default;
      ~DecodeGraph();
      DecodeGraph(const DecodeGraph&) = delete;
      DecodeGraph& operator=(const DecodeGraph&) = delete;
      DecodeGraph(DecodeGraph&&) noexcept;
      DecodeGraph& operator=(DecodeGraph&&) noexcept;
      // Record `body` on `stream` (ThreadLocal capture) into a fresh graph and instantiate it,
      // replacing any prior capture.
      void capture(cudaStream_t stream, const std::function<void()>& body);
      void launch(cudaStream_t stream);            // replay; precondition ready()
      [[nodiscard]] bool ready() const noexcept;   // exec_ != nullptr
      void reset() noexcept;                        // destroy exec_ + graph_
  private:
      cudaGraph_t graph_ = nullptr;
      cudaGraphExec_t exec_ = nullptr;
  };
  } // namespace qus
  ```
  `.cpp` implements `capture` (`cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal)` → `body()`
  → `cudaStreamEndCapture(&graph_)` → `cudaGraphInstantiate(&exec_, graph_, 0)`, after `reset()`), `launch`
  (`cudaGraphLaunch`), `reset`/dtor/move (destroy `exec_` then `graph_`), all `CUDA_CHECK`-guarded.
- **Model split** (`qwen3_6_27b.cpp` + `model.h`):
  - Remove the `kv_.advance()` call from `decode_step_impl<Tap>` (keep `work_.reset()` and
    `detail::advance_pos(io_.pos, s)` — device pos must advance inside the graph).
  - `decode_step()` (NullTap entry) and `decode_step_erased` (tap entry) each call `decode_step_impl(...)`
    then `kv_.advance()` — so the tool/tap path is unchanged and stays eager (never captured).
  - Add public `void decode_step_record();` = `decode_step_impl(NullTap)` only (no `kv_.advance()`). This is
    the device-only body the engine warms up and captures.
- **Engine harness** (`engine.cpp`), `Engine::decode_step()`:
  ```cpp
  require_loaded();
  if (kv_->pos >= kv_->max_context) throw std::out_of_range("Engine::decode_step exceeds max_ctx");
  if (options_.use_cuda_graph && decode_warmed_) {
      if (!decode_graph_.ready())
          decode_graph_.capture(ctx_->stream, [this] { card_->decode_step_record(); });
      decode_graph_.launch(ctx_->stream);
  } else {
      card_->decode_step_record();   // eager: first-call module warmup, or graphs disabled
      decode_warmed_ = true;
  }
  kv_->advance();                    // host pos advance, both paths
  return read_token();               // per-step sync + token D2H, outside any capture
  ```
  Step 1 (first decode) records eagerly (loads every decode kernel module — capturing a first-ever launch
  is illegal); step 2 captures+launches; step 3+ launch only. The captured body is position-independent, so
  one graph serves all later tokens and survives across `generate()` calls on the same load.
- **Lifetime:** in `Engine::load()`'s reset block call `decode_graph_.reset()` and set
  `decode_warmed_ = false` (buffers move on reload → graph invalid). `DecodeGraph`'s dtor cleans up on
  teardown. No mid-run re-capture (buffers are stable within a load).
- **Position correctness:** prefill sets device `pos` via `set_pos(io_.pos, T)`; the captured body's
  `detail::advance_pos` increments it after attention each replay, so token `N` reads the right `pos` with
  no per-step patching. Host `kv_.pos` advances each step for the bound check.

**Steps:**
- [ ] **Step 1 — `DecodeGraph`:** create the header + `.cpp`. Build `cmake --build build -j` (GLOB picks it up).
- [ ] **Step 2 — model split:** move `kv_.advance()` out of `decode_step_impl`; add `decode_step_record()`; have `decode_step()`/`decode_step_erased` call impl then `kv_.advance()`. Build.
- [ ] **Step 3 — engine harness + `use_cuda_graph` + lifetime + CLI flag** as above. Build (lib + `qus`).
- [ ] **Step 4 — parity (graph vs eager) — the primary correctness gate:**
  ```bash
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 --prompt "怎么用fem方法求解heat equation。" --max-context 8192 --max-new 256 > /tmp/graph.out
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 --prompt "怎么用fem方法求解heat equation。" --max-context 8192 --max-new 256 --no-cuda-graph > /tmp/eager.out
  diff /tmp/graph.out /tmp/eager.out
  ```
  Expected: identical (greedy decode is deterministic; the graph must not change values).
- [ ] **Step 5 — memory safety under replay:**
  `compute-sanitizer --tool memcheck ./build/src/qus … --max-new 32` (graph ON). Expected: `0 errors`
  (covers capture + the repeated `cudaGraphLaunch` lifetime).
- [ ] **Step 6 — launch-collapse + wall evidence (nsys):** re-profile decode per the nsys-inference-analysis skill and compare to the cited current report:
  ```bash
  nsys profile --force-overwrite=true --stats=false --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    -o profiles/nsys-decode-cuda-graph/heat_fem_graph \
    ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
      --prompt "怎么用fem方法求解heat equation。" --max-context 8192 --max-new 2916
  ```
  Expected: `cudaLaunchKernel`/step ≈ 1 (down from 1429); decode kernel-active share ↑ toward ~95%; decode
  wall ↓ toward the kernel-sum floor; record a short report under `docs/bench/`.
- [ ] **Step 7 — regressions:** `ctest --test-dir build` (incl. `qus_engine_memory_stats_test`, `qus_runtime_file_tap_test` — confirms the tap path still works eagerly). Expected: green.
- [ ] **Step 8 — commit:** `git add include/qus/runtime/decode_graph.h src/runtime/decode_graph.cpp include/qus/model/model.h src/model/qwen3_6_27b.cpp include/qus/runtime/engine.h src/runtime/engine.cpp include/qus/text/cli.h src/text/cli.cpp src/main.cpp && git commit -m "feat(q5090): capture/replay decode as one CUDA graph"` then commit the nsys report as `bench(q5090): record decode CUDA-graph wall`.

**Task 3 DoD:** graph-vs-eager greedy text identical; `compute-sanitizer` clean under replay; nsys shows
launch collapse + wall improvement; `ctest` green (tap path still eager); `Engine` holds no raw
`cudaGraph_t` (only a `DecodeGraph`); graph reset on `load()`; `--no-cuda-graph` forces the eager path.

---

## Task 4 — e2e report contract: reflect graph-default decode (coordinator dispatches one subagent)

After Task 3, `EngineOptions::use_cuda_graph` defaults `true`, so `bench/e2e_bench.cpp`'s
`engine.decode_step()` loop measures the **graph** path — but the report schema names it **eager**
(`decode_eager_tok_s*`, `"decode_metric": "decode_eager_tok_s"`). AGENTS.md treats this report/schema as a
real, downstream-consumed contract, so the label must match reality. **Chosen approach: make the report
honest about the production path** — rename the metric to a path-neutral `decode_tok_s*` and add an explicit
`decode_path` fact. (Rejected alternative: force the e2e bench eager via `use_cuda_graph=false` — that keeps
the historical series comparable but makes the flagship metric stop reflecting production, which defeats the
feature; the graph win would only live in nsys. We instead re-baseline against the production path.)

**Files:**
- Modify: `bench/e2e_bench_support.h` — rename `RepeatReport::decode_eager_tok_s{,_valid}()` →
  `decode_tok_s{,_valid}()`; add `std::string decode_path` to `RawReport`.
- Modify: `bench/e2e_bench_support.cpp` — JSON keys `decode_eager_tok_s`/`decode_eager_tok_s_valid`
  (`write_repeat`) and `decode_eager_tok_s_median` (`write_case_summary`) → `decode_tok_s*`; engine block
  `"decode_metric": "decode_tok_s"` and add `"decode_path": "<report.decode_path>"`.
- Modify: `bench/e2e_bench.cpp` — set `report.decode_path = engine_options.use_cuda_graph ? "cuda_graph" : "eager";`
  rename the progress-log `decode_eager_tok_s` → `decode_tok_s`; ensure capture is excluded from the timed
  window (see measurement hygiene below).
- Modify: `docs/bench/e2e-report-schema.md` — rename the field(s), document `decode_path`, update the
  `decode_metric` description.
- Modify: `tools/bench/e2e_report_common.py`, `tools/bench/compare_e2e_reports.py`,
  `tools/bench/make_baseline_summary.py` — rename `decode_eager_tok_s*` references; accept/emit `decode_path`.
- Modify: `tests/test_e2e_bench_support.cpp`, `tests/test_bench_report_tools.py` — update assertions to the
  new field names + `decode_path` (this is a whitelisted real report/schema contract test).
- Regenerate: `docs/bench/baselines/*.json` (the m3 gate baselines) with the graph-on bench — the decode
  numbers improve; that improvement **is** the recorded win.

**Measurement hygiene (introduced by graphs):** the one-time module-warmup + capture happen on the first
two decode steps of the first measured repeat. Ensure they are excluded from the reported decode time —
run the e2e bench with `--warmup-repeats >= 1` for graphed reports (a warmup repeat triggers warmup+capture
so measured repeats are pure replay), and state this in the schema doc. (For long-decode cases the one
capture step is amortized, but the warmup repeat makes the metric clean regardless.)

**Steps:**
- [ ] **Step 1 — C++ rename + `decode_path`:** edit `e2e_bench_support.{h,cpp}` + `e2e_bench.cpp`; build `cmake --build build -j --target qus_e2e_bench qus_e2e_bench_support_test`.
- [ ] **Step 2 — C++ contract test:** `ctest --test-dir build -R e2e_bench_support --output-on-failure`. Expected: green with the new field names.
- [ ] **Step 3 — schema + python tooling + python test:** update the schema doc + the three `tools/bench/*.py` + `tests/test_bench_report_tools.py`; run `pytest tests/test_bench_report_tools.py -q`. Expected: green.
- [ ] **Step 4 — regenerate gate baselines** with `--warmup-repeats 1` (graph on), via the existing baseline-summary tooling; confirm `decode_path: "cuda_graph"` and `decode_tok_s*` present and the validator passes.
- [ ] **Step 5 — full ctest + pytest:** `ctest --test-dir build` and the bench-tool pytest. Expected: green.
- [ ] **Step 6 — commit:** `git add bench/e2e_bench_support.h bench/e2e_bench_support.cpp bench/e2e_bench.cpp docs/bench/e2e-report-schema.md tools/bench/e2e_report_common.py tools/bench/compare_e2e_reports.py tools/bench/make_baseline_summary.py tests/test_e2e_bench_support.cpp tests/test_bench_report_tools.py docs/bench/baselines && git commit -m "feat(q5090): e2e report reflects cuda-graph decode (decode_tok_s + decode_path)"`

**Task 4 DoD:** no `decode_eager_tok_s` remains (`rg` audit); the report carries `decode_tok_s*` +
`decode_path`; C++ + python report-contract tests green; gate baselines regenerated with `decode_path:
"cuda_graph"`; the validator/compare/summary tools accept the new schema.

---

## Review (independent subagent, after Task 4 — strict: CUDA kernels + numerics + GPU memory + runtime lifetime + report schema)

Per AGENTS.md (CUDA kernels, numerical behavior, GPU-memory lifetime, multi-step runtime flow → strict
review). Verify:

- **Position invariance:** the decode attention launch (`grid`, partial buffer extents, kernel scalar
  args) is identical for every `pos`; nothing in the captured body reads host `kv.pos`. (Inspect the
  launcher + wrapper; confirm `S` is a compile-time constant.)
- **Numerics:** fp64 oracle green incl. `--long-decode`; the fixed-split flash merge is mathematically the
  standard partition (append exactly once per kv-head; neutral splits contribute `m=-inf, l=0`); greedy
  end-to-end parity vs eager holds.
- **GPU memory:** `compute-sanitizer` clean for both the op test and a graphed `qus` run; the split/append/
  neutral index math is in-bounds against `cache_*` and `partial_*`; partial buffers go through `ws`/`ArenaScope`.
- **Runtime lifetime + encapsulation:** graph state lives only in the RAII `DecodeGraph` (Engine holds no
  raw `cudaGraph_t`); the graph is `reset()` on `load()` and freed on teardown; capture happens after a
  module-warmup step; the tap path and prefill remain eager; `kv_.advance()` is host-side and outside the
  captured body while `detail::advance_pos` is inside it; `--no-cuda-graph` / `use_cuda_graph=false` forces
  the eager path and matches graph output token-for-token.
- **Report schema (Task 4):** no `decode_eager_tok_s` remains; the report carries `decode_tok_s*` +
  `decode_path`, `decode_metric` matches, and the C++/python report-contract tests + regenerated baselines
  agree; the bench excludes warmup/capture from the timed window.
- **Evidence:** nsys shows `cudaLaunchKernel`/step ≈ 1 and a decode-wall improvement vs
  `q5090-v2-current-qus-decode-nsys-report.md`; Task 2's `S` choice is `ncu`/bench-backed.
- **History:** linear `master`; Task 1 `refactor(q5090):`, Task 2 `perf(q5090):` + `bench(q5090):`, Task 3
  `feat(q5090):` + `bench(q5090):`, Task 4 `feat(q5090):`; no schedule/format/weight files touched beyond
  the Task-3 step split and the Task-4 report contract.

## Self-review (against AGENTS.md + the user's task shape)

- **Goal/non-goals, execution mode, scope/ownership per task, task breakdown, reading lists, DoD +
  verification commands, risk-scaled review** — all present ✓.
- **Tasks** — the user's three (fixed-split GQA, GQA tune on mainline, CUDA graph) plus **Task 4** added
  because graph-default decode changes the e2e report contract (review Issue 4) ✓.
- **Subagent-driven** — one fresh subagent per task; strict independent review ✓.
- **Shared split constant (review Issues 2/3)** — `kGqaDecodeSplits` lives in the C++/CUDA-shared public
  header (`.cpp` wrapper never includes a kernel `.cuh`; kernel uses `gridDim.y`); bench reads it so byte
  model can't diverge; Task 2 owns the value ✓.
- **Launcher header (review Issue 1)** — `src/kernels/launcher/gqa_attention.h` is in Task 1's files +
  commit; the `tile_n`/`tile_count`/`q_heads_per_cta` prototype is changed ✓.
- **Encapsulation** — graph lifetime is a self-contained RAII `DecodeGraph`; `Engine` exposes a clean
  `use_cuda_graph` option (no `getenv`, no raw graph handles) ✓.
- **Testing policy** — no new low-value tests; kernel correctness rides the existing fp64 oracle
  (`qus_gqa_attention_test`) + `compute-sanitizer` + greedy parity; Task 4's report-schema tests are the
  whitelisted real CLI/report-contract category ✓.
- **No backward-compat shims** — the `tile_n`/`tile_count` decode path and the `decode_eager_tok_s` field
  are **deleted/renamed**, not kept beside the new ones; `use_cuda_graph` is a live-feature capability
  switch (the eager body is the required warmup/tap path), not a legacy fallback ✓.
- **One graph for all context lengths** — fixed `S` decouples graph count from `pos`/`max_context`; batch=1
  ⇒ exactly one decode graph ✓.
- **Prefill / schedule / formats / weights untouched** beyond the Task-3 host/device step split and the
  Task-4 report contract ✓.
