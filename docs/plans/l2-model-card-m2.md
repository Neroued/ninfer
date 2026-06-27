# M2 — Model Card + Engine Integration & Validation — Plan (Codex-self-contained)

> **Current status:** M2 implementation exists; this plan is retained as implementation history and
> validation reference. The current tree contains the L2 model card, Engine, q5090 oracle,
> block-parity tooling, and greedy-match tooling. Performance optimization remains M3+.

> Self-contained for Codex. Reuses the execution machinery in
> [`docs/plans/l1-tier1-simple-ops.md`](l1-tier1-simple-ops.md) (subagent workflow + prompt templates
> + ncu procedure) and the frozen test framework. The design is
> [`docs/l2-model-card-design.md`](../l2-model-card-design.md) — **read it fully** (the §4 schedule is
> aligned to the real L1 op signatures). All 13 L1 ops are implemented and tested.

**Historical goal:** Build the L2 model card (`config.h` + `model.h` + `qwen3_6_27b.cpp`) and the `Engine`, wiring
every L1 op into the prefill/decode schedule, then **validate the wiring** against a self-contained
oracle over our own dequantized weights: block parity + end-to-end greedy match (NOT vLLM/llama.cpp,
which can't read q5090). This is the M2 correctness baseline (`design.md` §12) — the first time the full
pipeline runs and is proven correct. **Performance is M3+, not gated here.**

**Architecture:** The card is a thin `Qwen3_6_27B` object (bindings + schedule); the `Engine` owns the
resources + outer loop (design §2). Straight-line C++ over `constexpr` dims; the §4 sketch is the
spec. token-ids in → token-ids out, greedy.

**Tech Stack:** C++20, CUDA 13.1 (sm_120), CMake ≥ 3.28, gcc 13.3, build dir `build/`. Real weights:
the q5090 file under `out/` (see `out/manifest.json`); reference model in `~/vllm` (or `~/llama.cpp`).

---

## Hard rules (retained)
- Build the card per `l2-model-card-design.md` §1–§11 (don't re-derive decisions). No virtual dispatch,
  no per-step `cudaMalloc`, device-side `pos`/`token`, work_ bump-reset each step (design §6).
- L1 ops + the frozen test framework are read-only.
- **FORMAT before every commit:** `clang-format -i <new/changed .h/.cuh/.cpp/.cu>` then
  `clang-format --dry-run --Werror <…>` (exit 0; repo `.clang-format`).
- Work on `master`, one commit per task.

## Components built by current code
- `include/qus/model/config.h` — `ModelConfig` (constexpr, design §3.1) + `is_full`/`full_idx`/`gdn_idx`
  + `kCfg` + derived consts (`kAttnScale=1/√256`, `kGdnScale=1/√128`, `rms_eps`, rope dims).
- `include/qus/model/model.h` — `Qwen3_6_27B`: `FullLayerW`/`GdnLayerW`/`MlpW` (over `const Weight*` /
  `const Tensor*`, design §3.3), `StepState` (design §3.5), ctor (binds) + `prefill`/`decode_step`,
  holds references to Engine resources, optional `Tap` template (design §8).
- `src/model/qwen3_6_27b.cpp` — `bind()` (`WeightStore` → per-layer structs via
  `(TextCore, SourceKind, layer)`), `attn_mix`/`gdn_mix`/`mlp_tail`/`run_layers`, `prefill`/`decode_step`
  (exactly design §4).
- `include/qus/runtime/engine.h` + `src/runtime/engine.cpp` — `Engine`: owns `DeviceContext`, 3
  `DeviceArena`s, `WeightStore`, `KVCache`, `GdnState`, `StepState`, the card; `load(path)` (load
  weights, size caches, construct card) + `generate(prompt, max_new) -> vector<int>` (prefill + decode
  loop + next-token D2H readback + EOS).
- `StepState`: `token[1]` I32, `pos[1]` I32, `logits[vocab]` (dtype follows `lm_head` out; bf16 v1) —
  persistent device buffers (design §3.5).

**Cache construction (confirm dims):** `KVCache(cache_arena, /*full_layers=*/16, max_context,
/*num_kv_heads=*/4, /*head_dim=*/256, bf16)`; `GdnState(cache_arena, /*gdn_layers=*/48,
/*conv_dim=*/10240, /*conv_width=*/4, /*value_heads=*/48, /*value_head_dim=*/128, /*key_head_dim=*/128)`
with **`ssm[gidx]` laid out `[dk,dv,Hv]` AR-transposed** (design §4 notes). Verify `GdnState`'s ctor
produces that layout; if not, fix the ctor (small L0 change) — the grouped GDN kernel depends on it.

---

## Validation (the M2 gate)
The L1 ops are already proven vs fp64, so M2 only proves the **wiring** (schedule order, grouped
head-map, q/gate + qkv splits, scales, KV/state lifecycle, bindings). The reference is a small
self-contained **oracle over our OWN dequantized weights** — NOT vLLM/llama.cpp (which cannot read the
q5090 format, and would add quant noise + heavy internal hooks).

1. **Wiring smokes (no real model):** `bind()` + each block helper run on a fixture (shapes / finite /
   sanitizer) — m2-2 / m2-3.
2. **Block parity (the primary, cheap gate):** each block (`attn_mix` / `gdn_mix` / `mlp_tail`) at the
   real dims with **random weights** (synthesized via the q5090 packer helper) vs the oracle's matching
   block — exact to bf16 tolerance since weights are identical. Catches head-map / order / split / scale
   / KV-state bugs in milliseconds, with no 27B model.
3. **End-to-end greedy match (real weights, bounded):** the oracle loads the **real** q5090 (reusing the
   converter's `decode_tensor`), runs **one** short fixed prompt + a handful of decode steps with
   inter-op activations rounded to bf16 (mirroring the engine); our `Engine::generate` must produce the
   **same greedy tokens**. Per-layer cosine (≥ 0.999) is dumped only to **localize** a mismatch — not a
   mandatory 64-layer gate. No vLLM, no 128-token CPU run, no 52 GB load (layers stream).

---

## Current completion status

- **m2-1:** implemented and covered by `tests/test_model_config.cpp`.
- **m2-2:** implemented via model bindings / `StepState`; covered by `tests/test_model_bind.cpp`.
- **m2-3:** implemented via `attn_mix` / `gdn_mix` / `mlp_tail` / `run_layers`; covered by
  `tests/test_model_blocks.cpp`.
- **m2-4:** implemented via `Engine::load`, `prefill`, `decode_step`, and `generate`; covered by
  engine load/smoke structure and parity tooling.
- **m2-5:** implemented in `tools/parity/ref_model.py`.
- **m2-6:** implemented in `tools/parity/block_parity.py` plus `tools/parity/block_dump.cpp`,
  including decode `T=1` and chunked prefill coverage.
- **m2-7:** implemented in `tools/parity/greedy_match.py` with `FileTap` layer localization.

This status records that the implementation artifacts exist and have user-side testing history. This
documentation sync does not rerun real-weight parity, long CUDA tests, sanitizers, or benchmarks.

## Tasks (historical checklist; originally build + test/sanitizer + clang-format + commit)

### Task m2-1 — `ModelConfig` (config.h)
- **Reading:** design §3.1, `qwen3.6-27b-architecture.md` §2.
- **Files:** `include/qus/model/config.h` (struct + helpers + `kCfg` + derived scales/eps);
  `tests/test_model_config.cpp` (`static_assert`s: `n_full()==16`, `n_gdn()==48`, `full_idx(63)==15`,
  `gdn_idx(62)==47`, `is_full` pattern). Register in `tests/CMakeLists.txt`.
- **DoD:** test PASS, format clean. Commit `feat(model): ModelConfig (frozen dims + schedule helpers)`.

### Task m2-2 — bindings + StepState + `bind()`
- **Reading:** design §2/§3.2–§3.5, `weight-handle-design.md`, `include/qus/core/{weight_store,tensor}.h`.
- **Files:** `include/qus/model/model.h` (FullLayerW/GdnLayerW/MlpW/StepState + class skeleton);
  `src/model/qwen3_6_27b.cpp` (`bind()` mapping `(TextCore, SourceKind, layer)` → fields, per design §3.3
  + the §3.4 packer table). Test `tests/test_model_bind.cpp` (NEEDS_SOURCE_DIR): load a small fixture
  q5090, construct the card, assert every `Weight*`/`Tensor*` is non-null with the expected `(n,k,qtype)`
  / shape for a sampled full + gdn layer.
- **DoD:** bind test PASS (or SKIP w/o GPU/fixture), format clean. Commit `feat(model): bindings + bind()`.

### Task m2-3 — schedule helpers (`attn_mix`/`gdn_mix`/`mlp_tail`/`run_layers`)
- **Reading:** design §4 (the exact schedule), the L1 op headers, `op_tester.h`.
- **Files:** the helpers in `qwen3_6_27b.cpp` exactly per §4 (out-param ops, phase-split by `ph`, q/k/v
  views, `GdnState`/`KVCache` slots). Test `tests/test_model_blocks.cpp`: run `attn_mix` (one full layer)
  and `gdn_mix` (one gdn layer) + `mlp_tail` on fixture weights + random activations, decode (`T=1`) and a
  small prefill (`T=4`); assert output shapes, finite values, and `compute-sanitizer` clean. (Numerical
  correctness of the full stack is gated at parity, not here.)
- **DoD:** block smoke PASS, sanitizer clean, format clean. Commit `feat(model): block schedule helpers`.

### Task m2-4 — drivers + `Engine` (end-to-end smoke)
- **Reading:** design §2/§4/§6, `engine.h`, `include/qus/core/{arena,device,kv_cache,state_store}.h`.
- **Files:** `prefill`/`decode_step` in `qwen3_6_27b.cpp`; `engine.h`/`engine.cpp` (resource ownership +
  `load` + `generate` loop + StepState + D2H next-token readback + EOS + a basic decode-tok/s readout).
  Test `tests/test_engine_smoke.cpp`: on the fixture, `generate(prompt, 8)` runs end to end, returns 8
  ids, no `cudaMalloc` per step (assert workspace arena reset), sanitizer clean.
- **DoD:** smoke PASS, sanitizer clean, format clean. Commit `feat(runtime): Engine load + generate loop`.

### Task m2-5 — the model oracle (`tools/parity/ref_model.py`)
- **Reading:** `tools/q5090_convert/layouts.py::decode_tensor` (+ `qtypes.py`), `qwen3.6-27b-architecture.md`
  §6/§7/§8/§9, the L1 CPU refs (`tests/kernels/gdn_ref.h`, the attention/linear refs) as the math spec,
  `out/manifest.json`.
- **Files:** `tools/parity/ref_model.py` — a from-scratch PyTorch/numpy forward of the Qwen3.6 schedule
  that (a) loads weights from **our** q5090 via `decode_tensor` (dequant → fp32), **streaming one layer at
  a time** to the GPU; (b) implements the §6/§9 schedule — grouped head-map `h_v//3`, the q/gate + qkv
  splits, the `1/√256` / `1/√128` scales, GDN recurrence, partial RoPE, the `(1+w)` vs gated-norm
  convention — **rounding inter-op activations to bf16** to mirror the engine; (c) exposes a `block(...)`
  entry (for m2-6) and a `forward(prompt, n_decode)` returning greedy tokens + optional per-layer hidden
  dumps. Commit the fixed prompt + its token ids.
- **DoD:** runs on the real q5090 for a short prompt; documented. Commit `feat(parity): q5090 model oracle (ref_model.py)`.

### Task m2-6 — block parity (primary gate)
- **Reading:** m2-5 oracle `block(...)`; design §4; the q5090 packer helper `tests/kernels/q5090_pack.h`.
- **Files:** a parity harness (C++ `tests/test_model_blocks_parity.cpp`, or a small CLI driving the engine
  + the Python oracle): build random per-layer weights (packer helper for quant + random dense/norms), run
  `attn_mix` / `gdn_mix` / `mlp_tail` (decode `T=1` + prefill `T∈{4,64}`), compare to the oracle's matching
  block.
- **DoD:** every block matches the oracle within `attention_bf16` / `linear_bf16` tolerance; sanitizer
  clean; format clean. The first failing block localizes the wiring bug. Commit `test(parity): block parity vs oracle`.

### Task m2-7 — end-to-end greedy match (real weights)
- **Reading:** m2-5 oracle `forward`; `Engine::generate`.
- **Files:** `tests/test_greedy_match.cpp` (gated on the real q5090 via env var; SKIP loudly if absent):
  run `generate(prompt, N)` greedy (e.g. `N=16`) and assert **token-for-token equality** with the oracle.
  On mismatch, dump per-layer cosine vs the oracle (the `FileTap`) to find the first divergent layer.
- **DoD:** exact greedy match for the prompt; record a basic decode tok/s. Commit
  `test(parity): end-to-end greedy match vs oracle (M2 baseline)`.

---

## Dependencies & notes
- **Real weights:** the q5090 file (e.g. `out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus`, per `out/manifest.json`)
  + a GPU with ~24–26 GB free. Only m2-7 needs it; SKIP loudly if absent. m2-1…m2-6 need only a fixture /
  random weights.
- **Oracle:** `tools/parity/ref_model.py` reuses the converter's `decode_tensor` (already validated by
  `verify.py`) — **no vLLM/llama.cpp, no original safetensors, no 52 GB load** (weights stream per layer).
- **Head map:** grouped GDN (`h_v//3`) + schedule order get their proof at **block parity (m2-6)**; a
  failing block points straight at the bug.
- Performance (decode tok/s → roofline, fusion, CUDA-graph) is **M3–M5**, not part of M2.

## Done criteria
The card + Engine build; `bind()` / block / engine smokes pass (sanitizer clean); **block parity** vs the
oracle passes for every block; **end-to-end greedy match** vs the oracle passes on the real weights; all
code clang-format clean. At that point the v1 text forward compute graph is complete and correct.

## Self-review notes (author)
- Reference is a self-contained oracle over OUR dequantized weights (`ref_model.py`), not vLLM/llama.cpp
  (format-incompatible + heavy). Block parity (m2-6) is the cheap primary gate; end-to-end greedy match
  (m2-7) is the bounded real-weights gate; per-layer cosine is a localize-on-failure aid.
- Tasks build bottom-up (config → bind → blocks → engine), then oracle → block parity → greedy match.
- The schedule is implemented exactly per design §4; GdnState ssm layout + grouped head-map are the likely
  parity-failure suspects. clang-format in every task DoD; reuses the Tier-1 subagent workflow/templates.
