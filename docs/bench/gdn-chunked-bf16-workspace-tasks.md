# GDN Chunked BF16 Workspace-Reduction Task List

> Date: 2026-07-01
> Type: task checklist (not a formal implementation plan).
> Scope: reduce the prefill workspace peak of the GDN linear-attention block by moving the
> `gated_delta_rule_chunked` path to native bf16 I/O and, where numerically safe, bf16 storage of
> internal buffers. Correctness/parity remains owned by `tests/kernels/test_gated_delta_rule.cpp`.

## Motivation

The GDN linear-attention block is the prefill **workspace peak**: ~305 KiB/token, which forces
large `--work-bytes` and caps single-shot prefill length on a 32 GB card. The peak occurs inside
`kernels::gated_delta_rule_chunked` and decomposes as:

- ~110 KiB/token: the `gdn_mix` block's own live bf16 buffers ([qwen3_6_27b.cpp](../../src/model/qwen3_6_27b.cpp):435-486).
- 64 KiB/token: FP32 cast temporaries `q_f32/k_f32/v_f32/out_f32` ([gated_delta_rule.cpp](../../src/kernels/wrapper/gated_delta_rule.cpp):124-129).
- 120 KiB/token: chunked-algorithm scratch `g_cumsum/W/U/v_new/h_chunk` ([gdn_chunked_common.cuh](../../src/kernels/kernel/gdn_chunked_common.cuh):56-73).

This is a **memory** win, not a throughput win: the `gdn` category is only ~1.6–2.6% of prefill
kernel time (`gqa_attention_prefill` is 49–66% at 2k–4k, see
[q5090-v3-prefill-length-nsys-kernel-breakdown.md](q5090-v3-prefill-length-nsys-kernel-breakdown.md)).
The goal is headroom for tens-of-k prefill, not faster prefill.

## Invariants (must hold in every task)

- FP32 always: the persistent `ssm_state`, the running cross-chunk state accumulator in
  state-passing, `g_cumsum` and `exp(g)` gating, and all matmul/FMA accumulation.
- Never loosen the `gdn_output_bf16` / `gdn_state_fp32` presets
  ([op_check.h](../../tests/kernels/op_check.h):62-64). A state-tolerance failure is a signal that a
  buffer must stay FP32 — not a reason to widen the preset. If a genuinely new precision class is
  justified later, add a *new named preset* per the op-test standard rather than editing these.
- Each task is independently revertible and gated; do not stack an unverified task on another.

## Baseline (measured, `work peak`)

Config: `S=128, H_qk=16, H_v=48, chunk=64`. Per the nsys breakdown:

| pp | work peak (MiB) | KiB/token |
| ---: | ---: | ---: |
| 512 | 152.4 | ~305 |
| 1024 | 304.8 | ~305 |
| 2048 | 609.5 | ~305 |
| 4096 | 1219.0 | ~305 |

Per-buffer bytes/token inside the chunked call:

| buffer | dtype | B/token | notes |
| --- | --- | ---: | --- |
| `q_f32`, `k_f32` | fp32 | 8192 each | `S·H_qk·4` cast temporaries |
| `v_f32`, `out_f32` | fp32 | 24576 each | `S·H_v·4` cast temporaries |
| `g_cumsum` | fp32 | 192 | `H_v·4` |
| `W`, `U` | fp32 | 24576 each | `H_v·S·4` |
| `v_new` | fp32 | 24576 | `H_v·S·4` |
| `h_chunk` | fp32 | 49152 | `(1/64)·H_v·S·S·4` per-chunk state snapshot |

## Precision gate (PG) — run after every task

```bash
cmake --build build -j --target qus_gated_delta_rule_test qus_gated_delta_rule_bench
ctest --test-dir build -R qus_gated_delta_rule_test --output-on-failure
compute-sanitizer --tool memcheck ./build/tests/qus_gated_delta_rule_test
./build/bench/qus_gated_delta_rule_bench --prefill --sweep --csv
./build/bench/qus_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus -p 4096 -o json \
  --output-file profiles/bench/gdn_task_check.json     # record work peak
```

Pass criteria: correctness test green with **unchanged** tolerances (esp. `gdn_state_fp32` under the
slow-decay case from Task 1), sanitizer clean, and a recorded `work peak` delta vs baseline.

---

## Task 0 — Baseline capture

- **Objective:** freeze the before-reference so every later delta is auditable.
- **Do:** run the PG once on the current tree; record correctness=green, the `--prefill --sweep`
  timings, and `work peak` at pp512/4096 (and a long case, e.g. pp16384 with `--work-bytes`).
- **Output:** a short note of the baseline numbers (this file's Baseline table already captures the
  key ones).

## Task 1 — Test/bench instrument prep (before any precision change)

- **Objective:** make the test able to *detect* bf16-intermediate state error, and the bench able to
  *report* the workspace win.
- **Scope:**
  - [test_gated_delta_rule.cpp](../../tests/kernels/test_gated_delta_rule.cpp): add a chunked
    **slow-decay** case, e.g. a variant of `chunked_case` that builds inputs with `stress_g=true`
    (`g∈[-1,-0.05]`) at `T=4096` (today all chunked cases hardcode `stress_g=false`, line 240). Slow
    decay lengthens effective memory and is the adversarial condition for cross-chunk state error.
  - [gated_delta_rule_bench.cu](../../bench/gated_delta_rule_bench.cu): print the workspace bytes
    (it already computes `wrapper_workspace_bytes`); add a column so per-op workspace is measurable.
- **Keep FP32:** everything (no kernel math changes in this task).
- **Gate:** PG must pass on the **current FP32 kernel**, including the new slow-decay case. This
  establishes the stricter oracle used to judge Tasks 3–5.

## Task 2 — SAFE: native bf16 I/O (delete the FP32 cast layer)

- **Objective:** remove the 64 KiB/token cast temporaries with no numerical change.
- **Scope:**
  - Change the chunked launcher and stage kernels to read bf16 `q/k/v` and write bf16 `out`,
    converting to fp32 in-register / feeding bf16 tensor-core MMA:
    [gated_delta_rule_chunked.cu](../../src/kernels/launcher/gated_delta_rule_chunked.cu),
    [gated_delta_rule_recurrent.cu](../../src/kernels/launcher/gated_delta_rule_recurrent.cu),
    and the three stage kernels
    ([gdn_prepare_wy_wu](../../src/kernels/kernel/gdn_prepare_wy_wu.cuh),
    [gdn_state_passing](../../src/kernels/kernel/gdn_state_passing.cuh),
    [gdn_chunk_output](../../src/kernels/kernel/gdn_chunk_output.cuh)).
  - Delete the cast buffers and cast launches in
    [gated_delta_rule.cpp](../../src/kernels/wrapper/gated_delta_rule.cpp):124-160 and the cast
    kernels in [gdn_cast.cuh](../../src/kernels/kernel/gdn_cast.cuh).
  - Update the `--kernel-only` bench path (currently FP32) to bf16 inputs, and update
    `chunked_arena_bytes` in the test to drop the removed cast buffers.
- **Keep FP32:** `W/U/v_new/h_chunk`, `g_cumsum`, state, all accumulation. (Only the boundary I/O
  changes here.)
- **Expected:** −64 KiB/token → ~241 KiB/token.
- **Gate:** PG with **unchanged tolerances**. bf16→fp32 is exact and the model's `qn/kn/vv/o` are
  already bf16, so the correctness test passing at the current tolerances is the proof this is safe.

## Task 3 — bf16 storage of `W` and `U`

- **Objective:** halve the two WY-factor buffers.
- **Scope:** store `W`, `U` as bf16 in the workspace layout
  ([gdn_chunked_common.cuh](../../src/kernels/kernel/gdn_chunked_common.cuh) + the prepare/consumer
  kernels); producers/consumers read bf16 and accumulate fp32.
- **Keep FP32:** the T_inv forward-substitution math, state, accumulation.
- **Expected:** −24 KiB/token (2×12288).
- **Gate:** PG. Keep only if `gdn_state_fp32` still passes under slow-decay @T=4096; otherwise revert
  `W/U` to fp32 and record the result in Task 6.

## Task 4 — bf16 storage of `v_new`

- **Objective:** halve the corrected-values buffer.
- **Scope:** store `v_new` as bf16 (produced in state-passing, consumed in chunk-output); read bf16,
  accumulate fp32.
- **Keep FP32:** state accumulator, `g_cumsum`, accumulation.
- **Expected:** −12 KiB/token.
- **Gate:** PG; keep-or-revert on the state tolerance.

## Task 5 — bf16 `h_chunk` snapshot (accumulator stays FP32)

- **Objective:** halve the largest single buffer while protecting the state.
- **Scope:** keep the running cross-chunk state accumulator FP32 in
  [gdn_state_passing](../../src/kernels/kernel/gdn_state_passing.cuh); store only the per-chunk
  **snapshot** consumed by `chunk_output`'s `q·hᵀ` as bf16.
- **Keep FP32:** the accumulator and `state_out`/`ssm_state` (never bf16).
- **Expected:** −24 KiB/token. Highest risk — it touches the state carrier's stored form.
- **Gate:** PG; keep-or-revert strictly on `gdn_state_fp32` under slow-decay @T=4096.

## Task 6 — Consolidate and document

- **Objective:** lock in the outcome and update the record.
- **Do:** run the full PG at pp4096 and a long prefill (pp16384/24576); record the final `work peak`,
  the per-buffer bf16/FP32 decision matrix (which of W/U/v_new/h_chunk survived), and the resulting
  max prefill length at a given `--work-bytes`. Update the workspace numbers in
  [q5090-v3-prefill-length-nsys-kernel-breakdown.md](q5090-v3-prefill-length-nsys-kernel-breakdown.md).

## Projected workspace (if all tasks land)

| stage | KiB/token | pp16384 peak |
| --- | ---: | ---: |
| baseline | ~305 | ~4.76 GiB |
| after Task 2 | ~241 | ~3.76 GiB |
| after Task 3 | ~217 | ~3.39 GiB |
| after Task 4 | ~205 | ~3.20 GiB |
| after Task 5 | ~181 | ~2.83 GiB |

Task 2 is the guaranteed ~21% cut; Tasks 3–5 add up to ~40% total but each is gated on the FP32
state tolerance and independently revertible. Reducing the `gdn_mix` block's own dead-but-live bf16
buffers (~110 KiB/token) is a separate, orthogonal lever not covered here.
