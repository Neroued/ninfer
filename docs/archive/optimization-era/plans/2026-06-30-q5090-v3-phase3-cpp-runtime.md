# q5090 v3 migration — Phase 3: C++ runtime (parser + kernels) + perf re-tune

> **Status:** Phase 3 of the v3 plane-split migration
> ([2026-06-30-q5090-v3-layout-migration-roadmap.md](2026-06-30-q5090-v3-layout-migration-roadmap.md)).
> Phase 1 (spec, [../q5090_packed_file_format_v3.md](../q5090_packed_file_format_v3.md)) is done;
> Phase 2 (Python converter + v3 weights,
> [2026-06-30-q5090-v3-phase2-python-converter.md](2026-06-30-q5090-v3-phase2-python-converter.md))
> is **in progress**. Phase 3 depends on the Phase-2 artifact
> `out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus` for its end-to-end verification.

## Goal

Make the C++ runtime read and run the v3 plane-split layout, and re-tune the now-unblocked Q5/Q6 GEMVs
to the DRAM roofline — **preserving the integrated kernel structure** (warp/block-per-row, split-K,
occupancy, vectorized loads) and swapping only the per-thread unpack to the cheap
`sign_extend(low | high<<4)` form. v3 removes the unpack wall that left every Q5 kernel stalled at
46–58% DRAM in the roofline round.

## Execution mode — two stages

Phase 3 splits at a real boundary: a **correctness migration** (atomic) followed by a **performance
re-tune** (parallelizable). Both run through subagents, but differently — Stage A is **one** strongest-
model subagent doing the whole atomic task (un-split) plus a paired review; Stage B is the
**multi-subagent parallel** push-down loop.

### Stage A — correctness migration (one atomic subagent + paired review; NOT split into subtasks)

- **One implementer subagent, strongest model, does the whole task.** Stage A is dispatched to a single
  subagent on the strongest available model that performs the entire correctness migration in one pass
  (no intermediate commits), followed by a **paired independent review subagent** (Stage A review,
  below).
- **Why not split into subtasks:** the runtime cannot load or run a v3 `.qus` until the parser, the
  `Weight` descriptor, and **every** Q5/Q6 consumer all speak the three-plane layout — a half-migrated
  runtime mis-reads the planes of any un-swapped kernel and produces garbage or crashes. There is no
  intermediate state that builds-and-runs, hence no verifiable subtask boundary; the subagent does it
  atomically and it is verified + reviewed **once**.
- Delivers **correct** v3 reading + Q5/Q6 plane-split unpack (kernel bodies may be untuned — Stage B
  tunes them). The v2 reader path is deleted (no dual format).

### Stage B — perf re-tune (subagent-driven, parallel worktrees)

- Reuses the **roofline push-down protocol** verbatim
  ([2026-06-30-q5090-v2-decode-gemv-roofline-pushdown.md](2026-06-30-q5090-v2-decode-gemv-roofline-pushdown.md)):
  one detached worktree per kernel, a fresh strongest-model subagent per round (round 1 = seed design,
  rounds 2+ = ncu discovery), cold-cache `qus_linear_op_bench` DRAM% gate + fp64 oracle, a
  coordinator-owned ledger, and squash-to-one-`perf(q5090)`-commit linear integration.
- Scope is only the Q5/Q6 kernels v3 unblocks (Stage A leaves them correct-but-untuned).

## Non-goals / hard constraints

- **No CUDA Graphs.** Launch cost is the *next* round (deferred until v3 lands). See "Relation to the
  launch-bound finding" — it materially shapes Stage B's gate but graphs themselves are out of scope.
- **No format change.** v3's byte contract is frozen by Phase 1; Phase 3 implements it, it does not
  edit the spec.
- **No model-schedule change**, no new fusion, no registry/dispatch reshuffle beyond passing the new
  high-plane pointer.
- **No Q4 / W8 kernel changes.** Q4 packing is unchanged (the nibble plane *is* a Q4 code plane);
  `mlp_gate_up`, `gdn_in_qk`, and the `attn_in` Q4 kernel stay verbatim. W8 stays single-plane.
- **No backward compatibility.** The v2 reader/parser path is removed; the runtime reads v3 only.

## Stage A — scope & ownership (the one atomic task)

| file | change |
| --- | --- |
| `include/qus/core/tensor.h` (`Weight` struct) | add a **high-plane pointer** `qhigh` (and `high_plane_bytes`) alongside `qdata`/`scales`; `qhigh == nullptr` for Q4/W8/dense. |
| `include/qus/core/weight_store_parser.h`, `src/core/weight_store_parser.cpp` | parse the v3 `TensorEntry` plane fields (`nibble_plane_bytes`/`high_plane_bytes`/`scale_plane_bytes`); accept magic `Q5090MIXEDV3`/version 3; structural checks per spec §13 (3-plane sizes, ROW_SPLIT vs CONTIGUOUS `padded_shape`, `K_pad=align_up(K,128)`). |
| `src/core/weight_store.cpp` (`make_quant_descriptor`, ~193) | compute the three 256-aligned relative plane offsets (nibble/high/scale) per spec §9.2/§9.3; set `weight.qdata` (nibble), `weight.qhigh` (high), `weight.scales` (scale), each shifted by `segment.row_begin*G*{nib,hi,2}`; assert against the parsed plane-byte fields. |
| `src/kernels/linear/codec/linear_codec.cuh` (`Q5Codec`, `Q6Codec`), `src/kernels/linear/reference/linear_generic.h`, `src/kernels/linear/reference/linear_generic_lowbit*.{cu,cuh}` | two-plane unpack: `low` from the nibble plane (reuse the Q4 nibble loader), `high` from the high plane, `q = sign_extend(low | high<<4)`. Drives the generic lowbit GEMV/GEMM (prefill + fallback), whose launchers must pass `w.qhigh` for Q5/Q6. |
| `src/kernels/linear/gemv/linear_rowsplit_gemv_mlp_down.{cu,cuh}` | unpack body → plane-split; keep split-K + reduce **structure** (Stage B revisits the split factor / whether the reduce is still needed). Plumb `qhigh`. |
| `..._proj_6144.{cu,cuh}`, `..._out_6144.{cu,cuh}` | unpack body → plane-split; keep current occupancy/split-K structure; plumb `qhigh`. |
| `..._attn_in_7168.{cu,cuh}` (Q5 kernel only) | Q5 unpack body → plane-split; the Q4 kernel in the same file is untouched. |
| `..._lm_head.{cu,cuh}` | Q6 unpack body → plane-split (nibble + 2-bit high). |
| `src/kernels/wrapper/embed_gather.cpp`, `src/kernels/launcher/embed_gather.{cu,h}`, `src/kernels/kernel/embed_gather.cuh` | Q6 single-row dequant reads nibble + high planes. |
| `src/kernels/linear/linear.cpp` + the rowsplit launcher signatures | thread `w.qhigh` to the Q5/Q6 launchers (Q4/W8 unaffected). |
| `bench/linear_op_bench.cu` (`make_row_split_payload`/`make_weight`), `bench/linear_bench.cu`, `bench/embed_gather_bench.cu` | generate v3 three-plane payloads and set `qhigh` so the active low-bit benches measure the real v3 kernels and do not preserve v2 payload assumptions. |
| `tests/fixtures/make_q5090_fixture.py`, `tests/kernels/q5090_pack.h`, `tests/kernels/test_embed_gather.cpp`, `tests/test_q5090_pack_golden.cpp`, `tests/test_q5090_parser.cpp`, `tests/test_weight_store*.cpp`, `tests/test_model_bind.cpp`, `tests/test_engine_memory_stats.cpp`, `tests/test_model_blocks.cpp`, `tools/parity/q5090_structural_dump.h` | C++ packer/golden/parser, fixture-backed runtime tests, embed-gather oracle, and structural parity dump updated to v3 three-plane; assert plane offsets + dequant. |

**Delete:** the v2 reader/parse path and any v2-only assumptions (magic/version/2-plane offset math).

**Out of scope (Stage A):** Q4/W8 kernels, model schedule, CUDA graphs, the spec.

### Stage A reading list

- Spec §4 (TensorEntry plane fields), §9.1 (decode), §9.2 (3-plane sizes/offsets), §9.3 (plane
  addressing / `BlockView` contract), §13 (validation).
- `src/core/weight_store.cpp` `make_quant_descriptor` (193–227) + the load loop (338–379);
  `include/qus/core/weight_store_parser.h`; `src/core/weight_store_parser.cpp`;
  the Q5/Q6 kernels' current `accumulate_*_group`; `linear_codec.cuh`;
  `src/kernels/linear/reference/linear_generic_lowbit*.{cu,cuh}`;
  `src/kernels/wrapper/embed_gather.cpp`; `src/kernels/kernel/embed_gather.cuh`;
  `bench/linear_op_bench.cu` `make_row_split_payload`/`make_weight`;
  `bench/linear_bench.cu`; `bench/embed_gather_bench.cu`.

### Stage A definition of done (run once, all must pass)

```bash
cmake --build build -j                                         # lib + benches + tools + tests
ctest --test-dir build                                        # green (parser/pack-golden/weight-store/model)
ctest --test-dir build -R linear                              # fp64 oracle: Q5/Q6 shapes on v3
compute-sanitizer --tool memcheck ./build/tests/qus_linear_test   # 3-plane reads in-bounds
# end-to-end on the Phase-2 v3 artifact:
./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
  --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --prompt "怎么用fem方法求解heat equation。" --max-context 8192 --max-new 64
```

- Build + full `ctest` green; fp64 oracle green for every Q5/Q6 shape on v3; `compute-sanitizer` clean.
- **Token parity:** greedy decode on the v3 artifact matches the v2 baseline token-for-token (v3 is
  value-preserving, so identical greedy output is the correctness gate; use the existing greedy
  snapshot / `tools/parity`).
- No v2 reader path remains (`rg` audit).
- Commit (single squashed commit, e.g. `feat(q5090): runtime reads v3 plane-split layout`).

## Stage B — perf re-tune loop (subagent-driven, parallel)

After Stage A is green, re-tune the now-plane-split Q5/Q6 kernels to the DRAM roofline using the
push-down protocol. Owned kernels (one worktree each): `mlp_down`, `proj_6144`, `out_6144`, `attn_in`
(Q5), `lm_head` (Q6). (Q4 kernels are already at roofline; not retuned.)

**Launch-aware gate (new, important).** The current nsys report
([../bench/q5090-v2-current-qus-decode-nsys-report.md](../bench/q5090-v2-current-qus-decode-nsys-report.md))
showed decode is **host-launch-bound**: the previous round's `proj_6144` split-K cut GPU time but added
`+96` reduce launches/step and **regressed wall time** (`+2.66 s`) even though kernel sum dropped. So
Stage B's gate is **not** per-op DRAM% alone:

- A kernel that reaches the roofline **without** adding launches is strictly preferred. With v3's cheap
  unpack, re-evaluate whether `proj_6144`/`out_6144` still need **split-K at all** — if the plane-split
  unpack + existing occupancy reach roofline, **drop the split-K reduce kernel**, which simultaneously
  raises kernel throughput **and removes the `+96` launches/step that caused the wall regression.**
- Integration is judged by **nsys on the full decode** (kernel sum **and** wall time **and** launches/
  step), not only the cold-cache per-op bench. A per-op win that increases launches/step and regresses
  decode wall is **not** accepted.
- Per-kernel correctness gate stays the fp64 oracle; per-kernel speed gate stays cold-cache DRAM%
  approaching `roof_us`; the coordinator ledger + linear `perf(q5090)` integration are as in the
  push-down plan.

### Stage B integration gate (coordinator, once)

```bash
cmake --build build -j && ctest --test-dir build              # green
ctest --test-dir build -R linear                              # fp64 oracle all Q5/Q6 shapes
compute-sanitizer --tool memcheck ./build/tests/qus_linear_test
# nsys decode re-profile on the v3 artifact; compare to the v2 current report:
nsys profile --force-overwrite=true --stats=false --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o profiles/nsys-long-decode/heat_fem_v3_after \
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
    --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
    --prompt "怎么用fem方法求解heat equation。" --max-context 8192 --max-new 2916
```

**DoD:** kernel sum down vs the v3-Stage-A baseline; **launches/step not increased** (ideally reduced
by dropping split-K); decode wall **not regressed** (and improved if split-K is dropped); the new top
decode kernel named in the ledger.

## Relation to the launch-bound finding (why wall-time mostly waits for the graph round)

v3 + Stage B reduce **kernel-math** time and (ideally) **launch count**. But decode is currently
~28% GPU-idle from per-launch host latency; the bulk of that wall-time is reclaimed by **CUDA Graphs**,
the explicitly deferred next round. So Phase 3's honest success criteria are: (1) correctness on v3;
(2) lower kernel sum / higher per-op DRAM%; (3) **no wall regression** (and a wall *improvement* if the
plane-split lets us drop the launch-adding split-K). The large wall-time jump is expected to land in the
subsequent CUDA-graph round, which converts these kernel/launch gains into throughput.

## Stage A review (paired — independent review subagent, strongest model; strict)

A dedicated review subagent runs immediately after the Stage A implementer subagent (q5090 format,
numerics, weight loading, CUDA kernels, GPU memory → strict review per AGENTS.md):

- **Format/ABI** — `make_quant_descriptor` plane offsets match spec §9.2/§9.3; the `Weight.qhigh`
  contract (null for Q4/W8) is consistent; parser rejects malformed/short v3 payloads (fail loud).
- **Numerical** — Q5/Q6 plane-split unpack reproduces the policy values (fp64 oracle); token parity vs
  v2 on the v3 artifact.
- **CUDA memory** — three-plane + segment-row-offset reads in-bounds; `compute-sanitizer` clean.
- **Scope** — Q4/W8 kernels and the schedule untouched; v2 reader removed; no CUDA graphs; v3 byte
  contract unchanged.

(**Stage B** carries its own review via the roofline push-down protocol's review phase — per-kernel
cold-cache evidence, fp64 oracle, `compute-sanitizer`, and the launch-aware nsys integration check.)

## Dependencies & sequencing

- Stage A *implementation* can begin before Phase 2 finishes, but its **e2e/token-parity DoD requires
  the Phase-2 v3 artifact**. Land Stage A only after Phase 2 produces `…_v3.qus`.
- Stage B follows Stage A (needs correct v3 kernels to tune).
- The CUDA-graph round follows Phase 3 (converts the kernel/launch gains into wall-time).
