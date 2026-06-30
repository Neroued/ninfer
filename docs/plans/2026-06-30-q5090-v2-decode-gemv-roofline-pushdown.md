# q5090 v2 — Decode GEMV Roofline Push-down (per-kernel extreme optimization)

> **For agentic workers (Codex):** REQUIRED SUB-SKILLS:
> `superpowers:subagent-driven-development` (the Group A tuning loops),
> `superpowers:using-git-worktrees` (one detached worktree per parallel kernel loop),
> `superpowers:systematic-debugging` (when a kernel regresses or `compute-sanitizer` trips).
> Performance subagents MUST read first: `/home/neroued/.cursor/skills/profile-cuda/SKILL.md`,
> `/home/neroued/.codex/skills/ncu-kernel-profile/SKILL.md`,
> `/home/neroued/.codex/skills/nsys-inference-analysis/SKILL.md`.
>
> Successor to [docs/plans/2026-06-30-q5090-v2-decode-gemv-fusion-and-splitk.md](2026-06-30-q5090-v2-decode-gemv-fusion-and-splitk.md).
> That round did fusion + a first per-kernel tuning pass. This round drives **every decode GEMV to its
> cold-cache bandwidth roofline** at the operator layer, and **nothing else**. CUDA Graphs are **out of
> scope** (a near-free launch-cost mop-up for a later round; explicit user decision). The on-disk Q5/Q6
> byte layout is **frozen** here; whether an unpack-friendly relayout is worth it is a **separate,
> larger change decided after this plan** from its results (see "Out of scope: the layout question"
> below). Evidence:
> [docs/bench/q5090-v2-current-top-kernels-ncu-report.md](../bench/q5090-v2-current-top-kernels-ncu-report.md),
> [docs/bench/q5090-v2-tuned-numeric-long-decode-nsys-report.md](../bench/q5090-v2-tuned-numeric-long-decode-nsys-report.md).
> Format byte facts: [docs/q5090_packed_file_format_v2.md](../q5090_packed_file_format_v2.md) §9.2.

## Goal

Minimize the decode **GPU kernel sum** by pushing each decode-critical row-split GEMV to its
cold-cache DRAM roofline (or a documented non-bandwidth ceiling). The kernel sum is `15.14 ms/token`
(44.139 s over 2915 tokens) and **86.8% of it is row-split GEMV** (38.303 s). Lifting the laggards
(`proj`/`out` at ~38% DRAM, `mlp_down` unpack-bound at ~45%) to roofline is the prize:

| metric | now | target this round |
| --- | ---: | ---: |
| decode kernel sum | `15.14 ms/tok` | `~11.3 ms/tok` |
| decode wall (no graphs) | `18.78 ms/tok` (53.2 tok/s) | `~14.9 ms/tok` (~67 tok/s) |
| decode wall (after later free CUDA-graph round) | — | `~11.5 ms/tok` (~85 tok/s) |

The single authoritative per-kernel metric is **cold-cache `achieved_dram_pct`** from
`qus_linear_op_bench` (its stream-copy ceiling ≈ `1512 GB/s` on this RTX 5090), cross-checked with
`ncu`. No claim is accepted without a round-0 baseline and an after measurement on the same rig.

## Non-goals / hard constraints

- **No CUDA Graphs, no launch-count change, no scheduling change.** This round only rewrites GEMV
  kernel bodies. Launch cost is a later round.
- **No model-schedule edits.** `src/model/qwen3_6_27b.cpp`, `include/qus/model/model.h` are untouched.
  No new fusion of projections (the `in_v`+`in_z` decode-fusion idea is a schedule change, out of
  scope).
- **No registry/dispatch edits.** `src/kernels/linear/plan/linear_plan.{h,cpp}` and
  `src/kernels/linear/linear.cpp` already route every target shape/qtype to its specialized launcher
  and already pass `WorkspaceArena& ws`. A loop rewrites **only its own** `.cu`/`.cuh` (launcher body +
  private `__global__` kernels/reduce). This is what makes the loops parallel-safe.
- **No on-disk layout or numeric-format change.** The v2 §9.1 Q5/Q6 byte packing is frozen. Each loop
  works within the current layout and **records its final `ncu` limiter** so the separate post-plan
  layout decision has evidence — but no loop touches the converter, the `.qus` file, the parser byte
  contract, or any `qtype`.
- **Correctness gate is the fp64/tolerance oracle**, not token identity. A kernel may reorder its
  reduction (e.g. split-K) freely as long as the op-level fp64 oracle stays green. Greedy token
  identity is a smoke check, never an optimization gate.
- **Prefill untouched.** All targets are the `T==1` decode specializations. The `LargeT`/GEMM path and
  prefill call sites are not in scope.

## Execution mode — subagent-driven, worktree-parallel, linear mainline history

A **coordinator** dispatches bounded work to fresh subagents, integrates onto `master`, and verifies.
The kernels are independent (each owns a disjoint `.cu`/`.cuh` pair, edits no shared file, and is
measured by `qus_linear_op_bench` + `ncu` — **never** a full-model run), so the loops run **in
parallel, one git worktree each**. No loop loads the 15 GB model or KV cache, so concurrent loops do
not contend for GPU memory; `ncu` replay only serializes wall-clock on the SM.

```
Task 0 (coordinator, direct)         baseline snapshot + ledger init   -> bench(q5090): commit on master
   │
   ├── A1 mlp_gate_up_34816 Q4  ┐
   ├── A2 mlp_down Q5           │
   ├── A3 proj_6144 Q5          │  parallel DETACHED worktrees, one kernel file each,
   ├── A4 out_6144 Q5           │  per-kernel design + tuning loop; coordinator integrates
   ├── A5 gdn_in_qk_4096 Q4     │  each finished kernel SEQUENTIALLY onto master as one
   ├── A6 attn_in_7168 Q4+Q5    │  perf(q5090): commit (disjoint files -> linear, no merges)
   └── A7 lm_head Q6 (optional) ┘
   │
Integration gate (coordinator, once)  build + ctest + fp64 + sanitizer + nsys re-profile -> bench(q5090): ledger
   │
Cleanup                               git worktree remove (all) + git worktree prune
```

### Coordinator procedure (the whole run, in order)

1. **Task 0** (direct): build the bench, record the baseline, init the ledger; commit
   `bench(q5090): record roofline-pushdown baseline`.
2. **Dispatch A1–A7 in parallel:** for each tag, create a detached worktree and dispatch one subagent
   per the Subagent assignment below.
3. **Integrate as loops return:** evidence-check, then apply+commit on `master` sequentially
   (Coordinator accept-and-integrate). Update the ledger.
4. **Integration gate** (once, after all integrated).
5. **Cleanup** (remove + prune worktrees).
6. **Review** subagent.

### Subagent assignment (exactly what each loop subagent is dispatched with — keep the prompt thin)

A loop subagent receives only a pointer, not re-explained detail: *"Read this plan. Execute task
**A\<i\>** — your design section + the Per-loop protocol + the Per-loop DoD. Edit only your owned
`.cu`/`.cuh`. Measure with `qus_linear_op_bench` + `ncu` + `ctest -R linear` only; never run the full
model; never edit a shared file. Return your evidence bundle under
`profiles/ncu-roofline-pushdown/<tag>/`."* Every specific (file, limiter, lever, target, commands,
gate) lives in this plan — the dispatch prompt carries none of it.

### Git workflow (points the user fixed)

- **No named feature branches.** A git worktree cannot share `master`'s checkout, so each parallel
  worktree is created **detached** at the base commit:
  `git worktree add --detach ../qus-tune-<tag> <base-commit>`.
- **Develop on the mainline.** Subagents iterate inside their detached worktree (own build dir),
  editing only their kernel `.cu`/`.cuh`. They do **not** create branches and need not commit there —
  they leave the final, verified files in the worktree working tree.
- **Linear history, no merges.** The coordinator integrates each *finished + verified* kernel
  **sequentially in the main `master` checkout**: harvest the two-file diff
  (`git -C ../qus-tune-<tag> diff -- <files> | git -C <main> apply`), then
  `git add <files> && git commit`. Because the kernel files are disjoint, sequential applies never
  conflict, so `master` advances as a straight line of one commit per kernel — **never** a merge
  commit, never a feature-branch merge.
- **Commit messages follow the mainline convention** (Conventional Commits, as in
  `git log`: `perf(q5090): tune mlp gate-up decode gemv`). One commit per kernel, e.g.
  `perf(q5090): vectorize mlp gate-up decode gemv loads`,
  `perf(q5090): split-k proj-6144 decode gemv`. Baseline/ledger/evidence under `profiles/` are
  separate `bench(q5090): ...` commits (matching `bench(q5090): record …` / `update … ledger`).
- **Clean up at the end.** After the integration gate is green, remove every worktree
  (`git worktree remove ../qus-tune-<tag>`) and `git worktree prune`. Only the main `master` worktree
  remains; because the worktrees were detached, there are no leftover branch refs to delete.

Each loop **follows its design scheme below AND runs a tuning loop**: implement the design, profile,
iterate one limiter per round until the gate. If `ncu` disproves the design's core hypothesis, the
subagent reports that back (it does not silently abandon the design). The coordinator maintains the
ledger as the single source of truth across the parallel loops.

### Coordinator ledger (Task 0 creates it; only the coordinator writes it)

`profiles/ncu-roofline-pushdown/tuning_status.md`:

```markdown
| tag | shape | qtype | worktree | status | rounds | base cold_us | base DRAM% | best cold_us | best DRAM% | Δ | final limiter (ncu) | layout-relevant? | evidence dir | integrated |
|-----|-------|-------|----------|--------|-------:|-------------:|-----------:|-------------:|-----------:|---|---------------------|------------------|--------------|------------|
| mlp_gate_up_34816 | MlpGateUp34816x5120 | Q4 | ../qus-tune-mlp_gate_up_34816 | pending | 0 | — | — | — | — | — | — | n/a | profiles/ncu-roofline-pushdown/mlp_gate_up_34816/ | no |
```

- `status` ∈ `pending → baselined → in-progress → gate-met | non-bw-wall | blocked → integrated`.
- `final limiter (ncu)`: the dominant limiter the loop ended on (DRAM / L1-TEX-unpack / occupancy / …).
- `layout-relevant?` (Q5/Q6 rows only): `yes` if the loop hit its occupancy ceiling and `ncu` **still**
  shows the 5-/6-bit unpack (L1/TEX + branch divergence) as the dominant limiter below the DRAM
  roofline. This is **pure evidence for the separate post-plan layout decision** — it does not trigger
  any work in this plan.
- Before dispatching a loop, the coordinator confirms the row is `pending`/`baselined` and its worktree
  isn't already created; never two loops on the same kernel file.

### Per-loop protocol (identical for every loop)

0. **Round 0 baseline.** In the loop's detached worktree, cold-cache bench + `ncu` the target as-is;
   save to `profiles/ncu-roofline-pushdown/<tag>/round0_baseline.*`. Must reproduce the coordinator's
   canonical baseline for this shape.
1. **Implement the design scheme** (this task's section).
2. **Profile + verify.** `ncu` the changed kernel (save `round<N>_<limiter>.*`); run the shape's fp64
   oracle; revert anything that fails the oracle.
3. **Iterate**, one limiter per round, until the gate or a documented wall.
4. **Report** a `round0 → best` table (cold_us, achieved_GB/s, DRAM%, % of `roof_us`) + a
   one-line-per-round `{limiter → change → result}` log to `profiles/ncu-roofline-pushdown/<tag>/summary.md`.

**Gate:** cold-cache `achieved_dram_pct` materially up and approaching `roof_us` for the shape, **or**
`ncu` documents a non-bandwidth wall — in which case name that wall in `summary.md` (and, for Q5/Q6,
set `layout-relevant? = yes`) and stop. Do not fight a format-shaped problem inside the kernel beyond
its occupancy ceiling. The shape's fp64 oracle is green; `git diff --stat` in the worktree shows only
the owned `.cu`/`.cuh`.

## Reference facts (verified in tree)

- RTX 5090 (GB202, sm_120, 32 GB, CUDA 13.1); single non-blocking compute stream
  (`src/core/device.cu`); branch `master`. Cold-cache stream-copy ceiling ≈ `1512 GB/s`
  (`bench/linear_op_bench.cu`).
- Decode = 64 layers (16 full-attention + 48 GDN), dense MLP `gate_up=[34816,5120]`,
  `down=[5120,17408]`, `hidden=5120`, `vocab=248320` (`include/qus/model/config.h`).
- Dispatch + per-call host validation: `src/kernels/linear/linear.cpp` (216–311). Every specialized
  rowsplit launcher already takes `(x, w, out, WorkspaceArena& ws, cudaStream_t)`; `mlp_down`'s
  `ArenaScope` (`linear_rowsplit_gemv_mlp_down.cu` 23–32, 177–179) is the split-K scratch pattern to
  clone.
- v2 ROW_SPLIT byte facts (spec §9.2): each row's codes are one contiguous run; each row's code run is
  **16-byte aligned** (all in-scope `K` are multiples of 128); the code plane is one uninterrupted
  byte stream; scales live in a separate row-major fp16 plane. These are what make `LDG.128`
  vectorization legal.
- Per-op bench CLI (`bench/linear_op_bench.cu`): `--shape <Name> --qtype <Q4|Q5|Q6> --repeat N
  --csv-out PATH` (cold, L2-flushed; prints `cold_us`, `ach_GB/s`, `DRAM_%`, `roof_us`);
  `--all-targets` runs all 8 rows. It dispatches through `kernels::linear`, so it measures the real
  kernel.
- fp64 oracle: `tests/kernels/test_linear.cpp` (+ `tests/kernels/op_tester.h`,
  `tests/kernels/q5090_pack.h`); run via `ctest --test-dir build -R linear`.

## Canonical per-kernel baselines (from the NCU report; the ledger re-measures fresh)

| tag | shape | qtype | cold_us | DRAM% (bench) | NCU SOL limiter | occupancy / waves/SM | structural state |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| `mlp_gate_up_34816` | MlpGateUp34816x5120 | Q4 | 100.3 | 62.5 | DRAM 75.6, SM 61.7 | 86.2% / 4.27 | warp-per-row, **1-byte loads** |
| `mlp_down` | MlpDown5120x17408 | Q5 | 66.9 | 57.9 | **L1/TEX 86.6, DRAM 45.4, br-eff 48%** | 90.4% / 5.02 | split-K=8 + reduce, **shuffle unpack** |
| `proj_6144` | Proj6144x5120 | Q5 | 35.8 | 38.3 | **DRAM 38.2, 0.75 waves** | 69.2% / 0.75 | **no split-K (underfilled)** + shuffle unpack |
| `out_6144` | Out5120x6144 | Q5 | 36.1 | 37.8 | **DRAM 35.4, 0.63 waves** | 60.1% / 0.63 | **no split-K (underfilled)** + shuffle unpack |
| `gdn_in_qk_4096` | GdnInQK4096x5120 | Q4 | — | — | (sample as round 0) | block-per-row 4096 | **1-byte loads** |
| `attn_in_7168` | AttnInQKV7168x5120 | Q4+Q5 | — | — | (sample as round 0) | hybrid | 1-byte (Q4) / shuffle (Q5) |
| `lm_head` | LmHead248320x5120 | Q6 | 759.1 | 86.7 | DRAM 62.9, SM 91.7 | 96.2% / 30.4 | healthy; funnel unpack hidden by N |

---

## Task 0 — Baseline snapshot + ledger (coordinator, direct; no subagent)

Build the bench, record the canonical cold-cache baseline for all 8 targets, sample `ncu` for the five
top kernels, and initialize the ledger. This is the authoritative "before" every loop compares against.

```bash
cmake --build build --target qus_linear_op_bench -j
mkdir -p profiles/ncu-roofline-pushdown/baseline
./build/bench/qus_linear_op_bench --all-targets --repeat 200 \
  --csv-out profiles/ncu-roofline-pushdown/baseline/op_bench.csv \
  | tee profiles/ncu-roofline-pushdown/baseline/op_bench.txt
# Per-target cold ncu (detailed + scheduler/warpstate) for the 5 top kernels, per ncu-kernel-profile skill.
```

**DoD:** `op_bench.csv` present with `achieved_dram_pct`/`roofline_us` per target; ledger created with
one `pending` row per tag and the baseline columns filled; the base commit recorded (the detached
worktrees are created at it). Commit the baseline + ledger on `master`:
`bench(q5090): record roofline-pushdown baseline`.

---

## Group A — per-kernel design schemes + tuning loops (parallel detached worktrees)

Each task: **one detached worktree, one kernel file pair, one subagent**, design below + tuning-loop
protocol above, evidence-gated integration. Priority = integration order (biggest decode share first).

### A1 — `mlp_gate_up_34816` Q4  (22.4% of decode; the top kernel)

- **Owns:** `src/kernels/linear/gemv/linear_rowsplit_gemv_mlp_gate_up_34816.{cu,cuh}`.
- **Limiter (round-0 hypothesis):** memory-throughput-bound but capped by **1-byte global loads**.
  Today each lane issues one `LDG.U8` per group (`...mlp_gate_up_34816.cu` 57: `code_row[group*32 +
  lane]`), 80 groups/row → 80 narrow sector loads; DRAM only 62% cold.
- **Design:** keep warp-per-row + final warp-reduce, but stream the row's 2560-byte code run with
  **`LDG.128` (`uint4`)** loads — each thread pulls 16 contiguous code bytes (= 32 Q4 codes = 32
  consecutive K-values inside one group; group is 32 B so a `uint4` is half a group ⇒ one scale) and
  unpacks in registers. Cuts code-load instructions ~16×. The row code run is 16-byte aligned (§9.2),
  so `uint4` loads are legal. Map lanes so the per-lane `x` reads stay coalesced (`x` is reused from
  L2; weights dominate traffic). Optionally vectorize the scale-plane read too (the per-group
  `__shfl` scale broadcast is cheap — Q4 has 100% branch efficiency — so it is secondary).
- **Target:** `achieved_dram_pct` ≈ 90% (cold_us ~100 → ~70).

### A2 — `mlp_down` Q5  (18.4%; the unpack-bound kernel)

- **Owns:** `src/kernels/linear/gemv/linear_rowsplit_gemv_mlp_down.{cu,cuh}`.
- **Limiter (confirmed by NCU):** **not DRAM** — L1/TEX 86.6%, branch-eff 48%, DRAM 45.4%. The 5-bit
  unpack uses a divergent `lane<10/<20/<30` region branch + cross-lane `__shfl` + `__funnelshift_r`
  (`...mlp_down.cu` 86–118, `accumulate_group` 38–61).
- **Design (in-kernel, no format change):** remove the cross-lane shuffle. Have **each lane read its
  own bit-slice directly**: value pair `(2L, 2L+1)` lives at bit `10*L`, i.e. byte `floor(10*L/8)` of
  the 40-byte group, so lane `L` loads a `uint32` at that (in-bounds) byte offset and `funnelshift`s
  locally — no `__shfl`, no region branch. (Alternative the loop may compare: 3 lanes cooperatively
  load the 40 B group as `uint4`+`uint2` into shared/registers, then all lanes funnelshift from the
  shared copy.) Then **re-tune `kSplitK`** (currently 8): with the unpack cheaper, fewer splits may hit
  the same occupancy with less reduce overhead (`...mlp_down.cu` 21, 158–192). Keep the split-K +
  reduce structure; scratch via `ArenaScope`/`ws`.
- **Target:** push DRAM% as high as the current layout allows. **If, at its occupancy ceiling, `ncu`
  still shows the unpack / L1-TEX as the dominant wall, record `final limiter = L1-TEX unpack` and
  `layout-relevant? = yes` in `summary.md` and stop** — that is evidence for the separate post-plan
  layout decision, not a cue to change the byte layout here.

### A3 — `proj_6144` Q5  (11.4%; underfilled)

- **Owns:** `src/kernels/linear/gemv/linear_rowsplit_gemv_proj_6144.{cu,cuh}`. (Serves GDN `in_v` and
  `in_z`, 96 launches/step.)
- **Limiter:** **underfilled** — warp-per-row, `grid = N/4 = 1536` blocks, 0.75 waves/SM, DRAM 38%
  (`...proj_6144.cu` 157–160). Also carries the A2 Q5 shuffle unpack.
- **Design:** add **split-K** (clone `mlp_down`'s `dim3(grid, kSplitK)` body + private reduce kernel,
  scratch from `ws` via `ArenaScope`); tune the split factor to fill ~170 SMs (need ≳2720 resident
  4-warp blocks for one full wave ⇒ split ×2–×4). Apply the **A2 register-local Q5 unpack**. Same
  `layout-relevant?` reporting rule as A2.
- **Target:** ≈ 75–80% DRAM (cold_us ~36 → ~17).

### A4 — `out_6144` Q5  (7.5%; underfilled)

- **Owns:** `src/kernels/linear/gemv/linear_rowsplit_gemv_out_6144.{cu,cuh}`. (Serves attn `o_proj`
  and GDN `out_proj`, 64 launches/step.)
- **Limiter:** **underfilled** — warp-per-row, `grid = N/4 = 1280` blocks, 0.63 waves/SM, DRAM 35%
  (`...out_6144.cu` 125–129). Same Q5 unpack.
- **Design:** identical to A3 — split-K + register-local Q5 unpack; same `layout-relevant?` rule.
- **Target:** ≈ 75–80% DRAM (cold_us ~36 → ~17).

### A5 — `gdn_in_qk_4096` Q4  (2.4%)

- **Owns:** `src/kernels/linear/gemv/linear_rowsplit_gemv_gdn_in_qk_4096.{cu,cuh}`.
- **Limiter:** already **block-per-row** (`grid = 4096`, 4 warps cooperate, shared reduce —
  `...gdn_in_qk_4096.cu` 42–101), so occupancy is reasonable; the lever is the same **1-byte Q4 loads**
  as A1 (line 71/84). Confirm with round-0 `ncu`.
- **Design:** `LDG.128` vectorized Q4 loads as in A1, on the block-per-row structure (each warp owns 20
  groups). Keep the shared-memory cross-warp reduce.
- **Target:** ≈ 85–90% DRAM.

### A6 — `attn_in_7168` Q4 **and** Q5  (3.3% combined; one file, both kernels)

- **Owns:** `src/kernels/linear/gemv/linear_rowsplit_gemv_attn_in_7168.{cu,cuh}`.
- **Limiter:** hybrid scheme already present — warp-per-row for the 6144 proj rows, warps-cooperate for
  the 1024 KV-tail rows (`...attn_in_7168.cu` 80–161 Q4, 163–330 Q5). Q4 path has 1-byte loads; Q5
  path has the shuffle unpack; the proj-row region is mildly underfilled.
- **Design:** apply **A1 `LDG.128`** to the Q4 kernel and **A2 register-local unpack** to the Q5
  kernel; consider split-K on the proj-row region if round-0 `ncu` shows underfill. Tune both kernels
  in this one loop (they share the file). Same `layout-relevant?` rule for the Q5 kernel.
- **Target:** ≈ 80% DRAM both qtypes.

### A7 — `lm_head` Q6  (4.4%; optional, low priority)

- **Owns:** `src/kernels/linear/gemv/linear_rowsplit_gemv_lm_head.{cu,cuh}`.
- **Limiter:** already healthy (86.7% DRAM, 96% occupancy) — its huge `N` (62080 blocks) hides the
  funnel unpack. Small headroom only.
- **Design:** `LDG.128` for the 48-byte Q6 group + register-local unpack (`...lm_head.cu` 68–95). Run
  **only if** A1–A6 finish with time to spare, or if the post-Group-A nsys shows `lm_head` rising in
  share.
- **Target:** ≈ 92% DRAM.

### Per-loop DoD (the evidence bundle each subagent returns)

- `round0_baseline` + best-round `ncu`/op-bench artifacts + `summary.md` (before→after table +
  per-round limiter→change→result log) under `profiles/ncu-roofline-pushdown/<tag>/`.
- Cold-cache `achieved_dram_pct` **materially beats round-0 and approaches `roof_us`**, OR `ncu`
  documents a non-bandwidth wall named in `summary.md` (Q5/Q6: `layout-relevant? = yes`).
- Shape's fp64 oracle green (`ctest --test-dir build -R linear`); `compute-sanitizer --tool memcheck
  ./build/tests/qus_linear_test` clean (covers any new split-K scratch path).
- `git diff --stat` in the worktree touches only the owned `.cu`/`.cuh`.

### Coordinator accept-and-integrate (per loop, evidence-gated, linear on master)

For each finished loop, **before** integrating: open `summary.md`; confirm (a) real cold-cache
improvement or a documented wall, (b) the limiter was `ncu`-discovered, (c) fp64 oracle green, (d) the
worktree diff touches only the owned file. Then integrate **onto `master`, sequentially**:

```bash
git -C ../qus-tune-<tag> diff -- <owned .cu> <owned .cuh> | git -C <main> apply
cmake --build build -j && ctest --test-dir build -R linear     # re-verify on master
git add <owned .cu> <owned .cuh>
git commit -m "perf(q5090): <imperative summary of the kernel change>"
```

Record `best *`, `Δ`, `final limiter`, `layout-relevant?`, `integrated=yes` in the ledger (commit the
ledger update as `bench(q5090): …`). Disjoint files ⇒ the applies never conflict ⇒ `master` stays a
straight line (no merge commits). A loop failing (a)–(d) is re-dispatched (resume the subagent with its
prior evidence) or marked `blocked`.

### Integration gate (coordinator, after integrating all loops — ONCE, sequential)

```bash
cmake --build build -j && ctest --test-dir build               # green
ctest --test-dir build -R linear                               # fp64 oracle, every tuned shape
compute-sanitizer --tool memcheck ./build/tests/qus_linear_test # split-K scratch paths
# nsys decode re-profile (same invocation as the tuned-numeric report) — measure kernel-sum drop:
nsys profile --force-overwrite=true --stats=false --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o profiles/ncu-roofline-pushdown/decode_after \
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
    --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
    --prompt "怎么用fem方法求解heat equation。" --max-context 8192 --max-new 4096
```

**DoD:** ctest green; fp64 oracle green for every tuned shape; `compute-sanitizer` clean; nsys shows a
measurable decode kernel-sum / tok-s improvement; the new top decode kernel is named in the ledger.
Commit the final nsys evidence + ledger as `bench(q5090): record roofline-pushdown result`.

### Cleanup (coordinator, after the gate)

```bash
git worktree remove ../qus-tune-<tag>   # for every tag
git worktree prune
git worktree list                       # only the main master worktree remains
```

### Review (independent subagent, after integration — risk: CUDA kernels + numerics + GPU memory)

Per AGENTS.md (CUDA kernels, numerical behavior, GPU-memory lifetime → strict review): every speedup is
cold-cache `ncu`/`nsys`-backed with a round-0→best comparison; each loop's `final limiter` /
`layout-relevant?` verdict is consistent with its `ncu` evidence; fp64 oracle green on every shape;
split-K scratch goes through `ws`/`ArenaScope` and `compute-sanitizer` is clean; the history is linear
(`git log --graph --oneline` shows no merge commits) with one `perf(q5090): …` commit per kernel; no
loop edited a shared registry/schedule/format file; no worktrees remain.

---

## Out of scope: the layout question (decided separately, after this plan)

The NCU report already shows `mlp_down` is unpack-bound (86.6% L1/TEX, 48% branch-eff, 45% DRAM), and
the v2 §9.1 LSB-first bitstream is the likely reason the small-`N` Q5 GEMVs cannot reach the DRAM
roofline by in-kernel means alone. **Whether to change the on-disk Q5/Q6 layout (e.g. a plane-split
`nibble + high-bits` encoding) is intentionally NOT part of this plan.** It is a much larger change —
converter, file spec, `.qus` artifact regeneration, parser byte contract, golden tests, and every
Q5/Q6 kernel — and will be its own plan if and only if the results here justify it.

This plan's job is to feed that decision with evidence: after the integration gate, each Q5/Q6 row's
`final limiter` and `layout-relevant?` columns in the ledger say exactly which kernels remained
unpack-bound at their occupancy ceiling and how far below `roof_us` they stalled. We analyze those
results separately and decide then.

---

## Self-review (against AGENTS.md + the user's task shape)

- **Per-kernel design schemes** — A1–A7 each carry a concrete limiter hypothesis, technique, structure,
  and target ✓. Each loop "follows the design AND runs a tuning loop" ✓.
- **Subagent-driven, worktree-parallel** — loops own disjoint kernel files, edit no shared
  registry/schedule/format file, and measure with the per-op bench (no model load) ⇒ parallel-safe, low
  GPU contention ✓.
- **Linear mainline history (no branches, no merges)** — detached worktrees for parallel iteration;
  the coordinator applies each finished kernel's two-file diff and commits it sequentially on `master`;
  one `perf(q5090): …` commit per kernel; worktrees removed + pruned at the end ✓.
- **Layout change excluded** — no Task C; the on-disk layout is frozen; the relayout decision is a
  separate post-plan analysis fed by the ledger's `final limiter` / `layout-relevant?` evidence ✓.
- **Coordinator ledger** — `tuning_status.md` tracks status/rounds/Δ/limiter/integrated per kernel ✓.
- **Plan-writing rules** — goal/non-goals, execution mode, scope/ownership per task, task breakdown,
  reading lists (inline per task), DoD + verification commands, risk-scaled review ✓.
- **No CUDA Graphs / no schedule change / no bytes-per-token change / no format change** — honored ✓.
- **Testing policy** — gates use the existing fp64 oracle, `compute-sanitizer`, the per-op bench, and
  nsys/ncu evidence; no low-value tests added ✓.
