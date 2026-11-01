# Merging upstream into the two-GPU fork

This fork is upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) plus a two-GPU tensor
parallel layer (`--tp 2`, [Two-GPU tensor parallelism](tensor-parallel.md)). This guide says where
that layer lives, which lines it still needs inside upstream's files, how to resolve the conflicts
a merge typically produces, and which gate a merge must pass before it is kept.

The rule the layout follows: **our code lives in our files; upstream's files carry only hooks**
(an include, a registration line, one call). A hook is small and sits where upstream rarely edits,
so a merge either applies cleanly or conflicts on a line whose resolution is obvious.

## 1. Where our code lives

| Path | What |
|---|---|
| `src/models/qwen3_5/execution/tp.{h,cpp}`, `execution/tp2/` | `TpExecution`, TextContext's tp 2 bindings and split MTP head (`text_tp2.inc`), the split attention/GDN/FFN/MTP forms (`*_split.{h,cpp}`), split projection helpers and workspace recipes; `split.h` includes them all |
| `src/models/qwen3_5/program/tp2/` | ProgramImpl's rank 1 runtime, mirrors, mailbox probe and step-down (`program_impl_tp2.inc`), the tp2 workspace plan and option checks (`startup_tp2.inc`), MTP bridge, split verification and split prefill head |
| `src/models/qwen3_5/load/sharding.{h,cpp}` | the placement rules (`shard_rule`) |
| `src/ops/wrapper/tp2/`, `include/ninfer/ops/tp2/` | the column/row-parallel Op forms and their declarations |
| `src/ops/{attn_input_proj,gdn_input_proj,linear_swiglu,context_kv_materialize}/tp2/` | shard and half problems built from upstream's own launcher sources (§2.4), including the DFlash2 drafter's Q8 QKV shard and its four-KV-head context materialization |
| `src/ops/{linear_topk,softmax_attention/sliding_window,rmsnorm_rope,dynamic_grouped_conv}/tp2/` | the split DFlash2 drafter's Op forms: the vocabulary-split top sixteen, the 16/4 head blocks, the convolution finish |
| `src/ops/linear/tp2/`, `src/ops/linear/*/shapes/n*` (half shapes) | split `linear()` forms; the half problems' registrations |
| `src/ops/common/{allreduce,peer_mailbox}.cu`, `src/ops/kernel/peer_exchange.cuh`, `src/ops/launcher/{argmax_split,concat_rows}.*`, `include/ninfer/ops/{allreduce,peer_mailbox}.h` | transport and gathers |
| `src/ops/tp2/sources.cmake` | every tp2 Op source |
| `src/core/tp2/device_tuning.h` | the per-GPU table (§5) |
| `src/core/tp2/` | dual-device CUDA Graph capture (`decode_graph_peer.h`, `decode_graph_tp2.inc`), KV mirrors (`paged_kv_cache_tp2.inc`) |
| `src/runtime/engine/tp2/`, `src/serve/tp2/` | tp 2 Engine option checks, startup log lines |
| `tools/tp2/` | `gate.sh` (§4), `surface.sh` (§3), `mailbox_probe.cu` |
| `tests/ops/test_*_split.cpp`, `test_allreduce.cpp`, `test_attention_headlocal.cpp`, `tests/models/qwen3_5/test_*tp2*.cpp` | two-device tests |

## 2. The four ways our code attaches to upstream

### 2.1 Own translation units

Code that needs nothing private from upstream is an ordinary `.cpp`/`.cu` of ours
(`execution/tp2/*_split.cpp`, `allreduce.cu`, ...). Sources are registered by our own
`sources.cmake` files, each included with one line from the upstream list they extend:
`src/ops/basic_sources.cmake` → `src/ops/tp2/sources.cmake`,
`src/models/qwen3_5/execution_sources.cmake` → `execution/tp2/sources.cmake`.

### 2.2 Translation-unit fragments (`tp2/*.inc`, included at the end of an upstream `.cpp`)

When our code needs an upstream file's private helpers (its anonymous namespace), it lives in a
fragment that the upstream file includes **once, as its last line, after its closing namespace**:

```cpp
} // namespace ninfer::ops

#include "ops/wrapper/tp2/linear_add_tp2.inc"
```

The fragment includes what it needs, reopens the namespace and defines its functions; it sees the
upstream file's helpers because it is the same translation unit. Nothing is copied, so a fix upstream
makes to a shared helper reaches the two-device code too, and the compiled code is the one of the
original single file. Where an upstream function calls into the fragment before the fragment is
seen, a one-line forward declaration stays in the upstream file's anonymous namespace (e.g.
`log_tensor_parallel` in `serve/operational_log.cpp`), and an inline tp2 block of an upstream function
became one call (`construct_tensor_parallel()` and `bind_tensor_parallel_execution()` in
ProgramImpl's constructor, `validate_tensor_parallel_options()` in two places).

### 2.3 Class-body fragments (`*.inc` included inside an upstream class)

`TextContext` and `ProgramImpl` need tp2 members. Their declarations live in
`execution/tp2/text_context_{public,private}.inc` and `program/tp2/program_impl_{public,private}.inc`,
included inside the class body by one line each. `program_impl_public.inc` sits right after
`workspace_plan`, where the tp2 members stood before: rank 1's storage and the mailbox must be
declared before rank 0's storage and the graph families so they are destroyed after them. **Do not
move that include.** The other fragments can be anywhere in their section.

### 2.4 Shards compiled from upstream's own sources (`src/ops/*/tp2/*_shard_*.cu`)

A shard (`attn_input_proj` [7168,5120], `gdn_input_proj` [8192,5120]) or half (`linear_swiglu`
[17408,5120]) is the parent's problem with fewer output rows and the same K, so it must take the
parent's routes, schedules and token cutoffs. Instead of editing upstream's launchers to accept a
second output type, each shard translation unit includes upstream's launcher source after a names
header that renames the output type (or geometry) and the entry points:

```cpp
#include "ops/attn_input_proj/tp2/fp8_attn_input_shard_prelude.h"   // family headers, real names
#include "ops/attn_input_proj/tp2/fp8_attn_input_shard_names.h"     // #define Fp8AttentionInputOutput ...
#include "ops/attn_input_proj/fp8/fp8_attn_input_a8.cu"             // upstream's source
```

The prelude includes the family's headers first, under their real names, so that only the
upstream source's own code is renamed. The plan/dispatch source is compiled the same way, so the
shard's dispatcher is upstream's route logic calling the shard's launchers; the two-device wrapper
calls it (`fp8_attn_input_shard_dispatch`). What is genuinely the shard's own stays explicit in
these files: the FP8 split-K partial reservations (`*_shard_partial_capacity_bytes`) and the
NVFP4 attention shard's TMA scale tile at T=1024 (`nvfp4_attn_input_shard_a4.cu`, via
`tp2::DeviceTuning`).

After a merge that rewrites these launchers the shard follows automatically. Two changes would make
it wrong silently, and `tools/tp2/check_shard_sources.py` (the gate's `shards` stage) fails on both:
a template, inline or kernel definition outside an anonymous namespace in a re-included source (the
parent's and the shard's bodies would share one name and the linker would keep one), and a renamed
identifier appearing in a header the prelude does not include first. The rest breaks at compile or
link time:

- upstream renames or removes the output type, a launcher, or a source file → update the names
  header or the include in `tp2/`;
- upstream adds a non-static function to one of these sources → duplicate symbol at link time;
  add a rename for it in the names header (see how `fp8_attn_input_partial_capacity_bytes` is
  renamed to an unused name in `fp8_attn_input_shard_a8.cu`);
- upstream starts using a parent constant for something the shard must change (a row count, not
  K) → the split tests (`*_split_test`) fail; add a rename or an explicit shard override.

`linear_add` does not use this pattern: its half problems differ from their parents in K, and
upstream selects the schedule family by comparing K with 6144/17408, so the halves need the
`output_family(K)` hooks in upstream's launchers (§3).

## 3. What still lives in upstream files

`tools/tp2/surface.sh [<upstream-ref> [<ref>]]` measures the fork's footprint inside upstream's
files (files, lines, hunks, and the files with the most hunks); run it before and after a merge. The
remaining hunks fall into these kinds; each file's hunks are small unless noted.

| Kind | Where | Why it cannot move |
|---|---|---|
| Hooks: one include, one call, one registration line | `text.cpp/h`, `program_impl.cpp/h`, `startup.cpp`, `prefill.cpp`, `speculative/{mtp,target_verification}.cpp`, `decode_graph.{h,cpp}`, `paged_kv_cache.cpp`, `serve/operational_log.cpp`, `model_instance.cpp`, `basic_sources.cmake`, `execution_sources.cmake`, `linear/*/shapes.h` and `sources.cmake` | by design |
| Two-device branches inside upstream functions | `program/{graphs,prefill,decode,storage/context,transactions/*}.cpp`, `speculative/mtp.cpp` round body, `execution/{vision,draft,parameters}.cpp`, `startup.cpp` (per-rank layouts, the shared workspace helpers) | they read and write the function's locals at many points; `prepare_graphs()` is the largest (retry loop and mailbox step-down) |
| Shape registrations inside upstream Ops | `linear_add` (FP8/NVFP4 `output_family(K)`, half cases), `linear_swiglu` wrapper (half shape, routing), `causal_conv1d` (split geometries), `gdn_projected_conv.cu` (5120-channel geometry), `mtp_pack` (12/2 layout), `gdn_gating_proj` (24-head shard kernels), GDN replay and recurrence (8/24 heads), softmax attention (12/2 instances); for Qwen3.8-Flash-Next: GDN replay fold (36 layers, full and 8/24 shard), `gdn_gating` (24-head shard), the sigmoid epilogue of `rmsnorm` (`gated_rmsnorm_sigmoid`) and the public `gdn_projected_conv_{snapshot,record}` wrappers | the Op selects on the shape inside its own code |
| Engine-wide, public or product surface | `include/ninfer/types.h`, `apps/*/options.cpp`, `serve_options.cpp`, `kv_capacity.cpp`, `engine.cpp`, `resource_manager.h` (Host-less reclaim), artifact `binder`/`materializer`/`views` (multi-device materialization) | public API, CLI and loading paths |
| Head counts and geometries the two-device drafter instantiates | `softmax_attention/common/context_query.cuh` and `sliding_window/kernel.cuh` (query/KV head counts as template parameters, defaulting to 32/8), `context_kv_materialize/materialize.cu` (the KV-head count as one named, overridable constant), `weight_input.cpp` (the drafter's `[2048|512|512,5120]` QKV shard), `state/state_image.cpp` (the mirror replays DFlash rings when the mirror has them) | the kernels hard-coded one geometry; parameterizing keeps one source for both |
| Generic changes, candidates for upstream | `FuncAttrPerDevice` at every `cudaFuncSetAttribute` site, device SM count and balanced splits in the INT8 attention plan and kernels, A16 sliced-K with several row tiles per CTA (`nvfp4_a16_sliced_k_mma.cuh`), explicit rejection of unregistered head counts, per-step KV row publication (`publish_kv_rows`) | useful at one GPU too; proposing them upstream removes the conflicts at the root |

**Before taking upstream's side of a conflict**, check what else of ours the file carries:
`git log --no-merges <upstream>..<ours> -- <file>`. Not every fork change in an upstream file is tp2
plumbing; some are bit-identical performance work that the gate only sees as ms/round (it happened
during this refactor: the 70-SM sliced-K schedules of `91582814` in `nvfp4_linear_swiglu_small_t.cu`
went missing, texts stayed identical, the 60 greedy prompts measured +2.8 % ms/round).

Tests: the tp2 cases added to upstream test files (`test_decode_graph.cpp`, `test_kv_cache.cpp`,
`test_serve_options.cpp`, `test_cli_options.cpp`, ...) are still inline.

## 4. The gate (mandatory for every merge and every tp2 change)

```bash
# once per upstream base or intended behavior change: record the reference from the known-good build
GATE_PROMPTS=prompts60.jsonl tools/tp2/gate.sh --record <good-build> <ref-dir>
# every candidate, under a GPU lease on the workstation
tools/tp2/gate.sh [--attention] [--stages ...] <build> <ref-dir>
```

Stages: the unit/Op/split ctest set with the tp2 real tests (`--attention` adds the 12-minute
softmax attention suite, needed when `softmax_attention/` changes); the tp1 golden on the
synthetic model; `ninfer-perplexity --tp 2` at 65536/32768 and 4096/2048 with INT8 KV (every
printed digit must equal the reference; for a commit that changes bits on purpose,
`GATE_PPL_PAIRED=1` judges a changed table by a paired per-window test instead, PASS when not
significantly worse and within 0.5 %); the 60 greedy prompts through `ninfer-serve` with the
production flags (texts must be identical; ms/round is reported against the reference, ±1 % is
noise on the workstation, beyond that rerun alternated A/B before deciding); a short DFlash2 K=7 run.
About 13 minutes. `tools/tp2/README.md` has the paths and options.

A merge of upstream changes numerics by design; the gate is then run twice: the tp1 golden against
upstream's own build (`tools/golden/`), and the tp2 stages recorded afresh from the merged build
after PPL and GSM8K have been judged acceptable (the procedure of the 30/09 merge dossier).

## 5. Adding a new GPU

The per-GPU values of the fork live in one table, `kDeviceTuningRows` in
[`src/core/tp2/device_tuning.h`](../../src/core/tp2/device_tuning.h), matched by compute capability
and SM count (first match wins). Its last row matches every device and carries the values measured
on two RTX 5070 Ti; with it, every device runs exactly what it ran before the table existed. For a
new board:

1. Build and run the gate on the new pair with the current table (`tools/tp2/gate.sh --record` on
   the new hardware gives its reference; the texts will differ from another board's).
2. Measure the fields that are per-GPU (the table documents each one's procedure): the CUDA Graph
   allowances (the server logs `cuda graphs | rank N: observed ... allowance ...`; take
   max(3 x observed, 8 MiB)), the NVFP4 attention shard's TMA tile bound (`bench/ops` on the
   [7168,5120] shard at T=1024/1025), and whether the attention plan should size its waves for
   another SM count than the device's.
3. Add a row above the catch-all with the new values and a `source` naming the board, clocks,
   date and commit; nothing else is edited.
4. For schedules and crossovers of the shard and half problems (inherited from upstream's
   parents), follow the procedure of [Two-GPU tensor parallelism](tensor-parallel.md#11-known-limits-and-not-done-items)
   (Schedules and thresholds are per GPU); a retuned value becomes a table field read by the
   `src/ops/*/tp2/` shard files, never an edit of upstream's launchers.
5. Run the full gate, with `--attention` if the attention value changed, and GSM8K on a change
   that is not bit-identical (perplexity at 1024-token chunks does not see decode-width routes).

The header of `device_tuning.h` lists the per-GPU choices that are still compile-time inside
upstream's Op code (shared by every device) and would have to become table fields first.

## 6. Adding a new model to tp2

Two-GPU execution exists for the dense Qwen3.5 family (Qwen3.6/3.8 artifacts). What is generic and
what a new model family implements:

| Generic (reuse as is) | Model-specific today (Qwen3.5), to implement for a new family |
|---|---|
| `ExecutionContext`, peer access, `--tp/--devices` options, symmetric KV capacity (`runtime/engine`) | placement rules per logical parameter: `shard_rule` in [`load/sharding.cpp`](../../src/models/qwen3_5/load/sharding.cpp) (Rows/Columns/Replicated/PrimaryOnly/SingleDevice) |
| sharded materialization (`artifact::Binder` shard resolver, `tensor_slice`, `materialize` on an `ExecutionContext`) | the config split: `shard_text_config` in [`execution/tp.cpp`](../../src/models/qwen3_5/execution/tp.cpp) (which heads, channels and widths halve) |
| transport: `allreduce_sum`, `PeerEvents`, `PeerMailbox`, `concat_rows`, split argmax | the split schedule: `execution/tp2/` (the `*_split` forms the Qwen4 forward and the MTP head use: which projection is column- or row-parallel, where the two all-reduces of a layer sit) |
| split Op forms (`include/ninfer/ops/tp2/`): column/row-parallel `linear`, `linear_add`, `linear_swiglu`, `attn_input_proj`, `gdn_input_proj`, `gdn_gating_proj` | the shard shapes of the new model's dimensions: registered in each Op like any problem (`linear` shapes files; the `tp2/*_shard_*` sets of the fused Ops, one more set per new shard shape) |
| dual-device CUDA Graph capture (`core/tp2`), KV and StateImage mirrors | the Program integration: rank 1's runtime and mirrors (`program/tp2/program_impl_tp2.inc`), the rounds' peer ingress, `TpExecution` |
| `tp2::DeviceTuning`, `tools/tp2/gate.sh` | per-rank persistent layouts and the tp2 workspace plan (`program/tp2/startup_tp2.inc`), the tp2 checks of the planner and the Engine |

The order that worked for Qwen3.5: shard rules and config split with a shard-map test; the split
Op forms at the new shapes with `*_split_test` against the single-device Op; a synthetic two-layer
model whose tp 2 logits match tp 1 within two BF16 ulps (`test_text_context_tp2.cpp`); the real
artifact tests; then the gate recorded for the new model (its own prompts and perplexity corpus).

A drafter on two devices (the tensor-parallel DFlash2 drafter) is the same work at a smaller scale:
its placement rules in `shard_rule` (today PrimaryOnly), its Q8 shard shapes as more
`src/ops/*/tp2/` sets compiled from upstream's Q8 sources, the split drafter round in
`execution/tp2/` and `program/tp2/` with one call from the upstream DFlash2 round, and its state in
the mirrors.
