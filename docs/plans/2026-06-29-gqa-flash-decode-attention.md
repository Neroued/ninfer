# GQA Flash Decode Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current correctness-first GQA decode attention path with a fixed-shape Qwen3.6-27B split-KV flash-decoding implementation that targets high RTX 5090 memory throughput for long decode.

**Architecture:** The new decode path uses a decode-optimal KV cache layout, a split-KV partial kernel that processes one `(kv_head, q_subgroup, token_tile)`, and a fused reducer that combines tile-local online-softmax partials into the final BF16 attention output. The implementation treats the inner thread mapping as a first-class profiled branch: compare block-reduce-256 against warp-per-query-head/8-dims-per-thread instead of assuming one mapping is optimal. The plan deliberately deletes the old decode layout/append behavior rather than preserving compatibility.

**Tech Stack:** C++20, CUDA C++, BF16 CUDA intrinsics, Nsight Compute (`ncu`), Nsight Systems (`nsys`), existing L0 `DeviceArena`/`WorkspaceArena`, existing L1 wrapper/launcher/kernel layering.

---

## Coordinator Rules

- Use subagent-driven development for execution. Do not implement this plan inline unless the user explicitly changes execution mode.
- Every spawned implementer, reviewer, profiler, and auditor subagent must use:
  - `model: "gpt-5.5"`
  - `reasoning_effort: "xhigh"`
  - `service_tier: "priority"`
- Treat `gpt-5.5` as the best currently exposed model. If it is unavailable, stop and report BLOCKED; do not silently fall back.
- Use `multi_agent_v1.spawn_agent` with an explicit model override for every subagent. Example:

```json
{
  "agent_type": "worker",
  "model": "gpt-5.5",
  "reasoning_effort": "xhigh",
  "service_tier": "priority",
  "message": "Use the full task text copied from the selected section of this plan."
}
```

- Tell every worker: "You are not alone in the codebase. Do not revert edits made by other workers. Work only in your assigned files, and adapt to already-landed changes."
- Do not dispatch code-editing workers in parallel when their write sets overlap.
- Dispatch review/profiling workers only after the code checkpoint they review has been committed or clearly isolated.
- Treat Task 1 plus Task Group 2 as one functional integration boundary. Task 1 alone intentionally changes the KV physical layout before the attention wrapper and kernels understand it, so the coordinator must not run full-tree verification or make a bisectable commit until Sub-Round 2D is complete.
- After each implementation task or explicitly grouped integration boundary:
  1. Dispatch a spec-compliance reviewer using `gpt-5.5`.
  2. Fix all spec issues.
  3. Dispatch a code-quality reviewer using `gpt-5.5`.
  4. Fix all Critical and Important issues before moving on.
- Use no backward compatibility shims, old aliases, old flags, old layout branches, or deprecation paths.
- Follow the repository testing policy. Tests are allowed here only because they cover numerical attention correctness and GPU memory/lifetime risk. Do not add source-structure tests.

## Profiling Skill Requirements

Performance agents must read and follow these skill files before collecting profiles:

- `/home/neroued/.codex/skills/ncu-kernel-profile/SKILL.md`
- `/home/neroued/.codex/skills/nsys-inference-analysis/SKILL.md`

The required preflight command before any `ncu` round is:

```bash
~/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
```

Use the concrete `ncu` commands written in the performance rounds below. Do not report a performance
claim from a command that was not saved with an exact report path under `profiles/ncu-attn/`.

The required end-to-end `nsys` pattern after the per-op kernel is viable is:

```bash
mkdir -p profiles/nsys
nsys profile --force-overwrite=true \
  -o profiles/nsys/fem_m4096_flash_decode \
  --trace=cuda,nvtx \
  --sample=none --cpuctxsw=none \
  ./build/src/qus out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus \
    --tokenizer /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
    --prompt "怎么用fem方法求解heat equation。" \
    --max-context 8192 \
    --max-new 4096 \
  > profiles/nsys/fem_m4096_flash_decode.stdout.txt \
  2> profiles/nsys/fem_m4096_flash_decode.stderr.txt

python3 ~/.codex/skills/nsys-inference-analysis/scripts/nsys_inference_summary.py \
    profiles/nsys/fem_m4096_flash_decode.nsys-rep \
    --out profiles/nsys/fem_m4096_flash_decode.nsys-summary.md
```

## Scope And Ownership

### Project-Owned Files Expected To Change

- `include/qus/core/kv_cache.h`
  - Owns the resident KV cache contract and physical layout metadata.
- `src/core/kv_cache.cpp`
  - Allocates the new decode-optimal KV tensors and exposes layout-safe slot helpers.
- `src/runtime/engine.cpp`
  - Computes default cache arena bytes from padded KV context rather than unpadded `max_ctx`.
- `include/qus/kernels/gqa_attention.h`
  - Updates the decode API to accept `WorkspaceArena&`.
- `src/kernels/wrapper/gqa_attention.cpp`
  - Validates the new KV layout, computes decode tile policy, allocates decode scratch, and calls launchers.
- `src/kernels/launcher/gqa_attention.h`
  - Declares the new split decode launch entry and helper workspace byte functions if needed.
- `src/kernels/launcher/gqa_attention_decode.cu`
  - Replaces old append + 24-block decode launch with split-KV launch orchestration.
- `src/kernels/kernel/gqa_attention_decode.cuh`
  - Replaces old decode kernels with partial and fused output-reduce kernels.
- `src/kernels/kernel/gqa_attention_prefill.cuh`
  - Updates prefill cache fill and cache reads to the new KV layout.
- `src/kernels/launcher/gqa_attention_prefill.cu`
  - Updates fill launch assumptions only if needed by the layout change.
- `src/model/qwen3_6_27b.cpp`
  - Passes `work_` into decode attention and updates local tensor layout use only where needed.
- `tests/kernels/test_gqa_attention.cpp`
  - Updates CPU reference cache layout and existing numerical tests.
- `bench/gqa_attention_bench.cu`
  - Updates cache layout assumptions, decode positions, and byte accounting for the new multi-kernel decode op.
- `docs/bench/m3-long-decode-attention-report.md`
  - Do not modify unless the final reporting task explicitly updates benchmark conclusions.

### Coordination Points

- `include/qus/core/kv_cache.h`, `src/core/kv_cache.cpp`, and `src/kernels/wrapper/gqa_attention.cpp` are shared contract files. Only one implementation worker may edit them at a time.
- `tests/kernels/test_gqa_attention.cpp` and `bench/gqa_attention_bench.cu` are shared verification files. Update them after the relevant API/layout task lands.
- `src/model/qwen3_6_27b.cpp` is a model-card integration point. Edit only after the new L1 decode API compiles.

## Goal Shape

The optimization target is decode `T=1` for Qwen3.6-27B:

- full attention layers: 16
- query heads: 24
- KV heads: 4
- GQA group size: 6 query heads per KV head
- head dimension: 256 BF16 values
- main exposed sequence window: positions around 2048 to 2882
- stress/projection window: 8192 and 32768 in the per-op bench

RTX 5090 target constants used by this plan:

- 170 SMs
- sustained HBM peak model: 1792 GB/s
- L2 size: 96 MB
- register file: 256 KB/SM
- L1/shared memory budget: 128 KB/SM

Useful KV traffic roofline for the 16 full-attention layers is:

| decode position | 16-layer useful KV bytes/token | HBM floor at 1792 GB/s |
| ---: | ---: | ---: |
| 2048 | 134 MB | 75 us |
| 2882 | 189 MB | 105 us |
| 8192 | 537 MB | 300 us |
| 32768 | 2.15 GB | 1.2 ms |

`ncu` DRAM busy is not enough for acceptance because scratch traffic can make total DRAM throughput look high while useful KV throughput remains mediocre. Every performance report must compute:

```text
useful_kv_bytes_per_layer = (pos + 1) * kv_heads * head_dim * sizeof(bf16) * 2  // K + V reads
useful_kv_gbs = useful_kv_bytes_per_layer / measured_layer_decode_seconds
```

For 16-layer round-robin timing, multiply `useful_kv_bytes_per_layer` by 16 and compare against the table above. Report total modeled bytes separately from useful KV bytes.

The new KV physical index is:

```cpp
// cache tensor logical shape: [head_dim=256, padded_context, kv_heads=4]
offset = d + 256 * (position + padded_context * kv_head);
```

The initial decode partial grid shape is:

```cpp
q_heads_per_cta = 6;                       // initial variant; profiler may switch to 3 or 2
q_subgroups = ceil_div(6, q_heads_per_cta);
grid.x = kv_heads * q_subgroups;
grid.y = ceil((host_kv_pos + 1) / tile_n); // token tiles
block.x = 256;                             // one thread per head_dim lane
```

This block-reduce-256 mapping is only the first correctness baseline. The plan must also evaluate this candidate:

```cpp
// Warp-per-query-head mapping candidate.
q_heads_per_cta = 6;              // preferred unless profiling proves register/SFU pressure dominates
block.x = 32 * q_heads_per_cta;   // one warp per local query head
// lane owns d = lane + 32*k for k in [0, 8)
```

The warp-per-query-head mapping recomputes each score with a warp reduction and no block-wide shared-memory barrier in the hot token loop. It may increase L1 traffic for K/V reuse inside the CTA, so the profiler must compare barrier stalls, L1/L2 sector traffic, DRAM throughput, useful KV GB/s, and total decode time before choosing it.

The scratch layout is indexed by global query head and tile, so the same reducer works for
`q_heads_per_cta = 6`, `3`, or `2`:

```cpp
partial_acc[d + 256 * (q_head + 24 * tile)]   // stored as BF16, computed in FP32 locally
partial_m[q_head + 24 * tile]
partial_l[q_head + 24 * tile]
```

At the main exposed window, finer `tile_n` buys more CTAs and shallower per-CTA work, but it directly increases partial scratch traffic. The tuning goal is the largest `tile_n` that saturates useful KV bandwidth at the target positions. `q_heads_per_cta=6` is the preferred default because it minimizes repeated KV reads; `3` and `2` are profile-driven fallback branches only if `6` is demonstrably limited by registers, synchronization, SFU pressure, or occupancy after tile/mapping tuning.

## Non-Goals

- No paged attention.
- No prefix cache.
- No multi-batch decode.
- No compatibility with the old KVCache physical layout.
- No CUDA Graph fixed-grid compatibility in this phase.
- No approximate softmax, approximate `expf`, polynomial activation replacements, or math shortcuts.
- No changes to q5090 weight format.
- No generic model runtime abstraction.

## Implementation Rounds

### Task 1: KVCache Layout Contract

**Execution mode:** Sequential worker, `gpt-5.5`, `xhigh`. This task is a non-committed checkpoint inside the Task 1 + Task Group 2 integration boundary.

**Files:**
- Modify: `include/qus/core/kv_cache.h`
- Modify: `src/core/kv_cache.cpp`
- Modify: `src/runtime/engine.cpp`
- Modify: `tests/test_kv_cache.cpp` only if existing KVCache tests assert the old physical layout.

**Reading list:**
- `include/qus/core/tensor.h`
- `src/core/tensor.cpp`
- `include/qus/core/kv_cache.h`
- `src/core/kv_cache.cpp`
- `src/runtime/engine.cpp`
- `tests/test_kv_cache.cpp`
- `docs/qwen3_6_27b_q5090_final_quant_format_v1.md` lines describing full attention heads.

**Requirements:**
- Allocate each `kv.k[layer]` and `kv.v[layer]` as BF16 tensors with shape `{head_dim, padded_context, num_kv_heads}`.
- Set `padded_context = align_up(max_context, 128)` inside `KVCache`.
- Add a `std::uint32_t padded_context` field to `KVCache`.
- Keep `num_kv_heads`, `head_dim`, `max_context`, `pos`, and `dtype`.
- Replace the old all-KV-head `KVSlot` contract with a per-head contiguous slot contract:

```cpp
struct KVHeadSlot {
    Tensor k; // shape {head_dim}
    Tensor v; // shape {head_dim}
};

KVHeadSlot slot(std::uint32_t layer, std::uint32_t position, std::int32_t kv_head) const;
KVHeadSlot append_slot(std::uint32_t layer, std::int32_t kv_head) const;
```

- Delete the old `KVSlot` type and delete `slot(layer, position)` / `append_slot(layer)` overloads.
- Validate `kv_head` in `[0, num_kv_heads)`.
- The returned slot must point to a contiguous `head_dim` vector for one KV head at one position. Do not return a strided "all KV heads at one position" tensor because `Tensor::bytes()` describes element count, not physical stride span.
- Delete assumptions that cache tensors have shape `{num_kv_heads, head_dim, max_context}`.
- Update `Engine::default_cache_bytes()` to use `padded_context` for KV allocation bytes, not raw `max_ctx`.

**Steps:**

- [ ] Update `KVCache` to store `padded_context`.
- [ ] Change allocation shape to `{head_dim, padded_context, num_kv_heads}`.
- [ ] Update arena size preflight to use the padded shape.
- [ ] Replace `KVSlot` with `KVHeadSlot`.
- [ ] Replace `slot_at()` with a helper that slices one `kv_head` and one `position`, producing contiguous `{head_dim}` K/V tensors.
- [ ] Update `Engine::default_cache_bytes()` to include padded KV context bytes.
- [ ] Update `tests/test_kv_cache.cpp` to fill and verify each KV head separately; do not use a contiguous memset over all heads at one position.
- [ ] Run:

```bash
cmake --build build --target qus_kv_cache_test -j
./build/tests/qus_kv_cache_test
```

Expected: the KVCache test exits 0.

- [ ] Do not commit this task by itself. Continue directly into Task Group 2 in the same worktree so attention wrapper validation, prefill cache fill, and decode kernels are updated before the first bisectable commit.

**Definition of done:**
- KVCache tensors expose the new shape.
- No old shape compatibility branch remains.
- Existing KVCache behavior still rejects invalid positions, invalid heads, and does not advance on `slot()`.
- No helper exposes a strided all-head slot as if it were contiguous.
- The coordinator understands that `qus_gqa_attention_test` and the model path are expected to be broken until Sub-Round 2D lands.

### Task Group 2: Attention L1 Rewrite

**Execution mode:** One sequential worker, `gpt-5.5`, `xhigh`, owns Sub-Rounds 2A through 2D as a single integration boundary.

**Why grouped:** The decode API, prefill cache layout, partial kernel, fused reducer, and model-card call site share one compile boundary. Dispatching them to separate code-editing subagents would create unresolved intermediate states and overlapping ownership of the same kernel and wrapper files.

**Group verification boundary:** Run the full `qus_gqa_attention_test` only after Sub-Round 2D. Review the whole group after the Sub-Round 2D commit.

### Sub-Round 2A: Attention API And Wrapper Scratch Contract

**Execution mode:** Same worker as Task Group 2.

**Files:**
- Modify: `include/qus/kernels/gqa_attention.h`
- Modify: `src/kernels/launcher/gqa_attention.h`
- Modify: `src/kernels/wrapper/gqa_attention.cpp`
- Modify: `src/model/qwen3_6_27b.cpp`

**Reading list:**
- `include/qus/kernels/gqa_attention.h`
- `src/kernels/wrapper/gqa_attention.cpp`
- `src/kernels/launcher/gqa_attention.h`
- `src/model/qwen3_6_27b.cpp` `attn_mix`
- `src/kernels/wrapper/gated_delta_rule.cpp` for `WorkspaceArena` scratch scope pattern.

**Requirements:**
- Change decode signature to:

```cpp
void gqa_attention_decode(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                          float scale, KVCache& kv, int layer, WorkspaceArena& ws,
                          Tensor& out, cudaStream_t stream);
```

- Validate cache shape `{256, kv.padded_context, 4}`.
- Validate `kv.padded_context >= kv.max_context`.
- Use host `kv.pos` to select `tile_n`, `tile_count`, and the initial `q_heads_per_cta`.
- Device `pos` remains passed into kernels and guards actual logical position.
- Pass both `kv.padded_context` and `kv.max_context` through the launcher to the partial kernel.
- Decode wrapper allocates scratch from `WorkspaceArena` and rewinds it before returning.
- No old decode overload remains.
- Model card passes `work_` for decode attention.

**Tile policy:**

```cpp
tile_n = (kv.pos + 1 <= 4096) ? 32 :
         (kv.pos + 1 <= 16384) ? 64 : 128;
tile_count = ceil_div(kv.pos + 1, tile_n);
q_heads_per_cta = 6; // first candidate; Perf Round 1 may change to 3 or 2
```

**Scratch contract:**

```text
partial_acc: BF16 [256, 24, tile_count]   // tile-local accumulators are FP32 before store
partial_m:   FP32 [24, tile_count]
partial_l:   FP32 [24, tile_count]
```

The wrapper must compute scratch bytes from this contract and reject arena overflow through the existing `WorkspaceArena` path. Do not allocate a separate temporary buffer outside `WorkspaceArena`.

**Steps:**

- [ ] Update public header signature.
- [ ] Update launcher header signature.
- [ ] Add wrapper helper functions for checked `tile_n`, `tile_count`, `q_heads_per_cta`, `q_subgroups`, and scratch allocation.
- [ ] Update validation to the new KV shape.
- [ ] Update `Qwen3_6_27B::attn_mix` to pass `work_` in decode only.
- [ ] Continue directly to Sub-Round 2B without committing an inert launcher or compatibility overload.

**Definition of done:**
- There is exactly one decode attention API.
- Decode attention has explicit workspace ownership.
- No project-owned caller uses the old signature.

### Sub-Round 2B: Prefill Cache Fill On New Layout

**Execution mode:** Same worker as Task Group 2.

**Files:**
- Modify: `src/kernels/kernel/gqa_attention_prefill.cuh`
- Modify: `src/kernels/launcher/gqa_attention_prefill.cu` if launch arguments need padded context.
- Modify: `src/kernels/wrapper/gqa_attention.cpp` if prefill validation needs padded context.
- Modify: `tests/kernels/test_gqa_attention.cpp`

**Reading list:**
- `src/kernels/kernel/gqa_attention_prefill.cuh`
- `src/kernels/launcher/gqa_attention_prefill.cu`
- `tests/kernels/test_gqa_attention.cpp`

**Requirements:**
- Replace prefill cache indexing with:

```cpp
offset = d + 256 * (position + padded_context * kv_head);
```

- Pass `padded_context` to prefill kernels that read or write cache.
- Update CPU reference cache layout in `tests/kernels/test_gqa_attention.cpp`.
- Keep prefill attention correctness; prefill speed is not the target.

**Steps:**

- [ ] Update `gqa_prefill_cache_index()` to accept `padded_context`.
- [ ] Update fill kernel and prefill attention kernel call sites.
- [ ] Update tests' `cache_index()` helper to the new physical layout.
- [ ] Continue directly to Sub-Round 2C. Do not commit until Sub-Round 2D creates a complete attention implementation.

**Definition of done:**
- Prefill writes exactly the new KV layout.
- CPU reference and GPU kernels agree on cache physical order.
- No old cache index helper remains in attention tests.

### Sub-Round 2C: Split-KV Decode Partial Kernel

**Execution mode:** Same worker as Task Group 2.

**Files:**
- Modify: `src/kernels/kernel/gqa_attention_decode.cuh`
- Modify: `src/kernels/launcher/gqa_attention_decode.cu`
- Modify: `src/kernels/launcher/gqa_attention.h`

**Reading list:**
- `src/kernels/kernel/gqa_attention_decode.cuh`
- `src/kernels/launcher/gqa_attention_decode.cu`
- `src/kernels/linear/gemv/linear_lowbit_gemv.cuh` for local reduction style only.
- CUDA BF16 intrinsic documentation available in the local toolkit headers.

**Requirements:**
- Delete `gqa_attention_decode_append_kernel`.
- Delete the old 24-block `gqa_attention_decode_kernel`.
- Add a templated partial kernel:

```cpp
template <int TileN, int QHeadsPerCta>
__global__ void gqa_attention_decode_partial_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_new,
    const __nv_bfloat16* v_new,
    const std::int32_t* pos,
    __nv_bfloat16* cache_k,
    __nv_bfloat16* cache_v,
    std::int32_t padded_context,
    std::int32_t max_context,
    float scale,
    __nv_bfloat16* partial_acc,
    float* partial_m,
    float* partial_l);
```

- `blockIdx.x` encodes `(kv_head, q_subgroup)`.
- `blockIdx.y` is token tile.
- Implement the block-reduce-256 mapping first and keep the launch wrapper structured so Perf Round 1 can add the warp-per-query-head mapping without changing the public API or scratch layout.
- `QHeadsPerCta` is initially 6. Compile-time variants 3 and 2 must remain possible, but they are fallback branches after mapping/tile tuning, not the primary occupancy knob.
- `kv_head = blockIdx.x / q_subgroups`.
- `q_subgroup = blockIdx.x % q_subgroups`.
- Each CTA computes local query heads `0..QHeadsPerCta-1`, mapped to global query head `kv_head * 6 + q_subgroup * QHeadsPerCta + local_q`.
- If the mapped global query head is outside that KV head's 6-head group, write no partial for that local lane.
- Each thread owns one `d` lane.
- Read `p = pos[0]` once at kernel start.
- If `p < 0 || p >= max_context`, write neutral partials for this `(q_head, tile)` and return. Do not write to KV cache.
- For a tile with `tile_start > p`, write neutral partials for this `(q_head, tile)` and return. This protects the final tile and host/device position mismatch cases.
- For every token in the tile:
  - If `token > p`, break the loop before any K/V load.
  - If `token >= max_context`, break the loop before any cache write or cache load.
  - If `token == p`, use `k_new` and `v_new` directly for score and AV accumulation. Do not read cache for the current token.
  - Persist `k_new/v_new` into cache for future decode steps only from a single-writer path: `q_subgroup == 0` and the tile contains `p`. This write may happen before or after local computation because no CTA reads cache for `token == p`.
  - If `token < p`, load K once for that `(token, d, kv_head)` from cache.
  - Compute dot contributions for all local q heads owned by this CTA.
  - Reduce local scores across 256 lanes in fp32.
  - Update online softmax state and local fp32 accumulators.
  - If `token < p`, load V once and reuse it for all local accumulators.
- Write `partial_m`, `partial_l`, and BF16 `partial_acc`. Accumulate the tile in FP32 before converting `partial_acc` to BF16.
- Neutral partial means `partial_m = -infinity`, `partial_l = 0`, and `partial_acc = 0`.
- Avoid dynamic shared memory.
- Use `expf`; do not use approximate math.
- Any later vectorized cache load must preserve the invariant that `token == p` never uses the cache load path. Reviewers must reject vectorized changes that read `d+1` or paired K/V values from cache for the current token.

**Reduction primitive requirement:**
- For the block-reduce-256 mapping, implement a local helper that reduces `QHeadsPerCta` fp32 values from 256 lanes to one value per local q head.
- Use warp shuffles plus a small shared array sized for `QHeadsPerCta * 8` warp partials in the block-reduce-256 helper.
- For the warp-per-query-head mapping, each warp reduces one query head score with warp shuffles only; no `__syncthreads()` is allowed inside that mapping's hot token loop.
- Keep the helper in `gqa_attention_decode.cuh`; do not add a generic framework abstraction.

**Steps:**

- [ ] Replace old decode kernels with index helpers and the partial kernel.
- [ ] Add launcher routing for `TileN=32`, `TileN=64`, `TileN=128` and `QHeadsPerCta=6`.
- [ ] Keep the launcher structure able to select `QHeadsPerCta=3` and `QHeadsPerCta=2` without redesigning scratch layout, but leave selection at `6` until a profiler round produces evidence for changing it.
- [ ] Continue directly to Sub-Round 2D. Do not commit until the fused reducer is linked and tested.

**Definition of done:**
- The partial kernel can be profiled by name.
- It has enough CTAs at pos 2048 and pos 2882 to occupy RTX 5090.
- It never reads token positions greater than device `pos[0]`.
- It never writes positions greater than or equal to `max_context`.
- It never reads cache for `token == pos[0]`; the current token uses `k_new/v_new` directly.
- It persists the current token to cache through exactly one CTA per KV head, so `q_heads_per_cta=3/2` cannot introduce duplicate global writes.
- In the `QHeadsPerCta=6` variant, it reads each K/V element once per KV head per token tile, not once per q head.

### Sub-Round 2D: Fused Output Reducer Kernel

**Execution mode:** Same worker as Task Group 2.

**Files:**
- Modify: `src/kernels/kernel/gqa_attention_decode.cuh`
- Modify: `src/kernels/launcher/gqa_attention_decode.cu`

**Reading list:**
- `src/kernels/kernel/gqa_attention_decode.cuh`
- `src/kernels/kernel/argmax.cuh` for simple reduction style.
- `tests/kernels/test_gqa_attention.cpp` CPU softmax reference.

**Requirements:**
- Add one fused output reducer. Do not add a separate stats reducer in the default implementation:

```cpp
template <int DChunk>
__global__ void gqa_attention_decode_reduce_output_kernel(
    const __nv_bfloat16* partial_acc,
    const float* partial_m,
    const float* partial_l,
    std::int32_t tile_count,
    __nv_bfloat16* out);
```

- Output reducer grid: `x = q_head`, `y = ceil(256 / DChunk)`.
- Use `DChunk=32` for initial implementation.
- Output layout remains `out[d + 256 * q_head]`.
- The reducer must support any `tile_count >= 1`; do not assume `tile_count <= blockDim.x`.
- Each output reducer block recomputes `head_m` and `head_l` locally from `partial_m/partial_l`. This intentionally rereads tiny stats for each `d_chunk` to remove one kernel launch and the `head_m/head_l` HBM round-trip.
- The local stats pass must use a strided loop:

```cpp
for (int tile = threadIdx.x; tile < tile_count; tile += blockDim.x) {
    // include partial_m/partial_l for this q_head and tile
}
```

- The output pass must loop across all tiles for each output dimension, converting BF16 `partial_acc` to FP32 before combining. If one CTA per `(q_head, d_chunk)` becomes serially expensive at long context, Perf Round 3 owns the second-level reducer change.
- Neutral partials with `partial_l == 0` must not create NaN output. The final output denominator is expected to be positive for valid `pos`; reviewers must check this invariant.

**Numerical combine formula:**

```cpp
head_m = max_s(partial_m[s])
head_l = sum_s(partial_l[s] * expf(partial_m[s] - head_m))
out[d] = sum_s(partial_acc[s, d] * expf(partial_m[s] - head_m)) / head_l
```

**Steps:**

- [ ] Implement fused output reducer.
- [ ] Launch partial and fused output reducer in order.
- [ ] Ensure scratch tensor indexing matches wrapper allocation.
- [ ] Verify reducer logic with `tile_count=257` by running the long decode correctness mode added in Task 3.
- [ ] Run:

```bash
cmake --build build --target qus_gqa_attention_test -j
./build/tests/qus_gqa_attention_test
```

Expected: exits 0.

- [ ] Commit:

```bash
git add include/qus/core/kv_cache.h src/core/kv_cache.cpp src/runtime/engine.cpp tests/test_kv_cache.cpp \
        src/kernels/kernel/gqa_attention_prefill.cuh src/kernels/launcher/gqa_attention_prefill.cu \
        src/kernels/kernel/gqa_attention_decode.cuh src/kernels/launcher/gqa_attention_decode.cu \
        src/kernels/launcher/gqa_attention.h include/qus/kernels/gqa_attention.h \
        src/kernels/wrapper/gqa_attention.cpp src/model/qwen3_6_27b.cpp tests/kernels/test_gqa_attention.cpp
git commit -m "perf(attn): add split-kv flash decode attention"
```

**Definition of done:**
- Decode attention produces numerically valid BF16 output for existing random, stress, and prefill-to-decode cases.
- The old append kernel no longer exists.
- Decode op uses two kernels: partial and fused output reduce.
- The fused output reducer handles `tile_count=257` through strided loops.

### Task 3: Correctness Coverage And Sanitizer Verification

**Execution mode:** Sequential worker, `gpt-5.5`, `xhigh`.

**Files:**
- Modify: `tests/kernels/test_gqa_attention.cpp`

**Reading list:**
- `tests/kernels/test_gqa_attention.cpp`
- `docs/l1-op-test-standard.md`
- Repository `AGENTS.md` testing policy.

**Requirements:**
- Keep tests behavior/numerical, not source-structure.
- Cover decode positions:
  - `1`
  - `17`
  - `2048`
  - `2882`
  - `8191`
- Add an opt-in long correctness mode that covers decode position `32768`, so reducer behavior with `tile_count=257` is verified without making the default sanitizer path prohibitively slow.
- Keep stress case.
- Keep prefill-to-decode consistency.
- Set `kv.pos` to the intended decode position before calling decode, because the wrapper uses host position for launch sizing.
- Preserve the assertion that decode op does not advance `KVCache.pos`.

**Steps:**

- [ ] Update decode test cases to set `kv.pos = pos`.
- [ ] Add decode case at `pos=2882`.
- [ ] Add decode case at `pos=8191`.
- [ ] Add `--long-decode` CLI handling to `qus_gqa_attention_test`; when present, run exactly one additional public decode correctness case at `pos=32768`.
- [ ] Run:

```bash
cmake --build build --target qus_gqa_attention_test -j
./build/tests/qus_gqa_attention_test
./build/tests/qus_gqa_attention_test --long-decode
```

Expected: both commands exit 0.

- [ ] Run:

```bash
compute-sanitizer --tool memcheck ./build/tests/qus_gqa_attention_test
compute-sanitizer --tool racecheck ./build/tests/qus_gqa_attention_test
```

Expected: both default-shape sanitizer commands report no sanitizer errors. Do not run the `--long-decode` case under sanitizer unless a reviewer specifically requests it; it is for reducer boundary correctness, not memory lifetime coverage.

- [ ] Commit:

```bash
git add tests/kernels/test_gqa_attention.cpp
git commit -m "test(attn): cover split-kv decode shapes"
```

**Definition of done:**
- Numerical tests cover the shape that triggered the long decode collapse.
- Long-mode numerical testing covers `pos=32768` and `tile_count=257`.
- Memcheck and racecheck evidence exists before performance tuning starts.

### Task 4: Bench Harness And Byte Accounting

**Execution mode:** Sequential worker, `gpt-5.5`, `xhigh`.

**Files:**
- Modify: `bench/gqa_attention_bench.cu`
- Modify: `bench/qus_bench_common.h` only if a small indexed bench helper is needed; prefer keeping new logic local to `gqa_attention_bench.cu`.

**Reading list:**
- `bench/gqa_attention_bench.cu`
- `bench/qus_bench_common.h`
- `docs/l1-op-test-standard.md` performance section.

**Requirements:**
- Decode bench positions:
  - `2048`
  - `2882`
  - `8192`
  - `32768`
- Set `kv.pos` before each decode bench call.
- Include `WorkspaceArena` for decode scratch.
- Size the bench `WorkspaceArena` from the worst requested `decode_pos` and selected `tile_n`, including `pos=32768` / `tile_count=257`; `--decode-pos 32768 --profile-once` must not rely on incidental arena slack.
- Print combined op timing for the full partial + fused-reduce sequence.
- Add `--decode-pos <N>` to run exactly one requested decode position.
- Add `--profile-once` to launch one decode attention op for the requested position with no `bench_loop` warmup, no probe loop, and no repeated timing loop. This mode exists for `ncu` kernel capture and must not print median timing.
- Add `--cold-cache` for `--profile-once`; it must touch a large device buffer with a kernel whose name does not match `gqa_attention_decode_*` immediately before the one profiled attention op.
- Add `--round-robin-layers 16` for timing mode; allocate a 16-layer `KVCache` and rotate the layer argument across launches so hot-cache single-layer reuse does not masquerade as HBM bandwidth.
- Default `--decode` may still print hot-cache informational results for all positions, but those results are not acceptance evidence.
- Update bytes estimate to print useful KV bytes, scratch bytes, and total modeled bytes separately:
  - K cache reads: `window * 4 * 256 * 2`
  - V cache reads: `window * 4 * 256 * 2`
  - new K/V writes: `4 * 256 * 2 * 2`
  - q reads: `24 * 256 * 2`
  - output writes: `24 * 256 * 2`
  - BF16 partial acc writes and reads: `2 * tile_count * 24 * 256 * 2`
  - partial m/l writes: `tile_count * 24 * 2 * 4`
  - partial m/l reads by fused reducer: `ceil_div(256, DChunk) * tile_count * 24 * 2 * 4`
- Compute and print `useful_kv_gbs` separately from total modeled GB/s.
- The printed GB/s remains informational; ncu is the gate.

**Steps:**

- [ ] Add workspace allocation to the bench.
- [ ] Derive the workspace allocation from the maximum selected decode position, not from a hard-coded small case.
- [ ] Update decode runner to set `kv.pos`.
- [ ] Add positions listed above.
- [ ] Add `--decode-pos`, `--profile-once`, `--cold-cache`, and `--round-robin-layers 16`.
- [ ] Ensure `--profile-once --decode-pos 2882 --cold-cache` launches exactly one matching partial kernel and one matching fused output reducer.
- [ ] Update bytes accounting.
- [ ] Run:

```bash
cmake --build build --target qus_gqa_attention_bench -j
./build/bench/qus_gqa_attention_bench --decode
./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --round-robin-layers 16
```

Expected: all commands exit 0; default `--decode` prints all four decode positions, `--profile-once` prints a single profile marker without median timing, and round-robin timing prints the requested position.

- [ ] Commit:

```bash
git add bench/gqa_attention_bench.cu
git commit -m "bench(attn): measure split-kv decode shapes"
```

**Definition of done:**
- Bench can drive the exact position targeted by ncu without warmup capture ambiguity.
- The bench includes the report-triggering long decode range and a longer stress range.
- Bench has a cold-cache profile mode and a 16-layer round-robin timing mode.
- Bench workspace sizing is sufficient for `--decode-pos 32768 --profile-once --cold-cache`.
- Bench reports useful KV GB/s separately from total modeled GB/s.

## Performance Tuning Rounds

Performance work is not allowed before Tasks 1-4 pass correctness and sanitizer gates.

### Perf Round 0: Baseline Profile Of New Kernels

**Execution mode:** Profiler subagent, `gpt-5.5`, `xhigh`.

**Files:**
- Create local-only artifacts under `profiles/ncu-attn/`.
- Do not edit source files.

**Reading list:**
- `/home/neroued/.codex/skills/ncu-kernel-profile/SKILL.md`
- `bench/gqa_attention_bench.cu`
- `src/kernels/kernel/gqa_attention_decode.cuh`
- `docs/bench/m3-long-decode-attention-report.md`

**Steps:**

- [ ] Run preflight:

```bash
~/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
```

- [ ] Capture partial kernel basic profile:

```bash
mkdir -p profiles/ncu-attn
ncu --force-overwrite \
    -o profiles/ncu-attn/gqa_flash_partial_baseline_basic \
    --set basic \
    --kernel-name regex:'gqa_attention_decode_partial_kernel' \
    --launch-skip 0 --launch-count 1 \
    ./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
```

- [ ] Capture partial kernel SOL/memory profile:

```bash
ncu --force-overwrite \
    -o profiles/ncu-attn/gqa_flash_partial_baseline_sol_mem \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
    --kernel-name regex:'gqa_attention_decode_partial_kernel' \
    --launch-skip 0 --launch-count 1 \
    ./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
```

- [ ] Capture a long-context cold-cache partial profile for the HBM bandwidth claim:

```bash
ncu --force-overwrite \
    -o profiles/ncu-attn/gqa_flash_partial_baseline_pos32768_cold_sol_mem \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
    --kernel-name regex:'gqa_attention_decode_partial_kernel' \
    --launch-skip 0 --launch-count 1 \
    ./build/bench/qus_gqa_attention_bench --decode --decode-pos 32768 --profile-once --cold-cache
```

- [ ] Run the 16-layer round-robin timing mode:

```bash
./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --round-robin-layers 16
./build/bench/qus_gqa_attention_bench --decode --decode-pos 32768 --round-robin-layers 16
```

- [ ] Capture fused reducer profile:

```bash
ncu --force-overwrite \
    -o profiles/ncu-attn/gqa_flash_reduce_output_fused_baseline_basic \
    --set basic \
    --kernel-name regex:'gqa_attention_decode_reduce_output_kernel' \
    --launch-skip 0 --launch-count 1 \
    ./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
```

- [ ] Export text reports:

```bash
for f in profiles/ncu-attn/gqa_flash_*_baseline*.ncu-rep; do
  ncu --import "$f" --page details > "${f%.ncu-rep}.ncu.txt"
done
```

**Report required from profiler subagent:**
- Confirm each regex matched the intended kernel.
- Report duration, grid size, waves/SM, achieved occupancy, SM throughput, DRAM throughput, L2 throughput, L2 hit rate, barrier/synchronization stalls, scheduler stalls, useful KV GB/s, total modeled GB/s, and top `ncu` warnings.
- Separate hot-cache informational timing, cold-cache `ncu`, and 16-layer round-robin timing. Do not merge these into one bandwidth number.
- Explicitly compare `pos=2882` and `pos=32768`.
- Identify whether the first bottleneck to attack is inner mapping, tile policy, scratch traffic, memory coalescing, or reducer time.

**Definition of done:**
- The team has a metric-backed baseline for partial and fused reducer kernels.
- No source edits are made in this round.

### Perf Round 1: Inner Mapping And Tile Policy

**Execution mode:** Implementer subagent followed by profiler subagent, both `gpt-5.5`, `xhigh`.

**Files:**
- Modify: `src/kernels/wrapper/gqa_attention.cpp`
- Modify: `src/kernels/launcher/gqa_attention_decode.cu`
- Modify: `src/kernels/kernel/gqa_attention_decode.cuh`
- Modify: `bench/gqa_attention_bench.cu` only if the tile policy changes byte accounting.

**Bottleneck target:**
- `gqa_attention_decode_partial_kernel` barrier stalls, scheduler stalls, waves/SM, achieved occupancy, useful KV GB/s, and scratch overhead at positions 2048 and 2882.

**Candidate changes, attempted one at a time:**
- Implement and profile the warp-per-query-head mapping:
  - one warp owns one local query head;
  - each lane owns 8 `d` values (`lane + 32*k`);
  - score reduction uses warp shuffles only;
  - no `__syncthreads()` is allowed in that mapping's token loop.
- Compare warp-per-query-head against the block-reduce-256 mapping using the same `tile_n`, same `q_heads_per_cta=6`, same BF16 `partial_acc`, and the same cold-cache/round-robin bench mode.
- Tune tile policy among:
  - `TileN=16/32/64` for `window <= 4096`
  - `TileN=32/64/128` for `4096 < window <= 16384`
  - `TileN=64/128/256` for longer windows
- Treat `tile_n` as the primary occupancy lever. Smaller tiles increase CTA count but also increase BF16 `partial_acc` scratch traffic, so choose the largest tile that saturates useful KV bandwidth.
- Tune `q_heads_per_cta` among `6`, `3`, and `2` only if profiler evidence shows the selected mapping at `6` is register-limited, SFU-limited, or cannot fill the GPU after tile tuning. Record the extra modeled KV/L1/L2 traffic from dropping below 6 before accepting it.

**Steps:**

- [ ] Pick exactly one mapping or tile-policy change from profiler evidence.
- [ ] Implement only that change.
- [ ] Run:

```bash
cmake --build build --target qus_gqa_attention_test qus_gqa_attention_bench -j
./build/tests/qus_gqa_attention_test
./build/tests/qus_gqa_attention_test --long-decode
./build/bench/qus_gqa_attention_bench --decode
```

- [ ] If the change touches inner mapping or selects `q_heads_per_cta < 6`, run:

```bash
compute-sanitizer --tool memcheck ./build/tests/qus_gqa_attention_test
compute-sanitizer --tool racecheck ./build/tests/qus_gqa_attention_test
```

Expected: no sanitizer errors in the default-shape test.

- [ ] Re-run partial kernel SOL/memory ncu:

```bash
ncu --force-overwrite \
    -o profiles/ncu-attn/gqa_flash_partial_tile_round_sol_mem \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
    --kernel-name regex:'gqa_attention_decode_partial_kernel' \
    --launch-skip 0 --launch-count 1 \
    ./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
```

- [ ] Commit only if metrics improve or expose a necessary tradeoff:

```bash
git add src/kernels/wrapper/gqa_attention.cpp src/kernels/launcher/gqa_attention_decode.cu src/kernels/kernel/gqa_attention_decode.cuh bench/gqa_attention_bench.cu
git commit -m "perf(attn): tune flash decode tile policy"
```

**Definition of done:**
- The plan has profile evidence comparing block-reduce-256 and warp-per-query-head, or a documented blocker explaining why one mapping could not be built.
- For pos around 2048 to 2882, the chosen mapping and tile policy reach the best useful KV GB/s observed so far without hiding excessive scratch traffic behind high total DRAM busy.
- Remaining occupancy or barrier warnings are either fixed or explicitly tied to register/shared-memory/SFU limits.

### Perf Round 2: Memory Coalescing And Vectorization

**Execution mode:** Implementer subagent followed by profiler subagent, both `gpt-5.5`, `xhigh`.

**Files:**
- Modify: `src/kernels/kernel/gqa_attention_decode.cuh`
- Modify: `src/kernels/kernel/gqa_attention_prefill.cuh` only if cache write vectorization requires matching layout helper changes.

**Bottleneck target:**
- Partial kernel MemoryWorkloadAnalysis, especially L1/L2 sectors, L2 throughput, DRAM throughput, and memory pipe busy.

**Candidate changes, attempted one at a time:**
- Use vectorized BF16 pair loads for cached K/V only when `token < p`; `token == p` must keep using `k_new/v_new` directly and must never read cache.
- Unroll the 6-query local loop.
- Keep Q values for 6 local q heads in registers.
- Minimize shared memory synchronization inside the 6-score reduction helper.
- Align scratch writes so `partial_acc` stores are contiguous by `d`.

**Steps:**

- [ ] Choose one memory-access change from the latest `ncu` evidence.
- [ ] Implement only that change.
- [ ] Run:

```bash
cmake --build build --target qus_gqa_attention_test qus_gqa_attention_bench -j
./build/tests/qus_gqa_attention_test
./build/tests/qus_gqa_attention_test --long-decode
./build/bench/qus_gqa_attention_bench --decode
```

- [ ] Run sanitizer after every vectorized-load or mapping-adjacent memory change:

```bash
compute-sanitizer --tool memcheck ./build/tests/qus_gqa_attention_test
compute-sanitizer --tool racecheck ./build/tests/qus_gqa_attention_test
```

Expected: no sanitizer errors in the default-shape test. Racecheck is not relied on as the only proof of append safety; reviewers must inspect that the `token == p` path uses registers and no cache vector load.

- [ ] Re-run:

```bash
ncu --force-overwrite \
    -o profiles/ncu-attn/gqa_flash_partial_memory_round_sol_mem \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
    --kernel-name regex:'gqa_attention_decode_partial_kernel' \
    --launch-skip 0 --launch-count 1 \
    ./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
```

- [ ] Commit if evidence supports the change:

```bash
git add src/kernels/kernel/gqa_attention_decode.cuh src/kernels/kernel/gqa_attention_prefill.cuh
git commit -m "perf(attn): improve flash decode memory access"
```

**Definition of done:**
- Partial kernel no longer has obvious uncoalesced global load/store warnings.
- Any vectorized K/V load is restricted to cached tokens `token < pos[0]`; current-token K/V comes from `k_new/v_new` registers.
- If DRAM throughput is still low, the profiler report identifies the next limiter from metrics, not speculation.

### Perf Round 3: Reducer Bottleneck Removal

**Execution mode:** Implementer subagent followed by profiler subagent, both `gpt-5.5`, `xhigh`.

**Files:**
- Modify: `src/kernels/kernel/gqa_attention_decode.cuh`
- Modify: `src/kernels/launcher/gqa_attention_decode.cu`
- Modify: `src/kernels/wrapper/gqa_attention.cpp` only if scratch layout changes.
- Modify: `bench/gqa_attention_bench.cu` if scratch bytes change.

**Bottleneck target:**
- Combined decode op time after the partial kernel is no longer the dominant problem.

**Candidate changes, attempted one at a time:**
- Tune `DChunk` for output reduce: `16`, `32`, `64`.
- Change partial scratch layout if output reduce has poor coalescing.
- Add a second-level output reduce only if long-context `tile_count` makes one CTA per `(q_head, d_chunk)` serially too slow.
- Add an optional FP32 `partial_acc` comparison only if BF16 `partial_acc` fails numerical acceptance; do not keep both storage modes in production.

**Steps:**

- [ ] Profile fused output reducer basic metrics.
- [ ] Choose one reducer change.
- [ ] Implement only that change.
- [ ] Run:

```bash
cmake --build build --target qus_gqa_attention_test qus_gqa_attention_bench -j
./build/tests/qus_gqa_attention_test
./build/bench/qus_gqa_attention_bench --decode
```

- [ ] Re-profile the reducer targeted by the change:

```bash
ncu --force-overwrite \
    -o profiles/ncu-attn/gqa_flash_reduce_round_basic \
    --set basic \
    --kernel-name regex:'gqa_attention_decode_reduce_output_kernel' \
    --launch-skip 0 --launch-count 1 \
    ./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
```

- [ ] Commit if metrics support the change:

```bash
git add src/kernels/kernel/gqa_attention_decode.cuh src/kernels/launcher/gqa_attention_decode.cu src/kernels/wrapper/gqa_attention.cpp bench/gqa_attention_bench.cu
git commit -m "perf(attn): tune flash decode reducers"
```

**Definition of done:**
- Reducers do not dominate combined decode attention time at pos 2048, 2882, 8192, or 32768.
- If reducers remain significant, the next architectural reducer change is written down with profile evidence.

### Perf Round 4: End-To-End Long Decode Nsys Check

**Execution mode:** Profiler subagent, `gpt-5.5`, `xhigh`.

**Files:**
- Create local-only artifacts under `profiles/nsys/`.
- Optionally create a docs report only in the final reporting task.

**Reading list:**
- `/home/neroued/.codex/skills/nsys-inference-analysis/SKILL.md`
- `docs/bench/m3-long-decode-attention-report.md`
- `src/model/qwen3_6_27b.cpp`
- `bench/gqa_attention_bench.cu`

**Steps:**

- [ ] Run the long decode nsys command from the Profiling Skill Requirements section.
- [ ] Generate summary:

```bash
python3 ~/.codex/skills/nsys-inference-analysis/scripts/nsys_inference_summary.py \
    profiles/nsys/fem_m4096_flash_decode.nsys-rep \
    --out profiles/nsys/fem_m4096_flash_decode.nsys-summary.md
```

- [ ] Generate stats:

```bash
nsys stats --report cuda_gpu_kern_sum profiles/nsys/fem_m4096_flash_decode.nsys-rep \
    > profiles/nsys/fem_m4096_flash_decode.cuda_gpu_kern_sum.txt
nsys stats --report cuda_gpu_mem_time_sum profiles/nsys/fem_m4096_flash_decode.nsys-rep \
    > profiles/nsys/fem_m4096_flash_decode.cuda_gpu_mem_time_sum.txt
nsys stats --report cuda_api_sum profiles/nsys/fem_m4096_flash_decode.nsys-rep \
    > profiles/nsys/fem_m4096_flash_decode.cuda_api_sum.txt
```

**Report required from profiler subagent:**
- Decode kernel summed time.
- New attention kernel grouped time by partial and fused output reducer.
- Attention ms/token by decode step bucket if possible.
- Whether attention still grows linearly with position and whether the slope is now close to modeled memory traffic.
- Top remaining bottleneck after attention.

**Definition of done:**
- E2E trace proves whether the per-op gains move the real FEM long decode workload.
- The report does not add overlapping CUDA API wait time to GPU kernel time.

### Perf Round 5: Repeat Until Stop Condition

**Execution mode:** Coordinator plus one implementer and one profiler per round, all `gpt-5.5`, `xhigh`.

**Stop condition:**
- Continue profile-driven tuning until one of these is true:
  - `gqa_attention_decode_partial_kernel` reaches `dram__throughput.avg.pct_of_peak_sustained_elapsed >= 85%` on a cold-cache or 16-layer round-robin long-context capture and useful KV throughput is also near the roofline target for that position, or
  - 16-layer round-robin timing reaches at least 70% of the 1792 GB/s useful KV roofline at pos 2882 and at least 80% at pos 32768, with scratch bytes reported separately, or
  - detailed `ncu` evidence shows the partial kernel is no longer memory-bandwidth limited and the remaining limiter requires a larger architectural change outside this plan, or
  - end-to-end nsys shows another kernel family dominates decode and attention is no longer the top bottleneck.

**Rules:**
- Each tuning round changes exactly one identified limiter.
- Each round must include before/after `ncu` artifacts.
- Do not count current-kernel speedup as success.
- Do not stop on in-process GB/s alone.
- Do not stop on total modeled GB/s alone; useful KV GB/s must be reported and compared against the roofline table.
- Do not use hot-cache single-layer `dram__throughput` as an acceptance metric; hot-cache runs are diagnostic only.
- For small windows such as 2048/2882, report L2 throughput and DRAM throughput separately. A high L2 hit rate is acceptable only if the cold-cache or round-robin long-context profile also supports the bandwidth claim.

## Review Phase

### Per-Task Review

After every implementation task or explicitly grouped integration boundary:

- Spec reviewer checks:
  - assigned files only;
  - no compatibility shims;
  - task requirements met;
  - no forbidden tests;
  - verification commands were run.
- Code-quality reviewer checks:
  - CUDA indexing correctness;
  - memory coalescing intent matches implementation;
  - scratch lifetime safety;
  - no hidden allocations;
  - no unrelated refactors.

Both reviewers must be `gpt-5.5`, `xhigh`, `priority`.

### Final Independent Reviews

Dispatch these reviewers after Perf Round 4:

1. **Numerical correctness reviewer**
   - Reads `tests/kernels/test_gqa_attention.cpp`, decode kernels, and prefill kernels.
   - Checks the online softmax combine formula and CPU reference agreement.
   - Checks final-tile handling: tokens greater than `pos[0]` must be excluded from max, denominator, and AV accumulation.
   - Checks append/read ordering: `token == pos[0]` uses `k_new/v_new` directly and no vectorized cache path can read current-token cache data.
   - Checks `--long-decode` coverage for `pos=32768` and reducer `tile_count=257`.

2. **CUDA memory/lifetime reviewer**
   - Reads wrapper scratch allocation, launcher arguments, KVCache layout, and sanitizer output.
   - Checks out-of-bounds risks for `pos=0`, `pos=2882`, `pos=8191`, `pos=32768`, `pos=max_context-1`, and device `pos >= max_context`.
   - Checks that no API exposes a strided all-KV-head slot as contiguous memory.
   - Checks that the current-token cache persist path has exactly one writer per KV head for all selected `q_heads_per_cta` values.

3. **Performance evidence reviewer**
   - Reads ncu/nsys artifacts.
   - Checks that claims are metric-backed and not based on speedup over the old kernel.
   - Checks that HBM bandwidth claims use cold-cache or 16-layer round-robin evidence, not hot-cache single-layer reuse.
   - Checks that useful KV GB/s, scratch bytes, and total modeled GB/s are reported separately.
   - Checks that block-reduce-256 and warp-per-query-head were compared, or that a documented build blocker prevented one branch.

## Final Verification Commands

Run from the repository root:

```bash
cmake --build build --target qus_gqa_attention_test qus_gqa_attention_bench qus -j
./build/tests/qus_gqa_attention_test
./build/tests/qus_gqa_attention_test --long-decode
compute-sanitizer --tool memcheck ./build/tests/qus_gqa_attention_test
compute-sanitizer --tool racecheck ./build/tests/qus_gqa_attention_test
./build/bench/qus_gqa_attention_bench --decode
./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --round-robin-layers 16
./build/bench/qus_gqa_attention_bench --decode --decode-pos 32768 --profile-once --cold-cache
~/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
ncu --force-overwrite \
    -o profiles/ncu-attn/gqa_flash_partial_final_sol_mem \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
    --kernel-name regex:'gqa_attention_decode_partial_kernel' \
    --launch-skip 0 --launch-count 1 \
    ./build/bench/qus_gqa_attention_bench --decode --decode-pos 2882 --profile-once --cold-cache
ncu --force-overwrite \
    -o profiles/ncu-attn/gqa_flash_partial_final_pos32768_cold_sol_mem \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats \
    --kernel-name regex:'gqa_attention_decode_partial_kernel' \
    --launch-skip 0 --launch-count 1 \
    ./build/bench/qus_gqa_attention_bench --decode --decode-pos 32768 --profile-once --cold-cache
```

Run the FEM long-decode nsys command from Perf Round 4 before making end-to-end claims.

## Completion Criteria

- Old decode attention kernel and append kernel are gone.
- KVCache physical layout is decode-contiguous.
- KVCache slot helpers expose only per-KV-head contiguous vectors, or no slot helper at all.
- Decode attention uses split-KV partials and a fused reducer kernel.
- Partial kernel excludes tokens greater than device `pos[0]` and guards `pos[0] < max_context`.
- `q_heads_per_cta=6` is retained unless profile evidence justifies `3` or `2`; if changed, the extra KV/L1/L2 traffic and correctness gates are documented.
- Block-reduce-256 and warp-per-query-head mappings have been compared with profile evidence.
- Numerical correctness tests pass for real long-decode positions.
- Reducer correctness covers `tile_count=257`.
- Memcheck and racecheck have fresh clean evidence or reviewed known benign output.
- Per-op ncu artifacts show the current limiting factor and achieved throughput, with cold-cache or 16-layer round-robin evidence for HBM claims.
- E2E nsys artifacts show the new attention contribution in the long decode workload.
- Any remaining bottleneck is documented with exact kernel names and profile evidence.
