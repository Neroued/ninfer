# Active KV in RAM (`--kv-ram-active`)

`--kv-ram-active` (`EngineOptions::kv_ram_active`, Qwen3.8-Flash-Next, one request at a time) keeps
the K/V of the text attention layers in pinned, device-mapped host memory instead of VRAM, for the
conversation being generated as well as for the idle ones. Only the QSA indexer's block keys stay
on the GPU. Before each QSA attention call, the K/V rows the call reads are copied into a device
**staging region**:

* decode/verify: the blocks the indexer selected, through a per-layer block cache, so only blocks
  absent from the cache cross PCIe;
* prefill: the visible pages of the layer.

The attention kernel is unchanged and reads the same bytes in the same order, so every result is
bit-identical to the default placement. The VRAM the K/V no longer uses goes to routed-expert slots
(with `--elastic`, through the elastic KV/expert backing).

This document owns the placement, the staging contract and its cache, the invalidation rules, and
the cost of the mode. The general paged-KV contract is in [Paged KV](paged-kv-cache.md); the Op
contracts are in [`sparse_attention.h`](../../include/ninfer/ops/sparse_attention.h).

## Placement

QSA (Qwen Sparse Attention, [model reference](qwen4-exp-model.md)) stores, per attention layer and
per token: K and V codes and scales (INT8-G64: 2 x 264 B per KV head), the indexer record (272 B:
the raw 128-word key and three positions) and the pooled block key (64 B; one 4-token block key is
spread over its 4 token slots).

* `qsa_append` writes all of them.
* The block-publication step reads the record of the 4 tokens of a completed block.
* `qsa_select` scans every block key of the context at every call.
* `qsa_attention` reads the K/V rows of the selected positions (up to `budget + 3 = 2051` per
  column).

With the option, `plan_decoder_state` (`src/models/qwen3_5/state/decoder_state.cpp`) marks the K/V
code/scale planes and the record planes of the **Main Text** pool `KVPlaneGeometry::host`:

* `plan_device_kv_page_pool` lays those planes out in a separate host layout;
* `PagedKVCache` allocates one `PinnedHostBuffer` (mapped, 2 MiB pages) per rank for that layout,
  after the weights and inside the same page-locked limit: the Engine sizes the planes before the
  load (`active_kv_host_pinned_bytes`) and the materializer keeps at most the limit minus them of
  the host-mapped weights page-locked (`MaterializationPlan::host_pinned_reserved_bytes`); a
  failed page-locking of the planes stops the startup with their size and the remedies;
* the block-key planes stay in the device (elastic) backing;
* the MTP pool is unchanged.

Every logical page, replica, table and transaction of the paged-KV contract is unchanged: a host
plane is a plane whose bytes are in host memory.

* **Raw copies.** `DeviceKVPagePool` copies, zeroes and exchanges host records for host planes with
  stream-ordered kernels (`core/uva_copy.h`). A host-to-host `cudaMemcpyAsync` runs synchronously on
  the CPU and is not ordered after earlier work on the stream.
* **Accounting.** `payload_bytes()` counts device planes only, so `kv_full`, the elastic ranges and
  the slot plan see the smaller pool. Bytes per token:

  | | default, tp 1 | default, tp 2 per GPU | with the option, either |
  |---|---|---|---|
  | Main Text pool | 16.3 KiB | 10.1 KiB | 0.75 KiB |
  | MTP pool | 1.36 KiB | 0.84 KiB | unchanged |

* **Host memory:** the planes of `max_context` tokens ([Cost](#memory)).

## Staging region

`plan_qsa_staging` adds one region per rank to the persistent (main) layout, owned by the
`PagedKVCache` and exposed as `ops::QsaStagingRegion`. Region slot `s` is a 4-token block at rows
`(s % 16) * 4 ..` of region page `s / 16`. The region holds:

* K/V planes over `total_pages` pages and an identity block table, so `qsa_attention` addresses it
  like any pool;
* `cache_blocks` slots per attention layer, then `W x ceil(max_selected / 4)` overflow slots (W =
  draft tokens + 1), shared by the layers, which run in order on one stream;
* per-slot tags (I64) and claims (I64), eight counter words per layer, and optional statistics.

`cache_blocks` defaults to 4,096 (`NINFER_KVRAM_CACHE_BLOCKS`) and is raised until the region holds
one layer of `max_context` (5,472 blocks for 262,144 tokens). The prefill copies one layer into the
start of the region.

Being persistent, the region is part of the fixed reservation ([Cost](#memory)): the startup slot
count drops by about one slot per layer, and the option pays only at long contexts
([Cost](#vram-and-when-it-pays)).

## Staging Ops

`qsa(...)` in [`qwen4_forward/qsa.h`](../../src/models/qwen3_5/execution/qwen4_forward/qsa.h) composes the
Ops for a pool that has a region; the Program (one lane) calls them per layer.

* **Prefill: `qsa_stage_visible_pages`.** The call copies the visible pages of the lane's row
  (every K/V plane and head) to region pages `0 ..` and attends with the unchanged selection
  through the identity table. It also sets every tag to -1, because the copied pages overwrite
  cache slots.
* **Decode and verify: `qsa_stage_selected_blocks`.** One warp per 4-entry group of a column's
  selection (one block, by `qsa_select`'s contract):
  - **Cacheable blocks.** A block wholly before the window's first position `p0` is cacheable. It
    maps to slot `block % cache_blocks`. A 64-bit claim `(epoch << 32 | key)` taken with
    `atomicCAS` lets the columns of an MTP window share the block. The winner compares the slot's
    tag `(generation << 32 | row << 20 | block)` and copies the block only when it differs.
  - **Overflow.** A cacheable block whose slot another block claimed in the same call goes to the
    overflow area, and so does every uncacheable block (the growing tail, the positions of the
    window).
  - **Rewrite.** The selection is rewritten to `slot * 4 + (position & 3)`. The attention reads the
    region with the same counts.
  - **End of call.** The last CTA advances the layer's epoch, resets the overflow cursor, records
    `p0`, and starts a new generation when `p0` went back, which drops every cached block of the
    layer.
  - **CUDA Graphs.** No host metadata, a fixed launch topology: the call is captured in the decode
    graphs like the other QSA Ops.
* **`qsa_staging_invalidate`** sets every tag to -1.
* **`qsa_count_selected_blocks`** (diagnostics) counts the distinct blocks a prefill call's
  selections touch.

Speculative rows are never cached. Positions `>= p0` are uncacheable, so a rejected token
re-appended by the next window is always read fresh.

## Invalidation

A cached block is trusted until its committed bytes change or its table row maps another sequence.
Both go through the page pool, which calls its write listener in stream order. `PagedKVCache`
installs `qsa_staging_invalidate` as that listener, with the region's device current.

* **Page writes.** The listener runs on every `zero_run`, `copy_run` (copy-on-write, relocation)
  and host or disk restore (`copy_from_host*`).
* **Table publications.** `KVExecutionTablePool::publish*` calls it on a row's first publication
  after `acquire`, and on any publication that maps an already published logical page to another
  physical page. A mirror pool (rank 1) invalidates its own region on its own stream.
* **Prefill.** Every prefill call clears the tags.
* **Window start going back.** A new generation inside the call covers a truncation that is
  followed by decode without a prefill.

## Tensor parallelism, MTP, elastic VRAM

* **tp 2.** Each rank's pool has its own host buffer (its KV head) and its own region. Both ranks
  compute the same selection, so they make the same cache decisions independently.
* **MTP.** Verify windows use the cache, deduplicated across their columns. The MTP layer's pool is
  in VRAM and is untouched.
* **Elastic VRAM.** The elastic arena backs only the block-key planes (12 x 4 KiB per page), so a KV
  grow step takes about 14x less from the expert columns. As in the default placement, the split
  between base slots and elastic columns leaves enough columns for the KV to grow to `max_context`.
* **Idle conversations.** They still move to the host KV tier (`--host-kv-mib`). For host planes
  that move is a host-to-host copy of bytes already in RAM: correct but redundant.

## Cost

This section is the authority for the costs of `--kv-ram-active`; the user guides link here. The
sizes follow from the layouts; the rates and timings were measured on 2x RTX 5070 Ti 16 GB (each
on PCIe gen5 x8, no P2P), a Ryzen 9 9900X with 96 GB DDR5, CUDA 13.1, with `--profile flash-next`
(tp 2) and `flash-next-1gpu` (tp 1), and scale with the PCIe link and the host memory.

### Memory

* **Pinned host RAM:** the planes of the whole pool, 3.9 GiB (tp 1) or 2.3 GiB per rank (tp 2) at
  262,144 tokens; they count in the page-locked limit (`--host-pinned-max-gib`).
* **Staging region (VRAM):** one layer of `max_context` plus the block cache, 277 MiB (tp 1) or
  138 MiB per rank (tp 2) at 262,144 tokens; each block-cache block costs 2,112 B per KV head and
  layer (`NINFER_KVRAM_CACHE_BLOCKS`).

### Scattered host-to-device reads

On PCIe gen5 x8 a zero-copy gather kernel reading 512 B-4 KiB chunks at random offsets of pinned
memory reaches about 27 GB/s, close to a contiguous copy (29 GB/s), while `cudaMemcpyAsync` per
chunk reaches 0.2-2.8 GB/s and `cudaMemcpyBatchAsync` 1.9-13 GB/s. Kernels therefore do the
transfer, and the cost is the number of bytes that cross PCIe. `ninfer_qsa_staging_bench`
([bench/README.md](../../bench/README.md)) measures the two Ops.

### Per call

Per decode step and rank, the cost is `(new blocks) x 2,112 B x heads x 12 layers` at about
26.5 GB/s, plus the resolve launches: about 0.5 ms (tp 2) or 1 ms (tp 1) for a cold T=1 step, and
four times that for a cold MTP window of 4 columns that share no block. A prefill call stages
`n_visible x 528 B x heads` per layer, for example 15 / 31 / 62 ms per 2,048-token chunk at 64k /
128k / 262k at tp 2.

### VRAM and when it pays

The VRAM freed per GPU at tp 2 is `9,600 B x N - 145 MB` for a context of N tokens (0.48 GB at 64k,
1.11 GB at 128k, 2.37 GB at 262k; at tp 1 0.74, 1.75 and 3.8 GB), about +10 (tp 2) / +17 (tp 1)
expert slots per layer at 128k. Against that, the staging region costs about one expert slot per
layer from the first token. Measured on 2x RTX 5070 Ti with `--profile flash-next` and
`flash-next-1gpu`, the mode loses a few per cent of decode rate at 16k, is about neutral at 64k
and gains from 128k, where the expert slots stay near their short-context count instead of falling
with the KV; the gain is larger with MTP than without (each token pays the cache resolve of every
layer). The placement is fixed for the Engine's lifetime: enable it for workloads whose contexts
are usually above about 100k tokens, and leave it off for chat-length contexts.

## Limits

* **One lane.** Generation runs one request at a time (`max_concurrency 1`, already required by
  `--elastic`), with one lane in prefill and decode.
* **Prefill staging scales with the context.** It costs one layer of `max_context` in VRAM and
  `n_visible` bytes of PCIe per layer and chunk. Beyond 262k tokens it must stage only the union
  of the chunk's selections (`NINFER_KVRAM_STATS` reports that union), or the chunk must be split.
  Across 2,048-column chunks the selections touch 73-95 % of the visible blocks, so staging the
  union saves little at the supported context.
* **What still scales with the context in VRAM.** Block keys (768 B per token per GPU) and the
  1-layer MTP pool. The host buffer is sized for `max_context`.
* **Host-arena parking of host planes copies RAM to RAM.**

## Tests

* `tests/test_kv_cache.cpp` (`exercise_host_planes`):
  - host-plane accounting and binding;
  - zero/copy following a queued kernel write on the same stream;
  - exact host-record round trip;
  - which writes and publications reach the write listener.
* `tests/ops/test_qsa_staging.cpp`:
  - the exact oracle: every rewritten entry addresses region rows byte-identical to its source
    rows, entries past a count are untouched, staged pages equal their source pages, the page after
    them is not written;
  - attention over the region equals attention over device pages bit for bit, for all five KV
    profiles and one/two KV heads;
  - decode and verify windows over a deep context that hit, miss, share, collide and overflow;
  - re-appended uncommitted rows, a truncation, invalidation, and CUDA Graph replay.
* Engine level: the mode is bit-identical by construction, so the check is an exact comparison of
  the two placements. `ninfer_qwen4_tf_driver --kv-ram-active` (teacher-forced dumps and
  generation, tp 1 and 2, with and without MTP) and the greedy probe of
  `bench/capy_flash/long_context.py greedy` against two `ninfer-serve` instances, one with the
  option and one without, with frozen expert residency (`NINFER_QWEN4_VALIDATION=1
  NINFER_QWEN4_EXPERT_FREEZE=all` on both), must produce the same bytes. The `ninfer` CLI has no
  `--kv-ram-active`.
