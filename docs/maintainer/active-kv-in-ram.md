# Active KV in RAM (`--kv-ram-active`)

`--kv-ram-active` (`EngineOptions::kv_ram_active`, Qwen3.8-Flash-Next, any `--max-concurrency`)
keeps the K/V of the text attention layers in page-locked, device-mapped host memory instead of
VRAM, for the conversations being generated as well as for the idle ones. The RAM follows the pages
in use: it grows 8,192 tokens at a time as conversations grow and is given back when they end. Only
the QSA indexer's block keys stay on the GPU. Before each QSA attention call, the K/V rows the call
reads are copied into a device **staging region**:

* decode/verify: the blocks the indexer selected for every lane of the batch, through a per-layer
  block cache the lanes share, so only blocks absent from the cache cross PCIe;
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
code/scale planes and the record planes of the **Main Text** pool and of the **MTP** pool
`KVPlaneGeometry::host`:

* `plan_device_kv_page_pool` lays those planes out in a separate host layout;
* each pool's `PagedKVCache` reserves that layout as one `ElasticArena` of `ElasticMemory::Host` per rank (CUDA
  virtual memory: one address range, physical granules of 2 MiB of page-locked memory of the GPU's
  host NUMA node, mapped for the GPU and the CPU at the same address), with one elastic range per
  Host plane;
* `DeviceKVPagePool::set_host_backing` backs the first `kActiveKvHostStepPages` (128 pages, 8,192
  tokens) of every Host plane; before a page past the backed ones is handed out the pool backs the
  next multiple of the step on its rank and on the mirror (a failure fails the request that asked
  for the page, with the pool unchanged); `trim_host_backing` releases what lies above the
  allocated extent, rounded up to the step, at every prompt and after the idle-KV compaction;
* the page-locked limit still makes room for the whole layout: the Engine sizes the planes before
  the load (`active_kv_host_pinned_bytes`: `kv_capacity` pages, or `max_concurrency x max_context`
  with an automatic capacity) and the materializer keeps at most the limit minus them of the
  host-mapped weights page-locked (`MaterializationPlan::host_pinned_reserved_bytes`), so the KV can
  always grow to `kv_capacity`; the RAM it does not use stays with the page cache (the pageable
  expert banks, the n-gram table);
* the block-key planes stay in the device (elastic) backing;
* the MTP pool has no region of its own: it stages through the text pool's region, planned with
  one more cache layer per MTP layer after the text layers (`PagedKVCache::share_staging`,
  `staging_layer`), whose cached blocks its writes invalidate too.

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
  | MTP pool | 1.36 KiB | 0.84 KiB | 0.06 KiB |

* **Host memory:** the planes of the pages in use, at most `kv_capacity` tokens
  ([Cost](#memory)).

## Staging region

`plan_qsa_staging` adds one region per rank to the persistent (main) layout, owned by the
`PagedKVCache` and exposed as `ops::QsaStagingRegion`. Region slot `s` is a 4-token block at rows
`(s % 16) * 4 ..` of region page `s / 16`. The region holds:

* K/V planes over `total_pages` pages and an identity block table, so `qsa_attention` addresses it
  like any pool; the identity row has one entry per lane (`max_concurrency`);
* `cache_blocks` slots per attention layer, shared by the lanes, then
  `W x max_concurrency x ceil(max_selected / 4)` overflow slots (W = draft tokens + 1), shared by
  the layers, which run in order on one stream;
* per-slot tags (I64) and claims (I64), eight counter words per layer, the first position of each
  table row's last call per layer, and optional statistics.

`cache_blocks` defaults to 4,096 (`NINFER_KVRAM_CACHE_BLOCKS`) and is raised until the region holds
one layer of `max_context`, at most 262,144 tokens (`NINFER_KVRAM_REGION_TOKENS`; 5,472 blocks for
262,144 tokens). The prefill copies one layer into the start of the region, or, when the visible
context is longer than the region, the blocks its selections name.

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
* **Decode and verify: `qsa_stage_selected_blocks`.** One call per layer covers the `W x B`
  columns of the batch's B lanes. One warp per 4-entry group of a column's selection (one block, by
  `qsa_select`'s contract):
  - **Cacheable blocks.** A block wholly before its lane's window (first position `p0` of the
    lane) is cacheable. It maps to slot `(block + row x cache_blocks / 8) % cache_blocks` (`row` =
    the lane's table row; the lanes share the layer's slots). A 64-bit claim
    `(epoch << 32 | key)` taken with `atomicCAS` lets the columns of an MTP window share the
    block. The winner compares the slot's tag `(generation << 32 | row << 20 | block)` and copies
    the block only when it differs; the row in the key keeps the lanes' blocks apart.
  - **Overflow.** A cacheable block whose slot another block claimed in the same call goes to the
    overflow area, and so does every uncacheable block (the growing tail, the positions of the
    window).
  - **Rewrite.** The selection is rewritten to `slot * 4 + (position & 3)`. The attention reads the
    region with the same counts.
  - **End of call.** The last CTA advances the layer's epoch, resets the overflow cursor, records
    each lane's `p0` for its table row, and starts a new generation when some lane's `p0` went
    back below its row's previous one, which drops every cached block of the layer.
  - **CUDA Graphs.** No host metadata, a fixed launch topology: the call is captured in the decode
    graphs like the other QSA Ops.
* **Short chunks and prefill beyond the region: `qsa_stage_selected_union`.** When the visible
  pages of a prompt chunk do not fit the region, or when the chunk's columns cannot select as many
  blocks as are visible (`W x 513 < visible blocks`: the new turn of a resumed conversation, whose
  selections touch a few per cent of a long context), the chunk's columns go by ranges of
  `region blocks / 513` columns
  (127 at 262,144 tokens with 2 lanes and MTP), so that the worst case (no two columns sharing a
  block) still fits. For each range a bitmap marks the blocks its selections name, a one-CTA scan
  gives every marked block its slot (marked blocks before it, ascending), one warp per bitmap word
  copies the word's blocks, and the selection is rewritten onto the slots as in decode; the
  attention of the range reads the region. Every tag becomes -1.
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
  followed by decode without a prefill (tracked per table row).
* **Concurrency.** A new request binding a row, a page copy or a prefill of one lane clears the
  cache of every lane: the next step of the others fetches their selected blocks again.

## Tensor parallelism, MTP, elastic VRAM

* **tp 2.** Each rank's pool has its own host buffer (its KV head) and its own region. Both ranks
  compute the same selection, so they make the same cache decisions independently.
* **MTP.** Verify windows use the cache, deduplicated across their columns. The MTP layer reads its
  pool through the region's last cache layer; its draft steps rewrite positions the previous round
  had written, so its first position goes back at most rounds and its cached blocks are dropped
  then (a new generation of that layer only).
* **Elastic VRAM.** The elastic arena backs only the block-key planes (12 x 4 KiB per page), so a KV
  grow step takes about 14x less from the expert columns. As in the default placement, the split
  between base slots and elastic columns leaves enough columns for the KV to grow to the whole
  pool (`kv_capacity`), with any number of requests.
* **Idle conversations.** They still move to the host KV tier (`--host-kv-mib`). For host planes
  that move is a host-to-host copy of bytes already in RAM: correct but redundant.

## Cost

This section is the authority for the costs of `--kv-ram-active`; the user guides link here. The
sizes follow from the layouts; the rates and timings were measured on 2x RTX 5070 Ti 16 GB (each
on PCIe gen5 x8, no P2P), a Ryzen 9 9900X with 96 GB DDR5, CUDA 13.1, with `--profile flash-next`
(tp 2) and `flash-next-1gpu` (tp 1), and scale with the PCIe link and the host memory.

### Memory

* **Pinned host RAM:** the planes of the pages in use, rounded up to 8,192 tokens per pool: with
  int8 KV and MTP about 16.9 KiB per token (tp 1) or 10.2 KiB per token and rank (tp 2), 4.2 GiB
  (tp 1) or 2.5 GiB per rank (tp 2) for 262,144 tokens, plus up to one 2 MiB granule per Host
  plane (the scale planes are smaller than a granule: about 52 MiB per rank for the 26 of
  Flash-Next). The whole pool (`kv_capacity`) counts in the page-locked limit
  (`--host-pinned-max-gib`) from the start.
* **Staging region (VRAM):** one layer of `max_context` plus the block cache, 277 MiB (tp 1) or
  138 MiB per rank (tp 2) at 262,144 tokens; each block-cache block costs 2,112 B per KV head and
  layer (`NINFER_KVRAM_CACHE_BLOCKS`), and the overflow grows by `W x 513` blocks per lane (4.3 MiB
  per KV head with MTP 3).

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

### Several requests

Measured with `--tp 2`, MTP 3, int8 KV (unless noted), `--max-concurrency 4`, the lanes resuming
cached conversations (`bench/capy_flash/long_context.py concurrent --warm`): tokens per second of
every lane together while all of them decode, one run per cell.

| context per lane | placement | 1 lane | 2 lanes | 4 lanes |
|---|---|---|---|---|
| 32k | KV in VRAM | 132.7 | 186.7 | 141.3 |
| 32k | `--kv-ram-active` | 141.8 | 177.3 | 153.8 |
| 100k | `--kv-ram-active` | 134.3 | 177.5 | - |
| 100k | `--kv-ram-active --elastic` | 134.6 | 165.2 | 157.2 |
| 100k | `--kv-ram-active`, nvfp4 KV | 125.8 | 174.2 | - |

(-: the four warmed conversations did not all stay cached, so their prompts ran again.) Three coding agents replayed at once (the
three sessions of `bench/capy_flash/sessions.json` in parallel, 53 turns, `--max-concurrency 3
--draft-tokens 1`, shipped profile otherwise): 170.7 s in all with the option and 274.3 s with the
KV in VRAM (elastic: the three conversations' KV takes expert columns, 20 base + 66 elastic slots
against 81 + 5), first token after 1.77 s against 2.93 s (median). The lanes
share each layer's 4,096 cache blocks: at 4 x 32k, 43 % of the selected blocks are fetched
(collisions between the lanes), 24 % with 8,192 blocks and 16 % with 16,384, but the larger
region takes VRAM from the expert slots and the decode is not faster (8,192: 137.8 / 169.0 / 145.4
against 142.1 / 178.4 / 152.4 with 4,096, same run), so the default stays 4,096. The
shipped profile (one request, `--elastic`) at 100k: 141.1-141.6 tokens/s without the option (four
runs) and 139.0-140.3 with it (four of five runs; one run gave 129.0 with the same tokens). Four
lanes of MTP 3 make 16 columns, above the 8 of the routed experts' fast path, so four lanes gain
little over two. With `--draft-tokens 1` (windows of 2 columns) the 32k lanes give 127.7 / 191.1 /
184.0: below MTP 3 for one request, above it from two (MTP 3 in the same run: 141.3 / 178.2 /
148.9). The RAM bounds the lanes: with Flash-Next loaded (expert banks page-locked) about
10 GB of 96 GB stay available, which four 100k conversations (about 8 GB of planes at tp 2 with
the MTP pool) nearly exhaust; nvfp4 KV takes about 30 % less RAM (2.6 instead of 3.7 GiB per rank
for 409,600 text tokens).

## Limits

* **Concurrency.** Decode and verify batch every lane; a prompt is prefilled one lane at a time.
  Without `--elastic` the block keys of `kv_capacity` tokens are fixed in VRAM (768 + 64 B per
  token and GPU at tp 2, text and MTP pools).
* **The union costs more than whole pages for full chunks.** Measured at 100k tokens (Flash-Next,
  `--tp 2`, perf-max): a 100k prompt prefills at 4.33k tokens/s staging whole pages and 2.56k with
  a region of about 61k tokens (`NINFER_KVRAM_CACHE_BLOCKS=1024 NINFER_KVRAM_REGION_TOKENS=32768`:
  29-column ranges), so the region keeps whole layers up to its cap. For the new turn of a resumed
  conversation the union is the cheaper path: resuming a cached 100k conversation (shipped
  profile, one prompt token to recompute) takes 58.6 ms to the first token with the option and
  57.2 ms without (65.9 vs 58.7 ms when that turn staged every visible page).
* **Prefill staging up to 262,144 tokens scales with the context.** It costs one layer of
  `max_context` in VRAM (at most 262,144 tokens) and `n_visible` bytes of PCIe per layer and chunk.
  Across 2,048-column chunks the selections touch 73-95 % of the visible blocks, so below the cap
  the whole pages are cheaper than the union. Beyond it the chunk stages the union of the
  selections of each column range: the region no longer grows with the context, and the bytes per
  range are the distinct blocks its columns select (`NINFER_KVRAM_STATS` reports both).
* **What still scales with the context in VRAM.** Block keys (832 B per token per GPU at tp 2,
  text and MTP pools).
* **Host-arena parking of host planes copies RAM to RAM.**

## Tests

* `tests/test_elastic_arena.cu`: Host arenas (zero on backing, one address for GPU and CPU, RAM
  released and re-backed) and `set_host_backing` (growth in steps on the pool and its mirror before a
  page is handed out, bytes kept, `trim_host_backing` down to the allocated extent).
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
  - batched calls of three lanes at different depths whose rows map the same blocks to different
    pages, a call of a subset of the lanes, and one lane's truncation;
  - a prompt chunk staged as the union of its selections, whole and by column ranges, beyond the
    region;
  - re-appended uncommitted rows, a truncation, invalidation, and CUDA Graph replay.
* Engine level: the mode is bit-identical by construction, so the check is an exact comparison of
  the two placements. `ninfer_qwen4_tf_driver --kv-ram-active` (teacher-forced dumps and
  generation, tp 1 and 2, with and without MTP; `--concurrency N` submits several prompts at once) and the greedy probe of
  `bench/capy_flash/long_context.py greedy` against two `ninfer-serve` instances, one with the
  option and one without, with frozen expert residency (`NINFER_QWEN4_VALIDATION=1
  NINFER_QWEN4_EXPERT_FREEZE=all` on both), must produce the same bytes. The `ninfer` CLI has no
  `--kv-ram-active`.
