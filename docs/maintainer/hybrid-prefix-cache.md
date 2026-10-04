# Hybrid prefix cache: design and decisions

The hybrid prefix cache (HPC) is NInfer's prefix cache. It reuses prompt prefixes through
content-addressed 64-token KV blocks and sparse recurrent-state snapshots, kept on the Device and in
one pinned Host tier. It replaced the checkpoint-catalog cache (private continuations, shared
prefixes, the ResourceManager pressure planner and the Host State/KV checkpoint pools); §1 explains
why and §14 lists what was removed.

This document is the authority for the prefix cache. §0–§11 describe the design as implemented,
marking the parts of the original design that were not built; §12–§15 cover optional features,
tests, configuration and risks; the [decision record](#16-decision-record) (§16) lists the decisions
taken while building and operating it, with the measurements behind them and the alternatives that
were tried and reverted.

Scope: prefix reuse, retention and eviction for `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` on
one RTX 5090 (`sm_120a`), with `max_concurrency` 1..8, every KV profile (BF16, INT8-G64,
FP8-E4M3FN-row256, NVFP4-G16, K8V4), and every speculative backend (none, MTP, DFlash, DFlash2).

The cache reuses the active-execution machinery unchanged: `LogicalKVPageStore`,
`KVAddressSpaceStore`, execution rows, the active `StateStore` images and the prefill, decode and
speculative paths. Tree nodes hold non-writer references to logical pages, and an active address
space maps them through the existing page-prefix fork. Scheduling is unchanged: FIFO admission
with the blocked-head backfill proof (§6.3). `--no-prefix-reuse` disables the cache.

---

## Status

The cache configures itself: the only capacity a deployment chooses is `--host-cache-mib`
(default 8192, 0 = Device only); every other value is derived from the rest of the configuration
(§14.2).

| area | state |
|---|---|
| §5 index, §9 eviction (device LRU, host superseded-first then GDSF, dead KV), §7.1 tap planner | implemented; host-only unit tests (`ninfer_prefix_cache_index_test`) |
| §5.4 automatic Device sizing | implemented: the Main pool is not clamped to `C·L`, and `ninfer-serve` defaults `--kv-capacity` to `auto`, so free VRAM becomes Device block cache |
| §5.4 unified Host slab pool | implemented: KV blocks, snapshot images (split over slabs) and snapshot tails share one pinned pool; GDSF and the dead-KV sweep decide the split at run time |
| §6 admission as the Engine's materialization transaction | implemented: staging reserves every Device page and the state slot, Host restores run on a dedicated restore stream, and activation forks the lane at once; its Device work queues behind the copies it reads, layer by layer (§6.4, §6.5, §12.1) |
| §6.3 persistent backfill proof | implemented: the Engine's FIFO backfill protection is unchanged, and the Program proves a borrower from page entitlements (§6.3) |
| §6.6 prefetching the blocked head | implemented: Host-only path blocks are copied into spare Device cache while the FIFO head waits |
| §7.1 taps | implemented: exact taps (explicit, generation opener, structural) split the chunk; flexible taps (prompt tail, ladder) are realized at chunk boundaries at no extra forward pass |
| §7.4 tap publication | implemented: deferred until the anchoring blocks commit (a frontier inside a block, or an MTP backend one token behind); exact-frontier tails are copied into cache-owned pages |
| §7.6 block publication, write-through | implemented: blocks join the tree at every commit; a lane's blocks are written to Host in one batch when it releases them |
| §7.7 endpoint snapshots at finish and consistent abort | implemented |
| §6.2 cost model | implemented: the Engine's context-cost coefficients (prefill and Host-to-Device transfer) rank sources and value snapshots |
| §5.5 persistence across restarts | implemented (opt-in `--prefix-cache-file`) |
| §12.2 in-flight prefix coalescing | not included: it needs a request quoted while another prefills, and the Engine admits only while no request is in staged prefill (§16.2) |
| reuse-loss attribution | implemented: the request log's `materialization` record carries `cached_prefix_tokens` (longest cached block prefix, reusable or not), and `restored_host_bytes` |
| §7.2–§7.3 zero-split GDN state tap and phase alignment | not implemented: an exact tap costs one prefill split (about 15 ms per turn on 27B); flexible taps avoid it, and requests resuming from an endpoint skip the opener tap (§7.1). The implemented tap already copies the whole StateImage (conv, recurrent, hidden, DFlash local ring); §7.8 records what the split does to greedy output |
| §11.2 KV transfer Op | copy-engine path only (`cudaMemcpy2DAsync` runs over consecutive pages and slabs) |
| §12 optional features other than 12.1, §13.2 Op qualification | not implemented |
| §13.3 real-artifact scenarios | `ninfer_qwen3_5_hybrid_prefix_real_test`: Host vs Device restore exactness (with and without MTP), tap resume vs cold prefill exactness (without speculation and with DFlash2), generation-opener and system-block reuse, Device-only mode, protocol cache hints, Vision, persistence across a restart; `NINFER_HYBRID_KV_DTYPE` runs them for every KV storage (bf16, int8, fp8, nvfp4, k8v4) |

## 0. Summary

| | checkpoint-catalog cache (replaced) | hybrid prefix cache |
|---|---|---|
| Unit of reuse | owner/checkpoint (private continuation, shared prefix, 5 checkpoint kinds) | content-addressed 64-token KV block + sparse state snapshot |
| Identity | per-owner token ledgers, session index, shortlist keys, markers | exact token path in one radix tree (hash only for lookup) |
| KV sharing across sessions | only through an optional shared-prefix publication that must beat a private baseline | unconditional: identical blocks are one physical page |
| State checkpoint capture | prefill chunk split at the frontier + copy | a copy of the committed state at a chunk boundary; only exact taps (client breakpoints, generation opener, structural boundaries) split a chunk (§7.1) |
| Admission | bounded heuristic search over complete targets; up to 250 ms planning allowance per admission boundary | O(prompt blocks) lookup + arithmetic capacity check, no search |
| Pressure | monotone degradation graph, joint post-state projection, ordered stage peaks, one global transaction | independent eviction per tier: device LRU and host GDSF (GreedyDual-Size-Frequency) |
| Host tier | variable-extent arena with allocator geometry and a separate StateImage slot count | one pinned slab pool; KV blocks and snapshots share it with no fixed split between them |
| Device ↔ host | demote on pressure (copy before release) | asynchronous write-through when a lane releases its blocks; device eviction never copies |
| Code | ≈25 k lines in `runtime/engine/context_cache`, `program/{planning,transactions,storage}`, host arena, resource contracts | ≈6.1 k lines in `runtime/prefix_cache`, `program/prefix` and the Engine's `HybridResourceManager`, plus ≈1.8 k lines of integration |

What the design changes for users:

- **More reuse.** Every committed full block of every request becomes reusable, including
  cancelled and partially prefilled requests. Snapshots are deduplicated by content, so one
  snapshot serves every session that shares its prefix. Tap density is limited only by host memory,
  not by per-continuation anchor counts or catalog slots.
- **Lower TTFT.** No planning search. Host restores never hold an admission: they overlap the
  restored request's first prefill pass layer by layer while other lanes keep decoding (§6.5).
  Only exact taps split a prefill chunk, and the template's rewrite frontiers no longer do.
- **Throughput unchanged.** Kernels, page layout, block tables and CUDA Graph keys are unchanged.
  A flexible tap costs one Device copy of the state image (about 0.25 ms) at a chunk boundary.
  Cold prefill and decode match the replaced cache at the same pool size (§16.1).
- **Better memory efficiency.** Shared prefixes are stored once. KV that no snapshot can reach
  ("dead KV") is reclaimed first. The host pool has no fragmentation and no split between KV and
  state.

---

## 1. Why the checkpoint-catalog cache was replaced

Evidence is taken from the replaced sources and their reference,
`docs/maintainer/resource-scheduling-and-context-cache.md` (removed with them).

1. **Ownership instead of content.** Reuse depends on which *owner* holds a checkpoint: private
   continuation, shared prefix, SessionIndex binding, or one of five checkpoint kinds
   (`SessionEndpoint`, `TurnClosure`, `ResponseReplay`, `LongAnchor`, `SharedStablePrefix`). It is
   also bounded by catalog slots (`max_private_continuations`, `max_shared_prefixes`,
   `max_long_anchors_per_continuation`). Two sessions with the same 30 k-token system+tools prefix
   share KV only if a shared prefix was published. Publication must be *strictly better than the
   private-only baseline* (resource-scheduling §7.2, §8.3).
2. **Planning cost on the critical path.** Materialization runs a bounded heuristic search over
   complete pressure targets. The Program projects joint post-states, stage peaks and Host
   allocator geometry for each target (§7.3–§8.7). Every admission boundary has a 250 ms
   planning allowance. That time sits directly in TTFT for the head request and
   stalls the worker for everyone else.
3. **Split chunks.** A checkpoint at an arbitrary frontier splits a prefill chunk (`text.cpp`
   `prefill_split_frontier_`), producing a short, inefficient chunk. `rewrite_execution_frontiers`
   adds more splits so replay matches the original decomposition.
4. **Host fragmentation and partitioning.** Host KV is a variable-extent arena whose feasibility
   depends on extent geometry, not free bytes (paged-kv §5.3). Host StateImages are a separate
   slot count. `--host-cache-mib` resolves a *static* split between them at startup.
5. **Fragility.** Correctness depends on invariants spread across the ResourceManager and Program
   boundary: capability generations, resource revisions, owner edges, claims, seal windows,
   borrowed-read ownership, and absolute ResourceResult adoption. Recent fix history
   (planning races, shared catalog saturation, lease reclaim) shows the maintenance cost.
6. **Size.** About 25 k lines: `context_cache/` 7.0 k, `planning/` pressure/recovery/request 6.6 k,
   `transactions/` 4.5 k, `storage/` 5.2 k, Host arena + prefix identity + resource contracts
   1.8 k.

The fundamental constraint does not change. Qwen3.5 is hybrid: 48/64 (27B) or 30/40 (35B-A3B)
layers are Gated DeltaNet with a recurrent state that cannot be rolled back. A prefix is therefore
reusable only at a position where the complete recurrent state exists. HPC keeps that rule and
builds everything else around making such states **cheap to create, cheap to store, and exactly
addressable**.

---

## 2. Prior art used

| Source | Idea adopted |
|---|---|
| vLLM automatic prefix caching | Full blocks keyed by (parent, tokens, extra keys). Reference-counted. Freed blocks go to an LRU queue in reverse order, so tail blocks are evicted first. Evictable blocks count as free capacity. |
| vLLM hybrid/Mamba prefix caching | Recurrent state is cached only at block-aligned positions. A hit is the deepest aligned position that has state. |
| SGLang RadixAttention / MambaRadixCache | Radix tree over token blocks. Recurrent-state entries attach to tree nodes with their own eviction. Leaf-first eviction. |
| SGLang HiCache | Host tier with write-through, batched or kernel-based non-contiguous transfers, and layer-ordered loading. |
| llama.cpp context checkpoints | A few full-state checkpoints per sequence for recurrent/SWA models. |
| LMCache | Layer-wise pipelined KV loading overlapped with compute. |
| GreedyDual-Size-Frequency (Cherkasova, 1998) | Size- and cost-aware eviction with O(1) aging via an inflation value `L`. |

NInfer-specific additions in the design (1 and 2 are not implemented, §7.2):

1. The GDN **state tap**: chunked GDN already produces the state at every 64-token intra-chunk
   boundary, so exposing it costs one extra state write.
2. **Phase alignment** by identity-transition padding, so taps land on absolute page boundaries.
3. One pinned **slab pool** for both KV and state.
4. **Dead-KV** accounting: KV that no snapshot can reach has zero reuse value.

---

## 3. Model facts that size the design

### 3.1 KV bytes

Per token, all full-attention layers, K+V including scales (from paged-kv §4.3):

| profile | per token/head | Qwen3.6/3.8-27B (16 FA × 4 KV heads) | page (64 tok) | 35B-A3B (10 × 2) | page |
|---|---:|---:|---:|---:|---:|
| BF16 | 1024 B | 64.0 KiB | 4.00 MiB | 20.0 KiB | 1.25 MiB |
| INT8-G64 | 528 B | 33.0 KiB | 2.06 MiB | 10.3 KiB | 0.64 MiB |
| FP8-row256 | 516 B | 32.3 KiB | 2.02 MiB | 10.1 KiB | 0.63 MiB |
| K8V4 | 402 B | 25.1 KiB | 1.57 MiB | 7.9 KiB | 0.49 MiB |
| NVFP4-G16 | 288 B | 18.0 KiB | 1.13 MiB | 5.6 KiB | 0.35 MiB |

MTP adds one layer's worth of pages to the bundle (+1/16 for 27B). A DFlash draft with
full-attention layers adds its own BF16 pool.

### 3.2 State image

Fixed per-sequence state (qwen3_5-model "State and numerical boundaries", dflash "Backend storage"):

| component | 27B | 35B-A3B |
|---|---:|---:|
| GDN recurrent, FP32 `[128,128,Hv]` × GDN layers | 144 MiB | 60 MiB |
| conv history, `(2·16·128 + Hv·128)·3` BF16 × GDN layers | 2.81 MiB | 1.41 MiB |
| continuation hidden | 10 KiB | 4 KiB |
| DFlash2 local rings (official config) | 40 MiB | config-derived |
| **total (DFlash2)** | **186.8 MiB** (matches the 195,897,344 B Host StateImage in README) | — |

One 27B snapshot equals about 91 INT8 pages (≈5.8 k tokens of KV), 166 NVFP4 pages, or 47 BF16
pages. **Snapshots dominate storage cost. Their placement is the main policy decision.**

### 3.3 RTX 5090 transfer and compute budget

These are the design's planning values. The implementation ranks and values with the Engine's
context-cost coefficients (§6.2, §11.4). On the measuring machine, whose GPU link runs at PCIe 5.0
x8, restores move 24–27 GB/s.

| operation | value used for design |
|---|---|
| HBM D2D copy of one 186.8 MiB image (read+write) | ≈ 0.25 ms |
| PCIe 5.0 x16 pinned H2D / D2H (practical) | ≈ 50 GB/s each direction |
| H2D snapshot | ≈ 3.9 ms |
| H2D 100 k tokens INT8 KV (3.38 GB) | ≈ 68 ms |
| Prefill, 27B, ~121 k context (README median) | ≈ 4.2 k tok/s, so 100 k tokens ≈ 24 s |
| GDN tap write (144 MiB) inside a chunk (zero-split tap, not implemented) | ≈ 0.09 ms |
| Write-through rate at 4.2 k tok/s INT8 prefill | ≈ 140 MB/s (0.3 % of PCIe) |

Consequences:

- A host-resident prefix is about 300× cheaper to restore than to recompute. **The host tier
  carries most of the reuse value.** The device tier is a latency optimisation.
- A state snapshot is cheap to create (a Device copy of the image takes about 0.25 ms) and cheap
  to move (4–8 ms over PCIe). **Snapshots can be dense. The limit is host bytes, not time.**

---

## 4. Architecture

```text
                 Frontend (gateway threads)
   tokens, media digests, boundary hints, per-block chained hashes
                          │ PreparedPrompt
                          ▼
┌──────────────────────── Engine worker (single mutation owner) ─────────────────────────┐
│ Scheduler (unchanged FIFO + backfill) ─ HybridResourceManager (lanes, admission)       │
│      │ quote / reserve / progress / backfill proof / prefetch / terminal settlement    │
│      ▼                                                                                 │
│ Program                                                                                │
│  ├─ PrefixIndex (model-agnostic, src/runtime/prefix_cache/)                            │
│  │    block tree ─ nodes: 64 tokens + extra key, device page ids, host slab, pins      │
│  │    snapshots attached to nodes (+ optional tail)                                    │
│  │    device LRU (tail-first)           host eviction (superseded, GDSF, dead KV)      │
│  │    TapPlanner                         CacheCostModel (prefill and H2D costs)        │
│  ├─ Qwen3.5 binding (src/models/qwen3_5/program/prefix/)                               │
│  │    HybridPrefixCache: Host slab pool, residency, restore and write-through copies   │
│  │    admission, taps, block commit, finish; StateImage parts; persistence             │
│  ├─ Device typed KV pools (unchanged layout) + active block tables (unchanged)         │
│  ├─ Device state pool: C active slots + D snapshot slots                               │
│  └─ streams: restore (dedicated), write-through (the Program's transfer stream)        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Ownership against AGENTS.md boundaries:

- **Core** owns physical primitives: the paged KV cache's record-addressed Host copies, which
  copy runs of consecutive pages and slabs as one strided transfer per plane.
- **Ops** gain nothing: the zero-split GDN tap (§7.2–§7.3) and a KV page transfer kernel (§11.2)
  are not implemented. Taps are Device copies of the committed StateImage.
- **Program** owns the whole cache: exact identity, physical residency, refs, eviction, taps.
  There is no cross-layer logical/physical split, because the cache's policy decisions are
  local physical facts.
- **Runtime** keeps the scheduler, the admission policy and the common request contracts.
  `HybridResourceManager` owns the Engine's lanes and forwards admission, backfill proofs and
  terminal settlement (§8.2), and `runtime/prefix_cache/` holds the model-agnostic index, tap
  planner and cost model.
- **Frontend** supplies tokens, media digests, boundary hints and block hashes. It produces no
  cache candidates, and protocol session keys play no part in the cache.

---

## 5. Data model

### 5.1 Block key and exact identity

Block `b` of a prompt covers absolute positions `[64b, 64b+64)`.

```text
BlockKey(b) = ( parent NodeId,
                tokens[64b .. 64b+63],               // TokenId, exact
                extra = media_key(b) )               // 0 when no Vision item overlaps block b
media_key(b) = xxh3_64 over, for each Vision item whose token span overlaps block b and every
               earlier item (Vision changes later MRoPE positions via rope_delta):
               (content_digest[32], modality, grid, span begin/length)
lookup_hash(b) = xxh3_64(lookup_hash(b-1) || tokens || extra), lookup_hash(-1) = seed
```

- `lookup_hash` only selects a child in the parent's child map. A hit always compares
  `tokens` and `extra` exactly. A forced hash collision must not produce a false hit (tested,
  §13.1).
- Everything else that affects KV or state is fixed for the Engine lifetime: model artifact,
  KV profile, speculative backend, RoPE options. None of it is part of the key. Positions are
  a function of tokens and media, so they are covered by the key.
- Thinking mode, templates, tools and roles are all rendered into tokens, so they need no key
  field.
- The Frontend computes `lookup_hash` for every full block during preparation, off the worker, and
  stores it in `PreparedPrompt::block_hashes`.

### 5.2 Nodes

The records below list the state the design keeps; `PrefixIndex`
(`src/runtime/prefix_cache/prefix_index.h`) holds it in its own layout.

```cpp
struct BlockNode {                      // host-side; stored in a slab-allocated arena
    NodeId   id; uint32_t generation;   // generation-checked handles
    NodeId   parent;
    uint32_t depth;                     // block index b; the node covers [64b, 64b+64)
    uint64_t lookup_hash, extra;
    TokenId  tokens[64];                // exact tokens (256 B)
    ChildMap children;                  // small open-addressed map keyed by lookup_hash
    // Residency
    int32_t  device_page[kMaxPools];    // per enabled pool; -1 when not device resident
    int32_t  host_slab;                 // -1 when not host resident
    Residency device_state;             // Absent | Resident | Filling(event)
    Residency host_state;               // Absent | Resident | Filling(event)
    // Liveness
    uint32_t active_refs;               // active sequences whose table maps this node
    uint32_t restore_pins;              // in-flight transfers reading/writing this node
    uint32_t live_snapshots_below;      // snapshots whose path includes this node
    uint64_t last_access_tick;
    IntrusiveLink device_lru;           // present iff active_refs==0 && device Resident
    IntrusiveLink host_dead;            // present iff live_snapshots_below==0 && no refs && host Resident
    SnapshotId first_snapshot;          // snapshots anchored here (intrusive list)
};
```

A node is only inserted when **all enabled pools** have committed all 64 positions of the block
(§7.6). A node may be device-only, host-only or both. A node with neither copy is deleted, along
with its subtree (§9.4).

### 5.3 Snapshots

```cpp
struct Snapshot {
    SnapshotId id; uint32_t generation;
    NodeId   anchor;                    // node of the last full block below frontier (root if F<64)
    uint32_t frontier;                  // F: the snapshot state is after tokens [0,F)
    uint8_t  tail_len;                  // F - 64*(anchor.depth+1), 0..63
    TokenId  tail_tokens[63];           // exact; only tail_len are meaningful
    // Tail KV (only when tail_len>0): one private, immutable page bundle
    int32_t  tail_device_page[kMaxPools];  int32_t tail_host_slab;
    // State image
    int16_t  device_slot;               // index into the D device snapshot slots, or -1
    SlabList host_slabs;                // ceil(image_bytes / slab_bytes) slabs, or empty
    Residency device_state, host_state;
    uint32_t restore_pins;
    SnapshotKind kind;                  // Tap | Endpoint | OutputBoundary | Boundary (§9.3)
    bool superseded;                    // §9.3
    // Host eviction (GDSF)
    double   gdsf_h; uint32_t freq;
    uint64_t last_hit_tick;
};
```

- **Tap snapshots** have `tail_len == 0`: frontier on a page boundary, KV fully in the tree.
- **Endpoint snapshots** are taken at request finish or cancel at an exact frontier. They own the
  partial last page, which moves from the finishing sequence without a copy.
- A snapshot is **valid** iff every node on `path(anchor)`, the tail bundle (if any) and the image
  each have at least one resident copy (device or host). Validity is maintained eagerly:
  removing the last copy of a node deletes its subtree's snapshots (§9.4).
- There is at most one snapshot per `(anchor, tail tokens)`. Publishing a duplicate is a no-op
  that refreshes recency.

### 5.4 Memory pools

- **Device KV pools**: unchanged geometry, plane order, block-table matrix and consumer views
  (paged-kv §3–§4, §6.4, §10–§11). A page group is in exactly one state: `Free`, `Active`
  (owned by one sequence, private), `Cached` (owned by a node or snapshot tail), or `Filling`
  (transfer destination). `available = free + evictable_cached`.
- **Main pool size**: automatic sizing keeps the capacity curve. The upper clamp `M_max = C·L`
  is removed (only the int32 token representation bounds it): pages beyond the active
  entitlements are cache-only device capacity. `ninfer-serve` defaults `--kv-capacity` to `auto`.
  Active admission still requires `M ≥ max(L, C)`.
- **Device state pool**: `C` active slots plus `D = device_snapshot_slots` snapshot slots in the
  existing slot-indexed layout (`ssm_states [128,128,Hv,C+D]` per layer, and so on). Snapshot slots
  are also the staging destination for taps (§7.4). Default `D = C + 1` with a Host tier (one
  resident snapshot per live conversation plus one staging slot) and `C + 2` without one.
- **HostSlabPool**: one pinned allocation of `host_cache_bytes`, divided into slabs. A slab holds
  one block bundle: the Main page packed with its Host page layout, then the backend page at the
  next 4 KiB boundary; `slab_bytes` is that record rounded up to 4 KiB. A snapshot image uses
  `image_slabs = ceil(image_bytes / slab_bytes)` slabs, not necessarily contiguous (the image's
  packed byte `o` lives in slab `o / slab_bytes`), and a snapshot with a tail one slab more. The
  free list is an index stack, O(1). There is no fragmentation and no geometry check: capacity is
  `free_slabs`. A nonzero budget smaller than one snapshot plus one block is rejected at startup.
  The pool is pinned in chunks of at most 4 GiB of whole slabs (one very large pinned allocation can
  fail or stall); a slab never crosses chunks. A strided copy run joins consecutive
  pages only while their slab records share a chunk and advance by one pitch no larger than the
  device's maximum copy pitch. Any two records form a candidate run, so records in different
  chunks, or gigabytes apart in one, must be split explicitly. A 52 GB Host tier otherwise produced
  an invalid `cudaMemcpy2DAsync` that aborted the server.

### 5.5 Persistence (opt-in)

With `persistent_file` (`--prefix-cache-file`), the Host tier outlives the process. `ninfer-serve`
resolves the path to an absolute one at launch. It rejects a directory, a missing parent
directory, or `--host-cache-mib 0`, so an unusable location fails before any caching:

- **Save**, in the worker's orderly stop when the Engine is destroyed (clean shutdown on
  `SIGINT`/`SIGTERM`; an unfinished save leaves the previous file in place).
  `Program::shutdown_cleanup` releases every lane first, so requests still in flight write their
  committed blocks through, and saves before the cleanup drops the cache. Every Host write
  lands, then every retained (not superseded, §9.3) Host-resident snapshot whose anchor path is
  Host-resident, and exactly those paths, are written parents first with their slab bytes (a
  temporary file renamed into place). Dead KV, superseded snapshots and Device-only content are
  not saved. A worker that ended on a fatal error saves nothing, so the previous file stays in
  place.
- **Load**, at Engine construction before any request: the file is used only when its fingerprint
  (absolute artifact path, size and modification time, prefill signature, KV storage, speculative
  backend, and the product binary's identity — executable size and modification time, so any
  rebuild invalidates it) and its Host geometry equal the running Engine's, and its
  size matches its tables. The file (format 3) holds the header, the block table, the snapshot
  table and only then the slab bytes, so the loader chooses before it reads any slab. A Host tier
  that holds the whole file restores all of it. A smaller one restores whole snapshots, greedily
  by the GDSF density of §9.3 without its time-dependent base — (1 + hits) × the prefill a
  restore saves from the root, per byte of image, tail and anchor path — each with the path
  blocks it needs and no others, and reads only those slabs (`plan_host_restore`); the startup
  log then warns `prefix cache partly restored` with the kept and saved counts and the Host tier
  the file needs. A tier that holds none of the snapshots restores nothing, since a restore keeps
  only the blocks its snapshots resume through; the log then warns `prefix cache not restored`
  with the tier the file needs, and that the save at shutdown replaces the file. Entries are rebuilt Host-only, parents before children, without evicting
  anything, and their first requests restore blocks and images through the ordinary Host restore
  path. A damaged file loads nothing.
- The startup log shows the read of an accepted file as a progress phase (`loading prefix cache`,
  then `prefix cache read | bytes | time | rate`), then reports what was restored or why nothing
  was; the shutdown log reports the save.

---

## 6. Admission (lookup, choice, capacity)

### 6.1 Lookup

`PrefixCache::match(prompt)` runs on the worker:

1. Walk from the root. For block `b < floor((n-1)/64)`, find the child by `block_hashes[b]`,
   then compare tokens and extra exactly. Stop at the first miss. This gives the matched path
   `N_0..N_{k-1}`, `k` full blocks.
2. Collect candidate snapshots: every snapshot anchored on the matched path (walk each node's
   list) whose tail tokens equal `prompt[64·(anchor.depth+1) .. F)`, with **`F ≤ n − 1`**. At
   least one prompt token is always prefilled, so logits, continuation hidden and MTP/DFlash
   bridges come from the normal path. No full-hit path exists.
3. Frontiers inside a Vision token span are not valid resume points. Tap placement never creates
   them, and match rejects them defensively.

Complexity is O(k + snapshots on path). A 240 k prompt is 3,750 hash probes and exact 256 B
compares: tens of microseconds.

### 6.2 Choice

Candidates: root, the deepest valid snapshot `s*`, and the deepest valid snapshot whose image and
path are fully device resident, `s_dev`. At most three. For each:

```text
cost(s) = restore_seconds(s) + prefill_seconds(base=F_s, tokens=n-F_s)
restore_seconds = (host_path_bytes + host_image_bytes + host_tail_bytes) / measured_h2d_bw
                  + per_batch_latency
```

`prefill_seconds` is the existing calibrated machine model (`context_cost.cpp`: linear tokens
plus `B·S + S(S+1)/2` attention pairs, keyed by `prefill_signature`). Pick the minimum. Ties
favour fewer transferred bytes. This is O(1) and deterministic, and the cost model only ranks
feasible choices.

### 6.3 Capacity

```text
need_pages(s) = host_only_path_blocks(s)          // H2D destinations (shared once restored)
              + (s.tail_len>0 ? 1 : 0)            // private COW copy of the tail
              + growth_pages(prompt n, base F_s)  // existing entitlement: remaining prompt +
                                                  // bounded output window + per-lane cushion
need_state_slot = 1 active slot (a lane implies a free active slot)
```

The request is admissible with `s` iff `need_pages(s) ≤ available_pages`, where evictable pages
exclude pages pinned by this admission's own path. If not, try the next cheaper-to-materialise
candidate: `s_dev`, then root.

- Root infeasible because active entitlements hold the pages: `TemporarilyBlocked`.
- Root infeasible even on an empty cache: `PermanentlyInfeasible`.

This is the same classification as today, computed by arithmetic.

**Backfill proof** (engine-architecture §5.2). The Engine's FIFO protection is unchanged: a
blocked head freezes its donors (the requests active when it blocked), and a later request may
borrow capacity only with a Program proof. A lane's Device pages are its shared path pages plus its
private reservation, which together equal its entitlement, and once the donors finish every page
no lane maps is free or evictable cache. So the Program proves, per pool,

```text
Σ_borrowers entitlement + entitlement(candidate) + entitlement(head) ≤ pool capacity
borrowers + 2 ≤ max_concurrency
```

with the head's entitlement from its root plan, so no cache state is assumed to survive the wait.
The candidate must also be admissible now (its quote is `Ready`). Shared pages are counted once per
lane, which only makes the proof conservative. No growth reserve is needed, because an admitted
request never grows past its entitlement (paged-kv §6.2).

Every unpinned Device-resident cache entry (node or snapshot tail) owns its page bundle
exclusively, so evicting one returns exactly one page to each pool and the arithmetic is exact.
Exact-frontier tap tails are copied into their own pages for this reason (§7.4). Candidates with a
block another admission is still filling are skipped.

### 6.4 Admit

Admission is the Engine's materialization transaction (`hybrid_reserve_materialization`, then
`progress_context_transaction` until `Published`). The first progress step stages it
(`hybrid_stage`), in the same Engine call as the quote:

1. Re-validate the quoted snapshot and path. Pin every path node up to `F_s` (removed from the
   device LRU) and pin `s`.
2. Evict from the device LRU until the pools hold every page the admission needs, then **reserve**
   them: `entitlement − F_s/64` for the fork, plus one per host-only path block and host-only tail.
   Reserve the lane's StateImage destination slot.
3. If anything is host-only, open a restore batch on the dedicated restore stream (ordered after
   the compute stream's queued work, since destinations may have been read by earlier kernels):
   - host-only blocks: materialize destination pages from the reservation, adopt them as cache
     references, mark the nodes `Filling`, and copy the slabs in runs of consecutive pages and
     slabs;
   - host-only tail: the same into a cache-owned page (the snapshot's tail is Device-resident
     again afterwards);
   - host-only image: copy the image slabs straight into the reserved destination slot.
   The batch is enqueued in the order a forward pass reads it (§6.5).

Activation (`hybrid_activate`) then builds the lane at once, without waiting for the copies: the
reservations return to the pools and the page-prefix fork takes them back; path pages are shared; a
partial tail is D2D-copied into a private page; the state is reset, D2D-copied from the snapshot's
device slot, or restored in place. The lane takes over the path pins and the snapshot pin is
dropped; a restore batch holds its own pins on the nodes and snapshot it copies until it lands.

Admission never evicts anything that another sequence has pinned. Evictions come from the device
LRU (§9.1) and complete synchronously: dropping a device copy is bookkeeping, since the data is
already backed up or deliberately discarded.

### 6.5 Restoring

A restore batch never holds its admission. `submit_restore` enqueues it on the restore stream in
the order a forward pass reads it, recording an event after each part:

1. the **prelude**: the snapshot tail bundle, every restored block's backend page, and the image's
   continuation hidden and DFlash local state;
2. then, for each model layer in forward order, that layer's state: a full-attention layer's KV
   planes of every restored block, or a linear-attention layer's conv and recurrent state.

`land_restore` makes the compute stream wait for the prelude (activation copies the tail and the
MTP bridge reads the backend pages and hidden) and hands the lane a ticket for the per-layer
events. The lane's first prefill pass waits for each layer's event just before that layer
(`TextContext::run_layers`), so it computes the early layers while later ones are still arriving;
later passes are stream-ordered behind it. The chunk function itself looks the events up by
ticket (`PrefillContext::take_layer_ready`) just before its pass, because `poll()` retires a
landed batch and recycles its events, and program code between chunks polls: a batch already
retired has landed, so the pass has nothing to wait for. A lane released before its first pass
makes the compute stream wait for the whole batch, because the state slot it returns may still be
a destination.

The index keeps the batch's nodes and tail `Filling`, and the batch keeps them and its snapshot
pinned, until `poll()` (every admission quote, and the block publication after each KV commit)
or `drain()` sees the last event. So no eviction,
dead-KV reclaim or Host write can free a slab or page a copy still reads or writes. Other admissions
skip `Filling` candidates for that short window.

Consequences:

- A Host revisit no longer waits for the host to notice the copy finished: under concurrency that
  was the rest of the other lane's decode round (~12 ms measured), and with an idle Engine a timer
  tick.
- The copy overlaps the restored request's own first prefill chunk: for a long context, the prefill
  of new tokens hides most of a multi-GB restore.
- Other lanes' decode rounds queue behind that prefill chunk as for any prefill. When the chunk is
  much shorter than the copy (a few new tokens after a very long restored context), they can wait up
  to the copy's remaining time once.

Failure: a CUDA error in a restore job is Engine-wide, as today. A request cancelled while staged
(before activation) waits for the restore stream, returns every destination and reservation, and
releases its pins.

### 6.6 Prefetching the blocked head

When the Device pool cannot hold every conversation, each admission restores most of its path
from the Host tier (the production agent log of 2026-09-26: 2.59 TB, a median 2.7 GB per turn).
For a short turn the first prefill pass waits for those copies, and other lanes' decode queues
behind it. The FIFO head usually waited seconds to minutes for a lane or for pages before that,
so its restore can happen during the wait instead.

- When the head stays blocked and no borrower is admitted past it (§6.3), the Engine asks the
  Program to prefetch for it. The Program matches the prompt as a quote would, takes the source admission
  would choose, and copies up to 256 of its Host-only path blocks into Device pages. They are
  ordinary cached blocks: `Filling` until the batch lands, then unpinned at the MRU end of the
  device LRU. Admission later finds them Device-resident and restores only the rest.
- Pages come from the free pool and from host-backed cached blocks, least recently used first.
  The head's path is pinned meanwhile. A prefetch never drops a block's last copy, never waits for
  a transfer, and never runs beside an open admission or another prefetch. Admissions may evict
  the prefetched blocks like any cache.
- One batch is about 550 MB for 27B INT8 KV (about 20 ms of PCIe). The Engine retries for the same
  head only while the last attempt copied blocks or the spare room grew, because matching walks
  the whole prompt path.
- A quote whose candidates cross a landing prefetch waits for it (`settle_prefetch`) instead of
  skipping those candidates for a shallower source. A lane's own restore batches are never waited
  for synchronously (§6.5).

In the production log about 57 % of the restored bytes could have been copied this way. The rest
did not fit beside the active sequences while the head waited.

---

## 7. Prefill, taps, decode, finish

### 7.1 Tap planning

`plan_taps(n, base F_s, hints, existing frontiers, Vision exclusions)` returns taps sorted by
position, each `{p, placement}` with `F_s < p ≤ n − 1`, `p` not strictly inside a Vision token span
(a candidate inside one moves to the span start), and no existing or already-planned snapshot at
or near `p` on this path.

Candidates, in priority order:

| priority | source | placement | rationale |
|---|---|---|---|
| 1 | explicit markers (OpenAI breakpoints, Anthropic `cache_control`, C++ `PromptInput` markers) | exact | the client names reuse points |
| 2 | start of the final assistant generation opener | exact | the next turn re-renders the history after it (for example, stripped thinking) |
| 3 | end of tools, end of the leading System/Developer block | exact | compaction and new sessions sharing the preamble |
| 4 | `n − 1` ("prompt tail") | flexible | regenerate / retry of the same prompt |
| 5 | geometric ladder: for k = 0,1,…, `n − G·2^k` (G = `tap_ladder_tokens`), snapped back to a message boundary within the minimum gap | flexible | edits deep in history. Spacing grows with distance, so there are about log2(n/G) taps |

- **Exact** taps split the prefill chunk at `p` (one more forward pass each) and snapshot there.
  Semantic boundaries are where later prompts diverge, so the exact position is worth the split.
- **Flexible** taps never split: the snapshot is taken at the end of the chunk that reaches `p`, or
  at the start of the prompt's final chunk when `p` lies in it (the prompt tail always does). A
  chunk boundary costs nothing beyond the snapshot copy.
- Semantic boundaries (generation opener, structural) closer than 64 tokens form one cluster and
  only its earliest position is kept: a short user turn puts the system-block end next to the
  opener, and a snapshot at the system-block end serves new conversations sharing it while the
  next turn recomputes fewer than 64 tokens.
- A snapshot serves every prompt that diverges after it, so proximity only makes a *later*
  candidate redundant: a non-explicit candidate is dropped when an accepted tap, an existing
  snapshot on the path or the base lies at most 64 tokens before it. Ladder candidates keep
  `tap_min_gap` (default `max(1024, chunk)`) on both sides instead. The opener therefore absorbs
  the prompt tail in chat traffic, leaving one split per request.
- Only markers with explicit evidence (a client-named breakpoint) are priority 1. Protocol-automatic
  markers (OpenAI default caching, Anthropic automatic `cache_control`) are planned as structural
  boundaries, but they mark the conversation's latest turn rather than a prefix conversations
  share.
- Explicit and structural taps (not protocol-automatic ones) are published as `Boundary`
  snapshots: conversations may share them, so a lineage moving past one supersedes it only while
  no other conversation has continued from it (§9.3). A cluster keeps this from any member it
  drops.
- A request that resumed from an endpoint snapshot proves its client echoes generated turns token
  for token (agent loops, preserved reasoning), so its own endpoint serves the next turn and the
  generation opener is not tapped. Measured on a tool-calling agent trace (27B NVFP4, INT8 KV,
  `--preserve-thinking`), the opener split cost about 15 ms of a ~70–130 ms turn TTFT. Requests
  that resumed from a tap keep the exact opener: their client re-renders history.
- Hints are the Frontend's existing `message_boundaries` plus the structural boundaries it already
  computes.
- At most `max_new_taps` (default 8 with a Host tier, 2 without) taps are kept per request.

Tap placement is a function of the prompt, so it is deterministic. Endpoint snapshots are exact
(§7.7).

Exact taps are the only prefill splits. The replaced cache also split prefill at the chat
template's rewrite execution frontiers (after each assistant header, after `<think>`, after the
reasoning close) for its rewrite checkpoints and execution provenance; the prefix cache captures
nothing there, so prefill ignores them. Each split is a whole extra pass over the model. A short
chat turn used to prefill as four passes (reuse point, opener, `<think>`, reasoning close). Now it
takes two (to the opener, then the rest), or one when the opener is dropped, unless a client
breakpoint or structural boundary falls inside the new tokens.

### 7.2 Phase-aligned chunked GDN

**Not implemented** (§7.2 and §7.3 together form the zero-split tap). The implemented tap copies
the lane's whole committed StateImage at a chunk boundary (≈0.25 ms D2D), so only exact taps pay a
split. The split also changes the floating-point decomposition of the prompt (§7.8): after a split
at a position that is not a multiple of 64, the scan's output differs from an unsplit prefill.
Phase alignment would remove that, but chunk width selects other routes too (§7.8), so on its own
it would not make a prefill with taps match one without. Build this only if measurement shows the
remaining exact-tap split matters.

The chunked GDN kernel's intra-chunk grid is relative to the call start. To make every internal
boundary an absolute multiple of 64, the Op gains a `phase = base % 64` parameter:

- The kernel treats the call as if it were left-padded by `phase` virtual tokens with
  `k = 0, v = 0, g = 0, β = 0`. For such a token, `α = exp(0) = 1` exactly and
  `delta = 0 · (v − αS·k) = 0` exactly, so `S ← 1·S + outer(0, k) = S` **bit-exactly**.
- Padded outputs are not written.
- Padding is by predicated loads, with no workspace copy.
- `phase = 0` is the current behaviour, bit for bit.

Prefill chunks remain `--prefill-chunk` tokens long. The **first** chunk after a non-aligned base
is `chunk − phase` tokens long, so every later chunk starts page-aligned. This is one boundary
change in `TextContext::prefill`.

### 7.3 Tap Op contracts

**Not implemented** (the design of the zero-split tap, with §7.2). What is implemented captures
more than this list: the tap is a copy of the lane's StateImage, which holds every per-layer
conv history and recurrent state, the continuation hidden, and for a DFlash/DFlash2 Program the
local K/V ring (`StateImageSpec`). A resume therefore re-materializes nothing beyond the KV blocks
the tree already holds.

For each prefill unit with taps `T_u = T ∩ (unit_begin, unit_end]`, `|T_u| ≤ taps_per_unit`
(default 4, bounded by free snapshot slots):

1. **GDN state tap**: `gated_delta_net(..., StateTapSet taps)`. For each tap `j`, the kernel writes
   the FP32 state after absolute position `p_j` for every value head into
   `snapshot_slot[j].ssm_state[layer]`.
   - Stores use `st.global.cs` (streaming) to avoid displacing prefill working sets in L2.
   - Oracle: the FP64 naive recurrence evaluated to `p_j` (op-development §numerical).
   - Exactness criterion: the tapped state is **bit-identical** to the final state of a separate
     call on the same inputs ending at `p_j` with the same phase. Same chunk grid, same
     arithmetic.
2. **Conv tap**: per GDN layer, copy the `kernel_width − 1 = 3` pre-conv projected columns
   `[p_j − 3, p_j)`, taken from the current unit's projection output or the previous conv history
   when `p_j − unit_begin < 3`, into the slot's conv history. Exact copy, exact oracle.
3. **Hidden tap**: the continuation hidden at column `p_j − 1`, stored for completeness. It is
   never used for a full hit (§6.1).
4. **DFlash/DFlash2 ring tap**: after the unit's context K/V materialisation, and *before* the
   ring is overwritten, assemble `slot.ring = ring(positions [p_j − S, p_j))`. Each position comes
   from the pre-unit ring if `< unit_begin`, otherwise from the unit's freshly materialised K/V.
   The materialiser writes this unit's context K/V to a unit staging buffer first; the ring commit
   and the tap gathers both read from it. Exact copy.
   - The DFlash Full (paged) pool is committed through the chunk at unit commit, so its pages
     belong to the tree like Main pages.
5. **MTP**: MTP KV is paged. Tap snapshots carry no MTP state. A resume performs the Program's
   existing MTP bridge from the resumed frontier, as for any Program-resumed frontier today.

If no snapshot slot is free when the unit starts, the tap is skipped and counted
(`taps_skipped_no_slot`). In practice D2H of a slot (≈4 ms) finishes long before the next unit
(≥0.3 s).

### 7.4 Tap publication

At the chunk boundary the tap takes a staging device snapshot slot, copies the lane's committed
StateImage into it (including the continuation hidden of column `p − 1`, which prefill keeps while
taps remain) and waits as a **pending tap** of the lane. It is published as soon as the blocks it
anchors on are committed tree nodes:

- the anchor `p/64 − 1`, which for an MTP backend (one token behind the text frontier) commits
  only after the next chunk even when `p` is page-aligned;
- for `p % 64 ≠ 0`, block `p/64` as well, whose first `p % 64` columns are copied into a
  cache-owned tail page, so every cache entry keeps an exclusive page (§6.3).

Publication creates a `Snapshot{Tap}` with `device_slot` = the staging slot and enqueues D2H
write-through of the image (and tail) into host slabs; `live_snapshots_below++` along the path.
Once write-through completes, the slot becomes an **evictable device snapshot** (§9.2). A finishing
lane publishes its remaining pending taps from its own last page, or drops them.

### 7.5 Recomputed region

When `F_s < 64k`, blocks `[F_s/64 .. k)` already exist in the tree but are recomputed, because the
recurrent state must be rebuilt. The sequence writes them into its own private pages. At block
commit (§7.6), each such block finds the existing node, so the private page stays private
(`Duplicate`) and is freed at finish. The tree is not modified. KV-write elision, which maps the
existing pages and discards those writes, is an optional optimisation (§12.4).

### 7.6 Block commit (prefill and decode)

After any commit that advances every enabled pool's committed frontier across a block end
`64(b+1)`:

- If `child(N_{b−1}, key(b))` exists with a Device copy: the sequence's page stays private
  (`Duplicate`). If it exists host-only, it adopts the sequence's page as its Device copy. Otherwise
  **insert** a node owning the sequence's page. The node stays pinned for this sequence.
- D2H write-through of the lane's Device-only nodes is enqueued in one batch when the lane releases
  its path (finish, abort or cancel), before the nodes become evictable. Batching coalesces runs of
  consecutive pages and slabs into one strided copy per plane, and pinned nodes need no backup yet.

The block-table row is unchanged, so no GPU work is needed.

Only committed tokens count. Speculative provisional lead, rejected columns and MTP/DFlash lag are
excluded by requiring every enabled pool's *committed* frontier (paged-kv §9.1).

Decode blocks become cacheable the moment they fill. A later request that shares a long generated
output, such as an agent re-sending the assistant turn, can resume from the endpoint snapshot
(§7.7).

### 7.7 Finish and cancel

At terminal settlement (Finish, or Cancel after at least one committed unit):

1. If the request is cache-enabled, every pool has committed the frontier (the backend up to its
   own frontier, `F_end − 1` for MTP) and `F_end − (deepest snapshot on path) ≥ 64`:
   - The active StateImage itself becomes the snapshot image (frozen, no copy) in a device snapshot
     slot; if none is free, one is evicted (§9.2), otherwise the endpoint is skipped.
   - The partial last page (`F_end % 64 ≠ 0`) moves to the snapshot as its tail. There is no COW,
     because the writer is terminating. The backend page keeps its own committed extent.
   - Publish `Snapshot{Endpoint}` and enqueue D2H write-through of the image and tail.
2. For every pinned node: `active_refs--`. A node that reaches 0 goes to the device LRU tail
   (§9.1), inserted in **reverse block order**, so deeper blocks are evicted before shallower
   ones.
3. Free `Duplicate` and growth pages. Release the active state slot.

`Disabled` requests (Serve warmup, `--no-prefix-reuse`, CLI generation) never insert nodes or
snapshots. Their pages are freed at terminal.

### 7.8 Reproducibility contract

Greedy generation is a function of the prompt and of the floating-point decomposition used to
prefill it. The cache preserves the decomposition and cannot make two decompositions agree:

- **A resume is bit-exact with a cold prefill under the same cache settings.** A snapshot is a copy
  of state a request computed on its own chunk grid, and tap positions are a function of the
  prompt, so a request that resumes from a tap and one that prefills from the root run the same
  grid from the tap onward (§7.1). `resume-exact` in the real hybrid test checks this for no
  speculation and DFlash2.
- **Splitting a chunk changes the decomposition.** Exact taps split one chunk, so a prefill with
  taps generates different greedy tokens than `--no-prefix-reuse` or `--cache-taps-per-request 0`,
  just as another `--prefill-chunk` does. Endpoint resumes continue from decode-computed state, so
  they differ from a prefill of the echoed tokens in the same way. Only `--no-prefix-reuse` makes
  every output independent of the cache.
- **Prefill Ops choose their arithmetic by call width.** Comparing per-layer activations of the
  same prompt prefilled with different chunks isolated each source (27B NVFP4 artifact, FP8 A8
  projections, bf16 KV); none is in prefix-cache code:
  - FP8 A8 linears split K for the tiles of the last partial wave (`fp8_tma_split_k_plan`), so
    which columns reduce in parts depends on the call width;
  - the FP8 epilogue's scale and residual arithmetic contracts differently in the full-tile and
    ragged-tile instantiations (identical with explicit `__fmul_rn`/`__fadd_rn`);
  - the BF16 GDN gating projection picks its split count from the width (`k27Routes`);
  - causal attention runs chunks of at most 256 queries (80 for FP8 KV) on the grouped kernel with
    split keys, and wider chunks on the tiled kernel;
  - below 17 columns the FP8 projections take the A16 route instead of A8, and the gating
    projection its small-width routes, so the few tokens after a generation-opener split are
    computed at a different activation precision;
  - the chunked GDN scan's grid is relative to the call start (§7.2), so a split that is not a
    multiple of 64 changes every later position.

  With the first three neutralised, prefills with chunks 4096 and 3968 agree bit for bit at all
  37,937 positions of one prompt; an exact tap then still differs through the last two. Phase
  alignment alone therefore does not give bit equality with an unsplit prefill.

Measured on the Qwen3.8-27B official NVFP4 artifact, RTX 5090 (WSL2), `--prefill-chunk 4096`, eight
conversations of 17–44K prompt tokens each (system block, a haystack of repository documents and a
question; thinking on), 300 greedy tokens per answer. The first prefill is a 10–22K-token prefix of
the second, so the second resumes from a prompt-tail or ladder tap when its snapshot survives
(`taps_created` and `host_snapshot_evictions` show the Host tier overflowing after the third
conversation, so only the first one to three resumed). "Identical" counts conversations whose 300
tokens match; the number after it is the generated-token index of the first difference.

| comparison | bf16 KV, no speculation | K8V4 KV, DFlash2 |
|---|---|---|
| the same run twice | 8/8 identical | 8/8 identical |
| cache with taps, resumed vs cold prefill | 8/8 identical (1 resumed) | 8/8 identical (3 resumed) |
| `--no-prefix-reuse` vs cache with taps | 1/8 identical; first differences at 38–211 | 0/8 identical; 8–93 |
| `--no-prefix-reuse` vs taps 0 | 8/8 identical | 8/8 identical |
| `--no-prefix-reuse`, chunk 4096 vs 3968 | 1/8 identical; 8–113 | 0/8 identical; 8–83 |
| `--no-prefix-reuse`, chunk 4096 vs 2048 | 1/8 identical; 8–116 | not run |

Tap splits move greedy output as much as a chunk-size change does, and a resume adds nothing to
either.

The differences are rounding-level per Op and within each Op's qualification tolerance; they matter
for reproducibility, not accuracy. On one 37,937-token prompt the last layer's hidden states for
chunk 3968 against chunk 4096 have median cosine similarity 0.992 (1st percentile 0.29), and
K8V4 against bf16 KV, a supported storage choice, gives 0.990 (0.28): long-context greedy decoding
amplifies any perturbation of that size into a different but equally valid continuation.

---

## 8. Engine and API

### 8.1 Program API

```cpp
struct HybridAdmissionQuote {               // hybrid_quote(): the selected source, exact at admission
    runtime::Readiness readiness;           // Ready | TemporarilyBlocked | PermanentlyInfeasible
    runtime::LaneId destination;
    runtime::RequestPlanSummary summary;    // carries the reuse frontier, reported as cached_tokens
    std::shared_ptr<detail::HybridQuoteImpl> impl;
};
[[nodiscard]] HybridAdmissionQuote hybrid_quote(const PreparedPrompt&, const RequestBasePlan&,
                                                runtime::LaneId destination);
[[nodiscard]] runtime::ContextTransactionReserveStatus
    hybrid_reserve_materialization(HybridAdmissionQuote&&, PreparedPrompt&&,
                                   runtime::CancellationFlagView);   // §6.4; progressed by
                                                                     // progress_context_transaction
[[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan& blocked_head, const HybridAdmissionQuote&,
                              std::span<const SequenceHandle> persistent_borrowers) const;   // §6.3
[[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const PreparedPrompt&,
                                                           const RequestBasePlan&);  // §6.6
[[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
void set_hybrid_cost(const runtime::prefix_cache::CacheCostModel&);                   // §6.2
[[nodiscard]] HybridCachePersistence attach_hybrid_cache_file(...);                  // §5.5
[[nodiscard]] std::optional<HybridCachePersistence> hybrid_shutdown_save() const;
[[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept;
// Unchanged: advance_prefill, decode, append_forced_tokens, commit, abort_pending, finish, abort.
```

### 8.2 Engine

The Engine constructs `HybridResourceManager`, which tracks lane ownership and forwards admission,
backfill proofs and terminal settlement; retention policy lives in the Program's index. Scheduling,
prefill, decode and commit orchestration are unchanged.

- Admission is the ordinary context transaction: quote, reserve, then
  `progress_context_transaction` until the started sequence is published (§6.4). Restores never
  hold it (§6.5).
- `ensure_base_plan` builds no candidates, and no capture exists: taps are internal to prefill.
- A blocked FIFO head keeps the existing protection and backfill path; the Program's proof is
  §6.3.
- While the FIFO head stays blocked and no borrower is admitted, the Engine asks for a prefetch of
  its Host-only path (§6.6).
- `HybridPrefixCacheStats` feed `record_prefix_selection` and the request-log fields.

### 8.3 Frontend and protocol mapping

- `PreparedPrompt` carries `block_hashes` and `tap_hints` (sorted `{position, priority}`). The
  checkpoint-catalog `PreparedContextCache` (session key, retention, opportunities) is gone; the
  `PromptIdentity` rewrite checkpoint supplies the generation-opener tap.
- `ContextCacheHints` is unchanged and validated as before:
  - explicit-evidence markers map to priority-1 tap hints, automatic protocol markers to
    `Automatic` hints (planned as structural, published as ordinary taps);
  - `allow_engine_automatic_shared_prefixes = false` (every OpenAI request, and Anthropic
    requests with request-level `cache_control`) omits the Engine's own tap at the end of the tool
    definitions, as it omitted that shared-prefix candidate before;
  - `session_key`, `retention` and `update_session_index` are accepted and ignored. Content
    addressing finds session chains without them.
- External protocol behaviour is unchanged:
  - OpenAI `prompt_cache_*` and breakpoints, and Anthropic `cache_control` validation, limits
    (four distinct breakpoints) and error cases are unchanged.
  - Breakpoints mean "tap here" (priority 1). Protocol write-policy modes (`mode:"explicit"`,
    Anthropic `cache_control`-only requests) do not suppress automatic taps: taps are cheap and
    content-deduplicated, so markers can only add reuse.
  - `cached_tokens` and `cache_read_input_tokens` report `reuse_tokens`. The streaming
    `message_start` timing is unchanged, because the quote fixes the value at admission.

---

## 9. Eviction

### 9.1 Device KV (LRU, tail-first)

- One intrusive list of `Cached` device pages whose owner has no refs and no pins: nodes with
  `active_refs == 0 && restore_pins == 0`, and snapshot tails with `restore_pins == 0`.
- On a hit, touched nodes move to the MRU end.
- On release, nodes are appended in reverse block order.
- Allocation pops from the LRU head:
  - **Host-backed** entries (host `Resident`): drop the device copy. Free.
  - **Unbacked** entries, which occur only when the host is disabled, full, or write-through is
    still filling: the node loses its last copy, so apply §9.4.

Two lists are kept, *backed* and *unbacked*, and backed entries are always consumed first.

### 9.2 Device snapshot slots

Slots hold at most `D` images. A sequence about to snapshot past its previous snapshot (the one
it resumed from, or its previous tap in the same prompt) supersedes it first (§9.3), so the slot
comes from its own lineage when it can. A tap or endpoint takes a slot in this order:

1. a free slot;
2. a slot owned by a superseded snapshot, oldest supersession first; the snapshot keeps its Host
   copy or, without one, is deleted;
3. a slot owned by a Device-only snapshot (no Host copy: the Host tier is full, disabled or
   evicted it): the one with the lowest GDSF priority (§9.3), and only when the new snapshot's
   estimated priority (no hits, and the blocks since the sequence's previous snapshot its own) is
   at least as high. The snapshot is lost, and `L` rises to its priority;
4. the least recently hit Host-backed snapshot's slot, which loses only its Device copy.

Otherwise the tap or endpoint is skipped. A Device-only owner's slot is its last copy, so value
decides there rather than recency: least-recently-hit eviction let a short request's snapshots
push out the snapshot a long conversation resumes from, and taking Host-backed slots first left
owners that Host eviction had made Device-only in their slots for good (§16.5).

### 9.3 Host (superseded snapshots first, then GDSF; dead-KV first)

**Supersession.** In a conversation the next request resumes from the newest snapshot of its
lineage, not from the one the current request resumed from. When a sequence is about to publish a
deeper snapshot on its path, the snapshot it resumed from becomes *superseded*, and so does the
previous tap it captured in the same prompt when the new snapshot is a tap, before the new snapshot
takes a Device slot or Host slabs: only a request diverging before the newer snapshot (a retry, an
edit) can still use them. The endpoint leaves the prompt's last tap retained, since a next turn
whose template re-renders the reply (stripped thinking) resumes from that tap. A hit on a superseded
snapshot retains it again, until that lineage moves on. A `Boundary` snapshot (a tap at a client
breakpoint or structural boundary) may be shared by conversations, so it is superseded only while no
other conversation has continued from it: while the tree below it is a single chain. Until then it
is just its lineage's snapshot.

Allocation of `k` slabs:

1. **Dead KV sweep**: pop host-resident nodes with `live_snapshots_below == 0`,
   `active_refs == 0` and no pins, leaves first. These can never produce a hit.
2. Otherwise evict the oldest superseded snapshot entirely (both copies; `L` is unchanged), then
   repeat step 1.
3. Otherwise evict the retained snapshot with minimum `H`, then repeat step 1 (its exclusive path
   has just become dead). Update `L := H(victim)`. A victim still complete on the Device (image in
   a device slot and tail resident) only gives up its host copy and stays restorable.

A snapshot image write stops at step 3 when the victim is worth more than the new snapshot
(`H(victim) > H(new)`): GDSF never inserts what it would evict next, and the new snapshot stays
Device-only (§9.2). KV block writes (write-through at release) are not bounded this way.

```text
H(s)  = L_s + F(s) · C(s) / Z(s)                // L_s: L when s was published or last hit
C(s)  = prefill_seconds(base=F_a, tokens=F_s−F_a) − restore_seconds(image)   // clamp ≥ 0
        where a = nearest retained (not superseded) snapshot above s on its path, whatever its
        tail (or root, F_a = 0)
Z(s)  = image_bytes + tail_bytes + exclusive_path_bytes(s)
        exclusive_path_bytes: host bytes of the nodes on path(s) below which s is the only
        retained snapshot
F(s)  = 1 + hits(s)                             // hits since publication
```

- Each snapshot records its nearest snapshot above it on its path (`ancestor`) and the snapshots
  that record it (`dependents`); a new snapshot between them takes over the dependents below it,
  and a removed one hands its dependents to its ancestor. `C` and `Z` are recomputed for the
  snapshot and for those whose nearest retained ancestor or exclusive path it changes, when it is
  published, hit, superseded, retained again or removed.
- Valuing against the nearest *retained* ancestor makes a conversation's newest endpoint carry the
  whole conversation-specific prefill (its superseded predecessors go first), and a shared
  boundary its own prefix. Valuing only against aligned (`tail_len == 0`) ancestors, as before,
  priced every tailed endpoint as a full re-prefill from the root: stale endpoints of long
  conversations crowded out shared boundaries and fresh endpoints, which were evicted before the
  next turn (the production agent log of 2026-09-26: 68 turns re-prefilled 758K tokens behind a
  stuck resume point, 24 subagent turns prefilled from the root beside a cached 19.8K-token
  shared prefix).
- `C` uses the same calibrated model as §6.2, so eviction and admission agree about value.
- Aging comes from `L`, so snapshots that are no longer hit fall behind new ones without timers.
- Victim selection scans the snapshot table, O(snapshots); valuation walks at most one path per
  affected snapshot. Snapshot count is bounded by `host_cache_bytes / image_bytes` (about 1,300
  for a 230 GiB Host tier with 27B DFlash2 images), so the policy is independent of the tier size:
  a small tier reaches step 2 and 3 often, a large one rarely.
- Pinned objects (`restore_pins`, `active_refs`, in-flight fills) are never evicted.

### 9.4 Losing the last copy

A node whose device and host copies are both gone is deleted together with its subtree. Every
snapshot anchored in the subtree is deleted, and `live_snapshots_below` is decremented along the
paths. A node with `active_refs > 0` always has a device copy, so this never affects an active
sequence. The subtree walk runs only in the unbacked cases of §9.1 and §9.2, and is bounded by
the subtree size.

### 9.5 Active guarantee

- Active entitlement is unchanged: admission reserves the full prompt-plus-output entitlement
  (paged-kv §6.2), and an active request never grows past it.
- Cached pages never count against it: `available = free + evictable`, and admission evicts cache
  to satisfy a reservation, never another request's reserved pages.
- Nothing active is ever evicted.

---

## 10. Invariants

1. Each device page group and each host slab has exactly one owner state:
   Free/Active/Cached/Filling, or Free/Node/SnapshotImage/SnapshotTail/Filling.
2. A tree node is immutable: its tokens, extra key and payload never change after insertion.
3. Every node on an active sequence's mapped path has `active_refs > 0` and a device-resident
   copy until the sequence releases it.
4. A snapshot exists only while it is valid (§5.3). Match returns only valid snapshots, and only
   with `F ≤ n − 1`.
5. Exact identity: a match implies exact equality of tokens and extra keys for every block and
   tail token. Hashes never decide equality.
6. A Filling destination is never mapped into a block table, and its readers wait on its event.
   A Filling source is never freed.
7. Taps are bit-identical to an unsplit run ending at the tap, for the same phase.
   Identity-padding for phase alignment is bit-exact.
8. `Disabled` requests leave no nodes or snapshots behind.
9. The worker is the only mutator. Transfers change residency only in `poll_transfers()` or at
   commit.
10. Cost models only rank choices. Capacity and validity checks never depend on predictions.

---

## 11. RTX 5090 / sm_120a optimisation requirements

### 11.1 Streams

- Restores run on a dedicated non-blocking restore stream, ordered after the compute stream's
  queued work (destinations may have been read by earlier kernels), with an event after the
  prelude and after each layer (§6.5).
- Write-through runs on the Program's transfer stream. Every write shares that stream and every
  restore the restore stream, so each list completes in order.
- Stream priorities are not set. The design gave restores the highest and write-through the
  lowest priority; measurement has not shown the need.

### 11.2 KV page transfer

Device plane order is unchanged: one page is `planes` contiguous slices, 64 planes for 27B INT8
plus MTP. The Host slab layout packs a block's Main page with its Host page layout and the backend
page at the next 4 KiB boundary, so a transfer is a pure copy with no requantisation, and restore is
bit-exact for every KV profile.

Implemented: the copy engines. Runs of consecutive device pages whose slab records advance by one
constant pitch (within one pinned chunk and the device's maximum copy pitch) move as one
`cudaMemcpy2DAsync` per plane; a restore orders the planes by model layer.

Not implemented, kept as options if restore bandwidth or SM interference ever matters: an SM copy
kernel reading pinned slabs through mapped pointers (16 B vector accesses, a capped grid so other
lanes' decode keeps its SMs, layer-major work items), and `cudaMemcpyBatchAsync` over coalesced
runs. The design's acceptance bar was ≥ 85 % of the contiguous `cudaMemcpyAsync` peak for 2 MiB
copies in each direction; the copy-engine path reaches 24–27 GB/s on the measuring machine's
PCIe 5.0 x8 link.

### 11.3 Other rules

- A snapshot image copy is a set of Device copies over the image's parts (48 SSM + 48 conv +
  hidden + ring), about 0.25 ms for a 27B DFlash2 image.
- CUDA Graph keys, decode graphs and block-table publication are unchanged. Prefill is not
  graphed.
- All cache bookkeeping is host-side O(1) or O(path). Index arenas are reserved at startup from
  `M`, `D` and `host_cache_bytes`.

### 11.4 Cost coefficients

Admission choice (§6.2) and Host valuation (§9.3) use the Engine's context-cost model: prefill
chunk, token and attention-pair coefficients measured for the artifact's prefill signature, and the
Host-to-Device transfer coefficients of the RTX 5090 class (a local preset can replace them).
Uncalibrated terms keep the index's generic defaults. No separate calibration runs at startup.

---

## 12. Optional features (each gated by its own measurement)

1. **Layer-pipelined restore** (implemented, §6.5). The first prefill pass's layer `l` waits on
   the restore's event for layer `l` (`cudaStreamWaitEvent`), so the restore overlaps compute; the
   admission itself never waits for the copies.
   - Gate: TTFT reduction ≥ 30 ms p50 on host-restore requests of ≥ 50 k tokens.
2. **In-flight prefix coalescing** (not included). A FIFO head whose prompt shares more tokens
   with a lane still prefilling than the cache can serve would wait for that lane's snapshot at
   the divergence (the block boundary `⌊d/64⌋·64` below the divergence point `d`, added to the
   lane's remaining plan as an exact tap) instead of prefilling the shared prefix again. It
   requires quoting a waiting request while another request prefills; the Engine attempts
   admission only while no request is in staged prefill, so a follower is quoted after the
   leader's prefill and resumes from the leader's deepest snapshot on the shared path (its
   structural taps, prompt-tail or ladder snapshots). It becomes useful only with concurrent
   staged prefill, which is a separate scheduling change (§16.2).
   - Gate: total prefill tokens reduced on the concurrent-shared-prefix trace, with no TTFT
     regression for other requests.
3. **Output-boundary snapshots** (not implemented). The Frontend's output parser signals structural boundaries
   (after `</think>`, before `<tool_call>`), and the Program snapshots the state at that exact
   committed frontier.
   - Without speculation: D2D of the active slot at the round boundary.
   - With speculation: a two-segment ReplaySSM Fold, `S0 → slot` over the first `j` records, then
     `slot → active` over the rest. This is bit-identical by the Fold contract. Conv and ring taps
     are handled as in §7.3.
   - Gate: reuse gain on thinking-stripped agent traces.
4. **KV-write elision** (not implemented) for the recomputed region (§7.5): map existing pages and send those KV
   writes to a scratch page.
   - Gate: measurable page savings under pressure.

---

## 13. Test specification

The test design follows. §13.1 and §13.3 are implemented (`ninfer_prefix_cache_index_test`,
`ninfer_qwen3_5_hybrid_prefix_real_test`, see Status); §13.2 applies to the unimplemented Ops;
§13.4 was the acceptance plan, and §16 records what was measured.

### 13.1 Host-only unit tests (`tests/test_prefix_cache_index.cpp`)

- **BlockTree**
  - insert/match with exact tokens and extra keys;
  - a forced constant `lookup_hash` must not produce a false hit;
  - different media digest with the same tokens must miss;
  - tail snapshot matches on exact tail tokens only;
  - `F ≤ n − 1` is enforced;
  - duplicate snapshot publication is a no-op.
- **Liveness**
  - `live_snapshots_below` stays correct under interleaved publish, evict and subtree deletion;
  - losing the last copy deletes exactly the subtree's snapshots.
- **Device LRU**
  - tail-first order after release;
  - hits move the whole path to MRU;
  - backed entries are consumed before unbacked;
  - pinned entries are never returned.
- **Host GDSF**
  - hand-computed cases: two siblings, nested ancestor/descendant, frequency;
  - dead KV is reclaimed before any snapshot;
  - the victim's exclusive path is freed;
  - `L` is monotonic.
- **Supersession and admission**
  - a lineage's previous snapshot is superseded, and a hit retains it again;
  - a `Boundary` snapshot is superseded only while no other conversation continued below it;
  - Device slots go to superseded owners, then to the least valuable Device-only owner for a new
    snapshot worth as much, then to the least recently hit Host-backed owner;
  - a Host image write displaces only snapshots worth no more than the new one.
- **HostSlabPool**
  - exhaustion, reuse and chunked allocation;
  - a slab never crosses chunks.
- **TapPlanner** (property tests over random prompts and boundaries)
  - all taps 64-aligned, in `(F_s, n−1]`, outside Vision spans;
  - deduplicated against the path;
  - priority order and budget respected;
  - the ladder stays within about log2(n/G) + 1 taps;
  - deterministic.
- **Admission arithmetic**
  - candidate fallback order;
  - Blocked vs Infeasible classification;
  - backfill proof on constructed states;
  - quote staleness via `cache_epoch`.

### 13.2 Op qualification (op-development.md contract, real 27B and 35B-A3B shapes)

- **GDN tap and phase**
  - FP64 naive-recurrence oracle at every tap, phases 0..63, T in {1, 63, 64, 65, 2048, 4096};
  - **bitwise** equality of each tap with a truncated call on the same phase;
  - `phase > 0` with zero taps matches the unpadded oracle within the existing GDN criterion;
  - `phase = 0` is bit-identical to the pre-change kernel.
- **Conv, hidden and ring taps**: exact byte comparison against a host reference gather, including
  `p − unit_begin < 3` and ring wrap (`p > S`, `unit length > S`).
- **`paged_kv_transfer`**
  - exact bytes for BF16, INT8, FP8, NVFP4 and K8V4, with Main+MTP and Main+DraftFull bundles,
    random page IDs, both directions;
  - bandwidth ≥ 85 % of the contiguous peak;
  - a concurrent-decode interference measurement is recorded.

### 13.3 Real-artifact integration (the design's scenario set)

Each scenario runs for `--kv-dtype` in {bf16, int8, fp8, nvfp4, k8v4} × backend in {none, mtp,
dflash2}. Scenarios marked "full matrix" run for every combination; the rest use int8 × {none,
dflash2}.

| scenario | assertion |
|---|---|
| resume-equivalence (full matrix) | resumed vs root: first-token logits within tolerance; greedy top-1 agreement over 256 tokens ≥ the chunk-decomposition baseline (root with a different `--prefill-chunk`); CausalScoring NLL difference within the same baseline |
| multi-turn exact endpoint | turn k+1 reuses `F_end(k)` exactly, and the tail was COW'd |
| re-rendered history (thinking stripped) | reuse ≥ `floor64(n_k − 1)` |
| edit deep message | reuse = the deepest ladder tap below the edit |
| shared preamble across 8 sessions | device pages for the preamble counted once; all 8 hit |
| concurrency 8, identical prompts arriving together | correct outputs; no double restore of a Filling node |
| host-only restore (D snapshot slots evicted) | same outputs as device resume; `Restoring` observed |
| cancellation mid-prefill | taps and completed blocks published; next identical request reuses them |
| Vision: same image vs different image, identical tokens | hit vs miss; no resume inside a span |
| active entitlement | cache and admissions never take an active request's reserved pages; backfill borrowers leave the head admissible |
| host cache 0 | device-only cache works; unbacked eviction deletes subtrees correctly |
| `Disabled` request | tree and snapshot counts unchanged |
| protocol usage | `cached_tokens` / `cache_read_input_tokens` equal `reuse_tokens`; streaming `message_start` exact |

Fault injection covers:

- a failed host allocation at startup, which is a clear startup error;
- a forced CUDA error in a transfer job, which gives Engine-wide failure and cleanup with no leaked
  pins (checked by `prefix_cache_stats` after restart);
- `compute-sanitizer --tool memcheck` on one concurrency-8 scenario, for the lifetime claims.

### 13.4 Performance acceptance plan

The baseline is the replaced checkpoint-catalog cache, run on identical traces and flags, with
`--host-cache-mib` equal in both runs.

| metric | requirement |
|---|---|
| prompt tokens served from cache, agent trace (README workload) | ≥ baseline, target ≥ +5 points |
| admission planning time (quote + admit, host) | p99 < 1 ms (baseline: up to 250 ms allowance) |
| TTFT p50 / p95 on the agent trace | lower than baseline; report the breakdown (planning, restore, prefill) |
| cold-request prefill tok/s | within ±1 % |
| decode tok/s at c = 1, 2, 8 (with and without write-through active) | within ±1 % |
| tap overhead per 4 k-token chunk | < 0.5 % of chunk time |
| restore bandwidth | ≥ 85 % of measured PCIe peak |
| device memory | shared-preamble trace: ≥ 7× fewer preamble pages at c = 8 |


---

## 14. Replaced code and configuration

### 14.1 Removed with the checkpoint-catalog cache

- `src/runtime/engine/context_cache/`: `resource_manager.h`, `materialization_planner.h`,
  `shared_capture_planner.h`, `resource_search.h`, `context_portfolio_value.h`,
  `materialization_budget.h`. `context_cost.*` is reduced to the machine model.
- `src/models/qwen3_5/program/planning/`: `pressure.cpp`, `pressure_planner.{h,cpp}`,
  `checkpoint_recovery.cpp`, `resource_projection.h`, `rebuild_work.h`. `request_plan.cpp` is
  reduced to base planning.
- `src/models/qwen3_5/program/transactions/`: `materialization.cpp`, `capture.cpp`;
  `commit.cpp` keeps model commit and terminal settlement.
- `src/models/qwen3_5/program/prefix_identity.*`, the Frontend's `PreparedContextCache` and the
  output session's prefix-execution tracking.
- `src/runtime/contract/resources.h`: the pressure, target, candidate, checkpoint-ref, capture and
  recovery types.
- Engine options and `ninfer-serve` flags: `device_state_slots`, `host_state_slots`,
  `host_kv_capacity_bytes`, `max_private_continuations`, `max_shared_prefixes`,
  `max_long_anchors_per_continuation` (`--device-state-slots`, `--host-state-slots`,
  `--host-kv-mib`, `--max-private-continuations`, `--max-shared-prefixes`,
  `--max-long-anchors-per-continuation`).
- Request-log fields of the catalog: selection paths `private_*`/`shared_stable_prefix`, capture,
  pressure, search and Host State/KV counters (schema v22).
- Tests of owners, catalogs and pressure targets (`test_resource_manager`,
  `test_materialization_budget`, the catalog scenarios of `test_engine_prefix_real`).

Kept unchanged: the Scheduler and its admission policy (FIFO protection and backfill), the KV
address-space and logical-page stores, and the StateImage store. The logical-page Host replicas,
`HostKVArena` and the Host StateImage pool no longer have a product path (the Host tier is the slab
pool of §5.4); removing them is a separate cleanup.

### 14.2 Configuration surface

```cpp
struct ContextCacheOptions {
    bool enabled = true;                                    // false: --no-prefix-reuse
    std::size_t host_cache_bytes = kDefaultHostCacheBytes;  // --host-cache-mib (§5.4)
    std::optional<std::uint32_t> device_snapshot_slots;     // --device-snapshot-slots
    std::optional<std::uint32_t> max_new_taps;              // --cache-taps-per-request
    std::optional<std::uint32_t> tap_ladder_tokens;         // --cache-tap-ladder (G)
    std::optional<std::uint32_t> tap_min_gap_tokens;        // --cache-tap-min-gap
    std::filesystem::path persistent_file;                  // --prefix-cache-file (§5.5)
    std::string persistent_identity;                        // set by the product binary
};
```

Engine construction (`normalize_engine_options`) resolves every unset value, and
`Engine::options()` reports the effective ones. With `C = max_concurrency`, `chunk =
prefill_chunk` and a Host tier present when the budget is nonzero:

| value | default | reason |
|---|---|---|
| `host_cache_bytes` | 8 GiB (`kDefaultHostCacheBytes`); 0 disables the Host tier | the replaced cache's Host KV default |
| `device_snapshot_slots` | `C + 1` with a Host tier, `C + 2` without | one resident snapshot per live conversation, plus staging; without Host the slots are the only snapshot storage |
| `max_new_taps` | 8 with a Host tier, 2 without | without Host, taps would evict other conversations' snapshots |
| `tap_ladder_tokens` | `max(4096, 2·chunk)` | ladder taps land on chunk boundaries; a finer ladder only duplicates them |
| `tap_min_gap_tokens` | `max(1024, chunk)` | the same |
| KV capacity (`ninfer-serve`) | `--kv-capacity auto` unless given (`--max-context` with `--no-prefix-reuse`) | free VRAM becomes Device block cache |

With `enabled = false` the cache holds no Host tier, snapshot slots or taps, every request starts
from the root, and the snapshot, tap and file options are rejected.

---

## 15. Risks and mitigations

| risk | mitigation |
|---|---|
| Exact taps cost one prefill split each | Only semantic boundaries are exact (explicit, generation opener, structural); the prompt tail and ladder are flexible and cost no split. The proximity rule leaves one split per chat request, and none after an endpoint resume. The zero-split GDN tap (§7.2–§7.3) removes the rest if measurement shows it matters. |
| Snapshots crowd KV out of host memory | GDSF weighs bytes against recompute seconds on the same scale for both. Dead-KV sweep runs first. |
| Write-through competes with decode | Copy engines only, batched when a lane releases its blocks; decode rounds/s were unchanged with prefetch and write-through active (§16.3). Write-through can be throttled when decode is active without changing correctness. |
| Unbacked device eviction deletes subtrees | Only happens with no or full host cache. Backed entries are always preferred. |
| Large pinned allocations | Chunked allocation; the startup ledger reports the resolved size. |
| Session keys are ignored | Content addressing reproduces chain reuse; the protocol surface is unchanged. |

---

## 16. Decision record

The measurements here were taken while the design was developed, on a Windows build of NInfer
that carried both caches (one RTX 5090, CUDA 13.4, Qwen3.8-27B NVFP4). The Host-to-Device link of
the measuring machine runs at PCIe 5.0 x8 (the GPU supports x16), so restores and prefetches move
about 25 GB/s. That build also scheduled differently from this Engine (it admitted requests into
concurrent staged-prefill lanes and did not backfill past a blocked head), so its numbers show
what each decision did there, not how this implementation compares with the replaced cache. A
change that did not improve its target metric, or that regressed another, was reverted and is
recorded here with the reason.

### 16.1 Against the checkpoint-catalog cache

`ninfer-serve --max-context 32768 --max-concurrency 2 --kv-dtype int8 --kv-capacity auto
--prefill-chunk 2048 --host-cache-mib 12000`, a fresh server per workload; the development build
selected the replaced cache with a switch:

| workload | checkpoint catalog | hybrid |
|---|---|---|
| 4 conversations × 6 turns, thinking, shared ~5.9K-token system prompt: prompt tokens from cache | 83.0% | 90.9% |
| same: TTFT p50 / p90 / mean | 0.177 / 0.962 / 0.331 s | 0.149 / 0.283 / 0.200 s |
| same: TTFT median of later turns | 0.173 s | 0.129 s |
| 8 distinct ~11.4K-token prompts overflowing the Device pool, revisited: tokens reused on revisit | 0 of 8 | ~11,400 of each |
| cold ~8.6K-token prompts: server prefill / decode tok/s (median of 6) | 10.1k / 84.9 | 10.2k / 84.6 |
| 64-token prompt, 2048 output tokens: decode tok/s | 86.3 | 86.0–86.1 (86.1–86.3 with `--kv-capacity 65536`) |

With `--spec dflash2 --draft-tokens 7` and `--spec mtp --draft-tokens 3` the conversation workload
served 90.8% and 90.9% of prompt tokens from cache with no errors.

### 16.2 Decisions while building

1. **Built beside the catalog, then made the only cache.** The design was first built as a
   second mode, reusing the active-execution machinery and replacing only admission-source
   selection, retention, capture and pressure. Since it served every workload measured at least
   as well (§16.1, §16.3), it replaced the catalog: the catalog's policy code, options and tests
   were removed, and scheduling, including backfill (§6.3), kept its existing contract.
2. **Snapshots copy the committed state at chunk boundaries; the zero-split GDN tap was not
   built.** A tap copies the lane's StateImage into a snapshot slot (about 0.25 ms D2D), so only
   exact taps split a prefill chunk, at about 15 ms per split on 27B. §7.2–§7.3 remain the design
   for removing that split if measurement shows it matters.
3. **Prefill ignores the chat template's rewrite frontiers.** The replaced cache split prefill
   after each assistant header, after `<think>` and after the reasoning close for its rewrite
   checkpoints. The prefix cache captures nothing there, so a short chat turn prefills in two
   passes (to the opener, then the rest) instead of four (§7.1).
4. **No opener tap after an endpoint resume.** On a tool-calling agent trace (27B NVFP4, INT8 KV,
   `--preserve-thinking`) the opener split cost about 15 ms of a ~70–130 ms turn TTFT. A request
   that resumed from an endpoint proves its client echoes generated turns token for token, so its
   own endpoint serves the next turn (§7.1).
5. **Free VRAM becomes Device block cache.** Removing the `C·L` clamp and defaulting
   `--kv-capacity` to `auto` makes the pool several times larger. At the replaced cache's pool size
   decode is within noise; the automatic pool costs about 0.2–0.35% decode throughput, attributed
   by the capped runs to KV placement across the larger pool, not to per-round cache work (§16.1).
6. **One pinned Host slab pool, pinned in chunks.** KV blocks and snapshots share one pool with no
   fixed split (§5.4). A single very large pinned allocation can fail or stall (observed under
   Windows WDDM), so the pool is pinned in chunks of at most 4 GiB; strided copy runs never cross
   chunks, after a 52 GB Host tier produced an invalid `cudaMemcpy2DAsync` that aborted the server.
7. **In-flight prefix coalescing was left out** (§12.2). The development build admitted
   requests into concurrent staged-prefill lanes, and there, with `--max-concurrency 4`, four
   requests arriving together with a new ~13.9K-token system prompt served 74.9% of prompt tokens
   from cache (the three followers reuse 13,888 tokens each) instead of 11.0%; mean TTFT fell
   from 3.52 s to 1.48 s and the burst's wall time from 5.56 s to 1.80 s. The Engine here admits
   only while no request is in staged prefill, so followers already wait for the leader's
   prefill and then resume from its snapshots; the mechanism could never run and was removed.
8. **Restores never hold the admission, and the first prefill pass waits layer by layer** (§6.5).
   Copies run at 24–27 GB/s. Revisits of 8 ~11.4K-token prompts under two clients, each restoring
   its 147 MiB state image, have a median TTFT of 45.7 ms, against 60.0 ms when the admission
   waited for the copy. Revisits of two alternating ~90K-token conversations, each restoring
   2.6–2.8 GiB, have a TTFT of 268 ms with a short question (was 301 ms) and 1186 ms with a
   3.4K-token tool output (was 1208 ms). An Nsight Systems trace of a 44K-token revisit (1.49 GiB
   restored in 60 ms) shows the pipelining at work: the first prefill pass runs its
   linear-attention layers at once and waits ~2.2 ms at each attention layer for its KV, finishing
   as the copy does. The rest of that TTFT was not the restore: the 45 new tokens ran as four
   passes over the model, split at exact snapshot and template boundaries (~12–15 ms each), which
   decision 3 removed, and the prompt-attention kernel took ~2.3 ms per attention layer over the
   44K context in the one pass that used it.
### 16.3 Changes from a production agent log (2026-09-26)

A production request log (889 requests over 70 min, 27B NVIDIA NVFP4 + DFlash2, INT8 KV, C=2,
`--host-cache-mib 52000`, 247K-token Device pool) of an agent harness with about ten concurrent
conversations (17-tool subagents of 30K–208K tokens and a 20-tool orchestrator) served 94.9 % of
prompt tokens from cache but prefilled 5.02M tokens (994 s). It showed three problems:

1. **Host eviction kept the wrong snapshots.** The Host tier was full from minute 4. The GDSF value
   of a snapshot counted only aligned ancestors, so tailed endpoints were priced as full
   re-prefills from the root: about 100 stale chain endpoints (~19 GB) stayed, while shared
   system/tools taps and fresh endpoints went first. 68 turns re-prefilled 758K tokens because
   their resume point was stuck at an old aligned tap while the suffix grew to 42K tokens, and 24
   subagent turns prefilled from the root next to a cached 19.8K-token shared prefix.
2. **Only one lane ran for half the wall time.** Two 150K–200K contexts do not fit in the Device
   pool together, and the development build admitted strictly in FIFO order (it had no backfill):
   the average decode batch was 1.27 of 2.
3. **Every turn restored its whole context from the Host tier**: 2.59 TB in total, 2.7 GB per turn
   (median), while the FIFO head had usually been waiting seconds to minutes.

| change | section | A/B result | status |
|---|---|---|---|
| Superseded snapshots evicted first; valuation against the nearest retained ancestor | §9.3 | prompt tokens prefilled −10.8 % (seed 42) and −3.4 % (seed 43); main-session turns that re-prefilled the whole prompt 3 → 1 (seed 42); TTFT without queue wait −6 % and −3 % | kept |
| Persistent backfill proof with a 16K-token growth reserve | §6.3 | targeted scenario: four 12K-token requests queued behind a blocked 90K-token head finish in 1.6 s instead of 35–36 s; the long request already decoding takes 4.7 s (12 %) longer, about the borrowers' prefill time, and the head starts that much later (its answer ends at the same time); no answer is cut short. Agentic workload: 0–9 backfills per run, effect within sampling noise | not adopted in that build (below); this design keeps the Engine's backfill instead |
| Prefetch of the blocked head's Host-only blocks | §6.6 | admission Host restores −22 % (seed 42, 62.4 → 48.6 GB) and −25 % (seed 43, 68.4 → 51.3 GB), 10–11 GB prefetched per run; TTFT without queue wait −3 % and −6 %; decode rounds/s unchanged (57.8 vs 57.8, one request decoding) | kept |

The prefetch was measured on top of the backfill proof. The build that combined all three against
the base: at the 16 GB Host tier, prompt tokens prefilled −13.9 % (seed 42) and −2.7 % (seed 43),
TTFT without queue wait 0.82 → 0.74 s and 0.75 → 0.73 s; at 52 GB (seed 42), prompt tokens
prefilled −9.2 % (623,853 → 566,189), served from cache 89.1 → 90.1 %, no turn prefilled from the
root beside cached blocks (3 before), TTFT without queue wait 0.68 → 0.60 s (−12 %), workload wall
time −7.8 % for the same output volume (130K tokens). That 52 GB arm's mean TTFT including queue
wait rose from 2.1 to 2.8 s: in a closed loop, turns that finish sooner re-queue sooner, and the
queue-wait increase was spread over requests that no change touches. Backfill triggered at most
twice in it. The two kept changes together (without the backfill proof) ran the pressure workload at
scale 0.3 (54 requests, no failures, 1.2 GB prefetched); that combination was not A/B tested on its
own.

**Backfill.** That build let an answer grow past its admission reservation, so a borrower could
take the pages a long answer would grow into, and the proof needed a growth reserve that left room
only for small borrowers. Where it passed, it moved latency from the short requests to the long
ones rather than removing it, so it was not adopted there. The design here keeps the Engine's
existing backfill: an admitted request never grows past its entitlement, so the proof of §6.3 needs
no reserve, and the trade-off is the one the Engine already makes for every cache configuration.

Measurement: an agentic replay harness of the development tree (3 interleaved sessions and 11
subagents, 130 requests; seeds 42 and 43) on the production flags with the Device pool and Host
tier scaled down to that workload (`--kv-capacity 160000`, `--max-context 160000`,
`--host-cache-mib 16000`). Each change is compared with the build before it at a 16 GB Host tier,
where eviction matters, and all three with the base at 52 GB, where it rarely does. The policy does not depend on the tier size (`--host-cache-mib` 0 to 230 GiB): snapshot count
is bounded by `host_cache_bytes / image_bytes` (about 1,300 at 230 GiB), and victim selection is a
scan over snapshots.

Run notes:

- Seed 42, first run: the superseded-first arm and the second half of the prefetch arm ran under
  outside CPU load; tokenization, which none of these changes touch, was up to 2.3× slower, and
  their decode host time per round (204 and 338 µs against 80 µs) was not the builds'. Both arms
  were rerun with per-process CPU logging and the rerun values are the ones above; their
  tokenization matched the base.
- Backfill needs a blocked FIFO head with a shorter request behind it while a long answer decodes.
  The workload's lock-step sessions rarely queue that way (one lane decoding with requests waiting
  for 3–13 % of a run, against 50 % in the production log), so a targeted scenario measures it: a
  100K-token request answering with a count to 2,400, then a 90K-token request, then four 12K-token
  requests, greedy, `max_tokens` 64000, two repetitions per build, identical to within a second.
  In seed 43 the two long requests that borrowers overtook waited 6 and 8 s longer (23 s for one in
  the prefetch arm) while the borrowers saved about 1 s each; output lengths differ between arms,
  so this is not attributable, but it is worth watching in production.

### 16.4 Fixes from operation (2026-09-27)

None of these were A/B tested; each is a correctness, behavior or log fix.

| change | section | reason and evidence |
|---|---|---|
| A Host tier smaller than the saved file restores its most valuable snapshots and only their paths (file format 3, tables before slab bytes), with a startup warning | §5.5 | Format 2 restored every block before any snapshot, in file order, and still reported the file restored. For the production file (15,807 blocks and 98 snapshots of 91–92 slabs of 2.06 MiB, 49.97 GiB, saved at `--host-cache-mib 52000`), any tier below about 32,800 MiB filled with blocks and kept no snapshot, so nothing was resumable; between that and about 51,200 MiB the snapshots kept depended on file order |
| A Host tier that keeps none of the file's snapshots warns `prefix cache not restored` instead of `partly restored … the most valuable snapshots were kept` | §5.5 | A restore keeps only the blocks its snapshots resume through, so keeping no snapshot restores nothing, yet the line claimed the most valuable snapshots were kept; it now names the tier the file needs and that the save at shutdown replaces the file |
| A Host restore's per-layer events are looked up immediately before the first prefill chunk, not before the MTP bridge | §6.5 | Reported by funguf (Wallawalla47/ninfer-custom pull request 2) with an AddressSanitizer trace and 9 production crashes in `cuStreamWaitEvent` in one day (`--spec mtp`, Host tier): the view was taken before the MTP bridge, whose KV commit polls the cache, and a batch that had landed meanwhile was freed under it, so the first chunk waited on whatever the freed memory held. The report's fix copied the handles into the prefill context; looking them up at the chunk instead never hands CUDA an event the cache has recycled, needs no allocation, and skips the waits once the copies are complete. DFlash2 and no-speculation lanes run no bridge and were not exposed |
| The chunk function takes the restore's layer events itself (`PrefillContext::take_layer_ready`), so program code never holds the view | §6.5 | Hardening. A deterministic test of the fix above was asked for, but the retired-batch path is timing-dependent and a dangling view reading a recycled handle can pass a token-equality check, so a test would need a product seam or AddressSanitizer. Moving the lookup inside the chunk call instead leaves no place for program code to take the view early |

### 16.5 Changes from upstream's TTFT campaign (2026-09-30)

Upstream's public-HTTP TTFT campaign (`tools/bench/ttft`, its cache profiles translated to this
cache's options at equal Device state slots and pinned Host bytes) ran against the upstream port of
this cache in WSL2 on the same RTX 5090. Five of its cases resumed a conversation from the root, or
far behind its newest snapshot, where the replaced cache resumed from the conversation's own
checkpoint. Measured with one sample per case on the upstream port; the eviction code is the same
in both trees.

| change | section | result | status |
|---|---|---|---|
| A sequence supersedes the snapshot it resumed from before its new snapshot takes a slot | §9.2, §9.3 | `session-alternating` (2 Device snapshot slots, no Host tier; two sessions sharing a 4K prefix): A2 and B2 TTFT 305 → 22 ms (replaced cache 49 ms). Before, each session's endpoint took the slot of the other session's endpoint, not of the shared tap it had resumed from | kept |
| Device-only slots evicted by GDSF priority, only for a new snapshot worth as much, and before a Host-backed slot gives up its Device copy | §9.2 | `resume-after-interference-device` (same profile; two short requests between a 7.7K conversation's turns): resume 600 → 22 ms (replaced cache 78 ms); `resume-after-interference-catalog` 600 → 22 ms (632 ms) | kept |
| A snapshot image goes to the Host only by displacing snapshots worth no more; protocol-automatic markers published as ordinary taps rather than `Boundary` snapshots | §7.1, §8.3, §9.3 | `private-state-working-set-shift` (2 Device slots, 384 MiB Host, DFlash2; four 2K conversations in two phases): the second phase's continuations 204–215 → 40–49 ms (replaced cache 48 ms). New images had displaced the snapshots the waiting conversations resume from | kept |
| A `Boundary` snapshot is superseded like any other while no other conversation has continued from it | §9.3 | same case: B1 207 → 39 ms. Each conversation kept its system-block boundary beside its newer snapshot, so the other conversation's snapshot lost its slot. All eight continuations now take 20–40 ms (replaced cache 48–56 ms) | kept |
| A deeper tap supersedes the sequence's previous tap of the same prompt (the endpoint does not supersede the prompt's last tap) | §9.3 | `session-alternating-64k-host-swap` (4.5 GiB Host for two 64K sessions whose KV takes 4 GiB): A2 2381 → 176 ms, B2 164 → 113 ms (replaced cache 242 and 171 ms). Valued against the tap below it, each ladder tap was worth more per byte than the prompt's last tap, so GDSF kept the four ladder taps and evicted the last tap until A resumed from 48K | kept |
| Flexible taps (prompt tail, ladder) as a separate snapshot kind, evicted before every other retained snapshot | §9.3 | broke a regenerate-after-edit resume: with DFlash2 and a 1 GiB Host tier, the prompt-tail tap a request diverging late in the previous prompt needed went first (`restore-exact-dflash2` real test) | reverted |
| The same kind starting from half a hit in its GDSF value | §9.3 | no change on the two cases it targeted | reverted |
| Host admission by value (`F·C/Z`) instead of priority `H` | §9.3 | meant for small tiers, where `L` rises by a whole snapshot's value per eviction and almost any new image outranks the rest: stale snapshots worth more per byte then kept the tier from new ones, and the index test's Host backup found no slabs | reverted |

The full campaign after these changes (3 samples per case) gives a geometric-mean TTFT ratio of
0.658 against the replaced cache over 259 (case, role) observations: 172 faster by more than 10%, 18
slower. The slower roles are not eviction. The 55K rotation (six sessions resumed round-robin with
Device KV for about four) restores every session from Host, about 101 ms each, because Device KV is
LRU over a cyclic working set larger than the Device; the replaced cache keeps three sessions
resident (72 ms) and restores the other three in 125–245 ms, so its per-round mean is higher (128
against 99 ms) while three roles are faster. A repeat of an identical prompt prefills its last few
tokens (about 20 ms), where the replaced cache samples from the hidden state stored with its
checkpoint (6 ms): the prompt's last tap sits before the generation opener, and a snapshot at the
prompt's exact end would cost a whole state image per request for byte-identical repeats only. The
other slower roles are arrival races between concurrent requests and timing effects in profiles that
run with the cache off.
