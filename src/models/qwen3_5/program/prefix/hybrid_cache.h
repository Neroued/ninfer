#pragma once

#include "core/arena.h"
#include "core/startup.h"
#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"
#include "models/qwen3_5/program/storage/kv_store.h"
#include "models/qwen3_5/program/storage/state_store.h"
#include "models/qwen3_5/state/state_image.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Physical binding of the hybrid prefix index (docs/maintainer/hybrid-prefix-cache.md) to
// the Program's KV page stores, StateImage store and pinned Host slab pool. The index decides;
// this class holds the references behind its opaque ids and moves the bytes its decisions imply.
//
// - A block id owns one non-writer reference on a committed full Main page (and on the matching
//   backend page when a backend pool is enabled). Snapshot tails use the same id space.
// - A snapshot owns one immutable StateImage while its image is on the Device (the index reports a
//   device slot). Its Host copy lives in index-owned slabs, so dropping the Device copy releases
//   the StateImage and restoring it copies the slabs into a caller-reserved Device slot.
// - Host writes run on the transfer stream after the producer's queued work. Host restores run
//   on a dedicated restore stream so they overlap write-through traffic and compute.

struct HybridBlockPages {
    LogicalKVPageHandle text;
    std::optional<LogicalKVPageHandle> backend;
};

struct HybridCacheCounters {
    std::uint64_t admissions          = 0;
    std::uint64_t snapshot_hits       = 0;
    std::uint64_t reused_tokens       = 0;
    std::uint64_t blocks_inserted     = 0;
    std::uint64_t blocks_reattached   = 0;
    std::uint64_t blocks_duplicate    = 0;
    std::uint64_t taps_created        = 0;
    std::uint64_t taps_skipped        = 0;
    std::uint64_t endpoints_created   = 0;
    std::uint64_t host_image_writes   = 0;
    std::uint64_t host_block_writes   = 0;
    std::uint64_t host_image_restores = 0;
    std::uint64_t host_block_restores = 0; // prefetched blocks included
    std::uint64_t prefetched_blocks   = 0;
    std::uint64_t host_tail_restores  = 0;
    std::uint64_t host_write_bytes    = 0;
    std::uint64_t host_restore_bytes  = 0;
    std::uint64_t evicted_blocks      = 0;
};

// One layer of the model's forward pass, in execution order, as a restore sees it: whether it
// reads a full-attention layer's KV planes or a linear-attention layer's recurrent state, and its
// index among layers of that kind.
struct HybridRestoreLayer {
    bool attention      = false;
    std::uint32_t index = 0;
};

// Outcome of saving or loading the Host tier (--prefix-cache-file).
struct HybridPersistResult {
    bool ok = false;
    // Why nothing was saved or loaded (missing file, incompatible fingerprint, I/O error).
    std::string message;
    std::uint64_t blocks    = 0;
    std::uint64_t snapshots = 0;
    std::uint64_t bytes     = 0;
    double seconds          = 0.0;
    // Load only: the file's entries and the Host tier bytes they all take, against this tier.
    // A tier smaller than the file restores the snapshots it values most (plan_host_restore).
    std::uint64_t saved_blocks        = 0;
    std::uint64_t saved_snapshots     = 0;
    std::uint64_t required_host_bytes = 0;
    std::uint64_t host_bytes          = 0;
};

class HybridPrefixCache final : public runtime::prefix_cache::PrefixIndexBackend {
public:
    HybridPrefixCache(const runtime::prefix_cache::PrefixIndexConfig& config,
                      runtime::prefix_cache::TapPlannerConfig taps, LogicalKVPageStore& text_pages,
                      LogicalKVPageStore* backend_pages, StateImageStore& states,
                      qwen3_5::StateImageDevicePool& state_pool, const HybridHostLayout& host,
                      std::vector<HybridRestoreLayer> layers,
                      StartupPhaseScope* pin_progress = nullptr);
    ~HybridPrefixCache() override;

    HybridPrefixCache(const HybridPrefixCache&)            = delete;
    HybridPrefixCache& operator=(const HybridPrefixCache&) = delete;

    [[nodiscard]] runtime::prefix_cache::PrefixCacheIndex& index() noexcept { return *index_; }

    [[nodiscard]] const runtime::prefix_cache::PrefixCacheIndex& index() const noexcept {
        return *index_;
    }

    [[nodiscard]] const runtime::prefix_cache::TapPlannerConfig& taps() const noexcept {
        return taps_;
    }

    [[nodiscard]] HybridCacheCounters& counters() noexcept { return counters_; }

    [[nodiscard]] const HybridCacheCounters& counters() const noexcept { return counters_; }

    [[nodiscard]] bool host_tier() const noexcept { return !host_chunks_.empty(); }

    [[nodiscard]] const HybridHostLayout& host_layout() const noexcept { return host_layout_; }

    // Replaces the admission/eviction machine model (the Engine's calibrated context costs).
    void set_cost(const runtime::prefix_cache::CacheCostModel& cost);

    // ---- blocks -------------------------------------------------------------------------------
    // Inserts a committed full block. The cache takes references on `pages` only when the index
    // keeps them (a new node, or a host-only node regaining its Device copy); otherwise the caller
    // keeps its page private. The returned node is pinned once for the caller either way.
    [[nodiscard]] runtime::prefix_cache::InsertResult
    insert_block(runtime::prefix_cache::NodeRef parent, std::uint64_t lookup_hash,
                 std::span<const TokenId> tokens, std::uint64_t extra, HybridBlockPages pages);

    // Registers a snapshot tail bundle whose pages the caller hands over after its last write
    // (references taken now); returns its device id. `backend_columns` is the backend page's own
    // committed extent (MTP backend KV trails the text frontier by one token).
    [[nodiscard]] std::uint32_t register_tail(HybridBlockPages pages, std::uint32_t text_columns,
                                              std::uint32_t backend_columns);
    // Registers a snapshot tail bundle materialized as transfer destinations and filled by a
    // stream-ordered copy; the cache adopts the pages as its only references.
    [[nodiscard]] std::uint32_t register_tail_copy(HybridBlockPages pages, std::uint32_t columns);

    [[nodiscard]] const HybridBlockPages& block(std::uint32_t device_id) const;

    // ---- snapshot images ----------------------------------------------------------------------
    // Binds the immutable Device StateImage behind a published snapshot.
    void attach_image(runtime::prefix_cache::SnapshotRef snapshot, StateImageHandle image);
    [[nodiscard]] StateImageHandle image(runtime::prefix_cache::SnapshotRef snapshot) const;

    // ---- Host write-through (§9.3) --------------------------------------------------------------
    // Copies a Device-resident snapshot (image and tail) to Host slabs. Returns false when there is
    // no Host tier or no slabs could be freed. The snapshot is pinned until the copy completes.
    bool start_snapshot_host_write(runtime::prefix_cache::SnapshotRef snapshot,
                                   cudaStream_t producer, cudaStream_t transfer);
    // Copies every Device-only node of `nodes` to a Host slab in one batch. Nodes are pinned until
    // the copy completes. Stops early when the Host tier cannot free a slab.
    void start_block_host_writes(std::span<const runtime::prefix_cache::NodeRef> nodes,
                                 cudaStream_t producer, cudaStream_t transfer);
    // Publishes every completed Host write. Non-blocking.
    void poll();

    // Nodes and snapshots stay pinned while a Host write or a landing restore is in flight. When
    // those pins are all that stands between the pools and a request, waiting a few milliseconds
    // for the copies is the right answer, not a blocked admission or a truncated lease.
    [[nodiscard]] bool transfers_pending() const noexcept {
        return !pending_.empty() || !landing_.empty();
    }

    // Waits for every Host write and landing restore and publishes them.
    void drain();

    // ---- Host restores (one admission staged at a time) --------------------------------------
    // Opens a restore batch ordered after the producer's queued work (Device destinations may have
    // been read by earlier kernels). Destination pages are materialized transfer destinations; the
    // cache publishes them as its own references immediately and the index keeps their nodes
    // Filling until the batch lands.
    void open_restore(cudaStream_t producer);
    void restore_block(runtime::prefix_cache::NodeRef node, HybridBlockPages destination);
    void restore_tail(runtime::prefix_cache::SnapshotRef snapshot, HybridBlockPages destination);
    void restore_image(runtime::prefix_cache::SnapshotRef snapshot, std::int32_t device_slot);
    // Enqueues the batch in the order a forward pass reads it: a prelude (the snapshot tail, every
    // backend page, and the image's continuation hidden and DFlash state), then each model layer's
    // KV planes or recurrent state, recording an event after the prelude and after every layer.
    void submit_restore();

    [[nodiscard]] bool restore_open() const noexcept { return restore_.open; }

    [[nodiscard]] bool restore_pending() const noexcept { return restore_.submitted; }

    // Hands the submitted batch to the admission that staged it without waiting for the copies:
    // `consumer` waits for the prelude (activation's own Device work reads it), and the returned
    // ticket names the per-layer events the lane's first prefill pass waits on. The batch keeps
    // its nodes and snapshot pinned, and its nodes Filling, until poll() or drain() sees it land.
    [[nodiscard]] std::uint64_t land_restore(cudaStream_t consumer);
    // Per-layer events of a landing batch, in model layer order; empty once it has landed.
    [[nodiscard]] std::span<const cudaEvent_t>
    restore_layer_events(std::uint64_t ticket) const noexcept;
    // Orders `consumer` after every copy of a landing batch (a lane released before its first
    // prefill pass returns a state slot the batch may still be writing).
    void order_after_restore(std::uint64_t ticket, cudaStream_t consumer) const;
    // Waits for any submitted copies of the staged batch and returns every destination.
    void abort_restore() noexcept;
    // Hands a submitted batch of blocks to the cache without a consumer: a prefetch for a request
    // still waiting (spec §6.6). Its nodes stay Filling and pinned until poll() sees it land.
    void detach_prefetch();
    // Whether a prefetch batch may still be landing.
    [[nodiscard]] bool prefetch_landing() const noexcept;
    // Waits for every landing prefetch batch and publishes it.
    void settle_prefetch();

    // Releases every reference the cache holds. The index is rebuilt empty.
    void clear() noexcept;

    // ---- persistence (hybrid_persist.cpp) -------------------------------------------------------
    // Writes every Host-backed snapshot and the block paths it anchors on to `path` (through a
    // temporary file renamed into place). Requires no restore or write in flight.
    [[nodiscard]] HybridPersistResult save(const std::filesystem::path& path,
                                           std::string_view fingerprint) const;
    // Rebuilds a saved Host tier into this empty cache when the file's fingerprint and geometry
    // match. A mismatch or damaged file loads nothing; a smaller Host tier restores the snapshots
    // it values most and only their paths, reading nothing else. Reading a file whose header
    // matches publishes StartupPhase::PrefixCacheLoad to `observer`.
    [[nodiscard]] HybridPersistResult load(const std::filesystem::path& path,
                                           std::string_view fingerprint,
                                           const StartupObserver& observer);

    // PrefixIndexBackend
    void release_device_block(std::uint32_t device_id) noexcept override;
    void release_snapshot(runtime::prefix_cache::SnapshotRef snapshot) noexcept override;
    void drop_snapshot_device_image(runtime::prefix_cache::SnapshotRef snapshot) noexcept override;

private:
    struct PendingWrite {
        std::vector<runtime::prefix_cache::NodeRef> nodes;
        std::optional<runtime::prefix_cache::SnapshotRef> snapshot;
        cudaEvent_t done = nullptr;
    };

    // Copies between pages and Host slab records. A record's group is the pinned chunk holding
    // its slab: a strided transfer never joins records of two chunks.
    struct PageCopies {
        std::vector<DeviceKVPageHandle> text_pages;
        std::vector<std::byte*> text_records;
        std::vector<std::uint32_t> text_groups;
        std::vector<DeviceKVPageHandle> backend_pages;
        std::vector<std::byte*> backend_records;
        std::vector<std::uint32_t> backend_groups;

        void clear() noexcept {
            text_pages.clear();
            text_records.clear();
            text_groups.clear();
            backend_pages.clear();
            backend_records.clear();
            backend_groups.clear();
        }
    };

    struct RestoreBatch {
        bool open            = false;
        bool submitted       = false;
        std::uint64_t ticket = 0;
        std::vector<runtime::prefix_cache::NodeRef> nodes;
        std::optional<runtime::prefix_cache::SnapshotRef> tail;
        // The snapshot whose Host image lands in `image_slot`.
        std::optional<runtime::prefix_cache::SnapshotRef> image;
        std::int32_t image_slot = -1;
        PageCopies copies;
        PageCopies tail_copies;
        cudaEvent_t prelude = nullptr;
        // One per model layer; the last marks the whole batch.
        std::vector<cudaEvent_t> layers;
        // Pins a landing batch holds on its snapshot (tail or image source) until it lands.
        std::optional<runtime::prefix_cache::SnapshotRef> pinned_snapshot;
        // Copies blocks for a waiting request; no lane waits on its events.
        bool prefetch = false;
    };

    [[nodiscard]] std::uint32_t allocate_block_id(HybridBlockPages pages);
    void free_block_id(std::uint32_t id) noexcept;
    void retain(const HybridBlockPages& pages, std::uint32_t text_columns,
                std::uint32_t backend_columns);
    void release(const HybridBlockPages& pages) noexcept;
    void adopt_destination(const HybridBlockPages& pages, std::uint32_t columns);
    [[nodiscard]] std::byte* slab(std::uint32_t id) const noexcept;
    void add_copies(PageCopies& copies, const HybridBlockPages& pages, std::uint32_t slab_id);
    void enqueue_to_host(const PageCopies& copies, cudaStream_t stream) const;
    void enqueue_from_host(const PageCopies& copies, cudaStream_t stream) const;
    [[nodiscard]] cudaEvent_t take_event();
    void order_after(cudaStream_t producer, cudaStream_t consumer);
    void publish_write(PendingWrite& write);
    void recycle_events(RestoreBatch& batch) noexcept;
    void reset_restore() noexcept;
    // Marks a landed batch's copies resident and drops its pins.
    void finish_landing(RestoreBatch& batch);

    runtime::prefix_cache::PrefixIndexConfig config_;
    runtime::prefix_cache::TapPlannerConfig taps_;
    LogicalKVPageStore* text_pages_            = nullptr;
    LogicalKVPageStore* backend_pages_         = nullptr;
    StateImageStore* states_                   = nullptr;
    qwen3_5::StateImageDevicePool* state_pool_ = nullptr;
    HybridHostLayout host_layout_;
    // The Host slab pool is pinned in chunks of whole slabs: one very large pinned allocation
    // can fail or stall on Windows (WDDM) where several smaller ones succeed. A slab never
    // crosses chunks, and every copy record carries its chunk as its group, so a strided copy
    // run never spans two chunks (two records of different chunks would otherwise form a run).
    std::vector<PinnedHostBuffer> host_chunks_;
    std::uint32_t slabs_per_chunk_ = 0;
    cudaStream_t restore_stream_   = nullptr;
    std::vector<std::optional<HybridBlockPages>> blocks_;
    std::vector<std::uint32_t> free_blocks_;
    std::vector<StateImageHandle> images_;
    std::deque<PendingWrite> pending_;
    // Restore order within one forward pass, and the text KV planes each attention layer owns.
    std::vector<HybridRestoreLayer> layers_;
    std::size_t planes_per_attention_layer_ = 0;
    RestoreBatch restore_;
    // Batches handed to their lanes whose copies may still be in flight, in submission order (one
    // restore stream).
    std::deque<RestoreBatch> landing_;
    std::uint64_t next_ticket_ = 1;
    std::vector<cudaEvent_t> spare_events_;
    PageCopies write_scratch_;
    HybridCacheCounters counters_;
    std::optional<runtime::prefix_cache::PrefixCacheIndex> index_;
};

} // namespace ninfer::models::qwen3_5::detail
