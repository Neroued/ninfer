#pragma once

// Content-addressed pinned-host store for paged KV payloads and recurrent-state anchors.
//
// Pages are keyed by 64-bit chained content hashes supplied by the caller; a key therefore
// certifies the full token prefix it terminates, which is exactly the causal validity a KV
// page needs. Segments (a run of text/backend pages plus one anchor) are the LRU eviction
// unit; eviction never punctures a stored chain because removal is whole-segment.
//
// Single-threaded by contract: all calls happen on the Program's serve thread. Copies are
// asynchronous on the caller's stream; the caller synchronizes before publishing or reading.

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer {

struct KvHostCopySlice {
    void* device_ptr  = nullptr;
    std::size_t bytes = 0;
};

class KvHostCache {
public:
    enum class PageKind : std::uint8_t { Text, Backend };

    struct Config {
        std::size_t budget_bytes       = 0;
        std::size_t text_page_bytes    = 0;
        std::size_t backend_page_bytes = 0;
        std::size_t anchor_state_bytes = 0;
    };

    // NVCC: nested aggregates stay NSDMI-free; construct with {} at every use site.
    struct AnchorMeta {
        std::uint32_t frontier;
        std::uint32_t backend_tokens;
        std::int32_t rope_delta;
    };

    struct Stats {
        std::uint64_t stored_segments  = 0;
        std::uint64_t stored_pages     = 0;
        std::uint64_t stored_bytes     = 0;
        std::uint64_t hit_requests     = 0;
        std::uint64_t hit_tokens       = 0;
        std::uint64_t restored_bytes   = 0;
        std::uint64_t writeback_bytes  = 0;
        std::uint64_t evicted_segments = 0;
    };

    explicit KvHostCache(const Config& config);
    ~KvHostCache();

    KvHostCache(const KvHostCache&)            = delete;
    KvHostCache& operator=(const KvHostCache&) = delete;

    [[nodiscard]] bool has_page(PageKind kind, std::uint64_t key) const noexcept;
    // Pins an already-stored page for the open staging transaction (refcount +1, released by
    // seal/abort). Returns false when the key is absent — the caller stages it instead.
    [[nodiscard]] bool pin_page(PageKind kind, std::uint64_t key);
    [[nodiscard]] std::uint32_t present_prefix(PageKind kind,
                                               std::span<const std::uint64_t> keys) const;
    [[nodiscard]] std::optional<AnchorMeta> find_anchor(std::uint64_t chain_key) const;

    // Save path. stage_page copies D2H only when the key is new; every staged call must be
    // followed by exactly one seal_segment or abort_segment. Staging fails (returns false)
    // when the budget cannot host the payload even after evicting every idle segment.
    [[nodiscard]] bool stage_page(PageKind kind, std::uint64_t key,
                                  std::span<const KvHostCopySlice> device_slices,
                                  cudaStream_t stream);
    [[nodiscard]] bool stage_anchor(std::uint64_t chain_key, const AnchorMeta& meta,
                                    std::span<const KvHostCopySlice> state_slices,
                                    cudaStream_t stream);
    void seal_segment(std::uint64_t chain_key, std::span<const std::uint64_t> text_keys,
                      std::span<const std::uint64_t> backend_keys, std::uint64_t index_page_key);
    [[nodiscard]] std::span<const std::uint64_t>
    find_anchor_keys(std::uint64_t index_page_key) const;
    void abort_segment();

    // Restore path. Keys must be present; H2D copies land on the caller's stream.
    void restore_page(PageKind kind, std::uint64_t key,
                      std::span<const KvHostCopySlice> device_slices, cudaStream_t stream);
    void restore_anchor_state(std::uint64_t chain_key,
                              std::span<const KvHostCopySlice> state_slices, cudaStream_t stream);
    void note_hit(std::uint32_t tokens);
    void touch_segment(std::uint64_t chain_key);

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    // Monotonic counter of store mutations (segment sealed or dropped). Lets plan caches skip
    // invalidation when a completion changed nothing a probe could see.
    [[nodiscard]] std::uint64_t mutation_epoch() const noexcept { return mutation_epoch_; }
    [[nodiscard]] std::size_t budget_bytes() const noexcept { return config_.budget_bytes; }
    [[nodiscard]] std::size_t used_bytes() const noexcept { return used_bytes_; }

private:
    struct Slot {
        std::byte* host;
        std::size_t bytes;
    };
    struct PageEntry {
        Slot slot;
        std::uint32_t refcount;
    };
    struct Segment {
        AnchorMeta meta;
        Slot anchor_slot;
        std::vector<std::uint64_t> text_keys;
        std::vector<std::uint64_t> backend_keys;
        std::uint64_t index_page_key;
        std::list<std::uint64_t>::iterator lru_position;
    };

    [[nodiscard]] std::size_t page_bytes(PageKind kind) const noexcept;
    [[nodiscard]] std::unordered_map<std::uint64_t, PageEntry>& table(PageKind kind) noexcept;
    [[nodiscard]] const std::unordered_map<std::uint64_t, PageEntry>&
    table(PageKind kind) const noexcept;
    [[nodiscard]] bool reserve_slot(std::size_t bytes, Slot* out);
    void release_slot(const Slot& slot) noexcept;
    void unpin_staged() noexcept;
    [[nodiscard]] bool evict_one_idle_segment();
    void drop_segment(std::uint64_t chain_key, Segment& segment) noexcept;
    static void copy_slices(std::byte* host, std::span<const KvHostCopySlice> device_slices,
                            std::size_t expected_bytes, cudaMemcpyKind kind, cudaStream_t stream);

    Config config_;
    Stats stats_;
    std::vector<std::byte*> chunks_;
    std::vector<Slot> free_slots_text_;
    std::vector<Slot> free_slots_backend_;
    std::vector<Slot> free_slots_anchor_;
    std::size_t next_chunk_offset_  = 0;
    std::size_t current_chunk_left_ = 0;
    std::size_t used_bytes_         = 0;
    std::unordered_map<std::uint64_t, PageEntry> text_pages_;
    std::unordered_map<std::uint64_t, PageEntry> backend_pages_;
    std::unordered_map<std::uint64_t, Segment> segments_;
    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> anchors_by_page_;
    std::list<std::uint64_t> lru_;

    struct StagedPage {
        PageKind kind;
        std::uint64_t key;
        Slot slot;
    };
    std::vector<StagedPage> staged_pages_;
    std::vector<std::pair<PageKind, std::uint64_t>> staged_pins_;
    std::optional<std::pair<std::uint64_t, Segment>> staged_anchor_;
    std::size_t carved_bytes_   = 0;
    std::uint64_t mutation_epoch_ = 0;
};

} // namespace ninfer
