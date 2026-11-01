#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/sparse_attention.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
inline constexpr std::int32_t kKvFp8QuantGroup  = 256;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    // Qwen4Exp: every attention layer also keeps the QSA index planes (ops::QsaIndexPlanes:
    // records BF16 [136,64,1,pages] and pooled block keys BF16 [32,64,1,pages]) in the same page
    // pool, so they share the K/V frontier, paging, prefix reuse and page copies.
    bool qsa_index = false;
    // Active KV in RAM (--kv-ram-active, docs/maintainer/active-kv-in-ram.md): the text pool's K/V,
    // scale and QSA record planes live in pinned, device-mapped Host memory; only the QSA block
    // keys stay in device memory. A persistent device region (ops::QsaStagingRegion) serves the
    // attention: `staging_cache_blocks` 4-token blocks per attention layer (0 = default, raised so
    // the region holds one layer of `capacity`) plus an overflow area for `staging_window` columns
    // of `staging_selection_groups` blocks each (ceil(max_selected / 4)). Requires qsa_index.
    bool kv_ram_active                    = false;
    std::int32_t staging_window           = 1;
    std::int32_t staging_selection_groups = 0;
    std::uint32_t staging_cache_blocks    = 0;
};

// The device region of active KV in RAM (planned in the main persistent layout).
struct QsaStagingLayout {
    TensorRegion k_pages;
    TensorRegion v_pages;
    std::optional<TensorRegion> k_scale_pages;
    std::optional<TensorRegion> v_scale_pages;
    TensorRegion block_table; // identity I32 [pages, 1]
    TensorRegion table_row;   // I32 [1] = 0
    TensorRegion tags;
    TensorRegion claims;
    TensorRegion counters;
    TensorRegion stats;
    std::int32_t layers          = 0;
    std::int32_t cache_blocks    = 0;
    std::int32_t overflow_blocks = 0;
};

struct PagedKVCacheLayout {
    DeviceKVPagePoolLayout pages;
    KVExecutionTableLayout execution_tables;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    PagedKVStorageLayout layer_storage;
    bool qsa_index = false;
    // T64 elastic KV: the page planes are offsets into a second backing (the elastic arena), the
    // execution tables into the main one.
    bool elastic_pages = false;
    // Active KV in RAM: the Host planes' layout size and the device region (absent when off).
    std::size_t host_bytes = 0;
    std::optional<QsaStagingLayout> staging;

    [[nodiscard]] std::size_t payload_bytes() const noexcept { return pages.payload_bytes(); }
};

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor block_table_;
};

class PagedKVCache {
public:
    // `device` holds `backing`; `elastic` backs the page planes of a layout planned with
    // elastic_pages (else unused).
    PagedKVCache(int device, DeviceSpan backing, const PagedKVCacheLayout& layout,
                 DeviceSpan elastic = {});

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] DeviceKVPagePool& page_pool() noexcept { return pages_; }

    [[nodiscard]] const DeviceKVPagePool& page_pool() const noexcept { return pages_; }

    [[nodiscard]] KVExecutionTablePool& execution_tables() noexcept { return execution_tables_; }

    [[nodiscard]] const KVExecutionTablePool& execution_tables() const noexcept {
        return execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(const KVExecutionRowLease& row) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

    [[nodiscard]] bool has_qsa_index() const noexcept { return qsa_index_; }

    // The QSA index planes of attention layer `layer`, with the batched execution tables (the K/V
    // tables: every row addresses the same pages in every plane). Requires has_qsa_index().
    [[nodiscard]] ops::QsaIndexPlanes qsa_index_planes(std::uint32_t layer) const;

    // Active KV in RAM: the device region QSA attention reads (null when the pool is in VRAM).
    [[nodiscard]] const ops::QsaStagingRegion* qsa_staging() const noexcept {
        return staging_ ? &*staging_ : nullptr;
    }

    // The NINFER_KVRAM_STATS diagnostic: prints the staging region's per-layer counters summed
    // (and clears them) to stderr; synchronizes the region's device. No-op without a region.
    void report_staging_stats(const char* why) const;

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table) const;

    std::optional<PinnedHostBuffer> host_; // before pages_: it backs the Host planes
    DeviceKVPagePool pages_;
    KVExecutionTablePool execution_tables_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    PagedKVStorageLayout layer_storage_;
    bool qsa_index_ = false;
    std::optional<ops::QsaStagingRegion> staging_;
    int staging_device_ = -1;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

// `pages_builder` (T64 elastic KV, may be null): plans the page planes of both pools in a second
// layout, bound to DecoderState's `elastic` backing.
[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec,
                                                    LayoutBuilder* pages_builder = nullptr);

// Active KV in RAM: the bytes of the page-locked Host planes a DecoderState of `spec`
// (spec.kv_ram_active) allocates, which PagedKVCacheLayout::host_bytes records.
[[nodiscard]] std::size_t active_kv_host_plane_bytes(const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;

    DecoderState(int device, DeviceSpan backing, const DecoderStateLayout& layout,
                 DeviceSpan elastic = {});

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::models::qwen3_5
