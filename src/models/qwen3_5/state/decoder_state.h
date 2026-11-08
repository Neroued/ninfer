#pragma once

#include "core/arena.h"
#include "core/elastic_arena.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/sparse_attention.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::models::qwen3_5 {

inline constexpr std::int32_t kKvInt8QuantGroup = 64;
// Active KV in RAM: the Host planes' RAM grows and shrinks in steps of this many pages (8,192
// tokens; about 75 MiB per KV head at int8 over the 12 attention layers of Qwen3.8-Flash-Next).
inline constexpr std::uint32_t kActiveKvHostStepPages = 128;
inline constexpr std::int32_t kKvFp8QuantGroup        = 256;

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
    // Active KV in RAM (--kv-ram-active, docs/maintainer/active-kv-in-ram.md): the K/V, scale and
    // QSA record planes of the text and MTP pools live in page-locked, device-mapped Host memory
    // whose RAM follows the pages in use (an ElasticMemory::Host arena per pool,
    // kActiveKvHostStepPages at a time); only the QSA block keys stay in device memory. A
    // persistent device region (ops::QsaStagingRegion) serves the attention: `staging_cache_blocks`
    // 4-token blocks per attention layer (0 = default, raised so the region holds one layer of
    // `capacity`, at most `staging_region_tokens`, 0 = 262,144: longer prompt chunks stage the
    // union of their selections instead) plus an overflow area for `staging_window` columns of
    // each of the `kv_table_rows` lanes, of `staging_selection_groups` blocks each
    // (ceil(max_selected / 4)). Requires qsa_index.
    bool kv_ram_active                    = false;
    std::int32_t staging_window           = 1;
    std::int32_t staging_selection_groups = 0;
    std::uint32_t staging_cache_blocks    = 0;
    std::uint32_t staging_region_tokens   = 0;
};

// The device region of active KV in RAM (planned in the main persistent layout).
struct QsaStagingLayout {
    TensorRegion k_pages;
    TensorRegion v_pages;
    std::optional<TensorRegion> k_scale_pages;
    std::optional<TensorRegion> v_scale_pages;
    TensorRegion block_table; // identity I32 [pages, 1]
    TensorRegion table_row;   // I32 [table_rows] = 0
    TensorRegion tags;
    TensorRegion claims;
    TensorRegion counters;
    TensorRegion row_starts;
    TensorRegion stats;
    std::int32_t layers          = 0;
    std::int32_t table_rows      = 1;
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
    // Active KV in RAM: the Host planes' layout size and the device region (absent when off, and
    // in the MTP pool, which uses the text pool's region from layer `layers` of the text pool on).
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

    // Active KV in RAM: the device region QSA attention reads (null when the pool is in VRAM). The
    // MTP pool shares the text pool's region (share_staging); its layers come after the text ones.
    [[nodiscard]] const ops::QsaStagingRegion* qsa_staging() const noexcept {
        return staging_view_;
    }

    // The region's layer index of this pool's attention layer `layer`.
    [[nodiscard]] std::int32_t staging_layer(std::uint32_t layer) const noexcept {
        return staging_layer_base_ + static_cast<std::int32_t>(layer);
    }

    // Active KV in RAM, MTP pool: stage through `owner`'s region (planned with this pool's layers
    // after its own, from `layer_base`), invalidated by this pool's writes as by the owner's. The
    // owner must outlive this pool.
    void share_staging(const PagedKVCache& owner, std::int32_t layer_base);

    // The NINFER_KVRAM_STATS diagnostic: prints the staging region's per-layer counters summed
    // (and clears them) to stderr; synchronizes the region's device. No-op without a region.
    void report_staging_stats(const char* why) const;

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table) const;

    std::unique_ptr<ElasticArena> host_; // before pages_: it backs the Host planes (elastic)
    DeviceKVPagePool pages_;
    KVExecutionTablePool execution_tables_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    PagedKVStorageLayout layer_storage_;
    bool qsa_index_ = false;
    std::optional<ops::QsaStagingRegion> staging_;
    const ops::QsaStagingRegion* staging_view_ = nullptr;
    std::int32_t staging_layer_base_           = 0;
    int staging_device_                        = -1;
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

// Active KV in RAM: the bytes the page-locked Host planes of a DecoderState of `spec`
// (spec.kv_ram_active; text and MTP pools) can grow to (their address range), which the pools'
// PagedKVCacheLayout::host_bytes record. Their RAM is backed as pages are handed out
// (DeviceKVPagePool::set_host_backing).
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
