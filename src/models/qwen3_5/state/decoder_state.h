#pragma once

#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "models/common/context/storage.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5 {

struct DecoderStateSpec {
    std::uint32_t full_attention_layers = 0;
    std::uint32_t mtp_layers            = 0;
    std::uint32_t capacity              = 0;
    std::int32_t kv_heads               = 0;
    std::int32_t attention_head_dim     = 0;
    KvCacheStorage kv_storage           = KvCacheStorage::BFloat16;
    bool enable_mtp                     = false;
    std::int32_t kv_table_rows          = 1;
};

struct PagedKVCacheLayout {
    KVPageLayout pages;
    KVExecutionTableLayout execution_tables;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    PagedKVStorageLayout layer_storage;
};

[[nodiscard]] PagedKVCacheLayout plan_paged_kv_cache(LayoutBuilder& builder, std::uint32_t layers,
                                                     std::uint32_t capacity, std::int32_t kv_heads,
                                                     std::int32_t head_dim, KvCacheStorage storage,
                                                     std::int32_t table_rows);

class PagedKVCache;

class PagedKVCacheView {
public:
    PagedKVCacheView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return cache_ != nullptr; }

    [[nodiscard]] std::uint32_t max_context() const noexcept;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCache;
    PagedKVCacheView(const PagedKVCache& cache, Tensor page_offsets) noexcept;

    const PagedKVCache* cache_ = nullptr;
    Tensor page_offsets_;
};

class PagedKVCache {
public:
    PagedKVCache(MemorySpan arena, MemorySpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] MemorySpan arena() const noexcept { return arena_; }

    [[nodiscard]] const KVPageLayout& page_layout() const noexcept { return pages_; }

    [[nodiscard]] context::ExecutionTableStorage& tables() noexcept { return execution_tables_; }

    [[nodiscard]] const context::ExecutionTableStorage& tables() const noexcept {
        return execution_tables_;
    }

    [[nodiscard]] PagedKVCacheView execution_view(std::int32_t row) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor page_offsets) const;

    MemorySpan arena_;
    KVPageLayout pages_;
    context::ExecutionTableStorage execution_tables_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    PagedKVStorageLayout layer_storage_;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;

    DecoderState(MemorySpan arena, MemorySpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::models::qwen3_5
