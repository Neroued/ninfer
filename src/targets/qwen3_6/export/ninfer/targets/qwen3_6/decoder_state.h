#pragma once

#include "core/linear_attention_state.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::int32_t kKvQuantGroup = 64;

struct DecoderStateSpec {
    std::uint32_t full_attention_layers     = 0;
    std::uint32_t mtp_layers                = 0;
    std::uint32_t capacity                  = 0;
    std::int32_t kv_heads                   = 0;
    std::int32_t attention_head_dim         = 0;
    DType kv_dtype                          = DType::BF16;
    std::int32_t kv_quant_group             = 0;
    bool kv_packed_v                        = false;
    bool kv_rotate_k                        = false;
    bool kv_rotate_v                        = false;
    bool kv_packed_k                        = false;
    bool kv_e8_lattice                      = false;
    bool kv_e8_root                         = false;
    bool enable_mtp                         = false;
    std::int32_t kv_table_rows              = 1;
    std::uint32_t text_physical_page_groups = 0;
    std::uint32_t mtp_physical_page_groups  = 0;
    LinearAttentionStatePoolSpec linear_attention;
};

struct PagedKVCacheLayout {
    PagedKVPoolLayout pool;
    std::uint32_t layers      = 0;
    std::uint32_t max_context = 0;
    std::int32_t kv_heads     = 0;
    std::int32_t head_dim     = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
    bool packed_v             = false;
    bool rotate_k             = false;
    bool rotate_v             = false;
    bool packed_k             = false;
    bool e8_lattice           = false;
    bool e8_root              = false;

    [[nodiscard]] std::size_t payload_bytes() const noexcept { return pool.payload_bytes(); }
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
    PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout);

    PagedKVCache(const PagedKVCache&)            = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&)                 = delete;
    PagedKVCache& operator=(PagedKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t max_context() const noexcept { return max_context_; }

    [[nodiscard]] std::uint32_t layers() const noexcept { return layers_; }

    [[nodiscard]] PagedKVPool& pool() noexcept { return pool_; }

    [[nodiscard]] const PagedKVPool& pool() const noexcept { return pool_; }

    [[nodiscard]] PagedKVCacheView execution_view(const PagedKVAllocation& allocation) const;

    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t layer) const;

private:
    friend class PagedKVCacheView;
    [[nodiscard]] PagedKVLayerView layer_view(std::uint32_t layer, Tensor block_table) const;

    PagedKVPool pool_;
    std::uint32_t layers_      = 0;
    std::uint32_t max_context_ = 0;
    std::int32_t kv_heads_     = 0;
    std::int32_t head_dim_     = 0;
    DType dtype_               = DType::BF16;
    std::int32_t quant_group_  = 0;
    bool packed_v_             = false;
    bool rotate_k_             = false;
    bool rotate_v_             = false;
    bool packed_k_             = false;
    bool e8_lattice_           = false;
    bool e8_root_              = false;
};

struct DecoderStateLayout {
    PagedKVCacheLayout text_kv;
    std::optional<PagedKVCacheLayout> mtp_kv;
    LinearAttentionStatePoolLayout linear_attention;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept;
};

[[nodiscard]] DecoderStateLayout plan_decoder_state(LayoutBuilder& builder,
                                                    const DecoderStateSpec& spec);

struct DecoderState {
    PagedKVCache text_kv;
    std::optional<PagedKVCache> mtp_kv;
    LinearAttentionStatePool linear_attention;

    DecoderState(DeviceSpan backing, const DecoderStateLayout& layout);

    [[nodiscard]] PagedKVCache* mtp_cache() noexcept;
    [[nodiscard]] const PagedKVCache* mtp_cache() const noexcept;
};

} // namespace ninfer::targets::qwen3_6
