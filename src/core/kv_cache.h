#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cstdint>
#include <vector>

namespace ninfer {

inline constexpr std::int32_t kKvQuantGroup = 64;

struct KVHeadSlot {
    Tensor k;
    Tensor v;
    Tensor k_scale;
    Tensor v_scale;
};

struct KVCacheLayout {
    std::uint32_t max_context    = 0;
    std::uint32_t padded_context = 0;
    std::int32_t num_kv_heads    = 0;
    std::int32_t head_dim        = 0;
    DType dtype                  = DType::BF16;
    std::int32_t quant_group     = 0;
    std::vector<LayoutRegion> k;
    std::vector<LayoutRegion> v;
    std::vector<LayoutRegion> k_scale;
    std::vector<LayoutRegion> v_scale;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] KVCacheLayout plan_kv_cache(LayoutBuilder& builder, std::uint32_t full_layers,
                                          std::uint32_t max_context, std::int32_t num_kv_heads,
                                          std::int32_t head_dim, DType dtype = DType::BF16,
                                          std::int32_t quant_group = 0);

struct KVCache {
    std::vector<Tensor> k;
    std::vector<Tensor> v;
    std::vector<Tensor> k_scale;
    std::vector<Tensor> v_scale;
    std::uint32_t pos            = 0;
    std::uint32_t max_context    = 0;
    std::uint32_t padded_context = 0;
    std::int32_t num_kv_heads    = 0;
    std::int32_t head_dim        = 0;
    DType dtype                  = DType::BF16;
    std::int32_t quant_group     = 0;

    KVCache() = default;
    KVCache(DeviceSpan backing, const KVCacheLayout& layout);

    std::uint32_t layer_count() const noexcept;
    KVHeadSlot slot(std::uint32_t layer, std::uint32_t position, std::int32_t kv_head) const;
    KVHeadSlot append_slot(std::uint32_t layer, std::int32_t kv_head) const;
    void advance();
    void rewind(std::uint32_t position);
    void reset() noexcept;
};

} // namespace ninfer
