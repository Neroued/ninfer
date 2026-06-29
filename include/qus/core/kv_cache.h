#pragma once

#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cstdint>
#include <vector>

namespace qus {

struct KVHeadSlot {
    Tensor k;
    Tensor v;
};

struct KVCache {
    std::vector<Tensor> k;
    std::vector<Tensor> v;
    std::uint32_t pos            = 0;
    std::uint32_t max_context    = 0;
    std::uint32_t padded_context = 0;
    std::int32_t num_kv_heads    = 0;
    std::int32_t head_dim        = 0;
    DType dtype                  = DType::BF16;

    KVCache() = default;
    KVCache(DeviceArena& cache_arena, std::uint32_t full_layers, std::uint32_t max_context,
            std::int32_t num_kv_heads, std::int32_t head_dim, DType dtype = DType::BF16);

    std::uint32_t layer_count() const noexcept;
    KVHeadSlot slot(std::uint32_t layer, std::uint32_t position, std::int32_t kv_head) const;
    KVHeadSlot append_slot(std::uint32_t layer, std::int32_t kv_head) const;
    void advance();
    void reset() noexcept;
};

} // namespace qus
