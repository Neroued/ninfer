#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

/**
 * Fixed cyclic BF16-K/FP16-V storage with absolute-position addressing.
 *
 * A logical absolute position p resides in physical slot p % capacity. The view deliberately
 * carries no mutable frontier: callers supply the live absolute interval to the consuming Op.
 */
struct CyclicKVCacheLayerView {
    Tensor k;
    Tensor v;
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    std::int32_t lane_capacity    = 0;
};

struct CyclicKVCacheSlotView {
    Tensor k_layer0;
    Tensor v_layer0;
    std::size_t k_layer_bytes          = 0;
    std::size_t v_layer_bytes          = 0;
    std::ptrdiff_t k_layer_pitch_bytes = 0;
    std::ptrdiff_t v_layer_pitch_bytes = 0;
    std::uint32_t layers               = 0;
};

struct CyclicKVCacheLayout {
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    std::int32_t lane_capacity    = 0;
    std::vector<TensorRegion> k;
    std::vector<TensorRegion> v;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] CyclicKVCacheLayout
plan_cyclic_kv_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                     std::int32_t num_kv_heads, std::int32_t head_dim, std::int32_t lane_capacity);

class CyclicKVCache {
public:
    CyclicKVCache(DeviceSpan backing, const CyclicKVCacheLayout& layout);

    CyclicKVCache(const CyclicKVCache&)            = delete;
    CyclicKVCache& operator=(const CyclicKVCache&) = delete;
    CyclicKVCache(CyclicKVCache&&)                 = delete;
    CyclicKVCache& operator=(CyclicKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t layer_count() const noexcept;

    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::uint32_t padded_capacity() const noexcept { return padded_capacity_; }

    [[nodiscard]] std::int32_t num_kv_heads() const noexcept { return num_kv_heads_; }

    [[nodiscard]] std::int32_t head_dim() const noexcept { return head_dim_; }

    [[nodiscard]] std::int32_t lane_capacity() const noexcept { return lane_capacity_; }

    [[nodiscard]] CyclicKVCacheLayerView layer_view(std::uint32_t layer) const;
    [[nodiscard]] CyclicKVCacheSlotView slot_view(std::int32_t slot) const;

    // Copies one slot's complete fixed state. Source and destination geometry must match.
    void copy_slot_from(const CyclicKVCache& source, std::int32_t source_slot,
                        std::int32_t destination_slot, cudaStream_t stream);

private:
    std::vector<Tensor> k_;
    std::vector<Tensor> v_;
    std::uint32_t capacity_        = 0;
    std::uint32_t padded_capacity_ = 0;
    std::int32_t num_kv_heads_     = 0;
    std::int32_t head_dim_         = 0;
    std::int32_t lane_capacity_    = 0;
};

} // namespace ninfer
