#pragma once

#include "core/state_image.h"

namespace ninfer {

// Absolute position p resides at p % capacity. Each record contains dense [D,capacity,H] fields.
struct CyclicKVCacheLayerView {
    [[nodiscard]] Device device() const;
    StateFieldView k;
    StateFieldView v;
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
};

struct ConstCyclicKVCacheLayerView {
    ConstStateFieldView k;
    ConstStateFieldView v;
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;

    ConstCyclicKVCacheLayerView() = default;

    ConstCyclicKVCacheLayerView(ConstStateFieldView k, ConstStateFieldView v,
                                std::uint32_t capacity, std::uint32_t padded_capacity,
                                std::int32_t num_kv_heads, std::int32_t head_dim)
        : k(k), v(v), capacity(capacity), padded_capacity(padded_capacity),
          num_kv_heads(num_kv_heads), head_dim(head_dim) {}

    ConstCyclicKVCacheLayerView(const CyclicKVCacheLayerView& view)
        : k(view.k), v(view.v), capacity(view.capacity), padded_capacity(view.padded_capacity),
          num_kv_heads(view.num_kv_heads), head_dim(view.head_dim) {}

    [[nodiscard]] Device device() const;
};

struct CyclicKVCacheLayout {
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    std::uint32_t layers          = 0;
    StateField k;
    StateField v;
    LayoutRegion group;
};

[[nodiscard]] CyclicKVCacheLayout plan_cyclic_kv_cache(StateImageBuilder& builder,
                                                       std::uint32_t layers, std::uint32_t capacity,
                                                       std::int32_t num_kv_heads,
                                                       std::int32_t head_dim);

class CyclicKVCacheView {
public:
    CyclicKVCacheView(const StateImageStorage& storage, const CyclicKVCacheLayout& layout);

    [[nodiscard]] std::uint32_t layer_count() const noexcept { return layers_; }

    [[nodiscard]] std::uint32_t capacity() const noexcept { return all_layers_.capacity; }

    [[nodiscard]] std::uint32_t padded_capacity() const noexcept {
        return all_layers_.padded_capacity;
    }

    [[nodiscard]] CyclicKVCacheLayerView layer_view(std::uint32_t layer,
                                                    const Tensor& byte_offsets) const;

private:
    CyclicKVCacheLayerView all_layers_;
    std::uint32_t layers_;
};

} // namespace ninfer
