#pragma once

#include "core/state_image.h"
#include "ninfer/kv_cache_format.h"

namespace ninfer {

// Absolute position p resides at p % capacity. Each record contains dense [D,capacity,H] data
// fields and optional [D/128,capacity,H] U8 UE8M0 scale fields for Fp8E4m3fnGroup128Ue8m0 storage.
struct CyclicKVCacheLayerView {
    [[nodiscard]] Device device() const;
    StateFieldView k;
    StateFieldView v;
    StateFieldView k_scales{};
    StateFieldView v_scales{};
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    KvCacheStorage storage        = KvCacheStorage::BFloat16;
};

struct ConstCyclicKVCacheLayerView {
    ConstStateFieldView k;
    ConstStateFieldView v;
    ConstStateFieldView k_scales{};
    ConstStateFieldView v_scales{};
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    KvCacheStorage storage        = KvCacheStorage::BFloat16;

    ConstCyclicKVCacheLayerView() = default;

    ConstCyclicKVCacheLayerView(const CyclicKVCacheLayerView& view)
        : k(view.k), v(view.v), k_scales(view.k_scales), v_scales(view.v_scales),
          capacity(view.capacity), padded_capacity(view.padded_capacity),
          num_kv_heads(view.num_kv_heads), head_dim(view.head_dim), storage(view.storage) {}

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
    StateField k_scales{};
    StateField v_scales{};
    LayoutRegion group;
    KvCacheStorage storage        = KvCacheStorage::BFloat16;

    [[nodiscard]] constexpr bool has_scales() const noexcept {
        return storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    }
};

[[nodiscard]] CyclicKVCacheLayout
plan_cyclic_kv_cache(StateImageBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                     std::int32_t num_kv_heads, std::int32_t head_dim,
                     KvCacheStorage storage = KvCacheStorage::BFloat16);

class CyclicKVCacheView {
public:
    CyclicKVCacheView(const StateImageStorage& storage, const CyclicKVCacheLayout& layout);

    [[nodiscard]] std::uint32_t layer_count() const noexcept { return layers_; }

    [[nodiscard]] std::uint32_t capacity() const noexcept { return all_layers_.capacity; }

    [[nodiscard]] std::uint32_t padded_capacity() const noexcept {
        return all_layers_.padded_capacity;
    }

    [[nodiscard]] KvCacheStorage storage() const noexcept { return all_layers_.storage; }

    [[nodiscard]] CyclicKVCacheLayerView layer_view(std::uint32_t layer,
                                                    const Tensor& byte_offsets) const;

private:
    CyclicKVCacheLayerView all_layers_;
    std::uint32_t layers_;
};

} // namespace ninfer
