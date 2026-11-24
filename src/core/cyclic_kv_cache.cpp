#include "core/cyclic_kv_cache.h"

#include <limits>
#include <stdexcept>

namespace ninfer {

Device CyclicKVCacheLayerView::device() const {
    return ConstCyclicKVCacheLayerView(*this).device();
}

Device ConstCyclicKVCacheLayerView::device() const {
    const auto device = k.records.storage.device;
    if (v.records.storage.device != device || k.records.byte_offsets.device != device ||
        v.records.byte_offsets.device != device) {
        throw std::invalid_argument("Cyclic KV spans different devices");
    }
    return device;
}

CyclicKVCacheLayout plan_cyclic_kv_cache(StateImageBuilder& builder, std::uint32_t layers,
                                         std::uint32_t capacity, std::int32_t num_kv_heads,
                                         std::int32_t head_dim) {
    const std::uint64_t padded = (static_cast<std::uint64_t>(capacity) + 127U) & ~127ULL;
    if (layers == 0 || capacity == 0 || padded > INT32_MAX || num_kv_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument("Cyclic KV geometry is invalid");
    }
    const auto k =
        builder.add(DType::BF16, {head_dim, static_cast<std::int32_t>(padded), num_kv_heads},
                    layers, "Cyclic K");
    const auto v =
        builder.add(DType::FP16, {head_dim, static_cast<std::int32_t>(padded), num_kv_heads},
                    layers, "Cyclic V");
    return {capacity,
            static_cast<std::uint32_t>(padded),
            num_kv_heads,
            head_dim,
            layers,
            k,
            v,
            builder.group(k, v)};
}

CyclicKVCacheView::CyclicKVCacheView(const StateImageStorage& storage,
                                     const CyclicKVCacheLayout& layout)
    : all_layers_{storage.field(layout.k, {}), storage.field(layout.v, {}), layout.capacity,
                  layout.padded_capacity,      layout.num_kv_heads,         layout.head_dim},
      layers_(layout.layers) {
    if (layers_ == 0 || layout.capacity == 0 || layout.padded_capacity < layout.capacity ||
        layout.head_dim <= 0 || layout.num_kv_heads <= 0) {
        throw std::invalid_argument("Cyclic KV field geometry is inconsistent");
    }
    const std::array<std::int32_t, 4> shape{
        layout.head_dim, static_cast<std::int32_t>(layout.padded_capacity), layout.num_kv_heads, 1};
    for (const auto* field : {&all_layers_.k.field, &all_layers_.v.field}) {
        if (field->tensor.shape != shape || field->layers != layers_) {
            throw std::invalid_argument("Cyclic KV field shape is inconsistent");
        }
    }
    if (all_layers_.k.field.tensor.dtype != DType::BF16 ||
        all_layers_.v.field.tensor.dtype != DType::FP16) {
        throw std::invalid_argument("Cyclic KV requires BF16 K and FP16 V");
    }
}

CyclicKVCacheLayerView CyclicKVCacheView::layer_view(std::uint32_t layer,
                                                     const Tensor& byte_offsets) const {
    auto result                   = all_layers_;
    result.k                      = result.k.layer(layer);
    result.v                      = result.v.layer(layer);
    result.k.records.byte_offsets = byte_offsets;
    result.v.records.byte_offsets = byte_offsets;
    return result;
}

} // namespace ninfer
