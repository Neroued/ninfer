#include "core/cyclic_kv_cache.h"

#include "core/paged_kv_storage.h"

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
    if (storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        if (k_scales.records.storage.data == nullptr || v_scales.records.storage.data == nullptr ||
            k_scales.records.byte_offsets.data == nullptr ||
            v_scales.records.byte_offsets.data == nullptr) {
            throw std::invalid_argument("FP8 cyclic KV requires bound scale fields");
        }
        if (k_scales.records.storage.device != device ||
            v_scales.records.storage.device != device ||
            k_scales.records.byte_offsets.device != device ||
            v_scales.records.byte_offsets.device != device) {
            throw std::invalid_argument("Cyclic KV scales span different devices");
        }
    } else if (storage == KvCacheStorage::BFloat16) {
        if (k_scales.records.storage.data != nullptr || v_scales.records.storage.data != nullptr ||
            k_scales.records.byte_offsets.data != nullptr ||
            v_scales.records.byte_offsets.data != nullptr) {
            throw std::invalid_argument("BF16 cyclic KV must not carry scale fields");
        }
    } else {
        throw std::invalid_argument("Unsupported cyclic KV storage format");
    }
    return device;
}

CyclicKVCacheLayout plan_cyclic_kv_cache(StateImageBuilder& builder, std::uint32_t layers,
                                         std::uint32_t capacity, std::int32_t num_kv_heads,
                                         std::int32_t head_dim, KvCacheStorage storage) {
    const std::uint64_t padded = (static_cast<std::uint64_t>(capacity) + 127U) & ~127ULL;
    if (layers == 0 || capacity == 0 || padded > INT32_MAX || num_kv_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument("Cyclic KV geometry is invalid");
    }
    const auto schema     = cyclic_kv_storage_layout(storage, head_dim);
    const auto padded_i32 = static_cast<std::int32_t>(padded);
    const auto k =
        builder.add(schema.key.data_dtype, {schema.key.data_leading_extent, padded_i32, num_kv_heads},
                    layers, "Cyclic K");
    const auto v = builder.add(schema.value.data_dtype,
                               {schema.value.data_leading_extent, padded_i32, num_kv_heads}, layers,
                               "Cyclic V");
    if (!schema.key.has_scale()) {
        return {capacity, static_cast<std::uint32_t>(padded), num_kv_heads, head_dim, layers,
                k,        v,                                  {},           {},
                builder.group(k, v), storage};
    }
    const auto k_scales = builder.add(schema.key.scale_dtype,
                                      {schema.key.scale_leading_extent, padded_i32, num_kv_heads},
                                      layers, "Cyclic K scales");
    const auto v_scales = builder.add(schema.value.scale_dtype,
                                      {schema.value.scale_leading_extent, padded_i32, num_kv_heads},
                                      layers, "Cyclic V scales");
    return {capacity, static_cast<std::uint32_t>(padded), num_kv_heads, head_dim, layers,
            k,        v,                                  k_scales,     v_scales,
            builder.group(k, v_scales), storage};
}

CyclicKVCacheView::CyclicKVCacheView(const StateImageStorage& storage,
                                     const CyclicKVCacheLayout& layout)
    : all_layers_{storage.field(layout.k, {}),
                  storage.field(layout.v, {}),
                  layout.has_scales() ? storage.field(layout.k_scales, {}) : StateFieldView{},
                  layout.has_scales() ? storage.field(layout.v_scales, {}) : StateFieldView{},
                  layout.capacity,
                  layout.padded_capacity,
                  layout.num_kv_heads,
                  layout.head_dim,
                  layout.storage},
      layers_(layout.layers) {
    if (layers_ == 0 || layout.capacity == 0 || layout.padded_capacity < layout.capacity ||
        layout.head_dim <= 0 || layout.num_kv_heads <= 0) {
        throw std::invalid_argument("Cyclic KV field geometry is inconsistent");
    }
    const auto schema     = cyclic_kv_storage_layout(layout.storage, layout.head_dim);
    const auto padded_i32 = static_cast<std::int32_t>(layout.padded_capacity);
    const auto check_field = [&](const StateFieldLayout& field, DType dtype, std::int32_t leading,
                                 const char* message) {
        const std::array<std::int32_t, 4> expected{leading, padded_i32, layout.num_kv_heads, 1};
        if (field.tensor.dtype != dtype || field.tensor.shape != expected ||
            field.layers != layers_) {
            throw std::invalid_argument(message);
        }
    };
    check_field(all_layers_.k.field, schema.key.data_dtype, schema.key.data_leading_extent,
                "Cyclic K field geometry or dtype is inconsistent");
    check_field(all_layers_.v.field, schema.value.data_dtype, schema.value.data_leading_extent,
                "Cyclic V field geometry or dtype is inconsistent");
    if (schema.key.has_scale()) {
        check_field(all_layers_.k_scales.field, schema.key.scale_dtype,
                    schema.key.scale_leading_extent,
                    "Cyclic K scale field geometry or dtype is inconsistent");
        check_field(all_layers_.v_scales.field, schema.value.scale_dtype,
                    schema.value.scale_leading_extent,
                    "Cyclic V scale field geometry or dtype is inconsistent");
    }
}

CyclicKVCacheLayerView CyclicKVCacheView::layer_view(std::uint32_t layer,
                                                     const Tensor& byte_offsets) const {
    auto result                   = all_layers_;
    result.k                      = result.k.layer(layer);
    result.v                      = result.v.layer(layer);
    result.k.records.byte_offsets = byte_offsets;
    result.v.records.byte_offsets = byte_offsets;
    if (result.storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
        result.k_scales                      = result.k_scales.layer(layer);
        result.v_scales                      = result.v_scales.layer(layer);
        result.k_scales.records.byte_offsets = byte_offsets;
        result.v_scales.records.byte_offsets = byte_offsets;
    }
    return result;
}

} // namespace ninfer
