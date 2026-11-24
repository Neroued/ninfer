#pragma once

#include "core/dtype.h"
#include "ninfer/kv_cache_format.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

/** Physical data/scale planes for one K or V vector. */
struct PagedKVVectorLayout {
    DType data_dtype                  = DType::BF16;
    std::int32_t data_leading_extent  = 0;
    DType scale_dtype                 = DType::U8;
    std::int32_t scale_leading_extent = 0;

    [[nodiscard]] constexpr bool has_scale() const noexcept { return scale_leading_extent != 0; }

    [[nodiscard]] std::size_t physical_bytes() const {
        return static_cast<std::size_t>(data_leading_extent) * dtype_size(data_dtype) +
               static_cast<std::size_t>(scale_leading_extent) * dtype_size(scale_dtype);
    }

    friend bool operator==(const PagedKVVectorLayout&, const PagedKVVectorLayout&) = default;
};

/** Resolved physical plane schema for one paged K/V layer. */
struct PagedKVStorageLayout {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    std::int32_t head_dim  = 0;
    PagedKVVectorLayout key;
    PagedKVVectorLayout value;

    [[nodiscard]] constexpr std::size_t planes_per_layer() const noexcept {
        return 2ULL + static_cast<std::size_t>(key.has_scale()) +
               static_cast<std::size_t>(value.has_scale());
    }

    [[nodiscard]] constexpr std::size_t logical_vector_bytes() const noexcept {
        return static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    }

    [[nodiscard]] constexpr std::size_t logical_bytes_per_token_head() const noexcept {
        return 2ULL * logical_vector_bytes();
    }

    [[nodiscard]] std::size_t physical_bytes_per_token_head() const {
        return key.physical_bytes() + value.physical_bytes();
    }

    friend bool operator==(const PagedKVStorageLayout&, const PagedKVStorageLayout&) = default;
};

[[nodiscard]] inline PagedKVVectorLayout paged_kv_vector_layout(KvCacheVectorFormat format,
                                                                std::int32_t dim) {
    const auto info = kv_cache_format_info(format);
    if (dim <= 0 || (info.group_size && dim % info.group_size))
        throw std::invalid_argument("unsupported KV-cache vector geometry");
    DType data;
    switch (format) {
    case KvCacheVectorFormat::BFloat16:
        data = DType::BF16;
        break;
    case KvCacheVectorFormat::Float16:
        data = DType::FP16;
        break;
    case KvCacheVectorFormat::Int8Group64F16:
        data = DType::I8;
        break;
    case KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0:
        data = DType::FP8_E4M3FN;
        break;
    case KvCacheVectorFormat::Nvfp4Group16Ue4m3:
        data = DType::U8;
        break;
    default:
        throw std::invalid_argument("unknown KV-cache vector format");
    }
    return {data, info.value_bits == 4 ? dim / 2 : dim,
            info.scale == KvCacheScaleFormat::Float16 ? DType::FP16 : DType::U8,
            info.group_size ? dim / info.group_size : 0};
}

[[nodiscard]] inline PagedKVStorageLayout paged_kv_storage_layout(KvCacheStorage storage,
                                                                  std::int32_t head_dim) {
    const auto profile = kv_cache_profile_info(storage);
    if (head_dim <= 0 || (storage != KvCacheStorage::BFloat16 && head_dim != kD256KVCacheHeadDim))
        throw std::invalid_argument("unsupported paged KV-cache storage geometry");
    return {storage, head_dim, paged_kv_vector_layout(profile.key, head_dim),
            paged_kv_vector_layout(profile.value, head_dim)};
}

} // namespace ninfer
