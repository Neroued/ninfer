#pragma once

#include "ninfer/ops/kpool_mla.h"
#include "ops/common/host/operands.h"

#include <array>
#include <limits>

namespace ninfer::ops::detail::kpool {
namespace operand                       = common::operand;
inline constexpr const char* kOperation = "Kpool MLA";

inline void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(std::string(kOperation) + ": " + message);
}

// Check represented extents without imposing a contiguous outer layout. Sorting active axes
// accepts permuted/padded records while keeping features contiguous and fields nonoverlapping.
inline std::size_t strided(const Tensor& t, DType type, std::array<int, 4> shape, Device device,
                           const char* name) {
    operand::device(t.device, device, kOperation, name);
    require(t.dtype == type && operand::aligned(t.data, dtype_size(type)), name);
    std::array<int, 3> outer{1, 2, 3};
    std::sort(outer.begin(), outer.end(), [&](int a, int b) { return t.nb[a] < t.nb[b]; });
    for (int axis = 0; axis < 4; ++axis)
        require(t.ne[axis] == shape[axis] && t.ne[axis] > 0 && t.nb[axis] > 0 &&
                    t.nb[axis] % dtype_size(type) == 0,
                name);
    require(t.nb[0] == std::int64_t(dtype_size(type)), name);
    std::size_t span = operand::bytes(shape[0], dtype_size(type), kOperation);
    for (int axis : outer) {
        if (shape[axis] == 1) continue;
        require(std::uint64_t(t.nb[axis]) >= span, name);
        const auto extra = operand::bytes(shape[axis] - 1, t.nb[axis], kOperation);
        require(extra <= std::numeric_limits<std::size_t>::max() - span, name);
        span += extra;
    }
    (void)operand::range(t.data, span, kOperation);
    return span;
}

template <class Field>
inline void tail_field(const Field& f, int dim, int batch, int layers, Device device) {
    operand::device(f.records.storage.device, device, kOperation, "tail backing");
    require(f.field.layers == std::uint32_t(layers) && layers > 0, "tail layers");
    const auto& region = f.field.tensor.region;
    require(f.records.storage.data && region.offset <= f.records.storage.bytes &&
                region.bytes <= f.records.storage.bytes - region.offset,
            "tail capacity");
    auto meta = f.field.metadata();
    meta.data = const_cast<std::byte*>(static_cast<const std::byte*>(f.records.storage.data)) +
                region.offset;
    meta.device     = device;
    const auto span = strided(meta, DType::BF16, {dim, 3, 1, 1}, device, "tail field");
    require(region.bytes >= span, "tail field span");
    if (layers > 1) {
        require(f.field.layer_stride >= span && f.field.layer_stride % 2 == 0, "tail layer stride");
        require(std::size_t(layers - 1) <=
                    (f.records.storage.bytes - region.offset - span) / f.field.layer_stride,
                "tail layer capacity");
    }
    operand::tensor(f.records.byte_offsets, DType::I64, {batch}, 8, kOperation, "tail offsets",
                    device);
}

inline void plane(const ConstPagedPlaneView& p, DType dtype, int dim, int span, Device device) {
    operand::device(p.records.storage.device, device, kOperation, "cache backing");
    const auto& region  = p.plane.tensor.region;
    const auto expected = Tensor::metadata(dtype, {dim, 64 / span, 1});
    require(p.records.page_tokens == 64 && p.plane.geometry.dtype == dtype &&
                p.plane.geometry.leading_extent == dim && p.plane.geometry.head_extent == 1 &&
                p.plane.geometry.tokens_per_row == std::uint32_t(span) &&
                p.plane.tensor.dtype == dtype &&
                std::equal(p.plane.tensor.shape.begin(), p.plane.tensor.shape.end(), expected.ne) &&
                region.bytes == expected.bytes() && region.offset % 256 == 0 &&
                p.plane.head_stride_bytes == expected.bytes() && p.records.storage.data &&
                region.offset <= p.records.storage.bytes &&
                region.bytes <= p.records.storage.bytes - region.offset &&
                operand::aligned(p.data(), 256),
            "cache plane geometry/alignment");
    const auto& table = p.records.page_offsets;
    strided(table, DType::I64, {table.ne[0], table.ne[1], 1, 1}, device, "page offsets");
}

inline void index_cache(const ConstKpoolIndexCacheView& cache, int dim, Device device) {
    require(cache.format == KvCacheVectorFormat::BFloat16 ||
                cache.format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0,
            "unsupported index cache format");
    const auto format = paged_kv_vector_layout(cache.format, dim);
    plane(cache.keys, format.data_dtype, format.data_leading_extent, 4, device);
    if (!format.has_scale()) {
        require(!cache.scales.records.storage.data, "BF16 index cache has no scales");
        return;
    }
    plane(cache.scales, format.scale_dtype, format.scale_leading_extent, 4, device);
    operand::same_paged_records(cache.keys, cache.scales, kOperation, "index codes/scales");
    const auto& k = cache.keys.plane.tensor.region;
    const auto& s = cache.scales.plane.tensor.region;
    require(k.offset + k.bytes <= s.offset || s.offset + s.bytes <= k.offset,
            "index codes/scales overlap");
}

inline void latent_cache(const ConstKpoolMlaLatentCacheView& cache, int dim, Device device) {
    const auto format = paged_kv_vector_layout(kpool_mla_vector_format(cache.storage), dim);
    plane(cache.data, format.data_dtype, format.data_leading_extent, 1, device);
    if (!format.has_scale()) {
        require(!cache.scales.records.storage.data, "BF16 latent has no scales");
        return;
    }
    plane(cache.scales, format.scale_dtype, format.scale_leading_extent, 1, device);
    operand::same_paged_records(cache.data, cache.scales, kOperation, "latent codes/scales");
    const auto& data   = cache.data.plane.tensor.region;
    const auto& scales = cache.scales.plane.tensor.region;
    require(data.offset + data.bytes <= scales.offset ||
                scales.offset + scales.bytes <= data.offset,
            "latent codes/scales overlap");
}
} // namespace ninfer::ops::detail::kpool
