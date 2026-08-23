#pragma once

#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::bench {

struct QuantizedWeightFill {
    std::uint8_t low_byte   = 0x31;
    std::uint8_t high_byte  = 0xa5;
    std::uint16_t scale_f16 = 0x3c00;
};

struct PackedQuantizedWeight {
    DeviceBuffer storage;
    Weight weight{};
    std::uint64_t low_bytes    = 0;
    std::uint64_t high_offset  = 0;
    std::uint64_t high_bytes   = 0;
    std::uint64_t scale_offset = 0;
    std::uint64_t scale_bytes  = 0;

    [[nodiscard]] std::uint64_t model_weight_bytes() const noexcept {
        return low_bytes + high_bytes + scale_bytes;
    }
};

namespace detail {

struct QuantizedGeometry {
    std::int32_t group_size;
    std::int32_t high_bytes_per_group;
};

inline QuantizedGeometry quantized_geometry(QType qtype) {
    switch (qtype) {
    case QType::Q4G64_F16S:
        return {64, 0};
    case QType::Q5G64_F16S:
        return {64, 8};
    case QType::Q6G64_F16S:
        return {64, 16};
    case QType::W8G32_F16S:
        return {32, 0};
    default:
        throw std::invalid_argument("unsupported benchmark quantized format");
    }
}

inline std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* label) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        throw std::overflow_error(label);
    }
    return left * right;
}

inline std::uint64_t checked_add(std::uint64_t left, std::uint64_t right, const char* label) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1)) {
        throw std::overflow_error("benchmark quantized alignment overflow");
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

static __global__ void fill_f16_kernel(std::uint16_t* values, std::uint64_t count,
                                       std::uint16_t bits) {
    const std::uint64_t begin  = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = begin; index < count; index += stride) { values[index] = bits; }
}

inline int launch_grid(std::uint64_t elements) {
    return static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (elements + 255) / 256)));
}

} // namespace detail

inline PackedQuantizedWeight make_row_split_weight(QType qtype, std::int32_t n, std::int32_t k,
                                                   std::int32_t padded_k,
                                                   QuantizedWeightFill fill = {}) {
    const detail::QuantizedGeometry geometry = detail::quantized_geometry(qtype);
    if (n <= 0 || k <= 0 || padded_k < k || padded_k % geometry.group_size != 0) {
        throw std::invalid_argument("invalid benchmark RowSplit weight shape");
    }

    const std::uint64_t groups = detail::checked_mul(
        static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(padded_k / geometry.group_size),
        "benchmark weight group count overflow");
    const std::uint64_t low_bytes =
        detail::checked_mul(groups, 32, "benchmark low plane size overflow");
    const std::uint64_t high_bytes =
        detail::checked_mul(groups, static_cast<std::uint64_t>(geometry.high_bytes_per_group),
                            "benchmark high plane size overflow");
    const std::uint64_t scale_bytes =
        detail::checked_mul(groups, 2, "benchmark scale plane size overflow");
    const std::uint64_t high_offset  = detail::align_up(low_bytes, 256);
    const std::uint64_t scale_offset = detail::checked_add(
        high_offset, detail::align_up(high_bytes, 256), "benchmark scale plane offset overflow");
    const std::uint64_t payload_bytes =
        detail::checked_add(scale_offset, scale_bytes, "benchmark payload size overflow");
    if (payload_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("benchmark payload does not fit size_t");
    }

    PackedQuantizedWeight result{
        DeviceBuffer(static_cast<std::size_t>(payload_bytes)),
        {},
        low_bytes,
        high_offset,
        high_bytes,
        scale_offset,
        scale_bytes,
    };
    CUDA_CHECK(cudaMemset(result.storage.p, 0, result.storage.bytes));
    CUDA_CHECK(cudaMemset(result.storage.p, fill.low_byte, low_bytes));
    if (high_bytes != 0) {
        CUDA_CHECK(cudaMemset(static_cast<std::uint8_t*>(result.storage.p) + high_offset,
                              fill.high_byte, high_bytes));
    }
    detail::fill_f16_kernel<<<detail::launch_grid(groups), 256>>>(
        reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(result.storage.p) +
                                         scale_offset),
        groups, fill.scale_f16);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = result.weight;
    weight.payload          = result.storage.p;
    weight.payload_bytes    = payload_bytes;
    weight.high_plane_bytes = high_bytes;
    weight.qtype            = qtype;
    weight.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    weight.shape[0]         = n;
    weight.shape[1]         = k;
    weight.padded_shape[0]  = n;
    weight.padded_shape[1]  = padded_k;
    weight.ndim             = 2;
    weight.qdata            = result.storage.p;
    weight.qhigh            = high_bytes == 0
                                  ? nullptr
                                  : static_cast<const std::uint8_t*>(result.storage.p) + result.high_offset;
    weight.scales      = static_cast<const std::uint8_t*>(result.storage.p) + result.scale_offset;
    weight.n           = n;
    weight.k           = k;
    weight.group       = geometry.group_size;
    weight.layout      = QuantLayout::RowSplit;
    weight.scale_dtype = DType::FP16;
    return result;
}

inline PackedQuantizedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k) {
    if (n <= 0 || k <= 0 || (n % 128) != 0 || (k % 64) != 0) {
        throw std::invalid_argument("invalid benchmark NVFP4 weight shape");
    }
    const std::uint64_t elements =
        detail::checked_mul(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k),
                            "benchmark NVFP4 element count overflow");
    const std::uint64_t code_bytes   = elements / 2;
    const std::uint64_t scale_offset = detail::align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = elements / 16;
    const std::uint64_t divisor_offset =
        detail::checked_add(scale_offset, scale_bytes, "benchmark NVFP4 divisor offset overflow");
    const std::uint64_t payload_bytes =
        detail::checked_add(divisor_offset, sizeof(float), "benchmark NVFP4 payload size overflow");
    if (payload_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("benchmark NVFP4 payload does not fit size_t");
    }

    PackedQuantizedWeight result{
        DeviceBuffer(static_cast<std::size_t>(payload_bytes)),
        {},
        code_bytes,
        0,
        0,
        scale_offset,
        scale_bytes,
    };
    CUDA_CHECK(cudaMemset(result.storage.p, 0, result.storage.bytes));
    CUDA_CHECK(cudaMemset(result.storage.p, 0x22, code_bytes));
    CUDA_CHECK(
        cudaMemset(static_cast<std::uint8_t*>(result.storage.p) + scale_offset, 0x38, scale_bytes));
    constexpr float kWeightDivisor = 0.125F;
    CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(result.storage.p) + divisor_offset,
                          &kWeightDivisor, sizeof(kWeightDivisor), cudaMemcpyHostToDevice));

    Weight& weight              = result.weight;
    weight.payload              = result.storage.p;
    weight.payload_bytes        = payload_bytes;
    weight.qtype                = QType::NVFP4;
    weight.layout               = QuantLayout::BlockScaleK16M128x4;
    weight.scale_dtype          = DType::FP8_E4M3FN;
    weight.group_size           = 16;
    weight.group                = 16;
    weight.ndim                 = 2;
    weight.shape[0]             = n;
    weight.shape[1]             = k;
    weight.padded_shape[0]      = n;
    weight.padded_shape[1]      = k;
    weight.qdata                = result.storage.p;
    weight.qhigh                = nullptr;
    weight.scales               = static_cast<std::uint8_t*>(result.storage.p) + scale_offset;
    weight.n                    = n;
    weight.k                    = k;
    weight.weight_scale_divisor = kWeightDivisor;
    weight.input_scale_divisor  = 3.5F;
    return result;
}

inline PackedQuantizedWeight make_fp8_weight(std::int32_t n, std::int32_t k) {
    if (n <= 0 || k <= 0) { throw std::invalid_argument("invalid benchmark FP8 weight shape"); }
    const std::uint64_t code_bytes =
        detail::checked_mul(static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k),
                            "benchmark FP8 code size overflow");
    const std::uint64_t scale_offset = detail::align_up(code_bytes, 256);
    const std::uint64_t scale_bytes =
        detail::checked_mul(static_cast<std::uint64_t>(n), 2, "benchmark FP8 scale size overflow");
    const std::uint64_t payload_bytes =
        detail::checked_add(scale_offset, scale_bytes, "benchmark FP8 payload size overflow");
    if (payload_bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("benchmark FP8 payload does not fit size_t");
    }

    PackedQuantizedWeight result{
        DeviceBuffer(static_cast<std::size_t>(payload_bytes)),
        {},
        code_bytes,
        0,
        0,
        scale_offset,
        scale_bytes,
    };
    CUDA_CHECK(cudaMemset(result.storage.p, 0, result.storage.bytes));
    CUDA_CHECK(cudaMemset(result.storage.p, 0x31, code_bytes));
    detail::fill_f16_kernel<<<detail::launch_grid(n), 256>>>(
        reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(result.storage.p) +
                                         scale_offset),
        static_cast<std::uint64_t>(n), 0x3c00U);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = result.weight;
    weight.payload          = result.storage.p;
    weight.payload_bytes    = payload_bytes;
    weight.high_plane_bytes = 0;
    weight.qtype            = QType::FP8_E4M3FN_ROW_BF16S;
    weight.layout           = QuantLayout::RowScale;
    weight.scale_dtype      = DType::BF16;
    weight.group_size       = static_cast<std::uint32_t>(k);
    weight.group            = k;
    weight.ndim             = 2;
    weight.shape[0]         = n;
    weight.shape[1]         = k;
    weight.padded_shape[0]  = n;
    weight.padded_shape[1]  = k;
    weight.qdata            = result.storage.p;
    weight.qhigh            = nullptr;
    weight.scales           = static_cast<std::uint8_t*>(result.storage.p) + scale_offset;
    weight.n                = n;
    weight.k                = k;
    weight.scale_ne[0]      = n;
    weight.scale_nb[0]      = 2;
    weight.scale_nb[1]      = static_cast<std::int64_t>(n) * 2;
    weight.scale_nb[2]      = weight.scale_nb[1];
    weight.scale_nb[3]      = weight.scale_nb[1];
    return result;
}

inline Weight row_view(const Weight& parent, std::int32_t row_begin, std::int32_t rows) {
    if (parent.layout != QuantLayout::RowSplit || row_begin < 0 || rows <= 0 ||
        row_begin > parent.n - rows) {
        throw std::invalid_argument("benchmark RowSplit row view is out of range");
    }
    const detail::QuantizedGeometry geometry = detail::quantized_geometry(parent.qtype);
    const std::uint64_t groups_per_row =
        static_cast<std::uint64_t>(parent.padded_shape[1] / geometry.group_size);
    const std::uint64_t low_row_bytes = groups_per_row * 32;
    const std::uint64_t high_row_bytes =
        groups_per_row * static_cast<std::uint64_t>(geometry.high_bytes_per_group);
    const std::uint64_t scale_row_bytes = groups_per_row * 2;

    Weight view = parent;
    view.qdata  = static_cast<const std::uint8_t*>(parent.qdata) +
                 static_cast<std::uint64_t>(row_begin) * low_row_bytes;
    view.qhigh  = high_row_bytes == 0 ? nullptr
                                      : static_cast<const std::uint8_t*>(parent.qhigh) +
                                           static_cast<std::uint64_t>(row_begin) * high_row_bytes;
    view.scales = static_cast<const std::uint8_t*>(parent.scales) +
                  static_cast<std::uint64_t>(row_begin) * scale_row_bytes;
    view.n               = rows;
    view.shape[0]        = rows;
    view.padded_shape[0] = rows;
    return view;
}

} // namespace ninfer::bench
