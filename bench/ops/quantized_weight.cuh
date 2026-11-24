#pragma once

#include "core/buffer.h"
#include "core/weight.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::bench {

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
    case QType::Q4_G64_FP16:
        return {64, 0};
    case QType::Q5_G64_FP16:
        return {64, 8};
    case QType::Q6_G64_FP16:
        return {64, 16};
    case QType::Q8_G32_FP16:
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

// Each group generates a signed logical code once; low/high planes encode those same codes.
static __global__ void fill_row_split_kernel(std::uint8_t* low, std::uint8_t* high, __half* scales,
                                             std::uint64_t groups, int k, int padded_k, int bits,
                                             std::uint64_t seed, float extent) {
    const int group_size = bits == 8 ? 32 : 64;
    const int limit      = (1 << (bits - 1)) - 1;
    const int high_bytes = bits == 5 ? 8 : bits == 6 ? 16 : 0;
    for (auto g = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; g < groups;
         g += std::uint64_t(gridDim.x) * blockDim.x) {
        const int start = static_cast<int>(g % (padded_k / group_size)) * group_size;
        unsigned codes[64];
        for (int j = 0; j < group_size; ++j) {
            const int code =
                start + j < k
                    ? static_cast<int>(fixture::bits(g * group_size + j, seed) % (2 * limit + 1)) -
                          limit
                    : 0;
            codes[j] = static_cast<unsigned>(code) & ((1U << bits) - 1);
        }
        for (int j = 0; j < 32; ++j)
            low[g * 32 + j] =
                bits == 8 ? codes[j] : (codes[2 * j] & 15) | ((codes[2 * j + 1] & 15) << 4);
        for (int j = 0; j < high_bytes; ++j) {
            unsigned packed    = 0;
            const int per_byte = 8 / (bits - 4);
            for (int lane = 0; lane < per_byte; ++lane)
                packed |= (codes[j * per_byte + lane] >> 4) << (lane * (bits - 4));
            high[g * high_bytes + j] = packed;
        }
        scales[g] =
            __float2half_rn(extent / limit * fixture::uniform(g, seed ^ 0x51ca1eU, .5F, 1.5F));
    }
}

static __global__ void fill_fp8_weight_kernel(std::uint8_t* codes, __nv_bfloat16* scales,
                                              std::uint64_t count, int rows, std::uint64_t seed,
                                              float extent) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += std::uint64_t(gridDim.x) * blockDim.x) {
        codes[i] = __nv_fp8_e4m3(fixture::uniform(i, seed, -224.F, 224.F)).__x;
        if (i < rows)
            scales[i] = __float2bfloat16_rn(extent / 224.F *
                                            fixture::uniform(i, seed ^ 0x51ca1eU, .5F, 1.5F));
    }
}

static __global__ void fill_nvfp4_weight_kernel(std::uint8_t* codes, std::uint8_t* scales,
                                                std::uint64_t count, int k, std::uint64_t seed,
                                                float extent, float divisor) {
    for (auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count / 2;
         i += std::uint64_t(gridDim.x) * blockDim.x) {
        codes[i] =
            (fixture::bits(2 * i, seed) & 15U) | ((fixture::bits(2 * i + 1, seed) & 15U) << 4);
        if (i < count / 16) {
            const auto row    = i / (k / 16);
            const auto group  = i % (k / 16);
            const auto offset = ((row / 128) * (k / 64) + group / 4) * 512 + (row % 32) * 16 +
                                ((row % 128) / 32) * 4 + group % 4;
            scales[offset] = __nv_fp8_e4m3(extent * divisor / 6.F *
                                           fixture::uniform(i, seed ^ 0x51ca1eU, .5F, 1.5F))
                                 .__x;
        }
    }
}

inline int launch_grid(std::uint64_t elements) {
    return static_cast<int>(
        std::min<std::uint64_t>(65535, std::max<std::uint64_t>(1, (elements + 255) / 256)));
}

} // namespace detail

inline PackedQuantizedWeight make_row_split_weight(QType qtype, std::int32_t n, std::int32_t k,
                                                   std::int32_t padded_k,
                                                   std::uint64_t seed = 0x51U,
                                                   float extent       = 0.05F) {
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
        DeviceBuffer(ninfer::bench::device_context().device(),
                     static_cast<std::size_t>(payload_bytes)),
        {},
        low_bytes,
        high_offset,
        high_bytes,
        scale_offset,
        scale_bytes,
    };
    CUDA_CHECK(cudaMemset(result.storage.data(), 0, result.storage.size()));
    const int bits = qtype == QType::Q4_G64_FP16   ? 4
                     : qtype == QType::Q5_G64_FP16 ? 5
                     : qtype == QType::Q6_G64_FP16 ? 6
                                                   : 8;
    detail::fill_row_split_kernel<<<detail::launch_grid(groups), 256>>>(
        static_cast<std::uint8_t*>(result.storage.data()),
        high_bytes ? static_cast<std::uint8_t*>(result.storage.data()) + high_offset : nullptr,
        reinterpret_cast<__half*>(static_cast<std::uint8_t*>(result.storage.data()) + scale_offset),
        groups, k, padded_k, bits, seed, extent);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = result.weight;
    weight.device           = result.storage.device();
    weight.payload          = result.storage.data();
    weight.payload_bytes    = payload_bytes;
    weight.high_plane_bytes = high_bytes;
    weight.qtype            = qtype;
    weight.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    weight.shape[0]         = n;
    weight.shape[1]         = k;
    weight.padded_shape[0]  = n;
    weight.padded_shape[1]  = padded_k;
    weight.ndim             = 2;
    weight.qdata            = result.storage.data();
    weight.qhigh            = high_bytes == 0 ? nullptr
                                              : static_cast<const std::uint8_t*>(result.storage.data()) +
                                         result.high_offset;
    weight.scales = static_cast<const std::uint8_t*>(result.storage.data()) + result.scale_offset;
    weight.n           = n;
    weight.k           = k;
    weight.group       = geometry.group_size;
    weight.layout      = QuantLayout::RowSplit;
    weight.scale_dtype = DType::FP16;
    return result;
}

inline PackedQuantizedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                               std::uint64_t seed = 0x51U, float extent = 0.05F) {
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
        DeviceBuffer(ninfer::bench::device_context().device(),
                     static_cast<std::size_t>(payload_bytes)),
        {},
        code_bytes,
        0,
        0,
        scale_offset,
        scale_bytes,
    };
    CUDA_CHECK(cudaMemset(result.storage.data(), 0, result.storage.size()));
    constexpr float kWeightDivisor = 128.0F;
    detail::fill_nvfp4_weight_kernel<<<detail::launch_grid(code_bytes), 256>>>(
        static_cast<std::uint8_t*>(result.storage.data()),
        static_cast<std::uint8_t*>(result.storage.data()) + scale_offset, elements, k, seed, extent,
        kWeightDivisor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(static_cast<std::uint8_t*>(result.storage.data()) + divisor_offset,
                          &kWeightDivisor, sizeof(kWeightDivisor), cudaMemcpyHostToDevice));

    Weight& weight              = result.weight;
    weight.device               = result.storage.device();
    weight.payload              = result.storage.data();
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
    weight.qdata                = result.storage.data();
    weight.qhigh                = nullptr;
    weight.scales               = static_cast<std::uint8_t*>(result.storage.data()) + scale_offset;
    weight.n                    = n;
    weight.k                    = k;
    weight.weight_scale_divisor = kWeightDivisor;
    weight.input_scale_divisor  = 3.5F;
    return result;
}

inline PackedQuantizedWeight make_fp8_weight(std::int32_t n, std::int32_t k,
                                             std::uint64_t seed = 0x51U, float extent = 0.05F) {
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
        DeviceBuffer(ninfer::bench::device_context().device(),
                     static_cast<std::size_t>(payload_bytes)),
        {},
        code_bytes,
        0,
        0,
        scale_offset,
        scale_bytes,
    };
    CUDA_CHECK(cudaMemset(result.storage.data(), 0, result.storage.size()));
    detail::fill_fp8_weight_kernel<<<detail::launch_grid(code_bytes), 256>>>(
        static_cast<std::uint8_t*>(result.storage.data()),
        reinterpret_cast<__nv_bfloat16*>(static_cast<std::uint8_t*>(result.storage.data()) +
                                         scale_offset),
        code_bytes, n, seed, extent);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Weight& weight          = result.weight;
    weight.device           = result.storage.device();
    weight.payload          = result.storage.data();
    weight.payload_bytes    = payload_bytes;
    weight.high_plane_bytes = 0;
    weight.qtype            = QType::FP8_E4M3FN_ROW_BF16;
    weight.layout           = QuantLayout::RowScale;
    weight.scale_dtype      = DType::BF16;
    weight.group_size       = static_cast<std::uint32_t>(k);
    weight.group            = k;
    weight.ndim             = 2;
    weight.shape[0]         = n;
    weight.shape[1]         = k;
    weight.padded_shape[0]  = n;
    weight.padded_shape[1]  = k;
    weight.qdata            = result.storage.data();
    weight.qhigh            = nullptr;
    weight.scales           = static_cast<std::uint8_t*>(result.storage.data()) + scale_offset;
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
