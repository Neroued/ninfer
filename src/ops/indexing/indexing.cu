#include "ninfer/ops/indexing.h"
#include "core/device.h"
#include "core/platform.h"
#include "ops/common/host/operands.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops {
namespace {
template <class Unit>
__global__ void scatter_kernel(const Unit* source, const std::int32_t* indices, Unit* destination,
                               std::int32_t units, std::int64_t source_stride,
                               std::int64_t destination_stride) {
    const auto column           = static_cast<std::int32_t>(blockIdx.x);
    const auto source_base      = column * source_stride;
    const auto destination_base = indices[column] * destination_stride;
    for (int unit = threadIdx.x; unit < units; unit += blockDim.x)
        destination[destination_base + unit] = source[source_base + unit];
}

inline constexpr int kScatterBatchThreads = 128;

// A 2 KiB feature tile exposes parallelism even when the block has few columns.
__launch_bounds__(kScatterBatchThreads) __global__
    void scatter_bf16_batch_kernel(const uint4* __restrict__ source,
                                   const std::int32_t* __restrict__ lanes,
                                   const std::int32_t* __restrict__ valid_columns,
                                   uint4* __restrict__ destination, std::int32_t vectors_per_column,
                                   std::int32_t width, std::int64_t destination_column_stride,
                                   std::int64_t destination_lane_stride) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t batch  = static_cast<std::int32_t>(blockIdx.y);
    if (column >= valid_columns[batch]) return;
    const std::int64_t source_base =
        (static_cast<std::int64_t>(batch) * width + column) * vectors_per_column;
    const std::int64_t destination_base =
        static_cast<std::int64_t>(lanes[batch]) * destination_lane_stride +
        static_cast<std::int64_t>(column) * destination_column_stride;
    // Keep the tile loop CTA-uniform; only the store has a per-thread tail.
    for (std::int64_t base = static_cast<std::int64_t>(blockIdx.z) * kScatterBatchThreads;
         base < vectors_per_column;
         base += static_cast<std::int64_t>(gridDim.z) * kScatterBatchThreads) {
#pragma unroll
        for (int offset = threadIdx.x; offset < kScatterBatchThreads;
             offset += kScatterBatchThreads) {
            const std::int64_t vector = base + offset;
            if (vector < vectors_per_column)
                destination[destination_base + vector] = source[source_base + vector];
        }
    }
}

using ScatterLaunch = void (*)(const Tensor&, const Tensor&, Tensor&, cudaStream_t);

template <int Width>
void launch_scatter(const Tensor& source, const Tensor& indices, Tensor& destination,
                    cudaStream_t stream) {
    const int columns = source.ne[1];
    const int items   = source.ne[0] / Width;
    const auto* map   = static_cast<const std::int32_t*>(indices.data);
    using Unit        = std::conditional_t<Width == 8, uint4,
                                           std::conditional_t<Width == 2, __nv_bfloat162, __nv_bfloat16>>;
    scatter_kernel<<<columns, Width == 8 ? 128 : 256, 0, stream>>>(
        static_cast<const Unit*>(source.data), map, static_cast<Unit*>(destination.data), items,
        source.nb[1] / static_cast<std::int64_t>(sizeof(Unit)),
        destination.nb[1] / static_cast<std::int64_t>(sizeof(Unit)));
}

#if NINFER_TARGET_SPARK
template <int Blocks, int Threads>
__launch_bounds__(Threads) __global__
    void scatter_d2560_kernel(const uint4* source, const std::int32_t* indices, uint4* destination,
                              std::int64_t source_stride, std::int64_t destination_stride) {
    constexpr int Vectors = 2560 / 8, Tile = Vectors / Blocks;
    static_assert(Vectors % Blocks == 0);
    const auto column = blockIdx.x / Blocks, split = blockIdx.x % Blocks;
    const auto source_base      = std::int64_t(column) * source_stride;
    const auto destination_base = std::int64_t(indices[column]) * destination_stride;
    for (int i = threadIdx.x; i < Tile; i += Threads) {
        const auto v                      = split * Tile + i;
        destination[destination_base + v] = source[source_base + v];
    }
}

template <int Blocks, int Threads>
void launch_scatter_d2560(const Tensor& source, const Tensor& indices, Tensor& destination,
                          cudaStream_t stream) {
    scatter_d2560_kernel<Blocks, Threads><<<source.ne[1] * Blocks, Threads, 0, stream>>>(
        static_cast<const uint4*>(source.data), static_cast<const std::int32_t*>(indices.data),
        static_cast<uint4*>(destination.data), source.nb[1] / 16, destination.nb[1] / 16);
}
#endif

ScatterLaunch select_scatter(const Tensor& source, const Tensor& destination) {
    if (source.ne[0] % 8 == 0 && common::operand::aligned(source.data, 16) &&
        common::operand::aligned(destination.data, 16) && source.nb[1] % 16 == 0 &&
        destination.nb[1] % 16 == 0) {
#if NINFER_TARGET_SPARK
        if (source.ne[0] == 2560) return launch_scatter_d2560<5, 64>;
#endif
        return launch_scatter<8>;
    }
    if (source.ne[0] % 2 == 0 && common::operand::aligned(source.data, 4) &&
        common::operand::aligned(destination.data, 4) && source.nb[1] % 4 == 0 &&
        destination.nb[1] % 4 == 0)
        return launch_scatter<2>;
    return launch_scatter<1>;
}

void launch_scatter_bf16_batch(const Tensor& source, const Tensor& lanes,
                               const Tensor& valid_columns, Tensor& destination,
                               cudaStream_t stream) {
    constexpr int block = kScatterBatchThreads;
    // Keep a grid-stride feature loop beyond CUDA's z-dimension limit.
    const int feature_tiles = std::min((source.ne[0] / 8 + block - 1) / block, 65535);
    const dim3 grid(source.ne[1], source.ne[2], feature_tiles);
    scatter_bf16_batch_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(source.data), static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(valid_columns.data), static_cast<uint4*>(destination.data),
        source.ne[0] / 8, source.ne[1],
        destination.nb[1] / static_cast<std::int64_t>(sizeof(uint4)),
        destination.nb[2] / static_cast<std::int64_t>(sizeof(uint4)));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void scatter(const Tensor& src, const Tensor& indices, Tensor& dst, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (src.dtype != DType::BF16 || dst.dtype != DType::BF16 || indices.dtype != DType::I32) {
        throw std::invalid_argument("scatter: src/dst must be BF16 and indices must be I32");
    }
    if (src.ne[2] != 1 || src.ne[3] != 1 || dst.ne[2] != 1 || dst.ne[3] != 1 ||
        indices.ne[1] != 1 || indices.ne[2] != 1 || indices.ne[3] != 1 || src.ne[0] != dst.ne[0] ||
        indices.ne[0] != src.ne[1]) {
        throw std::invalid_argument("scatter: expected src [D,V], indices [V], dst [D,T]");
    }
    if (src.ne[0] < 0 || src.ne[1] < 0 || dst.ne[1] < 0) {
        throw std::invalid_argument("scatter: negative dimension");
    }
    if (src.ne[0] == 0 || src.ne[1] == 0) { return; }
    if (src.device != execution.device() || indices.device != execution.device() ||
        dst.device != execution.device()) {
        throw std::invalid_argument("scatter: operands must belong to the execution device");
    }
    common::operand::slotted_tensor(src, DType::BF16, {src.ne[0], src.ne[1]}, 1, 2, "scatter",
                                    "source", execution.device());
    common::operand::slotted_tensor(dst, DType::BF16, {dst.ne[0], dst.ne[1]}, 1, 2, "scatter",
                                    "destination", execution.device());
    common::operand::contiguous_nonnull(indices, "scatter", "indices", execution.device());
    if (src.data == nullptr || indices.data == nullptr || dst.data == nullptr) {
        throw std::invalid_argument("scatter: tensor data must be non-null");
    }
    const auto destination = common::operand::blocks(dst, "scatter");
    if (common::operand::overlaps(destination, common::operand::blocks(src, "scatter")) ||
        common::operand::overlaps(destination, common::operand::range(indices, "scatter")))
        throw std::invalid_argument("scatter: destination overlaps an input");
    select_scatter(src, dst)(src, indices, dst, stream);
    CUDA_CHECK(cudaGetLastError());
}

void scatter_bf16_batch(const Tensor& source, const Tensor& lanes, const Tensor& valid_columns,
                        Tensor& destination, CudaExecutionView execution) {
    if (source.device != execution.device() || lanes.device != execution.device() ||
        valid_columns.device != execution.device() || destination.device != execution.device()) {
        throw std::invalid_argument(
            "scatter_bf16_batch: operands must belong to the execution device");
    }
    const auto stream        = execution.stream();
    const std::int32_t width = source.ne[1];
    const std::int32_t batch = source.ne[2];
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16 ||
        lanes.dtype != DType::I32 || valid_columns.dtype != DType::I32 || source.ne[0] <= 0 ||
        (source.ne[0] % 8) != 0 || width <= 0 || batch <= 0 || source.ne[3] != 1 ||
        destination.ne[0] != source.ne[0] || destination.ne[1] != width || destination.ne[2] <= 0 ||
        destination.ne[3] != 1 || lanes.ne[0] != batch || valid_columns.ne[0] != batch ||
        lanes.ne[1] != 1 || valid_columns.ne[1] != 1 || lanes.ne[2] != 1 ||
        valid_columns.ne[2] != 1 || lanes.ne[3] != 1 || valid_columns.ne[3] != 1) {
        throw std::invalid_argument("scatter_bf16_batch: invalid [D,W,B] to [D,W,C] geometry");
    }
    if (!source.is_contiguous() || !lanes.is_contiguous() || !valid_columns.is_contiguous() ||
        source.data == nullptr || lanes.data == nullptr || valid_columns.data == nullptr ||
        destination.data == nullptr || destination.nb[0] != 2 ||
        (destination.nb[1] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        (destination.nb[2] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        ((reinterpret_cast<std::uintptr_t>(source.data) |
          reinterpret_cast<std::uintptr_t>(destination.data)) &
         15U) != 0U) {
        throw std::invalid_argument(
            "scatter_bf16_batch: tensors must have aligned dense-column storage");
    }
    launch_scatter_bf16_batch(source, lanes, valid_columns, destination, stream);
}

void extract_bf16_features(const Tensor& source, std::int32_t feature_offset, Tensor& destination,
                           CudaExecutionView execution) {
    if (source.device != execution.device() || destination.device != execution.device()) {
        throw std::invalid_argument(
            "extract_bf16_features: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16) {
        throw std::invalid_argument("extract_bf16_features: tensors must be BF16");
    }
    if (source.ne[0] <= 0 || source.ne[1] <= 0 || destination.ne[0] <= 0 ||
        destination.ne[1] != source.ne[1] || source.ne[2] != 1 || source.ne[3] != 1 ||
        destination.ne[2] != 1 || destination.ne[3] != 1 || feature_offset < 0 ||
        feature_offset > source.ne[0] - destination.ne[0]) {
        throw std::invalid_argument("extract_bf16_features: invalid rank-2 slice geometry");
    }
    if (!source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr) {
        throw std::invalid_argument(
            "extract_bf16_features: tensors must be contiguous and non-null");
    }
    if (source.data == destination.data) {
        throw std::invalid_argument("extract_bf16_features: source and destination must not alias");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width         = static_cast<std::size_t>(destination.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(source.ne[0]) * element_bytes;
    const std::size_t destination_pitch =
        static_cast<std::size_t>(destination.ne[0]) * element_bytes;
    const auto* source_ptr = static_cast<const unsigned char*>(source.data) +
                             static_cast<std::size_t>(feature_offset) * element_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(destination.data, destination_pitch, source_ptr, source_pitch,
                                 width, static_cast<std::size_t>(destination.ne[1]),
                                 cudaMemcpyDeviceToDevice, stream));
}

} // namespace ninfer::ops
