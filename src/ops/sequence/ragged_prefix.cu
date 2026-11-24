#include "ninfer/ops/sequence.h"
#include <cstdint>
#include <stdexcept>
#include "core/device.h"
#include <algorithm>
#include <cuda_bf16.h>

namespace ninfer::ops {

namespace {

inline constexpr int kRaggedPrefixVectorsPerBlock = 128;

__global__ void prepare_ragged_prefix_kernel(
    const uint4* __restrict__ source, const std::int32_t* __restrict__ lanes,
    const std::int32_t* __restrict__ starts, const std::int32_t* __restrict__ ends,
    uint4* __restrict__ destination, std::int32_t* __restrict__ positions,
    std::int32_t* __restrict__ counts, std::int32_t vectors_per_column, std::int32_t width,
    std::int64_t source_column_stride, std::int64_t source_lane_stride) {
    const int feature_tile         = blockIdx.z;
    const std::int32_t column      = blockIdx.x;
    const std::int32_t batch       = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t count       = ends[batch] - starts[batch];
    const bool live                = column < count;
    const std::int64_t source_base = static_cast<std::int64_t>(lanes[batch]) * source_lane_stride +
                                     static_cast<std::int64_t>(column) * source_column_stride;
    const std::int64_t destination_base =
        (static_cast<std::int64_t>(batch) * width + column) * vectors_per_column;
    const uint4 zero{};
    for (int vector = feature_tile * kRaggedPrefixVectorsPerBlock + threadIdx.x;
         vector < vectors_per_column; vector += gridDim.z * kRaggedPrefixVectorsPerBlock)
        destination[destination_base + vector] = live ? source[source_base + vector] : zero;
    if (feature_tile == 0 && threadIdx.x == 0) {
        positions[column + width * batch] = starts[batch] + (live ? column : max(count - 1, 0));
        if (column == 0) { counts[batch] = count; }
    }
}

void prepare_ragged_prefix_launch(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                                  const Tensor& ends, Tensor& destination, Tensor& positions,
                                  Tensor& counts, cudaStream_t stream) {
    const int vectors = source.ne[0] / 8;
    const int tiles   = (vectors + kRaggedPrefixVectorsPerBlock - 1) / kRaggedPrefixVectorsPerBlock;
    // Separate axes avoid per-CTA integer division. Grid stride preserves generic D when
    // feature tiles exceed CUDA's Z limit. Real feature widths use one vector per thread.
    const dim3 grid(source.ne[1], destination.ne[2], std::min(tiles, 65535));
    prepare_ragged_prefix_kernel<<<grid, kRaggedPrefixVectorsPerBlock, 0, stream>>>(
        static_cast<const uint4*>(source.data), static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(starts.data), static_cast<const std::int32_t*>(ends.data),
        static_cast<uint4*>(destination.data), static_cast<std::int32_t*>(positions.data),
        static_cast<std::int32_t*>(counts.data), source.ne[0] / 8, source.ne[1],
        source.nb[1] / static_cast<std::int64_t>(sizeof(uint4)),
        source.nb[2] / static_cast<std::int64_t>(sizeof(uint4)));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void prepare_ragged_prefix(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                           const Tensor& ends, Tensor& destination, Tensor& positions,
                           Tensor& counts, CudaExecutionView execution) {
    if (source.device != execution.device() || lanes.device != execution.device() ||
        starts.device != execution.device() || ends.device != execution.device() ||
        destination.device != execution.device() || positions.device != execution.device() ||
        counts.device != execution.device()) {
        throw std::invalid_argument(
            "prepare_ragged_prefix: operands must belong to the execution device");
    }
    const auto stream        = execution.stream();
    const std::int32_t width = source.ne[1];
    const std::int32_t batch = destination.ne[2];
    const auto vector_shape  = [batch](const Tensor& tensor) {
        return tensor.dtype == DType::I32 && tensor.ne[0] == batch && tensor.ne[1] == 1 &&
               tensor.ne[2] == 1 && tensor.ne[3] == 1 && tensor.is_contiguous() &&
               tensor.data != nullptr;
    };
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16 || source.ne[0] <= 0 ||
        (source.ne[0] % 8) != 0 || width <= 0 || source.ne[2] <= 0 || source.ne[3] != 1 ||
        destination.ne[0] != source.ne[0] || destination.ne[1] != width || batch <= 0 ||
        destination.ne[3] != 1 || positions.dtype != DType::I32 || positions.ne[0] != width ||
        positions.ne[1] != batch || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        !vector_shape(lanes) || !vector_shape(starts) || !vector_shape(ends) ||
        !vector_shape(counts)) {
        throw std::invalid_argument("prepare_ragged_prefix: invalid tensor geometry");
    }
    if (source.data == nullptr || destination.data == nullptr || !destination.is_contiguous() ||
        !positions.is_contiguous() || positions.data == nullptr || source.nb[0] != 2 ||
        (source.nb[1] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        (source.nb[2] % static_cast<std::int64_t>(sizeof(uint4))) != 0 ||
        ((reinterpret_cast<std::uintptr_t>(source.data) |
          reinterpret_cast<std::uintptr_t>(destination.data)) &
         15U) != 0U) {
        throw std::invalid_argument(
            "prepare_ragged_prefix: tensors must have aligned dense-column storage");
    }
    prepare_ragged_prefix_launch(source, lanes, starts, ends, destination, positions, counts,
                                 stream);
}

} // namespace ninfer::ops
