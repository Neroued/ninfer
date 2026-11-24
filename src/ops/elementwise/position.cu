#include "ninfer/ops/position.h"
#include "core/device.h"
#include "ops/common/numeric/math.h"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
__global__ void fill_i32_positions_kernel(std::int32_t* positions, std::int32_t count,
                                          std::int32_t start) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { positions[i] = start + i; }
}

__global__ void offset_i32_positions_kernel(const std::int32_t* source, const std::int32_t* delta,
                                            std::int32_t* destination, std::int32_t count) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { destination[i] = source[i] + delta[0]; }
}

void require_i32_vector(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::I32 || tensor.ne[0] <= 0 || tensor.ne[1] != 1 || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a non-empty contiguous I32 vector");
    }
}

void require_i32_scalar(const Tensor& tensor, const char* name) {
    require_i32_vector(tensor, name);
    if (tensor.ne[0] != 1) {
        throw std::invalid_argument(std::string(name) + " must be an I32 scalar");
    }
}

} // namespace

void fill_i32_positions(Tensor& positions, std::int32_t start, CudaExecutionView execution) {
    if (positions.device != execution.device()) {
        throw std::invalid_argument(
            "fill_i32_positions: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    require_i32_vector(positions, "fill_i32_positions positions");
    if (start < 0 || start > std::numeric_limits<std::int32_t>::max() - positions.ne[0]) {
        throw std::invalid_argument("fill_i32_positions: range is outside non-negative I32");
    }
    constexpr int block = 256;
    const int grid      = common::div_up(positions.ne[0], block);
    fill_i32_positions_kernel<<<grid, block, 0, stream>>>(
        static_cast<std::int32_t*>(positions.data), positions.ne[0], start);
    CUDA_CHECK(cudaGetLastError());
}

void offset_i32_positions(const Tensor& source, const Tensor& delta, Tensor& destination,
                          CudaExecutionView execution) {
    if (source.device != execution.device() || delta.device != execution.device() ||
        destination.device != execution.device()) {
        throw std::invalid_argument(
            "offset_i32_positions: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    require_i32_vector(source, "offset_i32_positions source");
    require_i32_scalar(delta, "offset_i32_positions delta");
    require_i32_vector(destination, "offset_i32_positions destination");
    if (source.ne[0] != destination.ne[0]) {
        throw std::invalid_argument("offset_i32_positions: source and destination shapes differ");
    }
    if (delta.data == destination.data) {
        throw std::invalid_argument("offset_i32_positions: delta must not alias destination");
    }
    constexpr int block = 256;
    const int grid      = common::div_up(source.ne[0], block);
    offset_i32_positions_kernel<<<grid, block, 0, stream>>>(
        static_cast<const std::int32_t*>(source.data), static_cast<const std::int32_t*>(delta.data),
        static_cast<std::int32_t*>(destination.data), source.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
