#include "model/position.h"

#include "qus/core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::model::detail {
namespace {

constexpr int kBlock = 256;

void require_i32_contiguous_nonnull(const Tensor& t, const char* name) {
    if (t.dtype != DType::I32) {
        throw std::invalid_argument(std::string(name) + ": tensor must be I32");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(name) + ": tensor must be contiguous");
    }
    if (t.data == nullptr) {
        throw std::invalid_argument(std::string(name) + ": tensor data must be non-null");
    }
}

void require_pos_shape(const Tensor& pos, const char* name) {
    if (pos.ne[0] != 1 || pos.ne[1] != 1 || pos.ne[2] != 1 || pos.ne[3] != 1) {
        throw std::invalid_argument(std::string(name) + ": pos must have shape [1]");
    }
}

void require_vector_shape(const Tensor& t, const char* name) {
    if (t.ne[0] <= 0 || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(name) + ": positions must have shape [T]");
    }
}

__global__ void fill_positions_kernel(std::int32_t* positions, std::int32_t n, std::int32_t start) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) { positions[i] = start + i; }
}

__global__ void offset_positions_kernel(const std::int32_t* src, std::int32_t* dst, std::int32_t n,
                                        const std::int32_t* delta) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) { dst[i] = src[i] + delta[0]; }
}

__global__ void set_pos_kernel(std::int32_t* pos, std::int32_t value) { pos[0] = value; }

__global__ void advance_pos_kernel(std::int32_t* pos) { ++pos[0]; }

} // namespace

void copy_i32(const std::int32_t* src, Tensor& dst, cudaStream_t stream) {
    if (src == nullptr) { throw std::invalid_argument("copy_i32: source is null"); }
    require_i32_contiguous_nonnull(dst, "copy_i32 dst");
    CUDA_CHECK(cudaMemcpyAsync(dst.data, src, dst.bytes(), cudaMemcpyHostToDevice, stream));
}

void fill_positions(Tensor& positions, int start, cudaStream_t stream) {
    require_i32_contiguous_nonnull(positions, "fill_positions");
    require_vector_shape(positions, "fill_positions");
    if (start < 0) { throw std::invalid_argument("fill_positions: start must be nonnegative"); }
    const int n = positions.ne[0];
    if (start > std::numeric_limits<std::int32_t>::max() - n) {
        throw std::overflow_error("fill_positions: range exceeds int32");
    }
    const int grid = (n + kBlock - 1) / kBlock;
    fill_positions_kernel<<<grid, kBlock, 0, stream>>>(static_cast<std::int32_t*>(positions.data),
                                                       n, start);
    CUDA_CHECK(cudaGetLastError());
}

void offset_positions(const Tensor& src, const Tensor& delta, Tensor& dst, cudaStream_t stream) {
    require_i32_contiguous_nonnull(src, "offset_positions src");
    require_i32_contiguous_nonnull(delta, "offset_positions delta");
    require_i32_contiguous_nonnull(dst, "offset_positions dst");
    require_vector_shape(src, "offset_positions src");
    require_pos_shape(delta, "offset_positions delta");
    require_vector_shape(dst, "offset_positions dst");
    if (src.ne[0] != dst.ne[0]) {
        throw std::invalid_argument("offset_positions: shapes must match");
    }
    const int n    = src.ne[0];
    const int grid = (n + kBlock - 1) / kBlock;
    offset_positions_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(src.data), static_cast<std::int32_t*>(dst.data), n,
        static_cast<const std::int32_t*>(delta.data));
    CUDA_CHECK(cudaGetLastError());
}

void set_pos(Tensor& pos, int value, cudaStream_t stream) {
    require_i32_contiguous_nonnull(pos, "set_pos");
    require_pos_shape(pos, "set_pos");
    set_pos_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(pos.data), value);
    CUDA_CHECK(cudaGetLastError());
}

void advance_pos(Tensor& pos, cudaStream_t stream) {
    require_i32_contiguous_nonnull(pos, "advance_pos");
    require_pos_shape(pos, "advance_pos");
    advance_pos_kernel<<<1, 1, 0, stream>>>(static_cast<std::int32_t*>(pos.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::model::detail
