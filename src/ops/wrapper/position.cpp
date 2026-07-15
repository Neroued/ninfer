#include "ninfer/ops/position.h"

#include "ops/launcher/position.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

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

void fill_i32_positions(Tensor& positions, std::int32_t start, cudaStream_t stream) {
    require_i32_vector(positions, "fill_i32_positions positions");
    if (start < 0 || start > std::numeric_limits<std::int32_t>::max() - positions.ne[0]) {
        throw std::invalid_argument("fill_i32_positions: range is outside non-negative I32");
    }
    detail::fill_i32_positions_launch(positions, start, stream);
}

void offset_i32_positions(const Tensor& source, const Tensor& delta, Tensor& destination,
                          cudaStream_t stream) {
    require_i32_vector(source, "offset_i32_positions source");
    require_i32_scalar(delta, "offset_i32_positions delta");
    require_i32_vector(destination, "offset_i32_positions destination");
    if (source.ne[0] != destination.ne[0]) {
        throw std::invalid_argument("offset_i32_positions: source and destination shapes differ");
    }
    if (delta.data == destination.data) {
        throw std::invalid_argument("offset_i32_positions: delta must not alias destination");
    }
    detail::offset_i32_positions_launch(source, delta, destination, stream);
}

} // namespace ninfer::ops
