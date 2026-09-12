#include "ninfer/ops/rmsnorm_rope.h"

#include "ops/rmsnorm_rope/launch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim       = 128;
constexpr std::int32_t kQueryHeads    = 32;
constexpr std::int32_t kKeyHeads      = 8;
constexpr std::int32_t kMaximumBatch  = 8;
constexpr std::int32_t kMaximumSingle = 2048;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_tensor(const Tensor& tensor, DType dtype, const std::array<std::int32_t, 4>& shape,
                    const char* label) {
    bool shape_matches = true;
    for (std::size_t dim = 0; dim < shape.size(); ++dim) {
        shape_matches = shape_matches && tensor.ne[dim] == shape[dim];
    }
    if (tensor.dtype != dtype || !shape_matches || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 4)) {
        throw std::invalid_argument(std::string("rmsnorm_rope: invalid ") + label);
    }
}

bool overlaps(const Tensor& first, const Tensor& second) {
    const auto first_begin  = reinterpret_cast<std::uintptr_t>(first.data);
    const auto second_begin = reinterpret_cast<std::uintptr_t>(second.data);
    return first_begin < second_begin + second.bytes() &&
           second_begin < first_begin + first.bytes();
}

void require_pair_nonoverlap(const Tensor& positions, const Tensor& q_norm_weight,
                             const Tensor& k_norm_weight, const Tensor& q, const Tensor& k) {
    if (overlaps(q, k) || overlaps(q, positions) || overlaps(q, q_norm_weight) ||
        overlaps(q, k_norm_weight) || overlaps(k, positions) || overlaps(k, q_norm_weight) ||
        overlaps(k, k_norm_weight)) {
        throw std::invalid_argument("rmsnorm_rope: pair mutable tensors overlap another operand");
    }
}

void require_single_nonoverlap(const Tensor& positions, const Tensor& norm_weight,
                               const Tensor& x) {
    if (overlaps(x, positions) || overlaps(x, norm_weight)) {
        throw std::invalid_argument(
            "rmsnorm_rope: single mutable tensor overlaps a read-only input");
    }
}

} // namespace

void rmsnorm_rope(const Tensor& positions, const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                  Tensor& q, Tensor& k, cudaStream_t stream) {
    const std::int32_t batch = q.ne[3];
    const std::int32_t width = q.ne[2];
    if (width < 2 || width > 16) throw std::invalid_argument("rmsnorm_rope: pair W must be 2..16");
    if (batch < 1 || batch > kMaximumBatch) {
        throw std::invalid_argument("rmsnorm_rope: pair B must be 1..8");
    }
    require_tensor(q, DType::BF16, {kHeadDim, kQueryHeads, width, batch}, "q");
    require_tensor(k, DType::BF16, {kHeadDim, kKeyHeads, width, batch}, "k");
    require_tensor(q_norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "q norm weight");
    require_tensor(k_norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "k norm weight");
    require_tensor(positions, DType::I32, {width, batch, 1, 1}, "positions");
    require_pair_nonoverlap(positions, q_norm_weight, k_norm_weight, q, k);
    detail::rmsnorm_rope_pair_launch(positions, q_norm_weight, k_norm_weight, q, k, width * batch,
                                     stream);
}

void rmsnorm_rope(const Tensor& positions, const Tensor& norm_weight, Tensor& x,
                  cudaStream_t stream) {
    const std::int32_t tokens = x.ne[2];
    if (tokens < 1 || tokens > kMaximumSingle) {
        throw std::invalid_argument("rmsnorm_rope: single T must be 1..2048");
    }
    require_tensor(x, DType::BF16, {kHeadDim, kKeyHeads, tokens, 1}, "x");
    require_tensor(norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "norm weight");
    require_tensor(positions, DType::I32, {tokens, 1, 1, 1}, "positions");
    require_single_nonoverlap(positions, norm_weight, x);
    detail::rmsnorm_rope_single_launch(positions, norm_weight, x, tokens, stream);
}

} // namespace ninfer::ops
