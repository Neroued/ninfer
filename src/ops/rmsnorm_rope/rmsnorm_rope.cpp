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
constexpr std::int32_t kTextHeadDim   = 256;
// The text form has no width of its own to cap: one warp owns one head, so the only ceiling is the
// launch grid, and even the largest supported context stays four orders of magnitude below it.
constexpr std::int64_t kMaximumTextGrid       = 2147483647;
constexpr std::int32_t kMaximumTextHeadGroups = 10;

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

void require_text_nonoverlap(const Tensor& positions, const Tensor& q_norm_weight,
                             const Tensor& k_norm_weight, const Tensor& q_in, const Tensor& k_in,
                             const Tensor& q_out, const Tensor& k_out) {
    for (const Tensor* mutable_tensor : {&q_out, &k_out}) {
        for (const Tensor* other : {&q_in, &k_in, &positions, &q_norm_weight, &k_norm_weight}) {
            if (overlaps(*mutable_tensor, *other)) {
                throw std::invalid_argument("rmsnorm_rope: text output overlaps an input");
            }
        }
    }
    if (overlaps(q_out, k_out)) {
        throw std::invalid_argument("rmsnorm_rope: text outputs overlap each other");
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

void rmsnorm_rope(const Tensor& positions, const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                  const Tensor& q_in, const Tensor& k_in, Tensor& q_out, Tensor& k_out,
                  cudaStream_t stream) {
    const std::int32_t tokens      = q_in.ne[2];
    const std::int32_t query_heads = q_in.ne[1];
    const std::int32_t key_heads   = k_in.ne[1];
    if (tokens < 1 ||
        static_cast<std::int64_t>(tokens) * kMaximumTextHeadGroups > kMaximumTextGrid) {
        throw std::invalid_argument(
            "rmsnorm_rope: text T must be positive and fit the launch grid");
    }
    if (!((query_heads == 16 && key_heads == 2) || (query_heads == 24 && key_heads == 4))) {
        throw std::invalid_argument("rmsnorm_rope: text (Q,K) must be (16,2) or (24,4)");
    }
    require_tensor(q_in, DType::BF16, {kTextHeadDim, query_heads, tokens, 1}, "text q in");
    require_tensor(k_in, DType::BF16, {kTextHeadDim, key_heads, tokens, 1}, "text k in");
    require_tensor(q_out, DType::BF16, {kTextHeadDim, query_heads, tokens, 1}, "text q out");
    require_tensor(k_out, DType::BF16, {kTextHeadDim, key_heads, tokens, 1}, "text k out");
    require_tensor(q_norm_weight, DType::BF16, {kTextHeadDim, 1, 1, 1}, "text q norm weight");
    require_tensor(k_norm_weight, DType::BF16, {kTextHeadDim, 1, 1, 1}, "text k norm weight");
    require_tensor(positions, DType::I32, {tokens, 1, 1, 1}, "text positions");
    require_text_nonoverlap(positions, q_norm_weight, k_norm_weight, q_in, k_in, q_out, k_out);
    detail::rmsnorm_rope_text_launch(positions, q_norm_weight, k_norm_weight, q_in, k_in, q_out,
                                     k_out, tokens, stream);
}

} // namespace ninfer::ops
