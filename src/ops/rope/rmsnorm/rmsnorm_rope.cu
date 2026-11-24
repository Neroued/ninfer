#include "ninfer/ops/rmsnorm_rope.h"
#include "ops/rope/rmsnorm/kernel.cuh"
#include "ops/common/host/operands.h"
#include "core/device.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

template <bool Pair>
void launch_fixed(const Tensor& positions, const Tensor* q_norm_weight, const Tensor& k_norm_weight,
                  Tensor* q, Tensor& k, std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(tokens, Pair ? 5 : 1);
    detail::rmsnorm_rope_d128_kernel<Pair><<<grid, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q_norm_weight == nullptr ? nullptr : static_cast<const __nv_bfloat16*>(q_norm_weight->data),
        static_cast<const __nv_bfloat16*>(k_norm_weight.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        static_cast<__nv_bfloat16*>(k.data));
}

constexpr std::int32_t kHeadDim       = 128;
constexpr std::int32_t kQueryHeads    = 32;
constexpr std::int32_t kKeyHeads      = 8;
constexpr std::int32_t kMaximumBatch  = 8;
constexpr std::int32_t kMaximumSingle = 2048;

void require_tensor(const Tensor& tensor, DType dtype, const std::array<std::int32_t, 4>& shape,
                    const char* label, Device device) {
    common::operand::device(tensor.device, device, "rmsnorm_rope", label);
    bool shape_matches = true;
    for (std::size_t dim = 0; dim < shape.size(); ++dim) {
        shape_matches = shape_matches && tensor.ne[dim] == shape[dim];
    }
    if (tensor.dtype != dtype || !shape_matches || !tensor.is_contiguous() ||
        !common::operand::aligned(tensor.data, 4)) {
        throw std::invalid_argument(std::string("rmsnorm_rope: invalid ") + label);
    }
}

bool overlaps(const Tensor& first, const Tensor& second) {
    return common::operand::overlaps(
        common::operand::range(first.data, first.bytes(), "rmsnorm_rope"),
        common::operand::range(second.data, second.bytes(), "rmsnorm_rope"));
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
                  Tensor& q, Tensor& k, CudaExecutionView execution) {
    const auto stream        = execution.stream();
    const std::int32_t batch = q.ne[3];
    const std::int32_t width = q.ne[2];
    if (width < 2 || width > 16) throw std::invalid_argument("rmsnorm_rope: pair W must be 2..16");
    if (batch < 1 || batch > kMaximumBatch) {
        throw std::invalid_argument("rmsnorm_rope: pair B must be 1..8");
    }
    require_tensor(q, DType::BF16, {kHeadDim, kQueryHeads, width, batch}, "q", execution.device());
    require_tensor(k, DType::BF16, {kHeadDim, kKeyHeads, width, batch}, "k", execution.device());
    require_tensor(q_norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "q norm weight",
                   execution.device());
    require_tensor(k_norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "k norm weight",
                   execution.device());
    require_tensor(positions, DType::I32, {width, batch, 1, 1}, "positions", execution.device());
    require_pair_nonoverlap(positions, q_norm_weight, k_norm_weight, q, k);
    launch_fixed<true>(positions, &q_norm_weight, k_norm_weight, &q, k, width * batch, stream);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_rope(const Tensor& positions, const Tensor& norm_weight, Tensor& x,
                  CudaExecutionView execution) {
    const auto stream         = execution.stream();
    const std::int32_t tokens = x.ne[2];
    if (tokens < 1 || tokens > kMaximumSingle) {
        throw std::invalid_argument("rmsnorm_rope: single T must be 1..2048");
    }
    require_tensor(x, DType::BF16, {kHeadDim, kKeyHeads, tokens, 1}, "x", execution.device());
    require_tensor(norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "norm weight",
                   execution.device());
    require_tensor(positions, DType::I32, {tokens, 1, 1, 1}, "positions", execution.device());
    require_single_nonoverlap(positions, norm_weight, x);
    launch_fixed<false>(positions, nullptr, norm_weight, nullptr, x, tokens, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
