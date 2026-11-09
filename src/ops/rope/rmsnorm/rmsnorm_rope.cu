#include "ninfer/ops/rmsnorm_rope.h"
#include "ops/rope/rmsnorm/kernel.cuh"
#include "ops/common/host/operands.h"
#include "core/device.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

template <int QueryHeads, int KeyHeads, int HeadsPerBlock, int LanesPerHead, bool WideAngles>
void launch(const Tensor& positions, const Tensor* q_norm_weight, const Tensor& k_norm_weight,
            Tensor* q, Tensor& k, int tokens, float epsilon,
            detail::RopeD128Frequencies frequencies, cudaStream_t stream) {
    constexpr int groups  = (QueryHeads + KeyHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int threads = HeadsPerBlock * LanesPerHead;
    detail::rmsnorm_rope_d128_kernel<QueryHeads, KeyHeads, HeadsPerBlock, LanesPerHead, WideAngles>
        <<<tokens * groups, threads, 0, stream>>>(
            static_cast<const std::int32_t*>(positions.data),
            q_norm_weight == nullptr ? nullptr
                                     : static_cast<const __nv_bfloat162*>(q_norm_weight->data),
            static_cast<const __nv_bfloat162*>(k_norm_weight.data),
            q == nullptr ? nullptr : static_cast<__nv_bfloat162*>(q->data),
            static_cast<__nv_bfloat162*>(k.data), epsilon, frequencies);
}

template <int QueryHeads, int KeyHeads>
void dispatch(const Tensor& positions, const Tensor* q_norm_weight, const Tensor& k_norm_weight,
              Tensor* q, Tensor& k, int tokens, float epsilon, float theta, cudaStream_t stream) {
    const auto frequencies = detail::rope_d128_frequencies(theta);
    const auto route       = [&]<bool WideAngles>() {
        if constexpr (QueryHeads > 0) {
            if (tokens < 32) {
                if constexpr (QueryHeads == 32)
                    launch<32, 8, 8, 32, WideAngles>(positions, q_norm_weight, k_norm_weight, q, k,
                                                           tokens, epsilon, frequencies, stream);
                else
                    launch<16, 4, 4, 16, WideAngles>(positions, q_norm_weight, k_norm_weight, q, k,
                                                           tokens, epsilon, frequencies, stream);
            } else
                launch<QueryHeads, KeyHeads, 20, 16, WideAngles>(positions, q_norm_weight,
                                                                       k_norm_weight, q, k, tokens,
                                                                       epsilon, frequencies, stream);
        } else {
            launch<0, KeyHeads, KeyHeads, 16, WideAngles>(positions, nullptr, k_norm_weight,
                                                                nullptr, k, tokens, epsilon, frequencies,
                                                                stream);
        }
    };
    if (theta >= 1.F)
        route.template operator()<false>();
    else
        route.template operator()<true>();
}

void require_parameters(float epsilon, float theta) {
    if (!(epsilon > 0) || !std::isfinite(epsilon) || !(theta > 0) || !std::isfinite(theta))
        throw std::invalid_argument("rmsnorm_rope: epsilon and theta must be positive and finite");
}

constexpr std::int32_t kHeadDim       = 128;
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
                  float epsilon, float theta, Tensor& q, Tensor& k, CudaExecutionView execution) {
    require_parameters(epsilon, theta);
    const auto stream        = execution.stream();
    const std::int32_t batch = q.ne[3];
    const std::int32_t width = q.ne[2];
    if (width < 2 || width > 16) throw std::invalid_argument("rmsnorm_rope: pair W must be 2..16");
    if (batch < 1 || batch > kMaximumBatch) {
        throw std::invalid_argument("rmsnorm_rope: pair B must be 1..8");
    }
    const int query_heads = q.ne[1], key_heads = k.ne[1];
    if (!((query_heads == 32 && key_heads == 8) || (query_heads == 16 && key_heads == 4)))
        throw std::invalid_argument("rmsnorm_rope: pair (Hq,Hkv) must be (32,8) or (16,4)");
    require_tensor(q, DType::BF16, {kHeadDim, query_heads, width, batch}, "q", execution.device());
    require_tensor(k, DType::BF16, {kHeadDim, key_heads, width, batch}, "k", execution.device());
    require_tensor(q_norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "q norm weight",
                   execution.device());
    require_tensor(k_norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "k norm weight",
                   execution.device());
    require_tensor(positions, DType::I32, {width, batch, 1, 1}, "positions", execution.device());
    require_pair_nonoverlap(positions, q_norm_weight, k_norm_weight, q, k);
    if (query_heads == 16)
        dispatch<16, 4>(positions, &q_norm_weight, k_norm_weight, &q, k, width * batch, epsilon,
                        theta, stream);
    else
        dispatch<32, 8>(positions, &q_norm_weight, k_norm_weight, &q, k, width * batch, epsilon,
                        theta, stream);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_rope(const Tensor& positions, const Tensor& norm_weight, float epsilon, float theta,
                  Tensor& x, CudaExecutionView execution) {
    require_parameters(epsilon, theta);
    const auto stream         = execution.stream();
    const std::int32_t tokens = x.ne[2];
    if (tokens < 1 || tokens > kMaximumSingle) {
        throw std::invalid_argument("rmsnorm_rope: single T must be 1..2048");
    }
    const int heads = x.ne[1];
    if (heads != 4 && heads != 8)
        throw std::invalid_argument("rmsnorm_rope: single Hkv must be 4 or 8");
    require_tensor(x, DType::BF16, {kHeadDim, heads, tokens, 1}, "x", execution.device());
    require_tensor(norm_weight, DType::BF16, {kHeadDim, 1, 1, 1}, "norm weight",
                   execution.device());
    require_tensor(positions, DType::I32, {tokens, 1, 1, 1}, "positions", execution.device());
    require_single_nonoverlap(positions, norm_weight, x);
    if (heads == 4)
        dispatch<0, 4>(positions, nullptr, norm_weight, nullptr, x, tokens, epsilon, theta, stream);
    else
        dispatch<0, 8>(positions, nullptr, norm_weight, nullptr, x, tokens, epsilon, theta, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
