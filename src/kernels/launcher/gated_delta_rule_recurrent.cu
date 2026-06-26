#include "kernels/launcher/gated_delta_rule.h"

#include "kernels/kernel/gated_delta_rule_recurrent.cuh"
#include "kernels/kernel/gdn_cast.cuh"
#include "qus/core/device.h"

#include <algorithm>
#include <cstdint>

namespace qus::kernels::detail {
namespace {

constexpr int kBlock = 256;

int linear_grid(std::int64_t n) {
    return static_cast<int>(
        std::max<std::int64_t>(1, (n + static_cast<std::int64_t>(kBlock) - 1) / kBlock));
}

} // namespace

void gdn_cast_bf16_to_f32_launch(const Tensor& in, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = in.numel();
    gdn_cast_bf16_to_f32_kernel<<<linear_grid(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in.data), static_cast<float*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void gdn_cast_f32_to_bf16_launch(const Tensor& in, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = in.numel();
    gdn_cast_f32_to_bf16_kernel<<<linear_grid(n), kBlock, 0, stream>>>(
        static_cast<const float*>(in.data), static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void gated_delta_rule_recurrent_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, float scale,
                                       Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const int n_block_dv = (q.ne[0] + kGdnBlockDv - 1) / kGdnBlockDv;
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(n_block_dv));
    const dim3 block(WARP_SIZE, kGdnNumWarps, 1);

    gated_delta_rule_recurrent_kernel<128><<<grid, block, 0, stream>>>(
        static_cast<const float*>(q.data), static_cast<const float*>(k.data),
        static_cast<const float*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<float*>(ssm_state.data),
        static_cast<float*>(out.data), T, heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
