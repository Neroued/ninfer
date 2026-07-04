#include "kernels/launcher/gated_delta_rule.h"

#include "kernels/kernel/gated_delta_rule_recurrent.cuh"
#include "qus/core/device.h"

#include <cstdint>

namespace qus::kernels::detail {

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

void gated_delta_rule_recurrent_bf16_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                            const Tensor& g, const Tensor& beta, float scale,
                                            Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const int n_block_dv = (q.ne[0] + kGdnBlockDv - 1) / kGdnBlockDv;
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(n_block_dv));
    const dim3 block(WARP_SIZE, kGdnNumWarps, 1);

    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(ssm_state.ne[0]) * static_cast<std::int64_t>(ssm_state.ne[1]) *
        static_cast<std::int64_t>(ssm_state.ne[2]);

    gated_delta_rule_recurrent_bf16_kernel<128, false><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<float*>(ssm_state.data),
        nullptr, static_cast<__nv_bfloat16*>(out.data), T, heads, scale, state_slot_stride, 1);
    CUDA_CHECK(cudaGetLastError());
}

void gated_delta_rule_recurrent_snapshot_bf16_launch(const Tensor& q, const Tensor& k,
                                                     const Tensor& v, const Tensor& g,
                                                     const Tensor& beta, float scale,
                                                     Tensor& ssm_states,
                                                     const Tensor& initial_slot, Tensor& out,
                                                     cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const int n_block_dv = (q.ne[0] + kGdnBlockDv - 1) / kGdnBlockDv;
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(n_block_dv));
    const dim3 block(WARP_SIZE, kGdnNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(ssm_states.ne[0]) * static_cast<std::int64_t>(ssm_states.ne[1]) *
        static_cast<std::int64_t>(ssm_states.ne[2]);
    const std::int32_t slots = ssm_states.ne[3];

    gated_delta_rule_recurrent_bf16_kernel<128, true><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<float*>(ssm_states.data),
        static_cast<const std::int32_t*>(initial_slot.data), static_cast<__nv_bfloat16*>(out.data),
        T, heads, scale, state_slot_stride, slots);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
