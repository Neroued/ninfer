#include "kernels/gated_delta_rule/launch.h"

#include "kernels/common/math.h"
#include "kernels/gated_delta_rule/recurrent.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::kernels::detail {

void gated_delta_rule_recurrent_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, float scale,
                                       Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const int n_block_dv = div_up(q.ne[0], kGdnBlockDv);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(n_block_dv));
    const dim3 block(kWarpSize, kGdnNumWarps, 1);

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
    const int n_block_dv = div_up(q.ne[0], kGdnBlockDv);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(n_block_dv));
    const dim3 block(kWarpSize, kGdnNumWarps, 1);

    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(ssm_state.ne[0]) * static_cast<std::int64_t>(ssm_state.ne[1]) *
        static_cast<std::int64_t>(ssm_state.ne[2]);

    gated_delta_rule_recurrent_bf16_kernel<128, false><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<float*>(ssm_state.data),
        static_cast<float*>(ssm_state.data), nullptr, static_cast<__nv_bfloat16*>(out.data), T,
        heads, scale, state_slot_stride, 1);
    CUDA_CHECK(cudaGetLastError());
}

void gated_delta_rule_recurrent_inout_bf16_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                                  const Tensor& g, const Tensor& beta, float scale,
                                                  const Tensor& ssm_state_in, Tensor& ssm_state_out,
                                                  Tensor& out, cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const int n_block_dv = div_up(q.ne[0], kGdnBlockDv);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(n_block_dv));
    const dim3 block(kWarpSize, kGdnNumWarps, 1);

    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(ssm_state_out.ne[0]) *
        static_cast<std::int64_t>(ssm_state_out.ne[1]) *
        static_cast<std::int64_t>(ssm_state_out.ne[2]);

    gated_delta_rule_recurrent_bf16_kernel<128, false><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<float*>(const_cast<void*>(ssm_state_in.data)),
        static_cast<float*>(ssm_state_out.data), nullptr, static_cast<__nv_bfloat16*>(out.data), T,
        heads, scale, state_slot_stride, 1);
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
    const int n_block_dv = div_up(q.ne[0], kGdnBlockDv);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(n_block_dv));
    const dim3 block(kWarpSize, kGdnNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(ssm_states.ne[0]) * static_cast<std::int64_t>(ssm_states.ne[1]) *
        static_cast<std::int64_t>(ssm_states.ne[2]);
    const std::int32_t slots = ssm_states.ne[3];

    gated_delta_rule_recurrent_bf16_kernel<128, true><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<float*>(ssm_states.data),
        static_cast<float*>(ssm_states.data),
        static_cast<const std::int32_t*>(initial_slot.data), static_cast<__nv_bfloat16*>(out.data),
        T, heads, scale, state_slot_stride, slots);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
