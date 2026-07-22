#include "ops/launcher/gated_delta_rule.h"

#include "ops/common/math.h"
#include "ops/kernel/gated_delta_rule_recurrent.cuh"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int HeadDim>
void launch_recurrent_fp32_typed(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                 const Tensor& beta, float scale, Tensor& ssm_state, Tensor& out,
                                 cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1,
                    static_cast<unsigned>(div_up(HeadDim, kGdnBlockDv)));
    const dim3 block(kWarpSize, kGdnNumWarps, 1);

    gated_delta_rule_recurrent_kernel<HeadDim><<<grid, block, 0, stream>>>(
        static_cast<const float*>(q.data), static_cast<const float*>(k.data),
        static_cast<const float*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<float*>(ssm_state.data),
        static_cast<float*>(out.data), T, heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_recurrent_fp32_dispatch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& g, const Tensor& beta, float scale,
                                    Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    switch (q.ne[0]) {
    case 16:
        return launch_recurrent_fp32_typed<16>(q, k, v, g, beta, scale, ssm_state, out, stream);
    case 32:
        return launch_recurrent_fp32_typed<32>(q, k, v, g, beta, scale, ssm_state, out, stream);
    case 64:
        return launch_recurrent_fp32_typed<64>(q, k, v, g, beta, scale, ssm_state, out, stream);
    case 128:
        return launch_recurrent_fp32_typed<128>(q, k, v, g, beta, scale, ssm_state, out, stream);
    default:
        throw std::invalid_argument("gated_delta_rule recurrent: unsupported head dimension");
    }
}

template <int HeadDim, bool Snapshot, bool NormalizeQK>
void launch_recurrent_bf16_typed(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                                 const Tensor& beta, float scale, const Tensor& state_read,
                                 Tensor& state_write, const Tensor* initial_slot,
                                 std::int32_t slots, Tensor& out, cudaStream_t stream) {
    const std::int64_t T = q.ne[2];
    const auto heads     = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1,
                    static_cast<unsigned>(div_up(HeadDim, kGdnBlockDv)));
    const dim3 block(kWarpSize, kGdnNumWarps, 1);
    const std::int64_t state_slot_stride = static_cast<std::int64_t>(state_write.ne[0]) *
                                           static_cast<std::int64_t>(state_write.ne[1]) *
                                           static_cast<std::int64_t>(state_write.ne[2]);
    const auto* initial =
        initial_slot == nullptr ? nullptr : static_cast<const std::int32_t*>(initial_slot->data);

    gated_delta_rule_recurrent_bf16_kernel<HeadDim, Snapshot, NormalizeQK>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
            static_cast<const float*>(beta.data), static_cast<float*>(state_read.data),
            static_cast<float*>(state_write.data), initial, static_cast<__nv_bfloat16*>(out.data),
            T, heads, scale, state_slot_stride, slots);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Snapshot, bool NormalizeQK>
void launch_recurrent_bf16_dispatch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& g, const Tensor& beta, float scale,
                                    const Tensor& state_read, Tensor& state_write,
                                    const Tensor* initial_slot, std::int32_t slots, Tensor& out,
                                    cudaStream_t stream) {
    switch (q.ne[0]) {
    case 16:
        return launch_recurrent_bf16_typed<16, Snapshot, NormalizeQK>(
            q, k, v, g, beta, scale, state_read, state_write, initial_slot, slots, out, stream);
    case 32:
        return launch_recurrent_bf16_typed<32, Snapshot, NormalizeQK>(
            q, k, v, g, beta, scale, state_read, state_write, initial_slot, slots, out, stream);
    case 64:
        return launch_recurrent_bf16_typed<64, Snapshot, NormalizeQK>(
            q, k, v, g, beta, scale, state_read, state_write, initial_slot, slots, out, stream);
    case 128:
        return launch_recurrent_bf16_typed<128, Snapshot, NormalizeQK>(
            q, k, v, g, beta, scale, state_read, state_write, initial_slot, slots, out, stream);
    default:
        throw std::invalid_argument("gated_delta_rule recurrent: unsupported head dimension");
    }
}

} // namespace

void gated_delta_rule_recurrent_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, float scale,
                                       Tensor& ssm_state, Tensor& out, cudaStream_t stream) {
    launch_recurrent_fp32_dispatch(q, k, v, g, beta, scale, ssm_state, out, stream);
}

void gated_delta_rule_recurrent_bf16_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                            const Tensor& g, const Tensor& beta, float scale,
                                            bool normalize_qk, Tensor& ssm_state, Tensor& out,
                                            cudaStream_t stream) {
    if (normalize_qk) {
        launch_recurrent_bf16_dispatch<false, true>(q, k, v, g, beta, scale, ssm_state, ssm_state,
                                                    nullptr, 1, out, stream);
    } else {
        launch_recurrent_bf16_dispatch<false, false>(q, k, v, g, beta, scale, ssm_state, ssm_state,
                                                     nullptr, 1, out, stream);
    }
}

void gated_delta_rule_recurrent_inout_bf16_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                                  const Tensor& g, const Tensor& beta, float scale,
                                                  bool normalize_qk, const Tensor& ssm_state_in,
                                                  Tensor& ssm_state_out, Tensor& out,
                                                  cudaStream_t stream) {
    if (normalize_qk) {
        launch_recurrent_bf16_dispatch<false, true>(q, k, v, g, beta, scale, ssm_state_in,
                                                    ssm_state_out, nullptr, 1, out, stream);
    } else {
        launch_recurrent_bf16_dispatch<false, false>(q, k, v, g, beta, scale, ssm_state_in,
                                                     ssm_state_out, nullptr, 1, out, stream);
    }
}

void gated_delta_rule_recurrent_snapshot_bf16_launch(const Tensor& q, const Tensor& k,
                                                     const Tensor& v, const Tensor& g,
                                                     const Tensor& beta, float scale,
                                                     bool normalize_qk, Tensor& ssm_states,
                                                     const Tensor& initial_slot, Tensor& out,
                                                     cudaStream_t stream) {
    const std::int32_t slots = ssm_states.ne[3];
    if (normalize_qk) {
        launch_recurrent_bf16_dispatch<true, true>(q, k, v, g, beta, scale, ssm_states, ssm_states,
                                                   &initial_slot, slots, out, stream);
    } else {
        launch_recurrent_bf16_dispatch<true, false>(q, k, v, g, beta, scale, ssm_states, ssm_states,
                                                    &initial_slot, slots, out, stream);
    }
}

} // namespace ninfer::ops::detail
