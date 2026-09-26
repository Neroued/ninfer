#include "ops/linear_attention/kimi_delta_attention/launch.h"
#include "ops/linear_attention/kimi_delta_attention/recurrent.cuh"

namespace ninfer::ops::detail::kimi_delta_attention {

void launch_recurrent(const Arguments& a, cudaStream_t stream) {
    const dim3 grid(a.value_heads, 1, kStateDim / kBlockDv);
    const dim3 block(kWarpSize, kNumWarps);
    recurrent_direct_kernel<<<grid, block, 0, stream>>>(
        a.q, a.k, a.v, a.g, a.beta, a.a_log, a.dt_bias, a.state_in, a.state_out, a.out, a.qk_heads,
        a.value_heads, a.tokens, a.lower_bound, a.scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_batch_update(const Arguments& a, const std::int32_t* slots, int batch,
                         cudaStream_t stream) {
    const dim3 grid(a.value_heads, batch, kStateDim / kBlockDv);
    const dim3 block(kWarpSize, kNumWarps);
    recurrent_batch_update_kernel<<<grid, block, 0, stream>>>(
        a.q, a.k, a.v, a.g, a.beta, a.a_log, a.dt_bias, a.state_out, slots, a.out, a.qk_heads,
        a.value_heads, static_cast<std::int64_t>(kStateDim) * kStateDim * a.value_heads,
        a.lower_bound, a.scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::kimi_delta_attention
