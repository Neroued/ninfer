#include "ops/linear_attention/kimi_delta_attention/launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/linear_attention/kimi_delta_attention/recurrent.cuh"

namespace ninfer::ops::detail::kimi_delta_attention {

void launch_recurrent(const Arguments& a, cudaStream_t stream) {
    const dim3 grid(a.value_heads, 1, kStateDim / kBlockDv);
    const dim3 block(common::kWarpSize, kNumWarps);
    recurrent_direct_kernel<<<grid, block, 0, stream>>>(
        a.q, a.k, a.v, a.g, a.beta, a.a_log, a.dt_bias, a.state_in, a.state_out, a.out, a.qk_heads,
        a.value_heads, a.tokens, a.lower_bound, a.scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_batch_update(const Arguments& a, const std::int64_t* offsets, int batch,
                         cudaStream_t stream) {
    const dim3 grid(a.value_heads, batch, kStateDim / kBlockDv);
    const dim3 block(common::kWarpSize, kNumWarps);
    recurrent_batch_update_kernel<<<grid, block, 0, stream>>>(
        a.q, a.k, a.v, a.g, a.beta, a.a_log, a.dt_bias, a.state_out, offsets, a.out, a.qk_heads,
        a.value_heads, a.lower_bound, a.scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::kimi_delta_attention
