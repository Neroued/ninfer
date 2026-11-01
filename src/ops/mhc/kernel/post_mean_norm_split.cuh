#pragma once
#include "ops/mhc/kernel/residual.cuh"
#include "ops/common/cuda/reduce.cuh"

namespace ninfer::ops::detail::mhc {
// Small-column decomposition spreads the post and mean over SMs. The scratch mean
// remains FP32; only post's four mixed stream values cross the BF16 boundary.
template <int Threads, int V>
__global__ void post_mean_norm_partial_kernel(Operands o, float* work) {
    constexpr int H = Threads * V, Parts = kHidden / H;
    auto m   = std::int64_t(blockIdx.x / Parts);
    int part = blockIdx.x % Parts, tid = threadIdx.x, d = part * H + tid * V;
    __shared__ float sums[Threads / 32];
    common::Bf16Vector<V> residual[4];
    mixed_vectors(o, m, d, o.post + m * 4, o.mix + m * 16, residual);
    float values[V], ss = 0;
#pragma unroll
    for (int v = 0; v < V; ++v) {
        float mean = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) mean += __bfloat162float(residual[j].word[v]);
        values[v]                 = mean * 0.25F;
        ss                        = fmaf(values[v], values[v], ss);
        work[m * kHidden + d + v] = values[v];
    }
    ss = common::block_reduce_sum<Threads>(ss, sums);
    if (tid == 0) work[std::int64_t(o.columns) * kHidden + m * Parts + part] = ss;
}

template <int Parts, int Threads, int V>
__global__ void post_mean_norm_finish_kernel(Operands o, const float* work) {
    constexpr int H = Threads * V;
    auto m          = std::int64_t(blockIdx.x / (kHidden / H));
    int d = (blockIdx.x % (kHidden / H)) * H + threadIdx.x * V, lane = threadIdx.x % 32;
    float ss = 0;
#pragma unroll
    for (int i = lane; i < Parts; i += 32)
        ss += work[std::int64_t(o.columns) * kHidden + m * Parts + i];
    ss               = common::warp_sum(ss);
    float inv        = rsqrtf(ss / kHidden + o.parameters.branch_epsilon);
    const auto gamma = common::Bf16Vector<V>::load(static_cast<const __nv_bfloat16*>(o.gamma) + d);
    common::Bf16Vector<V> output;
#pragma unroll
    for (int v = 0; v < V; ++v)
        output.word[v] =
            __float2bfloat16_rn(__bfloat162float(gamma.word[v]) * work[m * kHidden + d + v] * inv);
    output.store(static_cast<__nv_bfloat16*>(o.output) + m * kHidden + d);
}
} // namespace ninfer::ops::detail::mhc
