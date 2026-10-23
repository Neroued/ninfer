#pragma once

#include "ops/mhc/kernel/residual.cuh"
#include "ops/common/cuda/reduce.cuh"
#include <cooperative_groups.h>

namespace ninfer::ops::detail::mhc {
// A small group of CTAs shares one token's RMS reduction. All mixed streams
// cross the BF16 boundary in registers, and only the final hidden is published.
template <int Cluster, int Threads, int Vector>
__global__ void __cluster_dims__(Cluster, 1, 1) post_mean_norm_cluster_kernel(Operands o) {
    constexpr int Hidden = kHidden / Cluster, Groups = Hidden / (Threads * Vector);
    static_assert(Groups > 0 && Hidden % (Threads * Vector) == 0);
    auto cluster   = cooperative_groups::this_cluster();
    const int rank = cluster.block_rank(), tid = threadIdx.x, lane = tid & 31;
    const auto m = std::int64_t(blockIdx.x / Cluster);
    __shared__ float coefficients[20], sums[Threads / 32], total;
    if (tid < 4) coefficients[tid] = o.post[m * 4 + tid];
    if (tid < 16) coefficients[4 + tid] = o.mix[m * 16 + tid];
    __syncthreads();
    float values[Groups][Vector], square = 0;
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        common::Bf16Vector<Vector> residual[4];
        mixed_vectors(o, m, rank * Hidden + (group * Threads + tid) * Vector, coefficients,
                      coefficients + 4, residual);
#pragma unroll
        for (int v = 0; v < Vector; ++v) {
            float mean = 0;
#pragma unroll
            for (int s = 0; s < 4; ++s) mean += __bfloat162float(residual[s].word[v]);
            values[group][v] = mean * 0.25F;
            square           = fmaf(values[group][v], values[group][v], square);
        }
    }
    square = common::block_reduce_sum<Threads>(square, sums);
    if (tid == 0) total = square;
    auto reduced = cluster.barrier_arrive();
    common::Bf16Vector<Vector> gamma[Groups];
#pragma unroll
    for (int group = 0; group < Groups; ++group)
        gamma[group] =
            common::Bf16Vector<Vector>::load(static_cast<const __nv_bfloat16*>(o.gamma) +
                                             rank * Hidden + (group * Threads + tid) * Vector);
    cluster.barrier_wait(std::move(reduced));
    square              = lane < Cluster ? *cluster.map_shared_rank(&total, lane) : 0.F;
    square              = common::warp_sum(square);
    const float inverse = rsqrtf(square / kHidden + o.parameters.branch_epsilon);
    // Retain every CTA's shared total until all remote readers have consumed it.
    auto consumed = cluster.barrier_arrive();
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        common::Bf16Vector<Vector> output;
#pragma unroll
        for (int v = 0; v < Vector; ++v)
            output.word[v] = __float2bfloat16_rn(__bfloat162float(gamma[group].word[v]) *
                                                 values[group][v] * inverse);
        output.store(static_cast<__nv_bfloat16*>(o.output) + m * kHidden + rank * Hidden +
                     (group * Threads + tid) * Vector);
    }
    cluster.barrier_wait(std::move(consumed));
}
} // namespace ninfer::ops::detail::mhc
