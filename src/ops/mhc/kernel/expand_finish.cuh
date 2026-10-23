#pragma once
#include "ops/mhc/kernel/coefficients.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::mhc {
// With four identical streams, collapse = sum(pre)*embedding. Its RMS follows
// directly from the coefficient square sum, avoiding a residual reread and D reduction.
template <int Threads, int Vector = 8, int StaticParts = 0, bool Dependent = false>
__global__ void expand_finish_kernel(Operands o, const float* partial, int parts) {
    constexpr int Workers = Threads - 32, Chunk = Vector < 8 ? Vector : 8, Chunks = Vector / Chunk;
    constexpr int Groups = (kHidden + Workers * Vector - 1) / (Workers * Vector);
    const auto m         = std::int64_t(blockIdx.x);
    const int tid = threadIdx.x, warp = tid / 32;
    __shared__ float pre[4], gain;
    common::Bf16Vector<Chunk> saved_embedding[Chunks], saved_gamma[Chunks];
    if constexpr (Dependent) {
        static_assert(Workers * Vector == kHidden);
        // Both immutable inputs are independent of the projection's partials.
        if (warp != 0) {
#pragma unroll
            for (int c = 0; c < Chunks; ++c) {
                const int d        = (tid - 32) * Vector + c * Chunk;
                saved_embedding[c] = common::Bf16Vector<Chunk>::load(
                    static_cast<const __nv_bfloat16*>(o.residual) + m * kHidden + d);
                saved_gamma[c] =
                    common::Bf16Vector<Chunk>::load(static_cast<const __nv_bfloat16*>(o.gamma) + d);
            }
        }
        cudaGridDependencySynchronize();
    }
    const float value = reduce_projection<Threads, StaticParts>(partial, parts, m);
    float logit       = 0;
    if (warp == 0) {
        const float square = __shfl_sync(0xffffffffu, value, 24);
        logit              = activate_coefficients(o, value, m, pre);
        __syncwarp();
        if (tid == 0) {
            const float gate = (pre[0] + pre[1]) + (pre[2] + pre[3]);
            gain = gate * rsqrtf(gate * gate * (square / kWidth) + o.parameters.branch_epsilon);
        }
    }
    __syncthreads();
    if (warp == 0) {
        if constexpr (Dependent) {
            if (o.parameters.sinkhorn_iterations == 20)
                sinkhorn<20>(o, logit, m);
            else
                sinkhorn(o, logit, m);
        } else {
            sinkhorn(o, logit, m);
        }
    } else {
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            const int d = (group * Workers + tid - 32) * Vector;
            if (d < kHidden) {
#pragma unroll
                for (int c = 0; c < Chunks; ++c) {
                    common::Bf16Vector<Chunk> embedding, gamma;
                    if constexpr (Dependent) {
                        embedding = saved_embedding[c];
                        gamma     = saved_gamma[c];
                    } else {
                        embedding = common::Bf16Vector<Chunk>::load(
                            static_cast<const __nv_bfloat16*>(o.residual) + m * kHidden + d +
                            c * Chunk);
                        gamma = common::Bf16Vector<Chunk>::load(
                            static_cast<const __nv_bfloat16*>(o.gamma) + d + c * Chunk);
                    }
                    common::Bf16Vector<Chunk> output;
#pragma unroll
                    for (int v = 0; v < Chunk; ++v)
                        output.word[v] =
                            __float2bfloat16_rn(__bfloat162float(gamma.word[v]) *
                                                __bfloat162float(embedding.word[v]) * gain);
                    output.store(static_cast<__nv_bfloat16*>(o.output) + m * kHidden + d +
                                 c * Chunk);
                }
            }
        }
    }
}
} // namespace ninfer::ops::detail::mhc
