#pragma once
#include "ops/mhc/kernel/coefficients.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::mhc {
template <int Vector>
__device__ __forceinline__ void load_residual_words(Operands o, std::int64_t m, int d,
                                                    unsigned (&words)[4][Vector / 2]) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int v = 0; v < Vector / 8; ++v) {
            const auto* source = static_cast<const __nv_bfloat16*>(o.residual) + m * kWidth +
                                 s * kHidden + d + v * 8;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(words[s][v * 4]), "=r"(words[s][v * 4 + 1]),
                           "=r"(words[s][v * 4 + 2]), "=r"(words[s][v * 4 + 3])
                         : "l"(source));
        }
    }
}

template <int Threads, int Vector, int StaticParts = 0, bool Dependent = false,
          bool PreloadResidual = false, bool StreamingOutput = false>
__global__ void pre_finish_kernel(Operands o, const float* partial, int parts) {
    static_assert(Threads >= 64 && Threads <= 1024 && Threads % 32 == 0);
    constexpr int Warps = Threads / 32, Workers = Threads - 32;
    constexpr int Chunk = Vector < 8 ? Vector : 8;
    static_assert(Vector <= 8 || Workers * Vector == kHidden);
    constexpr int Groups   = (kHidden + Workers * Vector - 1) / (Workers * Vector);
    constexpr int SumWidth = [] {
        int width = 1;
        while (width < Warps - 1) width *= 2;
        return width;
    }();
    const auto m  = std::int64_t(blockIdx.x);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    __shared__ float pre[4], sums[Warps], inverse;
    unsigned words[4][Vector / 2];
    if constexpr (PreloadResidual) {
        static_assert(Dependent && Workers * Vector == kHidden && Vector % 8 == 0);
        // Only the immutable pre input may be read before the projection completes.
        if (warp != 0) load_residual_words<Vector>(o, m, (tid - 32) * Vector, words);
    }
    if constexpr (Dependent) cudaGridDependencySynchronize();
    const float projection = reduce_projection<Threads, StaticParts>(partial, parts, m);
    float logit            = 0;
    if (warp == 0) logit = activate_coefficients(o, projection, m, pre);
    __syncthreads();
    float values[Groups][Vector], square = 0;
    if (warp == 0) {
        if constexpr (Dependent) {
            if (o.parameters.sinkhorn_iterations == 20)
                sinkhorn<20>(o, logit, m);
            else
                sinkhorn(o, logit, m);
        } else {
            sinkhorn(o, logit, m);
        }
    } else if constexpr (Workers * Vector == kHidden && Vector % 8 == 0) {
        // One packed vector per worker covers D exactly; no tail group or scalar loads.
        const int d = (tid - 32) * Vector;
        if constexpr (!PreloadResidual) load_residual_words<Vector>(o, m, d, words);
#pragma unroll
        for (int j = 0; j < Vector / 2; ++j) {
            float2 value = {};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const auto pair = common::bf16x2_bits_to_float2(words[s][j]);
                value.x         = fmaf(pre[s], pair.x, value.x);
                value.y         = fmaf(pre[s], pair.y, value.y);
            }
            values[0][2 * j]     = value.x;
            values[0][2 * j + 1] = value.y;
            square               = fmaf(value.x, value.x, square);
            square               = fmaf(value.y, value.y, square);
        }
    } else {
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            const int d = (group * Workers + tid - 32) * Vector;
            if (d < kHidden) {
                common::Bf16Vector<Vector> x[4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    x[i] = common::Bf16Vector<Vector>::load(
                        static_cast<const __nv_bfloat16*>(o.residual) + m * kWidth + i * kHidden +
                        d);
#pragma unroll
                for (int v = 0; v < Vector; ++v) {
                    float value = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        value = fmaf(pre[i], __bfloat162float(x[i].word[v]), value);
                    values[group][v] = value;
                    square           = fmaf(value, value, square);
                }
            }
        }
    }
    if (warp != 0) {
        // Barrier 1 includes only the collapse workers. Warp 0 can finish Sinkhorn
        // independently while these warps reduce the RMS and publish block_input.
        square = common::warp_reduce_sum(square);
        if (lane == 0) sums[warp - 1] = square;
        asm volatile("barrier.sync 1, %0;" ::"r"(Workers) : "memory");
        if (warp == 1) {
            square = lane < Warps - 1 ? sums[lane] : 0.F;
            square = common::warp_reduce_sum<SumWidth>(square);
            if (lane == 0) inverse = rsqrtf(square / kHidden + o.parameters.branch_epsilon);
        }
        asm volatile("barrier.sync 1, %0;" ::"r"(Workers) : "memory");
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            const int d = (group * Workers + tid - 32) * Vector;
            if (d < kHidden) {
#pragma unroll
                for (int c = 0; c < Vector / Chunk; ++c) {
                    // Tensor bases promise 16-byte alignment, including wide worker tiles.
                    const auto gamma = common::Bf16Vector<Chunk>::load(
                        static_cast<const __nv_bfloat16*>(o.gamma) + d + c * Chunk);
                    common::Bf16Vector<Chunk> out;
#pragma unroll
                    for (int v = 0; v < Chunk; ++v)
                        out.word[v] = __float2bfloat16_rn(__bfloat162float(gamma.word[v]) *
                                                          values[group][c * Chunk + v] * inverse);
                    auto* target =
                        static_cast<__nv_bfloat16*>(o.output) + m * kHidden + d + c * Chunk;
                    if constexpr (StreamingOutput) {
                        // Keep the medium-token working set of inputs and partials cached.
                        static_assert(Chunk == 8);
                        const auto words = common::load_vec<uint4>(&out);
                        asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(target),
                                     "r"(words.x), "r"(words.y), "r"(words.z), "r"(words.w)
                                     : "memory");
                    } else {
                        out.store(target);
                    }
                }
            }
        }
    }
}
} // namespace ninfer::ops::detail::mhc
