#pragma once

#include "ops/mhc/kernel/residual.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/mhc/kernel/projection_math.cuh"

namespace ninfer::ops::detail::mhc {
// A CTA owns all streams and projection rows for its hidden/token tile. In-place
// post-pre reads every old stream before publishing the BF16 words used by the MMA.
template <Form F, int Hidden, int Tokens>
__launch_bounds__(ProjectionSchedule<Tokens>::kThreads) __global__
    void projection_mma_kernel(Operands o, float* partial, std::int64_t tile_offset) {
    using Schedule   = ProjectionSchedule<Tokens>;
    constexpr int BK = 64, Threads = Schedule::kThreads, Warps = Threads / 32;
    constexpr int Parts = kHidden / Hidden, Groups = Tokens / Warps;
    const auto tile = tile_offset + blockIdx.x;
    const auto m0   = (tile / Parts) * Tokens;
    const int part = tile % Parts, d0 = part * Hidden;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    __shared__ __align__(16) __nv_bfloat16 a[4][32 * BK], b[4][Tokens * BK];
    __shared__ float coefficients[F == Form::PostPre ? Tokens : 1][20];
    if constexpr (F == Form::PostPre) {
        for (int index = tid; index < Tokens * 20; index += Threads) {
            const int token = index / 20, c = index % 20;
            float value = 0;
            if (m0 + token < o.columns)
                value = c < 4 ? o.post[(m0 + token) * 4 + c] : o.mix[(m0 + token) * 16 + c - 4];
            coefficients[token][c] = value;
        }
        __syncthreads();
    }
    float accum[Schedule::kMmaRows][Schedule::kMmaTokens][4] = {};
    float squares[Groups]                                    = {};
    for (int h = 0; h < Hidden; h += BK) {
        // Padding the 24 native F rows stays on chip; weights retain their native layout.
        for (int index = tid; index < 4 * 32 * (BK / 8); index += Threads) {
            const int stream = index / (32 * (BK / 8));
            const int row = (index / (BK / 8)) % 32, col = (index % (BK / 8)) * 8;
            const int source = (row < 24 ? row : 0) * kWidth + stream * kHidden + d0 + h + col;
            common::cp_async_zfill<16, Cache::ca>(
                a[stream] + row * BK + projection_shared_column(row, col),
                static_cast<const __nv_bfloat16*>(o.projection) + source, row < 24 ? 16 : 0);
        }
        common::cp_commit();
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            const int token = warp + group * Warps, d = d0 + h + lane * 2;
            common::Bf16Vector<2> values[4] = {};
            if (m0 + token < o.columns) {
                if constexpr (F == Form::PostPre) {
                    mixed_vectors(o, m0 + token, d, coefficients[token], coefficients[token] + 4,
                                  values);
                } else if constexpr (F == Form::ExpandPre) {
                    const auto value = common::Bf16Vector<2>::load(
                        static_cast<const __nv_bfloat16*>(o.residual) + (m0 + token) * kHidden + d);
#pragma unroll
                    for (int stream = 0; stream < 4; ++stream) values[stream] = value;
                } else {
#pragma unroll
                    for (int stream = 0; stream < 4; ++stream)
                        values[stream] = common::Bf16Vector<2>::load(
                            static_cast<const __nv_bfloat16*>(o.residual) + (m0 + token) * kWidth +
                            stream * kHidden + d);
                }
                if constexpr (F != Form::Pre) {
#pragma unroll
                    for (int stream = 0; stream < 4; ++stream)
                        if constexpr (F == Form::ExpandPre) {
                            // Expanded residuals are not read by this Op's finish. Preserve
                            // the embedding, weights and partials in the cache instead.
                            auto* target = static_cast<__nv_bfloat16*>(o.updated) +
                                           (m0 + token) * kWidth + stream * kHidden + d;
                            const auto word = common::load_vec<unsigned>(&values[stream]);
                            asm volatile("st.global.cs.u32 [%0], %1;" ::"l"(target), "r"(word)
                                         : "memory");
                        } else {
                            values[stream].store(static_cast<__nv_bfloat16*>(o.updated) +
                                                 (m0 + token) * kWidth + stream * kHidden + d);
                        }
                }
            }
#pragma unroll
            for (int stream = 0; stream < 4; ++stream) {
                values[stream].store(b[stream] + token * BK +
                                     projection_shared_column(token, lane * 2));
#pragma unroll
                for (int v = 0; v < 2; ++v) {
                    const float value = __bfloat162float(values[stream].word[v]);
                    squares[group]    = fmaf(value, value, squares[group]);
                }
            }
        }
        common::cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int stream = 0; stream < 4; ++stream)
            projection_mma_stage<Tokens>(a[stream], b[stream], accum[0][0], warp, lane);
        __syncthreads();
    }
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        const auto m    = m0 + warp + group * Warps;
        const float sum = common::warp_sum(squares[group]);
        if (lane == 0 && m < o.columns) partial[(m * Parts + part) * 32 + 24] = sum;
    }
    const int row    = (warp / Schedule::kWarpsTokens) * 16 + (lane >> 2);
    const auto token = m0 + (warp % Schedule::kWarpsTokens) * 8 + 2 * (lane & 3);
    const auto store = [&](int r, std::int64_t m, float value) {
        if (r < 24 && m < o.columns) partial[(m * Parts + part) * 32 + r] = value;
    };
    const auto& v = accum[0][0];
    store(row, token, v[0]);
    store(row, token + 1, v[1]);
    store(row + 8, token, v[2]);
    store(row + 8, token + 1, v[3]);
}
} // namespace ninfer::ops::detail::mhc
