#pragma once

#include "ops/mhc/kernel/residual.cuh"
#include "ops/mhc/kernel/projection_math.cuh"
#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::detail::mhc {
// Stage a complete hidden partition at once. Loading uses the full CTA, while
// two MMA warps produce the 24 coefficients and the residual square sum in the
// unused eight rows. Each CTA owns every residual stream, including in-place writes.
template <Form F, int Hidden, int Threads>
__global__ __launch_bounds__(Threads) void projection_partition_kernel(Operands o, float* partial,
                                                                       std::int64_t tile_offset) {
    constexpr int Tokens = 8, Parts = kHidden / Hidden, Vector = Tokens * Hidden / Threads;
    static_assert(Hidden % 64 == 0 && kHidden % Hidden == 0);
    static_assert(Vector == 2 || Vector == 4 || Vector == 8);
    // The dependent finish can prepare independent inputs while this grid projects.
    // Its grid dependency wait owns visibility of partials and updated residuals.
    cudaTriggerProgrammaticLaunchCompletion();
    __shared__ __align__(16) __nv_bfloat16 a[4][24 * Hidden], x[4][Tokens * Hidden];
    __shared__ float coefficients[F == Form::PostPre ? Tokens : 1][20];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
    const auto tile = tile_offset + blockIdx.x, m0 = tile / Parts * Tokens;
    const int part = tile % Parts, d0 = part * Hidden;
    for (int i = tid; i < 4 * 24 * (Hidden / 8); i += Threads) {
        const int s = i / (24 * (Hidden / 8)), row = i / (Hidden / 8) % 24;
        const int col = i % (Hidden / 8) * 8;
        common::cp_async_zfill<16, Cache::ca>(
            a[s] + row * Hidden + projection_shared_column(row, col),
            static_cast<const __nv_bfloat16*>(o.projection) + row * kWidth + s * kHidden + d0 + col,
            16);
    }
    common::cp_commit();
    if constexpr (F == Form::PostPre) {
        for (int i = tid; i < Tokens * 20; i += Threads) {
            const auto m = m0 + i / 20;
            const int c  = i % 20;
            coefficients[i / 20][c] =
                m < o.columns ? (c < 4 ? o.post[m * 4 + c] : o.mix[m * 16 + c - 4]) : 0.F;
        }
        __syncthreads();
    }
    if constexpr (F == Form::Pre) {
        for (int i = tid; i < 4 * Tokens * (Hidden / 8); i += Threads) {
            const int s = i / (Tokens * (Hidden / 8)), t = i / (Hidden / 8) % Tokens;
            const int col = i % (Hidden / 8) * 8;
            const auto m  = m0 + t;
            common::cp_async_zfill<16, Cache::ca>(
                x[s] + t * Hidden + projection_shared_column(t, col),
                static_cast<const __nv_bfloat16*>(o.residual) + (m < o.columns ? m : 0) * kWidth +
                    s * kHidden + d0 + col,
                m < o.columns ? 16 : 0);
        }
        common::cp_commit();
    } else {
        const int t = tid * Vector / Hidden, col = tid * Vector % Hidden;
        common::Bf16Vector<Vector> values[4] = {};
        if (m0 + t < o.columns) {
            if constexpr (F == Form::PostPre) {
                mixed_vectors(o, m0 + t, d0 + col, coefficients[t], coefficients[t] + 4, values);
            } else {
                const auto value = common::Bf16Vector<Vector>::load(
                    static_cast<const __nv_bfloat16*>(o.residual) + (m0 + t) * kHidden + d0 + col);
#pragma unroll
                for (int s = 0; s < 4; ++s) values[s] = value;
            }
#pragma unroll
            for (int s = 0; s < 4; ++s)
                values[s].store(static_cast<__nv_bfloat16*>(o.updated) + (m0 + t) * kWidth +
                                s * kHidden + d0 + col);
        }
#pragma unroll
        for (int s = 0; s < 4; ++s)
            values[s].store(x[s] + t * Hidden + projection_shared_column(t, col));
    }
    common::cp_wait<0>();
    __syncthreads();
    if (warp < 2) {
        float accum[4] = {};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
#pragma unroll
            for (int k = 0; k < Hidden / 16; ++k) {
                const int row = warp * 16 + (lane & 7) + (((lane >> 3) & 1) * 8);
                const int col = k * 16 + (lane >> 4) * 8;
                const int t = lane & 7, xc = k * 16 + (((lane >> 3) & 1) * 8), xt = row - 24;
                const auto* ap = row < 24 ? a[s] + row * Hidden + projection_shared_column(row, col)
                                          : x[s] + xt * Hidden + projection_shared_column(xt, col);
                unsigned af[4], bf[2];
                common::ldmatrix_x4(af[0], af[1], af[2], af[3], common::smem_addr(ap));
                common::ldmatrix_x2(
                    bf[0], bf[1],
                    common::smem_addr(x[s] + t * Hidden + projection_shared_column(t, xc)));
                common::mma_bf16(accum[0], accum[1], accum[2], accum[3], af[0], af[1], af[2], af[3],
                                 bf[0], bf[1]);
            }
        }
        const int row = warp * 16 + (lane >> 2), token = 2 * (lane & 3);
        const auto store = [&](int r, int t, float value) {
            if (m0 + t < o.columns && (r < 24 || r == 24 + t))
                partial[((m0 + t) * Parts + part) * 32 + (r < 24 ? r : 24)] = value;
        };
        store(row, token, accum[0]);
        store(row, token + 1, accum[1]);
        store(row + 8, token, accum[2]);
        store(row + 8, token + 1, accum[3]);
    }
}
} // namespace ninfer::ops::detail::mhc
