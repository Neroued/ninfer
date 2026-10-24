#pragma once

#include "ops/softmax_attention/kpool_mla/sparse_mla/arguments.h"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"

#include <math_constants.h>

namespace ninfer::ops::detail::kpool {

template <int Width>
__device__ __forceinline__ int row_factor_offset(int row, int column) {
    return (row * Width + column) ^ ((row & 7) << 3);
}

// Native U_K is reduction-major; U_V is output-major. Transpose U_K fragments on
// chip. Both maps reuse a weight tile across queries, including different requests.
template <bool Key, int Queries, int Rows = 64, int K = 32, bool Direct = false, bool Merge = false>
__global__ __launch_bounds__(Rows > 64 ? 256 : 128) void mla_factor_kernel(
    const __grid_constant__ SparseMlaArguments a) {
    if constexpr (Key)
        if (a.direct) return;
    constexpr int Outputs = Key ? 512 : 256, Reduction = Key ? 256 : 512;
    constexpr int Threads  = Rows > 64 ? 256 : 128;
    constexpr int WarpRows = Rows / 16, WarpColumns = Threads / 32 / WarpRows,
                  PerWarp = Queries / WarpColumns;
    static_assert(Rows == 32 || Rows == 64 || Rows == 128);
    static_assert(PerWarp % 8 == 0);
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int warp_row = warp % WarpRows, warp_column = warp / WarpRows * PerWarp;
    const int head = blockIdx.y, m0 = blockIdx.x % (Outputs / Rows) * Rows;
    const int q0 = blockIdx.x / (Outputs / Rows) * Queries;
    __shared__ int live[Queries];
    __shared__ const __nv_bfloat16* direct_inputs[Direct ? Queries : 1];
    // A singleton softmax is exactly one; its query map cannot affect the result.
    const bool active = tid < Queries && a.work(q0 + tid).size > (Key ? 1 : 0);
    if (tid < Queries) {
        live[tid] = active;
        if constexpr (Direct)
            if (active) {
                const auto* table =
                    a.tables +
                    std::int64_t(a.table_rows[(a.begin + q0 + tid) / a.tokens]) * a.table_stride;
                direct_inputs[tid] = reinterpret_cast<const __nv_bfloat16*>(
                    reinterpret_cast<const char*>(a.latent) + table[0]);
            }
    }
    if (!__syncthreads_or(active)) {
        if constexpr (!Key)
            for (int i = tid; i < Queries * Rows; i += Threads) {
                const int q = q0 + i / Rows;
                if (q < a.width)
                    a.out[((a.begin + q) * 64 + head) * 256 + m0 + i % Rows] =
                        __float2bfloat16_rn(0.f);
            }
        return;
    }
    // A fused value map normalizes split partials directly into its MMA input tile.
    // Runtime singleton splits still consume the attention stage's mapped row.
    __shared__ float merge_weights[Merge ? Queries * 32 : 1];
    __shared__ int merge_splits[Merge ? Queries : 1];
    if constexpr (Merge) {
        for (int q = warp; q < Queries; q += Threads / 32) {
            const auto work  = a.work(q0 + q);
            const int splits = min(a.splits, (work.size + a.tile - 1) / a.tile);
            if (lane == 0) merge_splits[q] = splits;
            if (splits <= 1) continue;
            float maximum =
                lane < splits ? a.maximum[a.partial(q0 + q, lane, head)] : -CUDART_INF_F;
            maximum = common::warp_max(maximum);
            const float weight =
                lane < splits
                    ? common::exp2_approx_ftz(a.maximum[a.partial(q0 + q, lane, head)] - maximum)
                    : 0;
            const float denominator = common::warp_sum(
                lane < splits ? weight * a.denominator[a.partial(q0 + q, lane, head)] : 0);
            merge_weights[q * 32 + lane] = weight * __frcp_rn(denominator);
        }
        __syncthreads();
    }
    extern __shared__ __align__(16) __nv_bfloat16 storage[];
    constexpr int Stage = (Rows + Queries) * K;
    const auto stage    = [&](int k0, int buffer) {
        auto* ws = storage + buffer * Stage;
        auto* xs = ws + Rows * K;
        for (int i = tid; i < Rows * K / 8; i += Threads) {
            if constexpr (Key) {
                const int k = i / (Rows / 8), m = i % (Rows / 8) * 8;
                common::cp_async<16, Cache::cg>(ws + row_factor_offset<Rows>(k, m),
                                                   a.key_up + head * (256 * 512) + (k0 + k) * 512 +
                                                       m0 + m);
            } else {
                const int m = i / (K / 8), k = i % (K / 8) * 8;
                common::cp_async<16, Cache::cg>(ws + row_factor_offset<K>(m, k),
                                                   a.value_up + head * (256 * 512) + (m0 + m) * 512 +
                                                       k0 + k);
            }
        }
        for (int i = tid; i < Queries * K / 8; i += Threads) {
            const int q = i / (K / 8), k = i % (K / 8) * 8;
            auto* dst = xs + row_factor_offset<K>(q, k);
            if (live[q]) {
                if constexpr (Merge) {
                    if (merge_splits[q] > 1) {
                        float values[8]{};
                        for (int split = 0; split < merge_splits[q]; ++split) {
                            const float weight = merge_weights[q * 32 + split];
                            const auto* src =
                                a.numerator + a.partial(q0 + q, split, head) * 512 + k0 + k;
                            const float4 first = *reinterpret_cast<const float4*>(src);
                            const float4 last  = *reinterpret_cast<const float4*>(src + 4);
                            values[0] += weight * first.x;
                            values[1] += weight * first.y;
                            values[2] += weight * first.z;
                            values[3] += weight * first.w;
                            values[4] += weight * last.x;
                            values[5] += weight * last.y;
                            values[6] += weight * last.z;
                            values[7] += weight * last.w;
                        }
                        common::store_vec(dst,
                                             make_uint4(common::pack_bf16x2(values[0], values[1]),
                                                        common::pack_bf16x2(values[2], values[3]),
                                                        common::pack_bf16x2(values[4], values[5]),
                                                        common::pack_bf16x2(values[6], values[7])));
                        continue;
                    }
                }
                const auto* src = Key      ? a.query + ((a.begin + q0 + q) * 64 + head) * 256
                                     : Direct ? direct_inputs[q]
                                              : a.mapped + ((q0 + q) * 64 + head) * 512;
                common::cp_async<16>(dst, src + k0 + k);
            } else
                common::store_vec(dst, make_int4(0, 0, 0, 0));
        }
        common::cp_commit();
    };
    float acc[PerWarp / 8][4]{};
    stage(0, 0);
    for (int k0 = 0, iteration = 0; k0 < Reduction; k0 += K, ++iteration) {
        const int buffer = iteration % 2;
        common::cp_wait<0>();
        __syncthreads();
        if (k0 + K < Reduction) stage(k0 + K, 1 - buffer);
        const auto* ws = storage + buffer * Stage;
        const auto* xs = ws + Rows * K;
#pragma unroll
        for (int k = 0; k < K / 16; ++k) {
            unsigned af[4];
            if constexpr (Key) {
                const int r = k * 16 + (lane & 7) + (lane >> 4) * 8;
                const int c = warp_row * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x4_t(af[0], af[1], af[2], af[3],
                                      common::smem_addr(ws + row_factor_offset<Rows>(r, c)));
            } else {
                const int r = warp_row * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int c = k * 16 + (lane >> 4) * 8;
                common::ldmatrix_x4(af[0], af[1], af[2], af[3],
                                    common::smem_addr(ws + row_factor_offset<K>(r, c)));
            }
#pragma unroll
            for (int n = 0; n < PerWarp / 8; ++n) {
                unsigned bf[2];
                const int r = warp_column + n * 8 + (lane & 7), c = k * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x2(bf[0], bf[1],
                                    common::smem_addr(xs + row_factor_offset<K>(r, c)));
                common::mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], af[0], af[1], af[2],
                                 af[3], bf[0], bf[1]);
            }
        }
    }
#pragma unroll
    for (int n = 0; n < PerWarp / 8; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = q0 + warp_column + n * 8 + (lane % 4) * 2 + j % 2;
            const int m = m0 + warp_row * 16 + lane / 4 + j / 2 * 8;
            if (q >= a.width) continue;
            if constexpr (Key) {
                if (live[q - q0])
                    a.mapped[(q * 64 + head) * 512 + m] = __float2bfloat16_rn(acc[n][j]);
            } else
                a.out[((a.begin + q) * 64 + head) * 256 + m] =
                    __float2bfloat16_rn(live[q - q0] ? acc[n][j] : 0.f);
        }
}
} // namespace ninfer::ops::detail::kpool
