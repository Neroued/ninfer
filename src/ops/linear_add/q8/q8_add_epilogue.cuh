#pragma once
#include "ops/common/math.cuh"
#include "ops/linear/q8/q8_a16_mma.cuh"

namespace ninfer::ops::detail {
// The projection stays FP32 through the vectorized residual addition.
struct Q8AddMmaEpilogue {
    template <class Schedule>
    static constexpr int kSharedBytes =
        Schedule::kBlockRows * Schedule::kBlockTokens * sizeof(float);

    template <class Cfg, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(Output output_tile, unsigned char* scratch,
                                                float (&acc)[Cfg::kMmaRows][Cfg::kMmaTokens][4],
                                                int m0, int n0, int m, int n) const {
        constexpr int BM = Cfg::kBlockRows, BN = Cfg::kBlockTokens;
        constexpr int WM = Cfg::kWarpRows, WN = Cfg::kWarpTokens;
        constexpr int MT = Cfg::kMmaRows, NT = Cfg::kMmaTokens;
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int wm = warp / Cfg::kWarpGridTokens, wn = warp % Cfg::kWarpGridTokens;
        const int gid = lane >> 2, lid = lane & 3;
        static_assert((BM % 8) == 0);
        // Reuse the operand storage after all MMA reads complete. Preserve the FP32
        // projection until adding the residual; BF16 storage rounds the complete result.
        __syncthreads();
        float* projected_shared = reinterpret_cast<float*>(scratch);
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int local_r0 = wm * WM + mi * 16 + gid;
            const int local_r1 = local_r0 + 8;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int local_c0                         = wn * WN + ni * 8 + 2 * lid;
                const int local_c1                         = local_c0 + 1;
                const float* a                             = acc[mi][ni];
                projected_shared[local_c0 * BM + local_r0] = a[0];
                projected_shared[local_c1 * BM + local_r0] = a[1];
                projected_shared[local_c0 * BM + local_r1] = a[2];
                projected_shared[local_c1 * BM + local_r1] = a[3];
            }
        }
        __syncthreads();

        constexpr int kRowsPerPack = 8;
        constexpr int kPacksPerCol = BM / kRowsPerPack;
        constexpr int kPacks       = BN * kPacksPerCol;
        for (int pack = tid; pack < kPacks; pack += Cfg::kThreads) {
            const int local_col = pack / kPacksPerCol;
            const int row_pack  = pack - local_col * kPacksPerCol;
            const int local_row = row_pack * kRowsPerPack;
            const int col       = n0 + local_col;
            const int row       = m0 + local_row;
            if (Full || (col < n && row < m)) {
                if (Full || row + kRowsPerPack <= m) {
                    const float* source = projected_shared + local_col * BM + local_row;
                    const float4 low    = load_vec<float4>(source);
                    const float4 high   = load_vec<float4>(source + 4);
                    const float projected[]{low.x,  low.y,  low.z,  low.w,
                                            high.x, high.y, high.z, high.w};
                    Q8Bf16x8Bits residual;
                    residual.raw = load_vec<uint4>(output_tile.at(row, col));
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        residual.pair[pair] = __floats2bfloat162_rn(
                            __low2float(residual.pair[pair]) + projected[pair * 2],
                            __high2float(residual.pair[pair]) + projected[pair * 2 + 1]);
                    }
                    store_vec(output_tile.at(row, col), residual.raw);
                } else {
#pragma unroll
                    for (int i = 0; i < kRowsPerPack; ++i) {
                        if (row + i < m) {
                            __nv_bfloat16* destination = output_tile.at(row + i, col);
                            *destination               = __float2bfloat16_rn(
                                __bfloat162float(*destination) +
                                projected_shared[local_col * BM + local_row + i]);
                        }
                    }
                }
            }
        }
    }
};
} // namespace ninfer::ops::detail
