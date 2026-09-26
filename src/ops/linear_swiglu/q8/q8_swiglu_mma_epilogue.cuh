#pragma once
#include "ops/common/math.cuh"
#include "ops/linear/q8/q8_a16_mma.cuh"

namespace ninfer::ops::detail {
template <class Schedule>
struct Q8SwiGluMmaRows {
    static constexpr int kOutputRowsPerCta = Schedule::kBlockRows / 2;

    __device__ __forceinline__ int weight_row(int row0, int local_row, int rows) const {
        return row0 + local_row % kOutputRowsPerCta +
               (local_row >= kOutputRowsPerCta ? rows / 2 : 0);
    }
};

struct Q8SwiGluMmaEpilogue {
    template <class Schedule>
    static constexpr int kSharedBytes = Schedule::kBlockRows * Schedule::kBlockTokens * 2;

    template <class Cfg, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(Output output_tile, unsigned char* scratch,
                                                float (&acc)[Cfg::kMmaRows][Cfg::kMmaTokens][4],
                                                int m0, int n0, int m, int n) const {
        constexpr int BM = Cfg::kBlockRows, BN = Cfg::kBlockTokens;
        constexpr int WN = Cfg::kWarpTokens;
        constexpr int MT = Cfg::kMmaRows, NT = Cfg::kMmaTokens;
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int wm = warp / Cfg::kWarpGridTokens, wn = warp % Cfg::kWarpGridTokens;
        const int gid = lane >> 2, lid = lane & 3;
        if constexpr (Cfg::kWarpGridRows == 1) {
            static_assert((MT % 2) == 0);
            constexpr int kGateMt = MT / 2;
#pragma unroll
            for (int mi = 0; mi < kGateMt; ++mi) {
                const int r0 = m0 + mi * 16 + gid;
                const int r1 = r0 + 8;
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int c0          = n0 + wn * WN + ni * 8 + 2 * lid;
                    const int c1          = c0 + 1;
                    const float* gate_acc = acc[mi][ni];
                    const float* up_acc   = acc[mi + kGateMt][ni];
                    if constexpr (Full) {
                        output_tile.store(r0, c0, silu(gate_acc[0]) * up_acc[0]);
                        output_tile.store(r0, c1, silu(gate_acc[1]) * up_acc[1]);
                        output_tile.store(r1, c0, silu(gate_acc[2]) * up_acc[2]);
                        output_tile.store(r1, c1, silu(gate_acc[3]) * up_acc[3]);
                    } else {
                        if (r0 < m / 2 && c0 < n) {
                            output_tile.store(r0, c0, silu(gate_acc[0]) * up_acc[0]);
                        }
                        if (r0 < m / 2 && c1 < n) {
                            output_tile.store(r0, c1, silu(gate_acc[1]) * up_acc[1]);
                        }
                        if (r1 < m / 2 && c0 < n) {
                            output_tile.store(r1, c0, silu(gate_acc[2]) * up_acc[2]);
                        }
                        if (r1 < m / 2 && c1 < n) {
                            output_tile.store(r1, c1, silu(gate_acc[3]) * up_acc[3]);
                        }
                    }
                }
            }
        } else {
            static_assert(Cfg::kWarpGridRows == 2);
            auto* up_shared = reinterpret_cast<float*>(scratch);
            __syncthreads();
            if (wm == 1) {
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    const int local_r0 = mi * 16 + gid;
                    const int local_r1 = local_r0 + 8;
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        const int local_c0                  = wn * WN + ni * 8 + 2 * lid;
                        const int local_c1                  = local_c0 + 1;
                        const float* up_acc                 = acc[mi][ni];
                        up_shared[local_r0 * BN + local_c0] = up_acc[0];
                        up_shared[local_r0 * BN + local_c1] = up_acc[1];
                        up_shared[local_r1 * BN + local_c0] = up_acc[2];
                        up_shared[local_r1 * BN + local_c1] = up_acc[3];
                    }
                }
            }
            __syncthreads();
            if (wm == 0) {
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    const int local_r0 = mi * 16 + gid;
                    const int local_r1 = local_r0 + 8;
                    const int r0       = m0 + local_r0;
                    const int r1       = m0 + local_r1;
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        const int local_c0    = wn * WN + ni * 8 + 2 * lid;
                        const int local_c1    = local_c0 + 1;
                        const int c0          = n0 + local_c0;
                        const int c1          = n0 + local_c1;
                        const float* gate_acc = acc[mi][ni];
                        const float up00      = up_shared[local_r0 * BN + local_c0];
                        const float up01      = up_shared[local_r0 * BN + local_c1];
                        const float up10      = up_shared[local_r1 * BN + local_c0];
                        const float up11      = up_shared[local_r1 * BN + local_c1];
                        if constexpr (Full) {
                            output_tile.store(r0, c0, silu(gate_acc[0]) * up00);
                            output_tile.store(r0, c1, silu(gate_acc[1]) * up01);
                            output_tile.store(r1, c0, silu(gate_acc[2]) * up10);
                            output_tile.store(r1, c1, silu(gate_acc[3]) * up11);
                        } else {
                            if (r0 < m / 2 && c0 < n) {
                                output_tile.store(r0, c0, silu(gate_acc[0]) * up00);
                            }
                            if (r0 < m / 2 && c1 < n) {
                                output_tile.store(r0, c1, silu(gate_acc[1]) * up01);
                            }
                            if (r1 < m / 2 && c0 < n) {
                                output_tile.store(r1, c0, silu(gate_acc[2]) * up10);
                            }
                            if (r1 < m / 2 && c1 < n) {
                                output_tile.store(r1, c1, silu(gate_acc[3]) * up11);
                            }
                        }
                    }
                }
            }
        }
    }
};
} // namespace ninfer::ops::detail
