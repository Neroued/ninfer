#pragma once
#include "ops/linear/swiglu/paired_output.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_instances.cuh"
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
template <class Product>
struct BasicSwiGluRowMajorMmaEpilogue {
    Product product{};
    template <class Schedule>
    static constexpr int kSharedBytes = Schedule::kBlockRows * Schedule::kBlockTokens * 2;

    template <class Cfg, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(Output output_tile, unsigned char* scratch,
                                                float (&acc)[Cfg::kMmaRows][Cfg::kMmaTokens][4],
                                                int m0, int n0, int m, int n) const {
        constexpr int BN = Cfg::kBlockTokens;
        constexpr int WN = Cfg::kWarpTokens;
        constexpr int MT = Cfg::kMmaRows, NT = Cfg::kMmaTokens;
        constexpr int producers = [] {
            if constexpr (requires { Cfg::kProducerThreads; })
                return Cfg::kProducerThreads;
            else
                return 0;
        }();
        const auto synchronize = [] {
            if constexpr (producers)
                asm volatile("bar.sync 1, %0;" ::"r"(Cfg::kThreads - producers) : "memory");
            else
                __syncthreads();
        };
        const int tid = int(threadIdx.x) - producers, warp = tid >> 5, lane = tid & 31;
        const int wm  = warp / (Cfg::kBlockTokens / Cfg::kWarpTokens),
                  wn  = warp % (Cfg::kBlockTokens / Cfg::kWarpTokens);
        const int gid = lane >> 2, lid = lane & 3;
        if constexpr ((Cfg::kBlockRows / Cfg::kWarpRows) == 1) {
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
                        output_tile.store(r0, c0, product(gate_acc[0], up_acc[0]));
                        output_tile.store(r0, c1, product(gate_acc[1], up_acc[1]));
                        output_tile.store(r1, c0, product(gate_acc[2], up_acc[2]));
                        output_tile.store(r1, c1, product(gate_acc[3], up_acc[3]));
                    } else {
                        if (r0 < m / 2 && c0 < n) {
                            output_tile.store(r0, c0, product(gate_acc[0], up_acc[0]));
                        }
                        if (r0 < m / 2 && c1 < n) {
                            output_tile.store(r0, c1, product(gate_acc[1], up_acc[1]));
                        }
                        if (r1 < m / 2 && c0 < n) {
                            output_tile.store(r1, c0, product(gate_acc[2], up_acc[2]));
                        }
                        if (r1 < m / 2 && c1 < n) {
                            output_tile.store(r1, c1, product(gate_acc[3], up_acc[3]));
                        }
                    }
                }
            }
        } else {
            static_assert((Cfg::kBlockRows / Cfg::kWarpRows) == 2);
            auto* up_shared = reinterpret_cast<float*>(scratch);
            synchronize();
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
            synchronize();
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
                            output_tile.store(r0, c0, product(gate_acc[0], up00));
                            output_tile.store(r0, c1, product(gate_acc[1], up01));
                            output_tile.store(r1, c0, product(gate_acc[2], up10));
                            output_tile.store(r1, c1, product(gate_acc[3], up11));
                        } else {
                            if (r0 < m / 2 && c0 < n) {
                                output_tile.store(r0, c0, product(gate_acc[0], up00));
                            }
                            if (r0 < m / 2 && c1 < n) {
                                output_tile.store(r0, c1, product(gate_acc[1], up01));
                            }
                            if (r1 < m / 2 && c0 < n) {
                                output_tile.store(r1, c0, product(gate_acc[2], up10));
                            }
                            if (r1 < m / 2 && c1 < n) {
                                output_tile.store(r1, c1, product(gate_acc[3], up11));
                            }
                        }
                    }
                }
            }
        }
    }
};

// Pair an MMA atom's first eight rows with its second eight rows.
// Persistent gate and up planes stay unchanged.
template <class Schedule, int Intermediate>
struct SwiGluBf16SlicedSchedule : Schedule {
    __device__ static int weight_row(int row) {
        return (row / 16) * 8 + row % 8 + (row % 16 >= 8 ? Intermediate : 0);
    }
};

template <class Product>
struct SwiGluBf16FragmentEpilogue {
    Product product;

    template <class Output>
    __device__ void store_fragment(Output out, int row, int token, float4 v, int,
                                   int tokens) const {
        const int r = (row / 16) * 8 + row % 8;
        if (token < tokens) out.store(r, token, product(v.x, v.z));
        if (token + 1 < tokens) out.store(r, token + 1, product(v.y, v.w));
    }
};

// TMA fills one contiguous gate panel and one contiguous up panel per CTA.
template <int Intermediate>
struct SwiGluBf16TmaWeights {
    template <class Schedule>
    static CUtensorMap map(const Bf16A16Operands& p) {
        return bf16_tma_map<Schedule>(p.weight, p.rows, p.k, Schedule::kBlockRows / 2,
                                      Schedule::kBlockK);
    }

    template <class Schedule>
    __device__ static void load(void* dst, const CUtensorMap* map, int sector, int row,
                                std::uint64_t* barrier) {
        bf16_tma_load<Schedule>(dst, map, sector, row / 2, barrier);
        bf16_tma_load<Schedule>(static_cast<__nv_bfloat16*>(dst) +
                                    Schedule::kBlockRows / 2 * Schedule::kBlockK,
                                map, sector, row / 2 + Intermediate, barrier);
    }
};
} // namespace ninfer::ops::detail
