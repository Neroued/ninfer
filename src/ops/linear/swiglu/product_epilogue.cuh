#pragma once
#include "ops/linear/swiglu/paired_output.cuh"

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
                        output_tile.store(r0, c0,
                                          swiglu_product(product, r0, gate_acc[0], up_acc[0]));
                        output_tile.store(r0, c1,
                                          swiglu_product(product, r0, gate_acc[1], up_acc[1]));
                        output_tile.store(r1, c0,
                                          swiglu_product(product, r1, gate_acc[2], up_acc[2]));
                        output_tile.store(r1, c1,
                                          swiglu_product(product, r1, gate_acc[3], up_acc[3]));
                    } else {
                        if (r0 < m / 2 && c0 < n) {
                            output_tile.store(r0, c0,
                                              swiglu_product(product, r0, gate_acc[0], up_acc[0]));
                        }
                        if (r0 < m / 2 && c1 < n) {
                            output_tile.store(r0, c1,
                                              swiglu_product(product, r0, gate_acc[1], up_acc[1]));
                        }
                        if (r1 < m / 2 && c0 < n) {
                            output_tile.store(r1, c0,
                                              swiglu_product(product, r1, gate_acc[2], up_acc[2]));
                        }
                        if (r1 < m / 2 && c1 < n) {
                            output_tile.store(r1, c1,
                                              swiglu_product(product, r1, gate_acc[3], up_acc[3]));
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
                            output_tile.store(r0, c0,
                                              swiglu_product(product, r0, gate_acc[0], up00));
                            output_tile.store(r0, c1,
                                              swiglu_product(product, r0, gate_acc[1], up01));
                            output_tile.store(r1, c0,
                                              swiglu_product(product, r1, gate_acc[2], up10));
                            output_tile.store(r1, c1,
                                              swiglu_product(product, r1, gate_acc[3], up11));
                        } else {
                            if (r0 < m / 2 && c0 < n) {
                                output_tile.store(r0, c0,
                                                  swiglu_product(product, r0, gate_acc[0], up00));
                            }
                            if (r0 < m / 2 && c1 < n) {
                                output_tile.store(r0, c1,
                                                  swiglu_product(product, r0, gate_acc[1], up01));
                            }
                            if (r1 < m / 2 && c0 < n) {
                                output_tile.store(r1, c0,
                                                  swiglu_product(product, r1, gate_acc[2], up10));
                            }
                            if (r1 < m / 2 && c1 < n) {
                                output_tile.store(r1, c1,
                                                  swiglu_product(product, r1, gate_acc[3], up11));
                            }
                        }
                    }
                }
            }
        }
    }
};

template <class Product>
struct BasicSwiGluPairEpilogue {
    Product product{};

    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, swiglu_product(product, row, gate, up));
    }
};

template <class Product>
struct BasicSwiGluTokenMajorMmaEpilogue {
    Product product{};

    __device__ __forceinline__ float apply_pair(int row, int, float gate, float up) const {
        return swiglu_product(product, row, gate, up);
    }

    template <class Schedule>
    static constexpr int kSharedBytes = Schedule::kBlockTokens * (Schedule::kBlockRows / 2 + 8) * 2;

    template <class S, bool Full, class Output>
    __device__ __forceinline__ void
    finish_tile(Output output, unsigned char* scratch, float (&acc)[S::kMmaTokens][S::kMmaRows][4],
                int row_begin, int token_begin, int, int token_end) const {
        static_assert(S::kMmaRows % 2 == 0);
        if constexpr (requires { typename Output::Fp32; }) {
            constexpr int producers = [] {
                if constexpr (requires { S::kProducerThreads; })
                    return S::kProducerThreads;
                else
                    return 0;
            }();
            const int tid = int(threadIdx.x) - producers, warp = tid / 32, lane = tid & 31;
            const int wt = warp / S::kWarpsRows, wr = warp % S::kWarpsRows;
            constexpr int gate_fragments = S::kMmaRows / 2;
#pragma unroll
            for (int mt = 0; mt < S::kMmaTokens; ++mt) {
                const int token = token_begin + wt * S::kWarpTokens + mt * 16 + (lane >> 2);
#pragma unroll
                for (int mr = 0; mr < gate_fragments; ++mr) {
                    const int row = row_begin + wr * (S::kWarpRows / 2) + mr * 8 + 2 * (lane & 3);
                    const auto& gate = acc[mt][mr];
                    const auto& up   = acc[mt][mr + gate_fragments];
                    if (Full || token < token_end) {
                        output.store(row, token, swiglu_product(product, row, gate[0], up[0]));
                        output.store(row + 1, token,
                                     swiglu_product(product, row + 1, gate[1], up[1]));
                    }
                    if (Full || token + 8 < token_end) {
                        output.store(row, token + 8, swiglu_product(product, row, gate[2], up[2]));
                        output.store(row + 1, token + 8,
                                     swiglu_product(product, row + 1, gate[3], up[3]));
                    }
                }
            }
        } else {
            constexpr int rows           = S::kBlockRows / 2;
            constexpr int stride         = rows + 8;
            constexpr int gate_fragments = S::kMmaRows / 2;
            constexpr int producers      = [] {
                if constexpr (requires { S::kProducerThreads; })
                    return S::kProducerThreads;
                else
                    return 0;
            }();
            constexpr int threads = S::kThreads - producers;
            const int tid = static_cast<int>(threadIdx.x) - producers, warp = tid >> 5,
                      lane         = tid & 31;
            const auto synchronize = [] {
                if constexpr (producers)
                    asm volatile("bar.sync 1, %0;" ::"r"(threads) : "memory");
                else
                    __syncthreads();
            };
            const int wt = warp / S::kWarpsRows, wr = warp % S::kWarpsRows;
            auto* final_tile = reinterpret_cast<__nv_bfloat16*>(scratch);
            synchronize();
#pragma unroll
            for (int mt = 0; mt < S::kMmaTokens; ++mt) {
                const int token = wt * S::kWarpTokens + mt * 16 + (lane >> 2);
#pragma unroll
                for (int mr = 0; mr < gate_fragments; ++mr) {
                    const int row    = wr * (S::kWarpRows / 2) + mr * 8 + 2 * (lane & 3);
                    const auto& gate = acc[mt][mr];
                    const auto& up   = acc[mt][mr + gate_fragments];
                    *reinterpret_cast<__nv_bfloat162*>(final_tile + token * stride + row) =
                        __floats2bfloat162_rn(
                            swiglu_product(product, row_begin + row, gate[0], up[0]),
                            swiglu_product(product, row_begin + row + 1, gate[1], up[1]));
                    *reinterpret_cast<__nv_bfloat162*>(final_tile + (token + 8) * stride + row) =
                        __floats2bfloat162_rn(
                            swiglu_product(product, row_begin + row, gate[2], up[2]),
                            swiglu_product(product, row_begin + row + 1, gate[3], up[3]));
                }
            }
            synchronize();
            for (int item = tid; item < S::kBlockTokens * (rows / 8); item += threads) {
                const int token = item / (rows / 8), row = (item % (rows / 8)) * 8;
                if (Full || token_begin + token < token_end)
                    linear_store_bf16_vector(
                        output, row_begin + row, token_begin + token,
                        common::load_vec<uint4>(final_tile + token * stride + row));
            }
        }
    }
};
} // namespace ninfer::ops::detail
