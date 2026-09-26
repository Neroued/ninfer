#pragma once

// Q8G32 RowSplit x BF16 Tensor Core GEMM.
//
// out[M,N] = W[M,K] * x[K,N], where W stores one signed int8 code per element
// and one FP16 scale per 32 K elements. Raw codes and eight quantization groups' scales are
// staged with cp.async before dequantization into a swizzled BF16
// shared tile; x uses a two-stage cp.async pipeline. Tensor Cores execute
// m16n8k16 BF16 MMA with FP32 accumulation.

#include "ops/common/mma.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_operands.h"
#include "ops/linear/q8/q8_shared.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

union alignas(16) Q8Bf16x8Bits {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

static_assert(sizeof(Q8Bf16x8Bits) == 16);

__device__ __forceinline__ int q8_g32_swz64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <class Schedule>
struct Q8MmaIdentityRows {
    static constexpr int kOutputRowsPerCta = Schedule::kBlockRows;

    __device__ __forceinline__ int weight_row(int row0, int local_row, int) const {
        return row0 + local_row;
    }
};
template <class Schedule, class Epilogue>
inline constexpr int q8_mma_shared_bytes = [] {
    constexpr int extra = [] {
        if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
            return static_cast<int>(Epilogue::template kSharedBytes<Schedule>);
        else
            return 0;
    }();
    return Schedule::kSharedBytes > extra ? Schedule::kSharedBytes : extra;
}();

template <class Cfg, bool Full, bool FullK, class Output, class Epilogue, class RowPolicy>
__global__ __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocksPerSm) void q8_a16_mma_kernel(
    Q8LinearOperands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_begin, int token_count) {
    const auto* __restrict__ x      = operands.x;
    const auto* __restrict__ codes  = operands.codes;
    const auto* __restrict__ scales = operands.scales;
    const int m = operands.rows, k = operands.k, n = token_begin + token_count;
    const int padded_k              = operands.padded_k;
    constexpr int BM                = Cfg::kBlockRows;
    constexpr int BN                = Cfg::kBlockTokens;
    constexpr int BK                = Cfg::kBlockK;
    constexpr int WM                = Cfg::kWarpRows;
    constexpr int WN                = Cfg::kWarpTokens;
    constexpr int MT                = Cfg::kMmaRows;
    constexpr int NT                = Cfg::kMmaTokens;
    constexpr int KSUB              = Cfg::kMmaKSteps;
    constexpr int kOutputRowsPerCta = RowPolicy::kOutputRowsPerCta;

    struct OperandStorage {
        alignas(16) __nv_bfloat16 weights[BM * BK];
        alignas(16) __nv_bfloat16 activations[Cfg::kActivationStages][BN * BK];
        alignas(16) std::uint8_t codes[BM * BK];
        alignas(16) std::uint8_t scales[BM * Cfg::kScaleCacheBytes];
    };

    static_assert(sizeof(OperandStorage) == Cfg::kSharedBytes);
    auto* scratch = q8_shared_storage<q8_mma_shared_bytes<Cfg, Epilogue>>();
    auto& shared  = *reinterpret_cast<OperandStorage*>(scratch);
    auto& As      = shared.weights;
    auto& Bs      = shared.activations;
    auto& Cr      = shared.codes;
    auto& Sr      = shared.scales;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Cfg::kWarpGridTokens;
    const int wn   = warp % Cfg::kWarpGridTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int m0 = static_cast<int>(blockIdx.x) * kOutputRowsPerCta;
    const int n0 = token_begin + static_cast<int>(blockIdx.y) * BN;
    const int kg = padded_k / 32;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            acc[mi][ni][0] = 0.0f;
            acc[mi][ni][1] = 0.0f;
            acc[mi][ni][2] = 0.0f;
            acc[mi][ni][3] = 0.0f;
        }
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    auto stage_x = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Cfg::kThreads) {
            const int nl = item / (BK / 8);
            const int k8 = item - nl * (BK / 8);
            const int kk = k0 + k8 * 8;
            const int nn = n0 + nl;
            auto* dst    = &Bs[stage][nl * BK + q8_g32_swz64(nl, k8 * 8)];
            if constexpr (Full) {
                cp_async<16, Cfg::kActivationCache>(dst,
                                                    &x[static_cast<std::int64_t>(nn) * k + kk]);
            } else {
                const int valid =
                    nn < n && (FullK || kk < k) ? (FullK ? 16 : min(8, k - kk) * 2) : 0;
                ninfer::ops::cp_async_zfill<16, Cfg::kPredicatedCache>(
                    dst,
                    &x[static_cast<std::int64_t>(nn < n ? nn : 0) * k + (FullK || kk < k ? kk : 0)],
                    valid);
            }
        }
    };

    auto stage_w = [&](int kt) {
        constexpr int GROUPS            = BK / 32;
        constexpr int SCALE_CACHE_TILES = 8 / GROUPS;
        const int g0                    = kt * GROUPS;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 16); item += Cfg::kThreads) {
            const int row   = item / (BK / 16);
            const int chunk = item - row * (BK / 16);
            const int grow  = row_policy.weight_row(m0, row, m);
            auto* dst       = &Cr[row * BK + chunk * 16];
            if constexpr (Full) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                cp_async<16, Cfg::kWeightCache>(dst, &codes[gi * 32 + chunk * 16]);
            } else {
                const bool valid_row  = (grow < m);
                const std::int64_t gi = static_cast<std::int64_t>(valid_row ? grow : 0) * kg + g0;
                ninfer::ops::cp_async_zfill<16, Cfg::kPredicatedCache>(
                    dst, &codes[gi * 32 + chunk * 16], valid_row ? 16 : 0);
            }
        }
        if ((kt % SCALE_CACHE_TILES) == 0) {
            for (int row = tid; row < BM; row += Cfg::kThreads) {
                const int grow = row_policy.weight_row(m0, row, m);
                auto* dst      = &Sr[row * Cfg::kScaleCacheBytes];
                if constexpr (Full) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                    cp_async<16, Cfg::kWeightCache>(dst, &scales[gi * 2]);
                } else if (kg % 8 == 0) {
                    const bool valid_row   = grow < m;
                    const int valid_scales = valid_row && g0 < kg ? min(8, kg - g0) : 0;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(valid_row ? grow : 0) * kg + g0;
                    cp_async_zfill<16, Cfg::kPredicatedCache>(dst, &scales[gi * 2],
                                                              valid_scales * 2);
                } else {
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        if (grow < m && g0 + pair * 2 < kg)
                            cp_async<4>(dst + pair * 4,
                                        scales + (static_cast<std::int64_t>(grow) * kg + g0) * 2 +
                                            pair * 4);
                        else
                            *reinterpret_cast<unsigned*>(dst + pair * 4) = 0;
                    }
                }
            }
        }
    };

    auto dequant_w = [&](int kt) {
        constexpr int GROUPS            = BK / 32;
        constexpr int SCALE_CACHE_TILES = 8 / GROUPS;
        // q8_g32_swz64 permutes whole eight-element runs, so eight codes are the widest chunk
        // contiguous in As for every row. BK % 32 keeps gg inside GROUPS and the row stride
        // aligned for the vector store; 8 % GROUPS keeps the scale cache a whole number of tiles.
        constexpr int kChunksPerRow = BK / 8;
        static_assert(BK % 32 == 0 && (8 % GROUPS) == 0,
                      "an eight-code chunk must lie inside one Q8G32 group and the scale cache "
                      "must hold whole tiles");
        const int scale_tile_offset = (kt % SCALE_CACHE_TILES) * GROUPS * 2;
        for (int item = tid; item < BM * kChunksPerRow; item += Cfg::kThreads) {
            const int row   = item / kChunksPerRow;
            const int chunk = item - row * kChunksPerRow;
            const int col   = chunk * 8;
            const int gg    = col >> 5;
            const float scale =
                (FullK || kt * BK + col < k)
                    ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                          &Sr[row * Cfg::kScaleCacheBytes + scale_tile_offset + gg * 2])))
                    : 0.0f;
            const uint2 packed = *reinterpret_cast<const uint2*>(&Cr[row * BK + col]);
            Q8Bf16x8Bits decoded;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const unsigned word = (pair < 2 ? packed.x : packed.y) >> ((pair & 1) * 16);
                const int q0        = static_cast<int>(static_cast<std::int8_t>(word & 0xffu));
                const int q1 = static_cast<int>(static_cast<std::int8_t>((word >> 8) & 0xffu));
                decoded.pair[pair] = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                           static_cast<float>(q1) * scale);
            }
            store_vec(&As[row * BK + q8_g32_swz64(row, col)], decoded.raw);
        }
    };

    const int nkt = padded_k / BK;
    stage_x(0, 0);
    stage_w(0);
    ninfer::ops::cp_commit();

#pragma unroll 4
    for (int kt = 0; kt < nkt; ++kt) {
        const int stage = kt % Cfg::kStages;
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        dequant_w(kt);
        __syncthreads();

        const int next = kt + 1;
        if (next < nkt) {
            if constexpr (Cfg::kActivationStages == Cfg::kStages) {
                stage_x(next % Cfg::kStages, next);
            }
            stage_w(next);
            ninfer::ops::cp_commit();
        }

        unsigned af[2][MT][4];
        unsigned bf[2][NT][2];
        auto load_fragments = [&](int slot, int ks) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int ar = wm * WM + mi * 16 + a_rowoff;
                const int ac = ks * 16 + a_coloff;
                ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                            smem_addr(&As[ar * BK + q8_g32_swz64(ar, ac)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int br = wn * WN + ni * 8 + b_rin;
                const int bc = ks * 16 + b_koff;
                ldmatrix_x2(bf[slot][ni][0], bf[slot][ni][1],
                            smem_addr(&Bs[Cfg::kActivationStages == 1 ? 0 : stage]
                                         [br * BK + q8_g32_swz64(br, bc)]));
            }
        };

        if constexpr (Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::PingPong)
            load_fragments(0, 0);
#pragma unroll
        for (int ks = 0; ks < KSUB; ++ks) {
            const int slot = Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::PingPong ? ks & 1 : 0;
            if constexpr (Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::Serial) {
                load_fragments(0, ks);
            }
            if (Cfg::kFragmentPipeline == Q8MmaFragmentPipeline::PingPong && ks + 1 < KSUB) {
                load_fragments(slot ^ 1, ks + 1);
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                             bf[slot][ni][0], bf[slot][ni][1]);
                }
            }
        }

        if constexpr (Cfg::kActivationStages == 1) {
            if (next < nkt) {
                __syncthreads();
                stage_x(0, next);
                ninfer::ops::cp_commit();
            }
        }
    }

    const auto output_tile = linear_output_tile<kOutputRowsPerCta>(output, m0);
    if constexpr (requires {
                      epilogue.template finish_tile<Cfg, Full>(output_tile, scratch, acc, m0, n0, m,
                                                               n);
                  }) {
        epilogue.template finish_tile<Cfg, Full>(output_tile, scratch, acc, m0, n0, m, n);
    } else {
        const auto store = [&](int row, int token, float value) {
            if (Full || (row < m && token < n))
                output_tile.store(row, token, epilogue.apply(row, token, value));
        };
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int r0 = m0 + wm * WM + mi * 16 + gid;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int c0   = n0 + wn * WN + ni * 8 + 2 * lid;
                const float* a = acc[mi][ni];
                store(r0, c0, a[0]);
                store(r0, c0 + 1, a[1]);
                store(r0 + 8, c0, a[2]);
                store(r0 + 8, c0 + 1, a[3]);
            }
        }
    }
}
} // namespace ninfer::ops::detail
