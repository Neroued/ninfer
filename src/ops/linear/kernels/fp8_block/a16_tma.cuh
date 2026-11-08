#pragma once
#include "ops/common/quant/fp8.cuh"
#include "ops/linear/kernels/fp8_block/operands.h"
#include "ops/linear/kernels/fp8/native_common.cuh"
#include "ops/linear/kernels/fp8/fp8_a8_tma_mma.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_mma.cuh"

namespace ninfer::ops::detail {
template <int BR, int BT, int BK, int WR, int WT, int ST, int GR = 8>
struct Fp8BlockA16TmaSchedule : NativeFp8MmaSchedule<BR, BT, BK, WR, WT, ST, 1, GR> {
    static constexpr int kConsumerWarps   = (BR / WR) * (BT / WT);
    static constexpr int kConsumerThreads = kConsumerWarps * 32;
    static constexpr int kProducerThreads = 32;
    static constexpr int kThreads         = kConsumerThreads + 32;
    static constexpr bool kTmaSwizzle     = true;
    static constexpr int kSegmentsPerRow  = BK / 16;
    static constexpr auto kSwizzle        = Bf16MmaSwizzle::Tma128;
    static constexpr int kStorageBytes    = ST * (BR + 2 * BT) * BK;
    static_assert(128 % BR == 0 && BT <= 256 && kThreads <= 1024);
    static_assert(BK == 64 || BK == 128);
};

template <class S>
struct Fp8BlockTmaConsumerSchedule : S {
    static constexpr int kThreads         = S::kConsumerThreads;
    static constexpr int kProducerThreads = 0;
};
template <class S, class E>
inline constexpr int fp8_block_tma_scratch_bytes = [] {
    constexpr int bytes =
        linear_epilogue_bytes<Fp8BlockTmaConsumerSchedule<S>, E> > S::kStorageBytes
            ? linear_epilogue_bytes<Fp8BlockTmaConsumerSchedule<S>, E>
            : S::kStorageBytes;
    return (bytes + 127) / 128 * 128;
}();

template <class S, bool Full, bool Partial, class O, class E, bool PairedRows = false>
__device__ __forceinline__ void
fp8_block_a16_tma_body(const Fp8TmaDescriptors& maps, Fp8BlockA16Operands<> p, O output, E epilogue,
                       int token_offset, int tile_idx, int first_k_tile, int tiles,
                       float* partials) {
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, BK = S::kBlockK, ST = S::kStages;
    static_assert(!PairedRows || (!Partial && S::kWarpsRows == 2));
    extern __shared__ __align__(128) unsigned char raw[];
    auto* weights = raw;
    auto* inputs  = reinterpret_cast<__nv_bfloat16*>(raw + ST * BR * BK);
    auto* full    = reinterpret_cast<std::uint64_t*>(raw + fp8_block_tma_scratch_bytes<S, E>);
    auto* empty   = full + ST;
    int ri, ti;
    native_fp8_tile<S>(tile_idx, p.rows, p.tokens, ri, ti);
    const int rb = ri * (PairedRows ? BR / 2 : BR), tb = token_offset + ti * BT,
              K   = S::kStaticK ? S::kStaticK : p.k;
    const int tid = threadIdx.x, lane = tid & 31;
    const bool producer = tid >= S::kConsumerThreads;
    if (tid == 0) {
#pragma unroll
        for (int slot = 0; slot < ST; ++slot) {
            common::cta_mbarrier_init(full + slot, 1);
            common::cta_mbarrier_init(empty + slot, S::kConsumerWarps);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    float accum[S::kMmaRows][S::kMmaTokens][4]{};
    if (producer) {
#pragma unroll 1
        for (int kt = 0; kt < tiles; ++kt) {
            int slot = kt % ST;
            common::cta_mbarrier_wait(empty + slot, 1U ^ ((kt / ST) & 1U));
            if (lane == 0) {
                common::cta_mbarrier_arrive_expect_tx(full + slot, (BR + 2 * BT) * BK);
                common::tma_load_2d(weights + slot * BR * BK, &maps.weight,
                                    (kt + first_k_tile) * BK, rb, full + slot);
                if constexpr (PairedRows)
                    common::tma_load_2d(weights + slot * BR * BK + BR / 2 * BK, &maps.weight,
                                        (kt + first_k_tile) * BK, rb + p.rows / 2, full + slot);
                common::tma_load_3d(inputs + slot * BT * BK, &maps.activation, 0,
                                    (kt + first_k_tile) * (BK / 64), tb, full + slot);
            }
        }
    } else {
        const int warp = tid >> 5, wr = warp / S::kWarpsTokens, wt = warp % S::kWarpsTokens;
        // Keep the unscaled FP32 sum across both K64 transfers. Apply the represented
        // scale only after the complete K128 group, matching the K128 arithmetic.
        float partial[S::kMmaRows][S::kMmaTokens][4]{};
#pragma unroll 1
        for (int kt = 0; kt < tiles; ++kt) {
            int slot = kt % ST;
            common::cta_mbarrier_wait(full + slot, (kt / ST) & 1U);
            const auto* ws = weights + slot * BR * BK;
            const auto* xs = inputs + slot * BT * BK;
            unsigned af[S::kMmaRows][4], bf[S::kMmaTokens][2];
            const auto load = [&](int kk) {
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const int row   = wr * S::kWarpRows + r * 16 + (lane >> 2),
                              col   = kk + (lane & 3) * 2;
                    const auto pair = [&](int row, int col) {
                        return common::fp8_e4m3x2_to_bf16x2_bits(common::load_vec<std::uint16_t>(
                            ws + row * BK + fp8_mma_shared_byte<S>(row, col)));
                    };
                    af[r][0] = pair(row, col);
                    af[r][1] = pair(row + 8, col);
                    af[r][2] = pair(row, col + 8);
                    af[r][3] = pair(row + 8, col + 8);
                }
#pragma unroll
                for (int t = 0; t < S::kMmaTokens; ++t) {
                    const int row = wt * S::kWarpTokens + t * 8 + (lane & 7),
                              col = kk + ((lane >> 3) & 1) * 8;
                    common::ldmatrix_x2(
                        bf[t][0], bf[t][1],
                        common::smem_addr(xs + row * BK + bf16_mma_shared_col<S>(row, col)));
                }
            };
#pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                load(kk);
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
#pragma unroll
                    for (int t = 0; t < S::kMmaTokens; ++t) {
                        auto& a = partial[r][t];
                        common::mma_bf16(a[0], a[1], a[2], a[3], af[r][0], af[r][1], af[r][2],
                                         af[r][3], bf[t][0], bf[t][1]);
                    }
                }
            }
            if ((kt + 1) % (128 / BK) == 0) {
                // Each row warp stays within a represented N128 scale group. Paired
                // tiles load the second half from its own physical row range.
                const int scale_row = rb + (PairedRows && wr == 1 ? p.rows / 2 : 0);
                const float scale   = p.scales[std::int64_t(scale_row / 128) * (K / 128) +
                                             (kt + first_k_tile) * BK / 128];
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
#pragma unroll
                    for (int t = 0; t < S::kMmaTokens; ++t) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            accum[r][t][j]   = fmaf(partial[r][t][j], scale, accum[r][t][j]);
                            partial[r][t][j] = 0;
                        }
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0) common::cta_mbarrier_arrive(empty + slot);
        }
    }
    __syncthreads();
    if (producer) return;
    if constexpr (Partial) {
#pragma unroll
        for (int r = 0; r < S::kMmaRows; ++r)
#pragma unroll
            for (int t = 0; t < S::kMmaTokens; ++t)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    partials[((r * S::kMmaTokens + t) * 4 + j) * S::kConsumerThreads + tid] =
                        accum[r][t][j];
    } else {
        native_fp8_finish<Fp8BlockTmaConsumerSchedule<S>, Full>(
            output, epilogue, raw, accum, rb, tb, p.rows, token_offset + p.tokens);
    }
}

template <class S, bool Full, class O, class E, bool PairedRows = false>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void fp8_block_a16_tma_kernel(
    const __grid_constant__ Fp8TmaDescriptors maps, Fp8BlockA16Operands<> p, O output, E epilogue,
    int token_offset) {
    const int k = S::kStaticK ? S::kStaticK : p.k;
    fp8_block_a16_tma_body<S, Full, false, O, E, PairedRows>(
        maps, p, output, epilogue, token_offset, int(blockIdx.x), 0, k / S::kBlockK, nullptr);
}

template <class S, bool PairedRows = false>
Fp8TmaDescriptors fp8_block_a16_tma_descriptors(const Fp8BlockA16Operands<>& p) {
    validate_native_fp8<S>(p);
    if (p.rows % 128 || reinterpret_cast<std::uintptr_t>(p.x) % 16 ||
        reinterpret_cast<std::uintptr_t>(p.codes) % 16 ||
        reinterpret_cast<std::uintptr_t>(p.scales) % 4)
        throw std::invalid_argument(
            "FP8 block TMA requires N128, aligned codes/activations and FP32 scales");
    return Fp8TmaDescriptors{
        bf16_tma_map<S>(p.x, p.tokens, p.k, S::kBlockTokens, S::kBlockK),
        fp8_tma_map(p.codes, p.rows, p.k, S::kBlockRows / (PairedRows ? 2 : 1), S::kBlockK)};
}

template <class S, class E>
inline constexpr int fp8_block_a16_tma_bytes =
    fp8_block_tma_scratch_bytes<S, E> + 2 * S::kStages * sizeof(std::uint64_t);

// Descriptors and scratch are launch-owned. Transfer slices cannot split a K128 scale group.
template <class S, bool PairedRows = false, class O, class E>
void launch_fp8_block_a16_tma(const Fp8BlockA16Operands<>& p, O output, E epilogue,
                              cudaStream_t stream) {
    const auto maps     = fp8_block_a16_tma_descriptors<S, PairedRows>(p);
    constexpr int bytes = fp8_block_a16_tma_bytes<S, E>;
    static_assert(bytes <= 99 * 1024);
    common::for_each_token_slice(p.tokens, S::kBlockTokens, [&](int offset, int count) {
        auto slice        = p;
        slice.tokens      = count;
        const auto blocks = std::int64_t(common::div_up(p.rows, S::kBlockRows)) *
                            common::div_up(count, S::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("FP8 block TMA grid exceeds CUDA grid.x");
        const auto run = [&]<bool Full>() {
            constexpr auto fn = fp8_block_a16_tma_kernel<S, Full, O, E, PairedRows>;
            common::prepare_shared<bytes, fn, true>();
            fn<<<unsigned(blocks), S::kThreads, bytes, stream>>>(maps, slice, output, epilogue,
                                                                 offset);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % S::kBlockTokens == 0)
            run.template operator()<true>();
        else
            run.template operator()<false>();
    });
}
} // namespace ninfer::ops::detail
