#pragma once

#include "ops/linear/kernels/fp8/fp8_a8_tma_mma.cuh"
#include "ops/linear/kernels/fp8/native_common.cuh"
#include "ops/linear/kernels/mxfp8/operands.h"
#include "ops/linear/kernels/fp8/native_schedule.cuh"

namespace ninfer::ops::detail {

template <int BlockRows, int BlockTokens, int BlockK = 128, int WarpRows = 32, int WarpTokens = 32,
          int Stages = 2, int MinBlocksPerSm = 1, int RasterGroupRows = 8>
struct Mxfp8A8TmaSchedule
    : NativeFp8MmaSchedule<BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens, Stages,
                           MinBlocksPerSm, RasterGroupRows> {
    static constexpr int kConsumerWarps   = (BlockRows / WarpRows) * (BlockTokens / WarpTokens);
    static constexpr int kConsumerThreads = kConsumerWarps * 32;
    static constexpr int kProducerThreads = 32;
    static constexpr int kThreads         = kConsumerThreads + kProducerThreads;
    static constexpr int kStorageBytes =
        Stages * (BlockRows + BlockTokens) * (BlockK + BlockK / 32);
    static constexpr int kBarrierBytes = 2 * Stages * sizeof(std::uint64_t);
    static constexpr int kSharedBytes  = (kStorageBytes + 127) / 128 * 128 + kBarrierBytes;
    static_assert(BlockK == 128 || BlockK == 256);
    static_assert(BlockRows <= 256 && BlockTokens <= 256);
    static_assert(kThreads <= 1024 && kSharedBytes <= 99 * 1024);
};

// The producer occupies the final warp. Existing epilogues keep zero-based consumer indices.
template <class S>
struct Mxfp8TmaConsumerSchedule : S {
    static constexpr int kThreads         = S::kConsumerThreads;
    static constexpr int kProducerThreads = 0;
};

template <class S, class Rows>
inline constexpr int mxfp8_tma_panel_rows = [] {
    if constexpr (requires { Rows::kTmaRowsPerPanel; })
        return Rows::kTmaRowsPerPanel;
    else if constexpr (Rows::kPaired)
        return S::kBlockRows / 2;
    else
        return S::kBlockRows;
}();

template <class S, class Epilogue>
inline constexpr int mxfp8_tma_scratch_bytes = [] {
    constexpr int epilogue = linear_epilogue_bytes<Mxfp8TmaConsumerSchedule<S>, Epilogue>;
    constexpr int bytes    = S::kStorageBytes > epilogue ? S::kStorageBytes : epilogue;
    return (bytes + 127) / 128 * 128;
}();

// Each K128 sector is a separately swizzled TMA box; no global representation changes.
template <int Rows>
__device__ __forceinline__ int mxfp8_tma_code_offset(int row, int col) {
    return (col / 128) * Rows * 128 + row * 128 + ((col & 127) ^ ((row & 7) * 16));
}

template <class S, bool Full, class Output, class Epilogue, class Rows>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void mxfp8_a8_tma_kernel(
    const __grid_constant__ Fp8TmaDescriptors maps, Mxfp8A8Operands p, Output output,
    Epilogue epilogue, Rows rows) {
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, BK = S::kBlockK, ST = S::kStages;
    constexpr int Groups = BK / 32, Words = BK / 128;
    constexpr int Panels     = mxfp8_tma_panel_rows<S, Rows>;
    using Consumer           = Mxfp8TmaConsumerSchedule<S>;
    constexpr int code_bytes = ST * (BR + BT) * BK;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* weights       = storage;
    auto* inputs        = weights + ST * BR * BK;
    auto* weight_scales = storage + code_bytes;
    auto* input_scales  = weight_scales + ST * BR * Groups;
    auto* full  = reinterpret_cast<std::uint64_t*>(storage + mxfp8_tma_scratch_bytes<S, Epilogue>);
    auto* empty = full + ST;
    const int tid = threadIdx.x, lane = tid & 31;
    const bool producer = tid >= S::kConsumerThreads;
    int row_tile, token_tile;
    native_fp8_tile<S>(blockIdx.x, p.rows, p.tokens, row_tile, token_tile);
    const int row_begin   = row_tile * (BR / (Rows::kPaired ? 2 : 1));
    const int token_begin = token_tile * BT;
    const int k           = S::kStaticK ? S::kStaticK : p.k;
    const int k_tiles     = k / BK;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < ST; ++s) {
            // One TMA arrival and one asynchronous scale-copy arrival per producer lane.
            common::cta_mbarrier_init(full + s, 33);
            common::cta_mbarrier_init(empty + s, S::kConsumerWarps);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();

    float accum[S::kMmaRows][S::kMmaTokens][4]{};
    if (producer) {
#pragma unroll 1
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int slot = kt % ST;
            common::cta_mbarrier_wait(empty + slot, 1U ^ ((kt / ST) & 1U));
            if (lane == 0) {
                common::cta_mbarrier_arrive_expect_tx(full + slot, (BR + BT) * BK);
#pragma unroll
                for (int sector = 0; sector < Words; ++sector) {
                    common::tma_load_2d(inputs + slot * BT * BK + sector * BT * 128,
                                        &maps.activation, kt * BK + sector * 128, token_begin,
                                        full + slot);
#pragma unroll
                    for (int row = 0; row < BR; row += Panels) {
                        common::tma_load_2d(weights + slot * BR * BK + sector * BR * 128 +
                                                row * 128,
                                            &maps.weight, kt * BK + sector * 128,
                                            rows.weight_row(row_begin, row, p.rows), full + slot);
                    }
                }
            }
            for (int item = lane; item < BR * Words; item += 32) {
                const int row = item / Words, word = item % Words;
                const int source_row = rows.weight_row(row_begin, row, p.rows);
                const bool valid     = source_row < p.rows;
                common::cp_async_zfill<4>(
                    weight_scales + slot * BR * Groups + item * 4,
                    p.scales + (valid ? static_cast<std::int64_t>(source_row) * (k / 32) +
                                            kt * Groups + word * 4
                                      : 0),
                    valid ? 4 : 0);
            }
            for (int item = lane; item < BT * Words; item += 32) {
                const int token = token_begin + item / Words, word = item % Words;
                const bool valid = Full || token < p.tokens;
                common::cp_async_zfill<4>(input_scales + slot * BT * Groups + item * 4,
                                          p.x_scales +
                                              (valid ? static_cast<std::int64_t>(token) * (k / 32) +
                                                           kt * Groups + word * 4
                                                     : 0),
                                          valid ? 4 : 0);
            }
            // The full barrier tracks both transfer mechanisms, including every scale byte.
            asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(
                             common::smem_addr(full + slot))
                         : "memory");
        }
    } else {
        const int warp = tid >> 5;
        const int wr = warp / S::kWarpsTokens, wt = warp % S::kWarpsTokens;
#pragma unroll 1
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int slot = kt % ST;
            common::cta_mbarrier_wait(full + slot, (kt / ST) & 1U);
            const auto* ws = weights + slot * BR * BK;
            const auto* xs = inputs + slot * BT * BK;
            unsigned sw[S::kMmaRows][Words], sx[S::kMmaTokens][Words];
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                const int row =
                    wr * S::kWarpRows + mi * 16 + (lane >> 2) + ((lane & 3) == 1 ? 8 : 0);
#pragma unroll
                for (int word = 0; word < Words; ++word)
                    sw[mi][word] = common::load_vec<unsigned>(weight_scales + slot * BR * Groups +
                                                              row * Groups + word * 4);
            }
#pragma unroll
            for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                const int token = wt * S::kWarpTokens + ni * 8 + (lane >> 2);
#pragma unroll
                for (int word = 0; word < Words; ++word)
                    sx[ni][word] = common::load_vec<unsigned>(input_scales + slot * BT * Groups +
                                                              token * Groups + word * 4);
            }
            unsigned af[2][S::kMmaRows][4], bf[2][S::kMmaTokens][2];
            const auto load = [&](int frag, int step) {
#pragma unroll
                for (int mi = 0; mi < S::kMmaRows; ++mi) {
                    const int row =
                        wr * S::kWarpRows + mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int col = step * 32 + (lane >> 4) * 16;
                    common::ldmatrix_x4(
                        af[frag][mi][0], af[frag][mi][1], af[frag][mi][2], af[frag][mi][3],
                        common::smem_addr(ws + mxfp8_tma_code_offset<BR>(row, col)));
                }
#pragma unroll
                for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                    const int token = wt * S::kWarpTokens + ni * 8 + (lane & 7);
                    const int col   = step * 32 + ((lane >> 3) & 1) * 16;
                    common::ldmatrix_x2(
                        bf[frag][ni][0], bf[frag][ni][1],
                        common::smem_addr(xs + mxfp8_tma_code_offset<BT>(token, col)));
                }
            };
            load(0, 0);
#pragma unroll
            for (int step = 0; step < Groups; ++step) {
                const int frag = step & 1;
                if (step + 1 < Groups) load(frag ^ 1, step + 1);
#pragma unroll
                for (int mi = 0; mi < S::kMmaRows; ++mi) {
                    const unsigned sa = (sw[mi][step / 4] >> ((step % 4) * 8)) & 255U;
#pragma unroll
                    for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                        const unsigned sb = (sx[ni][step / 4] >> ((step % 4) * 8)) & 255U;
                        auto& a           = accum[mi][ni];
                        common::mma_fp8_e4m3_scaled(a[0], a[1], a[2], a[3], af[frag][mi][0],
                                                    af[frag][mi][1], af[frag][mi][2],
                                                    af[frag][mi][3], bf[frag][ni][0],
                                                    bf[frag][ni][1], sa, sb);
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0) common::cta_mbarrier_arrive(empty + slot);
        }
    }
    // No stage is borrowed after this point, including transfers issued by the producer.
    __syncthreads();
    if (producer) return;
    native_fp8_finish<Consumer, Full>(output, epilogue, storage, accum, row_begin, token_begin,
                                      p.rows, p.tokens);
}

template <class S, class Output, class Epilogue, class Rows = Fp8IdentityRows>
void launch_mxfp8_a8_tma(const Mxfp8A8Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream, Rows rows = {}) {
    constexpr int panel = mxfp8_tma_panel_rows<S, Rows>;
    static_assert(panel > 0 && panel % 8 == 0 && S::kBlockRows % panel == 0);
    validate_native_fp8<S>(p);
    if (!p.x_scales || reinterpret_cast<std::uintptr_t>(p.x) % 16 ||
        reinterpret_cast<std::uintptr_t>(p.codes) % 16 ||
        reinterpret_cast<std::uintptr_t>(p.x_scales) % 4 ||
        reinterpret_cast<std::uintptr_t>(p.scales) % 4 || p.k % S::kBlockK)
        throw std::invalid_argument("MXFP8 TMA requires aligned codes/scales and complete K tiles");
    const auto blocks = static_cast<std::int64_t>(common::div_up(p.rows, S::kBlockRows)) *
                        common::div_up(p.tokens, S::kBlockTokens);
    if (blocks > 2147483647LL) throw std::invalid_argument("MXFP8 TMA grid exceeds CUDA grid.x");
    const Fp8TmaDescriptors maps{fp8_tma_map(p.x, p.tokens, p.k, S::kBlockTokens, 128),
                                 fp8_tma_map(p.codes, p.rows, p.k, panel, 128)};
    const auto launch = [&]<bool Full>() {
        constexpr auto kernel = mxfp8_a8_tma_kernel<S, Full, Output, Epilogue, Rows>;
        constexpr int bytes   = mxfp8_tma_scratch_bytes<S, Epilogue> + S::kBarrierBytes;
        const auto shared     = common::prepare_shared<bytes, kernel, true>();
        kernel<<<static_cast<unsigned>(blocks), S::kThreads, shared, stream>>>(maps, p, output,
                                                                               epilogue, rows);
        CUDA_CHECK(cudaGetLastError());
    };
    if (p.tokens % S::kBlockTokens == 0 && p.rows % S::kBlockRows == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

} // namespace ninfer::ops::detail
