#pragma once

#include "ops/linear/kernels/fp8/fp8_a8_tma_mma.cuh"
#include "ops/linear/kernels/fp8/native_common.cuh"
#include "ops/linear/kernels/mxfp8/operands.h"
#include "ops/linear/kernels/mxfp8/a8_plan.h"
#include "ops/linear/kernels/fp8/native_schedule.cuh"

namespace ninfer::ops::detail {

template <int Rows, int Tokens, int K, int Stages, bool TmaScales>
struct Mxfp8TmaStageStorage {
    static constexpr int kScaleStages = TmaScales ? 2 : 0;
    static constexpr int kStorageBytes =
        Stages * (Rows + Tokens) * K +
        (TmaScales ? kScaleStages * (Rows + Tokens) * 16 : Stages * (Rows + Tokens) * (K / 32));
    static constexpr int kBarrierBytes = 2 * (Stages + kScaleStages) * sizeof(std::uint64_t);
    static constexpr int kSharedBytes  = (kStorageBytes + 127) / 128 * 128 + kBarrierBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

template <int BlockRows, int BlockTokens, int BlockK = 128, int WarpRows = 32, int WarpTokens = 32,
          int Stages = 2, int MinBlocksPerSm = 1, int RasterGroupRows = 8, bool TmaScales = false,
          int FragmentStages = 2, int RasterGroupTokens = 0>
struct Mxfp8A8TmaSchedule
    : NativeFp8MmaSchedule<BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens, Stages,
                           MinBlocksPerSm, RasterGroupRows>,
      Mxfp8TmaStageStorage<BlockRows, BlockTokens, BlockK, Stages, TmaScales> {
    static constexpr int kConsumerWarps     = (BlockRows / WarpRows) * (BlockTokens / WarpTokens);
    static constexpr int kConsumerThreads   = kConsumerWarps * 32;
    static constexpr int kProducerThreads   = 32;
    static constexpr int kThreads           = kConsumerThreads + kProducerThreads;
    static constexpr bool kTmaScales        = TmaScales;
    static constexpr int kFragmentStages    = FragmentStages;
    static constexpr int kRasterGroupTokens = RasterGroupTokens;
    static_assert(BlockK == 128 || BlockK == 256);
    static_assert(BlockRows <= 256 && BlockTokens <= 256);
    static_assert(kThreads <= 1024 && (FragmentStages == 1 || FragmentStages == 2));
};

struct Mxfp8ScaledTmaDescriptors {
    CUtensorMap activation, weight, activation_scales, weight_scales;
};

template <bool TmaScales>
using Mxfp8TmaDescriptors =
    std::conditional_t<TmaScales, Mxfp8ScaledTmaDescriptors, Fp8TmaDescriptors>;

// A scale box covers K512. Reuse its sixteen UE8M0 bytes across the narrower code tiles.
inline CUtensorMap mxfp8_scale_tma_map(const std::uint8_t* pointer, int rows, int k,
                                       int block_rows) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(k / 32),
                                     static_cast<std::uint64_t>(rows)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(k / 32)};
    const std::uint32_t box[]{16, static_cast<std::uint32_t>(block_rows)}, steps[]{1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<std::uint8_t*>(pointer), dimensions,
        strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("MXFP8 scale TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

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

template <class S, bool TmaScales>
using Mxfp8TmaStorage =
    Mxfp8TmaStageStorage<S::kBlockRows, S::kBlockTokens, S::kBlockK, S::kStages, TmaScales>;

template <class S, class Epilogue, bool TmaScales = S::kTmaScales>
inline constexpr int mxfp8_tma_scratch_bytes = [] {
    constexpr int epilogue = linear_epilogue_bytes<Mxfp8TmaConsumerSchedule<S>, Epilogue>;
    constexpr int staging  = Mxfp8TmaStorage<S, TmaScales>::kStorageBytes;
    constexpr int bytes    = staging > epilogue ? staging : epilogue;
    return (bytes + 127) / 128 * 128;
}();

// Each K128 sector is a separately swizzled TMA box; no global representation changes.
template <int Rows>
__device__ __forceinline__ int mxfp8_tma_code_offset(int row, int col) {
    return (col / 128) * Rows * 128 + row * 128 + ((col & 127) ^ ((row & 7) * 16));
}

template <class S, class Rows>
__device__ __forceinline__ void
mxfp8_tma_scale_stage(const Mxfp8ScaledTmaDescriptors& maps, const Rows& rows, int row_begin,
                      int token_begin, int row_count, int group, unsigned char* weight_scales,
                      unsigned char* input_scales, std::uint64_t* full, std::uint64_t* empty) {
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens;
    constexpr int Panels = mxfp8_tma_panel_rows<S, Rows>;
    const int slot       = group & 1;
    common::cta_mbarrier_wait(empty + slot, 1U ^ ((group / 2) & 1U));
    common::cta_mbarrier_arrive_expect_tx(full + slot, (BR + BT) * 16);
    common::tma_load_2d(input_scales + slot * BT * 16, &maps.activation_scales, group * 16,
                        token_begin, full + slot);
#pragma unroll
    for (int row = 0; row < BR; row += Panels)
        common::tma_load_2d(weight_scales + slot * BR * 16 + row * 16, &maps.weight_scales,
                            group * 16, rows.weight_row(row_begin, row, row_count), full + slot);
}

template <class S, bool Full, bool TmaScales, class Output, class Epilogue, class Rows>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void mxfp8_a8_tma_kernel(
    const __grid_constant__ Mxfp8TmaDescriptors<TmaScales> maps, Mxfp8A8Operands p, Output output,
    Epilogue epilogue, Rows rows) {
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, BK = S::kBlockK, ST = S::kStages;
    constexpr int Groups = BK / 32, Words = BK / 128;
    constexpr int SFSteps = 512 / BK, FS = S::kFragmentStages;
    using Storage            = Mxfp8TmaStorage<S, TmaScales>;
    constexpr int SC         = Storage::kScaleStages;
    constexpr int Panels     = mxfp8_tma_panel_rows<S, Rows>;
    using Consumer           = Mxfp8TmaConsumerSchedule<S>;
    constexpr int code_bytes = ST * (BR + BT) * BK;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* weights       = storage;
    auto* inputs        = weights + ST * BR * BK;
    auto* weight_scales = storage + code_bytes;
    auto* input_scales  = weight_scales + (TmaScales ? SC * BR * 16 : ST * BR * Groups);
    auto* full =
        reinterpret_cast<std::uint64_t*>(storage + mxfp8_tma_scratch_bytes<S, Epilogue, TmaScales>);
    auto* empty    = full + ST;
    auto* sf_full  = empty + ST;
    auto* sf_empty = sf_full + SC;
    const int tid = threadIdx.x, lane = tid & 31;
    const bool producer = tid >= S::kConsumerThreads;
    int row_tile, token_tile;
    native_fp8_tile<S>(blockIdx.x, p.rows, p.tokens, row_tile, token_tile);
    const int row_begin   = row_tile * (BR / (Rows::kPaired ? 2 : 1));
    const int token_begin = token_tile * BT;
    if constexpr (requires { output.clear_if_empty(0, 0, 0, 0); })
        if (output.clear_if_empty(row_begin, token_begin, min(BR, p.rows - row_begin),
                                  min(BT, p.tokens - token_begin)))
            return;
    const int k       = S::kStaticK ? S::kStaticK : p.k;
    const int k_tiles = k / BK;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < ST; ++s) {
            // cp.async scales contribute one arrival per producer lane; TMA scales have
            // separate full/empty barriers and outlive several code tiles.
            common::cta_mbarrier_init(full + s, TmaScales ? 1 : 33);
            common::cta_mbarrier_init(empty + s,
                                      TmaScales ? S::kConsumerWarps : S::kConsumerThreads);
        }
#pragma unroll
        for (int s = 0; s < SC; ++s) {
            common::cta_mbarrier_init(sf_full + s, 1);
            common::cta_mbarrier_init(sf_empty + s, S::kConsumerWarps);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();

    float accum[S::kMmaRows][S::kMmaTokens][4]{};
    if (producer) {
        if constexpr (TmaScales) {
            if (lane == 0)
                for (int group = 0; group < SC && group * SFSteps < k_tiles; ++group)
                    mxfp8_tma_scale_stage<S>(maps, rows, row_begin, token_begin, p.rows, group,
                                             weight_scales, input_scales, sf_full, sf_empty);
        }
#pragma unroll 1
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int slot = kt % ST;
            common::cta_mbarrier_wait(empty + slot, 1U ^ ((kt / ST) & 1U));
            if (lane == 0) {
                if constexpr (TmaScales) {
                    if (kt % SFSteps == 0 && kt / SFSteps >= SC)
                        mxfp8_tma_scale_stage<S>(maps, rows, row_begin, token_begin, p.rows,
                                                 kt / SFSteps, weight_scales, input_scales, sf_full,
                                                 sf_empty);
                }
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
            if constexpr (!TmaScales) {
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
                    common::cp_async_zfill<4>(
                        input_scales + slot * BT * Groups + item * 4,
                        p.x_scales + (valid ? static_cast<std::int64_t>(token) * (k / 32) +
                                                  kt * Groups + word * 4
                                            : 0),
                        valid ? 4 : 0);
                }
                // Close this slot's scale-copy group before publishing its arrival.
                // The independent TMA transaction may still be in flight; the full
                // barrier also waits for its expected bytes before consumers proceed.
                common::cp_commit();
                common::cp_wait<0>();
                common::cta_mbarrier_arrive(full + slot);
            }
        }
    } else {
        const int warp = tid >> 5;
        const int wr = warp / S::kWarpsTokens, wt = warp % S::kWarpsTokens;
#pragma unroll 1
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int slot = kt % ST;
            common::cta_mbarrier_wait(full + slot, (kt / ST) & 1U);
            const int sf_group = kt / SFSteps, sf_slot = sf_group & 1;
            if constexpr (TmaScales)
                if (kt % SFSteps == 0)
                    common::cta_mbarrier_wait(sf_full + sf_slot, (sf_group / 2) & 1U);
            const auto* ws = weights + slot * BR * BK;
            const auto* xs = inputs + slot * BT * BK;
            unsigned sw[S::kMmaRows][Words], sx[S::kMmaTokens][Words];
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                const int row =
                    wr * S::kWarpRows + mi * 16 + (lane >> 2) + ((lane & 3) == 1 ? 8 : 0);
#pragma unroll
                for (int word = 0; word < Words; ++word)
                    sw[mi][word] = common::load_vec<unsigned>(
                        weight_scales +
                        (TmaScales ? sf_slot * BR * 16 + row * 16 + (kt % SFSteps) * Groups
                                   : slot * BR * Groups + row * Groups) +
                        word * 4);
            }
#pragma unroll
            for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                const int token = wt * S::kWarpTokens + ni * 8 + (lane >> 2);
#pragma unroll
                for (int word = 0; word < Words; ++word)
                    sx[ni][word] = common::load_vec<unsigned>(
                        input_scales +
                        (TmaScales ? sf_slot * BT * 16 + token * 16 + (kt % SFSteps) * Groups
                                   : slot * BT * Groups + token * Groups) +
                        word * 4);
            }
            unsigned af[FS][S::kMmaRows][4], bf[FS][S::kMmaTokens][2];
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
            if constexpr (FS == 2) load(0, 0);
#pragma unroll
            for (int step = 0; step < Groups; ++step) {
                const int frag = FS == 1 ? 0 : step & 1;
                if constexpr (FS == 1)
                    load(0, step);
                else if (step + 1 < Groups)
                    load(frag ^ 1, step + 1);
#pragma unroll
                for (int mi = 0; mi < S::kMmaRows; ++mi) {
                    const unsigned sa = sw[mi][step / 4];
#pragma unroll
                    for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                        const unsigned sb = sx[ni][step / 4];
                        auto& a           = accum[mi][ni];
                        common::mma_fp8_e4m3_scaled(a[0], a[1], a[2], a[3], af[frag][mi][0],
                                                    af[frag][mi][1], af[frag][mi][2],
                                                    af[frag][mi][3], bf[frag][ni][0],
                                                    bf[frag][ni][1], sa, sb, step % 4);
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            // cp.async scale slots are read by every consumer lane. Retire each
            // reader explicitly before the producer can overwrite the slot.
            if constexpr (!TmaScales) {
                common::cta_mbarrier_arrive(empty + slot);
            } else if (lane == 0) {
                common::cta_mbarrier_arrive(empty + slot);
                if (kt % SFSteps == SFSteps - 1) common::cta_mbarrier_arrive(sf_empty + sf_slot);
            }
        }
    }
    // No stage is borrowed after this point, including transfers issued by the producer.
    __syncthreads();
    if (producer) return;
    native_fp8_finish<Consumer, Full>(output, epilogue, storage, accum, row_begin, token_begin,
                                      p.rows, p.tokens);
}

template <class S, bool PaddedCodes = false, class Output, class Epilogue,
          class Rows = Fp8IdentityRows>
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
    const auto launch = [&]<bool TmaScales, bool Full>() {
        const auto maps = [&] {
            const int code_tokens =
                PaddedCodes ? mxfp8_a8_padded_tokens(p.tokens, S::kBlockTokens) : p.tokens;
            const auto xmap = fp8_tma_map(p.x, code_tokens, p.k, S::kBlockTokens, 128);
            const auto wmap = fp8_tma_map(p.codes, p.rows, p.k, panel, 128);
            if constexpr (TmaScales)
                return Mxfp8ScaledTmaDescriptors{
                    xmap, wmap, mxfp8_scale_tma_map(p.x_scales, p.tokens, p.k, S::kBlockTokens),
                    mxfp8_scale_tma_map(p.scales, p.rows, p.k, panel)};
            else
                return Fp8TmaDescriptors{xmap, wmap};
        }();
        constexpr auto kernel = mxfp8_a8_tma_kernel<S, Full, TmaScales, Output, Epilogue, Rows>;
        constexpr int bytes   = mxfp8_tma_scratch_bytes<S, Epilogue, TmaScales> +
                              Mxfp8TmaStorage<S, TmaScales>::kBarrierBytes;
        const auto shared = common::prepare_shared<bytes, kernel, true>();
        kernel<<<static_cast<unsigned>(blocks), S::kThreads, shared, stream>>>(maps, p, output,
                                                                               epilogue, rows);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto selected = [&]<bool TmaScales>() {
        if (p.tokens % S::kBlockTokens == 0 && p.rows % S::kBlockRows == 0)
            launch.template operator()<TmaScales, true>();
        else
            launch.template operator()<TmaScales, false>();
    };
    if constexpr (S::kTmaScales) {
        // Preserve the existing four-byte scale-plane support without repacking weights.
        if (p.k % 512 == 0 && reinterpret_cast<std::uintptr_t>(p.scales) % 16 == 0 &&
            reinterpret_cast<std::uintptr_t>(p.x_scales) % 16 == 0)
            return selected.template operator()<true>();
    }
    selected.template operator()<false>();
}

} // namespace ninfer::ops::detail
