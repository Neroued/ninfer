#pragma once

// Q5G64 RowSplit x BF16 Tensor Core GEMM.
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// Each CTA owns a BlockRows x BlockCols output tile and advances K in one
// 64-value quant group at a time. Raw Q5 code/high planes and FP16 scales are
// staged independently from the BF16 activation tile. Q5 values are decoded
// to BF16 in shared memory, then consumed by m16n8k16 BF16 MMA with FP32
// accumulation.
//
// The template describes only physical kernel structure. Rows, Cols, and K
// remain runtime dimensions; registered shapes select a closed schedule and a
// statically compiled boundary variant outside the kernel.

#include "ops/common/mma.cuh"
#include "ops/linear/q5/q5_rowsplit_launch.h"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q5FragmentPipeline {
    Serial,
    PingPong,
};

enum class Q5ScaleLoad {
    Scalar16,
    Pair32,
};

enum class Q5MmaEpilogue {
    Store,
    AddResidual,
};

template <int BlockRows_, int BlockCols_, int BlockK_, int WarpRows_, int WarpCols_,
          int PipelineStages_, int LaunchBoundsMinBlocks_, Q5FragmentPipeline FragmentPipeline_,
          Cache QuantCache_, Cache ActivationCache_, Q5ScaleLoad ScaleLoadMode_>
struct Q5RowSplitMmaGemmSchedule {
    static constexpr int kBlockRows = BlockRows_;
    static constexpr int kBlockCols = BlockCols_;
    static constexpr int kBlockK    = BlockK_;
    static constexpr int kWarpRows  = WarpRows_;
    static constexpr int kWarpCols  = WarpCols_;

    static constexpr int kPipelineStages                  = PipelineStages_;
    static constexpr int kLaunchBoundsMinBlocks           = LaunchBoundsMinBlocks_;
    static constexpr Q5FragmentPipeline kFragmentPipeline = FragmentPipeline_;
    static constexpr Cache kQuantCache                    = QuantCache_;
    static constexpr Cache kActivationCache               = ActivationCache_;
    static constexpr Q5ScaleLoad kScaleLoadMode           = ScaleLoadMode_;

    static constexpr int kWarpGridRows = kBlockRows / kWarpRows;
    static constexpr int kWarpGridCols = kBlockCols / kWarpCols;
    static constexpr int kWarps        = kWarpGridRows * kWarpGridCols;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kMmaRows      = kWarpRows / 16;
    static constexpr int kMmaCols      = kWarpCols / 8;
    static constexpr int kMmaKSteps    = kBlockK / 16;
    static constexpr int kGroupsPerK   = kBlockK / Q5RowSplitStorage::kGroupK;
    static constexpr int kScaleBytes =
        kScaleLoadMode == Q5ScaleLoad::Pair32 ? 4 : Q5RowSplitStorage::kScaleBytesPerGroup;

    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kPipelineStages * kBlockCols * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kPipelineStages * kBlockRows * kGroupsPerK * Q5RowSplitStorage::kCodeBytesPerGroup +
        kPipelineStages * kBlockRows * kGroupsPerK * Q5RowSplitStorage::kHighBytesPerGroup +
        kPipelineStages * kBlockRows * kGroupsPerK * kScaleBytes;

    static_assert(kBlockRows > 0 && kBlockCols > 0);
    static_assert(kBlockK == Q5RowSplitStorage::kGroupK,
                  "Q5 MMA currently stages exactly one quant group per K tile");
    static_assert(kBlockRows % kWarpRows == 0 && kBlockCols % kWarpCols == 0,
                  "Q5 MMA block tile must divide into warp tiles");
    static_assert(kWarpRows % 16 == 0 && kWarpCols % 8 == 0,
                  "Q5 MMA warp tile must be composed of m16n8 MMA tiles");
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8,
                  "Q5 MMA cp.async pipeline depth must fit cp_wait");
    static_assert(kLaunchBoundsMinBlocks >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024,
                  "Q5 MMA staged shared memory exceeds the static 48 KiB budget");
};

// XOR swizzle for a [rows][64] BF16 tile. The eight 16-byte column groups are
// permuted by the low row bits so ldmatrix reads do not repeatedly hit the same
// shared-memory bank group.
__device__ __forceinline__ int q5_mma_swizzle_k64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// clang-format off
template <class Schedule_, Q5KernelVariant Variant,
          Q5MmaEpilogue Epilogue = Q5MmaEpilogue::Store>
__global__ __launch_bounds__(Schedule_::kThreads, Schedule_::kLaunchBoundsMinBlocks)
void q5_rowsplit_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high,
    const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ out,
    std::int32_t rows,
    std::int32_t k,
    std::int32_t cols,
    std::int32_t padded_k) {
    // clang-format on
    using Schedule = Schedule_;
    static_assert(Variant == Q5KernelVariant::Full || Variant == Q5KernelVariant::Predicated,
                  "Q5 MMA requires a tiled kernel variant");
    static_assert(Epilogue == Q5MmaEpilogue::Store || Epilogue == Q5MmaEpilogue::AddResidual,
                  "Q5 MMA requires a supported epilogue");

    constexpr bool kFull = Variant == Q5KernelVariant::Full;
    constexpr int BM     = Schedule::kBlockRows;
    constexpr int BN     = Schedule::kBlockCols;
    constexpr int BK     = Schedule::kBlockK;
    constexpr int WM     = Schedule::kWarpRows;
    constexpr int WN     = Schedule::kWarpCols;
    constexpr int MT     = Schedule::kMmaRows;
    constexpr int NT     = Schedule::kMmaCols;
    constexpr int KSUB   = Schedule::kMmaKSteps;
    constexpr int S      = Schedule::kPipelineStages;
    constexpr int GPB    = Schedule::kGroupsPerK;
    constexpr int SB     = Schedule::kScaleBytes;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * GPB * Q5RowSplitStorage::kCodeBytesPerGroup];
    __shared__ __align__(16) std::uint8_t Hr[S][BM * GPB * Q5RowSplitStorage::kHighBytesPerGroup];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * GPB * SB];

    const int groups_per_row = padded_k / Q5RowSplitStorage::kGroupK;
    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;
    const int warp_row       = warp / Schedule::kWarpGridCols;
    const int warp_col       = warp % Schedule::kWarpGridCols;
    const int mma_row        = lane >> 2;
    const int mma_col        = lane & 3;

    const int row0 = static_cast<int>(blockIdx.x) * BM;
    const int col0 = static_cast<int>(blockIdx.y) * BN;

    float accum[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            accum[mi][ni][0] = 0.0f;
            accum[mi][ni][1] = 0.0f;
            accum[mi][ni][2] = 0.0f;
            accum[mi][ni][3] = 0.0f;
        }
    }

    const int k_tiles = padded_k / BK;

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    auto stage_activation = [&](int stage, int k_tile) {
        const int k0 = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Schedule::kThreads) {
            const int local_col = item / (BK / 8);
            const int k8        = item - local_col * (BK / 8);
            const int kk        = k0 + k8 * 8;
            const int col       = col0 + local_col;
            auto* dst = &Bs[stage][local_col * BK + q5_mma_swizzle_k64(local_col, k8 * 8)];
            if constexpr (kFull) {
                cp_async<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                if (col < cols && kk + 8 <= k) {
                    cp_async<16, Schedule::kActivationCache>(
                        dst, &x[static_cast<std::int64_t>(col) * k + kk]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
    };

    auto stage_quant = [&](int stage, int k_tile) {
        const int group0 = (k_tile * BK) / Q5RowSplitStorage::kGroupK;
#pragma unroll 1
        for (int item = tid; item < BM * GPB * 2; item += Schedule::kThreads) {
            const int row_group = item >> 1;
            const int half      = item & 1;
            const int local_row = row_group / GPB;
            const int group     = row_group - local_row * GPB;
            const int row       = row0 + local_row;
            auto* dst = &Cr[stage][row_group * Q5RowSplitStorage::kCodeBytesPerGroup + half * 16];
            if constexpr (kFull) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                cp_async<16, Schedule::kQuantCache>(
                    dst, &codes[group_index * Q5RowSplitStorage::kCodeBytesPerGroup + half * 16]);
            } else {
                if (row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                    cp_async<16, Schedule::kQuantCache>(
                        dst,
                        &codes[group_index * Q5RowSplitStorage::kCodeBytesPerGroup + half * 16]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }

#pragma unroll 1
        for (int row_group = tid; row_group < BM * GPB; row_group += Schedule::kThreads) {
            const int local_row = row_group / GPB;
            const int group     = row_group - local_row * GPB;
            const int row       = row0 + local_row;
            auto* dst           = &Hr[stage][row_group * Q5RowSplitStorage::kHighBytesPerGroup];
            if constexpr (kFull) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                cp_async<8>(dst, &high[group_index * Q5RowSplitStorage::kHighBytesPerGroup]);
            } else {
                if (row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + group0 + group;
                    cp_async<8>(dst, &high[group_index * Q5RowSplitStorage::kHighBytesPerGroup]);
                } else {
                    store_vec(dst, static_cast<std::uint64_t>(0));
                }
            }
        }

#pragma unroll 1
        for (int row_group = tid; row_group < BM * GPB; row_group += Schedule::kThreads) {
            const int local_row   = row_group / GPB;
            const int group       = row_group - local_row * GPB;
            const int row         = row0 + local_row;
            const int scale_group = group0 + group;
            auto* dst             = &Sr[stage][row_group * SB];
            if constexpr (kFull) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row) * groups_per_row + scale_group;
                if constexpr (Schedule::kScaleLoadMode == Q5ScaleLoad::Pair32) {
                    const int aligned_group = scale_group & ~1;
                    const std::int64_t aligned_index =
                        static_cast<std::int64_t>(row) * groups_per_row + aligned_group;
                    if (aligned_group + 1 < groups_per_row) {
                        cp_async<4>(
                            dst, &scales[aligned_index * Q5RowSplitStorage::kScaleBytesPerGroup]);
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(
                                &scales[group_index * Q5RowSplitStorage::kScaleBytesPerGroup]);
                        *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                    }
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(
                            &scales[group_index * Q5RowSplitStorage::kScaleBytesPerGroup]);
                }
            } else {
                if (row < rows) {
                    const std::int64_t group_index =
                        static_cast<std::int64_t>(row) * groups_per_row + scale_group;
                    if constexpr (Schedule::kScaleLoadMode == Q5ScaleLoad::Pair32) {
                        const int aligned_group = scale_group & ~1;
                        const std::int64_t aligned_index =
                            static_cast<std::int64_t>(row) * groups_per_row + aligned_group;
                        if (aligned_group + 1 < groups_per_row) {
                            cp_async<4>(
                                dst,
                                &scales[aligned_index * Q5RowSplitStorage::kScaleBytesPerGroup]);
                        } else {
                            *reinterpret_cast<std::uint16_t*>(dst) =
                                *reinterpret_cast<const std::uint16_t*>(
                                    &scales[group_index * Q5RowSplitStorage::kScaleBytesPerGroup]);
                            *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                        }
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(
                                &scales[group_index * Q5RowSplitStorage::kScaleBytesPerGroup]);
                    }
                } else {
                    dst[0] = 0;
                    dst[1] = 0;
                    if constexpr (Schedule::kScaleLoadMode == Q5ScaleLoad::Pair32) {
                        dst[2] = 0;
                        dst[3] = 0;
                    }
                }
            }
        }
    };

    auto stage_inputs = [&](int stage, int k_tile) {
        stage_activation(stage, k_tile);
        stage_quant(stage, k_tile);
    };

    auto decode_weight = [&](int stage, int k_tile) {
        const int scale_group = (k_tile * BK) / Q5RowSplitStorage::kGroupK;
        for (int local_row = warp; local_row < BM; local_row += Schedule::kWarps) {
            auto* dst = &As[local_row * BK];
#pragma unroll
            for (int group = 0; group < GPB; ++group) {
                const int staged_group = local_row * GPB + group;
                const std::uint8_t* scale_ptr =
                    &Sr[stage][staged_group * SB + (Schedule::kScaleLoadMode == Q5ScaleLoad::Pair32
                                                        ? ((scale_group + group) & 1) *
                                                              Q5RowSplitStorage::kScaleBytesPerGroup
                                                        : 0)];
                const __nv_bfloat162 weights = Q5MmaDecodeAtom::decode_pair(
                    Cr[stage], Hr[stage], scale_ptr, staged_group, lane);
                const int shared_col =
                    q5_mma_swizzle_k64(local_row, group * Q5RowSplitStorage::kGroupK + 2 * lane);
                store_vec(&dst[shared_col], weights);
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < k_tiles) { stage_inputs(stage, stage); }
        cp_commit();
    }

    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S;
        cp_wait<S - 1>();
        __syncthreads();

        decode_weight(stage, k_tile);
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned(&a_frag)[MT][4], unsigned(&b_frag)[NT][2]) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = warp_row * WM + mi * 16 + a_row_offset;
                const int col = k_step + a_col_offset;
                ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                            smem_addr(&As[row * BK + q5_mma_swizzle_k64(row, col)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int row = warp_col * WN + ni * 8 + b_inner_row;
                const int col = k_step + b_k_offset;
                ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                            smem_addr(&Bs[stage][row * BK + q5_mma_swizzle_k64(row, col)]));
            }
        };

        if constexpr (Schedule::kFragmentPipeline == Q5FragmentPipeline::PingPong) {
            unsigned a_frag[2][MT][4];
            unsigned b_frag[2][NT][2];
            load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                const int current = ki & 1;
                const int next    = (ki + 1) & 1;
                if (ki + 1 < KSUB) { load_fragments((ki + 1) * 16, a_frag[next], b_frag[next]); }
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                                 accum[mi][ni][3], a_frag[current][mi][0], a_frag[current][mi][1],
                                 a_frag[current][mi][2], a_frag[current][mi][3],
                                 b_frag[current][ni][0], b_frag[current][ni][1]);
                    }
                }
            }
        } else {
            unsigned a_frag[MT][4];
            unsigned b_frag[NT][2];
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                load_fragments(ki * 16, a_frag, b_frag);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2],
                                 accum[mi][ni][3], a_frag[mi][0], a_frag[mi][1], a_frag[mi][2],
                                 a_frag[mi][3], b_frag[ni][0], b_frag[ni][1]);
                    }
                }
            }
        }

        __syncthreads();
        const int prefetch_tile = k_tile + S;
        if (prefetch_tile < k_tiles) { stage_inputs(stage, prefetch_tile); }
        cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int output_row0 = row0 + warp_row * WM + mi * 16 + mma_row;
        const int output_row1 = output_row0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int output_col0 = col0 + warp_col * WN + ni * 8 + 2 * mma_col;
            const int output_col1 = output_col0 + 1;
            const float* values   = accum[mi][ni];
            auto store_value      = [&](std::int64_t index, float value) {
                __nv_bfloat16 result = __float2bfloat16_rn(value);
                if constexpr (Epilogue == Q5MmaEpilogue::AddResidual) {
                    result = __float2bfloat16_rn(__bfloat162float(residual[index]) +
                                                      __bfloat162float(result));
                }
                out[index] = result;
            };
            if constexpr (kFull) {
                store_value(static_cast<std::int64_t>(output_col0) * rows + output_row0, values[0]);
                store_value(static_cast<std::int64_t>(output_col1) * rows + output_row0, values[1]);
                store_value(static_cast<std::int64_t>(output_col0) * rows + output_row1, values[2]);
                store_value(static_cast<std::int64_t>(output_col1) * rows + output_row1, values[3]);
            } else {
                if (output_row0 < rows) {
                    if (output_col0 < cols) {
                        store_value(static_cast<std::int64_t>(output_col0) * rows + output_row0,
                                    values[0]);
                    }
                    if (output_col1 < cols) {
                        store_value(static_cast<std::int64_t>(output_col1) * rows + output_row0,
                                    values[1]);
                    }
                }
                if (output_row1 < rows) {
                    if (output_col0 < cols) {
                        store_value(static_cast<std::int64_t>(output_col0) * rows + output_row1,
                                    values[2]);
                    }
                    if (output_col1 < cols) {
                        store_value(static_cast<std::int64_t>(output_col1) * rows + output_row1,
                                    values[3]);
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
