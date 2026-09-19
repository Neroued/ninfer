#pragma once

// Q4G64 RowSplit K-split MMA contraction for small column extents.
//
// Eight K-split warps cooperatively own one 16-row output tile; each warp evaluates a disjoint
// 64-wide K slice of every 512-wide K group, then the CTA reduces FP32 partials in shared
// memory. Codes are decoded to exact BF16 integers, contracted per 64-wide group on the tensor
// cores, and each group's binary16 scale is applied once to the FP32 group accumulator.
//
// Staging is a cp.async ring of complete K groups (codes, scales and each warp's activation
// tile). Q4 stores half a byte per weight, so a single-buffered CTA leaves too few bytes in
// flight to reach DRAM bandwidth; with two stages the loads for group g+1 overlap the MMA work
// on group g. Wide column tiles fall back to one stage because two would not fit the static
// shared-memory limit.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Q4KSplitStoreEpilogue {};

struct Q4KSplitIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

template <int OutputRows, int InputRows>
struct Q4LinearGeometry {
    static constexpr int kOutputRows   = OutputRows;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

struct Q4KSplitMmaSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kMinBlocksPerSm    = 6;
    static constexpr auto kCodeCache        = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
    // Static shared memory available to one CTA for cp.async staging buffers.
    static constexpr int kStaticSharedBudget = 48 * 1024;
};

// One cp.async staging buffer: the CTA's 16-row Q4 code slab for one 512-wide K group, each
// K-split warp's activation tile, and the per-row group scales.
template <int TileCols>
struct alignas(16) Q4KSplitStage {
    std::uint8_t codes[Q4KSplitMmaSchedule::kRowsPerCta][Q4KSplitMmaSchedule::kGroupK / 2];
    __nv_bfloat16 activations[Q4KSplitMmaSchedule::kKWarps]
                             [TileCols * Q4KSplitMmaSchedule::kTileKPerWarp];
    std::uint16_t scales[Q4KSplitMmaSchedule::kRowsPerCta][Q4KSplitMmaSchedule::kKWarps];
};

// Pipeline depth: two stages when the ring fits the static shared-memory budget, else one.
template <int TileCols>
__host__ __device__ constexpr int q4_ksplit_stages() {
    return 2 * static_cast<int>(sizeof(Q4KSplitStage<TileCols>)) <=
                   Q4KSplitMmaSchedule::kStaticSharedBudget
               ? 2
               : 1;
}

// Two-stage residency is shared-memory bound (four 25 KiB CTAs per 100 KiB SM for 8-column
// tiles, two 41 KiB CTAs for 16-column tiles), so the register cap follows that residency
// instead of the single-stage six-CTA target.
template <int TileCols>
__host__ __device__ constexpr int q4_ksplit_min_blocks_per_sm() {
    return q4_ksplit_stages<TileCols>() == 2 ? 4 : Q4KSplitMmaSchedule::kMinBlocksPerSm;
}

__device__ __forceinline__ int q4_ksplit_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q4KSplitBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q4_ksplit_bf16_pair(std::uint8_t packed) {
    const int q0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
    const int q1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
    Q4KSplitBf16PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q4KSplitStoreEpilogue,
          class RowPolicy = Q4KSplitIdentityRows, bool MaskedColumns = false>
__launch_bounds__(Q4KSplitMmaSchedule::kThreads, q4_ksplit_min_blocks_per_sm<TileCols>()) __global__
    void q4_ksplit_mma_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out, Epilogue epilogue = {},
                              RowPolicy row_policy = {}, int columns = ActiveCols) {
    using Schedule              = Q4KSplitMmaSchedule;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kRowsPerCta   = Schedule::kRowsPerCta;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kTileCols     = TileCols;
    constexpr int kNt           = kTileCols / 8;
    constexpr int kStages       = q4_ksplit_stages<TileCols>();
    static_assert(kTileCols >= 8 && kTileCols <= 32 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);
    static_assert(kStages == 1 || kStages == 2);

    using Stage = Q4KSplitStage<kTileCols>;
    static_assert(sizeof(Stage) % 16 == 0, "stage buffers must keep 16-byte cp.async alignment");

    union SharedStorage {
        Stage staging[kStages];
        float partial[kWarps * kNt * 32 * 4];
    };
    static_assert(sizeof(SharedStorage) <= Q4KSplitMmaSchedule::kStaticSharedBudget);

    __shared__ __align__(16) SharedStorage shared;

    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int k_split      = warp;
    const int row0         = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;
    const int live_columns = MaskedColumns ? columns : ActiveCols;

    const auto stage_x = [&](int group_k0, Stage& stage) {
        auto& x_shared               = stage.activations;
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &x_shared[warp][col * kTileK + q4_ksplit_swizzle_64(col, k8 * 8)];
            if constexpr (MaskedColumns) {
                const int source = col < live_columns ? col : 0;
                cp_async_zfill<16>(dst,
                                   &x[static_cast<std::int64_t>(source) * kHidden + group_k0 +
                                      warp * kTileK + k8 * 8],
                                   col < live_columns ? 16 : 0);
            } else {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(col) * kHidden + group_k0 +
                                     warp * kTileK + k8 * 8]);
            }
        }
    };

    const auto stage_weight = [&](int group_k0, Stage& stage) {
        auto& code_shared  = stage.codes;
        auto& scale_shared = stage.scales;
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row_policy.weight_row(row0, row);
            for (int chunk = lane; chunk < kGroupK / 32; chunk += 32) {
                cp_async<16, Schedule::kCodeCache>(
                    &code_shared[row][chunk * 16],
                    codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes + group_k0 / 2 +
                        chunk * 16);
            }
        }
        for (int row = tid; row < kRowsPerCta; row += kWarps * 32) {
            const int weight_row = row_policy.weight_row(row0, row);
            cp_async<16>(&scale_shared[row][0],
                         scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                                   group_k0 / 64) *
                                      2);
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_koff = k_split * kTileK;
    float acc[kNt][4]   = {};

    stage_weight(0, shared.staging[0]);
    stage_x(0, shared.staging[0]);
    cp_commit();

#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0  = group_index * kGroupK;
        const bool has_next = group_index + 1 < kGroups;

        // Two stages: issue group g+1 into the other buffer before consuming group g, then
        // wait for everything except that newest group. One stage: group g+1 is issued after
        // this group's trailing barrier, so wait for all outstanding copies here.
        if constexpr (kStages == 2) {
            if (has_next) {
                Stage& next = shared.staging[(group_index + 1) % kStages];
                stage_weight(group_k0 + kGroupK, next);
                stage_x(group_k0 + kGroupK, next);
                cp_commit();
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        const Stage& current     = shared.staging[group_index % kStages];
        const auto& code_shared  = current.codes;
        const auto& x_shared     = current.activations;
        const auto& scale_shared = current.scales;
        float group_acc[kNt][4]  = {};

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int byte_col = warp_koff / 2 + ks * 8 + lid;
            const unsigned af0 = q4_ksplit_bf16_pair(code_shared[gid][byte_col]);
            const unsigned af1 = q4_ksplit_bf16_pair(code_shared[gid + 8][byte_col]);
            const unsigned af2 = q4_ksplit_bf16_pair(code_shared[gid][byte_col + 4]);
            const unsigned af3 = q4_ksplit_bf16_pair(code_shared[gid + 8][byte_col + 4]);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                unsigned bf0, bf1;
                const int br = nt * 8 + b_rin;
                ldmatrix_x2(bf0, bf1,
                            smem_addr(&x_shared[k_split][br * kTileK + q4_ksplit_swizzle_64(
                                                                           br, ks * 16 + b_koff)]));
                mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                         af0, af1, af2, af3, bf0, bf1);
            }
        }

        const float top_scale = __half2float(__ushort_as_half(scale_shared[gid][k_split]));
        const float bot_scale = __half2float(__ushort_as_half(scale_shared[gid + 8][k_split]));
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
            acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
            acc[nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[nt][2]);
            acc[nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[nt][3]);
        }

        if (has_next) {
            // Every warp has finished reading this buffer before it is refilled: by the
            // single-stage issue below, or by the two-stage prefetch one iteration later.
            __syncthreads();
            if constexpr (kStages == 1) {
                stage_weight(group_k0 + kGroupK, shared.staging[0]);
                stage_x(group_k0 + kGroupK, shared.staging[0]);
                cp_commit();
            }
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            store_vec(partial + ((k_split * kNt + nt) * 32 + lane) * 4,
                      make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            const float4 partner =
                load_vec<float4>(partial + (((k_split + 1) * kNt + nt) * 32 + lane) * 4);
            acc[nt][0] += partner.x;
            acc[nt][1] += partner.y;
            acc[nt][2] += partner.z;
            acc[nt][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((k_split * kNt + nt) * 32 + lane) * 4,
                          make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]));
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            float4 sum = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value =
                    load_vec<float4>(partial + ((split * kNt + nt) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int col0 = nt * 8 + 2 * lid;
            if constexpr (std::is_same_v<Epilogue, Q4KSplitStoreEpilogue>) {
                if (col0 < live_columns) {
                    out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row0 + gid] =
                        __float2bfloat16_rn(sum.x);
                    out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row0 + gid + 8] =
                        __float2bfloat16_rn(sum.z);
                }
                if (col0 + 1 < live_columns) {
                    out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row0 + gid] =
                        __float2bfloat16_rn(sum.y);
                    out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row0 + gid +
                        8] = __float2bfloat16_rn(sum.w);
                }
            } else {
                epilogue.template store<ActiveCols>(row0 + gid, col0, sum);
            }
        }
    }
}

} // namespace ninfer::ops::detail
