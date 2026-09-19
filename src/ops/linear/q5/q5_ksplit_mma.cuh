#pragma once

// Q5G64 RowSplit K-split MMA contraction for small column extents: out[N,T] = W[N,K] . x[K,T].
//
// Eight K-split warps cooperatively own one 16-row output tile; each warp evaluates a disjoint
// 64-wide K slice (exactly one quantization group) of every 512-wide K group, then the CTA
// reduces FP32 partials in shared memory. Q5 codes (nibble plane plus the fifth-bit plane) are
// decoded to exact BF16 integers, contracted per group on the tensor cores, and each group's
// binary16 scale is applied once to the FP32 group accumulator.
//
// Staging is a two-deep cp.async ring of complete K groups (codes, high bits, scales and each
// warp's activation tile) so the loads of group g+1 overlap the MMA work on group g. Compared
// with the warp-per-row SIMT route, the activation tile is read once per 16 rows instead of
// once per row and the per-weight FMA work moves to the tensor cores, which is what the T>1
// verification extents need to stay on the DRAM roofline.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

struct Q5KSplitMmaSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = Q5RowSplitStorage::kGroupK;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
    static constexpr int kCodeBytesPerGroup = Q5RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kHighBytesPerGroup = Q5RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kStaticSharedBudget = 48 * 1024;
};

// One cp.async stage: the CTA's 16-row code and high-bit slabs for one 512-wide K group, the
// binary16 scale of each row's eight groups, and each K-split warp's activation tile.
template <int TileCols>
struct alignas(16) Q5KSplitStage {
    std::uint8_t codes[Q5KSplitMmaSchedule::kRowsPerCta]
                      [Q5KSplitMmaSchedule::kKWarps * Q5KSplitMmaSchedule::kCodeBytesPerGroup];
    std::uint8_t high[Q5KSplitMmaSchedule::kRowsPerCta]
                     [Q5KSplitMmaSchedule::kKWarps * Q5KSplitMmaSchedule::kHighBytesPerGroup];
    std::uint16_t scales[Q5KSplitMmaSchedule::kRowsPerCta][Q5KSplitMmaSchedule::kKWarps];
    __nv_bfloat16 activations[Q5KSplitMmaSchedule::kKWarps]
                             [TileCols * Q5KSplitMmaSchedule::kTileKPerWarp];
};

// Ring depth: the deepest ring that fits the static shared-memory budget. The N=5120 shapes
// launch only 320 CTAs (about two per SM), so each group costs one memory latency unless
// several groups are in flight per CTA; extra stages cost nothing there.
template <int TileCols>
__host__ __device__ constexpr int q5_ksplit_stages() {
    constexpr int kStage  = static_cast<int>(sizeof(Q5KSplitStage<TileCols>));
    constexpr int kBudget = Q5KSplitMmaSchedule::kStaticSharedBudget;
    return 3 * kStage <= kBudget ? 3 : (2 * kStage <= kBudget ? 2 : 1);
}

// Residency is shared-memory bound; the register cap follows the CTAs that actually fit.
template <int TileCols>
__host__ __device__ constexpr int q5_ksplit_min_blocks_per_sm() {
    constexpr int kFit = (100 * 1024) / (q5_ksplit_stages<TileCols>() *
                                         static_cast<int>(sizeof(Q5KSplitStage<TileCols>)));
    return kFit < 1 ? 1 : (kFit > 4 ? 4 : kFit);
}

__device__ __forceinline__ int q5_ksplit_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// Fragment reads touch the same column of rows gid..gid+7, whose 256-byte code rows and
// 64-byte high-bit rows would otherwise map to one bank. XOR-swizzling the 16-byte chunk index
// by a row-derived key spreads the eight rows over eight distinct banks.
__device__ __forceinline__ int q5_ksplit_code_offset(int row, int byte) {
    return (((byte >> 4) ^ (row & 7)) << 4) | (byte & 15);
}

__device__ __forceinline__ int q5_ksplit_high_offset(int row, int byte) {
    return (((byte >> 4) ^ ((row >> 1) & 3)) << 4) | (byte & 15);
}

union Q5KSplitBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

// One packed code byte holds K pair (2b, 2b+1) in its low/high nibbles; the matching high-bit
// byte holds their fifth bits at 2*(b&3) and 2*(b&3)+1. Signed code: (u ^ 16) - 16.
__device__ __forceinline__ unsigned q5_ksplit_bf16_pair(std::uint8_t packed, std::uint8_t high,
                                                        int shift) {
    const int q0 = ((static_cast<int>(packed & 0x0fu) | (((high >> shift) & 1) << 4)) ^ 0x10) - 0x10;
    const int q1 =
        ((static_cast<int>(packed >> 4) | (((high >> (shift + 1)) & 1) << 4)) ^ 0x10) - 0x10;
    Q5KSplitBf16PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

// AddResidual selects the LinearAdd form: out already holds the residual and receives
// residual + projection; otherwise out receives the projection.
template <int OutputRows, int InputRows, int TileCols, int ActiveCols, bool AddResidual = false>
__launch_bounds__(Q5KSplitMmaSchedule::kThreads, q5_ksplit_min_blocks_per_sm<TileCols>()) __global__
    void q5_ksplit_mma_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ high,
                              const std::uint8_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out, int columns) {
    using Schedule             = Q5KSplitMmaSchedule;
    constexpr int kHidden      = InputRows;
    constexpr int kTileK       = Schedule::kTileKPerWarp;
    constexpr int kWarps       = Schedule::kKWarps;
    constexpr int kRowsPerCta  = Schedule::kRowsPerCta;
    constexpr int kGroupK      = Schedule::kGroupK;
    constexpr int kGroups      = kHidden / kGroupK;
    constexpr int kGroupsPerRow = kHidden / Q5RowSplitStorage::kGroupK;
    constexpr int kCodeRowBytes = kGroupsPerRow * Schedule::kCodeBytesPerGroup;
    constexpr int kHighRowBytes = kGroupsPerRow * Schedule::kHighBytesPerGroup;
    constexpr int kScaleRowBytes = kGroupsPerRow * 2;
    constexpr int kTileCols    = TileCols;
    constexpr int kNt          = kTileCols / 8;
    constexpr int kStages      = q5_ksplit_stages<TileCols>();
    constexpr int kPrefetch    = kStages - 1;
    static_assert(kTileCols >= 8 && kTileCols <= 32 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0 && kGroups >= 1);
    static_assert(OutputRows % kRowsPerCta == 0);
    // Every row's code, high-bit and scale planes must start 16-byte aligned for cp.async.
    static_assert(kCodeRowBytes % 16 == 0 && kHighRowBytes % 16 == 0 && kScaleRowBytes % 16 == 0);
    static_assert(kStages >= 1 && kStages <= 4);

    using Stage = Q5KSplitStage<kTileCols>;
    static_assert(sizeof(Stage) % 16 == 0, "stage buffers must keep 16-byte cp.async alignment");

    union SharedStorage {
        Stage staging[kStages];
        float partial[kWarps * kNt * 32 * 4];
    };
    static_assert(sizeof(SharedStorage) <= Schedule::kStaticSharedBudget);

    __shared__ __align__(16) SharedStorage shared;

    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int k_split      = warp;
    const int row0         = static_cast<int>(blockIdx.x) * kRowsPerCta;
    // blockIdx.y selects a kTileCols-wide column tile; a single-tile launch has offset 0.
    const int column_offset = static_cast<int>(blockIdx.y) * kTileCols;
    const int live_columns  = min(kTileCols, columns - column_offset);
    x += static_cast<std::int64_t>(column_offset) * kHidden;
    out += static_cast<std::int64_t>(column_offset) * OutputRows;

    const auto stage_x = [&](int group_k0, Stage& stage) {
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst = &stage.activations[warp][col * kTileK + q5_ksplit_swizzle_64(col, k8 * 8)];
            const int source = col < live_columns ? col : 0;
            cp_async_zfill<16>(dst,
                               &x[static_cast<std::int64_t>(source) * kHidden + group_k0 +
                                  warp * kTileK + k8 * 8],
                               col < live_columns ? 16 : 0);
        }
    };

    // Per row and 512-wide group: 256 code bytes (16 chunks), 64 high-bit bytes (4 chunks) and
    // 16 scale bytes (1 chunk); 21 lanes of the loader warp carry one row.
    const auto stage_weight = [&](int group_k0, Stage& stage) {
        constexpr int kCodeChunks = kWarps * Schedule::kCodeBytesPerGroup / 16;
        constexpr int kHighChunks = kWarps * Schedule::kHighBytesPerGroup / 16;
        static_assert(kCodeChunks + kHighChunks + 1 <= 32);
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row0 + row;
            if (lane < kCodeChunks) {
                cp_async<16, Cache::cg>(&stage.codes[row][q5_ksplit_code_offset(row, lane * 16)],
                                        codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes +
                                            group_k0 / 2 + lane * 16);
            } else if (lane < kCodeChunks + kHighChunks) {
                const int chunk = lane - kCodeChunks;
                cp_async<16, Cache::cg>(&stage.high[row][q5_ksplit_high_offset(row, chunk * 16)],
                                        high + static_cast<std::int64_t>(weight_row) * kHighRowBytes +
                                            group_k0 / 8 + chunk * 16);
            } else if (lane == kCodeChunks + kHighChunks) {
                cp_async<16>(&stage.scales[row][0],
                             scales + static_cast<std::int64_t>(weight_row) * kScaleRowBytes +
                                 (group_k0 / Q5RowSplitStorage::kGroupK) * 2);
            }
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int code_off  = k_split * Schedule::kCodeBytesPerGroup;
    const int high_off  = k_split * Schedule::kHighBytesPerGroup;
    const int hshift    = 2 * lid;
    float acc[kNt][4]   = {};

    if constexpr (kStages == 1) {
        stage_weight(0, shared.staging[0]);
        stage_x(0, shared.staging[0]);
        cp_commit();
    } else {
#pragma unroll
        for (int s = 0; s < kPrefetch; ++s) {
            if (s < kGroups) {
                stage_weight(s * kGroupK, shared.staging[s]);
                stage_x(s * kGroupK, shared.staging[s]);
            }
            cp_commit();
        }
    }

    // Full unrolling of a long K loop (34 groups at K=17408) thrashes the instruction cache;
    // a partial unroll that is a multiple of every ring depth keeps buffer indexing cheap.
    constexpr int kGroupUnroll = kGroups <= 12 ? kGroups : 6;
#pragma unroll kGroupUnroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0  = group_index * kGroupK;
        const bool has_next = group_index + 1 < kGroups;

        // Ring of kStages: keep kPrefetch groups in flight by refilling the buffer consumed in
        // the previous iteration (its trailing barrier retired every reader); one group is
        // committed every iteration so wait_group<kPrefetch> means "group g has landed". With a
        // single stage, group g+1 is issued after this group's trailing barrier instead.
        if constexpr (kStages >= 2) {
            const int fetch = group_index + kPrefetch;
            if (fetch < kGroups) {
                Stage& next = shared.staging[fetch % kStages];
                stage_weight(fetch * kGroupK, next);
                stage_x(fetch * kGroupK, next);
            }
            cp_commit();
            cp_wait<kPrefetch>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        const Stage& current    = shared.staging[group_index % kStages];
        float group_acc[kNt][4] = {};

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            // Byte b = ks*8 + lid (K pair 2b within the warp's group) for af0/af1 and b + 4 for
            // af2/af3; their fifth bits live in high bytes b>>2 = 2*ks and 2*ks + 1.
            const int byte_col = code_off + ks * 8 + lid;
            const int high_col = high_off + ks * 2;
            const auto code_at = [&](int row, int byte) {
                return current.codes[row][q5_ksplit_code_offset(row, byte)];
            };
            const auto high_at = [&](int row, int byte) {
                return current.high[row][q5_ksplit_high_offset(row, byte)];
            };
            const unsigned af0 = q5_ksplit_bf16_pair(code_at(gid, byte_col),
                                                     high_at(gid, high_col), hshift);
            const unsigned af1 = q5_ksplit_bf16_pair(code_at(gid + 8, byte_col),
                                                     high_at(gid + 8, high_col), hshift);
            const unsigned af2 = q5_ksplit_bf16_pair(code_at(gid, byte_col + 4),
                                                     high_at(gid, high_col + 1), hshift);
            const unsigned af3 = q5_ksplit_bf16_pair(code_at(gid + 8, byte_col + 4),
                                                     high_at(gid + 8, high_col + 1), hshift);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                unsigned bf0, bf1;
                const int br = nt * 8 + b_rin;
                ldmatrix_x2(bf0, bf1,
                            smem_addr(&current.activations[k_split]
                                                          [br * kTileK + q5_ksplit_swizzle_64(
                                                                             br, ks * 16 + b_koff)]));
                mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                         af0, af1, af2, af3, bf0, bf1);
            }
        }

        const float top_scale = __half2float(__ushort_as_half(current.scales[gid][k_split]));
        const float bot_scale = __half2float(__ushort_as_half(current.scales[gid + 8][k_split]));
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
            acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
            acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
            acc[nt][2] = fmaf(group_acc[nt][2], bot_scale, acc[nt][2]);
            acc[nt][3] = fmaf(group_acc[nt][3], bot_scale, acc[nt][3]);
        }

        if (has_next) {
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
            const int col0   = nt * 8 + 2 * lid;
            const auto store = [&](int col, int row, float value) {
                __nv_bfloat16* destination = out + static_cast<std::int64_t>(col) * OutputRows + row;
                if constexpr (AddResidual) { value += __bfloat162float(*destination); }
                *destination = __float2bfloat16_rn(value);
            };
            if (col0 < live_columns) {
                store(col0, row0 + gid, sum.x);
                store(col0, row0 + gid + 8, sum.z);
            }
            if (col0 + 1 < live_columns) {
                store(col0 + 1, row0 + gid, sum.y);
                store(col0 + 1, row0 + gid + 8, sum.w);
            }
        }
    }
}

// Exact-shape host launch. Capacity is the compile-time column capacity of one column tile;
// the live column count is a runtime argument, so one instance serves every T in
// (Capacity - 8, Capacity]. MaxColumns > Capacity launches ceil(T / Capacity) column tiles
// along blockIdx.y, streaming the weights once per tile; that stays far ahead of the wide GEMM
// tiles, which compute padded columns, until T approaches a few tiles.
//
// Capacity is compile time and sizes the accumulator fragments and the staged item count, so a
// single wide instance makes small T pay for columns it never fills: one Capacity=32 instance
// serving all of T<=96 costs the same at T=2 as at T=17. Callers should ladder Capacity to T
// (4/8/16, then 32 with column tiling) the way the shape dispatchers and the linear_add route do;
// measured on an RTX 5090, the ladder is 20-40% faster than a lone Capacity=32 instance across
// T=2..16 on every Q5 shape.
template <int OutputRows, int InputRows, int Capacity, int MaxColumns = Capacity,
          bool AddResidual = false>
void launch_q5_ksplit_mma(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream) {
    constexpr int kTileCols = (Capacity + 7) / 8 * 8;
    static_assert(MaxColumns >= Capacity);
    static_assert(MaxColumns == Capacity || Capacity == kTileCols,
                  "column tiling requires full-width tiles");
    if (weight.n != OutputRows || weight.padded_shape[1] != InputRows) {
        throw std::invalid_argument("q5 K-split MMA: weight geometry differs from instance");
    }
    const std::int32_t columns = x.ne[1];
    if (columns < 1 || columns > MaxColumns) {
        throw std::invalid_argument("q5 K-split MMA: column count exceeds instance capacity");
    }
    const dim3 grid(OutputRows / Q5KSplitMmaSchedule::kRowsPerCta,
                    static_cast<unsigned>((columns + kTileCols - 1) / kTileCols));
    q5_ksplit_mma_kernel<OutputRows, InputRows, kTileCols, Capacity, AddResidual>
        <<<grid, Q5KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
