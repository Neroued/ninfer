#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear/q5/q5_schedule.cuh"

namespace ninfer::ops::detail {

template <int BlockRows>
struct Q5IdentityRows {
    static constexpr int kOutputRowsPerCta = BlockRows;

    __host__ __device__ int output_rows(int rows) const { return rows; }

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

// Q5 integers are exactly representable in BF16. Scaling each 64-element
// contraction in FP32 avoids a scaled-weight BF16 materialization.
__device__ __forceinline__ unsigned q5_sliced_k_integer_pair(std::uint8_t code, std::uint8_t high,
                                                             int pair) {
    const int shift = pair * 2;
    const int q0    = ((int(code & 15) | (((high >> shift) & 1) << 4)) ^ 16) - 16;
    const int q1    = ((int(code >> 4) | (((high >> (shift + 1)) & 1) << 4)) ^ 16) - 16;

    union {
        __nv_bfloat162 value;
        unsigned bits;
    } result;

    result.value = __floats2bfloat162_rn(float(q0), float(q1));
    return result.bits;
}

__device__ __forceinline__ int q5_sliced_k_swizzle(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <class Schedule, bool Full, bool FullTokens, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q5_a16_sliced_k_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, Output output,
    Epilogue epilogue, int rows, int k, int tokens, int padded_k, int token_begin,
    RowPolicy row_policy) {
    constexpr int R  = Schedule::kBlockRows;
    constexpr int T  = Schedule::kBlockTokens;
    constexpr int W  = Schedule::kKWarps;
    constexpr int S  = Schedule::kStages;
    constexpr int MT = Schedule::kMmaRows;
    constexpr int NT = Schedule::kMmaTokens;

    union SharedStorage {
        struct {
            alignas(16) std::uint8_t codes[S][R][W][32];
            alignas(16) std::uint8_t high[S][R][W][8];
            alignas(16) std::uint16_t scales[S][R][W];
            alignas(16) __nv_bfloat16 x[S][W][T][64];
        } staging;

        float partial[W][MT][NT][32][4];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    __shared__ SharedStorage shared;
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int row_begin  = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;
    const int token0     = static_cast<int>(blockIdx.y) * Schedule::kTokenCapacity;
    const int logical_k  = Schedule::kStaticK ? Schedule::kStaticK : k;
    const int groups     = (logical_k + 63) / 64;
    const int row_groups = (Schedule::kStaticK ? Schedule::kStaticK : padded_k) / 64;
    const int tiles      = (groups + W - 1) / W;

    const auto stage_inputs = [&](int slot, int tile) {
        const int first_group = tile * W;
        for (int item = tid; item < R * W * 2; item += Schedule::kThreads) {
            const int half   = item & 1;
            const int group  = (item >> 1) % W;
            const int row    = item / (W * 2);
            const bool valid = Full || (row_policy.weight_row(row_begin, row) < rows &&
                                        first_group + group < groups);
            const std::int64_t index =
                valid ? std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups +
                            first_group + group
                      : 0;
            cp_async_zfill<16, Schedule::kWeightCache>(
                &shared.staging.codes[slot][row][group][half * 16], codes + index * 32 + half * 16,
                valid ? 16 : 0);
        }
        for (int item = tid; item < R * (W / 2); item += Schedule::kThreads) {
            const int row    = item / (W / 2);
            const int pair   = (item % (W / 2)) * 2;
            const bool valid = Full || (row_policy.weight_row(row_begin, row) < rows &&
                                        first_group + pair < groups);
            const auto index =
                valid ? std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups +
                            first_group + pair
                      : 0;
            cp_async_zfill<16, Schedule::kWeightCache>(&shared.staging.high[slot][row][pair][0],
                                                       high + index * 8, valid ? 16 : 0);
        }
        if constexpr (Full) {
            for (int row = tid; row < R; row += Schedule::kThreads) {
                const std::int64_t index =
                    std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups + first_group;
                cp_async<W * 2>(&shared.staging.scales[slot][row][0], scales + index * 2);
            }
        } else {
            // Partial K tiles only guarantee scale-pair alignment of each stored row.
            for (int item = tid; item < R * (W / 2); item += Schedule::kThreads) {
                const int row  = item / (W / 2);
                const int pair = (item % (W / 2)) * 2;
                const bool valid =
                    row_policy.weight_row(row_begin, row) < rows && first_group + pair < groups;
                const std::int64_t index =
                    valid ? std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups +
                                first_group + pair
                          : 0;
                cp_async_zfill<4>(&shared.staging.scales[slot][row][pair], scales + index * 2,
                                  valid ? 4 : 0);
            }
        }
        for (int item = lane; item < Schedule::kTokenCapacity * 8; item += 32) {
            const int token  = item / 8;
            const int k8     = (item % 8) * 8;
            const bool valid = (FullTokens || token0 + token < tokens) &&
                               (Full || (first_group + warp) * 64 + k8 + 8 <= logical_k);
            const std::int64_t index =
                valid ? std::int64_t(token0 + token) * logical_k + (first_group + warp) * 64 + k8
                      : 0;
            cp_async_zfill<16, Schedule::kActivationCache>(
                &shared.staging.x[slot][warp][token][q5_sliced_k_swizzle(token, k8)], x + index,
                valid ? 16 : 0);
        }
        cp_commit();
    };

    float acc[MT][NT][4] = {};
#pragma unroll
    for (int slot = 0; slot < S; ++slot) {
        if (slot < tiles)
            stage_inputs(slot, slot);
        else
            cp_commit();
    }
    constexpr int kUnroll = Schedule::kStaticK ? (Schedule::kStaticK / 64 + W - 1) / W : 1;
#pragma unroll kUnroll
    for (int tile = 0; tile < tiles; ++tile) {
        const int slot = tile % S;
        cp_wait<S - 1>();
        __syncthreads();
        float group_acc[MT][NT][4] = {};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int r0      = mi * 16 + gid;
                const int byte0   = ks * 8 + lid;
                const auto decode = [&](int row, int byte) {
                    return q5_sliced_k_integer_pair(shared.staging.codes[slot][row][warp][byte],
                                                    shared.staging.high[slot][row][warp][byte / 4],
                                                    byte % 4);
                };
                const unsigned a0 = decode(r0, byte0);
                const unsigned a1 = decode(r0 + 8, byte0);
                const unsigned a2 = decode(r0, byte0 + 4);
                const unsigned a3 = decode(r0 + 8, byte0 + 4);
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const int br = ni * 8 + (lane & 7);
                    const int bk = ks * 16 + (((lane >> 3) & 1) * 8);
                    unsigned b0, b1;
                    ldmatrix_x2(
                        b0, b1,
                        smem_addr(&shared.staging.x[slot][warp][br][q5_sliced_k_swizzle(br, bk)]));
                    mma_bf16(group_acc[mi][ni][0], group_acc[mi][ni][1], group_acc[mi][ni][2],
                             group_acc[mi][ni][3], a0, a1, a2, a3, b0, b1);
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const bool valid_group = Full || tile * W + warp < groups;
            const float top        = !valid_group ? 0.0f
                                                  : __half2float(__ushort_as_half(
                                                 shared.staging.scales[slot][mi * 16 + gid][warp]));
            const float bottom     = !valid_group
                                         ? 0.0f
                                         : __half2float(__ushort_as_half(
                                           shared.staging.scales[slot][mi * 16 + gid + 8][warp]));
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                acc[mi][ni][0] = fmaf(group_acc[mi][ni][0], top, acc[mi][ni][0]);
                acc[mi][ni][1] = fmaf(group_acc[mi][ni][1], top, acc[mi][ni][1]);
                acc[mi][ni][2] = fmaf(group_acc[mi][ni][2], bottom, acc[mi][ni][2]);
                acc[mi][ni][3] = fmaf(group_acc[mi][ni][3], bottom, acc[mi][ni][3]);
            }
        }
        if (tile + S < tiles) {
            // Only a reused stage needs a barrier here. The final reduction
            // barrier covers the drained stages without an extra CTA stall.
            __syncthreads();
            stage_inputs(slot, tile + S);
        } else if constexpr (S > 1) {
            cp_commit();
        }
    }
    cp_wait<0>();
    __syncthreads();
    constexpr bool kPairwise = Schedule::kReduction == Q5SlicedKReduction::Pairwise;
    if (!kPairwise || (warp & 1) != 0) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                store_vec(
                    shared.partial[warp][mi][ni][lane],
                    make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]));
            }
        }
    }
    __syncthreads();
    if constexpr (kPairwise) {
        // Pair neighboring K warps before the final owner loads shared partials.
        // This trades one CTA barrier for half as many loads by the owner warp.
        if ((warp & 1) == 0) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const float4 partner = load_vec<float4>(shared.partial[warp + 1][mi][ni][lane]);
                    acc[mi][ni][0] += partner.x;
                    acc[mi][ni][1] += partner.y;
                    acc[mi][ni][2] += partner.z;
                    acc[mi][ni][3] += partner.w;
                    if (warp != 0)
                        store_vec(shared.partial[warp][mi][ni][lane],
                                  make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2],
                                              acc[mi][ni][3]));
                }
            }
        }
        __syncthreads();
    }
    if (warp == 0) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                float4 sum;
                if constexpr (kPairwise)
                    sum =
                        make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
                else
                    sum = load_vec<float4>(shared.partial[0][mi][ni][lane]);
#pragma unroll
                for (int part = kPairwise ? 2 : 1; part < W; part += kPairwise ? 2 : 1) {
                    const float4 value = load_vec<float4>(shared.partial[part][mi][ni][lane]);
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int r0        = row_begin + mi * 16 + gid;
                const int t0        = token0 + ni * 8 + 2 * lid;
                const int token_end = token_begin + min(tokens, token0 + Schedule::kTokenCapacity);
                if constexpr (requires {
                                  epilogue.store_fragment(output, r0, token_begin + t0, sum, rows,
                                                          token_end);
                              }) {
                    // Fully reduced m16n8 fragment: x/y belong to r0, z/w to
                    // r0+8. The owning Op receives absolute bounds for its store.
                    epilogue.store_fragment(output, r0, token_begin + t0, sum, rows, token_end);
                } else {
                    const auto store = [&](int row, int token, float value) {
                        if (token - token0 < Schedule::kTokenCapacity && (Full || row < rows) &&
                            (FullTokens || token < tokens)) {
                            const int global_token = token_begin + token;
                            output.store(row, global_token,
                                         epilogue.apply(row, global_token, value));
                        }
                    };
                    store(r0, t0, sum.x);
                    store(r0, t0 + 1, sum.y);
                    store(r0 + 8, t0, sum.z);
                    store(r0 + 8, t0 + 1, sum.w);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
