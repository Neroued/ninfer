#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear/q6/q6_schedule.cuh"

namespace ninfer::ops::detail {

// Q6 integers are exactly representable in BF16. Scaling each 64-element
// contraction in FP32 avoids a scaled-weight BF16 materialization.
__device__ __forceinline__ unsigned q6_sliced_k_integer_pair(std::uint8_t code, std::uint8_t high,
                                                             int shift) {
    const int q0 = ((int(code & 15) | (((high >> shift) & 3) << 4)) ^ 32) - 32;
    const int q1 = ((int(code >> 4) | (((high >> (shift + 2)) & 3) << 4)) ^ 32) - 32;

    union {
        __nv_bfloat162 value;
        unsigned bits;
    } pair;

    pair.value = __floats2bfloat162_rn(float(q0), float(q1));
    return pair.bits;
}

__device__ __forceinline__ int q6_sliced_k_swizzle(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <class Schedule, bool Full, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q6_a16_sliced_k_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, Output output,
    Epilogue epilogue, int rows, int k, int tokens, int padded_k, int token_begin) {
    constexpr int R  = Schedule::kBlockRows;
    constexpr int T  = Schedule::kBlockTokens;
    constexpr int W  = Schedule::kKWarps;
    constexpr int S  = Schedule::kStages;
    constexpr int MT = Schedule::kMmaRows;
    constexpr int NT = Schedule::kMmaTokens;

    union SharedStorage {
        struct {
            alignas(16) std::uint8_t codes[S][R][W][32];
            alignas(16) std::uint8_t high[S][R][W][16];
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
    const int row_begin  = static_cast<int>(blockIdx.x) * R;
    const int token0     = static_cast<int>(blockIdx.y) * T;
    const int groups     = k / 64;
    const int row_groups = padded_k / 64;
    const int tiles      = (groups + W - 1) / W;

    const auto stage_inputs = [&](int slot, int tile) {
        const int first_group = tile * W;
        for (int item = tid; item < R * W * 2; item += Schedule::kThreads) {
            const int half   = item & 1;
            const int group  = (item >> 1) % W;
            const int row    = item / (W * 2);
            const bool valid = Full || (row_begin + row < rows && first_group + group < groups);
            const std::int64_t index =
                valid ? std::int64_t(row_begin + row) * row_groups + first_group + group : 0;
            cp_async_zfill<16, Schedule::kWeightCache>(
                &shared.staging.codes[slot][row][group][half * 16], codes + index * 32 + half * 16,
                valid ? 16 : 0);
        }
        for (int item = tid; item < R * W; item += Schedule::kThreads) {
            const int group  = item % W;
            const int row    = item / W;
            const bool valid = Full || (row_begin + row < rows && first_group + group < groups);
            const std::int64_t index =
                valid ? std::int64_t(row_begin + row) * row_groups + first_group + group : 0;
            cp_async_zfill<16, Schedule::kWeightCache>(&shared.staging.high[slot][row][group][0],
                                                       high + index * 16, valid ? 16 : 0);
        }
        if constexpr (Full) {
            for (int row = tid; row < R; row += Schedule::kThreads) {
                const std::int64_t index = std::int64_t(row_begin + row) * row_groups + first_group;
                cp_async<W * 2>(&shared.staging.scales[slot][row][0], scales + index * 2);
            }
        } else {
            // Partial K tiles only guarantee scale-pair alignment of each stored row.
            for (int item = tid; item < R * (W / 2); item += Schedule::kThreads) {
                const int row    = item / (W / 2);
                const int pair   = (item % (W / 2)) * 2;
                const bool valid = row_begin + row < rows && first_group + pair < groups;
                const std::int64_t index =
                    valid ? std::int64_t(row_begin + row) * row_groups + first_group + pair : 0;
                cp_async_zfill<4>(&shared.staging.scales[slot][row][pair], scales + index * 2,
                                  valid ? 4 : 0);
            }
        }
        for (int item = lane; item < T * 8; item += 32) {
            const int token  = item / 8;
            const int k8     = (item % 8) * 8;
            const bool valid = Full || (token0 + token < tokens && first_group + warp < groups);
            const std::int64_t index =
                valid ? std::int64_t(token0 + token) * k + (first_group + warp) * 64 + k8 : 0;
            cp_async_zfill<16, Schedule::kActivationCache>(
                &shared.staging.x[slot][warp][token][q6_sliced_k_swizzle(token, k8)], x + index,
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
#pragma unroll 1
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
                    return q6_sliced_k_integer_pair(shared.staging.codes[slot][row][warp][byte],
                                                    shared.staging.high[slot][row][warp][byte / 2],
                                                    (byte & 1) * 4);
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
                        smem_addr(&shared.staging.x[slot][warp][br][q6_sliced_k_swizzle(br, bk)]));
                    mma_bf16(group_acc[mi][ni][0], group_acc[mi][ni][1], group_acc[mi][ni][2],
                             group_acc[mi][ni][3], a0, a1, a2, a3, b0, b1);
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const float top =
                __half2float(__ushort_as_half(shared.staging.scales[slot][mi * 16 + gid][warp]));
            const float bottom = __half2float(
                __ushort_as_half(shared.staging.scales[slot][mi * 16 + gid + 8][warp]));
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                acc[mi][ni][0] = fmaf(group_acc[mi][ni][0], top, acc[mi][ni][0]);
                acc[mi][ni][1] = fmaf(group_acc[mi][ni][1], top, acc[mi][ni][1]);
                acc[mi][ni][2] = fmaf(group_acc[mi][ni][2], bottom, acc[mi][ni][2]);
                acc[mi][ni][3] = fmaf(group_acc[mi][ni][3], bottom, acc[mi][ni][3]);
            }
        }
        __syncthreads();
        if (tile + S < tiles)
            stage_inputs(slot, tile + S);
        else
            cp_commit();
    }
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            store_vec(shared.partial[warp][mi][ni][lane],
                      make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]));
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                float4 sum = load_vec<float4>(shared.partial[0][mi][ni][lane]);
#pragma unroll
                for (int part = 1; part < W; ++part) {
                    const float4 value = load_vec<float4>(shared.partial[part][mi][ni][lane]);
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int r0     = row_begin + mi * 16 + gid;
                const int t0     = token0 + ni * 8 + 2 * lid;
                const auto store = [&](int row, int token, float value) {
                    if (Full || (row < rows && token < tokens)) {
                        const int global_token = token_begin + token;
                        output.store(row, global_token, epilogue.apply(row, global_token, value));
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

} // namespace ninfer::ops::detail
