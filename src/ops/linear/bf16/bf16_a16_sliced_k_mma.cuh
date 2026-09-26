#pragma once
// Warps share an output tile and partition K. FP32 fragments are reduced before the epilogue.
#include "ops/linear/bf16/bf16_mma_common.cuh"

namespace ninfer::ops::detail {
template <class Schedule, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_sliced_k_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight, Output output,
    Epilogue epilogue, int rows, int input_rows, int tokens, int token_offset) {
    const int kHidden          = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    constexpr int kMmaRows     = Schedule::kBlockRows;
    constexpr int kMmaK        = Schedule::kWarpK;
    constexpr int kKWarps      = Schedule::kKWarps;
    constexpr int kBlockTokens = Schedule::kBlockTokens;
    constexpr int kTokenMmas   = Schedule::kMmaTokens;
    constexpr int kBlockK      = Schedule::kBlockK;
    const int kGroups          = kHidden / kBlockK;
    static_assert((kMmaK % 16) == 0);

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* staging = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* partial = reinterpret_cast<float*>(shared_raw);

    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int row0         = static_cast<int>(blockIdx.x) * Schedule::kBlockRows;
    const int token0       = token_offset + static_cast<int>(blockIdx.y) * kBlockTokens;
    const auto destination = linear_output_tile<Schedule::kBlockRows>(output, row0);

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;
    const int warp_k0      = warp * kMmaK;

    constexpr int kFragments   = Schedule::kMmaRows * kTokenMmas;
    float accum[kFragments][4] = {};

    const auto stage_group = [&](int stage, int group) {
        const int group_k0  = group * kBlockK;
        auto* weight_shared = staging + stage * Schedule::kStageElements;
        auto* x_shared      = weight_shared + kMmaRows * kBlockK;

        constexpr int kWeightVectors = kMmaRows * (kBlockK / 8);
        for (int item = tid; item < kWeightVectors; item += Schedule::kThreads) {
            const int row = item / (kBlockK / 8);
            const int k8  = item - row * (kBlockK / 8);
            cp_async<16, Schedule::kWeightCache>(
                &weight_shared[row * kBlockK + bf16_mma_shared_col<Schedule>(row, k8 * 8)],
                &weight[static_cast<std::int64_t>(row0 + row) * kHidden + group_k0 + k8 * 8]);
        }

        constexpr int kActivationVectors = kBlockTokens * (kBlockK / 8);
        for (int item = tid; item < kActivationVectors; item += Schedule::kThreads) {
            const int token        = item / (kBlockK / 8);
            const int k8           = item - token * (kBlockK / 8);
            const bool valid_token = token0 + token < tokens;
            const int source_token = valid_token ? token0 + token : 0;
            cp_async_zfill<16, Schedule::kActivationCache>(
                &x_shared[token * kBlockK + bf16_mma_shared_col<Schedule>(token, k8 * 8)],
                &x[static_cast<std::int64_t>(source_token) * kHidden + group_k0 + k8 * 8],
                valid_token ? 16 : 0);
        }
    };

    const auto compute_group = [&](int stage) {
        const auto* weight_shared = staging + stage * Schedule::kStageElements;
        const auto* x_shared      = weight_shared + kMmaRows * kBlockK;
#pragma unroll
        for (int k_step = 0; k_step < kMmaK / 16; ++k_step) {
#pragma unroll
            for (int mi = 0; mi < Schedule::kMmaRows; ++mi) {
                unsigned a_frag[4];
                const int a_col = warp_k0 + k_step * 16 + a_col_offset;
                const int a_row = mi * 16 + a_row_offset;
                ldmatrix_x4(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                            smem_addr(weight_shared + a_row * kBlockK +
                                      bf16_mma_shared_col<Schedule>(a_row, a_col)));
#pragma unroll
                for (int ni = 0; ni < kTokenMmas; ++ni) {
                    unsigned b_frag[2];
                    const int b_row = ni * 8 + b_inner_row;
                    const int b_col = warp_k0 + k_step * 16 + b_k_offset;
                    ldmatrix_x2(b_frag[0], b_frag[1],
                                smem_addr(x_shared + b_row * kBlockK +
                                          bf16_mma_shared_col<Schedule>(b_row, b_col)));
                    auto& c = accum[mi * kTokenMmas + ni];
                    mma_bf16(c[0], c[1], c[2], c[3], a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                             b_frag[0], b_frag[1]);
                }
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kGroups) {
            stage_group(stage, stage);
            cp_commit();
        }
    }

#pragma unroll 1
    for (int group = 0; group < kGroups; ++group) {
        if (group + Schedule::kStages <= kGroups) {
            cp_wait<Schedule::kStages - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const int stage = group % Schedule::kStages;
        compute_group(stage);
        __syncthreads();

        const int next = group + Schedule::kStages;
        if (next < kGroups) {
            stage_group(stage, next);
            cp_commit();
        }
    }

    if ((warp & 1) != 0) {
#pragma unroll
        for (int fragment = 0; fragment < kFragments; ++fragment) {
            const std::int64_t base =
                (static_cast<std::int64_t>(warp) * kFragments + fragment) * 32 * 4;
            store_vec(partial + base + static_cast<std::int64_t>(lane) * 4,
                      make_float4(accum[fragment][0], accum[fragment][1], accum[fragment][2],
                                  accum[fragment][3]));
        }
    }
    __syncthreads();

    if ((warp & 1) == 0) {
#pragma unroll
        for (int fragment = 0; fragment < kFragments; ++fragment) {
            const std::int64_t partner_base =
                (static_cast<std::int64_t>(warp + 1) * kFragments + fragment) * 32 * 4;
            const float4 partner =
                load_vec<float4>(partial + partner_base + static_cast<std::int64_t>(lane) * 4);
            accum[fragment][0] += partner.x;
            accum[fragment][1] += partner.y;
            accum[fragment][2] += partner.z;
            accum[fragment][3] += partner.w;
            if (warp != 0) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(warp) * kFragments + fragment) * 32 * 4;
                store_vec(partial + base + static_cast<std::int64_t>(lane) * 4,
                          make_float4(accum[fragment][0], accum[fragment][1], accum[fragment][2],
                                      accum[fragment][3]));
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int fragment = 0; fragment < kFragments; ++fragment) {
            float4 sum = make_float4(accum[fragment][0], accum[fragment][1], accum[fragment][2],
                                     accum[fragment][3]);
#pragma unroll
            for (int split = 2; split < kKWarps; split += 2) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(split) * kFragments + fragment) * 32 * 4;
                const float4 value =
                    load_vec<float4>(partial + base + static_cast<std::int64_t>(lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int mi = fragment / kTokenMmas, ni = fragment % kTokenMmas;
            const int token = token0 + ni * 8 + 2 * lid;
            bf16_finish_fragment<false>(destination, epilogue, row0 + mi * 16 + gid, token, sum,
                                        rows, tokens);
        }
    }
}
} // namespace ninfer::ops::detail
