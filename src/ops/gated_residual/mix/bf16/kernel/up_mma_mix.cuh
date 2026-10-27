#pragma once

#include "ops/gated_residual/mix/bf16/kernel/output.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_common.cuh"

namespace ninfer::ops::detail::hc::mix::bf16 {
template <int Features, int Tokens>
using FoldedUpSchedule =
    Bf16A16MmaSchedule<kStreams * Features, Tokens, 64, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

// Native rows s*H+d are gathered into four shared tiles. One CTA owns the complete four-stream
// projection/mixing result. No weight repacking, global logits, contributions, or atomics.
template <int Features, int Tokens>
__launch_bounds__(FoldedUpSchedule<Features, Tokens>::kThreads) __global__
    void up_mma_mix_kernel(Operands o, const __nv_bfloat16* activation, int token_offset) {
    using S          = FoldedUpSchedule<Features, Tokens>;
    constexpr int BK = S::kBlockK, Rows = kStreams * Features, Stages = S::kStages;
    static_assert(kWidth % Features == 0 && S::kMmaRows == 1 && S::kMmaTokens == 1);
    const int feature_tiles = kWidth / Features;
    const int d0            = (int(blockIdx.x) % feature_tiles) * Features;
    const int token0        = token_offset + (int(blockIdx.x) / feature_tiles) * Tokens;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    extern __shared__ __align__(16) unsigned char storage[];
    auto* a                                    = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b                                    = a + Stages * Rows * BK;
    float accum[S::kMmaRows][S::kMmaTokens][4] = {};
    const auto stage_inputs                    = [&](int slot, int group) {
        auto* as = a + slot * Rows * BK;
        auto* bs = b + slot * Tokens * BK;
        for (int i = tid; i < Rows * (BK / 8); i += S::kThreads) {
            const int row = i / (BK / 8), k8 = i % (BK / 8);
            const int native_row = (row / Features) * kWidth + d0 + row % Features;
            common::cp_async<16, S::kWeightCache>(
                as + row * BK + bf16_mma_shared_col<S>(row, k8 * 8),
                static_cast<const __nv_bfloat16*>(o.up) + std::int64_t(native_row) * kRank +
                    group * BK + k8 * 8);
        }
        for (int i = tid; i < Tokens * (BK / 8); i += S::kThreads) {
            const int token = i / (BK / 8), k8 = i % (BK / 8);
            const bool valid = token0 + token < o.columns;
            const auto source =
                valid ? std::int64_t(token0 + token) * kRank + group * BK + k8 * 8 : 0;
            common::cp_async_zfill<16, S::kActivationCache>(
                bs + token * BK + bf16_mma_shared_col<S>(token, k8 * 8), activation + source,
                valid ? 16 : 0);
        }
    };
    constexpr int groups = kRank / BK;
#pragma unroll
    for (int slot = 0; slot < Stages; ++slot) {
        stage_inputs(slot, slot);
        common::cp_commit();
    }
#pragma unroll
    for (int group = 0; group < groups; ++group) {
        if (group + Stages <= groups)
            common::cp_wait<Stages - 1>();
        else
            common::cp_wait<0>();
        __syncthreads();
        const int slot = group % Stages;
        bf16_mma_compute_stage<S>(a + slot * Rows * BK, b + slot * Tokens * BK, accum, warp, lane);
        __syncthreads();
        if (group + Stages < groups) {
            stage_inputs(slot, group + Stages);
            common::cp_commit();
        }
    }
    auto* contributions = reinterpret_cast<float*>(storage);
    const auto at       = [&](int r, int t) { return contribution_index<Rows, Tokens>(r, t); };
    const int wm = warp / S::kWarpsTokens, wn = warp % S::kWarpsTokens;
    const int row = wm * 16 + (lane >> 2), token = wn * 8 + 2 * (lane & 3);
    const auto store = [&](int r, int t, float value) {
        const int s = r / Features, d = d0 + r % Features;
        if (token0 + t < o.columns) {
            const auto z            = __bfloat162float(static_cast<const __nv_bfloat16*>(
                o.normalized)[std::int64_t(token0 + t) * kResidualWidth + s * kWidth + d]);
            contributions[at(r, t)] = common::sigmoid_approx(value) * z;
        }
    };
    const auto& v = accum[0][0];
    store(row, token, v[0]);
    store(row, token + 1, v[1]);
    store(row + 8, token, v[2]);
    store(row + 8, token + 1, v[3]);
    __syncthreads();
    if (wm * 16 < Features) {
        const auto finish = [&](int d, int t) {
            if (token0 + t < o.columns) {
                float sum = 0;
#pragma unroll
                for (int s = 0; s < kStreams; ++s) sum += contributions[at(s * Features + d, t)];
                static_cast<__nv_bfloat16*>(o.output)[std::int64_t(token0 + t) * kWidth + d0 + d] =
                    __float2bfloat16_rn(sum * 0.25F);
            }
        };
        finish(row, token);
        finish(row, token + 1);
        finish(row + 8, token);
        finish(row + 8, token + 1);
    }
}
} // namespace ninfer::ops::detail::hc::mix::bf16
