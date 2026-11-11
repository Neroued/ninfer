#pragma once

#include "ops/gated_residual/mix/bf16/kernel/output.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_mma.cuh"

namespace ninfer::ops::detail::hc::mix::bf16 {
struct StripedTmaWeightAccess {
    template <class Schedule>
    static CUtensorMap map(const Bf16A16Operands& p) {
        if (p.rows != kResidualWidth || p.k != kRank)
            throw std::invalid_argument("hc_mix: invalid native striped Up geometry");
        CUtensorMap result{};
        const std::uint64_t dimensions[]{64, kRank / 64, kWidth, kStreams};
        const std::uint64_t strides[]{128, kRank * 2, std::uint64_t(kWidth) * kRank * 2};
        const std::uint32_t box[]{64, Schedule::kBlockK / 64, Schedule::kBlockRows / kStreams,
                                  kStreams};
        const std::uint32_t steps[]{1, 1, 1, 1};
        const auto status = cuTensorMapEncodeTiled(
            &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<__nv_bfloat16*>(p.weight),
            dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (status != CUDA_SUCCESS)
            throw std::runtime_error("hc_mix: native striped TMA descriptor failed");
        return result;
    }

    template <class Schedule>
    __device__ __forceinline__ static void load(void* destination, const CUtensorMap* map,
                                                int k_sector, int row, std::uint64_t* barrier) {
        asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                     :
                     : "r"(common::smem_addr(destination)), "l"(map), "r"(0), "r"(k_sector),
                       "r"(row / kStreams), "r"(0), "r"(common::smem_addr(barrier))
                     : "memory");
    }
};

struct TmaMixEpilogue {
    static constexpr bool kTmaCollective = true;
    const __nv_bfloat16* normalized;
    template <class Schedule>
    static constexpr int kSharedBytes =
        Schedule::kBlockRows * Schedule::kBlockTokens * sizeof(float);

    template <class S, bool FullTokens, class Output>
    __device__ __forceinline__ void finish_tile(const Output& output, unsigned char* storage,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row_begin, int token_begin, int,
                                                int tokens) const {
        constexpr int Features = S::kBlockRows / kStreams, Tokens = S::kBlockTokens,
                      Rows = S::kBlockRows;
        const int tid = threadIdx.x - S::kProducerThreads, warp = tid / 32, lane = tid % 32;
        const int wm = warp / S::kWarpsTokens, wn = warp % S::kWarpsTokens;
        const int feature_begin = row_begin / kStreams;
        auto* partial           = reinterpret_cast<float*>(storage);
        const auto at    = [&](int r, int t) { return contribution_index<Rows, Tokens>(r, t); };
        const auto store = [&](int r, int t, float value) {
            if (FullTokens || token_begin + t < tokens) {
                const int s = r / Features, d = feature_begin + r % Features;
                const float z = __bfloat162float(
                    normalized[std::int64_t(token_begin + t) * kResidualWidth + s * kWidth + d]);
                partial[at(r, t)] = common::sigmoid_approx(value) * z;
            }
        };
#pragma unroll
        for (int mi = 0; mi < S::kMmaRows; ++mi) {
            const int r = wm * S::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
            for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                const int t   = wn * S::kWarpTokens + ni * 8 + 2 * (lane & 3);
                const auto& v = accum[mi][ni];
                if (threadIdx.x >= S::kProducerThreads) {
                    store(r, t, v[0]);
                    store(r, t + 1, v[1]);
                    store(r + 8, t, v[2]);
                    store(r + 8, t + 1, v[3]);
                }
            }
        }
        __syncthreads();
        if (threadIdx.x >= S::kProducerThreads && wm * S::kWarpRows < Features) {
            const auto finish = [&](int d, int t) {
                if constexpr (S::kWarpRows > Features) {
                    if (d >= Features) return;
                }
                if (FullTokens || token_begin + t < tokens) {
                    float sum = 0;
#pragma unroll
                    for (int s = 0; s < kStreams; ++s) sum += partial[at(s * Features + d, t)];
                    output.store(feature_begin + d, token_begin + t, sum * 0.25F);
                }
            };
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                const int d = wm * S::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
                for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                    const int t = wn * S::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    finish(d, t);
                    finish(d, t + 1);
                    finish(d + 8, t);
                    finish(d + 8, t + 1);
                }
            }
        }
    }
};

struct TmaPrefetchedMixEpilogue {
    static constexpr bool kTmaCollective = true;
    const __nv_bfloat16* normalized;
    template <class Schedule>
    static constexpr int kSharedBytes =
        Schedule::kBlockRows * Schedule::kBlockTokens * (sizeof(float) + sizeof(__nv_bfloat16));

    template <class S, bool FullTokens, class Output>
    __device__ __forceinline__ void finish_tile(const Output& output, unsigned char* storage,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row_begin, int token_begin, int,
                                                int tokens) const {
        constexpr int Features = S::kBlockRows / kStreams, Tokens = S::kBlockTokens,
                      Rows = S::kBlockRows;
        const int tid = threadIdx.x - S::kProducerThreads, warp = tid / 32, lane = tid % 32;
        const int wm = warp / S::kWarpsTokens, wn = warp % S::kWarpsTokens;
        const int feature_begin = row_begin / kStreams;
        auto* partial           = reinterpret_cast<float*>(storage);
        const auto at = [&](int r, int t) { return contribution_index<Rows, Tokens>(r, t); };
        static_assert(Rows % 64 == 0 && Features % 8 == 0);
        const auto z_at = [&](int r, int t) { return t * Rows + (r ^ ((t & 7) << 3)); };
        // Staging has retired before the collective callback. Keep BF16 Z separate from
        // the FP32 contribution prefix so no producer overwrites an unread input.
        auto* z_tile = reinterpret_cast<__nv_bfloat16*>(storage + Rows * Tokens * sizeof(float));
        for (int v = int(threadIdx.x); v < Tokens * (Rows / 8); v += S::kThreads) {
            const int t = v / (Rows / 8), r = (v % (Rows / 8)) * 8;
            uint4 words{};
            if (FullTokens || token_begin + t < tokens) {
                const int stream = r / Features, feature = feature_begin + r % Features;
                words = common::load_vec<uint4>(normalized +
                                                std::int64_t(token_begin + t) * kResidualWidth +
                                                stream * kWidth + feature);
            }
            common::store_vec(z_tile + z_at(r, t), words);
        }
        __syncthreads();
        const auto store = [&](int r, int t, float value) {
            if (FullTokens || token_begin + t < tokens) {
                const float z     = __bfloat162float(z_tile[z_at(r, t)]);
                partial[at(r, t)] = common::sigmoid_approx(value) * z;
            }
        };
#pragma unroll
        for (int mi = 0; mi < S::kMmaRows; ++mi) {
            const int r = wm * S::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
            for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                const int t   = wn * S::kWarpTokens + ni * 8 + 2 * (lane & 3);
                const auto& v = accum[mi][ni];
                if (threadIdx.x >= S::kProducerThreads) {
                    store(r, t, v[0]);
                    store(r, t + 1, v[1]);
                    store(r + 8, t, v[2]);
                    store(r + 8, t + 1, v[3]);
                }
            }
        }
        __syncthreads();
        if (threadIdx.x >= S::kProducerThreads && wm * S::kWarpRows < Features) {
            const auto finish = [&](int d, int t) {
                if constexpr (S::kWarpRows > Features) {
                    if (d >= Features) return;
                }
                if (FullTokens || token_begin + t < tokens) {
                    float sum = 0;
#pragma unroll
                    for (int s = 0; s < kStreams; ++s) sum += partial[at(s * Features + d, t)];
                    output.store(feature_begin + d, token_begin + t, sum * 0.25F);
                }
            };
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                const int d = wm * S::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
                for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                    const int t = wn * S::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    finish(d, t);
                    finish(d, t + 1);
                    finish(d + 8, t);
                    finish(d + 8, t + 1);
                }
            }
        }
    }
};

} // namespace ninfer::ops::detail::hc::mix::bf16
