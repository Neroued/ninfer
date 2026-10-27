#pragma once
#include "ops/common/numeric/math.cuh"
#include "ops/common/cuda/memory.cuh"

#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
// Matching gate/up fragments belong to the same warp. The persistent weight
// planes are unchanged; only the CTA's logical load rows are reordered.
template <class Schedule>
struct SwiGluTokenMajorMmaRows {
    static_assert(Schedule::kWarpRows % 16 == 0);
    static constexpr bool kPaired          = true;
    static constexpr bool kWarpPaired      = true;
    static constexpr bool kContiguous      = false;
    static constexpr bool kContiguousPairs = true;

    __device__ __forceinline__ int weight_row(int begin, int local_row, int rows) const {
        constexpr int half_warp = Schedule::kWarpRows / 2;
        const int warp          = local_row / Schedule::kWarpRows;
        const int within        = local_row % Schedule::kWarpRows;
        return begin + warp * half_warp + within % half_warp + (within >= half_warp ? rows / 2 : 0);
    }
};

struct SwiGluTokenMajorMmaEpilogue {
    __device__ __forceinline__ float apply_pair(int, int, float gate, float up) const {
        return common::silu(gate) * up;
    }

    template <class Schedule>
    static constexpr int kSharedBytes = Schedule::kBlockTokens * (Schedule::kBlockRows / 2 + 8) * 2;

    template <class S, bool Full, class Output>
    __device__ __forceinline__ void
    finish_tile(Output output, unsigned char* scratch, float (&acc)[S::kMmaTokens][S::kMmaRows][4],
                int row_begin, int token_begin, int, int token_end) const {
        static_assert(S::kMmaRows % 2 == 0);
        constexpr int rows           = S::kBlockRows / 2;
        constexpr int stride         = rows + 8;
        constexpr int gate_fragments = S::kMmaRows / 2;
        constexpr int producers      = [] {
            if constexpr (requires { S::kProducerThreads; })
                return S::kProducerThreads;
            else
                return 0;
        }();
        constexpr int threads = S::kThreads - producers;
        const int tid = static_cast<int>(threadIdx.x) - producers, warp = tid >> 5, lane = tid & 31;
        const auto synchronize = [] {
            if constexpr (producers)
                asm volatile("bar.sync 1, %0;" ::"r"(threads) : "memory");
            else
                __syncthreads();
        };
        const int wt = warp / S::kWarpsRows, wr = warp % S::kWarpsRows;
        auto* final_tile = reinterpret_cast<__nv_bfloat16*>(scratch);
        synchronize();
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
            const int token = wt * S::kWarpTokens + mt * 16 + (lane >> 2);
#pragma unroll
            for (int mr = 0; mr < gate_fragments; ++mr) {
                const int row    = wr * (S::kWarpRows / 2) + mr * 8 + 2 * (lane & 3);
                const auto& gate = acc[mt][mr];
                const auto& up   = acc[mt][mr + gate_fragments];
                // Invalid token fragments never reach the output. Avoid their exp/div
                // work while keeping every valid gate/up pair in FP32 through the product.
                if (Full || token_begin + token < token_end)
                    *reinterpret_cast<__nv_bfloat162*>(final_tile + token * stride + row) =
                        __floats2bfloat162_rn(common::silu(gate[0]) * up[0],
                                             common::silu(gate[1]) * up[1]);
                if (Full || token_begin + token + 8 < token_end)
                    *reinterpret_cast<__nv_bfloat162*>(final_tile + (token + 8) * stride + row) =
                        __floats2bfloat162_rn(common::silu(gate[2]) * up[2],
                                             common::silu(gate[3]) * up[3]);
            }
        }
        synchronize();
        for (int item = tid; item < S::kBlockTokens * (rows / 8); item += threads) {
            const int token = item / (rows / 8), row = (item % (rows / 8)) * 8;
            if (Full || token_begin + token < token_end)
                linear_store_bf16_vector(
                    output, row_begin + row, token_begin + token,
                    common::load_vec<uint4>(final_tile + token * stride + row));
        }
    }
};
} // namespace ninfer::ops::detail
