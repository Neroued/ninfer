#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/common/vector_output.cuh"

namespace ninfer::ops::detail {
// Matching gate/up fragments belong to the same warp. The persistent weight
// planes are unchanged; only the CTA's logical load rows are reordered.
template <class Schedule>
struct SwiGluTokenMajorMmaRows {
    static_assert(Schedule::kWarpRows % 16 == 0);
    static constexpr bool kPaired          = true;
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
        const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
        const int wt = warp / S::kWarpsRows, wr = warp % S::kWarpsRows;
        auto* final_tile = reinterpret_cast<__nv_bfloat16*>(scratch);
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
            const int token = wt * S::kWarpTokens + mt * 16 + (lane >> 2);
#pragma unroll
            for (int mr = 0; mr < gate_fragments; ++mr) {
                const int row    = wr * (S::kWarpRows / 2) + mr * 8 + 2 * (lane & 3);
                const auto& gate = acc[mt][mr];
                const auto& up   = acc[mt][mr + gate_fragments];
                *reinterpret_cast<__nv_bfloat162*>(final_tile + token * stride + row) =
                    __floats2bfloat162_rn(silu(gate[0]) * up[0], silu(gate[1]) * up[1]);
                *reinterpret_cast<__nv_bfloat162*>(final_tile + (token + 8) * stride + row) =
                    __floats2bfloat162_rn(silu(gate[2]) * up[2], silu(gate[3]) * up[3]);
            }
        }
        __syncthreads();
        for (int item = tid; item < S::kBlockTokens * (rows / 8); item += S::kThreads) {
            const int token = item / (rows / 8), row = (item % (rows / 8)) * 8;
            if (Full || token_begin + token < token_end)
                linear_store_bf16_vector(output, row_begin + row, token_begin + token,
                                         load_vec<uint4>(final_tile + token * stride + row));
        }
    }
};
} // namespace ninfer::ops::detail
