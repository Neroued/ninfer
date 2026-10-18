#pragma once
#include "ops/linear/swiglu/paired_output.cuh"

namespace ninfer::ops::detail {
template <class Product>
struct BasicSwiGluPairEpilogue {
    Product product{};

    template <class Output>
    __device__ __forceinline__ void apply_pair(Output output, int row, int token, float gate,
                                               float up) const {
        output.store(row, token, product(gate, up));
    }
};

template <class Product>
struct BasicSwiGluTokenMajorMmaEpilogue {
    Product product{};

    __device__ __forceinline__ float apply_pair(int, int, float gate, float up) const {
        return product(gate, up);
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
                *reinterpret_cast<__nv_bfloat162*>(final_tile + token * stride + row) =
                    __floats2bfloat162_rn(product(gate[0], up[0]), product(gate[1], up[1]));
                *reinterpret_cast<__nv_bfloat162*>(final_tile + (token + 8) * stride + row) =
                    __floats2bfloat162_rn(product(gate[2], up[2]), product(gate[3], up[3]));
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
