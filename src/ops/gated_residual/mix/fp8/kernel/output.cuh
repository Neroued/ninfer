#pragma once
#include "ops/gated_residual/mix/fp8/operands.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/fp8.cuh"
#include "ops/linear/kernels/fp8/fp8_a16_gemv.cuh"

namespace ninfer::ops::detail::hc::mix::fp8 {
struct DownEpilogue {
    __device__ float apply(int, int, float value) const { return common::silu(value * 0.25F); }
};

struct UpOutput {
    const __nv_bfloat16* z;
    __nv_bfloat16* x;
    int columns;
};

// All four streams for a feature tile belong to the same CTA, without repacking weights.
template <int Features>
struct StreamRows {
    static constexpr bool kPaired = false, kContiguousPairs = true;

    __device__ int weight_row(int begin, int row, int) const {
        return (row / Features) * kWidth + begin / 4 + row % Features;
    }
};

struct A16UpEpilogue {
    template <class S>
    static constexpr int kSharedBytes = S::kBlockRows * S::kBlockTokens * 4;

    template <class S, bool Full, class A>
    __device__ void finish_tile(UpOutput output, unsigned char* storage, A& values, int first,
                                int token0, int, int tokens) const {
        constexpr int Rows = S::kBlockRows, Tokens = S::kBlockTokens, F = Rows / 4;
        const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
        auto* c          = reinterpret_cast<float*>(storage);
        const auto store = [&](int r, int t, float v) {
            if (Full || token0 + t < tokens) {
                const int s = r / F, d = first / 4 + r % F;
                c[t * Rows + r] =
                    common::sigmoid(v) *
                    __bfloat162float(output.z[std::int64_t(token0 + t) * kTotal + s * kWidth + d]);
            }
        };
        const int wr = warp / S::kWarpsTokens, wt = warp % S::kWarpsTokens;
#pragma unroll
        for (int mr = 0; mr < S::kMmaRows; ++mr)
#pragma unroll
            for (int mt = 0; mt < S::kMmaTokens; ++mt) {
                const int r   = wr * S::kWarpRows + mr * 16 + (lane >> 2),
                          t   = wt * S::kWarpTokens + mt * 8 + 2 * (lane & 3);
                const auto& v = values[mr][mt];
                store(r, t, v[0]);
                store(r, t + 1, v[1]);
                store(r + 8, t, v[2]);
                store(r + 8, t + 1, v[3]);
            }
        __syncthreads();
        for (int i = tid; i < Tokens * F; i += S::kThreads) {
            const int t = i / F, d = i % F;
            if (Full || token0 + t < tokens) {
                float sum = 0;
#pragma unroll
                for (int stream = 0; stream < 4; ++stream) sum += c[t * Rows + stream * F + d];
                output.x[std::int64_t(token0 + t) * kWidth + first / 4 + d] =
                    __float2bfloat16_rn(sum * .25F);
            }
        }
    }
};

// Native row-scaled FP8 GEMV; each lane group owns all four stream gates.
__global__ __launch_bounds__(128) void up_gemv(Fp8A16Operands p, UpOutput output) {
    constexpr int Group = 8, Features = 128 / Group;
    const int d = int(blockIdx.x) * Features + int(threadIdx.x) / Group, lane = threadIdx.x % Group;
    __shared__ __align__(16) __nv_bfloat16 u[kRank];
    if (threadIdx.x < kRank / 8)
        common::store_vec(u + threadIdx.x * 8, common::load_vec<uint4>(p.x + threadIdx.x * 8));
    __syncthreads();
    float sum = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        float dot = 0;
#pragma unroll
        for (int k = lane * 8; k < kRank; k += Group * 8) {
            const auto bits =
                common::load_vec<uint2>(p.codes + (std::int64_t(s) * kWidth + d) * kRank + k);
            const unsigned words[2]{bits.x, bits.y};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const auto code = common::fp8_code2_to_float2(
                    static_cast<std::uint16_t>(words[q / 2] >> ((q % 2) * 16)));
                const auto x =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(u + k + 2 * q));
                dot = fmaf(code.x, x.x, dot);
                dot = fmaf(code.y, x.y, dot);
            }
        }
        dot = common::warp_reduce_sum<Group>(dot);
        if (lane == 0)
            sum += common::sigmoid(dot * __bfloat162float(p.scales[s * kWidth + d])) *
                   __bfloat162float(output.z[s * kWidth + d]);
    }
    if (lane == 0) output.x[d] = __float2bfloat16_rn(sum * 0.25F);
}

// Injection consumes original BF16 z even when the down projection uses A8.
__global__ __launch_bounds__(256) void injection(const __nv_bfloat16* z, const __nv_bfloat16* w,
                                                 float* out, int m) {
    const auto col = std::int64_t(blockIdx.x);
    const int s    = blockIdx.y;
    float dot      = 0;
    for (int k = threadIdx.x; k < kTotal; k += 256)
        dot = fmaf(__bfloat162float(z[col * kTotal + k]), __bfloat162float(w[s * kTotal + k]), dot);
    __shared__ float sums[8];
    dot = common::block_reduce_sum<256>(dot, sums);
    if (threadIdx.x == 0) out[col * 4 + s] = 2.F * common::sigmoid(dot * 0.25F);
}

// Pairwise, 8x8 microtiles match A8 MMA lanes. A 64-bit shared transaction
// touches each bank once per half warp; adjacent normalized BF16 values share a load.
struct A8UpEpilogue {
    template <class S>
    static constexpr int kSharedBytes = S::kBlockRows * S::kBlockTokens * 4;

    template <int Tokens>
    __device__ static int at(int r, int t) {
        return ((r / 8) * (Tokens / 8) + t / 8) * 64 + (t % 8) * 8 + r % 8;
    }

    template <class S, bool Full, class A>
    __device__ void finish_tile(UpOutput o, unsigned char* storage, A& values, int first,
                                int token0, int, int tokens) const {
        constexpr int Rows = S::kBlockRows, Tokens = S::kBlockTokens, F = Rows / 4;
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wr = warp % S::kWarpsRows,
                  wt     = warp / S::kWarpsRows;
        auto* c          = reinterpret_cast<float*>(storage);
        const auto store = [&](int r, int t, int token, float x, float y) {
            const int stream = r / F, d = first / 4 + r % F;
            const auto z = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                o.z + std::int64_t(token) * kTotal + stream * kWidth + d));
            *reinterpret_cast<float2*>(c + at<Tokens>(r, t)) =
                make_float2(common::sigmoid(x) * z.x, common::sigmoid(y) * z.y);
        };
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
            const int t = wt * S::kWarpTokens + mt * 16 + (lane >> 2);
            // Only private encoded activations are padded. Public normalized values
            // are loaded at a safe column; invalid columns never reach the output.
            const int token = min(token0 + t, o.columns - 1);
            const int next  = min(token0 + t + 8, o.columns - 1);
#pragma unroll
            for (int mr = 0; mr < S::kMmaRows; ++mr) {
                const int r   = wr * S::kWarpRows + mr * 8 + 2 * (lane & 3);
                const auto& v = values[mt][mr];
                store(r, t, token, v[0], v[1]);
                store(r, t + 8, next, v[2], v[3]);
            }
        }
        __syncthreads();
        for (int tile = warp; tile < Tokens / 8 * (F / 8); tile += S::kThreads / 32) {
            const int t = tile / (F / 8) * 8 + (lane >> 2), d = tile % (F / 8) * 8 + 2 * (lane & 3);
            if (token0 + t < o.columns) {
                float2 sum{0, 0};
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    auto x = *reinterpret_cast<const float2*>(c + at<Tokens>(s * F + d, t));
                    sum.x += x.x;
                    sum.y += x.y;
                }
                *reinterpret_cast<__nv_bfloat162*>(o.x + std::int64_t(token0 + t) * kWidth +
                                                   first / 4 + d) =
                    __floats2bfloat162_rn(sum.x * .25F, sum.y * .25F);
            }
        }
    }
};
} // namespace ninfer::ops::detail::hc::mix::fp8
