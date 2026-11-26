#include "launch.h"

#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"

#include <algorithm>
#include <cuda_bf16.h>

namespace ninfer::ops::markov {
namespace {

using Bf16 = __nv_bfloat16;

constexpr int kWarps   = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kRows    = 16 * kWarps;

struct Problem {
    const Bf16* latent;
    const Bf16* weights;
    const Bf16* bias;
    const Bf16* base;
    const int* live;
    Bf16* out;
    int rank, batch, vocab, domain;
    float softcap;
};

__device__ __forceinline__ float softcap(float x, float cap) {
    return cap * tanhf(__fdiv_rn(x, cap));
}

__device__ __forceinline__ Bf16 finish(float dot, Bf16 bias, Bf16 base, float cap) {
    const auto delta = __float2bfloat16_rn(dot + __bfloat162float(bias));
    const auto sum   = __float2bfloat16_rn(__bfloat162float(base) + __bfloat162float(delta));
    return __float2bfloat16_rn(softcap(__bfloat162float(sum), cap));
}

__device__ __forceinline__ unsigned finish_pair(float lo, float hi, unsigned bias, unsigned base,
                                                float cap) {
    const auto delta = __floats2bfloat162_rn(lo + __uint_as_float(bias << 16),
                                             hi + __uint_as_float(bias & 0xffff0000U));
    const __nv_bfloat162_raw base_pair{static_cast<unsigned short>(base),
                                       static_cast<unsigned short>(base >> 16)};
    // BF16 addition has the same rounding boundary as FP32 addition followed by BF16 conversion.
    const auto sum = __bfloat1622float2(__hadd2(delta, __nv_bfloat162(base_pair)));
    const __nv_bfloat162_raw result =
        __floats2bfloat162_rn(softcap(sum.x, cap), softcap(sum.y, cap));
    return unsigned(result.x) | (unsigned(result.y) << 16);
}

__device__ __forceinline__ void store(const Problem& p, float4 dot, int b, int v, bool enabled) {
    const auto index = std::int64_t(b) * p.vocab + v;
    if (p.domain - v >= 4 && (index & 3) == 0) {
        uint2 result{};
        if (enabled) {
            const auto bias = common::load_vec<uint2>(p.bias + v);
            const auto base = common::load_vec<uint2>(p.base + index);
            result.x        = finish_pair(dot.x, dot.y, bias.x, base.x, p.softcap);
            result.y        = finish_pair(dot.z, dot.w, bias.y, base.y, p.softcap);
        }
        common::store_vec(p.out + index, result);
    } else {
        const float values[]{dot.x, dot.y, dot.z, dot.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (v + i < p.vocab) {
                Bf16 value = __float2bfloat16_rn(0.0f);
                if (enabled && v + i < p.domain)
                    value = finish(values[i], p.bias[v + i], p.base[index + i], p.softcap);
                p.out[index + i] = value;
            }
        }
    }
}

template <int Columns, int KTile, int FixedRank, bool Masked, bool Resident>
__global__ void logits(const __grid_constant__ Problem p) {
    constexpr int kMmas    = Columns / 8;
    constexpr int kVectors = KTile / 8;
    __shared__ __align__(16) Bf16 tile[(kRows + Columns) * KTile];
    __shared__ std::uint64_t live_mask;
    auto* activations = tile + kRows * KTile;

    const int rank = FixedRank ? FixedRank : p.rank;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int v0        = blockIdx.x * kRows;
    const bool interior = p.domain - v0 >= kRows;
    const int a_row = warp * 16 + (lane & 15), a_col = (lane >> 4) * 8;
    const int b_row = lane & 7, b_col = (lane & 8) != 0 ? 8 : 0;
    const unsigned a_base = common::smem_addr(tile + a_row * KTile);
    const unsigned b_base = common::smem_addr(activations + b_row * KTile);

    const auto stage_weight = [&](int k0, int depth) {
        const auto copy = [&]<bool Full>() {
#pragma unroll
            for (int i = tid; i < kRows * kVectors; i += kThreads) {
                const int r = i / kVectors, k = (i % kVectors) * 8;
                const int v    = v0 + r;
                const bool ok  = Full || (v < p.domain && k < depth);
                common::cp_async_zfill<16, Cache::cg>(
                    tile + r * KTile + common::swizzle_xor_64(r, k),
                    ok ? (p.weights + std::int64_t(v) * rank + k0 + k) : p.weights, ok ? 16 : 0);
            }
        };
        if (interior && depth == KTile)
            copy.template operator()<true>();
        else
            copy.template operator()<false>();
    };

    // With a resident rank tile, the same weights serve every batch tile owned by the CTA.
    if constexpr (Resident) {
        if (v0 < p.domain) {
            stage_weight(0, rank);
            common::cp_commit();
            common::cp_wait<0>();
            __syncthreads();
        }
    }

    for (std::int64_t start = std::int64_t(blockIdx.y) * Columns; start < p.batch;
         start += gridDim.y * Columns) {
        const int b0        = static_cast<int>(start);
        const int count     = min(Columns, p.batch - b0);
        std::uint64_t mask  = count >= 64 ? ~0ULL : ((1ULL << count) - 1ULL);
        if constexpr (Masked) {
            if (warp == 0) {
                const unsigned lo = __ballot_sync(0xffffffff, lane < count && p.live[b0 + lane]);
                unsigned hi       = 0;
                if constexpr (Columns > 32) {
                    hi = __ballot_sync(0xffffffff, lane + 32 < count && p.live[b0 + lane + 32]);
                }
                if (lane == 0) live_mask = std::uint64_t(lo) | (std::uint64_t(hi) << 32);
            }
            __syncthreads();
            mask = live_mask;
        }
        if (mask == 0 || v0 >= p.domain) {
            for (int i = tid; i < kRows * count; i += kThreads) {
                const int v = v0 + i % kRows;
                if (v < p.vocab)
                    p.out[std::int64_t(b0 + i / kRows) * p.vocab + v] = __float2bfloat16_rn(0.0f);
            }
        } else {
            float sum[kMmas][4]{};
            for (int k0 = 0; k0 < rank; k0 += KTile) {
                const int depth = min(KTile, rank - k0);
                if constexpr (!Resident) stage_weight(k0, depth);

#pragma unroll
                for (int i = tid; i < Columns * kVectors; i += kThreads) {
                    const int r = i / kVectors, k = (i % kVectors) * 8;
                    const bool load = (mask & (1ULL << r)) && k < depth;
                    common::cp_async_zfill<16, Cache::cg>(
                        activations + r * KTile + common::swizzle_xor_64(r, k),
                        p.latent + (load ? std::int64_t(b0 + r) * rank + k0 + k : 0),
                        load ? 16 : 0);
                }
                common::cp_commit();
                common::cp_wait<0>();
                __syncthreads();

                // Each weight fragment feeds every batch fragment before the next K load.
#pragma unroll
                for (int k = 0; k < KTile; k += 16) {
                    unsigned a0, a1, a2, a3;
                    common::ldmatrix_x4(a0, a1, a2, a3,
                                        a_base + 2 * common::swizzle_xor_64(a_row, k + a_col));
#pragma unroll
                    for (int n = 0; n < kMmas; ++n) {
                        unsigned x0, x1;
                        common::ldmatrix_x2(x0, x1,
                                            b_base + n * 8 * KTile * sizeof(Bf16) +
                                                2 * common::swizzle_xor_64(b_row, k + b_col));
                        auto& c = sum[n];
                        common::mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, x0, x1);
                    }
                }
                if (k0 + KTile < rank) __syncthreads();
            }

#pragma unroll
            for (int n = 0; n < kMmas; ++n) {
                const auto quad =
                    common::mma_row_quad(make_float4(sum[n][0], sum[n][1], sum[n][2], sum[n][3]));
                const int b = n * 8 + quad.column;
                if (b < count)
                    store(p, quad.values, b0 + b, v0 + warp * 16 + quad.row,
                          (mask & (1ULL << b)) != 0);
            }
        }
        if (start + gridDim.y * Columns < p.batch) __syncthreads();
    }
}

template <int Columns, int KTile, int FixedRank, bool Resident = false>
void submit(Problem p, cudaStream_t stream) {
    const dim3 grid(1 + (p.vocab - 1) / kRows,
                    Resident ? 1 : std::min(65535, 1 + (p.batch - 1) / Columns));
    if (p.live)
        logits<Columns, KTile, FixedRank, true, Resident><<<grid, kThreads, 0, stream>>>(p);
    else
        logits<Columns, KTile, FixedRank, false, Resident><<<grid, kThreads, 0, stream>>>(p);
}
} // namespace

void launch_logits(const Tensor& latent, const Weight& weight, const Tensor& bias,
                   const Tensor& base, int domain, float softcap, const Tensor& live,
                   Tensor& output, CudaExecutionView execution) {
    const Problem p{static_cast<const Bf16*>(latent.data),
                    static_cast<const Bf16*>(weight.qdata),
                    static_cast<const Bf16*>(bias.data),
                    static_cast<const Bf16*>(base.data),
                    static_cast<const int*>(live.data),
                    static_cast<Bf16*>(output.data),
                    latent.ne[0],
                    latent.ne[1],
                    weight.n,
                    domain,
                    softcap};
    const auto launch = [&]<int KTile, int FixedRank>() {
        const auto stream = execution.stream();
        if (p.batch <= 8)
            submit<8, KTile, FixedRank>(p, stream);
        else if (p.batch <= 16)
            submit<16, KTile, FixedRank>(p, stream);
        else if (p.batch <= 24)
            submit<24, KTile, FixedRank>(p, stream);
        else if (p.batch <= 32)
            submit<32, KTile, FixedRank>(p, stream);
        else if (p.batch <= 48)
            submit<48, 64, FixedRank>(p, stream);
        else
            submit<64, 64, FixedRank>(p, stream);
    };
    const auto stream = execution.stream();
    if (p.batch > 64 && p.rank <= 64)
        submit<32, 64, 0, true>(p, stream);
    else if (p.batch > 64 && p.rank <= 128)
        submit<32, 128, 0, true>(p, stream);
    else if (p.rank == 256)
        launch.template operator()<128, 256>();
    else if (p.rank <= 64)
        launch.template operator()<64, 0>();
    else
        launch.template operator()<128, 0>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::markov
