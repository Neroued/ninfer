#include "ops/loop_message/rtx5090/kernel.cuh"
#include "ops/common/cuda/mma.cuh"
#include "core/pdl.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::loop_message {
namespace {
constexpr int kThreads = 256;
constexpr int kWarps   = kThreads / 32;

__device__ __forceinline__ float dot8(uint4 lhs, uint4 rhs, float sum) {
    const unsigned a[]{lhs.x, lhs.y, lhs.z, lhs.w}, b[]{rhs.x, rhs.y, rhs.z, rhs.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 av = common::bf16x2_bits_to_float2(a[j]);
        const float2 bv = common::bf16x2_bits_to_float2(b[j]);
        sum             = fmaf(av.x, bv.x, sum);
        sum             = fmaf(av.y, bv.y, sum);
    }
    return sum;
}

__device__ __forceinline__ float dot_pair4(float4 x, float scale, const Bf16* w, float sum) {
    const auto first  = __bfloat1622float2(__floats2bfloat162_rn(x.x * scale, x.y * scale));
    const auto second = __bfloat1622float2(__floats2bfloat162_rn(x.z * scale, x.w * scale));
    const auto* pair  = reinterpret_cast<const __nv_bfloat162*>(w);
    const auto lo = __bfloat1622float2(pair[0]), hi = __bfloat1622float2(pair[1]);
    sum = fmaf(first.x, lo.x, sum);
    sum = fmaf(first.y, lo.y, sum);
    sum = fmaf(second.x, hi.x, sum);
    return fmaf(second.y, hi.y, sum);
}

__global__ __launch_bounds__(kThreads, 4) void down_vector(Arguments a) {
    __shared__ RmsStats stats[2][kWarps];
    __shared__ float retry[2][kWarps];
    const int lane            = threadIdx.x % 32;
    const std::int64_t groups = (std::int64_t(a.rank) + kWarps - 1) / kWarps;
    for (std::int64_t block = blockIdx.x; block < std::int64_t(a.tokens) * a.streams * groups;
         block += gridDim.x) {
        const auto row = block / groups;
        const Token token(a, row);
        const int k      = int(block % groups) * kWarps + threadIdx.x / 32;
        const auto scale = input_scales<kThreads>(token, a.hidden, a.pair_eps, stats, retry);
        if (k < a.rank) {
            const auto* w = a.down + (token.stream * a.rank + k) * (2 * a.hidden);
            float ds = 0, ms = 0;
            for (int d = lane; d < a.hidden; d += 32) {
                const float x = float(__float2bfloat16_rn(token.dest[d] * scale.x));
                const float y = float(__float2bfloat16_rn(token.message[d] * scale.y));
                ds            = fmaf(x, float(w[d]), ds);
                ms            = fmaf(y, float(w[d + a.hidden]), ms);
            }
            const float sum = common::warp_reduce_sum(ds + ms);
            if (lane == 0) a.latent[row * a.rank + k] = __float2bfloat16_rn(sum);
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kThreads, 4) void down_vector_prefill(Arguments a) {
    // One CTA normalizes the row once and reuses the rounded pair for all ranks.
    extern __shared__ __align__(16) Bf16 pair[];
    __shared__ RmsStats stats[2][kWarps];
    __shared__ float retry[2][kWarps];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    for (std::int64_t row = blockIdx.x; row < std::int64_t(a.tokens) * a.streams;
         row += gridDim.x) {
        const Token token(a, row);
        const auto scale    = input_scales<kThreads>(token, a.hidden, a.pair_eps, stats, retry);
        const auto* dest    = reinterpret_cast<const float4*>(token.dest);
        const auto* message = reinterpret_cast<const float4*>(token.message);
        for (int i = tid; i < a.hidden / 4; i += kThreads) {
            const auto d = dest[i], m = message[i];
            auto* dst = reinterpret_cast<__nv_bfloat162*>(pair + 4 * i);
            auto* msg = reinterpret_cast<__nv_bfloat162*>(pair + a.hidden + 4 * i);
            dst[0]    = __floats2bfloat162_rn(d.x * scale.x, d.y * scale.x);
            dst[1]    = __floats2bfloat162_rn(d.z * scale.x, d.w * scale.x);
            msg[0]    = __floats2bfloat162_rn(m.x * scale.y, m.y * scale.y);
            msg[1]    = __floats2bfloat162_rn(m.z * scale.y, m.w * scale.y);
        }
        __syncthreads();
        for (int k = warp; k < a.rank; k += kWarps) {
            const auto* w       = a.down + (token.stream * a.rank + k) * (2 * a.hidden);
            const auto* weights = reinterpret_cast<const uint4*>(w);
            const auto* input   = reinterpret_cast<const uint4*>(pair);
            float sum           = 0.F;
            for (int i = lane; i < a.hidden / 4; i += 32) sum = dot8(input[i], weights[i], sum);
            sum = common::warp_reduce_sum(sum);
            if (lane == 0) a.latent[row * a.rank + k] = __float2bfloat16_rn(sum);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float projected_blend(const Token& token, const Bf16* up, const Bf16* h,
                                                 int hidden, int rank, int d) {
    const int lane = threadIdx.x % 8;
    float gain = 0, forget = 0;
    if (d < hidden) {
        const auto* g = reinterpret_cast<const uint4*>(up + std::int64_t(d) * rank);
        const auto* f = reinterpret_cast<const uint4*>(up + std::int64_t(d + hidden) * rank);
        const auto* v = reinterpret_cast<const uint4*>(h);
        for (int k = lane; k < rank / 8; k += 8) {
            const auto x = v[k];
            gain         = dot8(g[k], x, gain);
            forget       = dot8(f[k], x, forget);
        }
    }
    gain   = common::warp_reduce_sum<8>(gain);
    forget = common::warp_reduce_sum<8>(forget);
    return lane == 0 && d < hidden ? blend(token.dest[d], token.message[d], gain, forget) : 0;
}

template <bool Keep>
__global__ void up_vector(Arguments a) {
    extern __shared__ float blended[];
    __shared__ RmsStats partials[kWarps];
    __shared__ float sums[kWarps];
    for (std::int64_t row = blockIdx.x; row < std::int64_t(a.tokens) * a.streams;
         row += gridDim.x) {
        const Token token(a, row);
        const auto* h = a.latent + row * a.rank;
        const auto* w = a.up + token.stream * (2 * a.hidden) * a.rank;
        RmsStats stats{};
        for (int i = 0; i < a.hidden; i += kThreads / 8) {
            const int d   = i + threadIdx.x / 8;
            const float x = projected_blend(token, w, h, a.hidden, a.rank, d);
            if constexpr (Keep) {
                if (threadIdx.x % 8 == 0 && d < a.hidden) blended[d] = x;
            }
            stats.add(x);
        }
        publish_warp_stats(stats, partials);
        __syncthreads();
        stats = gather_rms_stats<kThreads>(partials);
        float scale;
        if constexpr (Keep) {
            scale = row_inverse_rms<kThreads>(stats, blended, a.hidden, a.norm_eps, sums);
        } else {
            const auto normal = norm_scale(stats, a.hidden, a.norm_eps);
            scale             = normal.inverse;
            if (__builtin_expect(normal.retry_max != 0.F, false)) {
                const RescaleSquares rescale(normal.retry_max);
                float sum = 0;
                for (int i = 0; i < a.hidden; i += kThreads / 8) {
                    const int d = i + threadIdx.x / 8;
                    rescale.add(projected_blend(token, w, h, a.hidden, a.rank, d), sum);
                }
                sum = common::block_reduce_sum<kThreads>(sum, sums);
                if (threadIdx.x == 0)
                    sums[0] = scaled_inverse_rms(sum, a.hidden, a.norm_eps, normal.retry_max);
                __syncthreads();
                scale = sums[0];
            }
        }
        if constexpr (Keep) {
            for (int d = threadIdx.x; d < a.hidden; d += kThreads)
                token.out[d] = __float2bfloat16_rn(blended[d] * scale * float(token.gamma[d]));
        } else {
            for (int i = 0; i < a.hidden; i += kThreads / 8) {
                const int d   = i + threadIdx.x / 8;
                const float x = projected_blend(token, w, h, a.hidden, a.rank, d);
                if (threadIdx.x % 8 == 0 && d < a.hidden)
                    token.out[d] = __float2bfloat16_rn(x * scale * float(token.gamma[d]));
            }
        }
        __syncthreads();
    }
}

constexpr int kDecodeThreads = 128;

__global__ void decode_down(Chunk a) {
    const int rows = a.tokens * a.streams, tid = threadIdx.x;
    const int rank = blockIdx.x % kRank, row = blockIdx.x / kRank % rows;
    const int split = blockIdx.x / kRank / rows;
    const Token token(a, row);
    const auto* w = a.args.down + (token.stream * kRank + rank) * (2 * kHidden) + split * kHidden;
    __shared__ RmsStats stats[kDecodeThreads / 32];
    __shared__ float sums[kDecodeThreads / 32];
    __shared__ __align__(16) Bf16 weights[kHidden];
    for (int i = tid * 8; i < kHidden; i += 8 * kDecodeThreads)
        common::cp_async<16, Cache::cg>(weights + i, w + i);
    common::cp_commit();
    pdl::trigger_dependents();

    constexpr int packs = kHidden / 4, groups = (packs + kDecodeThreads - 1) / kDecodeThreads;
    const auto* source = reinterpret_cast<const float4*>(split == 0 ? token.dest : token.message);
    float4 input[groups];
    RmsStats local{};
#pragma unroll
    for (int i = 0; i < groups; ++i) {
        const int k = i * kDecodeThreads + tid;
        if (k < packs) {
            input[i] = source[k];
            local.add(input[i]);
        }
    }
    publish_warp_stats(local, stats);
    __syncthreads();
    const float norm = row_inverse_rms<kDecodeThreads>(gather_rms_stats<kDecodeThreads>(stats),
                                                       reinterpret_cast<const float*>(source),
                                                       kHidden, a.args.pair_eps, sums);
    common::cp_wait<0>();
    __syncthreads();

    float dp = 0;
#pragma unroll
    for (int i = 0; i < groups; ++i) {
        const int k = i * kDecodeThreads + tid;
        if (k < packs) dp = dot_pair4(input[i], norm, weights + 4 * k, dp);
    }
    const float total = common::block_reduce_sum<kDecodeThreads>(dp, sums);
    if (tid == 0) a.projection[(split * rows + row) * kRank + rank] = total;
}

__global__ void decode_blend(Chunk a) {
    constexpr int Features = kDecodeFeatures, Tiles = kDecodeTiles;
    const int row = blockIdx.x / Tiles, tile = blockIdx.x % Tiles;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int rows = a.tokens * a.streams;
    const Token token(a, row);
    const auto* up = a.args.up + token.stream * (2 * kHidden * kRank) + tile * Features * kRank;
    __shared__ __align__(16) Bf16 weights[2 * Features * kRank], h[kRank];
    __shared__ RmsStats warp_stats[kDecodeThreads / 32];

    for (int i = tid * 8; i < 2 * Features * kRank; i += kDecodeThreads * 8) {
        const int source = i % (Features * kRank) + i / (Features * kRank) * kHidden * kRank;
        common::cp_async<16, Cache::cg>(weights + i, up + source);
    }
    common::cp_commit();
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    if (tid < kRank)
        h[tid] = __float2bfloat16_rn(a.projection[row * kRank + tid] +
                                     a.projection[(rows + row) * kRank + tid]);
    common::cp_wait<0>();
    __syncthreads();

    auto* out = a.blended + row * kHidden;
    RmsStats stats{};
    const int k0 = lane % 4 * 2;
    unsigned hp[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        hp[i][0] = packed(h + i * 16 + k0);
        hp[i][1] = packed(h + i * 16 + k0 + 8);
    }
#pragma unroll
    for (int f = 0; f < Features; f += (kDecodeThreads / 32) * 8) {
        const int local = f + warp * 8 + lane / 4;
        const auto* g   = weights + local * kRank + k0;
        const auto* q   = g + Features * kRank;
        float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
        for (int k = 0; k < kRank; k += 16) {
            common::mma_bf16(c0, c1, c2, c3, packed(g + k), packed(q + k), packed(g + k + 8),
                             packed(q + k + 8), hp[k / 16][0], hp[k / 16][1]);
        }
        if (k0 == 0) {
            const int j   = tile * Features + local;
            const float v = blend(token.dest[j], token.message[j], c0, c2);
            out[j]        = v;
            stats.add(v);
        }
    }
    publish_warp_stats(stats, warp_stats);
    __syncthreads();
    if (tid == 0) {
        RmsStats total{};
#pragma unroll
        for (int i = 0; i < kDecodeThreads / 32; ++i) total.merge(warp_stats[i]);
        a.sums[row * Tiles + tile] = total.pack();
    }
}
} // namespace

void launch_generic(Arguments a, cudaStream_t stream) {
    const auto rows = std::int64_t(a.tokens) * a.streams;
    constexpr int kRowsForSharedPair = 2 * 170;
    if (rows >= kRowsForSharedPair && a.hidden <= 8192) {
        down_vector_prefill<<<grid_size(rows), kThreads, 2 * a.hidden * sizeof(Bf16), stream>>>(a);
    } else {
        down_vector<<<grid_size(rows * (a.rank / kWarps)), kThreads, 0, stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
    if (a.hidden <= 8192)
        up_vector<true><<<grid_size(rows), kThreads, a.hidden * sizeof(float), stream>>>(a);
    else
        up_vector<false><<<grid_size(rows), kThreads, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

void launch_decode(Chunk a, cudaStream_t stream) {
    const int rows = a.tokens * a.streams;
    decode_down<<<rows * kRank * kDecodeSplits, kDecodeThreads, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(pdl::launch_dependent({unsigned(rows * kDecodeTiles), kDecodeThreads, 0, stream},
                                     decode_blend, a));
    launch_normalize(a, stream);
}
} // namespace ninfer::ops::detail::loop_message
