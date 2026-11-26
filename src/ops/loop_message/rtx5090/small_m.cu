#include "ops/loop_message/rtx5090/kernel.cuh"
#include "ops/common/cuda/mma.cuh"
#include "core/pdl.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/convert.cuh"

#include <cuda/atomic>

namespace ninfer::ops::detail::loop_message {
namespace {
constexpr int kThreads = 256;
constexpr int kWarps   = kThreads / 32;

template <int FixedWidth = 0>
__device__ __forceinline__ float2 pair_scales(const Token& row, int width, float eps, float* smem) {
    float ds = 0, ms = 0;
    const int count = (FixedWidth ? FixedWidth : width) / 4;
    const auto* d   = reinterpret_cast<const float4*>(row.dest);
    const auto* m   = reinterpret_cast<const float4*>(row.message);
    for (int i = threadIdx.x; i < count; i += kThreads) {
        ds = square_sum(d[i], ds);
        ms = square_sum(m[i], ms);
    }
    ds = common::warp_reduce_sum(ds);
    ms = common::warp_reduce_sum(ms);
    if ((threadIdx.x & 31) == 0) {
        smem[threadIdx.x / 32]          = ds;
        smem[kWarps + threadIdx.x / 32] = ms;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    ds             = common::warp_sum<kWarps>(smem[lane % kWarps]);
    ms             = common::warp_sum<kWarps>(smem[kWarps + lane % kWarps]);
    return {row_inverse_rms(ds, row.dest, width, eps),
            row_inverse_rms(ms, row.message, width, eps)};
}

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

__global__ void down_vector(Arguments a) {
    __shared__ float scales[2 * kWarps];
    const int lane            = threadIdx.x % 32;
    const std::int64_t groups = (std::int64_t(a.rank) + kWarps - 1) / kWarps;
    for (std::int64_t block = blockIdx.x; block < std::int64_t(a.tokens) * a.streams * groups;
         block += gridDim.x) {
        const auto row = block / groups;
        const Token token(a, row);
        const int k      = int(block % groups) * kWarps + threadIdx.x / 32;
        const auto scale = pair_scales(token, a.hidden, a.pair_eps, scales);
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
    __shared__ float sums[kWarps], scale;
    for (std::int64_t row = blockIdx.x; row < std::int64_t(a.tokens) * a.streams;
         row += gridDim.x) {
        const Token token(a, row);
        const auto* h = a.latent + row * a.rank;
        const auto* w = a.up + token.stream * (2 * a.hidden) * a.rank;
        float squares = 0;
        for (int i = 0; i < a.hidden; i += kThreads / 8) {
            const int d   = i + threadIdx.x / 8;
            const float x = projected_blend(token, w, h, a.hidden, a.rank, d);
            if constexpr (Keep) {
                if (threadIdx.x % 8 == 0 && d < a.hidden) blended[d] = x;
            }
            squares = fmaf(x, x, squares);
        }
        squares = common::block_reduce_sum<kThreads>(squares, sums);
        if constexpr (Keep) {
            if (threadIdx.x == 0) scale = row_inverse_rms(squares, blended, a.hidden, a.norm_eps);
            __syncthreads();
        } else {
            const bool retry = threadIdx.x == 0 && lost_squares(squares);
            if (retry) scale = square_rescale(squares);
            if (__syncthreads_or(retry)) {
                const float factor = scale;
                float scaled       = 0;
                for (int i = 0; i < a.hidden; i += kThreads / 8) {
                    const int d   = i + threadIdx.x / 8;
                    const float x = projected_blend(token, w, h, a.hidden, a.rank, d) * factor;
                    scaled        = fmaf(x, x, scaled);
                }
                scaled = common::block_reduce_sum<kThreads>(scaled, sums);
                if (threadIdx.x == 0)
                    scale = scaled_inverse_rms(scaled, a.hidden, a.norm_eps, factor);
            } else if (threadIdx.x == 0) {
                scale = inverse_rms(squares, a.hidden, a.norm_eps);
            }
            __syncthreads();
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

template <bool InitializeCounter>
__global__ void decode_down(Chunk a) {
    const int rows = a.tokens * a.streams, tid = threadIdx.x;
    const int rank = blockIdx.x % kRank, row = blockIdx.x / kRank % rows;
    const int split = blockIdx.x / kRank / rows;
    const Token token(a, row);
    const auto* w = a.args.down + (token.stream * kRank + rank) * (2 * kHidden) + split * kHidden;
    __shared__ float sums[kDecodeThreads / 32], norm;
    __shared__ __align__(16) Bf16 weights[kHidden];
    if constexpr (InitializeCounter) {
        if (rank == 0 && split == 0 && tid == 0) a.done[row] = 0;
    }

    for (int i = tid * 8; i < kHidden; i += 8 * kDecodeThreads)
        common::cp_async<16, Cache::cg>(weights + i, w + i);
    common::cp_commit();
    pdl::trigger_dependents();

    constexpr int packs = kHidden / 4, groups = (packs + kDecodeThreads - 1) / kDecodeThreads;
    const auto* source = reinterpret_cast<const float4*>(split == 0 ? token.dest : token.message);
    float4 input[groups];
    float ss = 0;
#pragma unroll
    for (int i = 0; i < groups; ++i) {
        const int k = i * kDecodeThreads + tid;
        if (k < packs) {
            input[i] = source[k];
            ss       = square_sum(input[i], ss);
        }
    }
    ss = common::block_reduce_sum<kDecodeThreads>(ss, sums);
    if (tid == 0)
        norm =
            row_inverse_rms(ss, reinterpret_cast<const float*>(source), kHidden, a.args.pair_eps);
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

template <bool Separate>
__global__ void decode_blend(Chunk a) {
    constexpr int Features = kDecodeFeatures, Tiles = kDecodeTiles;
    const int row = blockIdx.x / Tiles, tile = blockIdx.x % Tiles;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int rows = a.tokens * a.streams;
    const Token token(a, row);
    const auto* up = a.args.up + token.stream * (2 * kHidden * kRank) + tile * Features * kRank;
    __shared__ __align__(16) Bf16 weights[2 * Features * kRank], h[kRank];
    __shared__ float warp_sums[kDecodeThreads / 32], inverse;
    __shared__ bool last;

    for (int i = tid * 8; i < 2 * Features * kRank; i += kDecodeThreads * 8) {
        const int source = i % (Features * kRank) + i / (Features * kRank) * kHidden * kRank;
        common::cp_async<16, Cache::cg>(weights + i, up + source);
    }
    common::cp_commit();
    if constexpr (Separate) pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    if (tid < kRank)
        h[tid] = __float2bfloat16_rn(a.projection[row * kRank + tid] +
                                     a.projection[(rows + row) * kRank + tid]);
    common::cp_wait<0>();
    __syncthreads();

    auto* out    = a.blended + row * kHidden;
    float ss     = 0;
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
            ss            = fmaf(v, v, ss);
        }
    }
    ss = common::block_reduce_sum<kDecodeThreads>(ss, warp_sums);
    if (tid == 0) a.sums[row * Tiles + tile] = ss;
    if constexpr (!Separate) {
        // The release sequence publishes every tile and its sum to the last CTA.
        // Stage one initializes the counter on every eager call and graph replay.
        __syncthreads();
        if (tid == 0) {
            cuda::atomic_ref<unsigned, cuda::thread_scope_device> arrivals(a.done[row]);
            last = arrivals.fetch_add(1, cuda::memory_order_acq_rel) == Tiles - 1;
        }
        __syncthreads();
        if (!last) return;
        if (tid < 32) {
            float sum = 0;
#pragma unroll
            for (int j = tid; j < Tiles; j += 32) sum += a.sums[row * Tiles + j];
            sum = common::warp_reduce_sum(sum);
            if (tid == 0) inverse = row_inverse_rms(sum, out, kHidden, a.args.norm_eps);
        }
        __syncthreads();
        for (int d = tid * 2; d < kHidden; d += kDecodeThreads * 2) {
            const auto x = *reinterpret_cast<const float2*>(out + d);
            const auto g =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(token.gamma + d));
            *reinterpret_cast<__nv_bfloat162*>(token.out + d) =
                __floats2bfloat162_rn((x.x * inverse) * g.x, (x.y * inverse) * g.y);
        }
    }
}
} // namespace

void launch_generic(Arguments a, cudaStream_t stream) {
    const auto rows = std::int64_t(a.tokens) * a.streams;
    down_vector<<<grid_size(rows * (a.rank / kWarps)), kThreads, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
    if (a.hidden <= 8192)
        up_vector<true><<<grid_size(rows), kThreads, a.hidden * sizeof(float), stream>>>(a);
    else
        up_vector<false><<<grid_size(rows), kThreads, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

void launch_decode(Chunk a, Route route, cudaStream_t stream) {
    const int rows   = a.tokens * a.streams;
    const bool last  = route == Route::DecodeLast;
    const auto down  = last ? decode_down<true> : decode_down<false>;
    const auto blend = last ? decode_blend<false> : decode_blend<true>;
    down<<<rows * kRank * kDecodeSplits, kDecodeThreads, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(pdl::launch_dependent({unsigned(rows * kDecodeTiles), kDecodeThreads, 0, stream},
                                     blend, a));
    if (!last) launch_normalize(a, stream);
}
} // namespace ninfer::ops::detail::loop_message
