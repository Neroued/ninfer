#include "ops/loop_message/rtx5090/kernel.cuh"
#include "ops/loop_message/rtx5090/mma_tile.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "core/pdl.cuh"

#include <cuda/atomic>

namespace ninfer::ops::detail::loop_message {
namespace {
constexpr int kPrefillThreads   = 256;
// One warp per row performs the medium tile's RMS scan. Only the first
// eight warps issue MMA, while the full block stages its input and weights.
constexpr int kMediumDownThreads = 512;
constexpr int kPrefillWarps     = kPrefillThreads / 32;
constexpr int kNormalizeThreads = 256;
constexpr int kDownK            = 128;

template <int CounterPlanes>
__global__ void pair_inverses(Chunk a) {
    constexpr int Threads = 128;
    const Token token(a, blockIdx.x);
    pdl::trigger_dependents();
    __shared__ RmsStats stats[2][Threads / 32];
    __shared__ float retry[2][Threads / 32];
    const auto inverse = input_scales<Threads>(token, kHidden, a.args.pair_eps, stats, retry);
    pdl::wait_for_dependencies();
    if (threadIdx.x == 0) {
        a.inv[blockIdx.x] = inverse;
#pragma unroll
        for (int plane = 0; plane < CounterPlanes; ++plane)
            a.done[plane * a.tokens * a.streams + blockIdx.x] = 0;
    }
}

template <int Tokens, int Splits, bool Stats, int Ranks>
__global__ __launch_bounds__(Stats ? kPrefillThreads
                                   : kMediumDownThreads) void tiled_down(Chunk a) {
    constexpr int Threads  = Stats ? kPrefillThreads : kMediumDownThreads;
    constexpr int RowWarps = Threads / 32;
    constexpr int K    = kDownK;
    constexpr int Half = kHidden / K, HalfSplits = Splits > 1 ? Splits / 2 : 1;
    static_assert(kHidden % K == 0 && Tokens % 16 == 0 && (Stats || Splits % 2 == 0));
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    constexpr int RankTiles = kRank / Ranks, RankWarps = Ranks / 8;
    constexpr int WarpTokens = 16 * (kPrefillWarps / RankWarps);
    static_assert(Tokens % WarpTokens == 0);
    const int groups = (a.tokens + Tokens - 1) / Tokens;
    const int block = blockIdx.x / RankTiles, rank_tile = blockIdx.x % RankTiles;
    const int group = block % groups, stream = block / groups % a.streams;
    const int split = block / groups / a.streams;
    const int first = group * Tokens, count = min(Tokens, a.tokens - first);
    const int row = stream * a.tokens + first, rows = a.streams * a.tokens;
    const Token token(a, row);
    const auto* down =
        a.args.down + token.stream * (2 * kHidden * kRank) + rank_tile * Ranks * (2 * kHidden);
    const int begin =
        Splits == 1 ? 0
                    : (split / HalfSplits * Half + (split % HalfSplits) * Half / HalfSplits) * K;
    const int end =
        Splits == 1
            ? 2 * kHidden
            : (split / HalfSplits * Half + (split % HalfSplits + 1) * Half / HalfSplits) * K;
    const int inner = end - begin;
    __shared__ __align__(16) Bf16 w[2][Ranks * K], pair[Tokens * K];
    __shared__ float2 inverse[Tokens];

    const auto stage_weights = [&](int slot, int kk) {
#pragma unroll
        for (int i = tid; i < Ranks * K / 8; i += Threads) {
            const int r = i / (K / 8), c = (i % (K / 8)) * 8;
            common::cp_async<16>(w[slot] + swizzle<K>(r, c), down + r * (2 * kHidden) + kk + c);
        }
        common::cp_commit();
    };
    stage_weights(0, begin);
    pdl::trigger_dependents();
    if constexpr (Stats) pdl::wait_for_dependencies();
    if constexpr (Stats) {
        if (tid < Tokens) inverse[tid] = tid < count ? a.inv[row + tid] : float2{};
    } else {
        const bool dest    = begin < kHidden;
        const auto* source = dest ? token.dest : token.message;
#pragma unroll
        for (int t = warp; t < Tokens; t += RowWarps) {
            RmsStats first{}, second{};
            const auto* row_source = t < count ? source + t * kHidden : nullptr;
            if (row_source) {
                const auto* v = reinterpret_cast<const float4*>(row_source);
#pragma unroll 3
                for (int i = lane; i < kHidden / 4; i += 64) {
                    first.add(v[i]);
                    second.add(v[i + 32]);
                }
            }
            first.merge(second);
            first.sum_sq     = common::warp_sum(first.sum_sq);
            first.max_abs    = warp_max_abs(first.max_abs);
            const float norm = row_inverse_rms<32>(first, row_source, kHidden, a.args.pair_eps);
            if (lane == 0) inverse[t] = {norm, norm};
        }
    }
    common::cp_wait<0>();
    __syncthreads();
    float acc[Tokens / WarpTokens][4]{};
#pragma unroll 1
    for (int i = 0; i < inner; i += K) {
        const int slot = (i / K) & 1, kk = begin + i;
        if (i + K < inner) stage_weights(slot ^ 1, kk + K);
        const bool dest = kk < kHidden;
        const auto* x   = (dest ? token.dest : token.message) + kk % kHidden;
#pragma unroll
        for (int v = tid; v < Tokens * K / 8; v += Threads) {
            const int t = v / (K / 8), c = (v % (K / 8)) * 8;
            float4 lo{}, hi{};
            if (t < count) {
                const auto* src = reinterpret_cast<const float4*>(x + t * kHidden + c);
                lo              = src[0];
                hi              = src[1];
            }
            const float norm = dest ? inverse[t].x : inverse[t].y;
            auto* dst        = reinterpret_cast<__nv_bfloat162*>(pair + swizzle<K>(t, c));
            dst[0]           = __floats2bfloat162_rn(lo.x * norm, lo.y * norm);
            dst[1]           = __floats2bfloat162_rn(lo.z * norm, lo.w * norm);
            dst[2]           = __floats2bfloat162_rn(hi.x * norm, hi.y * norm);
            dst[3]           = __floats2bfloat162_rn(hi.z * norm, hi.w * norm);
        }
        __syncthreads();
        if (warp < kPrefillWarps) {
#pragma unroll
        for (int j = 0; j < K; j += 16) {
            unsigned wf[2];
            load_b<K>(w[slot], warp % RankWarps * 8, j, lane, wf);
#pragma unroll
            for (int t = 0; t < Tokens / WarpTokens; ++t) {
                unsigned xf[4];
                load_a<K>(pair, t * WarpTokens + warp / RankWarps * 16, j, lane, xf);
                mma(acc[t], xf, wf);
            }
        }
        }
        common::cp_wait<0>();
        __syncthreads();
    }
    if constexpr (!Stats) {
        if (warp >= kPrefillWarps) return;
    }
    const int r = rank_tile * Ranks + warp % RankWarps * 8 + (lane & 3) * 2;
#pragma unroll
    for (int t = 0; t < Tokens / WarpTokens; ++t) {
#pragma unroll
        for (int z = 0; z < 2; ++z) {
            const int m = t * WarpTokens + warp / RankWarps * 16 + lane / 4 + z * 8;
            if (m < count) {
                *reinterpret_cast<float2*>(a.projection + (split * rows + row + m) * kRank +
                                           r) = {acc[t][2 * z], acc[t][2 * z + 1]};
            }
        }
    }
    if constexpr (Stats) {
        // Each split publishes disjoint FP32 partials before arrival. The last
        // CTA reduces them in split order and publishes the rounded BF16 latent.
        __shared__ bool last;
        __syncthreads();
        if (tid == 0) {
            cuda::atomic_ref<unsigned, cuda::thread_scope_device> arrivals(
                a.done[rank_tile * rows + row]);
            last = arrivals.fetch_add(1, cuda::memory_order_acq_rel) == Splits - 1;
        }
        __syncthreads();
        if (last) {
            for (int i = tid * 2; i < count * Ranks; i += 2 * kPrefillThreads) {
                const int j = i / Ranks * kRank + rank_tile * Ranks + i % Ranks;
                float2 sum{};
#pragma unroll
                for (int k = 0; k < Splits; ++k) {
                    const auto v = *reinterpret_cast<const float2*>(a.projection +
                                                                    (k * rows + row) * kRank + j);
                    sum.x += v.x;
                    sum.y += v.y;
                }
                *reinterpret_cast<__nv_bfloat162*>(a.args.latent + row * kRank + j) =
                    __floats2bfloat162_rn(sum.x, sum.y);
            }
        }
    }
}

template <int Tokens, int Splits>
__global__ void tiled_up(Chunk a) {
    constexpr int Features = kUpFeatures, Slots = Tokens / 16;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int groups = (a.tokens + Tokens - 1) / Tokens;
    const int tile = blockIdx.x % kUpTiles, group = blockIdx.x / kUpTiles % groups;
    const int stream = blockIdx.x / kUpTiles / groups;
    const int first = group * Tokens, count = min(Tokens, a.tokens - first);
    const int row = stream * a.tokens + first, rows = a.streams * a.tokens;
    const int feature = tile * Features;
    const Token token(a, row);
    const auto* up = a.args.up + token.stream * (2 * kRank * kHidden) + feature * kRank;
    __shared__ __align__(16) Bf16 w[2 * Features * kRank], h[Tokens * kRank];
    __shared__ RmsStats statistics[Tokens][kPrefillWarps];
#pragma unroll
    for (int i = tid; i < 2 * Features * kRank / 8; i += kPrefillThreads) {
        const int r = i / (kRank / 8), k = (i % (kRank / 8)) * 8;
        const int d = r % Features + r / Features * kHidden;
        common::cp_async<16>(w + swizzle<kRank>(r, k), up + d * kRank + k);
    }
    common::cp_commit();
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    if constexpr (Splits == 1) {
#pragma unroll
        for (int i = tid; i < Tokens * kRank / 8; i += kPrefillThreads) {
            const int t = i / (kRank / 8), k = (i % (kRank / 8)) * 8;
            const auto* latent = a.args.latent;
            *reinterpret_cast<uint4*>(h + swizzle<kRank>(t, k)) =
                t < count ? *reinterpret_cast<const uint4*>(latent + (row + t) * kRank + k)
                          : uint4{};
        }
    } else {
#pragma unroll
        for (int i = tid; i < Tokens * kRank / 2; i += kPrefillThreads) {
            const int t = i / (kRank / 2), k = (i % (kRank / 2)) * 2;
            float2 sum{};
            if (t < count) {
#pragma unroll
                for (int s = 0; s < Splits; ++s) {
                    const auto v = *reinterpret_cast<const float2*>(
                        a.projection + (s * rows + row + t) * kRank + k);
                    sum.x += v.x;
                    sum.y += v.y;
                }
            }
            *reinterpret_cast<__nv_bfloat162*>(h + swizzle<kRank>(t, k)) =
                __floats2bfloat162_rn(sum.x, sum.y);
        }
    }
    common::cp_wait<0>();
    __syncthreads();
    float gain[Slots][4]{}, forget[Slots][4]{};
#pragma unroll
    for (int k = 0; k < kRank; k += 16) {
        unsigned g[2], f[2];
        load_b<kRank>(w, warp * 8, k, lane, g);
        load_b<kRank>(w, Features + warp * 8, k, lane, f);
#pragma unroll
        for (int t = 0; t < Slots; ++t) {
            unsigned x[4];
            load_a<kRank>(h, t * 16, k, lane, x);
            mma(gain[t], x, g);
            mma(forget[t], x, f);
        }
    }
    const int d = feature + warp * 8 + (lane & 3) * 2;
#pragma unroll
    for (int t = 0; t < Slots; ++t) {
#pragma unroll
        for (int z = 0; z < 2; ++z) {
            const int m  = t * 16 + lane / 4 + z * 8;
            RmsStats stats{};
            if (m < count) {
                const auto dst = *reinterpret_cast<const float2*>(token.dest + m * kHidden + d);
                const auto msg = *reinterpret_cast<const float2*>(token.message + m * kHidden + d);
                const float x  = blend(dst.x, msg.x, gain[t][2 * z], forget[t][2 * z]);
                const float y  = blend(dst.y, msg.y, gain[t][2 * z + 1], forget[t][2 * z + 1]);
                *reinterpret_cast<float2*>(a.blended + (row + m) * kHidden + d) = {x, y};
                stats.add(float2{x, y});
            }
            stats = warp_rms_stats<4>(stats);
            if (m < count && lane % 4 == 0) statistics[m][warp] = stats;
        }
    }
    __syncthreads();
    if (tid < count) {
        RmsStats stats{};
#pragma unroll
        for (int w = 0; w < kPrefillWarps; ++w) stats.merge(statistics[tid][w]);
        a.sums[(row + tid) * kUpTiles + tile] = stats.pack();
    }
}

__global__ void normalize_chunk(Chunk a) {
    constexpr int Threads = kNormalizeThreads;
    const int Tiles       = a.sum_tiles;
    const int row = blockIdx.x, tid = threadIdx.x;
    const Token token(a, row);
    __shared__ float scale;
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    if (tid < 32) {
        const float inverse = tile_inverse_rms(a.sums + row * Tiles, Tiles,
                                               a.blended + row * kHidden, kHidden, a.args.norm_eps);
        if (tid == 0) scale = inverse;
    }
    __syncthreads();
    const auto* blended = reinterpret_cast<const float4*>(a.blended + row * kHidden);
    for (int i = tid; i < kHidden / 4; i += Threads) {
        const auto v      = blended[i];
        const auto* gamma = reinterpret_cast<const __nv_bfloat162*>(token.gamma) + i * 2;
        const auto g0 = __bfloat1622float2(gamma[0]), g1 = __bfloat1622float2(gamma[1]);
        auto* out = reinterpret_cast<__nv_bfloat162*>(token.out) + i * 2;
        out[0]    = __floats2bfloat162_rn(v.x * scale * g0.x, v.y * scale * g0.y);
        out[1]    = __floats2bfloat162_rn(v.z * scale * g1.x, v.w * scale * g1.y);
    }
}

template <int Tokens, int Splits, bool Stats, int Ranks = kRank>
void run_prefill(Chunk a, cudaStream_t stream) {
    constexpr auto down        = tiled_down<Tokens, Splits, Stats, Ranks>;
    constexpr auto up          = tiled_up < Tokens, Stats ? 1 : Splits > ;
    const unsigned tiles       = (a.tokens + Tokens - 1) / Tokens * a.streams;
    const unsigned down_blocks = tiles * Splits * (kRank / Ranks);
    if constexpr (Stats) {
        constexpr auto inv  = pair_inverses<kRank / Ranks>;
        const unsigned rows = a.tokens * a.streams;
        // Later chunks may read inputs while the previous output is finishing.
        // Publication of the new inverse RMS values follows the stream dependency.
        if (a.first_token || a.first_stream)
            CUDA_CHECK(pdl::launch_dependent({rows, 128, 0, stream}, inv, a));
        else {
            inv<<<rows, 128, 0, stream>>>(a);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(pdl::launch_dependent({down_blocks, kPrefillThreads, 0, stream}, down, a));
    } else {
        down<<<down_blocks, kMediumDownThreads, 0, stream>>>(a);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(pdl::launch_dependent({tiles * kUpTiles, kPrefillThreads, 0, stream}, up, a));
}
} // namespace

void launch_normalize(Chunk a, cudaStream_t stream) {
    const auto rows = unsigned(a.tokens * a.streams);
    CUDA_CHECK(pdl::launch_dependent({rows, kNormalizeThreads, 0, stream}, normalize_chunk, a));
}

void launch_prefill(Chunk a, Route route, cudaStream_t stream) {
    if (route == Route::Medium)
        run_prefill<16, kMediumSplits, false>(a, stream);
    else
        run_prefill<32, kShortSplits, true, kShortRankTile>(a, stream);
    launch_normalize(a, stream);
}
} // namespace ninfer::ops::detail::loop_message
