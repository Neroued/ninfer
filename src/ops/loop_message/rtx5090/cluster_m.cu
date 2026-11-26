#include "ops/loop_message/rtx5090/mma_tile.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/convert.cuh"

#include <cooperative_groups.h>

namespace ninfer::ops::detail::loop_message {
namespace {
namespace cg              = cooperative_groups;
constexpr int kWarpTokens = 16;
constexpr int kThreads    = 256;
constexpr int kWarps      = kThreads / 32;
constexpr int kFeatures   = kWarps * 8;
template <int Tokens>
constexpr int kBlockThreads = kThreads * (Tokens / kWarpTokens);
template <int Cluster, int Tokens>
constexpr int kDownTile = Cluster == 8            ? 96
                          : Tokens == kWarpTokens ? 32
                                                  : 64;

__device__ __forceinline__ int feature_warp() { return threadIdx.x / 32 % kWarps; }

__device__ __forceinline__ int token_group() { return threadIdx.x / kThreads * kWarpTokens; }

template <int Cluster>
__device__ __forceinline__ float cluster_sum(cg::cluster_group cluster, const float* local) {
    float sum = 0;
#pragma unroll
    for (int c = 0; c < Cluster; ++c) sum += *cluster.map_shared_rank(local, c);
    return sum;
}

template <int K, int Tokens>
__device__ __forceinline__ void stage_down_weights(Bf16* dst, const Bf16* w, int first) {
#pragma unroll
    for (int i = threadIdx.x; i < kRank * K / 8; i += kBlockThreads<Tokens>) {
        const int r = i / (K / 8), c = i % (K / 8) * 8;
        common::cp_async<16>(dst + swizzle<K>(r, c), w + r * (2 * kHidden) + first + c);
    }
    common::cp_commit();
}

template <int Width, int K, int Tokens>
__device__ __forceinline__ void down_shard(const Bf16* down, const Bf16 (*pair)[Tokens * Width],
                                           Bf16 (*weights)[kRank * K], float (*partial)[kRank],
                                           int part) {
    static_assert(Width % K == 0);
    const int warp = feature_warp(), lane = threadIdx.x % 32;
    float acc[4]{};
#pragma unroll 1
    for (int k = 0; k < 2 * Width; k += K) {
        const bool slot = (k / K) & 1;
        if (k + K < 2 * Width) {
            const int next = k + K;
            stage_down_weights<K, Tokens>(weights[!slot], down,
                                          part * Width +
                                              (next < Width ? next : next - Width + kHidden));
        }
        const int half = k >= Width;
        const int c0   = k - half * Width;
#pragma unroll
        for (int i = 0; i < K; i += 16) {
            unsigned x[4], w[2];
            load_a<Width>(pair[half], token_group(), c0 + i, lane, x);
            load_b<K>(weights[slot], warp * 8, i, lane, w);
            mma(acc, x, w);
        }
        common::cp_wait<0>();
        __syncthreads();
    }
    const int r = warp * 8 + (lane % 4) * 2, t = token_group() + lane / 4;
    *reinterpret_cast<float2*>(partial[t] + r)     = {acc[0], acc[1]};
    *reinterpret_cast<float2*>(partial[t + 8] + r) = {acc[2], acc[3]};
}

template <int Cluster, int Tokens>
__device__ __forceinline__ void gather_latent(cg::cluster_group cluster, int part,
                                              const float (*partial)[kRank], Bf16* shard,
                                              Bf16* latent) {
    constexpr int Rows = Tokens / Cluster;
    const int tid      = threadIdx.x;
    cluster.sync();
#pragma unroll
    for (int i = tid; i < Rows * kRank; i += kBlockThreads<Tokens>)
        shard[i] = __float2bfloat16_rn(
            cluster_sum<Cluster>(cluster, &partial[part * Rows + i / kRank][i % kRank]));
    cluster.sync();
#pragma unroll
    for (int i = tid; i < Tokens * kRank / 8; i += kBlockThreads<Tokens>) {
        const int t = i / (kRank / 8), k = i % (kRank / 8) * 8;
        const auto* src = cluster.map_shared_rank(shard + t % Rows * kRank + k, t / Rows);
        *reinterpret_cast<uint4*>(latent + swizzle<kRank>(t, k)) =
            *reinterpret_cast<const uint4*>(src);
    }
}

template <int Features, int Tokens>
__device__ __forceinline__ void stage_up(Bf16* weights, const Bf16* up, int width, int first,
                                         int part) {
#pragma unroll
    for (int i = threadIdx.x; i < 2 * Features * kRank / 8; i += kBlockThreads<Tokens>) {
        const int r = i / (kRank / 8), k = i % (kRank / 8) * 8, d = first + r % Features;
        if (d < width)
            common::cp_async<16>(weights + swizzle<kRank>(r, k),
                                 up + (part * width + d + r / Features * kHidden) * kRank + k);
    }
    common::cp_commit();
}

__device__ __forceinline__ void gate_logits(const Bf16* weight, const Bf16* hidden, int warp,
                                            int lane, float (&gain)[4], float (&forget)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        unsigned h[4], g[2], f[2];
        load_a<kRank>(hidden, token_group(), k * 16, lane, h);
        load_b<kRank>(weight, warp * 8, k * 16, lane, g);
        load_b<kRank>(weight, kFeatures + warp * 8, k * 16, lane, f);
        mma(gain, h, g);
        mma(forget, h, f);
    }
}

template <int Cluster, int Tokens>
__device__ __forceinline__ void input_scales(cg::cluster_group cluster, const Token& row, int count,
                                             float epsilon, const RmsStats (*pair)[2], float* dest,
                                             float* message) {
    const int tid = threadIdx.x, warp = tid / 32;
    cluster.sync();
    bool retry = false;
    if (tid < Tokens) {
        NormScale scales[2]{};
        if (tid < count) {
            RmsStats sums[2]{};
#pragma unroll
            for (int c = 0; c < Cluster; ++c) {
                const auto* partial = cluster.map_shared_rank(pair[tid], c);
                sums[0].merge(partial[0]);
                sums[1].merge(partial[1]);
            }
            scales[0] = norm_scale(sums[0], kHidden, epsilon);
            scales[1] = norm_scale(sums[1], kHidden, epsilon);
        }
        retry = scales[0].retry_max != 0.F || scales[1].retry_max != 0.F;
        // Inverse RMS is nonnegative. A negative value is a pending retry.
        dest[tid]    = scales[0].retry_max != 0.F ? -scales[0].retry_max : scales[0].inverse;
        message[tid] = scales[1].retry_max != 0.F ? -scales[1].retry_max : scales[1].inverse;
    }
    if (__syncthreads_or(retry)) {
#pragma unroll
        for (int t = warp; t < Tokens; t += kBlockThreads<Tokens> / 32) {
            const auto recover = [&](const float* input, float* scale) {
                const float max_abs = -scale[t];
                if (max_abs > 0.F) {
                    const float sum =
                        reduce_rescaled_row<32>(input + t * kHidden, kHidden, max_abs);
                    if (tid % 32 == 0)
                        scale[t] = scaled_inverse_rms(sum, kHidden, epsilon, max_abs);
                }
            };
            recover(row.dest, dest);
            recover(row.message, message);
        }
        __syncthreads();
    }
}

template <int Cluster, int Tokens, bool Scaled = false>
__device__ __forceinline__ bool output_scales(cg::cluster_group cluster, RmsStats s0, RmsStats s1,
                                              RmsStats (*sums)[Tokens], RmsStats* totals,
                                              float* inverse, float* rescale, float epsilon) {
    const int tid = threadIdx.x, lane = tid % 32, warp = feature_warp();
    s0 = warp_rms_stats<4>(s0);
    s1 = warp_rms_stats<4>(s1);
    if (lane % 4 == 0) {
        sums[warp][token_group() + lane / 4]     = s0;
        sums[warp][token_group() + lane / 4 + 8] = s1;
    }
    __syncthreads();
    if (tid < Tokens) {
        RmsStats stats{};
#pragma unroll
        for (int w = 0; w < kWarps; ++w) stats.merge(sums[w][tid]);
        totals[tid] = stats;
    }
    cluster.sync();
    bool retry = false;
    if (tid < Tokens) {
        RmsStats stats{};
#pragma unroll
        for (int c = 0; c < Cluster; ++c) stats.merge(*cluster.map_shared_rank(totals + tid, c));
        if constexpr (Scaled) {
            if (rescale[tid] != 0.F)
                inverse[tid] = scaled_inverse_rms(stats.sum_sq, kHidden, epsilon, rescale[tid]);
        } else {
            const auto result = norm_scale(stats, kHidden, epsilon);
            rescale[tid]      = result.retry_max;
            inverse[tid]      = result.inverse;
            retry             = result.retry_max != 0.F;
        }
    }
    return __syncthreads_or(retry);
}

// The register layout is the MMA accumulator layout: a quartet of lanes
// owns eight adjacent features for each of two tokens.
template <int Cluster, int Tokens>
struct RegisterBlend {
    static constexpr int Width = kHidden / Cluster;
    static constexpr int Steps = (Width + kFeatures - 1) / kFeatures;
    static_assert(Width % 8 == 0);
    // Reread one feature tile for the wider cluster. Keeping both full-width
    // inputs in registers for 32 tokens would spill the projection.
    static constexpr bool ReloadTail = Cluster == 4 && Tokens > kWarpTokens;
    float2 value[Steps][2];

    __device__ __forceinline__ static float2 load_pair(const float* source, int part, int t, int d,
                                                       int count) {
        return t < count ? *reinterpret_cast<const float2*>(source + t * kHidden + part * Width + d)
                         : float2{};
    }

    __device__ __forceinline__ static int token(int z, int lane) {
        return token_group() + lane / 4 + z * 8;
    }

    __device__ __forceinline__ static int feature(int i, int warp, int lane) {
        return i * kFeatures + warp * 8 + (lane % 4) * 2;
    }

    __device__ __forceinline__ void load(const float* source, int part, int count,
                                         RmsStats (&stats)[2], float* raw) {
        const int tid = threadIdx.x, warp = feature_warp(), lane = tid % 32;
        if constexpr (Cluster == 8) {
            for (int i = tid; i < Tokens * Width / 4; i += kBlockThreads<Tokens>) {
                const int t = i / (Width / 4), d = 4 * (i % (Width / 4));
                const auto* input = t < count ? source + t * kHidden + part * Width + d : source;
                common::cp_async_zfill<16, Cache::cg>(raw + swizzle<Width>(t, d), input,
                                                      t < count ? 16 : 0);
            }
            common::cp_commit();
            common::cp_wait<0>();
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < Steps; ++i) {
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int t = token(z, lane), d = feature(i, warp, lane);
                float2 x{};
                if constexpr (Cluster == 8) {
                    if (d < Width) x = *reinterpret_cast<const float2*>(raw + swizzle<Width>(t, d));
                } else {
                    x = load_pair(source, part, t, d, count);
                }
                if (!ReloadTail || i + 1 < Steps) value[i][z] = x;
                stats[z].add(x);
            }
        }
    }

    __device__ __forceinline__ void pack_pair(const float* scale, Bf16* pair, const float* source,
                                              int part, int count) const {
        const int warp = feature_warp(), lane = threadIdx.x % 32;
#pragma unroll
        for (int z = 0; z < 2; ++z) {
            const int t            = token(z, lane);
            const float multiplier = scale[t];
#pragma unroll
            for (int i = 0; i < Steps; ++i) {
                const int d = feature(i, warp, lane);
                if (d < Width) {
                    const auto x = ReloadTail && i + 1 == Steps
                                       ? load_pair(source, part, t, d, count)
                                       : value[i][z];
                    *reinterpret_cast<__nv_bfloat162*>(pair + swizzle<Width>(t, d)) =
                        __floats2bfloat162_rn(x.x * multiplier, x.y * multiplier);
                }
            }
        }
    }

    __device__ __forceinline__ void apply(cg::cluster_group cluster, const Token& row, int count,
                                          const Bf16* up, const Bf16* latent,
                                          const RegisterBlend& message, Bf16* weight,
                                          RmsStats (*sums)[Tokens], RmsStats* total, float* inverse,
                                          float* rescale, float epsilon) {
        const int part = cluster.block_rank(), lane = threadIdx.x % 32, warp = feature_warp();
        constexpr int TileValues = 2 * kFeatures * kRank;
        stage_up<kFeatures, Tokens>(weight, up, Width, 0, part);
        common::cp_wait<0>();
        __syncthreads();
        RmsStats stats[2]{};
#pragma unroll
        for (int i = 0; i < Steps; ++i) {
            const int slot = i % 2;
            if (i + 1 < Steps)
                stage_up<kFeatures, Tokens>(weight + (slot ^ 1) * TileValues, up, Width,
                                            (i + 1) * kFeatures, part);
            if (feature(i, warp, lane) < Width) {
                float gain[4]{}, forget[4]{};
                gate_logits(weight + slot * TileValues, latent, warp, lane, gain, forget);
#pragma unroll
                for (int z = 0; z < 2; ++z) {
                    const int t = token(z, lane), d = feature(i, warp, lane);
                    const auto x = ReloadTail && i + 1 == Steps
                                       ? load_pair(row.dest, part, t, d, count)
                                       : value[i][z];
                    const auto y = ReloadTail && i + 1 == Steps
                                       ? load_pair(row.message, part, t, d, count)
                                       : message.value[i][z];
                    const float2 v{blend(x.x, y.x, gain[2 * z], forget[2 * z]),
                                   blend(x.y, y.y, gain[2 * z + 1], forget[2 * z + 1])};
                    value[i][z] = v;
                    stats[z].add(v);
                }
            } else {
                value[i][0] = {};
                value[i][1] = {};
            }
            if (i + 1 < Steps) {
                common::cp_wait<0>();
                __syncthreads();
            }
        }
        if (output_scales<Cluster, Tokens>(cluster, stats[0], stats[1], sums, total, inverse,
                                           rescale, epsilon)) {
            cluster.sync();
            RmsStats scaled[2]{};
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const float max_abs = rescale[token(z, lane)];
                if (max_abs > 0.F) {
                    const RescaleSquares scale(max_abs);
#pragma unroll
                    for (int i = 0; i < Steps; ++i) {
                        const auto v = value[i][z];
                        scale.add(v.x, scaled[z].sum_sq);
                        scale.add(v.y, scaled[z].sum_sq);
                    }
                }
            }
            output_scales<Cluster, Tokens, true>(cluster, scaled[0], scaled[1], sums, total,
                                                 inverse, rescale, epsilon);
        }
#pragma unroll
        for (int i = 0; i < Steps; ++i) {
            const int d = feature(i, warp, lane);
            if (d >= Width) continue;
            const auto g = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(row.gamma + part * Width + d));
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int t = token(z, lane);
                if (t < count) {
                    const auto v = value[i][z];
                    *reinterpret_cast<__nv_bfloat162*>(weight + swizzle<Width>(t, d)) =
                        __floats2bfloat162_rn(v.x * inverse[t] * g.x, v.y * inverse[t] * g.y);
                }
            }
        }
        // The up weights are no longer live. Transpose the accumulator layout
        // through the same shared allocation so each warp stores full sectors.
        __syncthreads();
        for (int i = threadIdx.x; i < count * Width / 8; i += kBlockThreads<Tokens>) {
            const int t = i / (Width / 8), d = 8 * (i % (Width / 8));
            const auto packed = *reinterpret_cast<const uint4*>(weight + swizzle<Width>(t, d));
            *reinterpret_cast<uint4*>(row.out + t * kHidden + part * Width + d) = packed;
        }
    }
};

template <int Cluster, int Tokens>
union PhaseStorage {
    static constexpr int Width = kHidden / Cluster;

    struct {
        Bf16 weight[2][kRank * kDownTile<Cluster, Tokens>];
        Bf16 pair[2][Tokens * Width];
    } down;

    float partial[Tokens][kRank];
    Bf16 up[std::max(4 * kFeatures * kRank, Tokens* Width)];
};

template <int Cluster, int Tokens>
__global__ __launch_bounds__(kBlockThreads<Tokens>,
                             kWarpTokens * 2 / Tokens) void __cluster_dims__(Cluster, 1, 1)
    prefill_cluster(Arguments a, int first_group, int first_tile, bool interleave) {
    using Fragment      = RegisterBlend<Cluster, Tokens>;
    constexpr int Width = Fragment::Width;
    const int tid = threadIdx.x, lane = tid % 32;
    auto cluster   = cg::this_cluster();
    const int part = cluster.block_rank(), tiles = (a.tokens - 1) / Tokens + 1;

    extern __shared__ __align__(16) unsigned char storage[];
    auto& work = *reinterpret_cast<PhaseStorage<Cluster, Tokens>*>(storage);

    // Neighboring lanes reduce neighboring tokens, so keep those stats adjacent.
    union Statistics {
        struct {
            RmsStats warp[kWarps][Tokens][2], part[Tokens][2];
        } input;

        struct {
            RmsStats warp[kWarps][Tokens], part[Tokens];
        } output;
    };

    __shared__ Statistics stats;
    __shared__ float ds[Tokens], ms[Tokens], rescale[Tokens];
    __shared__ __align__(16) Bf16 latent[Tokens * kRank], shard[Tokens * kRank / Cluster];

    const unsigned tile = first_tile + blockIdx.x / Cluster;
    const int m = (interleave ? first_group + int(tile / a.streams) : int(tile % tiles)) * Tokens;
    const int s = interleave ? int(tile % a.streams) : first_group + int(tile / tiles);
    const int count = min(Tokens, a.tokens - m);
    const Token token(a, s, m);
    const auto* dw = a.down + token.stream * (2 * kHidden * kRank);
    const auto* uw = a.up + token.stream * (2 * kHidden * kRank);
    stage_down_weights<kDownTile<Cluster, Tokens>, Tokens>(work.down.weight[0], dw, part * Width);

    // Keep the original FP32 activations for blend and materialize each
    // normalized BF16 value once for all the rank warps.
    Fragment dest, message;
    RmsStats dest_stats[2]{}, message_stats[2]{};
    // The eight-CTA path stages each FP32 operand in the BF16 pair's
    // allocation before either half of the pair is produced.
    auto* raw = reinterpret_cast<float*>(work.down.pair);
    dest.load(token.dest, part, count, dest_stats, raw);
    __syncthreads();
    message.load(token.message, part, count, message_stats, raw);
#pragma unroll
    for (int z = 0; z < 2; ++z) {
        const auto sd = warp_rms_stats<4>(dest_stats[z]);
        const auto sm = warp_rms_stats<4>(message_stats[z]);
        if (lane % 4 == 0) {
            auto& partial = stats.input.warp[feature_warp()][Fragment::token(z, lane)];
            partial[0]    = sd;
            partial[1]    = sm;
        }
    }
    __syncthreads();
    if (tid < Tokens) {
        RmsStats sd{}, sm{};
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            sd.merge(stats.input.warp[w][tid][0]);
            sm.merge(stats.input.warp[w][tid][1]);
        }
        stats.input.part[tid][0] = sd;
        stats.input.part[tid][1] = sm;
    }
    input_scales<Cluster, Tokens>(cluster, token, count, a.pair_eps, stats.input.part, ds, ms);
    dest.pack_pair(ds, work.down.pair[0], token.dest, part, count);
    message.pack_pair(ms, work.down.pair[1], token.message, part, count);
    common::cp_wait<0>();
    __syncthreads();
    down_shard<Width, kDownTile<Cluster, Tokens>, Tokens>(dw, work.down.pair, work.down.weight,
                                                          work.partial, part);
    gather_latent<Cluster, Tokens>(cluster, part, work.partial, shard, latent);
    __syncthreads();
    dest.apply(cluster, token, count, uw, latent, message, work.up, stats.output.warp,
               stats.output.part, ms, rescale, a.norm_eps);
    cluster.sync();
}

template <int Cluster, int Tokens = kWarpTokens>
void launch_cluster(Arguments a, cudaStream_t stream) {
    constexpr auto kernel = prefill_cluster<Cluster, Tokens>;
    constexpr int bytes   = sizeof(PhaseStorage<Cluster, Tokens>);
    const auto tiles      = std::int64_t((a.tokens - 1) / Tokens + 1) * a.streams;
    // Schedule consumers of the same destination tile together while it is hot in L2.
    const bool interleave     = a.broadcast_dest && a.streams > 1;
    const int tiles_per_group = interleave ? a.streams : (a.tokens - 1) / Tokens + 1;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    for (std::int64_t first = 0; first < tiles;) {
        const auto blocks = grid_size(tiles - first, Cluster);
        kernel<<<blocks, kBlockThreads<Tokens>, bytes, stream>>>(
            a, int(first / tiles_per_group), int(first % tiles_per_group), interleave);
        CUDA_CHECK(cudaGetLastError());
        first += blocks / Cluster;
    }
}

} // namespace

void launch_cluster_prefill(Arguments a, Route route, cudaStream_t stream) {
    if (route == Route::ClusterWide)
        launch_cluster<4, 2 * kWarpTokens>(a, stream);
    else if (route == Route::Cluster4)
        launch_cluster<4>(a, stream);
    else
        launch_cluster<8>(a, stream);
}

} // namespace ninfer::ops::detail::loop_message
