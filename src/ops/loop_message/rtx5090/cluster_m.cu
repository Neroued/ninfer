#include "ops/loop_message/rtx5090/mma_tile.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/convert.cuh"

#include <cooperative_groups.h>

namespace ninfer::ops::detail::loop_message {
namespace {
namespace cg            = cooperative_groups;
constexpr int kTokens   = 16;
constexpr int kThreads  = 256;
constexpr int kWarps    = kThreads / 32;
constexpr int kFeatures = kWarps * 8;

template <int Cluster>
__device__ __forceinline__ float cluster_sum(cg::cluster_group cluster, const float* local) {
    float sum = 0;
#pragma unroll
    for (int c = 0; c < Cluster; ++c) sum += *cluster.map_shared_rank(local, c);
    return sum;
}

template <int K>
__device__ __forceinline__ void stage_down_weights(Bf16* dst, const Bf16* w, int first) {
#pragma unroll
    for (int i = threadIdx.x; i < kRank * K / 8; i += kThreads) {
        const int r = i / (K / 8), c = i % (K / 8) * 8;
        common::cp_async<16>(dst + swizzle<K>(r, c), w + r * (2 * kHidden) + first + c);
    }
    common::cp_commit();
}

template <int Cluster, int K, bool Direct>
__device__ __forceinline__ void
down_shard(const Bf16* down, const float (*dest)[kHidden / Cluster],
           const float (*message)[kHidden / Cluster], const float* dest_scale,
           const float* message_scale, Bf16* input_tile, Bf16* weights, Bf16* second_weights,
           float (*partial)[kRank], int part, const Token* token = nullptr, int count = 0) {
    constexpr int Width = kHidden / Cluster;
    static_assert(Width % K == 0);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    float a[4]{};
#pragma unroll 1
    for (int k = 0; k < 2 * Width; k += K) {
        const bool slot = (k / K) & 1;
        const auto* w   = slot ? second_weights : weights;
        if (k + K < 2 * Width) {
            const int next = k + K;
            stage_down_weights<K>(slot ? weights : second_weights, down,
                                  part * Width + (next < Width ? next : next - Width + kHidden));
        }
        const bool first  = k < Width;
        const int c0      = first ? k : k - Width;
        const auto* scale = first ? dest_scale : message_scale;
        if constexpr (!Direct) {
#pragma unroll
            for (int i = tid; i < kTokens * K / 4; i += kThreads) {
                const int r = i / (K / 4), c = i % (K / 4) * 4;
                float4 v{};
                if (r < count) {
                    const auto* row = first ? token->dest : token->message;
                    v = *reinterpret_cast<const float4*>(row + r * kHidden + part * Width + c0 + c);
                }
                auto* p       = reinterpret_cast<__nv_bfloat162*>(input_tile + swizzle<K>(r, c));
                const float f = scale[r];
                p[0]          = __floats2bfloat162_rn(v.x * f, v.y * f);
                p[1]          = __floats2bfloat162_rn(v.z * f, v.w * f);
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < K; i += 16) {
            unsigned x[4], y[2];
            if constexpr (Direct) {
#pragma unroll
                for (int z = 0; z < 4; ++z) {
                    const int r     = lane / 4 + (z & 1) * 8;
                    const int c     = c0 + i + (z >> 1) * 8 + (lane % 4) * 2;
                    const auto* src = first ? dest : message;
                    const auto v = *reinterpret_cast<const float2*>(src[r] + (c ^ ((r & 3) * 8)));
                    x[z]         = common::pack_bf16x2(v.x * scale[r], v.y * scale[r]);
                }
            } else {
                load_a<K>(input_tile, 0, i, lane, x);
            }
            load_b<K>(w, warp * 8, i, lane, y);
            mma(a, x, y);
        }
        common::cp_wait<0>();
        __syncthreads();
    }
    const int r = warp * 8 + (lane % 4) * 2, t = lane / 4;
    *reinterpret_cast<float2*>(partial[t] + r)     = {a[0], a[1]};
    *reinterpret_cast<float2*>(partial[t + 8] + r) = {a[2], a[3]};
}

template <int Cluster>
__device__ __forceinline__ void gather_latent(cg::cluster_group cluster, int part,
                                              const float (*partial)[kRank], Bf16* shard,
                                              Bf16* latent) {
    constexpr int Rows = kTokens / Cluster;
    const int tid      = threadIdx.x;
    cluster.sync();
#pragma unroll
    for (int i = tid; i < Rows * kRank; i += kThreads)
        shard[i] = __float2bfloat16_rn(
            cluster_sum<Cluster>(cluster, &partial[part * Rows + i / kRank][i % kRank]));
    cluster.sync();
#pragma unroll
    for (int i = tid; i < kTokens * kRank / 8; i += kThreads) {
        const int t = i / (kRank / 8), k = i % (kRank / 8) * 8;
        const auto* src = cluster.map_shared_rank(shard + t % Rows * kRank + k, t / Rows);
        *reinterpret_cast<uint4*>(latent + swizzle<kRank>(t, k)) =
            *reinterpret_cast<const uint4*>(src);
    }
}

template <int Features>
__device__ __forceinline__ void stage_up(Bf16* weights, const Bf16* up, int width, int first,
                                         int part) {
#pragma unroll
    for (int i = threadIdx.x; i < 2 * Features * kRank / 8; i += kThreads) {
        const int r = i / (kRank / 8), k = i % (kRank / 8) * 8, d = first + r % Features;
        if (d < width)
            common::cp_async<16>(weights + swizzle<kRank>(r, k),
                                 up + (part * width + d + r / Features * kHidden) * kRank + k);
    }
    common::cp_commit();
}

__device__ __forceinline__ void gate_logits(const Bf16* weight, const unsigned (&hidden)[4][4],
                                            int warp, int lane, float (&gain)[4],
                                            float (&forget)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        unsigned g[2], f[2];
        load_b<kRank>(weight, warp * 8, k * 16, lane, g);
        load_b<kRank>(weight, kFeatures + warp * 8, k * 16, lane, f);
        mma(gain, hidden[k], g);
        mma(forget, hidden[k], f);
    }
}

template <int Cluster, bool Scaled = false>
__device__ __forceinline__ bool output_scales(cg::cluster_group cluster, float s0, float s1,
                                              float (*sums)[kWarps], float* totals, float* inverse,
                                              float* rescale, int count, float epsilon) {
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    s0 = common::warp_reduce_sum<4>(s0);
    s1 = common::warp_reduce_sum<4>(s1);
    if (lane % 4 == 0) {
        sums[lane / 4][warp]     = s0;
        sums[lane / 4 + 8][warp] = s1;
    }
    __syncthreads();
    if (tid < kTokens) {
        float sum = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) sum += sums[tid][w];
        totals[tid] = sum;
    }
    cluster.sync();
    bool retry = false;
    if (tid < kTokens) {
        const float sum = cluster_sum<Cluster>(cluster, totals + tid);
        if constexpr (Scaled) {
            if (rescale[tid] != 0)
                inverse[tid] = scaled_inverse_rms(sum, kHidden, epsilon, rescale[tid]);
        } else {
            retry        = tid < count && lost_squares(sum);
            rescale[tid] = retry ? square_rescale(sum) : 0.F;
            inverse[tid] = inverse_rms(sum, kHidden, epsilon);
        }
    }
    return __syncthreads_or(retry);
}

// The register layout is the MMA accumulator layout: a quartet of lanes
// owns eight adjacent features for each of two tokens.
template <int Cluster>
struct RegisterBlend {
    static constexpr int Width = kHidden / Cluster;
    static constexpr int Steps = (Width + kFeatures - 1) / kFeatures;
    static_assert(Width % 8 == 0);
    float2 value[Steps][2];

    __device__ __forceinline__ static int token(int z, int lane) { return lane / 4 + z * 8; }

    __device__ __forceinline__ static int feature(int i, int warp, int lane) {
        return i * kFeatures + warp * 8 + (lane % 4) * 2;
    }

    __device__ __forceinline__ void load(const float* source, int part, int count) {
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
        for (int i = 0; i < Steps; ++i) {
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int t = token(z, lane), d = part * Width + feature(i, warp, lane);
                value[i][z] = t < count && d < (part + 1) * Width
                                  ? *reinterpret_cast<const float2*>(source + t * kHidden + d)
                                  : float2{};
            }
        }
    }

    __device__ __forceinline__ void load_shared(const float (*raw)[Width]) {
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
        for (int i = 0; i < Steps; ++i) {
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int t = token(z, lane), d = feature(i, warp, lane);
                value[i][z] = d < Width
                                  ? *reinterpret_cast<const float2*>(raw[t] + (d ^ ((t & 3) * 8)))
                                  : float2{};
            }
        }
    }

    template <bool Load>
    __device__ __forceinline__ void
    apply(cg::cluster_group cluster, const Token& row, int count, const Bf16* up,
          const Bf16* latent, const RegisterBlend* message, Bf16* weight, float (*sums)[kWarps],
          float* total, float* inverse, float* rescale, float epsilon) {
        const int part = cluster.block_rank(), lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        constexpr int TileValues = 2 * kFeatures * kRank;
        unsigned h[4][4];
        stage_up<kFeatures>(weight, up, Width, 0, part);
        common::cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) load_a<kRank>(latent, 0, i * 16, lane, h[i]);
        float sq[2]{};
#pragma unroll
        for (int i = 0; i < Steps; ++i) {
            const int slot = i % 2;
            if (i + 1 < Steps)
                stage_up<kFeatures>(weight + (slot ^ 1) * TileValues, up, Width,
                                    (i + 1) * kFeatures, part);
            if (feature(i, warp, lane) < Width) {
                float gain[4]{}, forget[4]{};
                gate_logits(weight + slot * TileValues, h, warp, lane, gain, forget);
#pragma unroll
                for (int z = 0; z < 2; ++z) {
                    const int t = token(z, lane), d = part * Width + feature(i, warp, lane);
                    float2 x{}, y{};
                    if constexpr (Load) {
                        if (t < count) {
                            x = *reinterpret_cast<const float2*>(row.dest + t * kHidden + d);
                            y = *reinterpret_cast<const float2*>(row.message + t * kHidden + d);
                        }
                    } else {
                        x = value[i][z];
                        y = message->value[i][z];
                    }
                    const float2 v{blend(x.x, y.x, gain[2 * z], forget[2 * z]),
                                   blend(x.y, y.y, gain[2 * z + 1], forget[2 * z + 1])};
                    value[i][z] = v;
                    sq[z]       = fmaf(v.x, v.x, sq[z]);
                    sq[z]       = fmaf(v.y, v.y, sq[z]);
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
        if (output_scales<Cluster>(cluster, sq[0], sq[1], sums, total, inverse, rescale, count,
                                   epsilon)) {
            cluster.sync();
            const float factor[2]{rescale[token(0, lane)], rescale[token(1, lane)]};
            float scaled[2]{};
#pragma unroll
            for (int i = 0; i < Steps; ++i) {
#pragma unroll
                for (int z = 0; z < 2; ++z) {
                    const auto v  = value[i][z];
                    const float x = v.x * factor[z], y = v.y * factor[z];
                    scaled[z] = fmaf(x, x, scaled[z]);
                    scaled[z] = fmaf(y, y, scaled[z]);
                }
            }
            output_scales<Cluster, true>(cluster, scaled[0], scaled[1], sums, total, inverse,
                                         rescale, count, epsilon);
        }
#pragma unroll
        for (int i = 0; i < Steps; ++i) {
            const int d = part * Width + feature(i, warp, lane);
            if (d >= (part + 1) * Width) continue;
            const auto g =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(row.gamma + d));
#pragma unroll
            for (int z = 0; z < 2; ++z) {
                const int t = token(z, lane);
                if (t < count) {
                    const auto v = value[i][z];
                    *reinterpret_cast<__nv_bfloat162*>(row.out + t * kHidden + d) =
                        __floats2bfloat162_rn(v.x * inverse[t] * g.x, v.y * inverse[t] * g.y);
                }
            }
        }
    }
};

// The 8-CTA path keeps inputs in shared memory through down. Up reads
// both into registers and reuses their shared storage to stage weights.
__global__ __launch_bounds__(256, 2) void __cluster_dims__(8, 1, 1) prefill_cluster8(Arguments a) {
    constexpr int Cluster = 8, Width = kHidden / Cluster, K = 32;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    auto cluster   = cg::this_cluster();
    const int part = cluster.block_rank(), tiles = (a.tokens - 1) / kTokens + 1;

    union Inputs {
        struct {
            float dest[kTokens][Width], message[kTokens][Width];
        } raw;

        Bf16 up[4 * kFeatures * kRank];
    };

    __shared__ __align__(16) Inputs data;
    auto& dest    = data.raw.dest;
    auto& message = data.raw.message;
    __shared__ float2 pair_stats[kTokens];
    __shared__ float ds[kTokens], ms[kTokens], sums[kTokens][kWarps];
    __shared__ float rescale[kTokens];
    __shared__ __align__(16) Bf16 latent[kTokens * kRank], shard[kTokens / Cluster * kRank];

    union Storage {
        Bf16 weights[kRank * K];
        float partial[kTokens][kRank];
    };

    __shared__ __align__(16) Storage work;
    extern __shared__ __align__(16) Bf16 next[];
    for (std::int64_t tile = blockIdx.x / Cluster; tile < std::int64_t(tiles) * a.streams;
         tile += gridDim.x / Cluster) {
        const int m = int(tile % tiles) * kTokens, count = min(kTokens, a.tokens - m);
        const Token token(a, tile / tiles * a.tokens + m);
        const auto* dw = a.down + token.stream * (2 * kHidden * kRank);
        const auto* uw = a.up + token.stream * (2 * kHidden * kRank);
        stage_down_weights<K>(work.weights, dw, part * Width);

#pragma unroll
        for (int t = warp; t < kTokens; t += kWarps) {
            float sd = 0, sm = 0;
#pragma unroll
            for (int i = lane * 4; i < Width; i += 128) {
                float4 x{}, y{};
                if (t < count) {
                    const int d = t * kHidden + part * Width + i;
                    x           = *reinterpret_cast<const float4*>(token.dest + d);
                    y           = *reinterpret_cast<const float4*>(token.message + d);
                }
                const int index                                = i ^ ((t & 3) * 8);
                *reinterpret_cast<float4*>(dest[t] + index)    = x;
                *reinterpret_cast<float4*>(message[t] + index) = y;
                sd                                             = square_sum(x, sd);
                sm                                             = square_sum(y, sm);
            }
            sd = common::warp_reduce_sum(sd);
            sm = common::warp_reduce_sum(sm);
            if (lane == 0) pair_stats[t] = {sd, sm};
        }
        common::cp_wait<0>();
        cluster.sync();
        if (tid < count) {
            float2 sum{};
#pragma unroll
            for (int c = 0; c < Cluster; ++c) {
                const auto x = *cluster.map_shared_rank(pair_stats + tid, c);
                sum.x += x.x;
                sum.y += x.y;
            }
            ds[tid] = row_inverse_rms(sum.x, token.dest + tid * kHidden, kHidden, a.pair_eps);
            ms[tid] = row_inverse_rms(sum.y, token.message + tid * kHidden, kHidden, a.pair_eps);
        } else if (tid < kTokens) {
            ds[tid] = ms[tid] = 0;
        }
        __syncthreads();
        down_shard<Cluster, K, true>(dw, dest, message, ds, ms, nullptr, work.weights, next,
                                     work.partial, part);
        gather_latent<Cluster>(cluster, part, work.partial, shard, latent);

        RegisterBlend<Cluster> dest_fragment, message_fragment;
        dest_fragment.load_shared(dest);
        message_fragment.load_shared(message);
        __syncthreads();
        dest_fragment.template apply<false>(cluster, token, count, uw, latent, &message_fragment,
                                            data.up, sums, ds, ms, rescale, a.norm_eps);
        cluster.sync();
    }
}

// Retain two resident blocks per SM with the extra RMS recovery code.
__global__ void __cluster_dims__(4, 1, 1) __launch_bounds__(kThreads, 2)
    prefill_cluster4(Arguments a) {
    constexpr int Cluster = 4, K = 64;
    using Fragment      = RegisterBlend<Cluster>;
    constexpr int Width = Fragment::Width, Steps = Fragment::Steps;
    constexpr int WeightValues = 2 * kFeatures * kRank;
    const int tid = threadIdx.x, lane = tid % 32;
    auto cluster   = cg::this_cluster();
    const int part = cluster.block_rank(), tiles = (a.tokens - 1) / kTokens + 1;

    union Storage {
        struct {
            Bf16 weight[2][kRank * K], pair[kTokens * K];
        } down;

        float partial[kTokens][kRank];
        Bf16 weight_up[2 * WeightValues];
    };

    extern __shared__ __align__(16) unsigned char storage[];
    auto& work = *reinterpret_cast<Storage*>(storage);
    __shared__ float totals[kTokens], inverse[kTokens], sums[kTokens][kWarps];
    __shared__ float rescale[kTokens];
    __shared__ float2 pair_sums[kTokens][kWarps], scalars[kTokens];
    __shared__ __align__(16) Bf16 latent[kTokens * kRank], shard[kTokens * kRank / Cluster];
    for (std::int64_t tile = blockIdx.x / Cluster; tile < std::int64_t(tiles) * a.streams;
         tile += gridDim.x / Cluster) {
        const int m = int(tile % tiles) * kTokens, count = min(kTokens, a.tokens - m);
        const Token token(a, tile / tiles * a.tokens + m);
        const auto* dw = a.down + token.stream * (2 * kHidden * kRank);
        const auto* uw = a.up + token.stream * (2 * kHidden * kRank);
        stage_down_weights<K>(work.down.weight[0], dw, part * Width);
        Fragment dest, message;
        dest.load(token.dest, part, count);
        message.load(token.message, part, count);
#pragma unroll
        for (int z = 0; z < 2; ++z) {
            float sd = 0, sm = 0;
#pragma unroll
            for (int i = 0; i < Steps; ++i) {
                const auto d = dest.value[i][z], m = message.value[i][z];
                sd = fmaf(d.x, d.x, sd);
                sd = fmaf(d.y, d.y, sd);
                sm = fmaf(m.x, m.x, sm);
                sm = fmaf(m.y, m.y, sm);
            }
            sd = common::warp_reduce_sum<4>(sd);
            sm = common::warp_reduce_sum<4>(sm);
            if (lane % 4 == 0) pair_sums[Fragment::token(z, lane)][tid / 32] = {sd, sm};
        }
        __syncthreads();
        if (tid < kTokens) {
            float2 s{};
#pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                s.x += pair_sums[tid][w].x;
                s.y += pair_sums[tid][w].y;
            }
            scalars[tid] = s;
        }
        cluster.sync();
        if (tid < count) {
            float2 sum{};
#pragma unroll
            for (int c = 0; c < Cluster; ++c) {
                const auto s = *cluster.map_shared_rank(scalars + tid, c);
                sum.x += s.x;
                sum.y += s.y;
            }
            totals[tid] = row_inverse_rms(sum.x, token.dest + tid * kHidden, kHidden, a.pair_eps);
            inverse[tid] =
                row_inverse_rms(sum.y, token.message + tid * kHidden, kHidden, a.pair_eps);
        } else if (tid < kTokens) {
            totals[tid] = inverse[tid] = 0;
        }
        __syncthreads();
        common::cp_wait<0>();
        __syncthreads();
        down_shard<Cluster, K, false>(dw, nullptr, nullptr, totals, inverse, work.down.pair,
                                      work.down.weight[0], work.down.weight[1], work.partial, part,
                                      &token, count);
        gather_latent<Cluster>(cluster, part, work.partial, shard, latent);
        __syncthreads();
        dest.template apply<true>(cluster, token, count, uw, latent, &message, work.weight_up, sums,
                                  totals, inverse, rescale, a.norm_eps);
        cluster.sync();
    }
}

} // namespace

void launch_cluster_prefill(Arguments a, Route route, cudaStream_t stream) {
    const auto tiles = std::int64_t((a.tokens - 1) / kTokens + 1) * a.streams;
    if (route == Route::Cluster4) {
        constexpr int bytes = 4 * kFeatures * kRank * sizeof(Bf16);
        CUDA_CHECK(cudaFuncSetAttribute(prefill_cluster4,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        prefill_cluster4<<<grid_size(tiles, 4), kThreads, bytes, stream>>>(a);
    } else {
        constexpr int bytes = kRank * 32 * sizeof(Bf16);
        CUDA_CHECK(cudaFuncSetAttribute(prefill_cluster8,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        prefill_cluster8<<<grid_size(tiles, 8), kThreads, bytes, stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::loop_message
