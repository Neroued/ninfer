#include "ops/kv_cache/context_materialize/dispatch.h"
#include "ops/kv_cache/context_materialize/epilogue.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_tma_mma.cuh"

namespace ninfer::ops::detail::context_kv::bf16 {
namespace {

struct Epilogue {
    static constexpr bool kTmaCollective = true;
    template <class S>
    static constexpr int kSharedBytes = (S::kBlockRows < 128 ? S::kBlockRows : 128) *
                                        (S::kBlockTokens < 128 ? S::kBlockTokens : 128) * 4;
    const DeviceLayers* layers;
    const int *positions, *counts;
    float* scratch;
    int width, batch, min_count, prefix;
    const int* cached_columns  = nullptr;
    const int* cached_requests = nullptr;
    int first_token            = 0;

    __device__ __forceinline__ bool live(int token) const {
        if (cached_columns) return cached_columns[token - first_token] >= 0;
        if (token >= prefix * batch) return false;
        const int count = counts[token / prefix];
        return count >= min_count && count <= prefix && token % prefix < count;
    }

    __device__ __forceinline__ int physical_column(int token) const {
        return cached_columns ? cached_columns[token - first_token]
                              : context_column(token, width, prefix);
    }

    __device__ __forceinline__ int request_row(int token) const {
        return cached_requests ? cached_requests[token - first_token] : token / prefix;
    }

    template <class S, bool Full, class Output>
    __device__ void finish_tile(const Output&, unsigned char* shared,
                                float (&accum)[S::kMmaRows][S::kMmaTokens][4], int row_begin,
                                int token_begin, int, int) const {
        constexpr int Producers = [] {
            if constexpr (requires { S::kProducerThreads; })
                return S::kProducerThreads;
            else
                return 0;
        }();
        constexpr int R  = S::kBlockRows < 128 ? S::kBlockRows : 128;
        constexpr int T  = S::kBlockTokens < 128 ? S::kBlockTokens : 128;
        auto* scores     = reinterpret_cast<float*>(shared);
        const int tid    = int(threadIdx.x) - Producers;
        const int l      = blockIdx.z;
        const auto layer = layers->layer[l];
        const int warp   = threadIdx.x / 32;
#pragma unroll
        for (int rp = 0; rp < S::kBlockRows; rp += R) {
#pragma unroll
            for (int tp = 0; tp < S::kBlockTokens; tp += T) {
                if (tid >= 0) {
                    const int w = tid / 32, lane = tid % 32;
                    const int wm = w / S::kWarpsTokens, wn = w % S::kWarpsTokens;
#pragma unroll
                    for (int mi = 0; mi < S::kMmaRows; ++mi)
#pragma unroll
                        for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                            const int r = wm * S::kWarpRows + mi * 16 + lane / 4;
                            const int t = wn * S::kWarpTokens + ni * 8 + 2 * (lane % 4);
                            if (r >= rp && r < rp + R && t >= tp && t < tp + T) {
                                const int at       = (t - tp) * R + r - rp;
                                scores[at]         = accum[mi][ni][0];
                                scores[at + R]     = accum[mi][ni][1];
                                scores[at + 8]     = accum[mi][ni][2];
                                scores[at + R + 8] = accum[mi][ni][3];
                            }
                        }
                }
                __syncthreads();
                const int first_row = row_begin + rp;
                if constexpr (R == 128) {
                    if (first_row < 512) {
                        for (int local = warp; local < T; local += S::kThreads / 32) {
                            const int token = token_begin + tp + local;
                            if (!live(token)) continue;
                            const int physical = physical_column(token);
                            store_key_head(scores + local * 128, layer, positions, layer.offsets,
                                           physical, width, first_row / 128, layers->epsilon,
                                           layers->frequencies);
                        }
                    } else {
                        for (int item = threadIdx.x; item < R * T; item += S::kThreads) {
                            const int token = token_begin + tp + item / R;
                            if (!live(token)) continue;
                            const int physical = physical_column(token);
                            store_value(first_row + item % R - 512, physical,
                                        layer.offsets[request_row(token)], positions, layer,
                                        scores[item]);
                        }
                    }
                } else {
                    for (int item = threadIdx.x; item < R * T; item += S::kThreads) {
                        const int token = token_begin + tp + item / R;
                        if (!live(token)) continue;
                        const int row = first_row + item % R;
                        if (row < 512)
                            scratch[row + 512LL * (token + prefix * batch * l)] = scores[item];
                        else {
                            const int physical = physical_column(token);
                            store_value(row - 512, physical, layer.offsets[request_row(token)],
                                        positions, layer, scores[item]);
                        }
                    }
                }
                if (rp + R < S::kBlockRows || tp + T < S::kBlockTokens) __syncthreads();
            }
        }
    }
};

// The family owns count-aware packed-column staging. BF16 contraction math and
// schedules are shared with Linear; represented inputs and accumulators are unchanged.
template <class S>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void projection_kernel(
    const __nv_bfloat16* x, const int* positions, const int* counts,
    const __grid_constant__ DeviceLayers layers, float* scratch, int width, int batch,
    int min_count, int prefix) {
    const Epilogue metadata{&layers, positions, counts, scratch, width, batch, min_count, prefix};
    constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, BK = S::kBlockK, Stages = S::kStages;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<S>(blockIdx.x, 1024 / BR, (prefix * batch + BT - 1) / BT, tile_r,
                                 tile_t);
    const int row_begin = tile_r * BR, token_begin = tile_t * BT;
    __shared__ int mapped_columns[BT], mapped_requests[BT];
    bool active = false;
    for (int t = threadIdx.x; t < BT; t += S::kThreads) {
        const int token    = token_begin + t;
        const bool valid   = metadata.live(token);
        mapped_columns[t]  = valid ? context_column(token, width, prefix) : -1;
        mapped_requests[t] = valid ? token / prefix : 0;
        active |= valid;
    }
    if (!__syncthreads_or(active)) return;
    const Epilogue ep{&layers,   positions, counts,         scratch,         width,      batch,
                      min_count, prefix,    mapped_columns, mapped_requests, token_begin};
    extern __shared__ __align__(128) unsigned char shared[];
    auto* as           = reinterpret_cast<__nv_bfloat16*>(shared);
    auto* bs           = as + Stages * BR * BK;
    const auto* weight = reinterpret_cast<const __nv_bfloat16*>(layers.layer[blockIdx.z].key_codes);
    const auto stage   = [&](int slot, int kt) {
        for (int i = threadIdx.x; i < BR * (BK / 8); i += S::kThreads) {
            const int r = i / (BK / 8), k = i % (BK / 8) * 8;
            common::cp_async<16, S::kWeightCache>(
                as + slot * BR * BK + r * BK + bf16_mma_shared_col<S>(r, k),
                weight + std::int64_t(row_begin + r) * 4096 + kt * BK + k);
        }
        for (int i = threadIdx.x; i < BT * (BK / 8); i += S::kThreads) {
            const int t = i / (BK / 8), k = i % (BK / 8) * 8;
            const int mapped   = mapped_columns[t];
            const bool valid   = mapped >= 0;
            const int physical = valid ? mapped : 0;
            common::cp_async_zfill<16, S::kActivationCache>(
                bs + slot * BT * BK + t * BK + bf16_mma_shared_col<S>(t, k),
                x + std::int64_t(physical) * 4096 + kt * BK + k, valid ? 16 : 0);
        }
    };
    constexpr int Tiles = 4096 / BK;
#pragma unroll
    for (int s = 0; s < Stages; ++s) {
        stage(s, s);
        common::cp_commit();
    }
    float accum[S::kMmaRows][S::kMmaTokens][4] = {};
#pragma unroll 1
    for (int kt = 0; kt < Tiles; ++kt) {
        if (kt + Stages <= Tiles)
            common::cp_wait<Stages - 1>();
        else
            common::cp_wait<0>();
        __syncthreads();
        bf16_mma_compute_stage<S>(as + (kt % Stages) * BR * BK, bs + (kt % Stages) * BT * BK, accum,
                                  threadIdx.x / 32, threadIdx.x % 32);
        __syncthreads();
        if (kt + Stages < Tiles) {
            stage(kt % Stages, kt + Stages);
            common::cp_commit();
        }
    }
    bf16_finish_mma_tile<S, false>(LinearBf16Output{nullptr, 1024}, ep, shared, accum, row_begin,
                                   token_begin, 1024, prefix * batch, threadIdx.x / 32,
                                   threadIdx.x % 32);
}

struct alignas(128) Descriptors {
    Bf16TmaDescriptors layer[kLayers];
};

template <class S>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void tma_projection_kernel(
    const __grid_constant__ Descriptors descriptors, const __grid_constant__ DeviceLayers layers,
    const int* positions, const int* counts, float* scratch, int width, int batch, int min_count,
    int prefix) {
    int r, t;
    bf16_mma_tile_coordinates<S>(blockIdx.x, 1024 / S::kBlockRows,
                                 (prefix * batch + S::kBlockTokens - 1) / S::kBlockTokens, r, t);
    const Epilogue metadata{&layers, positions, counts, scratch, width, batch, min_count, prefix};
    __shared__ int mapped_columns[S::kBlockTokens], mapped_requests[S::kBlockTokens];
    bool active = false;
    for (int i = threadIdx.x; i < S::kBlockTokens; i += S::kThreads) {
        const int token    = t * S::kBlockTokens + i;
        const bool valid   = metadata.live(token);
        mapped_columns[i]  = valid ? context_column(token, width, prefix) : -1;
        mapped_requests[i] = valid ? token / prefix : 0;
        active |= valid;
    }
    if (!__syncthreads_or(active)) return;
    const Epilogue ep{&layers,
                      positions,
                      counts,
                      scratch,
                      width,
                      batch,
                      min_count,
                      prefix,
                      mapped_columns,
                      mapped_requests,
                      t * S::kBlockTokens};
    bf16_a16_tma_mma_body<S, false>(descriptors.layer[blockIdx.z], LinearBf16Output{nullptr, 1024},
                                    ep, 1024, 4096, 0, prefix * batch);
}

// All five layers share phases; no BF16 K projection intermediate is introduced.
__global__ __launch_bounds__(640) void key_post_kernel(const float* input, const int* positions,
                                                       const int* counts,
                                                       const __grid_constant__ DeviceLayers layers,
                                                       int width, int batch, int min_count,
                                                       int prefix) {
    const int token = blockIdx.x, request = token / prefix;
    const int count = counts[request];
    if (count < min_count || count > prefix || token % prefix >= count) return;
    const int physical = context_column(token, width, prefix);
    __shared__ float sine[64], cosine[64];
    if (threadIdx.x < 64)
        rope_d128_sincos(layers.frequencies, positions[physical], threadIdx.x, sine + threadIdx.x,
                         cosine + threadIdx.x);
    const int h = threadIdx.x / 32, lane = threadIdx.x % 32, j = lane * 2;
    const auto layer   = layers.layer[h / 4];
    const int head     = h % 4;
    const auto* source = input + (h / 4 * batch * prefix + token) * 512LL + head * 128;
    float x = source[j], y = source[j + 1], u = source[j + 64], v = source[j + 65];
    float square        = common::warp_reduce_sum(x * x + y * y + u * u + v * v);
    const float inverse = rsqrtf(__shfl_sync(0xffffffffU, square, 0) / 128.F + layers.epsilon);
    x *= inverse * __bfloat162float(layer.key_norm[j]);
    y *= inverse * __bfloat162float(layer.key_norm[j + 1]);
    u *= inverse * __bfloat162float(layer.key_norm[j + 64]);
    v *= inverse * __bfloat162float(layer.key_norm[j + 65]);
    __syncthreads();
    const auto dst = layer.offsets[request] / 2 +
                     kv::cyclic_offset<128, 2048>(positions[physical], head, layer.padded_capacity);
    auto* out = reinterpret_cast<__nv_bfloat162*>(layer.cache_k + dst);
    out[lane] =
        __floats2bfloat162_rn(x * cosine[j] - u * sine[j], y * cosine[j + 1] - v * sine[j + 1]);
    out[lane + 32] =
        __floats2bfloat162_rn(u * cosine[j] + x * sine[j], v * cosine[j + 1] + y * sine[j + 1]);
}

template <class S, bool Tma = false>
void launch(const Tensor& x, const Tensor& positions, const Tensor& counts,
            const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
            const ContextKVMaterializeProfile& profile,
            ContextKVMaterializeExecutionEnvelope envelope, const Tensor& scratch,
            cudaStream_t stream) {
    const auto device = make_device_layers(layers, profile);
    const int columns = envelope.max_count * x.ne[2];
    const dim3 grid(1024 / S::kBlockRows * ((columns + S::kBlockTokens - 1) / S::kBlockTokens), 1,
                    kLayers);
    const auto cp = [&]<class C>() {
        constexpr int bytes = linear_mma_shared_bytes<C, Epilogue>;
        common::prepare_shared<bytes, projection_kernel<C>, true>();
        projection_kernel<C><<<grid, C::kThreads, bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const int*>(positions.data),
            static_cast<const int*>(counts.data), device, static_cast<float*>(scratch.data),
            x.ne[1], x.ne[2], envelope.min_count, envelope.max_count);
    };
    if constexpr (Tma) {
        if (x.ne[2] == 1 || x.ne[1] == int(envelope.max_count)) {
            Descriptors descriptors{};
            for (int l = 0; l < kLayers; ++l)
                descriptors.layer[l] = make_bf16_tma_descriptors<S>(
                    {static_cast<const __nv_bfloat16*>(x.data),
                     static_cast<const __nv_bfloat16*>(layers[l].key_weight.qdata), 1024, 4096,
                     columns});
            constexpr int bytes = bf16_tma_scratch_bytes<S, Epilogue> + S::kBarrierBytes;
            common::prepare_shared<bytes, tma_projection_kernel<S>, true>();
            tma_projection_kernel<S><<<grid, S::kThreads, bytes, stream>>>(
                descriptors, device, static_cast<const int*>(positions.data),
                static_cast<const int*>(counts.data), static_cast<float*>(scratch.data), x.ne[1],
                x.ne[2], envelope.min_count, envelope.max_count);
        } else {
            using C = Bf16ScheduleInstance<
                Bf16A16MmaSchedule<S::kBlockRows, S::kBlockTokens, S::kBlockK, S::kWarpRows,
                                   S::kWarpTokens, S::kStages, 1, Cache::cg, Cache::ca,
                                   Bf16MmaFragmentPipeline::PingPong, S::kRaster,
                                   Bf16MmaSwizzle::Xor64, S::kRasterGroupRows>,
                4096>;
            cp.template operator()<C>();
        }
    } else
        cp.template operator()<S>();
    CUDA_CHECK(cudaGetLastError());
    if constexpr (S::kBlockRows < 128) {
        key_post_kernel<<<columns, 640, 0, stream>>>(
            static_cast<const float*>(scratch.data), static_cast<const int*>(positions.data),
            static_cast<const int*>(counts.data), device, x.ne[1], x.ne[2], envelope.min_count,
            envelope.max_count);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class S, bool Tma = false>
constexpr ContextKVMaterializeImplementation implementation() {
    return {launch<S, Tma>, S::kBlockRows < 128};
}

using Small = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<16, 16, 256, 16, 8, 2, 1, Cache::cg, Cache::ca,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    4096>;
using Medium = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 16, 256, 16, 8, 2, 1, Cache::cg, Cache::ca,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    4096>;
using Block = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 128, 16, 16, 2, 1, Cache::cg, Cache::ca,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    4096>;
using Narrow =
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<16, 8, 256, 16, 8, 2, 1, Bf16MmaRaster::Grouped, 2>,
                         4096>;
using Transition =
    Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 1, Bf16MmaRaster::Grouped, 2>,
                         4096>;
using Wide = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1, Bf16MmaRaster::Grouped, 2>, 4096>;
using Long = Bf16ScheduleInstance<
    Bf16A16TmaMmaSchedule<128, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::Grouped, 2>, 4096>;

} // namespace
} // namespace ninfer::ops::detail::context_kv::bf16

namespace ninfer::ops::detail {
ContextKVMaterializeImplementation select_bf16_context_kv_materialize(std::int32_t columns) {
    using namespace context_kv::bf16;
    if (columns <= 16) return implementation<Small>();
    if (columns <= 32) return implementation<Narrow, true>();
    if (columns <= 64) return implementation<Medium>();
    if (columns <= 128) return implementation<Block>();
    if (columns <= 256) return implementation<Transition, true>();
    if (columns <= 512) return implementation<Wide, true>();
    return implementation<Long, true>();
}
} // namespace ninfer::ops::detail
