#include "launch.h"

#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <algorithm>
#include <cuda_bf16.h>

namespace ninfer::ops::markov {
namespace {

using Bf16 = __nv_bfloat16;

constexpr int kWarps   = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kRows    = 16 * kWarps;
static_assert(kRows == kLogitTile);

struct Problem {
    const Bf16* latent;
    const Bf16* weights;
    const Bf16* bias;
    const Bf16* base;
    const int* live;
    Bf16* out;
    int rank, batch, vocab, domain;
    float softcap;
    LogitTileResult* partials = nullptr;

    __device__ __host__ int tiles() const { return 1 + (domain - 1) / kRows; }
};

struct LogitPartial {
    float maximum, mass;
    int id;
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

__device__ __forceinline__ LogitPartial reduce_quad(const Problem& p, float4 dot, int b, int v,
                                                    bool enabled) {
    const float absent = -CUDART_INF_F;
    float values[]{absent, absent, absent, absent};
    const auto index = std::int64_t(b) * p.vocab + v;
    if (enabled && p.domain - v >= 4 && (index & 3) == 0) {
        const auto bias = common::load_vec<uint2>(p.bias + v);
        const auto base = common::load_vec<uint2>(p.base + index);
        const auto a    = finish_pair(dot.x, dot.y, bias.x, base.x, p.softcap);
        const auto z    = finish_pair(dot.z, dot.w, bias.y, base.y, p.softcap);
        values[0]       = __uint_as_float(a << 16);
        values[1]       = __uint_as_float(a & 0xffff0000U);
        values[2]       = __uint_as_float(z << 16);
        values[3]       = __uint_as_float(z & 0xffff0000U);
    } else if (enabled) {
        const float dots[]{dot.x, dot.y, dot.z, dot.w};
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if (v + i < p.domain)
                values[i] =
                    __bfloat162float(finish(dots[i], p.bias[v + i], p.base[index + i], p.softcap));
    }
    float maximum = fmaxf(fmaxf(values[0], values[1]), fmaxf(values[2], values[3]));
    maximum       = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 8));
    maximum       = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 16));
    float mass    = 0.0f;
    int id        = INT_MAX;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (values[i] == absent) continue;
        mass += __expf(values[i] - maximum);
        if (values[i] == maximum) id = min(id, v + i);
    }
    mass += __shfl_xor_sync(0xffffffff, mass, 8);
    mass += __shfl_xor_sync(0xffffffff, mass, 16);
    id = min(id, __shfl_xor_sync(0xffffffff, id, 8));
    id = min(id, __shfl_xor_sync(0xffffffff, id, 16));
    return {maximum, mass, id};
}

__device__ __forceinline__ void merge(LogitPartial& a, LogitPartial b) {
    if (b.mass == 0.0f) return;
    if (a.maximum < b.maximum) {
        a = {b.maximum, fmaf(a.mass, __expf(a.maximum - b.maximum), b.mass), b.id};
    } else {
        a.mass = fmaf(b.mass, __expf(b.maximum - a.maximum), a.mass);
        if (a.maximum == b.maximum) a.id = min(a.id, b.id);
    }
}

template <int Columns, int KTile, int FixedRank, bool Masked, bool Resident, bool Reduce>
__global__ void logits(const __grid_constant__ Problem p) {
    constexpr int kMmas    = Columns / 8;
    constexpr int kVectors = KTile / 8;
    __shared__ __align__(16) char storage[(kRows + Columns) * KTile * sizeof(Bf16)];
    __shared__ std::uint64_t live_mask;
    auto* tile        = reinterpret_cast<Bf16*>(storage);
    auto* activations = tile + kRows * KTile;
    static_assert(!Reduce || kWarps * sizeof(LogitPartial) <= KTile * sizeof(Bf16));
    // The epilogue can reuse the activation tile after the last MMA.
    auto* partials = reinterpret_cast<LogitPartial*>(activations);

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
                const int v   = v0 + r;
                const bool ok = Full || (v < p.domain && k < depth);
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
        const int b0       = static_cast<int>(start);
        const int count    = min(Columns, p.batch - b0);
        std::uint64_t mask = count >= 64 ? ~0ULL : ((1ULL << count) - 1ULL);
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
            if constexpr (!Reduce) {
                for (int i = tid; i < kRows * count; i += kThreads) {
                    const int v = v0 + i % kRows;
                    if (v < p.vocab)
                        p.out[std::int64_t(b0 + i / kRows) * p.vocab + v] =
                            __float2bfloat16_rn(0.0f);
                }
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
                    // All vocabulary tiles reuse the same small batch of latent rows.
                    common::cp_async_zfill<16, Cache::ca>(
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

            if constexpr (Reduce) __syncthreads();
#pragma unroll
            for (int n = 0; n < kMmas; ++n) {
                const auto quad =
                    common::mma_row_quad(make_float4(sum[n][0], sum[n][1], sum[n][2], sum[n][3]));
                const int b = n * 8 + quad.column;
                if constexpr (Reduce) {
                    const auto partial =
                        reduce_quad(p, quad.values, b0 + b, v0 + warp * 16 + quad.row,
                                    (mask & (1ULL << b)) != 0);
                    if (lane < 8) partials[warp * Columns + b] = partial;
                } else if (b < count) {
                    store(p, quad.values, b0 + b, v0 + warp * 16 + quad.row,
                          (mask & (1ULL << b)) != 0);
                }
            }
            if constexpr (Reduce) {
                __syncthreads();
                if (tid < count && (mask & (1ULL << tid))) {
                    auto partial = partials[tid];
#pragma unroll
                    for (int w = 1; w < kWarps; ++w) merge(partial, partials[w * Columns + tid]);
                    const auto packed =
                        __float_as_uint(partial.maximum) | static_cast<unsigned>(partial.id - v0);
                    p.partials[std::int64_t(b0 + tid) * p.tiles() + blockIdx.x] = {packed,
                                                                                   partial.mass};
                }
            }
        }
        if (start + gridDim.y * Columns < p.batch) __syncthreads();
    }
}

template <int Columns, int KTile, int FixedRank, bool Reduce, bool Resident = false>
void submit(Problem p, cudaStream_t stream) {
    const dim3 grid(Reduce ? p.tiles() : 1 + (p.vocab - 1) / kRows,
                    Resident ? 1 : std::min(65535, 1 + (p.batch - 1) / Columns));
    if (p.live)
        logits<Columns, KTile, FixedRank, true, Resident, Reduce><<<grid, kThreads, 0, stream>>>(p);
    else
        logits<Columns, KTile, FixedRank, false, Resident, Reduce>
            <<<grid, kThreads, 0, stream>>>(p);
}

template <bool Reduce>
void launch(Problem p, cudaStream_t stream) {
    const auto launch_columns = [&]<int Columns>() {
        if (p.rank == 256)
            submit<Columns, Columns == 8 ? 128 : 64, 256, Reduce>(p, stream);
        else if (Columns > 32 || p.rank <= 64)
            submit<Columns, 64, 0, Reduce>(p, stream);
        else
            submit<Columns, 128, 0, Reduce>(p, stream);
    };
    if (p.batch > 64 && p.rank <= 64)
        submit<32, 64, 0, Reduce, true>(p, stream);
    else if (p.batch > 64 && p.rank <= 128)
        submit<32, 128, 0, Reduce, true>(p, stream);
    else if (p.batch <= 8)
        launch_columns.template operator()<8>();
    else if (p.batch <= 16)
        launch_columns.template operator()<16>();
    else if (p.batch <= 24)
        launch_columns.template operator()<24>();
    else if (p.batch <= 32)
        launch_columns.template operator()<32>();
    else if (p.batch <= 48)
        launch_columns.template operator()<48>();
    else
        launch_columns.template operator()<64>();
    CUDA_CHECK(cudaGetLastError());
}

constexpr int kFinishThreads = 256;

__global__ void logits_argmax_finish(const LogitTileResult* partials, const int* live, int tiles,
                                     int domain, int* ids, float* probs) {
    const int row = blockIdx.x, tid = threadIdx.x;
    if (live && live[row] == 0) {
        if (tid == 0) {
            ids[row]   = 0;
            probs[row] = __fdiv_rn(1.0f, static_cast<float>(domain));
        }
        return;
    }
    LogitPartial local{-CUDART_INF_F, 0.0f, INT_MAX};
    for (int i = tid; i < tiles; i += kFinishThreads) {
        const auto partial = partials[std::int64_t(row) * tiles + i];
        const auto key     = partial.maximum_and_offset;
        const auto id      = i * kRows + static_cast<int>(key & 0xffffU);
        merge(local, {__uint_as_float(key & 0xffff0000U), partial.mass, id});
    }

    const float maximum = common::block_max_broadcast<kFinishThreads>(local.maximum);
    const int id        = local.maximum == maximum ? local.id : INT_MAX;
    const auto warp_id  = __reduce_min_sync(0xffffffff, id);
    const float mass    = local.mass * __expf(local.maximum - maximum);
    __shared__ int warp_ids[kFinishThreads / 32];
    __shared__ float warp_masses[kFinishThreads / 32];
    const auto sum = common::warp_reduce_sum(mass);
    if (tid % 32 == 0) {
        warp_ids[tid / 32]    = warp_id;
        warp_masses[tid / 32] = sum;
    }
    __syncthreads();
    if (tid < 32) {
        const bool valid  = tid < kFinishThreads / 32;
        const auto winner = __reduce_min_sync(0xffffffff, valid ? warp_ids[tid] : INT_MAX);
        const auto total  = common::warp_reduce_sum(valid ? warp_masses[tid] : 0.0f);
        if (tid == 0) {
            ids[row]   = winner;
            probs[row] = __fdiv_rn(1.0f, total);
        }
    }
}

Problem make_problem(const Tensor& latent, const Weight& weight, const Tensor& bias,
                     const Tensor& base, int domain, float softcap, const Tensor& live) {
    return {static_cast<const Bf16*>(latent.data),
            static_cast<const Bf16*>(weight.qdata),
            static_cast<const Bf16*>(bias.data),
            static_cast<const Bf16*>(base.data),
            static_cast<const int*>(live.data),
            nullptr,
            latent.ne[0],
            latent.ne[1],
            weight.n,
            domain,
            softcap};
}
} // namespace

void launch_logits(const Tensor& latent, const Weight& weight, const Tensor& bias,
                   const Tensor& base, int domain, float softcap, const Tensor& live,
                   Tensor& output, CudaExecutionView execution) {
    auto p = make_problem(latent, weight, bias, base, domain, softcap, live);
    p.out  = static_cast<Bf16*>(output.data);
    launch<false>(p, execution.stream());
}

void launch_logits_argmax(const Tensor& latent, const Weight& weight, const Tensor& bias,
                          const Tensor& base, int domain, float softcap, const Tensor& live,
                          Tensor& ids, Tensor& probabilities, LogitTileResult* partials,
                          CudaExecutionView execution) {
    auto p            = make_problem(latent, weight, bias, base, domain, softcap, live);
    p.partials        = partials;
    const auto stream = execution.stream();
    launch<true>(p, stream);
    logits_argmax_finish<<<p.batch, kFinishThreads, 0, stream>>>(
        partials, p.live, p.tiles(), domain, static_cast<int*>(ids.data),
        static_cast<float*>(probabilities.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::markov
