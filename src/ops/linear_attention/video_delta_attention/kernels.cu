#include "ops/linear_attention/video_delta_attention/launch.h"
#include "ops/common/host/launch.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include <mma.h>
#include <algorithm>
#include "factor.cuh"
#include "scan.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.cuh"

namespace ninfer::ops::detail::vda {
namespace {
namespace wm               = nvcuda::wmma;
using BF                   = __nv_bfloat16;
using Acc                  = wm::fragment<wm::accumulator, 16, 16, 16, float>;
using AR                   = wm::fragment<wm::matrix_a, 16, 16, 16, BF, wm::row_major>;
using AC                   = wm::fragment<wm::matrix_a, 16, 16, 16, BF, wm::col_major>;
using BR                   = wm::fragment<wm::matrix_b, 16, 16, 16, BF, wm::row_major>;
using BC                   = wm::fragment<wm::matrix_b, 16, 16, 16, BF, wm::col_major>;
constexpr int SharedStride = 136;

__device__ const BF* row(Vector x, int b, int h, std::int64_t token) {
    return x.data + b * x.batch_stride + h * x.head_stride + token * x.token_stride;
}

__device__ void split(float x, BF& hi, BF& lo) {
    hi = __float2bfloat16_rn(x);
    lo = __float2bfloat16_rn(x - __bfloat162float(hi));
}

// [K or V]^T @ (beta*K): two BF16 contractions preserve the weighted operand's
// residual instead of globally materializing FP32 K and beta*K. FP32 accumulators.
template <int BK, bool Vectorized, bool Partitioned = false>
__global__ __launch_bounds__(256, 2) void frame_stats(Arguments a, float* p, float* j, int hb,
                                                      int hw, int batch, int frame_begin,
                                                      int partitions) {
    __shared__ BF left[BK * SharedStride], right_hi[BK * SharedStride], right_lo[BK * SharedStride];
    const int f = frame_begin + blockIdx.y, h = blockIdx.x, kind = blockIdx.z, tid = threadIdx.x,
              warp  = tid / 32;
    const int wmrow = warp / 4, wmcol = warp % 4;
    const auto location = resolve_group(a, batch, hb + h);
    const auto tiles    = (std::int64_t(a.tokens) + BK - 1) / BK;
    const auto begin =
        Partitioned ? std::int64_t(f) * tiles / partitions * BK : std::int64_t(f) * a.per_frame;
    const auto end =
        Partitioned ? min(std::int64_t(a.tokens), std::int64_t(f + 1) * tiles / partitions * BK)
                    : std::int64_t(f + 1) * a.per_frame;
    Acc c[4][2];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n) wm::fill_fragment(c[m][n], 0.f);
    for (std::int64_t start = begin; start < end; start += BK) {
        if constexpr (Vectorized) {
            for (int i = tid * 8; i < BK * 128; i += 256 * 8) {
                const int u = i / 128, d = i % 128;
                const auto token       = start + u;
                const int shared_index = u * SharedStride + d;
                alignas(16) BF key[8]{}, value[8]{}, high[8], low[8];
                float weight = 0;
                if (token < end) {
                    common::store_vec(key, common::load_vec<uint4>(
                                               row(a.k, location.batch, location.head, token) + d));
                    if (kind)
                        common::store_vec(value,
                                          common::load_vec<uint4>(
                                              row(a.v, location.batch, location.head, token) + d));
                    weight = a.beta[(std::int64_t(location.batch) * a.tokens + token) * a.heads +
                                    location.head];
                }
#pragma unroll
                for (int element = 0; element < 8; ++element)
                    split(__bfloat162float(key[element]) * weight, high[element], low[element]);
                common::store_vec(left + shared_index, common::load_vec<uint4>(kind ? value : key));
                common::store_vec(right_hi + shared_index, common::load_vec<uint4>(high));
                common::store_vec(right_lo + shared_index, common::load_vec<uint4>(low));
            }
        } else {
            for (int i = tid; i < BK * 128; i += 256) {
                const int u = i / 128, d = i % 128;
                const auto token = start + u;
                float x          = 0;
                BF l             = __float2bfloat16_rn(0.f);
                if (token < end) {
                    l = row(kind ? a.v : a.k, location.batch, location.head, token)[d];
                    x = __bfloat162float(row(a.k, location.batch, location.head, token)[d]) *
                        a.beta[(std::int64_t(location.batch) * a.tokens + token) * a.heads +
                               location.head];
                }
                const int shared_index = u * SharedStride + d;
                left[shared_index]     = l;
                split(x, right_hi[shared_index], right_lo[shared_index]);
            }
        }
        __syncthreads();
#pragma unroll
        for (int reduction = 0; reduction < BK; reduction += 16) {
            BR bh[2], bl[2];
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                wm::load_matrix_sync(bh[n],
                                     right_hi + reduction * SharedStride + (wmcol + n * 4) * 16,
                                     SharedStride);
                wm::load_matrix_sync(bl[n],
                                     right_lo + reduction * SharedStride + (wmcol + n * 4) * 16,
                                     SharedStride);
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                AC x;
                wm::load_matrix_sync(x, left + reduction * SharedStride + (wmrow + m * 2) * 16,
                                     SharedStride);
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    wm::mma_sync(c[m][n], x, bh[n], c[m][n]);
                    wm::mma_sync(c[m][n], x, bl[n], c[m][n]);
                }
            }
        }
        __syncthreads();
    }
    float* dest = (kind ? j : p) + (std::int64_t(f) * hw + h) * 128 * 128;
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n)
            wm::store_matrix_sync(dest + (wmrow + m * 2) * 16 * 128 + (wmcol + n * 4) * 16, c[m][n],
                                  128, wm::mem_row_major);
}

__global__ void reduce_state_statistics(float* gram, float* write, int heads, int partitions) {
    const int element = blockIdx.x * 256 + threadIdx.x;
    const int head    = blockIdx.y;
    float* matrices   = blockIdx.z ? write : gram;
    float sum         = 0;
    for (int part = 0; part < partitions; ++part)
        sum += matrices[(std::int64_t(part) * heads + head) * 128 * 128 + element];
    // Each element has one writer, and no other reduction reads that element.
    matrices[head * 128 * 128 + element] = sum;
}

// Build the effective state once per frame, in the transition buffer that the
// scan has finished consuming. Two BF16 components occupy exactly one FP32 matrix.
__global__ void prepare_readout(Arguments a, const float* boundaries, BF* effective, int hb, int hw,
                                int batch, int chunks) {
    __shared__ float decay[256];
    const int f = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, d = tid % 128;
    const auto location = resolve_group(a, batch, hb + h);
    const auto c        = (std::int64_t(f) + a.window.frame_offset) / a.window.chunk_frames;
    const auto lo =
        max(std::int64_t(0), (c - a.window.radius) * a.window.chunk_frames - a.window.frame_offset);
    const auto hi =
        min(std::int64_t(a.frames - 1),
            (c + a.window.radius + 1) * a.window.chunk_frames - a.window.frame_offset - 1);
    float product    = 1;
    const auto begin = tid < 128 ? lo : std::int64_t(f);
    const auto end   = tid < 128 ? std::int64_t(f) : hi;
    for (auto i = begin; i <= end; ++i)
        product *= fmaxf(
            a.alpha[((std::int64_t(location.batch) * a.frames + i) * a.heads + location.head) *
                        128 +
                    d],
            1e-12f);
    decay[tid] = product;
    __syncthreads();
    const auto left_chunk  = c - a.window.radius - 1;
    const auto right_chunk = c + a.window.radius + 1;
    const float* seed      = a.initial + (std::int64_t(batch) * a.heads + hb + h) * 128 * 128;
    const float* left = left_chunk >= 0 ? boundaries + (left_chunk * hw + h) * 128 * 128 : seed;
    const float* right =
        right_chunk < chunks ? boundaries + ((chunks + right_chunk) * hw + h) * 128 * 128 : seed;
    BF* high = effective + (std::int64_t(f) * hw + h) * 2 * 128 * 128;
    BF* low  = high + 128 * 128;
    for (int i = tid; i < 128 * 128; i += 256) {
        const int key = i % 128;
        split(left[i] * decay[key] + right[i] * decay[128 + key], high[i], low[i]);
    }
}

__device__ __forceinline__ float apply_sigmoid_gate(float value, float gate) {
    // The scaled form keeps a small sigmoid from underflowing before multiplication.
    if (gate < -80.f) return (value * 0x1p-128f) * expf(gate + 88.722839111673f);
    const float e = __expf(-fabsf(gate)), denominator = 1.f + e;
    float reciprocal = common::rcp_approx_ftz(denominator);
    reciprocal       = fmaf(reciprocal, fmaf(-denominator, reciprocal, 1.f), reciprocal);
    return value * (gate >= 0 ? reciprocal : e * reciprocal);
}

__device__ __forceinline__ float4 load_bf16_quad(const BF* pointer) {
    alignas(8) BF values[4];
    if (reinterpret_cast<std::uintptr_t>(pointer) % 8 == 0)
        common::store_vec(values, common::load_vec<uint2>(pointer));
    else {
#pragma unroll
        for (int i = 0; i < 4; ++i) values[i] = pointer[i];
    }
    return make_float4(__bfloat162float(values[0]), __bfloat162float(values[1]),
                       __bfloat162float(values[2]), __bfloat162float(values[3]));
}

template <int BQ, int BK, bool Gated>
__global__ __launch_bounds__(256, 2) void readout_kernel(Arguments a, const BF* effective, int hb,
                                                         int hw, int batch, int frame_begin) {
    extern __shared__ __align__(32) unsigned char storage[];
    constexpr int Stride = BK + 8;
    auto* mh             = reinterpret_cast<BF*>(storage);
    auto* ml             = mh + 128 * Stride;
    auto* qs             = ml + 128 * Stride;
    const int f = frame_begin + blockIdx.z, h = blockIdx.y, begin = blockIdx.x * BQ;
    const auto location = resolve_group(a, batch, hb + h);
    const int tid = threadIdx.x, warp = tid / 32;
    const int wmrow = warp / 4, wmcol = warp % 4;
    const BF* high = effective + (std::int64_t(f) * hw + h) * 2 * 128 * 128;
    const BF* low  = high + 128 * 128;
    Acc c[BQ / 32][2];
#pragma unroll
    for (int m = 0; m < BQ / 32; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n) wm::fill_fragment(c[m][n], 0.f);
#pragma unroll 1
    for (int tile = 0; tile < 128; tile += BK) {
        for (int i = tid * 8; i < 128 * BK; i += 256 * 8) {
            const int value = i / BK, key = i % BK;
            common::cp_async<16, Cache::ca>(mh + value * Stride + key,
                                            high + value * 128 + tile + key);
            common::cp_async<16, Cache::ca>(ml + value * Stride + key,
                                            low + value * 128 + tile + key);
        }
        for (int i = tid * 8; i < BQ * BK; i += 256 * 8) {
            const int u = begin + i / BK, key = i % BK;
            BF* destination = qs + i / BK * Stride + key;
            if (u < a.per_frame) {
                const BF* source =
                    row(a.q, location.batch, location.head, std::int64_t(f) * a.per_frame + u) +
                    tile + key;
                if (reinterpret_cast<std::uintptr_t>(source) % 16 == 0) {
                    common::cp_async<16, Cache::ca>(destination, source);
                } else {
#pragma unroll
                    for (int element = 0; element < 8; ++element)
                        destination[element] = source[element];
                }
            } else {
                common::store_vec(destination, make_uint4(0, 0, 0, 0));
            }
        }
        common::cp_commit();
        common::cp_wait<0>();
        __syncthreads();
#pragma unroll 1
        for (int k = 0; k < BK; k += 16) {
            BC bh[2], bl[2];
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                wm::load_matrix_sync(bh[n], mh + (wmcol + n * 4) * 16 * Stride + k, Stride);
                wm::load_matrix_sync(bl[n], ml + (wmcol + n * 4) * 16 * Stride + k, Stride);
            }
#pragma unroll
            for (int m = 0; m < BQ / 32; ++m) {
                AR x;
                wm::load_matrix_sync(x, qs + (wmrow + m * 2) * 16 * Stride + k, Stride);
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    wm::mma_sync(c[m][n], x, bh[n], c[m][n]);
                    wm::mma_sync(c[m][n], x, bl[n], c[m][n]);
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();
    auto* result = reinterpret_cast<float*>(storage); // matrix staging is now dead
#pragma unroll
    for (int m = 0; m < BQ / 32; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n)
            wm::store_matrix_sync(result + (wmrow + m * 2) * 16 * 128 + (wmcol + n * 4) * 16,
                                  c[m][n], 128, wm::mem_row_major);
    __syncthreads();
    const int lane = tid % 32;
    for (int r = warp; r < BQ; r += 8) {
        const auto token = std::int64_t(f) * a.per_frame + begin + r;
        if (begin + r >= a.per_frame) continue;
        const int d       = lane * 4;
        const auto values = common::load_vec<float4>(result + r * 128 + d);
        float x[4]{values.x, values.y, values.z, values.w};
        if constexpr (Gated) {
            float sum = 0;
#pragma unroll
            for (int n = 0; n < 4; ++n) sum = fmaf(x[n], x[n], sum);
#pragma unroll
            for (int offset = 16; offset; offset /= 2)
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            const float inv = rsqrtf(sum * (1.f / 128) + a.eps);
            const auto gates =
                load_bf16_quad(row(a.gate, location.batch, location.head, token) + d);
            const auto weights = load_bf16_quad(a.weight + d);
            const float g[4]{gates.x, gates.y, gates.z, gates.w};
            const float w[4]{weights.x, weights.y, weights.z, weights.w};
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const float weighted = x[n] * inv * w[n];
                x[n]                 = apply_sigmoid_gate(weighted, g[n]);
            }
        }
        BF* output =
            a.out +
            ((std::int64_t(location.batch) * a.tokens + token) * a.heads + location.head) * 128 + d;
        if (reinterpret_cast<std::uintptr_t>(output) % 8 == 0)
            common::store_vec(output, make_uint2(common::pack_bf16x2(x[0], x[1]),
                                                 common::pack_bf16x2(x[2], x[3])));
        else {
#pragma unroll
            for (int n = 0; n < 4; ++n) output[n] = __float2bfloat16_rn(x[n]);
        }
    }
}
} // namespace

void launch(const Arguments& a, float* p, float* j, float* boundary, int hb, int hw, int batch,
            bool init, cudaStream_t stream) {
    const auto aligned = [](Vector x) {
        return reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 && x.head_stride % 8 == 0 &&
               x.token_stride % 8 == 0 && x.batch_stride % 8 == 0;
    };
    if (init) {
        const int partitions = state_statistics_partitions(hw, a.tokens);
        const dim3 grid(hw, partitions, 2);
        if (aligned(a.k) && aligned(a.v))
            frame_stats<32, true, true>
                <<<grid, 256, 0, stream>>>(a, p, j, hb, hw, batch, 0, partitions);
        else
            frame_stats<32, false, true>
                <<<grid, 256, 0, stream>>>(a, p, j, hb, hw, batch, 0, partitions);
        if (partitions > 1)
            reduce_state_statistics<<<dim3(64, hw, 2), 256, 0, stream>>>(p, j, hw, partitions);
    } else
        for (std::int64_t f = 0; f < a.frames; f += 65535) {
            const dim3 grid(hw, std::min<std::int64_t>(65535, a.frames - f), 2);
            if (aligned(a.k) && aligned(a.v))
                frame_stats<32, true><<<grid, 256, 0, stream>>>(a, p, j, hb, hw, batch, f, 1);
            else
                frame_stats<32, false><<<grid, 256, 0, stream>>>(a, p, j, hb, hw, batch, f, 1);
        }
    constexpr auto factor = factor_internal::factor_kernel;
    common::prepare_shared<factor_internal::kXsBytes, factor, true>();
    factor<<<dim3(a.frames, hw), factor_internal::kBlockSize, factor_internal::kXsBytes, stream>>>(
        a, p, j, hb, hw, batch, init);
    if (!init) {
        const int chunks =
            (std::int64_t(a.frames) + a.window.frame_offset + a.window.chunk_frames - 1) /
            a.window.chunk_frames;
        constexpr int BQ = 64, ReadK = 64;
        scan_internal::launch<16, 16, 2>(a, p, j, boundary, hb, hw, batch, chunks, stream);
        auto* effective = reinterpret_cast<BF*>(p);
        prepare_readout<<<dim3(a.frames, hw), 256, 0, stream>>>(a, boundary, effective, hb, hw,
                                                                batch, chunks);
        constexpr int read_shared =
            std::max((2 * 128 + BQ) * (ReadK + 8) * int(sizeof(BF)), BQ * 128 * int(sizeof(float)));
        common::prepare_shared<read_shared, readout_kernel<BQ, ReadK, false>, true>();
        common::prepare_shared<read_shared, readout_kernel<BQ, ReadK, true>, true>();
        for (std::int64_t f = 0; f < a.frames; f += 65535) {
            const dim3 grid((std::int64_t(a.per_frame) + BQ - 1) / BQ, hw,
                            std::min<std::int64_t>(65535, a.frames - f));
            if (a.weight)
                readout_kernel<BQ, ReadK, true>
                    <<<grid, 256, read_shared, stream>>>(a, effective, hb, hw, batch, f);
            else
                readout_kernel<BQ, ReadK, false>
                    <<<grid, 256, read_shared, stream>>>(a, effective, hb, hw, batch, f);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::vda
