#pragma once

#include "ops/softmax_attention/qsa/sparse_gqa/cache.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail::qsa {

template <KvCacheStorage Storage, int KvHeads>
__device__ __forceinline__ void sparse_gqa_merge_row(const SparseGqaKernelArguments<KvHeads>& a,
                                                     std::int64_t q, int h, SparseGqaWork work,
                                                     float* weights) {
    const int lane = threadIdx.x % 32;
    float inverse;
    if (work.splits == 1) {
        inverse = __frcp_rn(a.sum[a.partial_row(q, 0, h)]);
    } else {
        float maximum = -CUDART_INF_F;
        for (int i = lane; i < work.splits; i += 32)
            maximum = fmaxf(maximum, a.maximum[a.partial_row(q, i, h)]);
        maximum   = common::warp_max(maximum);
        float sum = 0;
        for (int i = lane; i < work.splits; i += 32) {
            const auto row = a.partial_row(q, i, h);
            const auto w   = softmax_log2_state_weight(a.maximum[row], a.sum[row], maximum);
            weights[i]     = w;
            sum += w * a.sum[row];
        }
        inverse = __frcp_rn(common::warp_sum(sum));
        __syncwarp();
    }
    float values[8]{};
    for (int i = 0; i < work.splits; ++i) {
        const float weight = work.splits == 1 ? 1.f : weights[i];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const auto v = common::load_vec<float4>(a.numerator + a.partial_row(q, i, h) * 256 +
                                                    lane * 8 + 4 * j);
            values[4 * j] += weight * v.x;
            values[4 * j + 1] += weight * v.y;
            values[4 * j + 2] += weight * v.z;
            values[4 * j + 3] += weight * v.w;
        }
    }
#pragma unroll
    for (auto& v : values) v *= inverse;
    if constexpr (kSparseRotatedValue<Storage>)
        kv::normalized_hadamard_d256_contiguous_inplace(values, lane);
    sparse_gqa_store_vector(a, q, h, values);
}

template <KvCacheStorage Storage, int KvHeads>
__global__ void sparse_gqa_merge_kernel(SparseGqaKernelArguments<KvHeads> a) {
    if (!a.numerator) return;
    const auto q = a.begin + a.head_query(blockIdx.x);
    const int h = a.head_index(blockIdx.x), d = threadIdx.x * 2;
    const auto work = a.work(q);
    if (a.direct_single && work.size && work.splits <= 1) return;
    if (!work.size) {
        *reinterpret_cast<unsigned*>(a.out + (q * a.query_heads + h) * 256 + d) = 0;
        return;
    }
    __shared__ float weights[64], denominator;
    // A small partial set needs one warp, including the FP32 inverse value basis.
    // Preserve the four-warp load parallelism for large one-query reductions.
    if constexpr (NINFER_TARGET_RTX5090 && (KvHeads == 2 || KvHeads == 4)) {
        if (work.splits <= 16) {
            if (threadIdx.x < 32) sparse_gqa_merge_row<Storage>(a, q, h, work, weights);
            return;
        }
    }
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float m        = -CUDART_INF_F;
        for (int s = lane; s < work.splits; s += 32)
            m = fmaxf(m, a.maximum[a.partial_row(q, s, h)]);
        m         = common::warp_max(m);
        float sum = 0;
        for (int s = lane; s < work.splits; s += 32) {
            const auto i  = a.partial_row(q, s, h);
            const float w = softmax_log2_state_weight(a.maximum[i], a.sum[i], m);
            weights[s]    = w;
            sum += w * a.sum[i];
        }
        sum = common::warp_sum(sum);
        if (lane == 0) denominator = sum;
    }
    __syncthreads();
    float x = 0, y = 0;
    for (int s = 0; s < work.splits; ++s) {
        const auto v =
            *reinterpret_cast<const float2*>(a.numerator + a.partial_row(q, s, h) * 256 + d);
        x += weights[s] * v.x;
        y += weights[s] * v.y;
    }
    const float inv = __frcp_rn(denominator);
    if constexpr (kSparseRotatedValue<Storage>) {
        __shared__ float row[256];
        row[d]     = x * inv;
        row[d + 1] = y * inv;
        __syncthreads();
        if (threadIdx.x < 32) {
            float values[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) values[i] = row[threadIdx.x + 32 * i];
            sparse_gqa_inverse_gate(a, q, h, values);
        }
    } else
        sparse_gqa_store(a, q, h, d, x * inv, y * inv);
}

template <KvCacheStorage Storage, int Heads, int KvHeads>
__global__ void sparse_gqa_merge_warp_kernel(SparseGqaKernelArguments<KvHeads> a) {
    if (!a.numerator) return;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int groups = (a.query_heads + Heads - 1) / Heads;
    const auto q         = a.begin + blockIdx.x / groups;
    const int h          = blockIdx.x % groups * Heads + warp;
    if (h >= a.query_heads) return;
    const auto work = a.work(q);
    if (a.direct_single && work.size && work.splits <= 1) return;
    if (!work.size) {
        reinterpret_cast<uint4*>(a.out + (q * a.query_heads + h) * 256)[lane] =
            make_uint4(0, 0, 0, 0);
        return;
    }
    __shared__ float weights[Heads][64];
    sparse_gqa_merge_row<Storage>(a, q, h, work, weights[warp]);
}

} // namespace ninfer::ops::detail::qsa
