#pragma once
#include "ops/kv_cache/kernels/hadamard.cuh"

#include "ops/softmax_attention/qsa/sparse_gqa/kernel.cuh"

namespace ninfer::ops::detail::qsa {
template <KvCacheStorage Storage, int Warps, bool Vector, int KvHeads>
__launch_bounds__(Warps * 32) __global__
    void sparse_gqa_simt_kernel(SparseGqaKernelArguments<KvHeads> a) {
    const auto q   = a.begin + a.head_query(blockIdx.x);
    const int head = a.head_index(blockIdx.x), kvh = head / 12, split = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const auto work = a.work(q);
    // This route always publishes FP32 partials; merge owns basis inversion and the gate.
    if (!work.size) return;
    if (split >= work.splits) return;
    const int start = split * work.tiles / work.splits * a.key_rows;
    const int end   = min(work.size, (split + 1) * work.tiles / work.splits * a.key_rows);
    float query[8], acc[8] = {}, maximum = -CUDART_INF_F, sum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int d = kSparseQuantized<Storage> ? lane + 32 * i : lane * 8 + i;
        query[i]    = __bfloat162float(a.query[(q * a.query_heads + head) * 256 + d]);
    }
    if constexpr (kSparseQuantized<Storage>) kv::normalized_hadamard_d256_inplace(query, lane);
    const auto table = a.tables + std::int64_t(a.table_rows[q / a.tokens]) * a.table_stride;
    for (int j = start + warp; j < end; j += Warps) {
        const int ordinal  = j / 4;
        const int g        = ordinal < work.count
                                 ? (work.count == work.complete ? ordinal : a.ids[q * 512 + ordinal])
                                 : work.complete;
        const int physical = table[g / 16], row = 4 * (g % 16) + j % 4;
        const auto* kp = a.key.data + physical * a.key.page + kvh * a.key.head + row * a.key.row;
        const auto* vp =
            a.value.data + physical * a.value.page + kvh * a.value.head + row * a.value.row;
        float keys[8], values[8];
        if constexpr (kSparseQuantized<Storage>) {
            const auto* ks = a.key_scale.data + physical * a.key_scale.page +
                             kvh * a.key_scale.head + row * a.key_scale.row;
            const auto* vs = a.value_scale.data + physical * a.value_scale.page +
                             kvh * a.value_scale.head + row * a.value_scale.row;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                keys[i]   = sparse_gqa_decode<Storage, true>(kp, ks, lane + 32 * i);
                values[i] = sparse_gqa_decode<Storage, false>(vp, vs, lane + 32 * i);
            }
        } else {
            const auto k   = sparse_gqa_expand<Storage, true, Vector>(kp, nullptr, lane * 8);
            const auto v   = sparse_gqa_expand<Storage, false, Vector>(vp, nullptr, lane * 8);
            const auto* kw = reinterpret_cast<const std::uint16_t*>(&k);
            const auto* vw = reinterpret_cast<const std::uint16_t*>(&v);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                keys[i]   = __bfloat162float(__ushort_as_bfloat16(kw[i]));
                values[i] = __half2float(__ushort_as_half(vw[i]));
            }
        }
        float dot = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) dot = fmaf(query[i], keys[i], dot);
        dot              = common::warp_sum(dot) * (common::kLog2E / 16.f);
        const float next = fmaxf(maximum, dot), alpha = common::exp2_approx(maximum - next),
                    p = common::exp2_approx(dot - next);
        sum           = fmaf(alpha, sum, p);
        maximum       = next;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = fmaf(alpha, acc[i], p * values[i]);
    }
    __shared__ float output[Warps][256], maxima[Warps], sums[Warps];
    if (lane == 0) {
        maxima[warp] = maximum;
        sums[warp]   = sum;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        output[warp][kSparseQuantized<Storage> ? lane + 32 * i : lane * 8 + i] = acc[i];
    __syncthreads();
    if (warp != 0) return;
    maximum = -CUDART_INF_F;
    sum     = 0;
#pragma unroll
    for (int w = 0; w < Warps; ++w) maximum = fmaxf(maximum, maxima[w]);
    float weights[Warps];
#pragma unroll
    for (int w = 0; w < Warps; ++w) {
        weights[w] = softmax_log2_state_weight(maxima[w], sums[w], maximum);
        sum += weights[w] * sums[w];
    }
    const auto partial = a.partial_row(q, split, head);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 8 + i * 2;
        float x = 0, y = 0;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            x += weights[w] * output[w][d];
            y += weights[w] * output[w][d + 1];
        }
        *reinterpret_cast<float2*>(a.numerator + partial * 256 + d) = make_float2(x, y);
    }
    if (lane == 0) {
        a.maximum[partial] = maximum;
        a.sum[partial]     = sum;
    }
}
} // namespace ninfer::ops::detail::qsa
