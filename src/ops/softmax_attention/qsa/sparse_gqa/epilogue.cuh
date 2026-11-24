#pragma once

#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/arguments.h"

namespace ninfer::ops::detail::qsa {

template <int KvHeads>
__device__ __forceinline__ void sparse_gqa_store(const SparseGqaKernelArguments<KvHeads>& a,
                                                 std::int64_t q, int h, int d, float x, float y) {
    const auto offset = (q * a.query_heads + h) * 256 + d;
    const float2 gate =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(a.gate + offset));
    x *= __frcp_rn(1.f + common::exp2_approx(-gate.x * common::kLog2E));
    y *= __frcp_rn(1.f + common::exp2_approx(-gate.y * common::kLog2E));
    *reinterpret_cast<unsigned*>(a.out + offset) = common::pack_bf16x2(x, y);
}

// A warp holds a complete FP32 value row. Basis inversion precedes the elementwise gate;
// only the final public output crosses BF16 storage.
template <int KvHeads>
__device__ __forceinline__ void sparse_gqa_inverse_gate(const SparseGqaKernelArguments<KvHeads>& a,
                                                        std::int64_t q, int h, float (&values)[8]) {
    const int lane = threadIdx.x % 32;
    kv::normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto index = (q * a.query_heads + h) * 256 + lane + 32 * i;
        const float gate = __bfloat162float(a.gate[index]);
        a.out[index]     = __float2bfloat16_rn(
            values[i] * __frcp_rn(1.f + common::exp2_approx(-gate * common::kLog2E)));
    }
}

template <int Warps, int Queries, int KvHeads>
__device__ void sparse_gqa_inverse_rows(const SparseGqaKernelArguments<KvHeads>& a,
                                        std::int64_t first, std::int64_t last, int head,
                                        const float* rows) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int i = warp; i < Queries * 12; i += Warps) {
        const auto q = first + i / 12;
        const int h  = i % 12;
        if (q >= last) continue;
        float values[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) values[d] = rows[(i / 12 * 16 + h) * 256 + lane + 32 * d];
        sparse_gqa_inverse_gate(a, q, head * 12 + h, values);
    }
}

} // namespace ninfer::ops::detail::qsa
