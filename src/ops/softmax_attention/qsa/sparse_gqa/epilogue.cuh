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

// One warp writes its six-head group. Fetch the following head's gates while
// processing the current FP32 row, rather than serializing 24 narrow gate loads.
template <int KvHeads>
__device__ __forceinline__ void
sparse_gqa_compact_store_vector(const SparseGqaKernelArguments<KvHeads>& a, std::int64_t q,
                                int head, const float* rows) {
    static_assert(KvHeads == 4);
    const int lane  = threadIdx.x % 32;
    const auto base = (q * a.query_heads + head * a.ratio) * 256 + lane * 8;
    uint4 gates     = common::load_vec<uint4>(a.gate + base);
#pragma unroll 1
    for (int h = 0; h < a.ratio; ++h) {
        uint4 next{};
        if (h + 1 < a.ratio) next = common::load_vec<uint4>(a.gate + base + (h + 1) * 256);
        const auto low  = common::load_vec<float4>(rows + h * 256 + lane * 8);
        const auto high = common::load_vec<float4>(rows + h * 256 + lane * 8 + 4);
        const float values[8]{low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
        const unsigned words[4]{gates.x, gates.y, gates.z, gates.w};
        unsigned output[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float g0 = __uint_as_float(words[i] << 16);
            const float g1 = __uint_as_float(words[i] & 0xffff0000u);
            const float x =
                values[2 * i] * __frcp_rn(1.F + common::exp2_approx(-g0 * common::kLog2E));
            const float y =
                values[2 * i + 1] * __frcp_rn(1.F + common::exp2_approx(-g1 * common::kLog2E));
            output[i] = common::pack_bf16x2(x, y);
        }
        common::store_vec(a.out + base + h * 256,
                          make_uint4(output[0], output[1], output[2], output[3]));
        gates = next;
    }
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
    for (int i = warp; i < Queries * a.ratio; i += Warps) {
        const auto q = first + i / a.ratio;
        const int h  = i % a.ratio;
        if (q >= last) continue;
        float values[8];
#pragma unroll
        for (int d = 0; d < 8; ++d) values[d] = rows[(i / a.ratio * 16 + h) * 256 + lane + 32 * d];
        sparse_gqa_inverse_gate(a, q, head * a.ratio + h, values);
    }
}

} // namespace ninfer::ops::detail::qsa
