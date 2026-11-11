#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/normalization/rmsnorm/kernel/epilogue.cuh"

namespace ninfer::ops::detail::rmsnorm {

template <int Words>
struct alignas(Words * sizeof(unsigned)) NormVector {
    unsigned words[Words];
};

// Implements: include/ninfer/ops/rmsnorm.h. Match: RTX 5090, aligned D=20480 rows.
// A whole row remains in packed BF16 registers across the FP32 reduction. The vector width
// is an alignment requirement owned by the selector, not a restriction on the public Op.
template <Epilogue Kind, int D, int Block, int Words>
__launch_bounds__(Block) __global__
    void rmsnorm_vector_cta_kernel(const __nv_bfloat16* __restrict__ x,
                                   const __nv_bfloat16* __restrict__ weight,
                                   __nv_bfloat16* __restrict__ out, float eps) {
    static_assert(!is_gated<Kind>);
    static_assert(Words == 2 || Words == 4);
    static_assert(D % (Block * Words * 2) == 0);
    constexpr int Groups = D / (Block * Words * 2);
    using Vector         = NormVector<Words>;
    const auto row_base  = static_cast<std::int64_t>(blockIdx.x) * D;
    Vector values[Groups], weights[Groups];
    float sums[Words]{};
#pragma unroll
    for (int k = 0; k < Groups; ++k) {
        const int feature = (k * Block + threadIdx.x) * Words * 2;
        values[k]         = common::load_vec<Vector>(x + row_base + feature);
        weights[k]        = common::load_vec<Vector>(weight + feature);
#pragma unroll
        for (int w = 0; w < Words; ++w) {
            unsigned low, high;
            common::unpack_bf16x2_to_fp32_bits(values[k].words[w], low, high);
            const float lo = __uint_as_float(low), hi = __uint_as_float(high);
            sums[w] += lo * lo + hi * hi;
        }
    }
    float sum = 0.F;
#pragma unroll
    for (int w = 0; w < Words; ++w) sum += sums[w];
    __shared__ float warp_sums[Block / common::kWarpSize];
    __shared__ float inverse;
    const float reduced = common::block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) inverse = rsqrtf(reduced * (1.F / static_cast<float>(D)) + eps);
    __syncthreads();
    const float inv = inverse;
#pragma unroll
    for (int k = 0; k < Groups; ++k) {
        const int feature = (k * Block + threadIdx.x) * Words * 2;
        const Vector gain = weights[k];
        Vector result;
#pragma unroll
        for (int w = 0; w < Words; ++w) {
            unsigned low, high, weight_low, weight_high;
            common::unpack_bf16x2_to_fp32_bits(values[k].words[w], low, high);
            common::unpack_bf16x2_to_fp32_bits(gain.words[w], weight_low, weight_high);
            const float lo = __uint_as_float(low);
            const float hi = __uint_as_float(high);
            const float wl = __uint_as_float(weight_low), wh = __uint_as_float(weight_high);
            result.words[w] = common::pack_bf16x2(apply_epilogue<Kind>(lo, inv, wl, 0.F),
                                                  apply_epilogue<Kind>(hi, inv, wh, 0.F));
        }
        common::store_vec(out + row_base + feature, result);
    }
}

} // namespace ninfer::ops::detail::rmsnorm
