#pragma once
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
struct Q8DownWeightView {
    const std::uint8_t *codes, *scales;
};

template <class Output>
__global__ __launch_bounds__(128, 2) void q8_down_vector_kernel(const float* input,
                                                                Q8DownWeightView w, Output output) {
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32, lane = int(threadIdx.x) & 31;
    float accum[4]{};
#pragma unroll
    for (int base = lane * 8; base < 1024; base += 256) {
        const auto codes  = common::load_vec<uint2>(w.codes + std::int64_t(row) * 1024 + base);
        const float scale = __half2float(__ushort_as_half(
            common::load_ldg<std::uint16_t>(w.scales + (std::int64_t(row) * 32 + base / 32) * 2)));
        const float4 a    = common::load_vec<float4>(input + base),
                     b    = common::load_vec<float4>(input + base + 4);
#pragma unroll
        for (int word = 0; word < 2; ++word) {
            const unsigned packed = word ? codes.y : codes.x;
            const float4 x        = word ? b : a;
            accum[0] = fmaf(float(static_cast<std::int8_t>(packed)) * scale, x.x, accum[0]);
            accum[1] = fmaf(float(static_cast<std::int8_t>(packed >> 8)) * scale, x.y, accum[1]);
            accum[2] = fmaf(float(static_cast<std::int8_t>(packed >> 16)) * scale, x.z, accum[2]);
            accum[3] = fmaf(float(static_cast<std::int8_t>(packed >> 24)) * scale, x.w, accum[3]);
        }
    }
    const float value = common::warp_reduce_sum((accum[0] + accum[1]) + (accum[2] + accum[3]));
    if (lane == 0) output.store(row, 0, value);
}

template <class Schedule, class Output>
void q8_compensated_down_tma(const __nv_bfloat16* planes, const Weight& w, int tokens,
                             Output output, cudaStream_t stream, int plane_tokens = 0) {
    static_assert(Schedule::kActivationComponents == 2 && Schedule::kWeightComponents == 1);
    const Q8LinearOperands p{planes,
                             static_cast<const std::uint8_t*>(w.qdata),
                             static_cast<const std::uint8_t*>(w.scales),
                             w.n,
                             w.k,
                             plane_tokens ? plane_tokens : tokens,
                             w.padded_shape[1]};
    launch_q8_a16_tma_mma<Schedule>(p, output, LinearIdentityEpilogue{}, stream, tokens);
}

} // namespace ninfer::ops::detail::swiglu_ffn
