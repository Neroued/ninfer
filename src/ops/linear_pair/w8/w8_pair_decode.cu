#include "ops/linear_pair/w8/w8_pair_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows          = 1024;
constexpr int kHidden        = 2048;
constexpr int kGroupsPerRow  = kHidden / 32;
constexpr int kValuesPerLane = 8;

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w8_pair_k2048_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ first_codes,
    const std::uint8_t* __restrict__ first_scales, const std::uint8_t* __restrict__ second_codes,
    const std::uint8_t* __restrict__ second_scales, __nv_bfloat16* __restrict__ first_out,
    __nv_bfloat16* __restrict__ second_out) {
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase = kValuesPerPhase / 32;
    constexpr int kPhases         = kHidden / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int row        = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const auto* code_a   = first_codes + static_cast<std::int64_t>(row) * kHidden;
    const auto* code_b   = second_codes + static_cast<std::int64_t>(row) * kHidden;
    const auto* scales_a = first_scales + static_cast<std::int64_t>(row) * kGroupsPerRow * 2;
    const auto* scales_b = second_scales + static_cast<std::int64_t>(row) * kGroupsPerRow * 2;

    float acc_a = 0.0f;
    float acc_b = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned scale_bits_a = 0;
        unsigned scale_bits_b = 0;
        if (lane < kGroupsPerPhase) {
            const std::int64_t offset =
                static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2;
            scale_bits_a = *reinterpret_cast<const std::uint16_t*>(scales_a + offset);
            scale_bits_b = *reinterpret_cast<const std::uint16_t*>(scales_b + offset);
        }
        scale_bits_a        = __shfl_sync(kMask, scale_bits_a, lane >> 2);
        scale_bits_b        = __shfl_sync(kMask, scale_bits_b, lane >> 2);
        const float scale_a = __half2float(__ushort_as_half(scale_bits_a));
        const float scale_b = __half2float(__ushort_as_half(scale_bits_b));

        const int phase_k      = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 qa         = load_vec<uint2>(code_a + phase_k);
        const uint2 qb         = load_vec<uint2>(code_b + phase_k);
        const uint4 xv         = load_vec<uint4>(x + phase_k);
        const float2 values[4] = {
            bf16x2_bits_to_float2(xv.x),
            bf16x2_bits_to_float2(xv.y),
            bf16x2_bits_to_float2(xv.z),
            bf16x2_bits_to_float2(xv.w),
        };
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t word_a = (&qa.x)[word_index];
            const std::uint32_t word_b = (&qb.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int shift              = byte * 8;
                const float2 activation_pair = values[word_index * 2 + (byte >> 1)];
                const float activation = (byte & 1) == 0 ? activation_pair.x : activation_pair.y;
                const float weight_a =
                    static_cast<float>(static_cast<std::int8_t>(word_a >> shift)) * scale_a;
                const float weight_b =
                    static_cast<float>(static_cast<std::int8_t>(word_b >> shift)) * scale_b;
                acc_a = fmaf(weight_a, activation, acc_a);
                acc_b = fmaf(weight_b, activation, acc_b);
            }
        }
    }

    acc_a = warp_reduce_sum(acc_a);
    acc_b = warp_reduce_sum(acc_b);
    if (lane == 0) {
        first_out[row]  = __float2bfloat16_rn(acc_a);
        second_out[row] = __float2bfloat16_rn(acc_b);
    }
}

template <int RowsPerCta>
void launch_decode(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                   Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    if (x.ne[0] != kHidden || x.ne[1] != 1 || first_out.ne[0] != kRows || first_out.ne[1] != 1 ||
        second_out.ne[0] != kRows || second_out.ne[1] != 1 || first_weight.n != kRows ||
        first_weight.k != kHidden || second_weight.n != kRows || second_weight.k != kHidden) {
        throw std::invalid_argument("W8 pair decode requires [1024,2048] and T=1");
    }
    static_assert((kRows % RowsPerCta) == 0);
    w8_pair_k2048_decode_kernel<RowsPerCta><<<kRows / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(first_weight.qdata),
        static_cast<const std::uint8_t*>(first_weight.scales),
        static_cast<const std::uint8_t*>(second_weight.qdata),
        static_cast<const std::uint8_t*>(second_weight.scales),
        static_cast<__nv_bfloat16*>(first_out.data), static_cast<__nv_bfloat16*>(second_out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_pair_decode_r4_launch(const Tensor& x, const Weight& first_weight,
                              const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                              cudaStream_t stream) {
    launch_decode<4>(x, first_weight, second_weight, first_out, second_out, stream);
}

void w8_pair_decode_r8_launch(const Tensor& x, const Weight& first_weight,
                              const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                              cudaStream_t stream) {
    launch_decode<8>(x, first_weight, second_weight, first_out, second_out, stream);
}

void w8_pair_decode_r16_launch(const Tensor& x, const Weight& first_weight,
                               const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                               cudaStream_t stream) {
    launch_decode<16>(x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops::detail
