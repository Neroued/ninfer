#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kIntermediate = 6144;
constexpr int kK            = 2048;
constexpr int kGroupsPerRow = kK / 32;

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w8_linear_swiglu_decode_pair_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    constexpr int kValuesPerLane  = 8;
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase = kValuesPerPhase / 32;
    constexpr int kPhases         = kK / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int row           = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const int up_row        = row + kIntermediate;
    const auto* gate_row    = codes + static_cast<std::int64_t>(row) * kK;
    const auto* up_codes    = codes + static_cast<std::int64_t>(up_row) * kK;
    const auto* gate_scales = scales + static_cast<std::int64_t>(row) * kGroupsPerRow * 2;
    const auto* up_scales   = scales + static_cast<std::int64_t>(up_row) * kGroupsPerRow * 2;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned gate_scale_bits = 0;
        unsigned up_scale_bits   = 0;
        if (lane < kGroupsPerPhase) {
            gate_scale_bits = *reinterpret_cast<const std::uint16_t*>(
                gate_scales + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
            up_scale_bits = *reinterpret_cast<const std::uint16_t*>(
                up_scales + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
        }
        gate_scale_bits        = __shfl_sync(kMask, gate_scale_bits, lane >> 2);
        up_scale_bits          = __shfl_sync(kMask, up_scale_bits, lane >> 2);
        const float gate_scale = __half2float(__ushort_as_half(gate_scale_bits));
        const float up_scale   = __half2float(__ushort_as_half(up_scale_bits));

        const int phase_k       = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 gate_packed = load_vec<uint2>(gate_row + phase_k);
        const uint2 up_packed   = load_vec<uint2>(up_codes + phase_k);
        const uint4 values      = load_vec<uint4>(x + phase_k);
        const float2 xv[4]      = {
            bf16x2_bits_to_float2(values.x),
            bf16x2_bits_to_float2(values.y),
            bf16x2_bits_to_float2(values.z),
            bf16x2_bits_to_float2(values.w),
        };
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t gate_word = (&gate_packed.x)[word_index];
            const std::uint32_t up_word   = (&up_packed.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const int shift              = byte * 8;
                const float2 activation_pair = xv[word_index * 2 + (byte >> 1)];
                const float activation = (byte & 1) == 0 ? activation_pair.x : activation_pair.y;
                const float gate_value =
                    static_cast<float>(static_cast<std::int8_t>(gate_word >> shift)) * gate_scale;
                const float up_value =
                    static_cast<float>(static_cast<std::int8_t>(up_word >> shift)) * up_scale;
                gate_acc = fmaf(gate_value, activation, gate_acc);
                up_acc   = fmaf(up_value, activation, up_acc);
            }
        }
    }

    gate_acc = warp_reduce_sum(gate_acc);
    up_acc   = warp_reduce_sum(up_acc);
    if (lane == 0) { out[row] = __float2bfloat16_rn(silu(gate_acc) * up_acc); }
}

template <int RowsPerCta>
void launch_decode(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert(kIntermediate % RowsPerCta == 0);
    w8_linear_swiglu_decode_pair_kernel<RowsPerCta>
        <<<kIntermediate / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_linear_swiglu_decode_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream) {
    launch_decode<8>(x, w, out, stream);
}

void w8_linear_swiglu_decode_pair_r4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream) {
    launch_decode<4>(x, w, out, stream);
}

void w8_linear_swiglu_decode_pair_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream) {
    launch_decode<16>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
