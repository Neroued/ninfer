#include "ops/linear_add/w8/w8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlock = 8;
constexpr int kStages       = 2;
constexpr int kDecodeK      = 6144;
constexpr int kDecodeGroups = kDecodeK / 32;

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w8_linear_add_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ residual) {
    constexpr int kValuesPerLane  = 8;
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase = kValuesPerPhase / 32;
    constexpr int kPhases         = kDecodeK / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int row        = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const auto* code_row = codes + static_cast<std::int64_t>(row) * kDecodeK;
    const auto* scale_row =
        scales + static_cast<std::int64_t>(row) * kDecodeGroups * sizeof(std::uint16_t);

    float acc = 0.0f;
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned scale_bits = 0;
        if (lane < kGroupsPerPhase) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
        }
        scale_bits        = __shfl_sync(kMask, scale_bits, lane >> 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const int phase_k  = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 packed = load_vec<uint2>(code_row + phase_k);
        const uint4 values = load_vec<uint4>(x + phase_k);
        const float2 xv[4] = {
            bf16x2_bits_to_float2(values.x),
            bf16x2_bits_to_float2(values.y),
            bf16x2_bits_to_float2(values.z),
            bf16x2_bits_to_float2(values.w),
        };
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t word = (&packed.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const float2 activation_pair = xv[word_index * 2 + (byte >> 1)];
                const float activation = (byte & 1) == 0 ? activation_pair.x : activation_pair.y;
                const float weight_value =
                    static_cast<float>(static_cast<std::int8_t>(word >> (byte * 8))) * scale;
                acc = fmaf(weight_value, activation, acc);
            }
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { residual[row] = __float2bfloat16_rn(__bfloat162float(residual[row]) + acc); }
}

template <int RowsPerCta>
void launch_decode(const Tensor& x, const Weight& w, Tensor& residual_out, cudaStream_t stream) {
    static_assert((2048 % RowsPerCta) == 0);
    w8_linear_add_decode_kernel<RowsPerCta><<<2048 / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(residual_out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int ColsPerTile, W8KernelVariant Variant>
void launch_tt(const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
               __nv_bfloat16* residual_out, std::int32_t rows, std::int32_t k, std::int32_t cols,
               std::int32_t padded_k, std::int32_t full_slabs, cudaStream_t stream) {
    constexpr int kThreads = kRowsPerBlock * 32;
    const dim3 grid(static_cast<unsigned>(div_up(rows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    const W8ContiguousOutput output{residual_out, rows};
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages,
                                 Variant, W8Epilogue::Residual><<<grid, kThreads, 0, stream>>>(
        x, codes, scales, output, rows, k, cols, padded_k, full_slabs);
}

template <int ColsPerTile>
void launch_variant(W8KernelVariant variant, const Tensor& x, const Weight& w, Tensor& residual_out,
                    cudaStream_t stream) {
    const auto* xp       = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (x.ne[0] % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? x.ne[0] / 1024 : 0;
    const auto* codes             = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales            = static_cast<const std::uint8_t*>(w.scales);
    auto* out                     = static_cast<__nv_bfloat16*>(residual_out.data);

    switch (variant) {
    case W8KernelVariant::Full:
        launch_tt<ColsPerTile, W8KernelVariant::Full>(xp, codes, scales, out, residual_out.ne[0],
                                                      x.ne[0], x.ne[1], w.padded_shape[1],
                                                      full_slabs, stream);
        break;
    case W8KernelVariant::Predicated:
        launch_tt<ColsPerTile, W8KernelVariant::Predicated>(xp, codes, scales, out,
                                                            residual_out.ne[0], x.ne[0], x.ne[1],
                                                            w.padded_shape[1], full_slabs, stream);
        break;
    case W8KernelVariant::None:
        throw std::invalid_argument("w8 linear_add SIMT requires Full or Predicated variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_linear_add_decode_r4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream) {
    launch_decode<4>(x, w, residual_out, stream);
}

void w8_linear_add_decode_r8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream) {
    launch_decode<8>(x, w, residual_out, stream);
}

void w8_linear_add_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream) {
    launch_decode<16>(x, w, residual_out, stream);
}

void w8_linear_add_simt_r8_c4_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& residual_out, cudaStream_t stream) {
    launch_variant<4>(variant, x, w, residual_out, stream);
}

void w8_linear_add_simt_r8_c8_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& residual_out, cudaStream_t stream) {
    launch_variant<8>(variant, x, w, residual_out, stream);
}

} // namespace ninfer::ops::detail
