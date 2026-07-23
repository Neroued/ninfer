#include "ops/linear_add/w8/w8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_rowsplit_gemm_exact_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows                = 2048;
constexpr int kRowsPerCta          = 16;
constexpr int kFirstExactCols      = 2;
constexpr int kLastExactCols       = 32;
constexpr int kLegacyLastExactCols = 16;
using ProjectionLauncher           = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

struct W8LinearAddExactTEpilogue {
    __nv_bfloat16* residual;
    std::int32_t rows;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, float (&projected)[ActiveCols]) const {
#pragma unroll
        for (int token = 0; token < ActiveCols; ++token) {
            const std::int64_t index = static_cast<std::int64_t>(token) * rows + row;
            residual[index] =
                __float2bfloat16_rn(projected[token] + __bfloat162float(residual[index]));
        }
    }
};

template <int Hidden, int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& residual_out,
                        cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    static_assert((kRows % kRowsPerCta) == 0);
    auto* residual = static_cast<__nv_bfloat16*>(residual_out.data);
    const W8ContiguousOutput output{residual, kRows};
    const W8LinearAddExactTEpilogue epilogue{residual, kRows};
    w8_rowsplit_exact_t_splitk_kernel<Hidden, TileCols, ActiveCols, W8ContiguousOutput,
                                      W8LinearAddExactTEpilogue>
        <<<kRows / kRowsPerCta, 8 * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, epilogue);
}

template <int Hidden, std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<Hidden, kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kK4096ProjectionLaunchers = make_projection_launchers<4096>(
    std::make_index_sequence<kLegacyLastExactCols - kFirstExactCols + 1>{});
constexpr auto kK6144ProjectionLaunchers = make_projection_launchers<6144>(
    std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

} // namespace

void w8_linear_add_splitk_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                     Tensor& residual_out, cudaStream_t stream) {
    const std::int32_t last_exact = weight.k == 6144 ? kLastExactCols : kLegacyLastExactCols;
    if (variant != W8KernelVariant::None || x.ne[1] < kFirstExactCols || x.ne[1] > last_exact) {
        throw std::invalid_argument("W8 linear_add split-K MMA requires exact T=2..32");
    }
    if (weight.k == 6144) {
        kK6144ProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    } else {
        kK4096ProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void w8_linear_add_splitk_mma_composite_launch(const Tensor& x, const Weight& weight,
                                               Tensor& residual_out, cudaStream_t stream) {
    if (weight.k != 6144 || x.ne[1] < 33 || x.ne[1] > 65) {
        throw std::invalid_argument(
            "W8 linear_add composite split-K MMA requires K=6144 and T=33..65");
    }
    std::int32_t offset = 0;
    while (x.ne[1] - offset >= 32) {
        const Tensor x_slice  = x.slice(1, offset, 32);
        Tensor residual_slice = residual_out.slice(1, offset, 32);
        w8_linear_add_splitk_mma_launch(W8KernelVariant::None, x_slice, weight, residual_slice,
                                        stream);
        offset += 32;
    }
    const std::int32_t tail = x.ne[1] - offset;
    if (tail == 1) {
        const Tensor x_slice  = x.slice(1, offset, 1);
        Tensor residual_slice = residual_out.slice(1, offset, 1);
        w8_linear_add_decode_r16_launch(x_slice, weight, residual_slice, stream);
    } else if (tail >= 2) {
        const Tensor x_slice  = x.slice(1, offset, tail);
        Tensor residual_slice = residual_out.slice(1, offset, tail);
        w8_linear_add_splitk_mma_launch(W8KernelVariant::None, x_slice, weight, residual_slice,
                                        stream);
    }
}

} // namespace ninfer::ops::detail
