#include "ops/linear_add/w8/w8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_rowsplit_gemm_exact_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows           = 2048;
constexpr int kHidden         = 4096;
constexpr int kRowsPerCta     = 16;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 16;
using ProjectionLauncher      = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

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

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& residual_out,
                        cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    static_assert((kRows % kRowsPerCta) == 0);
    auto* residual = static_cast<__nv_bfloat16*>(residual_out.data);
    const W8ContiguousOutput output{residual, kRows};
    const W8LinearAddExactTEpilogue epilogue{residual, kRows};
    w8_rowsplit_exact_t_splitk_kernel<kHidden, TileCols, ActiveCols, W8ContiguousOutput,
                                      W8LinearAddExactTEpilogue>
        <<<kRows / kRowsPerCta, 8 * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, epilogue);
}

template <std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kProjectionLaunchers =
    make_projection_launchers(std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

} // namespace

void w8_linear_add_splitk_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                     Tensor& residual_out, cudaStream_t stream) {
    if (variant != W8KernelVariant::None || x.ne[1] < kFirstExactCols || x.ne[1] > kLastExactCols) {
        throw std::invalid_argument("W8 linear_add split-K MMA requires exact T=2..16");
    }
    kProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
