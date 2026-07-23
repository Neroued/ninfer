#include "ops/linear_pair/w8/w8_pair_gemm_mma.cuh"

#include "ops/common/math.h"
#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

template <int TileCols, W8KernelVariant Variant>
void launch_variant(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                    Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    constexpr int BM            = 32;
    constexpr int BN            = TileCols;
    const std::int32_t m        = first_weight.n;
    const std::int32_t k        = first_weight.k;
    const std::int32_t n        = x.ne[1];
    const std::int32_t padded_k = first_weight.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(m, BM)), static_cast<unsigned>(div_up(n, BN)), 1u);
    const auto* xp            = static_cast<const __nv_bfloat16*>(x.data);
    const auto* first_codes   = static_cast<const std::uint8_t*>(first_weight.qdata);
    const auto* first_scales  = static_cast<const std::uint8_t*>(first_weight.scales);
    const auto* second_codes  = static_cast<const std::uint8_t*>(second_weight.qdata);
    const auto* second_scales = static_cast<const std::uint8_t*>(second_weight.scales);
    auto* first_output        = static_cast<__nv_bfloat16*>(first_out.data);
    auto* second_output       = static_cast<__nv_bfloat16*>(second_out.data);
    w8_pair_gemm_mma_kernel<TileCols, Variant><<<grid, (TileCols / 16) * 32, 0, stream>>>(
        xp, first_codes, first_scales, second_codes, second_scales, first_output, second_output, m,
        k, n, padded_k);
}

template <int TileCols>
void launch_tile(W8KernelVariant variant, const Tensor& x, const Weight& first_weight,
                 const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                 cudaStream_t stream) {
    switch (variant) {
    case W8KernelVariant::Full:
        launch_variant<TileCols, W8KernelVariant::Full>(x, first_weight, second_weight, first_out,
                                                        second_out, stream);
        break;
    case W8KernelVariant::Predicated:
        launch_variant<TileCols, W8KernelVariant::Predicated>(x, first_weight, second_weight,
                                                              first_out, second_out, stream);
        break;
    case W8KernelVariant::None:
        throw std::invalid_argument("w8 pair MMA launch requires Full or Predicated variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

void w8_pair_gemm_mma_r32_c64_launch(W8KernelVariant variant, const Tensor& x,
                                     const Weight& first_weight, const Weight& second_weight,
                                     Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    launch_tile<64>(variant, x, first_weight, second_weight, first_out, second_out, stream);
}

void w8_pair_gemm_mma_r32_c80_launch(W8KernelVariant variant, const Tensor& x,
                                     const Weight& first_weight, const Weight& second_weight,
                                     Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    launch_tile<80>(variant, x, first_weight, second_weight, first_out, second_out, stream);
}

void w8_pair_gemm_mma_r32_c96_launch(W8KernelVariant variant, const Tensor& x,
                                     const Weight& first_weight, const Weight& second_weight,
                                     Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    launch_tile<96>(variant, x, first_weight, second_weight, first_out, second_out, stream);
}

void w8_pair_gemm_mma_r32_c112_launch(W8KernelVariant variant, const Tensor& x,
                                      const Weight& first_weight, const Weight& second_weight,
                                      Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    launch_tile<112>(variant, x, first_weight, second_weight, first_out, second_out, stream);
}

void w8_pair_gemm_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& first_weight,
                             const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                             cudaStream_t stream) {
    launch_tile<128>(variant, x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops::detail
