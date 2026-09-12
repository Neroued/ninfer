#include "ops/linear_pair/w8/w8_pair_gemm_mma.cuh"

#include "ops/common/math.h"
#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

namespace {

template <int TileCols, bool Full>
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
    w8_pair_gemm_mma_kernel<TileCols, Full><<<grid, (TileCols / 16) * 32, 0, stream>>>(
        xp, first_codes, first_scales, second_codes, second_scales, first_output, second_output, m,
        k, n, padded_k);
}

template <int TileCols>
void launch_tile(bool full, const Tensor& x, const Weight& first_weight,
                 const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                 cudaStream_t stream) {
    if (full) {
        launch_variant<TileCols, true>(x, first_weight, second_weight, first_out, second_out,
                                       stream);
    } else {
        launch_variant<TileCols, false>(x, first_weight, second_weight, first_out, second_out,
                                        stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int TileCols, bool Full>
void launch_simt_single(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int RowsPerCta      = 8;
    constexpr int Stages          = 2;
    const std::int32_t full_slabs = x.ne[0] / 1024;
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], RowsPerCta)),
                    static_cast<unsigned>(div_up(x.ne[1], TileCols)), 1u);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]};
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, TileCols, RowsPerCta, Stages, Full>
        <<<grid, RowsPerCta * 32, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                               static_cast<const std::uint8_t*>(weight.qdata),
                                               static_cast<const std::uint8_t*>(weight.scales),
                                               output, out.ne[0], x.ne[0], x.ne[1],
                                               weight.padded_shape[1], full_slabs);
}

template <int TileCols>
void launch_two_simt(bool full, const Tensor& x, const Weight& first_weight,
                     const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                     cudaStream_t stream) {
    if (full) {
        launch_simt_single<TileCols, true>(x, first_weight, first_out, stream);
        launch_simt_single<TileCols, true>(x, second_weight, second_out, stream);
    } else {
        launch_simt_single<TileCols, false>(x, first_weight, first_out, stream);
        launch_simt_single<TileCols, false>(x, second_weight, second_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_pair_simt_r8_c4_launch(bool full, const Tensor& x, const Weight& first_weight,
                               const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                               cudaStream_t stream) {
    launch_two_simt<4>(full, x, first_weight, second_weight, first_out, second_out, stream);
}

void w8_pair_gemm_mma_r32_c64_launch(bool full, const Tensor& x, const Weight& first_weight,
                                     const Weight& second_weight, Tensor& first_out,
                                     Tensor& second_out, cudaStream_t stream) {
    launch_tile<64>(full, x, first_weight, second_weight, first_out, second_out, stream);
}

void w8_pair_gemm_mma_r32_c128_launch(bool full, const Tensor& x, const Weight& first_weight,
                                      const Weight& second_weight, Tensor& first_out,
                                      Tensor& second_out, cudaStream_t stream) {
    launch_tile<128>(full, x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops::detail
