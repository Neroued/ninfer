#include "ops/common/numeric/math.h"
#include "ops/linear/pair/dispatch.h"
#include "ops/linear/pair/q8/dual_mma.cuh"
#include "ops/common/host/launch.h"
#include "ops/linear/kernels/q8/q8_simt_launch.cuh"

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
    const dim3 grid(static_cast<unsigned>(common::div_up(m, BM)),
                    static_cast<unsigned>(common::div_up(n, BN)), 1u);
    const auto* xp            = static_cast<const __nv_bfloat16*>(x.data);
    const auto* first_codes   = static_cast<const std::uint8_t*>(first_weight.qdata);
    const auto* first_scales  = static_cast<const std::uint8_t*>(first_weight.scales);
    const auto* second_codes  = static_cast<const std::uint8_t*>(second_weight.qdata);
    const auto* second_scales = static_cast<const std::uint8_t*>(second_weight.scales);
    auto* first_output        = static_cast<__nv_bfloat16*>(first_out.data);
    auto* second_output       = static_cast<__nv_bfloat16*>(second_out.data);
    q8_pair_gemm_mma_kernel<TileCols, Full><<<grid, (TileCols / 16) * 32, 0, stream>>>(
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

template <int TileCols>
void launch_two_simt(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                     Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    using Schedule = Q8A16SimtSchedule<8, TileCols, 1, 32, 2, Cache::cg, 1>;
    launch_q8_a16_simt<Schedule>(
        q8_linear_operands(x, first_weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(first_out.data), first_out.ne[0]},
        LinearIdentityEpilogue{}, stream);
    launch_q8_a16_simt<Schedule>(
        q8_linear_operands(x, second_weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(second_out.data), second_out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <int TileCols>
void launch_dual(const Tensor& x, const Weight& w0, const Weight& w1, Tensor& y0, Tensor& y1,
                 cudaStream_t stream) {
    const bool full = x.ne[1] % TileCols == 0;
    common::for_each_token_slice(x.ne[1], TileCols, [&](int offset, int count) {
        const auto sx = x.slice(1, offset, count);
        auto s0       = y0.slice(1, offset, count);
        auto s1       = y1.slice(1, offset, count);
        launch_tile<TileCols>(full, sx, w0, w1, s0, s1, stream);
    });
}
} // namespace

LinearPairLaunch select_q8_pair_n1024_k5120(int tokens) {
    if (tokens <= 85) return launch_two_simt<4>;
    if (tokens <= 960) return launch_dual<64>;
    return launch_dual<128>;
}
} // namespace ninfer::ops::detail
