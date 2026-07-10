// Launcher for the LargeT bf16 tensor-core GEMM (Q4/Q5/Q6 row-split), selected by
// fmt. Routed from the LargeT regime; the multi-step GEMV remains the SmallT path
// and the universal fallback.
#include "kernels/linear/gemm/linear_rowsplit_gemm_mma.cuh"

#include "kernels/linear/reference/linear_generic.h" // launch declaration
#include "qus/core/device.h"                         // CUDA_CHECK

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace qus::kernels::detail {
namespace {

int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <class Codec, class Cfg, bool Residual = false>
void launch_cfg(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* high,
                const std::uint8_t* scales, const __nv_bfloat16* residual, __nv_bfloat16* outp,
                std::int32_t n, std::int32_t k, std::int32_t t, std::int32_t padded_k,
                cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(ceil_div(n, Cfg::BM)),
                    static_cast<unsigned>(ceil_div(t, Cfg::BN)), 1u);
    const bool full_tiles =
        (n % Cfg::BM) == 0 && (t % Cfg::BN) == 0 && k == padded_k && (k % Cfg::BK) == 0;
    if (full_tiles) {
        linear_rowsplit_gemm_mma_kernel<Codec, Cfg, true, Residual>
            <<<grid, Cfg::THREADS, 0, stream>>>(xp, codes, high, scales, residual, outp, n, k, t,
                                                padded_k);
    } else {
        linear_rowsplit_gemm_mma_kernel<Codec, Cfg, false, Residual>
            <<<grid, Cfg::THREADS, 0, stream>>>(xp, codes, high, scales, residual, outp, n, k, t,
                                                padded_k);
    }
}

template <class Codec, bool Residual = false>
void dispatch_codec(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* high,
                    const std::uint8_t* scales, const __nv_bfloat16* residual, __nv_bfloat16* outp,
                    std::int32_t n, std::int32_t k, std::int32_t t, std::int32_t padded_k,
                    cudaStream_t stream) {
    using CgspCfg  = GemmCfg<64, 128, 64, 64, 32, 2, 1, false, true, true>;
    using ShortCfg = GemmCfg<64, 64, 64, 32, 32, 2, 3>;

    bool short_tile = false;
    if constexpr (std::is_same_v<Codec, Q4Codec>) {
        short_tile = t <= 128 && n == 4096 && k == 5120;
    } else if constexpr (std::is_same_v<Codec, Q5Codec>) {
        const bool down_or_out = (n == 5120 && k == 17408) || (n == 5120 && k == 6144);
        const bool projection  = n == 6144 && k == 5120;
        short_tile             = (t <= 128 && down_or_out) || (t <= 64 && projection);
    }

    if (short_tile) {
        launch_cfg<Codec, ShortCfg, Residual>(xp, codes, high, scales, residual, outp, n, k, t,
                                              padded_k, stream);
    } else {
        launch_cfg<Codec, CgspCfg, Residual>(xp, codes, high, scales, residual, outp, n, k, t,
                                             padded_k, stream);
    }
}

} // namespace

// Precondition (enforced by the regime seam in linear.cpp): k % 8 == 0, so each
// token row of x is 16-byte aligned for the cp.async<16> staging. Non-multiple-of-8
// k is routed to the multi-step GEMV instead.
void linear_rowsplit_gemm_mma_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     LinearFormat fmt, cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high            = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    auto* outp                  = static_cast<__nv_bfloat16*>(out.data);

    switch (fmt) {
    case LinearFormat::Q4G64_RowSplit:
        dispatch_codec<Q4Codec>(xp, codes, high, scales, nullptr, outp, n, k, t, padded_k, stream);
        break;
    case LinearFormat::Q5G64_RowSplit:
        dispatch_codec<Q5Codec>(xp, codes, high, scales, nullptr, outp, n, k, t, padded_k, stream);
        break;
    case LinearFormat::Q6G64_RowSplit:
        dispatch_codec<Q6Codec>(xp, codes, high, scales, nullptr, outp, n, k, t, padded_k, stream);
        break;
    default:
        throw std::invalid_argument("linear: mma GEMM requires a Q4/Q5/Q6 row-split format");
    }
    CUDA_CHECK(cudaGetLastError());
}

void linear_rowsplit_gemm_mma_residual_q5_launch(const Tensor& x, const Weight& w,
                                                 Tensor& residual_out, cudaStream_t stream) {
    const std::int32_t n        = residual_out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    dispatch_codec<Q5Codec, true>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<const __nv_bfloat16*>(residual_out.data),
        static_cast<__nv_bfloat16*>(residual_out.data), n, k, t, padded_k, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
