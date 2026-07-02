// Launcher for the LargeT bf16 tensor-core GEMM (Q4/Q5/Q6 row-split), selected by
// fmt. Routed from the LargeT regime; the multi-step GEMV remains the SmallT path
// and the universal fallback.
#include "kernels/linear/gemm/linear_rowsplit_gemm_mma.cuh"

#include "kernels/linear/reference/linear_generic.h" // launch declaration
#include "qus/core/device.h"                          // CUDA_CHECK

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace qus::kernels::detail {
namespace {

int ceil_div(int a, int b) { return (a + b - 1) / b; }

template <class Codec, class Cfg>
void launch_cfg(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* high,
                const std::uint8_t* scales, __nv_bfloat16* outp, std::int32_t n, std::int32_t k,
                std::int32_t t, std::int32_t padded_k, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(ceil_div(n, Cfg::BM)),
                    static_cast<unsigned>(ceil_div(t, Cfg::BN)), 1u);
    const bool full_tiles =
        (n % Cfg::BM) == 0 && (t % Cfg::BN) == 0 && k == padded_k && (k % Cfg::BK) == 0;
    if (full_tiles) {
        linear_rowsplit_gemm_mma_kernel<Codec, Cfg, true>
            <<<grid, Cfg::THREADS, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
    } else {
        linear_rowsplit_gemm_mma_kernel<Codec, Cfg, false>
            <<<grid, Cfg::THREADS, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
    }
}

// QUS_GEMM_CFG selects a compiled tile config for ncu sweeps. Tag = "BMxBNxWMxWN"
// (BK is fixed to 64). Unset / unknown -> the default. Add candidates here as the
// tuning loop needs them.
template <class Codec>
void dispatch_codec(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* high,
                    const std::uint8_t* scales, __nv_bfloat16* outp, std::int32_t n, std::int32_t k,
                    std::int32_t t, std::int32_t padded_k, cudaStream_t stream) {
    const char*            env = std::getenv("QUS_GEMM_CFG");
    const std::string_view tag = env ? std::string_view(env) : std::string_view();

    // Tag = "BMxBNxWMxWN" (BK 64). The default favors the MLP gate_up
    // fused-prefill shape while preserving the original configs for local sweeps.
    if (tag == "64x64x32x32") {
        launch_cfg<Codec, GemmCfg<64, 64, 64, 32, 32, 2, 3>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else if (tag == "128x64x32x32") {
        launch_cfg<Codec, GemmCfg<128, 64, 64, 32, 32, 2, 2>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else if (tag == "64x128x32x32") {
        launch_cfg<Codec, GemmCfg<64, 128, 64, 32, 32, 2, 2>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else if (tag == "64x128x16x64" || tag == "64x128x16x64m2") {
        launch_cfg<Codec, GemmCfg<64, 128, 64, 16, 64, 2, 2>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else if (tag == "64x128x64x16") {
        launch_cfg<Codec, GemmCfg<64, 128, 64, 64, 16, 2, 2, false>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else if (tag == "64x128x64x32cg") {
        launch_cfg<Codec, GemmCfg<64, 128, 64, 64, 32, 2, 1, false, true>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else if (tag == "64x128x64x32cgsp") {
        launch_cfg<Codec, GemmCfg<64, 128, 64, 64, 32, 2, 1, false, true, true>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else if (tag == "128x64x64x32") {
        launch_cfg<Codec, GemmCfg<128, 64, 64, 64, 32, 2, 2>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else if (tag == "64x64x64x32") {
        launch_cfg<Codec, GemmCfg<64, 64, 64, 64, 32, 2, 3>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else if constexpr (std::is_same_v<Codec, Q6Codec>) {
        launch_cfg<Codec, GemmCfg<64, 128, 64, 64, 32, 2, 1, false, true, true>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
    } else {
        launch_cfg<Codec, GemmCfg<64, 128, 64, 64, 16, 2, 2, false>>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
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
    const auto*        codes    = static_cast<const std::uint8_t*>(w.qdata);
    const auto*        high     = static_cast<const std::uint8_t*>(w.qhigh);
    const auto*        scales   = static_cast<const std::uint8_t*>(w.scales);
    const auto*        xp       = static_cast<const __nv_bfloat16*>(x.data);
    auto*              outp     = static_cast<__nv_bfloat16*>(out.data);

    switch (fmt) {
    case LinearFormat::Q4G64_RowSplit:
        dispatch_codec<Q4Codec>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
        break;
    case LinearFormat::Q5G64_RowSplit:
        dispatch_codec<Q5Codec>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
        break;
    case LinearFormat::Q6G64_RowSplit:
        dispatch_codec<Q6Codec>(xp, codes, high, scales, outp, n, k, t, padded_k, stream);
        break;
    default:
        throw std::invalid_argument("linear: mma GEMM requires a Q4/Q5/Q6 row-split format");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
