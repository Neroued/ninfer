// Launcher for the warp-per-row multi-step (T>1) row-split low-bit GEMM. This is
// the universal prefill/T>1 path for Q4/Q5/Q6 row-split weights (registered and
// generic shapes alike); the codec is selected by fmt.
#include "kernels/linear/gemv/linear_rowsplit_gemm_multistep.cuh"

#include "kernels/linear/reference/linear_generic.h" // launch declaration
#include "qus/core/device.h"                          // CUDA_CHECK

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kTt            = 8;  // activation columns per block tile
constexpr int kWarpsPerBlock = 8;  // one warp per output row
constexpr int kBlockThreads  = kWarpsPerBlock * 32;

int ceil_div(int a, int b) { return (a + b - 1) / b; }

} // namespace

void linear_rowsplit_gemm_multistep_launch(const Tensor& x, const Weight& w, Tensor& out,
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

    const dim3 grid(static_cast<unsigned>(ceil_div(n, kWarpsPerBlock)),
                    static_cast<unsigned>(ceil_div(t, kTt)), 1u);

    switch (fmt) {
    case LinearFormat::Q4G64_RowSplit:
        linear_rowsplit_gemm_multistep_kernel<Q4Codec, kTt>
            <<<grid, kBlockThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
        break;
    case LinearFormat::Q5G64_RowSplit:
        linear_rowsplit_gemm_multistep_kernel<Q5Codec, kTt>
            <<<grid, kBlockThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
        break;
    case LinearFormat::Q6G64_RowSplit:
        linear_rowsplit_gemm_multistep_kernel<Q6Codec, kTt>
            <<<grid, kBlockThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
        break;
    case LinearFormat::W8G32_RowSplit:
        linear_rowsplit_gemm_multistep_kernel<W8G32Codec, kTt>
            <<<grid, kBlockThreads, 0, stream>>>(xp, codes, nullptr, scales, outp, n, k, t,
                                                 padded_k);
        break;
    default:
        throw std::invalid_argument(
            "linear: multistep GEMM requires a Q4/Q5/Q6/W8G32 row-split format");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
