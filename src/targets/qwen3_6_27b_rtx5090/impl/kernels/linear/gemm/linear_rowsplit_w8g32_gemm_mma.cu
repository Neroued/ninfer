#include "kernels/linear/gemm/linear_rowsplit_w8g32_gemm_mma.cuh"

#include "kernels/common/math.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::kernels::detail {
namespace {

template <class Cfg>
void launch_w8_cfg(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t m        = w.n;
    const std::int32_t k        = w.k;
    const std::int32_t n        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(m, Cfg::BM)),
                    static_cast<unsigned>(div_up(n, Cfg::BN)), 1u);
    const bool full =
        (m % Cfg::BM) == 0 && (n % Cfg::BN) == 0 && k == padded_k && (k % Cfg::BK) == 0;
    const auto* xp     = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* outp         = static_cast<__nv_bfloat16*>(out.data);
    if (full) {
        linear_rowsplit_w8g32_gemm_mma_kernel<Cfg, true>
            <<<grid, Cfg::THREADS, 0, stream>>>(xp, codes, scales, outp, m, k, n, padded_k);
    } else {
        linear_rowsplit_w8g32_gemm_mma_kernel<Cfg, false>
            <<<grid, Cfg::THREADS, 0, stream>>>(xp, codes, scales, outp, m, k, n, padded_k);
    }
}

} // namespace

void linear_rowsplit_w8g32_gemm_mma_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream) {
    if (w.qtype != QType::W8G32_F16S || w.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("linear: W8G32 MMA GEMM requires W8G32 row-split weight");
    }
    if (w.n <= 1024) {
        using Cfg = W8G32GemmCfg<32, 128, 32, 16, 2>;
        launch_w8_cfg<Cfg>(x, w, out, stream);
    } else {
        using Cfg = W8G32GemmCfg<64, 128, 64, 16, 2, 2>;
        launch_w8_cfg<Cfg>(x, w, out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
