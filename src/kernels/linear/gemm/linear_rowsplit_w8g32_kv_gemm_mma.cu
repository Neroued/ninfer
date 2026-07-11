#include "kernels/linear/gemm/linear_rowsplit_w8g32_kv_gemm_mma.cuh"

#include "kernels/common/math.h"
#include "qus/core/device.h"
#include "qus/core/tensor.h"

#include <cstdint>

namespace qus::kernels::detail {

void linear_rowsplit_w8g32_kv_gemm_mma_launch(const Tensor& x, const Weight& k_weight,
                                              const Weight& v_weight, Tensor& k_out, Tensor& v_out,
                                              cudaStream_t stream) {
    constexpr int BM            = 32;
    constexpr int BN            = 128;
    const std::int32_t m        = k_weight.n;
    const std::int32_t k        = k_weight.k;
    const std::int32_t n        = x.ne[1];
    const std::int32_t padded_k = k_weight.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(m, BM)),
                    static_cast<unsigned>(div_up(n, BN)), 1u);
    const bool full = (m % BM) == 0 && (n % BN) == 0 && k == padded_k && (k % 64) == 0;
    const auto* xp  = static_cast<const __nv_bfloat16*>(x.data);
    const auto* kc  = static_cast<const std::uint8_t*>(k_weight.qdata);
    const auto* ks  = static_cast<const std::uint8_t*>(k_weight.scales);
    const auto* vc  = static_cast<const std::uint8_t*>(v_weight.qdata);
    const auto* vs  = static_cast<const std::uint8_t*>(v_weight.scales);
    auto* ko        = static_cast<__nv_bfloat16*>(k_out.data);
    auto* vo        = static_cast<__nv_bfloat16*>(v_out.data);
    if (full) {
        linear_rowsplit_w8g32_kv_gemm_mma_kernel<true>
            <<<grid, 256, 0, stream>>>(xp, kc, ks, vc, vs, ko, vo, m, k, n, padded_k);
    } else {
        linear_rowsplit_w8g32_kv_gemm_mma_kernel<false>
            <<<grid, 256, 0, stream>>>(xp, kc, ks, vc, vs, ko, vo, m, k, n, padded_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
