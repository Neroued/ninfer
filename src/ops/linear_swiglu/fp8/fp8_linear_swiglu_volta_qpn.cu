#include "core/device.h"
#include "core/tensor.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_volta_qpn.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Fp8VoltaQpnSchedule;
} // namespace

bool fp8_linear_swiglu_volta_qpn_supported(std::int32_t k, std::int32_t t) noexcept {
    if (t < 2 || t > 4 * S::kRowsPerTile) { return false; }
    if (k % S::kKPerBlock != 0) { return false; }
    return k / S::kKPerBlock >= kFp8SwigluWarps;
}

void fp8_linear_swiglu_volta_qpn_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const dim3 grid(
        static_cast<unsigned>((kFp8SwigluIntermediate + S::kColsPerCta - 1) / S::kColsPerCta));
    const auto* codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const __nv_bfloat16*>(weight.scales);
    const auto* xd      = static_cast<const __nv_bfloat16*>(x.data);
    auto* outd = static_cast<__nv_bfloat16*>(out.data);
    if (t <= S::kRowsPerTile) {
        fp8_linear_swiglu_volta_qpn_kernel<1><<<grid, kFp8SwigluThreads, 0, stream>>>(codes, scales, xd,
                                                                                k, t, outd);
    } else if (t <= 2 * S::kRowsPerTile) {
        fp8_linear_swiglu_volta_qpn_kernel<2><<<grid, kFp8SwigluThreads, 0, stream>>>(codes, scales, xd,
                                                                                k, t, outd);
    } else {
        fp8_linear_swiglu_volta_qpn_kernel<4><<<grid, kFp8SwigluThreads, 0, stream>>>(codes, scales, xd,
                                                                                k, t, outd);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
