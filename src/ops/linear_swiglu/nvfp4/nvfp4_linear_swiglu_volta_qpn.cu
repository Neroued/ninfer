#include "core/device.h"
#include "core/tensor.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_volta_qpn.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Nvfp4VoltaQpnSchedule;
} // namespace

bool nvfp4_linear_swiglu_volta_qpn_supported(std::int32_t k, std::int32_t t) noexcept {
    if (t < 2 || t > 4 * S::kRowsPerTile) { return false; }
    if (k % S::kChunkK != 0) { return false; }
    // Both instantiated SPLITK values (8, 16) must divide the per-warp chunk count.
    return (k / S::kChunkK) % 16 == 0;
}

void nvfp4_linear_swiglu_volta_qpn_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const dim3 grid(static_cast<unsigned>((kNvfp4SwigluIntermediate + S::kColsPerCta - 1) /
                                          S::kColsPerCta));
    const auto* codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales);
    const auto* xd      = static_cast<const __nv_bfloat16*>(x.data);
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    auto* outd = static_cast<__nv_bfloat16*>(out.data);
    // Schedule winners from a private sweep (bench/ops/nvfp4_swiglu_qpn_sweep.cu, deleted):
    // SPLITK16 NACC1 at kTiles=1 (T<=8, 225.4 GB/s vs SPLITK8's 179.7), SPLITK8 NACC1 at kTiles=2
    // (T=9..16, 232.0 GB/s). NACC=2 lost everywhere by 2.5-3.5x -- this kernel already carries
    // twice plain QPN2's accumulator and decode-register footprint (two independent projections),
    // and a second accumulator chain on top of that spills hard.
    if (t <= S::kRowsPerTile) {
        nvfp4_linear_swiglu_volta_qpn_kernel<1, 16, 1><<<grid, 16 * 32, 0, stream>>>(
            codes, scales, xd, k, t, inverse_weight_divisor, outd);
    } else {
        nvfp4_linear_swiglu_volta_qpn_kernel<2, 8, 1><<<grid, 8 * 32, 0, stream>>>(
            codes, scales, xd, k, t, inverse_weight_divisor, outd);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
