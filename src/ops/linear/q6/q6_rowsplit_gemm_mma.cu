#include "ops/linear/q6/q6_rowsplit_kernels.h"

#include "ops/common/math.h"
#include "ops/linear/q6/q6_rowsplit_gemm_mma.cuh"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using MmaR64C64Schedule =
    Q6RowSplitMmaGemmSchedule<64, 64, 64, 32, 32, 2, 3, Q6FragmentPipeline::PingPong, Cache::ca,
                              Cache::ca, Q6ScaleLoad::Scalar16>;
using MmaR64C128Schedule =
    Q6RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q6FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q6ScaleLoad::Pair32>;

template <class Cfg>
void launch_variant(Q6KernelVariant variant, const Tensor& x, const Weight& w, Tensor& out,
                    cudaStream_t stream) {
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high            = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    auto* outp                  = static_cast<__nv_bfloat16*>(out.data);
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(n, Cfg::kBlockRows)),
                    static_cast<unsigned>(div_up(t, Cfg::kBlockCols)), 1u);

    switch (variant) {
    case Q6KernelVariant::Full:
        q6_rowsplit_gemm_mma_kernel<Cfg, Q6KernelVariant::Full>
            <<<grid, Cfg::kThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
        break;
    case Q6KernelVariant::Predicated:
        q6_rowsplit_gemm_mma_kernel<Cfg, Q6KernelVariant::Predicated>
            <<<grid, Cfg::kThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k);
        break;
    case Q6KernelVariant::None:
        throw std::invalid_argument("q6 MMA launch requires Full or Predicated variant");
    default:
        throw std::invalid_argument("q6 MMA launch received an unknown variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q6_rowsplit_mma_r64_c64_launch(Q6KernelVariant variant, const Tensor& x, const Weight& w,
                                    Tensor& out, cudaStream_t stream) {
    launch_variant<MmaR64C64Schedule>(variant, x, w, out, stream);
}

void q6_rowsplit_mma_r64_c128_launch(Q6KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& out, cudaStream_t stream) {
    launch_variant<MmaR64C128Schedule>(variant, x, w, out, stream);
}

} // namespace ninfer::ops::detail
