#include "ops/linear/q5/q5_rowsplit_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q5/q5_rowsplit_gemm_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using MmaR64C64Schedule =
    Q5RowSplitMmaGemmSchedule<64, 64, 64, 32, 32, 2, 3, Q5FragmentPipeline::PingPong, Cache::ca,
                              Cache::ca, Q5ScaleLoad::Scalar16>;
using MmaR64C128Schedule =
    Q5RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q5FragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Q5ScaleLoad::Pair32>;

template <class Schedule, Q5MmaEpilogue Epilogue>
void launch_variant(Q5KernelVariant variant, const Tensor& x, const Weight& w, Tensor& out,
                    cudaStream_t stream) {
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high            = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    const auto* residual        = Epilogue == Q5MmaEpilogue::AddResidual
                                      ? static_cast<const __nv_bfloat16*>(out.data)
                                      : nullptr;
    auto* outp                  = static_cast<__nv_bfloat16*>(out.data);
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kBlockRows)),
                    static_cast<unsigned>(div_up(cols, Schedule::kBlockCols)), 1u);

    switch (variant) {
    case Q5KernelVariant::Full:
        q5_rowsplit_gemm_mma_kernel<Schedule, Q5KernelVariant::Full, Epilogue>
            <<<grid, Schedule::kThreads, 0, stream>>>(xp, codes, high, scales, residual, outp, rows,
                                                      k, cols, padded_k);
        break;
    case Q5KernelVariant::Predicated:
        q5_rowsplit_gemm_mma_kernel<Schedule, Q5KernelVariant::Predicated, Epilogue>
            <<<grid, Schedule::kThreads, 0, stream>>>(xp, codes, high, scales, residual, outp, rows,
                                                      k, cols, padded_k);
        break;
    case Q5KernelVariant::None:
        throw std::invalid_argument("q5 MMA launch requires Full or Predicated variant");
    default:
        throw std::invalid_argument("q5 MMA launch received an unknown variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q5_rowsplit_mma_r64_c64_launch(Q5KernelVariant variant, const Tensor& x, const Weight& w,
                                    Tensor& out, cudaStream_t stream) {
    launch_variant<MmaR64C64Schedule, Q5MmaEpilogue::Store>(variant, x, w, out, stream);
}

void q5_rowsplit_mma_r64_c128_launch(Q5KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& out, cudaStream_t stream) {
    launch_variant<MmaR64C128Schedule, Q5MmaEpilogue::Store>(variant, x, w, out, stream);
}

} // namespace ninfer::ops::detail
