#include "ops/linear/q6/q6_rowsplit_kernels.h"

#include "ops/common/math.h"
#include "ops/linear/q6/q6_rowsplit_gemm_simt.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using SimtR8C4Schedule = Q6RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;

} // namespace

void q6_rowsplit_simt_r8_c4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(n, SimtR8C4Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(t, SimtR8C4Schedule::kColsPerTile)), 1u);
    q6_rowsplit_gemm_simt_kernel<SimtR8C4Schedule, Q6KernelVariant::Predicated>
        <<<grid, SimtR8C4Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
            static_cast<__nv_bfloat16*>(out.data), n, k, t, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
