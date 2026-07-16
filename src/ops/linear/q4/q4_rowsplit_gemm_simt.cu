#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_kernels.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Q4SimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4SimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

template <class Schedule, Q4KernelVariant Variant>
void launch_schedule(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];

    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);

    q4_rowsplit_gemm_simt_kernel<Schedule, Variant><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows, k,
        cols, padded_k);
}

template <class Schedule>
void launch_variant(Q4KernelVariant variant, const Tensor& x, const Weight& w, Tensor& out,
                    cudaStream_t stream) {
    switch (variant) {
    case Q4KernelVariant::Full:
        launch_schedule<Schedule, Q4KernelVariant::Full>(x, w, out, stream);
        break;
    case Q4KernelVariant::Predicated:
        launch_schedule<Schedule, Q4KernelVariant::Predicated>(x, w, out, stream);
        break;
    case Q4KernelVariant::None:
        throw std::invalid_argument("q4 SIMT GEMM launch requires Full or Predicated variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_rowsplit_gemm_simt_r8_c4_launch(Q4KernelVariant variant, const Tensor& x, const Weight& w,
                                        Tensor& out, cudaStream_t stream) {
    launch_variant<Q4SimtR8C4Schedule>(variant, x, w, out, stream);
}

void q4_rowsplit_gemm_simt_r8_c8_launch(Q4KernelVariant variant, const Tensor& x, const Weight& w,
                                        Tensor& out, cudaStream_t stream) {
    launch_variant<Q4SimtR8C8Schedule>(variant, x, w, out, stream);
}

} // namespace ninfer::ops::detail
