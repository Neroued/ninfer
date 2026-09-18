#include "core/weight.h"
#include "ops/linear/q8/q8_rowsplit_gemm_mma.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Schedule, bool Full>
void launch_slice(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = w.n;
    const std::int32_t k        = w.k;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::BM)),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)), 1u);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), rows};
    q8_rowsplit_gemm_mma_kernel<Schedule, Full><<<grid, Schedule::THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), output, rows, k, cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const bool full = (w.n % Schedule::BM) == 0 && (x.ne[1] % Schedule::BN) == 0 &&
                      w.k == w.padded_shape[1] && (w.k % Schedule::BK) == 0;
    for_each_token_slice(x.ne[1], Schedule::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_slice<Schedule, true>(x_slice, w, out_slice, stream);
        } else {
            launch_slice<Schedule, false>(x_slice, w, out_slice, stream);
        }
    });
}

using MmaR32C64  = Q8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
using MmaR32C96  = Q8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
using MmaR32C128 = Q8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
using MmaR48C64  = Q8RowSplitMmaGemmSchedule<48, 64, 48, 16, 3>;
using MmaR64C96  = Q8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>;
// The wide route of the linear family is the one schedule measured to want cg on the predicated
// path; every other schedule, here and in the six other Ops that instantiate this same tile, keeps
// the inherited ca.
using MmaR64C128 =
    Q8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>::with_predicated_cache<Cache::cg>;
using MmaR96C96  = Q8RowSplitMmaGemmSchedule<96, 96, 48, 16, 2>;
using MmaR128C64 = Q8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>;
using MmaR128C80 = Q8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>;
// K128 single-activation-stage schedules shared with fused consumers.
using MmaR64x16C48K128A1 = Q8RowSplitMmaGemmSchedule<64, 48, 16, 24, 2, 2, 128, 1>;
using MmaR64x32C64K128A1 = Q8RowSplitMmaGemmSchedule<64, 64, 32, 16, 2, 2, 128, 1>;

} // namespace

#define NINFER_Q8_MMA_LAUNCHER(Name, Schedule)                                                     \
    void Name(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {                \
        launch_route<Schedule>(x, w, out, stream);                                                 \
    }

NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r32_c64, MmaR32C64)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r32_c96, MmaR32C96)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r32_c128, MmaR32C128)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r48_c64, MmaR48C64)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r64_c96, MmaR64C96)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r64_c128, MmaR64C128)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r96_c96, MmaR96C96)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r128_c64, MmaR128C64)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r128_c80, MmaR128C80)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r64x16_c48_k128_a1, MmaR64x16C48K128A1)
NINFER_Q8_MMA_LAUNCHER(launch_q8_mma_r64x32_c64_k128_a1, MmaR64x32C64K128A1)

#undef NINFER_Q8_MMA_LAUNCHER

} // namespace ninfer::ops::detail
