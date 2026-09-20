#include "core/weight.h"
#include "ops/linear/q6/q6_launch.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q6/q6_rowsplit_gemm_simt.cuh"
#include "core/device.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using SimtR8C4Schedule = Q6RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::cg, 1>;
using SimtR8C5Schedule = Q6RowSplitSimtGemmSchedule<8, 5, 16, 2, Cache::cg, 1>;
using SimtR8C6Schedule = Q6RowSplitSimtGemmSchedule<8, 6, 16, 2, Cache::cg, 1>;
using SimtR8C7Schedule = Q6RowSplitSimtGemmSchedule<8, 7, 16, 2, Cache::cg, 1>;

template <class Schedule>
void launch_schedule(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(n, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(t, Schedule::kColsPerTile)), 1u);
    q6_rowsplit_gemm_simt_kernel<Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.qhigh), static_cast<const std::uint8_t*>(w.scales),
        static_cast<__nv_bfloat16*>(out.data), n, k, t, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::kColsPerTile,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor out_slice     = out.slice(1, offset, count);
                             launch_schedule<Schedule>(x_slice, w, out_slice, stream);
                         });
}

} // namespace

void launch_q6_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<SimtR8C4Schedule>(x, w, out, stream);
}

void launch_q6_simt_r8_c5(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<SimtR8C5Schedule>(x, w, out, stream);
}

void launch_q6_simt_r8_c6(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<SimtR8C6Schedule>(x, w, out, stream);
}

void launch_q6_simt_r8_c7(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<SimtR8C7Schedule>(x, w, out, stream);
}


} // namespace ninfer::ops::detail
