#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"

namespace ninfer::ops::detail {

template <class Schedule, bool Full, bool FullK>
void launch_q4_simt_tile(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_k = w.padded_shape[1];

    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);

    q4_rowsplit_gemm_simt_kernel<Schedule, Full, false, 0, Q4SimtStoreEpilogue, false, false,
                                 Full || FullK><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), nullptr,
        out_ld, 0, rows, k, cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool FullK = false>
void launch_q4_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const bool full = (out.ne[0] % Schedule::kRowsPerCta) == 0 &&
                      ((x.ne[0] / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    for_each_token_slice(
        x.ne[1], Schedule::kColsPerTile, [&](std::int32_t offset, std::int32_t count) {
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor out_slice     = out.slice(1, offset, count);
            if (full) {
                launch_q4_simt_tile<Schedule, true, FullK>(x_slice, w, out_slice, stream);
            } else {
                launch_q4_simt_tile<Schedule, false, FullK>(x_slice, w, out_slice, stream);
            }
        });
}


} // namespace ninfer::ops::detail
