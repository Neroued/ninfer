#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_rowsplit_gemm_mma.cuh"

namespace ninfer::ops::detail {

template <class Schedule, bool Full, class Epilogue = Q4MmaStoreEpilogue>
void launch_q4_mma_tile(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];

    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kBlockRows)),
                    static_cast<unsigned>(div_up(cols, Schedule::kBlockCols)), 1u);

    q4_rowsplit_gemm_mma_kernel<Schedule, Full, Epilogue><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), rows, k,
        cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, class Epilogue = Q4MmaStoreEpilogue>
void launch_q4_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const bool full =
        (out.ne[0] % Schedule::kBlockRows) == 0 && (x.ne[1] % Schedule::kBlockCols) == 0;
    for_each_token_slice(x.ne[1], Schedule::kBlockCols,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor out_slice     = out.slice(1, offset, count);
                             if (full) {
                                 launch_q4_mma_tile<Schedule, true, Epilogue>(x_slice, w, out_slice,
                                                                              stream);
                             } else {
                                 launch_q4_mma_tile<Schedule, false, Epilogue>(x_slice, w,
                                                                               out_slice, stream);
                             }
                         });
}


} // namespace ninfer::ops::detail
