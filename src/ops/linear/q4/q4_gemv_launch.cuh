#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

namespace ninfer::ops::detail {

template <class Schedule, class Epilogue = Q4GemvStoreEpilogue>
void launch_q4_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    const std::size_t dynamic_shared_bytes =
        Schedule::kActivationAccess == Q4GemvActivationAccess::CtaSharedFullK
            ? static_cast<std::size_t>(k) * sizeof(__nv_bfloat16)
            : 0u;

    q4_rowsplit_gemv_kernel<Schedule, false, 0, Epilogue>
        <<<grid, block, dynamic_shared_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            nullptr, rows, k);
    CUDA_CHECK(cudaGetLastError());
}


} // namespace ninfer::ops::detail
