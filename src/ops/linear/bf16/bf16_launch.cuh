#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear/bf16/bf16_gemv.cuh"
#include "ops/linear/bf16/bf16_simt.cuh"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"

namespace ninfer::ops::detail {
template <class Geometry, class Schedule>
void launch_bf16_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const Bf16ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data)};
    bf16_gemv_kernel<Geometry, Schedule>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int Capacity, class Schedule>
void launch_bf16_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static_assert(Geometry::kOutputRows % Schedule::kRowsPerCta == 0);
    const Bf16SimtContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                          Geometry::kOutputRows};
    bf16_simt_kernel<Geometry, Capacity, Schedule, Bf16SimtContiguousOutput, true>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule, bool FullTokens>
void launch_bf16_mma_variant(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream) {
    static_assert(Geometry::kOutputRows % Schedule::kBlockRows == 0);
    static_assert(Geometry::kInputRows % Schedule::kBlockK == 0);
    const int blocks =
        Geometry::kOutputRows / Schedule::kBlockRows * div_up(x.ne[1], Schedule::kBlockCols);
    const Bf16MmaContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                         Geometry::kOutputRows};
    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attr = cudaFuncSetAttribute(
            bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens, Bf16MmaContiguousOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attr);
    }
    bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_bf16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] % Schedule::kBlockCols == 0)
        launch_bf16_mma_variant<Geometry, Schedule, true>(x, weight, out, stream);
    else
        launch_bf16_mma_variant<Geometry, Schedule, false>(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
