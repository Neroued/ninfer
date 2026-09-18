#pragma once
#include "core/device.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_simt.cuh"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include <algorithm>

namespace ninfer::ops::detail {
template <class Geometry, class Schedule>
void launch_fp8_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_gemv_kernel<Geometry, Schedule>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int Capacity, class Schedule, bool FullColumns = false>
void launch_fp8_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int blocks = Geometry::kOutputRows / Schedule::kRowsPerCta *
                           ((Capacity + Schedule::kTokenTile - 1) / Schedule::kTokenTile);
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_simt_kernel<Geometry, Capacity, Schedule, Fp8ContiguousOutput, Fp8IdentityEpilogue,
                    Fp8GemvIdentityRows, false, Fp8SimtFinalization::Elementwise, !FullColumns>
        <<<blocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, {}, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <int Chunk, Fp8Launch (*Select)(std::int32_t)>
void launch_fp8_a16_chunks(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += Chunk) {
        const int count = std::min(Chunk, x.ne[1] - offset);
        auto input      = x.slice(1, offset, count);
        auto output     = out.slice(1, offset, count);
        Select(count)(input, weight, output, stream);
    }
}

template <class Geometry, class Schedule, bool FullTokens>
void launch_fp8_a8_mma(const Weight& weight, Tensor& out, Fp8A8Workspace workspace,
                       std::int32_t tokens, cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    const int row_tiles   = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks      = row_tiles * token_tiles;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue,
                           Fp8ContiguousOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_fp8_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
                   cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    if (x.ne[1] % Schedule::kBlockTokens == 0)
        launch_fp8_a8_mma<Geometry, Schedule, true>(weight, out, scratch, x.ne[1], stream);
    else
        launch_fp8_a8_mma<Geometry, Schedule, false>(weight, out, scratch, x.ne[1], stream);
}
} // namespace ninfer::ops::detail
