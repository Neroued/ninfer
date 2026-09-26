#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_a16_gemv.cuh"
#include "ops/linear/q4/q4_operands.h"

namespace ninfer::ops::detail {
template <class Schedule, bool TriggerPdl = false, bool JoinPdl = false, bool Dependent = false,
          class Output, class Epilogue>
void launch_q4_a16_gemv(const Q4LinearOperands& operands, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    validate_q4_operands(operands);
    if (operands.tokens != 1) throw std::invalid_argument("Q4 GEMV requires T=1");
    if constexpr (Schedule::kStaticGroupsPerRow > 0) {
        if (operands.k != Schedule::kStaticGroupsPerRow * 64)
            throw std::invalid_argument("Q4 GEMV static K differs from operands");
    }
    const dim3 grid(div_up(operands.rows, Schedule::kBlockRows));
    const dim3 block(Schedule::kThreads);
    const std::size_t shared_bytes =
        Schedule::kActivationAccess == Q4GemvActivationAccess::CtaSharedFullK
            ? static_cast<std::size_t>(operands.k) * sizeof(__nv_bfloat16)
            : 0;
    if (Schedule::kSharedBytes + shared_bytes > 48 * 1024)
        throw std::invalid_argument("Q4 GEMV shared memory exceeds 48 KiB");
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {grid, block, shared_bytes, stream},
            q4_a16_gemv_kernel<Schedule, Output, Epilogue, TriggerPdl, JoinPdl>, operands.x,
            operands.codes, operands.scales, output, epilogue, operands.rows, operands.k,
            operands.padded_k));
    } else {
        q4_a16_gemv_kernel<Schedule, Output, Epilogue, TriggerPdl, JoinPdl>
            <<<grid, block, shared_bytes, stream>>>(operands.x, operands.codes, operands.scales,
                                                    output, epilogue, operands.rows, operands.k,
                                                    operands.padded_k);
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::ops::detail
