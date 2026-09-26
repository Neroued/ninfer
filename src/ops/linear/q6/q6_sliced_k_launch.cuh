#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q6/q6_operands.h"
#include "ops/linear/q6/q6_a16_sliced_k_mma.cuh"

namespace ninfer::ops::detail {

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue>
void launch_q6_a16_sliced_k_mma(const Q6LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream) {
    validate_q6_operands(operands);
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(static_cast<unsigned>(div_up(operands.rows, Schedule::kBlockRows)),
                        static_cast<unsigned>(div_up(count, Schedule::kBlockTokens)));
        const auto* x = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        // The full kernel copies all warp scales together; the masked kernel
        // copies four-byte scale pairs and supports the minimum input alignment.
        const bool full = operands.rows % Schedule::kBlockRows == 0 &&
                          count % Schedule::kBlockTokens == 0 && operands.k == operands.padded_k &&
                          operands.k % Schedule::kBlockK == 0 &&
                          reinterpret_cast<std::uintptr_t>(operands.scales) %
                                  (Schedule::kKWarps * sizeof(__half)) ==
                              0;
        if (full) {
            q6_a16_sliced_k_mma_kernel<Schedule, true><<<grid, Schedule::kThreads, 0, stream>>>(
                x, operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
                operands.k, count, operands.padded_k, offset);
        } else {
            q6_a16_sliced_k_mma_kernel<Schedule, false><<<grid, Schedule::kThreads, 0, stream>>>(
                x, operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
                operands.k, count, operands.padded_k, offset);
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninfer::ops::detail
