#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q4/q4_operands.h"
#include "ops/linear/q4/q4_a16_mma.cuh"

namespace ninfer::ops::detail {

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue>
void launch_q4_a16_mma(const Q4LinearOperands& operands, Output output, Epilogue epilogue,
                       cudaStream_t stream) {
    validate_q4_operands(operands);
    if (operands.padded_k % Schedule::kBlockK != 0) {
        throw std::invalid_argument("Q4 MMA padded K must contain complete K tiles");
    }
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(static_cast<unsigned>(div_up(operands.rows, Schedule::kBlockRows)),
                        static_cast<unsigned>(div_up(count, Schedule::kBlockTokens)));
        const auto* x   = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const bool full = operands.rows % Schedule::kBlockRows == 0 &&
                          count % Schedule::kBlockTokens == 0 && operands.k == operands.padded_k &&
                          operands.k % Schedule::kBlockK == 0;
        if (full) {
            q4_a16_mma_kernel<Schedule, true><<<grid, Schedule::kThreads, 0, stream>>>(
                x, operands.codes, operands.scales, output, epilogue, operands.rows, operands.k,
                count, operands.padded_k, offset);
        } else {
            q4_a16_mma_kernel<Schedule, false><<<grid, Schedule::kThreads, 0, stream>>>(
                x, operands.codes, operands.scales, output, epilogue, operands.rows, operands.k,
                count, operands.padded_k, offset);
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninfer::ops::detail
