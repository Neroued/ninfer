#pragma once

#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"

#include "core/device.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/q4/q4_operands.h"
#include "ops/linear/kernels/q4/q4_a16_mma.cuh"

namespace ninfer::ops::detail {

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue>
void launch_q4_a16_mma(const Q4LinearOperands& operands, Output output, Epilogue epilogue,
                       cudaStream_t stream) {
    validate_q4_operands(operands);
    if (operands.padded_k % Schedule::kBlockK != 0) {
        throw std::invalid_argument("Q4 MMA padded K must contain complete K tiles");
    }
    common::for_each_token_slice(
        operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const dim3 grid(
                static_cast<unsigned>(common::div_up(operands.rows, Schedule::kBlockRows)),
                static_cast<unsigned>(common::div_up(count, Schedule::kBlockTokens)));
            const auto* x = operands.x + static_cast<std::int64_t>(offset) * operands.k;
            const bool full_rows_k =
                operands.rows % Schedule::kBlockRows == 0 &&
                operands.k == operands.padded_k && operands.k % Schedule::kBlockK == 0;
            const bool full_tokens = count % Schedule::kBlockTokens == 0;
            const auto launch = [&]<bool FullRowsK, bool FullTokens>() {
                q4_a16_mma_kernel<Schedule, FullRowsK, Output, Epilogue, FullTokens>
                    <<<grid, Schedule::kThreads, 0, stream>>>(
                    x, operands.codes, operands.scales, output, epilogue, operands.rows, operands.k,
                    count, operands.padded_k, offset);
            };
            constexpr bool independent_tokens = [] {
                if constexpr (requires { Schedule::kIndependentTokenTail; })
                    return Schedule::kIndependentTokenTail;
                else return false;
            }();
            if constexpr (independent_tokens) {
                // Keep one token-tail variant across full and partial tiles. Row/K bounds
                // remain independent, without changing register policy at exact token tiles.
                if (full_rows_k) launch.template operator()<true, false>();
                else launch.template operator()<false, false>();
            } else {
                if (full_rows_k && full_tokens) launch.template operator()<true, true>();
                else launch.template operator()<false, false>();
            }
            CUDA_CHECK(cudaGetLastError());
        });
}

} // namespace ninfer::ops::detail
