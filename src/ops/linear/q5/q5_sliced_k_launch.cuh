#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q5/q5_operands.h"
#include "ops/linear/q5/q5_a16_sliced_k_mma.cuh"

namespace ninfer::ops::detail {
template <class Schedule, class Output, class Epilogue,
          class RowPolicy = Q5IdentityRows<Schedule::kBlockRows>>
void launch_q5_a16_sliced_k_mma(const Q5LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream, RowPolicy row_policy = {}) {
    validate_q5_operands(operands);
    if (operands.k % 8 != 0) throw std::invalid_argument("Q5 MMA requires K aligned to 8");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument("Q5 sliced-K static geometry requires K == padded K");
    }
    for_each_token_slice(operands.tokens, Schedule::kTokenCapacity, [&](int offset, int count) {
        const int output_rows = row_policy.output_rows(operands.rows);
        const dim3 grid(div_up(output_rows, RowPolicy::kOutputRowsPerCta),
                        div_up(count, Schedule::kTokenCapacity));
        const auto* x   = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const bool full = operands.rows % Schedule::kBlockRows == 0 &&
                          output_rows % RowPolicy::kOutputRowsPerCta == 0 &&
                          operands.k == operands.padded_k && operands.k % Schedule::kBlockK == 0 &&
                          reinterpret_cast<std::uintptr_t>(operands.scales) %
                                  (Schedule::kKWarps * sizeof(__half)) ==
                              0;
        const bool full_tokens = count % Schedule::kTokenCapacity == 0;
        const auto launch      = [&]<bool Full, bool FullTokens>() {
            q5_a16_sliced_k_mma_kernel<Schedule, Full, FullTokens>
                <<<grid, Schedule::kThreads, 0, stream>>>(
                    x, operands.codes, operands.high, operands.scales, output, epilogue,
                    operands.rows, operands.k, count, operands.padded_k, offset, row_policy);
        };
        if (full && full_tokens)
            launch.template operator()<true, true>();
        else if (full)
            launch.template operator()<true, false>();
        else if (full_tokens)
            launch.template operator()<false, true>();
        else
            launch.template operator()<false, false>();
        CUDA_CHECK(cudaGetLastError());
    });
}
} // namespace ninfer::ops::detail
