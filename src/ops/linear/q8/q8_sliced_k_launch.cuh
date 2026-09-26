#pragma once
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_a16_sliced_k_mma.cuh"

namespace ninfer::ops::detail {
template <class Schedule, class RowPolicy = Q8SlicedKIdentityRows, class Output, class Epilogue>
void launch_q8_a16_sliced_k_mma(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream, RowPolicy row_policy = {}) {
    validate_q8_operands(operands);
    if (operands.k % 8 != 0) throw std::invalid_argument("Q8 sliced-K MMA requires K aligned to 8");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument(
                "Q8 sliced-K static K requires matching, padding-free operands");
    }
    if constexpr (Schedule::kExactTokens) {
        if (operands.tokens != Schedule::kTokenCapacity)
            throw std::invalid_argument("Q8 sliced-K exact tokens differ from the schedule");
    }
    constexpr int ratio = Schedule::kBlockRows / RowPolicy::kOutputRowsPerCta;
    static_assert(Schedule::kBlockRows % RowPolicy::kOutputRowsPerCta == 0);
    if (operands.rows % ratio != 0)
        throw std::invalid_argument("Q8 sliced-K row mapping requires complete row pairs");
    for_each_token_slice(operands.tokens, Schedule::kTokenCapacity, [&](int offset, int count) {
        const dim3 grid(div_up(operands.rows / ratio, RowPolicy::kOutputRowsPerCta),
                        div_up(count, Schedule::kTokenCapacity));
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel =
                q8_a16_sliced_k_mma_kernel<Schedule, Full, Output, Epilogue, RowPolicy>;
            const int shared = q8_prepare_shared<Schedule::kSharedBytes, kernel>();
            kernel<<<grid, Schedule::kThreads, shared, stream>>>(operands, output, epilogue,
                                                                 row_policy, offset);
            CUDA_CHECK(cudaGetLastError());
        };
        if (operands.rows % Schedule::kBlockRows == 0 && operands.k == operands.padded_k &&
            operands.k % Schedule::kBlockK == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}
} // namespace ninfer::ops::detail
