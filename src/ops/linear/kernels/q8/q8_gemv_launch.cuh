#pragma once

#include "ops/common/numeric/math.h"
#include "core/device.h"
#include "ops/linear/kernels/q8/q8_a16_gemv.cuh"

namespace ninfer::ops::detail {
template <class Schedule, class Output, class Epilogue>
void launch_q8_a16_gemv(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    validate_q8_operands(operands);
    if (operands.tokens > Schedule::kBlockTokens || operands.k % 8 != 0)
        throw std::invalid_argument(
            "Q8 GEMV requires tokens within its capacity and K aligned to 8");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument(
                "Q8 GEMV static K requires matching, padding-free operands");
    }
    const auto launch = [&]<bool FullRows>() {
        if constexpr (Schedule::kBlockTokens == 1)
            q8_a16_gemv_kernel<Schedule, FullRows>
                <<<common::div_up(operands.rows, Schedule::kBlockRows), Schedule::kThreads, 0,
                   stream>>>(operands, output, epilogue);
        else
            q8_a16_gemv_tile_kernel<Schedule, FullRows>
                <<<common::div_up(operands.rows, Schedule::kBlockRows), Schedule::kThreads, 0,
                   stream>>>(operands, output, epilogue);
        CUDA_CHECK(cudaGetLastError());
    };
    if (operands.rows % Schedule::kBlockRows == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}
} // namespace ninfer::ops::detail
