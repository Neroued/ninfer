#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q6/q6_operands.h"
#include "ops/linear/q6/q6_a16_simt.cuh"

namespace ninfer::ops::detail {

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue>
void launch_q6_a16_simt(const Q6LinearOperands& operands, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    validate_q6_operands(operands);
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(static_cast<unsigned>(div_up(operands.rows, Schedule::kBlockRows)),
                        static_cast<unsigned>(div_up(count, Schedule::kBlockTokens)));
        const auto* x = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        q6_a16_simt_kernel<Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
            x, operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
            operands.k, count, operands.padded_k, offset);
        CUDA_CHECK(cudaGetLastError());
    });
}

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue>
void launch_q6_a16_gemv(const Q6LinearOperands& operands, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    static_assert(Schedule::kBlockTokens == 1);
    if (operands.tokens != 1) throw std::invalid_argument("Q6 GEMV requires T=1");
    launch_q6_a16_simt<Schedule>(operands, output, epilogue, stream);
}

} // namespace ninfer::ops::detail
