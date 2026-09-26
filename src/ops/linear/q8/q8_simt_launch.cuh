#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_a16_simt.cuh"
#include "ops/linear/q8/q8_operands.h"

namespace ninfer::ops::detail {
template <class Schedule, bool Full, bool FullK, bool TriggerPdl, bool JoinPdl, bool Dependent,
          class Output, class Epilogue>
void launch_q8_a16_simt_slice(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                              int token_begin, int tokens, cudaStream_t stream) {
    const dim3 grid(div_up(operands.rows, Schedule::kBlockRows),
                    div_up(tokens, Schedule::kBlockTokens));
    const dim3 block(Schedule::kThreads);
    const auto* x = operands.x + static_cast<std::int64_t>(token_begin) * operands.k;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {grid, block, 0, stream},
            q8_a16_simt_kernel<Schedule, Full, FullK, Output, Epilogue, TriggerPdl, JoinPdl>, x,
            operands.codes, operands.scales, output, epilogue, operands.rows, operands.k, tokens,
            operands.padded_k, token_begin));
    } else {
        q8_a16_simt_kernel<Schedule, Full, FullK, Output, Epilogue, TriggerPdl, JoinPdl>
            <<<grid, block, 0, stream>>>(x, operands.codes, operands.scales, output, epilogue,
                                         operands.rows, operands.k, tokens, operands.padded_k,
                                         token_begin);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class Schedule, bool TriggerPdl = false, bool JoinPdl = false, bool Dependent = false,
          class Output, class Epilogue>
void launch_q8_a16_simt(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    validate_q8_operands(operands);
    constexpr int W   = Schedule::kWarpsPerRow;
    const bool full_k = !Schedule::kPredicated && operands.k % 64 == 0 &&
                        (operands.k / 64) % W == 0 &&
                        (operands.k / 32 / W) % Schedule::kGroupsPerWarpStage == 0;
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const bool full = !Schedule::kPredicated && operands.rows % Schedule::kBlockRows == 0 &&
                          count % Schedule::kBlockTokens == 0;
        if (full && full_k)
            launch_q8_a16_simt_slice<Schedule, true, true, TriggerPdl, JoinPdl, Dependent>(
                operands, output, epilogue, offset, count, stream);
        else if (full)
            launch_q8_a16_simt_slice<Schedule, true, false, TriggerPdl, JoinPdl, Dependent>(
                operands, output, epilogue, offset, count, stream);
        else if (full_k)
            launch_q8_a16_simt_slice<Schedule, false, true, TriggerPdl, JoinPdl, Dependent>(
                operands, output, epilogue, offset, count, stream);
        else
            launch_q8_a16_simt_slice<Schedule, false, false, TriggerPdl, JoinPdl, Dependent>(
                operands, output, epilogue, offset, count, stream);
    });
}

} // namespace ninfer::ops::detail
