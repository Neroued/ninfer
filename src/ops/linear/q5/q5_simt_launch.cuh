#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q5/q5_a16_simt.cuh"
#include "ops/linear/q5/q5_a16_direct_simt.cuh"
#include "ops/linear/q5/q5_operands.h"

namespace ninfer::ops::detail {
template <class Schedule, bool Full, bool FullK, bool TriggerPdl, bool JoinPdl, bool Dependent,
          class Output, class Epilogue>
void launch_q5_a16_simt_slice(const Q5LinearOperands& operands, Output output, Epilogue epilogue,
                              int token_begin, int tokens, cudaStream_t stream) {
    const dim3 grid(div_up(operands.rows, Schedule::kBlockRows),
                    div_up(tokens, Schedule::kBlockTokens));
    const dim3 block(Schedule::kThreads);
    const auto* x = operands.x + static_cast<std::int64_t>(token_begin) * operands.k;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {grid, block, 0, stream},
            q5_a16_simt_kernel<Schedule, Full, FullK, Output, Epilogue, TriggerPdl, JoinPdl>, x,
            operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
            operands.k, tokens, operands.padded_k, token_begin));
    } else {
        q5_a16_simt_kernel<Schedule, Full, FullK, Output, Epilogue, TriggerPdl, JoinPdl>
            <<<grid, block, 0, stream>>>(x, operands.codes, operands.high, operands.scales, output,
                                         epilogue, operands.rows, operands.k, tokens,
                                         operands.padded_k, token_begin);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class Schedule, bool TriggerPdl = false, bool JoinPdl = false, bool Dependent = false,
          class Output, class Epilogue>
void launch_q5_a16_simt(const Q5LinearOperands& operands, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    validate_q5_operands(operands);
    constexpr int W   = Schedule::kWarpsPerRow;
    const bool full_k = !Schedule::kPredicated && operands.k % 128 == 0 &&
                        (operands.k / 128) % W == 0 &&
                        (operands.k / 64 / W) % Schedule::kGroupsPerWarpStage == 0;
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const bool full = !Schedule::kPredicated && operands.rows % Schedule::kBlockRows == 0 &&
                          count % Schedule::kBlockTokens == 0;
        if (full && full_k)
            launch_q5_a16_simt_slice<Schedule, true, true, TriggerPdl, JoinPdl, Dependent>(
                operands, output, epilogue, offset, count, stream);
        else if (full)
            launch_q5_a16_simt_slice<Schedule, true, false, TriggerPdl, JoinPdl, Dependent>(
                operands, output, epilogue, offset, count, stream);
        else if (full_k)
            launch_q5_a16_simt_slice<Schedule, false, true, TriggerPdl, JoinPdl, Dependent>(
                operands, output, epilogue, offset, count, stream);
        else
            launch_q5_a16_simt_slice<Schedule, false, false, TriggerPdl, JoinPdl, Dependent>(
                operands, output, epilogue, offset, count, stream);
    });
}

template <class Schedule, bool TriggerPdl = false, bool JoinPdl = false, bool Dependent = false,
          class Output, class Epilogue>
void launch_q5_a16_direct_simt(const Q5LinearOperands& operands, Output output, Epilogue epilogue,
                               cudaStream_t stream) {
    validate_q5_operands(operands);
    if (operands.k % 8 != 0) throw std::invalid_argument("Q5 direct SIMT requires K aligned to 8");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK)
            throw std::invalid_argument("Q5 direct SIMT static K differs from operands");
    }
    if constexpr (Schedule::kExactTokens) {
        if (operands.tokens != Schedule::kBlockTokens)
            throw std::invalid_argument("Q5 exact direct SIMT token count differs from schedule");
    }
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(div_up(operands.rows, Schedule::kBlockRows),
                        div_up(count, Schedule::kBlockTokens));
        const dim3 block(Schedule::kThreads);
        const auto* x     = operands.x + std::int64_t(offset) * operands.k;
        const auto launch = [&]<bool FullK, bool FullTokens>() {
            if constexpr (Dependent) {
                CUDA_CHECK(pdl::launch_dependent(
                    {grid, block, 0, stream},
                    q5_a16_direct_simt_kernel<Schedule, FullK, FullTokens, Output, Epilogue,
                                              TriggerPdl, JoinPdl>,
                    x, operands.codes, operands.high, operands.scales, output, epilogue,
                    operands.rows, operands.k, count, operands.padded_k, offset));
            } else {
                q5_a16_direct_simt_kernel<Schedule, FullK, FullTokens, Output, Epilogue, TriggerPdl,
                                          JoinPdl><<<grid, block, 0, stream>>>(
                    x, operands.codes, operands.high, operands.scales, output, epilogue,
                    operands.rows, operands.k, count, operands.padded_k, offset);
                CUDA_CHECK(cudaGetLastError());
            }
        };
        const auto select_tokens = [&]<bool FullK>() {
            if constexpr (Schedule::kExactTokens)
                launch.template operator()<FullK, true>();
            else if (count % Schedule::kBlockTokens == 0)
                launch.template operator()<FullK, true>();
            else
                launch.template operator()<FullK, false>();
        };
        if constexpr (Schedule::kStaticK > 0)
            select_tokens.template operator()<Schedule::kStaticK % Schedule::kBlockK == 0>();
        else if (operands.k % Schedule::kBlockK == 0)
            select_tokens.template operator()<true>();
        else
            select_tokens.template operator()<false>();
    });
}
} // namespace ninfer::ops::detail
