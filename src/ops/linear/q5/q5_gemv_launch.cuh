#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q5/q5_a16_gemv.cuh"
#include "ops/linear/q5/q5_operands.h"

namespace ninfer::ops::detail {
template <class Schedule, bool TriggerPdl = false, bool JoinPdl = false, bool Dependent = false,
          class Output, class Epilogue>
void launch_q5_a16_gemv(const Q5LinearOperands& operands, Output output, Epilogue epilogue,
                        cudaStream_t stream) {
    validate_q5_operands(operands);
    if (operands.tokens != 1 || operands.k % 128 != 0)
        throw std::invalid_argument("Q5 GEMV requires T=1 and K aligned to 128");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument("Q5 GEMV static geometry requires K == padded K");
    }
    const dim3 grid(div_up(operands.rows, Schedule::kBlockRows));
    const dim3 block(Schedule::kThreads);
    const std::size_t shared_bytes = Schedule::kStageX ? std::size_t(operands.k) * 2 : 0;
    if (shared_bytes + Schedule::kSharedBytes > 48 * 1024)
        throw std::invalid_argument("Q5 GEMV shared memory exceeds 48 KiB");
    constexpr int W        = Schedule::kWarpsPerRow;
    const bool wide_scales = Schedule::kGroupsPerWarpTile % 8 == 0 &&
                             (operands.k / 64) % (W * 8) == 0 &&
                             (operands.padded_k / 64) % 8 == 0 &&
                             reinterpret_cast<std::uintptr_t>(operands.scales) % 16 == 0;
    const auto launch = [&]<bool WideScales>() {
        if constexpr (Dependent) {
            CUDA_CHECK(pdl::launch_dependent(
                {grid, block, shared_bytes, stream},
                q5_a16_gemv_kernel<Schedule, WideScales, Output, Epilogue, TriggerPdl, JoinPdl>,
                operands.x, operands.codes, operands.high, operands.scales, output, epilogue,
                operands.rows, operands.k, operands.padded_k));
        } else {
            q5_a16_gemv_kernel<Schedule, WideScales, Output, Epilogue, TriggerPdl, JoinPdl>
                <<<grid, block, shared_bytes, stream>>>(
                    operands.x, operands.codes, operands.high, operands.scales, output, epilogue,
                    operands.rows, operands.k, operands.padded_k);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    if (wide_scales)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}
} // namespace ninfer::ops::detail
