#include "ops/linear_attention/kimi_delta_attention/recurrent.cuh"
#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "core/platform.h"
#include "ops/linear_attention/kimi_delta_attention/kernels/record.cuh"

#include <algorithm>

namespace ninfer::ops::detail::kimi_delta_attention {
namespace {
constexpr int kGridRows = 65535;
}

void launch_recurrent(const Arguments& args, cudaStream_t stream) {
    recurrent_kernel<false, false>
        <<<dim3(args.value_heads, 1, kStateDim / kBlockDv), dim3(common::kWarpSize, kNumWarps), 0,
           stream>>>(args, nullptr, nullptr, nullptr, {}, 0);
    CUDA_CHECK(cudaGetLastError());
}

void launch_batch_update(const Arguments& args, const std::int64_t* sources,
                         const std::int64_t* destinations, const std::int32_t* valid, int batch,
                         cudaStream_t stream) {
    for (std::int64_t begin = 0; begin < batch; begin += kGridRows) {
        const int rows = std::min<std::int64_t>(kGridRows, batch - begin);
#if NINFER_TARGET_SPARK
        if (batch >= 16) {
            recurrent_kernel<true, false, kDvPerWarp, true>
                <<<dim3(args.value_heads * (kStateDim / kBlockDv), rows),
                   dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                    args, sources, destinations, valid, {}, static_cast<int>(begin));
        } else
#endif
            recurrent_kernel<true, false><<<dim3(args.value_heads, rows, kStateDim / kBlockDv),
                                            dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                args, sources, destinations, valid, {}, static_cast<int>(begin));
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_record(const Arguments& args, const std::int64_t* sources, const std::int32_t* valid,
                   RecurrenceRecords records, int batch, cudaStream_t stream) {
    for (std::int64_t begin = 0; begin < batch; begin += kGridRows) {
        const int batch_rows = std::min<std::int64_t>(kGridRows, batch - begin);
        if (args.tokens == 1) {
            recurrent_kernel<true, true><<<dim3(args.value_heads, batch_rows, kStateDim / kBlockDv),
                                           dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                args, sources, nullptr, valid, records, static_cast<int>(begin));
        } else {
            const auto schedule = select_record_schedule(args.value_heads, args.tokens, batch);
            if (schedule.steps == 16) {
                constexpr int tiles = kStateDim / (kNumWarps * 16);
                staged_record_kernel<16, 16, true, 1, true>
                    <<<dim3(args.value_heads * tiles, batch_rows),
                       dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                        args, sources, valid, records, static_cast<int>(begin));
                CUDA_CHECK(cudaGetLastError());
                continue;
            }
            const auto submit = [&]<int Rows>() {
                constexpr int tiles = kStateDim / (kNumWarps * Rows);
                if (schedule.group_state_tiles) {
                    staged_record_kernel<4, Rows, true>
                        <<<dim3(args.value_heads * tiles, batch_rows),
                           dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                            args, sources, valid, records, static_cast<int>(begin));
                    return;
                }
                staged_record_kernel<4, Rows><<<dim3(args.value_heads, batch_rows, tiles),
                                                dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                    args, sources, valid, records, static_cast<int>(begin));
            };
            if (schedule.rows == 8)
                submit.template operator()<8>();
            else if (schedule.rows == 2)
                submit.template operator()<2>();
            else
                submit.template operator()<4>();
        }
        CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ninfer::ops::detail::kimi_delta_attention
