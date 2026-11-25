#include "ops/linear_attention/kimi_delta_attention/recurrent.cuh"
#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "core/platform.h"
#include "ops/linear_attention/kimi_delta_attention/kernels/record.cuh"

#include <algorithm>

namespace ninfer::ops::detail::kimi_delta_attention {
namespace {
constexpr int kGridRows = 65535;

template <class Fn>
void dispatch_state_dtype(DType dtype, Fn&& fn) {
    switch (dtype) {
    case DType::FP32:
        fn.template operator()<float>();
        return;
    case DType::BF16:
        fn.template operator()<__nv_bfloat16>();
        return;
    default:
        throw std::logic_error("kimi_delta_attention: unsupported recurrent state dtype");
    }
}
} // namespace

void launch_recurrent(const Arguments& args, cudaStream_t stream) {
    dispatch_state_dtype(args.state_dtype, [&]<class StateT>() {
        recurrent_kernel<false, false, StateT>
            <<<dim3(args.value_heads, 1, kStateDim / kBlockDv), dim3(common::kWarpSize, kNumWarps),
               0, stream>>>(args, nullptr, nullptr, nullptr, {}, 0);
    });
    CUDA_CHECK(cudaGetLastError());
}

void launch_ragged_recurrent(const Arguments& args, RaggedSegments segments, int recurrent_before,
                             cudaStream_t stream) {
    dispatch_state_dtype(args.state_dtype, [&]<class StateT>() {
        for (std::int64_t begin = 0; begin < segments.count; begin += kGridRows) {
            const int rows = std::min<std::int64_t>(kGridRows, segments.count - begin);
#if NINFER_TARGET_SPARK
            if (segments.count >= 16) {
                recurrent_kernel<true, false, StateT, kDvPerWarp, true>
                    <<<dim3(args.value_heads * (kStateDim / kBlockDv), rows),
                       dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                        args, args.source_offset, args.destination_offset, nullptr, {},
                        static_cast<int>(begin), RaggedRecurrent{segments, recurrent_before});
            } else
#endif
                recurrent_kernel<true, false, StateT>
                    <<<dim3(args.value_heads, rows, kStateDim / kBlockDv),
                       dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                        args, args.source_offset, args.destination_offset, nullptr, {},
                        static_cast<int>(begin), RaggedRecurrent{segments, recurrent_before});
            CUDA_CHECK(cudaGetLastError());
        }
    });
}

void launch_batch_update(const Arguments& args, const std::int64_t* sources,
                         const std::int64_t* destinations, const std::int32_t* valid, int batch,
                         cudaStream_t stream) {
    dispatch_state_dtype(args.state_dtype, [&]<class StateT>() {
        for (std::int64_t begin = 0; begin < batch; begin += kGridRows) {
            const int rows = std::min<std::int64_t>(kGridRows, batch - begin);
#if NINFER_TARGET_SPARK
            if (batch >= 16) {
                recurrent_kernel<true, false, StateT, kDvPerWarp, true>
                    <<<dim3(args.value_heads * (kStateDim / kBlockDv), rows),
                       dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                        args, sources, destinations, valid, {}, static_cast<int>(begin));
            } else
#endif
                recurrent_kernel<true, false, StateT>
                    <<<dim3(args.value_heads, rows, kStateDim / kBlockDv),
                       dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                        args, sources, destinations, valid, {}, static_cast<int>(begin));
            CUDA_CHECK(cudaGetLastError());
        }
    });
}

void launch_record(const Arguments& args, const std::int64_t* sources, const std::int32_t* valid,
                   RecurrenceRecords records, int batch, cudaStream_t stream) {
    dispatch_state_dtype(args.state_dtype, [&]<class StateT>() {
        for (std::int64_t begin = 0; begin < batch; begin += kGridRows) {
            const int batch_rows = std::min<std::int64_t>(kGridRows, batch - begin);
            if (args.tokens == 1) {
                recurrent_kernel<true, true, StateT>
                    <<<dim3(args.value_heads, batch_rows, kStateDim / kBlockDv),
                       dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                        args, sources, nullptr, valid, records, static_cast<int>(begin));
            } else {
                const auto schedule = select_record_schedule(args.value_heads, args.tokens, batch);
                if (schedule.steps == 16) {
                    constexpr int tiles = kStateDim / (kNumWarps * 16);
                    staged_record_kernel<16, 16, StateT, true, 1, true>
                        <<<dim3(args.value_heads * tiles, batch_rows),
                           dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                            args, sources, valid, records, static_cast<int>(begin));
                    CUDA_CHECK(cudaGetLastError());
                    continue;
                }
                const auto submit = [&]<int Rows>() {
                    constexpr int tiles = kStateDim / (kNumWarps * Rows);
                    if (schedule.group_state_tiles) {
                        staged_record_kernel<4, Rows, StateT, true>
                            <<<dim3(args.value_heads * tiles, batch_rows),
                               dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
                                args, sources, valid, records, static_cast<int>(begin));
                        return;
                    }
                    staged_record_kernel<4, Rows, StateT>
                        <<<dim3(args.value_heads, batch_rows, tiles),
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
    });
}
} // namespace ninfer::ops::detail::kimi_delta_attention
