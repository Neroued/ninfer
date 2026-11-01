#include "ops/linear_attention/kimi_delta_attention/recurrent.cuh"
#include "ops/linear_attention/kimi_delta_attention/plan.h"
#include "core/platform.h"
#include "ops/linear_attention/kimi_delta_attention/kernels/record.cuh"

namespace ninfer::ops::detail::kimi_delta_attention {

void launch_recurrent(const Arguments& args, cudaStream_t stream) {
    recurrent_kernel<false, false>
        <<<dim3(args.value_heads, 1, kStateDim / kBlockDv), dim3(common::kWarpSize, kNumWarps), 0,
           stream>>>(args, nullptr, nullptr, nullptr, {});
    CUDA_CHECK(cudaGetLastError());
}

void launch_batch_update(const Arguments& args, const std::int64_t* sources,
                         const std::int64_t* destinations, int batch, cudaStream_t stream) {
    recurrent_kernel<true, false>
        <<<dim3(args.value_heads, batch, kStateDim / kBlockDv), dim3(common::kWarpSize, kNumWarps),
           0, stream>>>(args, sources, destinations, nullptr, {});
    CUDA_CHECK(cudaGetLastError());
}

void launch_record(const Arguments& args, const std::int64_t* sources, const std::int32_t* valid,
                   RecurrenceRecords records, int batch, cudaStream_t stream) {
    if (args.tokens == 1) {
        recurrent_kernel<true, true><<<dim3(args.value_heads, batch, kStateDim / kBlockDv),
                                       dim3(common::kWarpSize, kNumWarps), 0, stream>>>(
            args, sources, nullptr, valid, records);
    } else {
        const auto submit = [&]<int Rows>() {
            staged_record_kernel<4, Rows>
                <<<dim3(args.value_heads, batch, kStateDim / (kNumWarps * Rows)),
                   dim3(common::kWarpSize, kNumWarps), 0, stream>>>(args, sources, valid, records);
        };
#if NINFER_TARGET_SPARK
        if (select_record_rows(args.value_heads, args.tokens, batch) == 8)
            submit.template operator()<8>();
        else
#endif
            submit.template operator()<4>();
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::kimi_delta_attention
