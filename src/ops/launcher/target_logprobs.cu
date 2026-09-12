// Implements: include/ninfer/ops/target_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/target_logprobs.h"

#include "core/device.h"
#include "ops/kernel/target_logprobs.cuh"

namespace ninfer::ops::detail {

void target_logprobs_launch(const Tensor& logits, const Tensor& target_ids, std::int32_t valid_rows,
                            Tensor& output, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    target_logprobs_kernel<kTargetLogprobsBlock><<<columns, kTargetLogprobsBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(target_ids.data), static_cast<float*>(output.data),
        valid_rows, logits.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
