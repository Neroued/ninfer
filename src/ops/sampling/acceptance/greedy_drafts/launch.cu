#include "ops/sampling/acceptance/greedy_drafts/plan.h"
#include "ops/sampling/acceptance/greedy_drafts/kernel.cuh"
#include "core/device.h"

namespace ninfer::ops::greedy_drafts_detail {
using namespace sampling_detail;
using namespace common;

void launch(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
            const Tensor& current_extents, const Tensor& base_positions, Tensor& licensed_tokens,
            Tensor& licensed_counts, std::int32_t token_domain, const SamplingConfig* configs,
            const Tensor& live_rows, const Plan& plan, MemorySpan workspace,
            CudaExecutionView execution) {
    const auto stream                = execution.stream();
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t cols          = drafts.ne[0] + 1;
    const std::int32_t batch         = drafts.ne[1];
    const auto* live                 = static_cast<const std::int32_t*>(live_rows.data);
    const auto& layout               = plan.workspace;
    if (layout.bytes == 0) {
        speculative_accept_greedy_drafts_kernel<RowSchedule>
            <<<batch, RowSchedule::kSamplerBlock, 0, stream>>>(
                static_cast<const std::int32_t*>(target_tokens.data),
                static_cast<const __nv_bfloat16*>(logits.data), token_rows(drafts),
                static_cast<const std::int32_t*>(current_extents.data),
                static_cast<const std::int32_t*>(base_positions.data),
                static_cast<std::int32_t*>(licensed_tokens.data),
                static_cast<std::int32_t*>(licensed_counts.data), configs, token_domain,
                physical_rows, drafts.ne[0], live);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int32_t partial_blocks = plan.partial_blocks;
    const std::int32_t groups         = plan.groups;
    const Workspace scratch           = layout.bind(workspace);
    for (int begin = 0; begin < batch;) {
        const int rows = std::min(batch - begin, 65535);
        const SpeculativeSelectionInput input{
            .logits          = static_cast<const __nv_bfloat16*>(logits.data),
            .drafts          = token_rows(drafts),
            .current_extents = static_cast<const std::int32_t*>(current_extents.data),
            .configs         = configs,
            .live_rows       = live,
            .token_domain    = token_domain,
            .physical_rows   = physical_rows,
            .columns         = cols,
            .draft_count     = drafts.ne[0],
            .workspace       = scratch,
            .row_stride      = layout.bytes,
            .first_row       = begin};
        launch_speculative_selection(input, rows, partial_blocks, plan.partial_tile, execution);
        const dim3 batched_group_grid(static_cast<unsigned int>(groups),
                                      static_cast<unsigned int>(cols),
                                      static_cast<unsigned int>(rows));
        speculative_sampling_group_finalize_kernel<ParallelSchedule, false>
            <<<batched_group_grid, ParallelSchedule::kSamplerGroupBlock, 0, stream>>>(
                static_cast<const std::int32_t*>(target_tokens.data), token_rows(drafts), nullptr,
                nullptr, static_cast<const std::int32_t*>(current_extents.data),
                static_cast<const std::int32_t*>(base_positions.data),
                static_cast<std::int32_t*>(licensed_tokens.data),
                static_cast<std::int32_t*>(licensed_counts.data), configs, token_domain, cols,
                partial_blocks, groups, scratch, layout.bytes, live, begin);
        CUDA_CHECK(cudaGetLastError());
        begin += rows;
    }
}

} // namespace ninfer::ops::greedy_drafts_detail
