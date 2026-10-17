#pragma once
#include "ops/sampling/kernels/workspace.h"
#include "ninfer/ops/sampling.h"
#include <algorithm>
#include "core/platform.h"

namespace ninfer::ops::greedy_drafts_detail {
using RowSchedule = sampling_detail::TopKSchedule<256, 2, 256, 2, 25>;
#if NINFER_TARGET_SPARK
using ParallelSchedule = sampling_detail::TopKSchedule<128, 4, 128, 4, 25>;
#else
using ParallelSchedule = sampling_detail::TopKSchedule<256, 2, 256, 2, 25>;
#endif

inline constexpr int kMaxColumns = 16;

struct Plan {
    sampling_detail::WorkspaceLayout workspace;
    int partial_blocks = 0;
    int groups         = 0;
};

inline Plan select_plan(int token_domain, int columns) {
    Plan plan;
    // The final merge has finite capacity; larger domains use a single CTA.
    plan.partial_blocks = common::div_up(token_domain, ParallelSchedule::kSamplerPartialTileItems);
    plan.groups = common::div_up(plan.partial_blocks, ParallelSchedule::kSamplerPartialsPerGroup);
    if (token_domain > RowSchedule::kSamplerTileItems && columns <= kMaxColumns &&
        plan.groups * sampling_detail::kSamplerCandidateCap <=
            ParallelSchedule::kSamplerGroupTileItems)
        plan.workspace =
            sampling_detail::make_workspace(columns, plan.partial_blocks, plan.groups, true);
    return plan;
}

inline std::size_t capacity(int token_domain, int min_columns, int max_columns) {
    if (min_columns > kMaxColumns) return 0;
    return select_plan(token_domain, std::min(max_columns, kMaxColumns)).workspace.bytes;
}

void launch(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
            const Tensor& current_extents, Tensor& lengths, Tensor& anchors,
            Tensor& licensed_tokens, Tensor& licensed_counts, Tensor& accepted,
            std::int32_t token_domain, const SamplingConfig* configs, const Plan& plan,
            MemorySpan workspace, cudaStream_t stream);
} // namespace ninfer::ops::greedy_drafts_detail
