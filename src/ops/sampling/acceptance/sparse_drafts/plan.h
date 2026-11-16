#pragma once
#include "ops/sampling/kernels/workspace.h"
#include "ninfer/ops/sampling.h"
#include <algorithm>
#include "core/platform.h"

namespace ninfer::ops::sparse_drafts_detail {
#if NINFER_TARGET_SPARK
using Schedule = sampling_detail::TopKSchedule<128, 4, 128, 4, 25>;
#else
using Schedule = sampling_detail::TopKSchedule<256, 2, 256, 2, 25>;
#endif

struct Plan {
    sampling_detail::WorkspaceLayout workspace;
    int partial_blocks = 0;
    int groups         = 0;
    bool raw_greedy    = false;
    int partial_tile   = 512;
};

inline Plan select_plan(int token_domain, int columns, bool raw_greedy = false) {
    Plan plan;
    plan.raw_greedy = raw_greedy;
    if (raw_greedy) return plan;
    plan.partial_tile   = (token_domain >= 4096 ? 4096 : 512);
    plan.partial_blocks = common::div_up(token_domain, plan.partial_tile);
    plan.groups         = common::div_up(plan.partial_blocks, Schedule::kSamplerPartialsPerGroup);
    plan.workspace =
        sampling_detail::make_workspace(columns, plan.partial_blocks, plan.groups, true);
    return plan;
}

inline std::size_t capacity(int token_domain, int max_columns) {
    return select_plan(token_domain, max_columns).workspace.bytes;
}

void launch(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
            const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
            const Tensor& base_positions, Tensor& licensed_tokens, Tensor& licensed_counts,
            std::int32_t token_domain, const SamplingConfig* configs, const Tensor& live_rows,
            const Plan& plan, MemorySpan workspace, CudaExecutionView execution);
} // namespace ninfer::ops::sparse_drafts_detail
