#pragma once
#include "ops/sampling/kernels/workspace.h"
#include "ninfer/ops/sampling.h"
#include <algorithm>
#include "core/platform.h"

namespace ninfer::ops::sample_detail {
using RowSchedule = sampling_detail::TopKSchedule<256, 2, 256, 2, 25>;
#if NINFER_TARGET_SPARK
using ParallelSchedule = sampling_detail::TopKSchedule<128, 4, 128, 4, 25>;
#else
using ParallelSchedule = sampling_detail::TopKSchedule<256, 2, 256, 2, 25>;
#endif

inline constexpr int kColumnSlab       = 65535;
inline constexpr int kThresholdColumns = 8;

struct Plan {
    sampling_detail::WorkspaceLayout workspace;
    int partial_blocks = 0;
    int groups         = 0;
    int partial_tile   = 512;
    std::size_t bytes  = 0;
};

inline Plan select_plan(int token_domain, int columns, bool probability) {
    Plan plan;
    // The final merge has finite capacity; larger domains use a single CTA.
    plan.partial_tile   = token_domain >= 4096 && columns >= kThresholdColumns ? 4096 : 512;
    plan.partial_blocks = 1 + (token_domain - 1) / plan.partial_tile;
    plan.groups = common::div_up(plan.partial_blocks, ParallelSchedule::kSamplerPartialsPerGroup);
    if (token_domain > RowSchedule::kSamplerTileItems &&
        plan.groups * sampling_detail::kSamplerCandidateCap <=
            ParallelSchedule::kSamplerGroupTileItems) {
        plan.workspace = sampling_detail::make_workspace(std::min(columns, kColumnSlab),
                                                         plan.partial_blocks, plan.groups, false);
        plan.bytes     = plan.workspace.bytes;
        if (probability)
            plan.bytes +=
                std::size_t(std::min(columns, kColumnSlab)) * plan.partial_blocks * sizeof(float);
    }
    return plan;
}

inline std::size_t capacity(int token_domain, int min_columns, int max_columns, bool probability) {
    auto bytes = select_plan(token_domain, max_columns, probability).bytes;
    if (min_columns < kThresholdColumns)
        bytes =
            std::max(bytes, select_plan(token_domain, std::min(max_columns, kThresholdColumns - 1),
                                        probability)
                                .bytes);
    return bytes;
}

void launch(const Tensor& logits, Tensor& out, float* probabilities, std::int32_t token_domain,
            const SamplingConfig* configs, const Tensor& logical_positions, std::int32_t purpose,
            const Tensor& live_rows, const Plan& plan, MemorySpan workspace, cudaStream_t stream);
} // namespace ninfer::ops::sample_detail
