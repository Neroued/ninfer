#pragma once
#include "ops/sampling/kernels/workspace.h"
#include "ninfer/ops/sampling.h"
#include "core/platform.h"
#include <algorithm>

namespace ninfer::ops::sparse_drafts_detail {
#if NINFER_TARGET_SPARK
using Schedule = sampling_detail::TopKSchedule<128, 4, 128, 4, 25>;
#else
using Schedule = sampling_detail::TopKSchedule<256, 2, 256, 2, 25>;
#endif

inline constexpr int kFinalMergePartials   = 32;
inline constexpr int kTargetSms            = 170;
inline constexpr int kMaxStreamingPartials = 48;
inline constexpr int kMaxStreamingWaves    = 2;

inline bool use_streaming(int token_domain) {
    return NINFER_TARGET_RTX5090 && token_domain > 7 * 2048 && token_domain <= 100 * 2048;
}

inline int streaming_partials(int batch, int columns, int sm_count) {
    const int sms   = std::clamp(sm_count, 1, kTargetSms);
    const auto work = std::int64_t(batch) * columns;
    // Keep the small grid close to one CTA per SM. Around 32 columns,
    // approaching two waves from below leaves room for the row finalizer.
    // Larger batches have enough work to keep two waves in flight.
    const auto parts = work >= 24 && work <= 48
                           ? (2 * sms) / work
                           : common::div_up<std::int64_t>(work >= 96 ? 2 * sms : sms, work);
    return std::clamp(static_cast<int>(parts), 1, kMaxStreamingPartials);
}

struct Plan {
    sampling_detail::WorkspaceLayout workspace;
    std::size_t row_bytes = 0;
    int partial_blocks    = 0;
    int groups            = 0;
    int partial_tile      = 512;
    bool raw_greedy       = false;
    bool streaming        = false;
};

inline Plan select_plan(int token_domain, int columns, int batch, bool raw_greedy = false,
                        int sm_count = kTargetSms) {
    Plan plan;
    plan.raw_greedy = raw_greedy;
    if (raw_greedy) return plan;
    plan.streaming = use_streaming(token_domain);
    if (plan.streaming) {
        plan.partial_blocks = streaming_partials(batch, columns, sm_count);
        // Match a 256-byte segment when the column base permits it.
        constexpr int alignment = 128;
        plan.partial_tile = alignment * common::div_up(common::div_up(token_domain, alignment),
                                                       plan.partial_blocks);
        plan.row_bytes    = std::size_t(columns) * plan.partial_blocks * 2 * sizeof(std::uint64_t);
    } else {
        plan.partial_tile   = token_domain >= 4096 ? 4096 : 512;
        plan.partial_blocks = common::div_up(token_domain, plan.partial_tile);
        plan.groups = common::div_up(plan.partial_blocks, Schedule::kSamplerPartialsPerGroup);
        plan.workspace =
            sampling_detail::make_workspace(columns, plan.partial_blocks, plan.groups, true);
        plan.row_bytes = plan.workspace.bytes;
    }
    return plan;
}

void launch(const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
            const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
            const Tensor& base_positions, Tensor& licensed_tokens, Tensor& licensed_counts,
            std::int32_t token_domain, const SamplingConfig* configs, const Tensor& live_rows,
            const Plan& plan, MemorySpan workspace, CudaExecutionView execution);
} // namespace ninfer::ops::sparse_drafts_detail
