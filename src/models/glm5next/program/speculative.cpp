#include "models/glm5next/program/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
Tensor RankProgramImpl::draft_proposals(int batch, int columns) {
    const auto& region = frame.speculative->drafts;
    return parameters.mtp ? resources.packed(region, {batch, columns}).permute({1, 0, 2, 3})
                          : resources.packed(region, {columns, batch});
}

void RankProgramImpl::prepare_speculative() {
    auto& f           = *compute;
    const auto& work  = std::get<SpeculativeWork>(f.request.work.operation);
    const auto& io    = frame;
    const auto rank   = parameters.profile->local.rank;
    const auto& route = *work.speculative;
    const int b = work.rows.size(), d = static_cast<int>(route.mtp_steps),
              width = dimension(route.verify_width);
    f.physical      = work.batch;
    f.tokens        = width;
    f.sequence      = false;
    std::vector<std::uint32_t> valid(b);
    for (int i = 0; i < b; ++i) valid[i] = work.draft_extents[i] + 1;
    prepare_inputs(work.rows, f.physical, width, {valid.data(), work.rows.size()});
    const auto i32 = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host(r).data);
    };
    const auto physical_rows = f.batch.rows;
    std::fill_n(i32(io.speculative->extents), physical_rows, 0);
    if (io.mtp)
        for (const auto& step : io.mtp->draft_steps) {
            std::fill_n(i32(step.positions), physical_rows, 0);
            std::fill_n(i32(step.valid_columns), physical_rows, 0);
        }
    f.draft_visible.assign(parameters.mtp ? d : int(route.head_columns != 0), 0);
    for (int i = 0; i < b; ++i) {
        const auto logical              = f.physical.members[i];
        const auto& r                   = work.rows[logical];
        i32(io.speculative->extents)[i] = work.draft_extents[logical];
        if (parameters.draft && work.draft_extents[logical]) {
            // Active rows always run the full masked proposal block.
            const auto end     = r.main_frontier + shape.draft_tokens + 1;
            f.draft_visible[0] = std::max(f.draft_visible[0], end);
        }
        if (parameters.mtp)
            for (int s = 0; s < d; ++s) {
                const auto& step  = io.mtp->draft_steps[s];
                const bool active = std::uint32_t(s) < work.draft_extents[logical];
                i32(step.positions)[i] =
                    r.mtp_frontier + std::min<std::uint32_t>(s, work.draft_extents[logical]);
                i32(step.valid_columns)[i] = active;
                if (active && r.attention_rank == rank)
                    f.draft_visible[s] = std::max(f.draft_visible[s], r.mtp_frontier + s + 1);
            }
    }
    if (parameters.mtp)
        for (const auto& step : io.mtp->draft_steps)
            fill_local_metadata(step, 1, f.batch.attention.rows);
    f.hidden               = target_hidden(f.batch);
    const bool read_drafts = !f.decision;
    if (graphs) {
        const auto& graph_route =
            f.graph_route.emplace(f.physical.shape, route.verify_width, route.mtp_steps,
                                  route.head_columns, route.graph_entry);
        std::vector<ops::ContextBucket> draft_buckets;
        draft_buckets.reserve(f.draft_visible.size());
        for (const auto bound : f.draft_visible)
            draft_buckets.push_back(ops::select_context_bucket(std::max(1U, bound)));
        const auto main_bucket =
            ops::select_context_bucket(std::max(1U, f.batch.attention.max_visible_tokens));
        const ProposalReplay proposal{.buckets = draft_buckets, .readback = read_drafts};
        if (graph_route.width > 1 &&
            graphs->topology() == models::ProposalTopology::SplitProposal) {
            graphs->launch_proposal(graph_route, proposal);
            (void)graphs->start_target(graph_route, {.bucket = main_bucket, .head = true});
        } else {
            (void)graphs->start_target(graph_route,
                                       {.bucket = main_bucket, .head = true, .proposal = proposal});
        }
    } else {
        execute_proposal(f.batch, f.draft_visible, route.head_columns, read_drafts);
        execute_target(f.batch, true);
    }
    f.head_ready = true;
    f.phase      = ComputePhase::Decision;
    decide();
}

void RankProgramImpl::accept_speculative(int rows, int width, int head_columns) {
    if (parameters.mtp)
        accept_mtp(rows, width);
    else
        accept_dflash(rows, width, head_columns);
}
} // namespace ninfer::models::glm5next::detail
