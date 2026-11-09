#include "models/glm5next/program/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
Tensor RankProgramImpl::draft_proposals(int batch) {
    const auto& region = resources.plan.frame.speculative->drafts;
    return parameters.mtp ? resources.tensor(region).slice(0, 0, batch).permute({1, 0, 2, 3})
                          : resources.packed(region, {dimension(shape.draft_tokens), batch});
}

void RankProgramImpl::prepare_speculative() {
    auto& f          = *compute;
    const auto& work = std::get<SpeculativeWork>(f.request.work.operation);
    const auto& io   = resources.plan.frame;
    const auto rank  = parameters.profile->local.rank;
    const int b = work.rows.size(), k = shape.draft_tokens, width = k + 1;
    f.physical = f.request.work.batch;
    f.tokens   = width;
    f.sequence = false;
    std::array<std::uint32_t, kMaximumConcurrency> valid{}, skip_first{};
    for (int i = 0; i < b; ++i) {
        valid[i]      = work.draft_extents[i] + 1;
        skip_first[i] = work.draft_extents[i] != 0;
    }
    prepare_inputs(work.rows, f.physical, width, {valid.data(), work.rows.size()});
    if (parameters.mtp) prepare_history_metadata(work.rows, {skip_first.data(), work.rows.size()});
    const auto i32 = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host(r).data);
    };
    f.draft_visible.assign(parameters.mtp ? k : 1, 0);
    for (int i = 0; i < b; ++i) {
        const auto logical              = f.physical.members[i];
        const auto& r                   = work.rows[logical];
        i32(io.speculative->extents)[i] = work.draft_extents[logical];
        if (parameters.draft) f.draft_visible[0] = std::max(f.draft_visible[0], r.main_frontier);
        if (parameters.mtp)
            for (int s = 0; s < k; ++s) {
                const auto& d     = io.mtp->draft_steps[s];
                const bool active = std::uint32_t(s) < work.draft_extents[logical];
                i32(d.positions)[i] =
                    r.mtp_frontier + std::min<std::uint32_t>(s, work.draft_extents[logical]);
                i32(d.valid_columns)[i] = active;
                if (active && r.attention_rank == rank)
                    f.draft_visible[s] = std::max(f.draft_visible[s], r.mtp_frontier + s + 1);
            }
    }
    if (parameters.mtp)
        for (int s = 0; s < k; ++s)
            fill_local_metadata(io.mtp->draft_steps[s], 1, f.batch.attention.rows);
    f.hidden               = resources.packed(io.pending_hidden,
                                              {dimension(parameters.profile->config.text.hidden), width, b});
    const bool read_drafts = !f.decision;
    f.graph_round          = graphs != nullptr;
    if (graphs)
        graphs->replay(f.batch, f.draft_visible, true, read_drafts);
    else
        execute_round(f.batch, f.draft_visible, true, read_drafts);
    f.head_ready = true;
    f.phase      = ComputePhase::Decision;
    decide();
}

void RankProgramImpl::accept_speculative(int rows) {
    if (parameters.mtp)
        accept_mtp(rows);
    else
        accept_dflash(rows);
}
} // namespace ninfer::models::glm5next::detail
