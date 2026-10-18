#include "models/qwen3_5/program/internal.h"

#include "models/qwen3_5/execution/draft.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

void RankProgramImpl::commit_draft_history(ComputeOperation& operation) {
    if (!resources.dflash) return;
    if (!operation.commit) throw std::logic_error("draft history commit has no accepted prefix");
    std::uint32_t maximum = 0;
    for (const auto& row : operation.commit->rows)
        if (row.terminal && !row.discarded) maximum = std::max(maximum, row.accepted_tokens);
    if (maximum == 0) return;

    auto& frame       = resources.frame;
    auto& draft       = *frame.dflash;
    auto& state       = *resources.dflash;
    const auto width  = static_cast<std::int32_t>(draft_window + 1);
    const auto batch  = static_cast<std::int32_t>(operation.batch);
    const auto cuda   = device.execution_view();
    const auto prefix = [](const Tensor& storage, std::initializer_list<std::int32_t> shape) {
        return Tensor({storage.data, storage.bytes(), storage.device}, storage.dtype, shape);
    };
    auto scope = resources.work.scope();
    Tensor features =
        resources.work.alloc(DType::BF16, {state.pending_features.ne[0], width, batch});
    auto positions = prefix(draft.append_positions, {width, batch});
    auto counts    = draft.append_counts.slice(0, 0, batch);
    ops::prepare_ragged_prefix(state.pending_features, frame.residency_slots.slice(0, 0, batch),
                               frame.draft_append_starts.slice(0, 0, batch),
                               frame.draft_append_ends.slice(0, 0, batch), features, positions,
                               counts, cuda);
    execution::draft_append_context({parameters, resources.work, cuda, state.local,
                                     state.full ? &*state.full : nullptr, proposal_head},
                                    features, positions, counts,
                                    frame.commit_destination_slots.slice(0, 0, batch),
                                    frame.backend_table_rows.slice(0, 0, batch), {0, maximum});
}

} // namespace ninfer::models::qwen3_5::detail
