#include "models/qwen4_exp/session/planning.h"
#include "models/qwen4_exp/program/planning.h"
#include "models/qwen4_exp/session/internal.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {
namespace {
std::uint32_t count(std::uint64_t value) {
    if (value > UINT32_MAX) throw std::overflow_error("Qwen4Exp context directory is too large");
    return static_cast<std::uint32_t>(value);
}
} // namespace

SessionPlan plan_session(const ExecutionPlan& execution) {
    SessionPlan plan{execution.shape};
    plan.checkpoint_interval = execution.checkpoint_interval;
    const auto rank          = execution.profile->local.rank;
    const auto& image        = execution.persistent.state.images;
    const auto capacity      = execution.capacity.device_bytes + execution.host_context_bytes;
    const auto states = count(capacity / image.image_bytes + execution.shape.max_concurrency * 2);
    const auto histories = count(4ULL * states + 3ULL * execution.shape.max_concurrency + 2);
    plan.context.ranks.push_back(
        {rank, execution.capacity.device_bytes, execution.host_context_bytes});
    plan.context.state = {states, {{0, {{rank, image.image_bytes, image.alignment}}}}};
    const auto add     = [&](const QsaCacheLayout& cache) {
        plan.context.pools.push_back({
                .pool_id = count(plan.context.pools.size()),
                .logical_pages =
                count(capacity / cache.pages.page_bytes + execution.shape.max_concurrency * 2),
                .histories     = histories,
                .page_capacity = cache.tables.spec.logical_page_capacity,
                .table_rows    = static_cast<std::uint32_t>(cache.tables.spec.table_rows),
                .parts         = {{rank, {cache.pages.page_bytes, cache.pages.alignment}}},
        });
    };
    add(execution.persistent.main);
    if (execution.persistent.mtp) add(*execution.persistent.mtp);
    plan.context.metadata_budget_bytes = execution.capacity.metadata_budget_bytes;
    if (execution.workspace.vision)
        plan.vision_max_features = execution.workspace.vision->max_features;
    return plan;
}

std::size_t session_metadata_bytes(const ExecutionPlan& execution) {
    const auto plan    = plan_session(execution);
    const auto records = std::size_t(plan.context.state.logical_capacity) * 8;
    return context::ContextDirectory::metadata_bytes(plan.context) +
           detail::Checkpoints::metadata_bytes(records) +
           context::RecoveryIndex<>::metadata_bytes(records);
}

runtime::ContextMetadataPlan session_metadata_plan(const ExecutionPlan& execution,
                                                   const ContextPoolGeometry& geometry) {
    const auto schema = plan_session(execution).context;
    auto metadata     = context::ContextDirectory::metadata_plan(schema);
    const auto checkpoint_cost =
        detail::Checkpoints::metadata_bytes(8) - detail::Checkpoints::metadata_bytes(0);
    const auto recovery_cost =
        context::RecoveryIndex<>::metadata_bytes(8) - context::RecoveryIndex<>::metadata_bytes(0);
    const auto history_cost = context::HistoryDirectory::metadata_bytes(4, 0, 0) -
                              context::HistoryDirectory::metadata_bytes(0, 0, 0);
    metadata.fixed_bytes +=
        detail::Checkpoints::metadata_bytes(std::size_t(schema.state.logical_capacity) * 8) +
        context::RecoveryIndex<>::metadata_bytes(std::size_t(schema.state.logical_capacity) * 8);
    metadata.bytes_per_record.at(geometry.spec_classes[0].index) +=
        checkpoint_cost + recovery_cost + history_cost * schema.pools.size();
    return metadata;
}

} // namespace ninfer::models::qwen4_exp
