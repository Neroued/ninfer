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
    const auto histories =
        count(std::uint64_t(states) + 3ULL * execution.shape.max_concurrency + 2);
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
    if (plan.context.metadata_budget_bytes)
        for (auto& pool : plan.context.pools)
            pool.histories = context::HistoryDirectory::capacity_for_metadata(
                plan.context.metadata_budget_bytes);
    if (execution.workspace.vision)
        plan.vision_max_features = execution.workspace.vision->max_features;
    return plan;
}

runtime::ContextMetadataPlan session_metadata_plan(const ExecutionPlan& execution,
                                                   const ContextPoolGeometry& geometry) {
    auto schema           = plan_session(execution).context;
    const auto rank       = execution.profile->local.rank;
    const auto ranks      = execution.profile->local.size;
    const auto local_rank = schema.ranks.front();
    const auto state_part = schema.state.layouts.front().parts.front();
    schema.ranks.clear();
    schema.state.layouts.front().parts.clear();
    for (auto& pool : schema.pools) {
        const auto part = pool.parts.front();
        pool.parts.clear();
        for (std::uint32_t i = 0; i < ranks; ++i) pool.parts.push_back({i, part.record});
    }
    for (std::uint32_t i = 0; i < ranks; ++i) {
        schema.ranks.push_back({i, 0, i == rank ? local_rank.host_bytes : 0});
        schema.state.layouts.front().parts.push_back(
            {i, state_part.image_bytes, state_part.alignment});
    }
    auto metadata = context::ContextDirectory::metadata_plan(schema, rank);
    const auto checkpoint_cost =
        detail::Checkpoints::metadata_bytes(1) - detail::Checkpoints::metadata_bytes(0);
    const auto recovery_cost =
        context::RecoveryIndex<>::metadata_bytes(1) - context::RecoveryIndex<>::metadata_bytes(0);
    std::size_t history_cost = 0;
    for (const auto& pool : schema.pools)
        history_cost += context::HistoryDirectory::metadata_bytes(1, 0, pool.page_capacity, ranks) -
                        context::HistoryDirectory::metadata_bytes(0, 0, pool.page_capacity, ranks);
    if (rank == 0)
        metadata.fixed_bytes +=
            TokenCountBuilder::capacity_bytes(execution.shape.purpose == EnginePurpose::Generation
                                                  ? execution.profile->public_tokens
                                                  : 0) +
            detail::Checkpoints::metadata_bytes(std::size_t(schema.state.logical_capacity)) +
            context::RecoveryIndex<>::metadata_bytes(std::size_t(schema.state.logical_capacity));
    metadata.bytes_per_record.at(geometry.spec_classes[0].index) +=
        (checkpoint_cost + recovery_cost + history_cost + ranks - 1) / ranks;
    return metadata;
}

} // namespace ninfer::models::qwen4_exp
