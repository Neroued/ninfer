#include "models/qwen3_5/planning.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/common/context/storage.h"
#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {
namespace {

std::uint32_t directory_count(std::uint64_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("Qwen3.5 Context metadata capacity exceeds uint32");
    return static_cast<std::uint32_t>(count);
}

context::ContextDirectorySpec context_spec(const detail::SequencePlanImpl& plan) {
    context::ContextDirectorySpec result;
    const auto& resources = plan.persistent;
    const auto host       = resources.host_context_bytes;
    result.ranks.push_back({0, plan.context_bytes, host});
    const auto records  = resources.context_records();
    const auto geometry = describe_context_pool(records);
    const auto blocks   = plan.context_bytes / geometry.block_bytes + host / geometry.block_bytes;
    const auto maximum_records = [&](std::size_t layout) {
        return blocks * geometry.classes[geometry.spec_classes[layout].index].records_per_block;
    };
    const auto& image = resources.state_images.images;
    result.state.logical_capacity =
        directory_count(maximum_records(0) + 2ULL * plan.max_concurrency);
    result.state.layouts.push_back({0, {{0, image.image_bytes, image.alignment}}});
    const auto histories = directory_count(std::uint64_t(result.state.logical_capacity) +
                                           3ULL * plan.max_concurrency + 2);
    const auto add_pool  = [&](std::uint32_t id, const PagedKVCacheLayout& layout) {
        context::KVDirectorySpec spec;
        spec.pool_id       = id;
        spec.logical_pages = directory_count(maximum_records(id + 1));
        spec.histories     = histories;
        spec.page_capacity = layout.execution_tables.spec.logical_page_capacity;
        spec.table_rows    = layout.execution_tables.spec.table_rows;
        spec.parts.push_back({0, {layout.pages.page_bytes, layout.pages.alignment}});
        result.pools.push_back(std::move(spec));
    };
    add_pool(0, resources.decoder.text_kv);
    if (resources.decoder.mtp_kv)
        add_pool(1, *resources.decoder.mtp_kv);
    else if (resources.dflash && resources.dflash->full)
        add_pool(1, *resources.dflash->full);
    result.metadata_budget_bytes = plan.context_metadata_bytes;
    if (result.metadata_budget_bytes)
        for (auto& pool : result.pools)
            pool.histories =
                context::HistoryDirectory::capacity_for_metadata(result.metadata_budget_bytes);
    return result;
}

} // namespace

SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlan::SequencePlan(SequencePlan&&) noexcept = default;

SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;

SequencePlan::~SequencePlan() = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

std::size_t SequencePlan::context_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->context_bytes : 0;
}

std::uint32_t SequencePlan::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->max_concurrency : 0;
}

std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->device_reservation_bytes : 0;
}

std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->workspace.capacity : 0;
}

std::size_t SequencePlan::host_capacity_bytes() const noexcept {
    return impl_ ? impl_->context_cache.host_capacity_bytes.value_or(0) : 0;
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept = default;

SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;

SequencePlanner::~SequencePlanner() = default;

const runtime::ContextCapacityPlan& SequencePlanner::capacity_plan() const noexcept {
    static const runtime::ContextCapacityPlan empty;
    return impl_ != nullptr ? impl_->capacity_plan : empty;
}

SequencePlan SequencePlanner::finalize(runtime::ContextCapacity capacity) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan(detail::finalize_sequence_plan_impl(std::move(impl_), capacity));
}

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options) {
    return make_sequence_planner(parameters.profile, device, options);
}

SequencePlanner make_sequence_planner(std::shared_ptr<const execution::ModelProfile> profile,
                                      DeviceContext& device, const EngineOptions& options,
                                      std::shared_ptr<const moe::Plan> moe) {
    auto impl =
        detail::make_sequence_planner_impl(std::move(profile), device, options, std::move(moe));
    auto spec                    = context_spec(*impl->minimum);
    impl->capacity_plan.metadata = context::ContextDirectory::metadata_plan(spec);
    auto& metadata               = impl->capacity_plan.metadata;
    const auto checkpoints =
        std::size_t(spec.state.logical_capacity) + 2ULL * options.max_concurrency;
    metadata.fixed_bytes +=
        TokenCountBuilder::capacity_bytes(
            impl->minimum->causal_scoring ? 0 : impl->minimum->profile->public_tokens) +
        detail::Checkpoints::metadata_bytes(checkpoints) +
        context::RecoveryIndex<>::metadata_bytes(checkpoints);
    const auto geometry    = describe_context_pool(impl->capacity_plan.records);
    const auto state_class = geometry.spec_classes.front().index;
    metadata.bytes_per_record[state_class] +=
        detail::Checkpoints::metadata_bytes(1) - detail::Checkpoints::metadata_bytes(0) +
        context::RecoveryIndex<>::metadata_bytes(1) - context::RecoveryIndex<>::metadata_bytes(0);
    for (const auto& pool : spec.pools)
        metadata.bytes_per_record[state_class] +=
            context::HistoryDirectory::metadata_bytes(1, 0, pool.page_capacity) -
            context::HistoryDirectory::metadata_bytes(0, 0, pool.page_capacity);
    return SequencePlanner(std::move(impl));
}

SessionPlan plan_session(const SequencePlan& sequence) {
    if (!sequence.impl_) throw std::invalid_argument("sequence plan is empty");
    const auto& source = *sequence.impl_;
    SessionPlan out;
    out.profile             = source.profile;
    out.capacity            = source.capacity;
    out.max_concurrency     = source.max_concurrency;
    out.prefill_chunk       = source.prefill_chunk;
    out.draft_window        = source.draft_window;
    out.speculative_backend = source.speculative_backend;
    out.context_cache       = source.context_cache;
    out.kv_storage          = source.kv_storage;
    out.proposal_head       = source.proposal_head;
    out.causal_scoring      = source.causal_scoring;
    out.vision_max_features = source.workspace.vision ? source.workspace.vision->max_features : 0;
    const auto& image       = source.persistent.state_images;
    out.state_transfer      = image.images.transfer_work();
    if (image.dflash_local) {
        out.dflash_local_group    = image.dflash_local->group;
        out.dflash_local_transfer = TransferWork{image.dflash_local->group.bytes, 1};
    }
    out.context = context_spec(source);
    out.kv_pools.push_back({kTextKvPool, "text"});
    if (source.persistent.decoder.mtp_kv)
        out.kv_pools.push_back({kBackendKvPool, "mtp"});
    else if (source.persistent.dflash && source.persistent.dflash->full)
        out.kv_pools.push_back({kBackendKvPool, "draft_full"});
    return out;
}

} // namespace ninfer::models::qwen3_5
