#include "models/qwen3_5/planning.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/common/context/storage.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {

SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlan::SequencePlan(SequencePlan&&) noexcept = default;

SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;

SequencePlan::~SequencePlan() = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->capacity : 0;
}

std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->kv_capacity : 0;
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

const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const noexcept {
    static const runtime::SequenceCapacityCurve empty;
    return impl_ != nullptr ? impl_->curve : empty;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return SequencePlan(detail::finalize_sequence_plan_impl(std::move(impl_), main_page_groups));
}

SequencePlanner make_sequence_planner(const execution::Parameters& parameters,
                                      DeviceContext& device, const EngineOptions& options) {
    return make_sequence_planner(parameters.profile, device, options);
}

SequencePlanner make_sequence_planner(std::shared_ptr<const execution::ModelProfile> profile,
                                      DeviceContext& device, const EngineOptions& options,
                                      std::shared_ptr<const moe::Plan> moe) {
    return SequencePlanner(
        detail::make_sequence_planner_impl(std::move(profile), device, options, std::move(moe)));
}

SessionPlan plan_session(const SequencePlan& sequence) {
    if (!sequence.impl_) throw std::invalid_argument("sequence plan is empty");
    const auto& source = *sequence.impl_;
    const auto checked = [](std::uint64_t n) {
        if (n > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("context directory capacity exceeds uint32");
        return static_cast<std::uint32_t>(n);
    };
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
    out.vision_workspace    = source.workspace.vision;
    const auto& image       = source.persistent.state_images;
    out.state_transfer      = image.images.transfer_work();
    if (image.dflash_local) {
        out.dflash_local_group    = image.dflash_local->group;
        out.dflash_local_transfer = TransferWork{image.dflash_local->group.bytes, 1};
    }
    const auto host           = source.context_cache.host_capacity_bytes.value_or(0);
    const auto logical_states = checked(image.images.slot_count + host / image.images.image_stride);
    out.context.state = {logical_states, static_cast<std::uint32_t>(image.images.slot_count),
                         image.images.image_stride};
    std::size_t minimum_stride    = image.images.image_stride;
    std::size_t minimum_kv_stride = std::numeric_limits<std::size_t>::max();
    const auto histories = checked(4ULL * logical_states + 3ULL * source.max_concurrency + 2);
    const auto add_pool  = [&](runtime::KvPoolId id, std::string_view label,
                              const PagedKVCacheLayout& layout) {
        const auto host_layout = plan_host_kv_page_layout(layout.pages.spec.geometry);
        const auto stride      = host_layout.page_stride;
        minimum_stride         = std::min(minimum_stride, stride);
        minimum_kv_stride      = std::min(minimum_kv_stride, stride);
        context::KVDirectorySpec spec;
        spec.pool_id       = id.index;
        spec.device_pages  = layout.pages.spec.page_group_count;
        spec.logical_pages = checked(spec.device_pages + host / stride);
        spec.histories     = histories;
        spec.page_capacity = layout.execution_tables.spec.logical_page_capacity;
        spec.table_rows    = layout.execution_tables.spec.table_rows;
        spec.geometry      = layout.pages.spec.geometry;
        out.kv_pools.push_back({id, label, nullptr, spec.device_pages, stride});
        out.context.pools.push_back(std::move(spec));
    };
    add_pool(kTextKvPool, "text", source.persistent.decoder.text_kv);
    if (source.persistent.decoder.mtp_kv)
        add_pool(kBackendKvPool, "mtp", *source.persistent.decoder.mtp_kv);
    else if (source.persistent.dflash && source.persistent.dflash->full)
        add_pool(kBackendKvPool, "draft_full", *source.persistent.dflash->full);
    out.context.host = {0, host, minimum_stride, checked(host / minimum_kv_stride)};
    return out;
}

} // namespace ninfer::models::qwen3_5
