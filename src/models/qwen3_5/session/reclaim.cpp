#include "models/qwen3_5/session/internal.h"

#include "models/common/context/physical_reclaim.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <unordered_set>

namespace ninfer::models::qwen3_5::detail {
namespace {

using PhysicalFacts = context::PhysicalReclaim;

PhysicalFacts physical_facts(const SessionImpl& session, std::span<const CheckpointHandle> allowed,
                             std::span<const CheckpointHandle> excluded) {
    const std::array pools{
        context::ReclaimPool{session.text_kv_histories, session.text_kv_pages},
        context::ReclaimPool{session.backend_kv_histories, session.backend_kv_pages}};
    return PhysicalFacts(session.checkpoints, session.directory,
                         std::span(pools).first(session.directory.pool_count()), allowed, excluded);
}

struct StateDemotion {
    context::StateId handle;
    std::uint64_t content_epoch = 0;
    bool host_resident          = false;
};

struct PageDemotion {
    context::PageId handle;
    std::uint32_t position          = 0;
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;
    bool host_covers_prefix         = false;
};

} // namespace

struct DemotionPlan {
    const SessionImpl* owner = nullptr;
    runtime::ContextTransferResource resource;
    std::vector<CheckpointHandle> allowed;
    std::vector<CheckpointHandle> excluded;
    std::optional<StateDemotion> state;
    std::vector<PageDemotion> pages;
};

struct ReclaimPlan {
    ReclaimPlan(const SessionImpl& owner, std::span<const CheckpointHandle> allowed,
                std::span<const CheckpointHandle> excluded, ContextDemand demand)
        : owner(&owner), demand(std::move(demand)),
          facts(physical_facts(owner, allowed, excluded)) {}

    const SessionImpl* owner;
    ContextDemand demand;
    PhysicalFacts facts;
};

struct ReleasePlan {
    std::shared_ptr<const ReclaimPlan> evaluation;
    std::vector<CheckpointHandle> sources;
    ContextReleaseSet released;
};

std::uint32_t SessionImpl::checkpoint_recovery_frontier(CheckpointHandle retained,
                                                        const RequestBasePlan& base,
                                                        std::uint32_t target) const {
    if (!valid_checkpoint(retained) || checkpoint(retained).frontier() > target ||
        !checkpoint_matches(retained, base))
        return 0;
    return checkpoint(retained).frontier();
}

std::optional<ContextReleaseSet>
SessionImpl::checkpoint_release_set(std::span<const CheckpointHandle> handles) const {
    struct ImageOwners {
        context::StateId image;
        std::uint32_t references = 0;
    };

    std::vector<ImageOwners> images;
    std::vector<const context::HistoryPrefix*> main, backend;
    std::unordered_set<std::uint32_t> seen;
    for (const auto handle : handles) {
        if (!can_release_checkpoint(handle)) return {};
        if (!seen.insert(handle.index).second) continue;
        const auto& record = checkpoint(handle);
        const auto image   = record.continuation().image;
        const auto found   = std::find_if(images.begin(), images.end(),
                                          [&](const auto& entry) { return entry.image == image; });
        if (found == images.end())
            images.push_back({image, 1});
        else
            ++found->references;
        main.push_back(&record.text_prefix());
        if (const auto* prefix = record.backend_prefix()) backend.push_back(prefix);
    }
    ContextReleaseSet result;
    for (const auto handle : handles)
        if (std::none_of(result.catalog.begin(), result.catalog.end(),
                         [&](const auto& item) { return item == checkpoints.retirement(handle); }))
            result.catalog.push_back(checkpoints.retirement(handle));
    result.histories = text_kv_histories->released_histories(main);
    if (backend_kv_histories) {
        const auto retired = backend_kv_histories->released_histories(backend);
        result.histories.insert(result.histories.end(), retired.begin(), retired.end());
    }
    const auto append = [&](ContextReleaseSet records) { result.merge(records); };
    for (const auto& image : images)
        if (image.references == state_store->checkpoint_references(image.image) &&
            state_store->can_release_after_checkpoint_references(image.image, image.references))
            append(directory.state_release(image.image));
    for (const auto page : text_kv_histories->released_pages(main))
        append(directory.page_release(page));
    if (backend_kv_histories)
        for (const auto page : backend_kv_histories->released_pages(backend))
            append(directory.page_release(page));
    return result;
}

namespace {

void append_demotions(const SessionImpl& session, const PhysicalFacts::Group& group,
                      std::span<const CheckpointHandle> excluded, const ContextDemand& demand,
                      std::vector<ContextDemotion>& output) {
    if (!session.host_context_enabled || demand.memory != context::ContextMemory::Device) return;
    const auto make_plan = [&] {
        auto plan     = std::make_shared<DemotionPlan>();
        plan->owner   = &session;
        plan->allowed = group.sources;
        plan->excluded.assign(excluded.begin(), excluded.end());
        return plan;
    };
    for (const auto image : group.states) {
        const auto& states = *session.state_store;
        if (states.role(image) != context::StateRole::CheckpointImmutable ||
            !states.can_release_after_checkpoint_references(image,
                                                            states.checkpoint_references(image)))
            continue;
        auto plan       = make_plan();
        const bool host = states.host_resident(image);
        plan->state     = StateDemotion{image, states.content_epoch(image), host};
        output.push_back(
            {group.sources, host ? 0 : states.image_bytes(),
             session.directory.state_release(image).replicas(context::ContextMemory::Device),
             std::move(plan)});
    }
    for (std::uint32_t pool = 0; pool < group.pages.size(); ++pool) {
        const auto& pages   = session.directory.pool(pool).pages();
        auto plan           = make_plan();
        plan->resource.pool = runtime::KvPoolId{static_cast<std::uint16_t>(pool)};
        ContextDemotion candidate;
        candidate.sources = group.sources;
        std::vector<std::size_t> record_ends;
        for (const auto& entry : group.pages[pool]) {
            const auto page = entry.handle;
            if (!pages.device_resident(page) || pages.active_history_references(page) ||
                pages.source_pins(page) || !pages.can_pin_source(page))
                continue;
            const bool host = pages.host_replica_current(page);
            plan->pages.push_back({page, entry.position, pages.content_epoch(page),
                                   pages.committed_columns(page), host});
            const auto records =
                session.directory.page_release(page).replicas(context::ContextMemory::Device);
            candidate.released.records.insert(candidate.released.records.end(),
                                              records.records.begin(), records.records.end());
            record_ends.push_back(candidate.released.records.size());
            if (!host) candidate.host_bytes += pages.layout().parts.front().record.bytes;
        }
        if (!plan->pages.empty()) {
            if (session.directory.assess(demand, candidate.released).feasible()) {
                std::size_t low = 1, high = plan->pages.size();
                while (low < high) {
                    const auto middle = low + (high - low) / 2;
                    ContextReleaseSet prefix;
                    prefix.records.assign(candidate.released.records.begin(),
                                          candidate.released.records.begin() +
                                              record_ends[middle - 1]);
                    if (session.directory.assess(demand, prefix).feasible())
                        high = middle;
                    else
                        low = middle + 1;
                }
                plan->pages.resize(low);
                candidate.released.records.resize(record_ends[low - 1]);
                candidate.host_bytes =
                    std::count_if(plan->pages.begin(), plan->pages.end(),
                                  [](const auto& page) { return !page.host_covers_prefix; }) *
                    pages.layout().parts.front().record.bytes;
            }
            candidate.impl = std::move(plan);
            output.push_back(std::move(candidate));
        }
    }
}

std::optional<ContextRelease>
quote_release(const std::shared_ptr<ReclaimPlan>& evaluation,
              std::span<const CheckpointHandle> sources,
              context::ReleaseInventory<ContextReleaseSet>::Selection& selection) {
    auto probe = selection.probe(sources);
    if (selection.released().empty()) return {};
    auto plan        = std::make_shared<ReleasePlan>();
    plan->evaluation = evaluation;
    plan->sources.assign(sources.begin(), sources.end());
    plan->released = selection.released();
    return ContextRelease{plan->sources, plan->released, std::move(plan)};
}

} // namespace

ContextReclaimPlan SessionImpl::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                             std::span<const CheckpointHandle> excluded,
                                             const ContextDemand& demand) const {
    ContextReclaimPlan result;
    result.impl_     = std::make_shared<ReclaimPlan>(*this, allowed, excluded, demand);
    result.inventory = result.impl_->facts.inventory;
    if (has_context_operation()) return result;
    auto selection = result.inventory->select();
    for (const auto& group : result.impl_->facts.groups) {
        append_demotions(*this, group, excluded, demand, result.demotions);
        if (auto quote = quote_release(result.impl_, group.sources, selection))
            result.releases.push_back(std::move(*quote));
    }
    return result;
}

context::ReleasePlan<ContextRelease, ContextReleaseSet>
SessionImpl::plan_releases(std::span<const CheckpointHandle> allowed,
                           std::span<const CheckpointHandle> excluded,
                           const ContextDemand& demand) const {
    auto evaluation = std::make_shared<ReclaimPlan>(*this, allowed, excluded, demand);
    context::ReleasePlan<ContextRelease, ContextReleaseSet> result{{}, evaluation->facts.inventory};
    if (has_context_operation()) return result;
    auto selection = result.inventory->select();
    for (const auto& group : evaluation->facts.groups)
        if (auto quote = quote_release(evaluation, group.sources, selection))
            result.releases.push_back(std::move(*quote));
    return result;
}

context::HostWriteRequirements
SessionImpl::demotion_host_requirements(const ContextDemotion& quote) const {
    if (!quote.impl || quote.impl->owner != this)
        throw std::invalid_argument("demotion quote belongs to another session");
    const auto& plan = *quote.impl;
    context::HostWriteRequirements write;
    write.kv_pages.resize(directory.pool_count());
    if (plan.state)
        write.state_images = !plan.state->host_resident;
    else
        for (const auto& page : plan.pages)
            if (!page.host_covers_prefix) ++write.kv_pages[plan.resource.pool->index];
    return write;
}

runtime::ContextStart SessionImpl::start_demote(const ContextDemotion& quote,
                                                std::span<const CheckpointHandle> host_victims) {
    if (!quote.impl || quote.impl->owner != this || has_context_operation())
        return runtime::ContextStart::Stale;
    const auto& plan = *quote.impl;
    if (quote.sources != plan.allowed) return runtime::ContextStart::Stale;
    for (const auto source : plan.allowed)
        if (!can_release_checkpoint(source) ||
            std::find(plan.excluded.begin(), plan.excluded.end(), source) != plan.excluded.end())
            return runtime::ContextStart::Stale;
    const auto facts = physical_facts(*this, plan.allowed, plan.excluded);
    const auto group =
        std::find_if(facts.groups.begin(), facts.groups.end(),
                     [&](const auto& value) { return value.sources == plan.allowed; });
    if (group == facts.groups.end()) return runtime::ContextStart::Stale;
    ContextReleaseSet actual;
    std::size_t host_bytes = 0;
    if (plan.state) {
        const auto& expected = *plan.state;
        if (std::find(group->states.begin(), group->states.end(), expected.handle) ==
                group->states.end() ||
            state_store->role(expected.handle) != context::StateRole::CheckpointImmutable ||
            !state_store->device_resident(expected.handle) ||
            state_store->content_epoch(expected.handle) != expected.content_epoch ||
            state_store->host_resident(expected.handle) != expected.host_resident ||
            !state_store->can_release_after_checkpoint_references(
                expected.handle, state_store->checkpoint_references(expected.handle)))
            return runtime::ContextStart::Stale;
        actual = directory.state_release(expected.handle).replicas(context::ContextMemory::Device);
        host_bytes = expected.host_resident ? 0 : state_store->image_bytes();
    } else {
        const auto pool   = plan.resource.pool->index;
        const auto& pages = directory.pool(pool).pages();
        for (const auto& expected : plan.pages) {
            if (std::none_of(group->pages.at(pool).begin(), group->pages.at(pool).end(),
                             [&](const auto& value) { return value.handle == expected.handle; }) ||
                !pages.device_resident(expected.handle) ||
                pages.active_history_references(expected.handle) ||
                pages.source_pins(expected.handle) || !pages.can_pin_source(expected.handle) ||
                pages.content_epoch(expected.handle) != expected.content_epoch ||
                pages.committed_columns(expected.handle) != expected.committed_columns ||
                pages.host_replica_current(expected.handle) != expected.host_covers_prefix)
                return runtime::ContextStart::Stale;
            actual.merge(
                directory.page_release(expected.handle).replicas(context::ContextMemory::Device));
            if (!expected.host_covers_prefix)
                host_bytes += pages.layout().parts.front().record.bytes;
        }
    }
    if (actual != quote.released || host_bytes != quote.host_bytes)
        return runtime::ContextStart::Stale;
    for (const auto victim : host_victims)
        if (std::find(plan.allowed.begin(), plan.allowed.end(), victim) != plan.allowed.end())
            return runtime::ContextStart::Stale;
    const auto released = checkpoint_release_set(host_victims);
    if (!released) return runtime::ContextStart::Stale;
    const auto write     = demotion_host_requirements(quote);
    const auto placement = directory.plan_host(write, *released);
    if (!placement.feasible()) return runtime::ContextStart::Stale;
    if (placement.requires_maintenance() && (execution || !events.empty()))
        return runtime::ContextStart::Deferred;
    auto frame = executor().try_reserve(runtime::FrameKind::Context);
    if (!frame) return runtime::ContextStart::Deferred;
    for (const auto victim : host_victims)
        if (!release_checkpoint(victim))
            throw std::logic_error("prepared Host victim changed before demotion");
    auto reserved = directory.reserve(directory.host_demand(write));
    if (!reserved) throw std::logic_error("prepared Host placement changed before demotion");
    if (!reserved.reservation->ready()) {
        return defer_demotion(quote, std::move(*reserved.reservation), reserved.assessment,
                              std::move(*frame));
    }
    return submit_demotion(quote, std::move(*reserved.reservation), std::move(*frame));
}

runtime::ContextStart SessionImpl::defer_demotion(const ContextDemotion& quote,
                                                  context::ContextReservation targets,
                                                  const ContextAssessment& assessment,
                                                  runtime::ExecutionReservation frame) {
    auto maintenance   = directory.prepare_maintenance(std::move(targets), assessment);
    context_operation_ = std::make_unique<PendingContext>(std::move(maintenance),
                                                          MaintenanceOperation{.demotion = quote});
    context_operation_->reservation.emplace(std::move(frame));
    try {
        (void)submit_context(*context_operation_, ContextWork{});
    } catch (...) {
        abort_context();
        throw;
    }
    return runtime::ContextStart::Started;
}

runtime::ContextStart SessionImpl::submit_demotion(const ContextDemotion& quote,
                                                   context::ContextReservation targets,
                                                   runtime::ExecutionReservation frame) {
    const auto& plan = *quote.impl;
    auto transaction = std::make_unique<PendingContext>(directory, DemoteOperation{});
    transaction->reservation.emplace(std::move(frame));
    auto& migration = transaction->transaction;
    if (plan.state) {
        if (plan.state->host_resident) {
            if (!state_store->drop_device_replica(plan.state->handle))
                throw std::logic_error("prepared State replica could not be released");
            return runtime::ContextStart::Started;
        }
        auto transfer =
            state_store->reserve_device_to_host(plan.state->handle, targets.take_state());
        if (!transfer) throw std::logic_error("prepared State demotion lost its destination");
        migration.add_state_copy(std::move(*transfer), false);
    } else {
        const auto pool = plan.resource.pool->index;
        auto& pages     = directory.pool(pool).pages();
        std::vector<context::PageId> missing;
        for (auto entry = plan.pages.rbegin(); entry != plan.pages.rend(); ++entry)
            if (!entry->host_covers_prefix) missing.push_back(entry->handle);
        migration.add_backups(pool, missing, targets.take_history(pool), true);
        for (const auto& page : plan.pages)
            if (page.host_covers_prefix && !pages.drop_device_replica(page.handle))
                throw std::logic_error("quoted redundant Device page changed during demotion");
        if (!migration.spill_page_groups()) return runtime::ContextStart::Started;
    }
    context_operation_ = std::move(transaction);
    try {
        (void)submit_context(*context_operation_, ContextWork{});
    } catch (...) {
        abort_context();
        throw;
    }
    return runtime::ContextStart::Started;
}

} // namespace ninfer::models::qwen3_5::detail
