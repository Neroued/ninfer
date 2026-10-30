#include "models/qwen4_exp/session/internal.h"

#include "models/common/context/physical_reclaim.h"
#include "models/common/qwen/recovery_store.h"

#include <algorithm>
#include <unordered_set>

namespace ninfer::models::qwen4_exp::detail {
namespace {

template <class T>
void append_unique(std::vector<T>& values, const T& value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) values.push_back(value);
}

bool has_host(const SessionImpl& session) {
    const auto ranks = session.directory.records().ranks();
    return std::any_of(ranks.begin(), ranks.end(),
                       [](const auto& rank) { return rank.host_bytes; });
}

context::PhysicalReclaim physical_facts(const SessionImpl& session,
                                        std::span<const CheckpointHandle> allowed,
                                        std::span<const CheckpointHandle> excluded = {}) {
    std::vector<context::ReclaimPool> pools;
    for (std::uint16_t i = 0; i < session.directory.pool_count(); ++i) {
        const auto& pool = session.directory.pool(i);
        pools.push_back({&pool.histories(), &pool.pages()});
    }
    return {session.checkpoints, session.directory, pools, allowed, excluded};
}

void seal(Demotion& plan) {
    const auto& session = *plan.owner;
    for (const auto image : plan.images) {
        plan.image_epochs.push_back(session.directory.states().content_epoch(image));
    }
    for (std::uint16_t pool = 0; pool < session.directory.pool_count(); ++pool) {
        const auto& pages = session.directory.pool(pool).pages();
        for (const auto page : plan.pages[pool]) {
            plan.page_versions[pool].push_back(
                {pages.content_epoch(page), pages.committed_columns(page)});
        }
    }
}

bool valid_demotion(const SessionImpl& session, const Demotion& plan) {
    for (const auto handle : plan.records) {
        if (!session.checkpoints.erasable(handle)) return false;
    }
    const auto facts = physical_facts(session, plan.records);
    std::unordered_set<std::uint32_t> images;
    std::array<std::unordered_set<std::uint32_t>, 2> pages;
    for (const auto& group : facts.groups) {
        for (const auto image : group.states) images.insert(image.index);
        for (std::uint16_t pool = 0; pool < session.directory.pool_count(); ++pool) {
            for (const auto& page : group.pages[pool]) pages[pool].insert(page.handle.index);
        }
    }
    const auto& states = session.directory.states();
    if (plan.images.size() != plan.image_epochs.size()) return false;
    for (std::size_t i = 0; i < plan.images.size(); ++i) {
        const auto image = plan.images[i];
        if (!states.valid(image) || !images.contains(image.index) ||
            !states.device_resident(image) || states.content_epoch(image) != plan.image_epochs[i] ||
            states.role(image) != context::StateRole::CheckpointImmutable ||
            !states.can_release_after_checkpoint_references(image,
                                                            states.checkpoint_references(image))) {
            return false;
        }
    }
    for (std::uint16_t pool = 0; pool < session.directory.pool_count(); ++pool) {
        const auto& store = session.directory.pool(pool).pages();
        if (plan.pages[pool].size() != plan.page_versions[pool].size()) return false;
        for (std::size_t i = 0; i < plan.pages[pool].size(); ++i) {
            const auto page    = plan.pages[pool][i];
            const auto version = plan.page_versions[pool][i];
            if (!store.valid(page) || !pages[pool].contains(page.index) ||
                !store.device_resident(page) || !store.can_pin_source(page) ||
                store.source_pins(page) || store.active_history_references(page) ||
                store.content_epoch(page) != version.epoch ||
                store.committed_columns(page) != version.columns) {
                return false;
            }
        }
    }
    return true;
}

std::shared_ptr<Demotion> demotion_for_group(const SessionImpl& session,
                                             const context::PhysicalReclaim::Group& group,
                                             const ContextDemand& demand) {
    struct Candidate {
        Image image;
        context::PageId page;
        ContextReleaseSet released;
        std::size_t copy_bytes = 0;
    };

    std::vector<Candidate> candidates;
    const auto& states = session.directory.states();
    for (const auto image : group.states) {
        if (states.role(image) != context::StateRole::CheckpointImmutable ||
            !states.can_release_after_checkpoint_references(image,
                                                            states.checkpoint_references(image))) {
            continue;
        }
        auto released =
            session.directory.state_release(image).replicas(context::ContextMemory::Device);
        const auto bytes =
            states.host_resident(image) ? 0 : session.directory.released_bytes(released);
        candidates.push_back({image, {}, std::move(released), bytes});
    }
    for (std::uint16_t pool = 0; pool < session.directory.pool_count(); ++pool) {
        const auto& pages = session.directory.pool(pool).pages();
        for (const auto& item : group.pages[pool]) {
            const auto page = item.handle;
            if (!pages.device_resident(page) || pages.source_pins(page) ||
                pages.active_history_references(page) || !pages.can_pin_source(page)) {
                continue;
            }
            auto released =
                session.directory.page_release(page).replicas(context::ContextMemory::Device);
            const auto bytes =
                pages.host_replica_current(page) ? 0 : session.directory.released_bytes(released);
            candidates.push_back({{}, page, std::move(released), bytes});
        }
    }
    if (candidates.empty()) return {};
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.copy_bytes < right.copy_bytes;
    });
    ContextReleaseSet released;
    std::vector<std::size_t> ends{0};
    for (const auto& candidate : candidates) {
        released.records.insert(released.records.end(), candidate.released.records.begin(),
                                candidate.released.records.end());
        ends.push_back(released.records.size());
    }
    std::size_t count = candidates.size();
    if (session.directory.assess(demand, released).feasible()) {
        std::size_t low = 1, high = count;
        while (low < high) {
            const auto middle = low + (high - low) / 2;
            ContextReleaseSet prefix;
            prefix.records.assign(released.records.begin(),
                                  released.records.begin() + ends[middle]);
            if (session.directory.assess(demand, prefix).feasible())
                high = middle;
            else
                low = middle + 1;
        }
        count = low;
    }
    auto plan     = std::make_shared<Demotion>();
    plan->owner   = &session;
    plan->records = group.sources;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& candidate = candidates[i];
        if (candidate.image.valid())
            plan->images.push_back(candidate.image);
        else
            plan->pages[candidate.page.pool_id].push_back(candidate.page);
    }
    seal(*plan);
    return plan;
}

} // namespace

std::vector<Image> SessionImpl::checkpoint_images(std::span<const CheckpointHandle> records) const {
    std::vector<Image> images;
    std::unordered_set<std::uint32_t> seen;
    for (const auto handle : records) {
        if (!checkpoints.valid(handle)) continue;
        const auto image = checkpoints.at(handle).snapshot.continuation().image;
        if (seen.insert(image.index).second) images.push_back(image);
    }
    return images;
}

std::vector<context::PageId>
SessionImpl::checkpoint_pages(std::span<const CheckpointHandle> records, std::uint32_t pool) const {
    std::vector<context::PageId> pages;
    std::unordered_set<std::uint32_t> seen;
    for (const auto handle : records) {
        if (!checkpoints.valid(handle)) continue;
        const auto& prefix    = checkpoints.at(handle).snapshot.prefix(pool);
        const auto& histories = directory.pool(pool).histories();
        for (std::uint32_t i = 0; i < context::kv_pages_for_frontier(prefix.frontier()); ++i) {
            const auto page = histories.logical_page(prefix.history(), i);
            if (seen.insert(page.index).second) pages.push_back(page);
        }
    }
    return pages;
}

ContextReleaseSet SessionImpl::footprint(std::span<const CheckpointHandle> input,
                                         bool released) const {
    ContextReleaseSet result;
    if (released) {
        if (std::any_of(input.begin(), input.end(),
                        [&](auto handle) { return !checkpoints.erasable(handle); }))
            return result;
        const auto facts = physical_facts(*this, input);
        auto selected    = facts.inventory->select();
        auto probe       = selected.probe(input);
        return selected.released();
    }
    for (const auto image : checkpoint_images(input)) result.merge(directory.state_release(image));
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
        for (const auto page : checkpoint_pages(input, pool)) {
            const auto facts = directory.page_release(page);
            result.records.insert(result.records.end(), facts.records.begin(), facts.records.end());
            result.pages.push_back(page);
        }
    }
    result.normalize();
    return result;
}

std::size_t SessionImpl::host_released(std::span<const CheckpointHandle> records) const {
    return directory.released_bytes(
        footprint(records, true).replicas(context::ContextMemory::Host));
}

std::shared_ptr<Demotion> SessionImpl::demotion(std::span<const CheckpointHandle> records) const {
    if (!has_host(*this)) return {};
    auto plan   = std::make_shared<Demotion>();
    plan->owner = this;
    for (const auto handle : records) {
        if (checkpoints.valid(handle)) append_unique(plan->records, handle);
    }
    for (const auto image : checkpoint_images(plan->records)) {
        if (directory.states().device_resident(image) && !directory.states().source_pins(image)) {
            plan->images.push_back(image);
        }
    }
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
        const auto& pages = directory.pool(pool).pages();
        for (const auto page : checkpoint_pages(plan->records, pool)) {
            if (pages.device_resident(page) && !pages.source_pins(page) &&
                !pages.active_history_references(page))
                plan->pages[pool].push_back(page);
        }
    }
    seal(*plan);
    return plan->empty() ? nullptr : plan;
}

ContextDemotion SessionImpl::describe_demotion(std::shared_ptr<const Demotion> plan) const {
    ContextDemotion result;
    result.sources = plan->records;
    for (const auto image : plan->images) {
        result.released.merge(
            directory.state_release(image).replicas(context::ContextMemory::Device));
    }
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
        for (const auto page : plan->pages[pool]) {
            const auto records =
                directory.page_release(page).replicas(context::ContextMemory::Device);
            result.released.records.insert(result.released.records.end(), records.records.begin(),
                                           records.records.end());
        }
    }
    result.released.normalize();
    result.host_bytes = directory.host_write_bytes(host_requirements(*plan));
    result.impl       = std::move(plan);
    return result;
}

ContextDemotion SessionImpl::combine_demotions(std::span<const ContextDemotion> quotes) const {
    auto plan   = std::make_shared<Demotion>();
    plan->owner = this;
    ContextDemotion result;
    std::unordered_set<std::uint32_t> images;
    std::array<std::unordered_set<std::uint32_t>, 2> pages;
    for (const auto& quote : quotes) {
        if (!quote.impl || quote.impl->owner != this)
            throw std::invalid_argument("demotion quote belongs to another session");
        const auto& part = *quote.impl;
        for (const auto source : part.records) append_unique(plan->records, source);
        for (std::size_t i = 0; i < part.images.size(); ++i) {
            if (!images.insert(part.images[i].index).second)
                throw std::invalid_argument("demotion batch contains overlapping State records");
            plan->images.push_back(part.images[i]);
            plan->image_epochs.push_back(part.image_epochs[i]);
        }
        for (std::uint32_t pool = 0; pool < directory.pool_count(); ++pool)
            for (std::size_t i = 0; i < part.pages[pool].size(); ++i) {
                const auto page = part.pages[pool][i];
                if (!pages[pool].insert(page.index).second)
                    throw std::invalid_argument("demotion batch contains overlapping KV records");
                plan->pages[pool].push_back(page);
                plan->page_versions[pool].push_back(part.page_versions[pool][i]);
            }
        result.host_bytes += quote.host_bytes;
        result.released.merge(quote.released);
    }
    result.sources = plan->records;
    result.impl    = std::move(plan);
    return result;
}

context::HostWriteRequirements SessionImpl::host_requirements(const Demotion& plan) const {
    context::HostWriteRequirements write;
    write.kv_pages.resize(directory.pool_count());
    for (const auto image : plan.images)
        write.state_images += !directory.states().host_resident(image);
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
        for (const auto page : plan.pages[pool]) {
            write.kv_pages[pool] += !directory.pool(pool).pages().host_replica_current(page);
        }
    }
    return write;
}

bool SessionImpl::prepare_demotion(context::ContextTransaction& migration, const Demotion& plan,
                                   context::ContextReservation& targets,
                                   const context::HostDemandIds& target_ids) {
    std::vector<Image> redundant;
    auto state_targets =
        target_ids.state.valid() ? targets.take(target_ids.state) : context::RecordReservation{};
    for (const auto image : plan.images) {
        auto& states = directory.states();
        if (!states.valid(image) || !states.device_resident(image) || states.source_pins(image))
            return false;
        if (states.host_resident(image))
            redundant.push_back(image);
        else {
            auto transfer = states.reserve_device_to_host(image, state_targets.split(1));
            if (!transfer) return false;
            migration.add_state_copy(std::move(*transfer), false);
        }
    }
    std::array<std::vector<context::PageId>, 2> redundant_pages;
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
        auto& store = directory.pool(pool).pages();
        std::vector<context::PageId> pages;
        for (const auto page : plan.pages[pool]) {
            if (!store.valid(page) || !store.device_resident(page) || store.source_pins(page) ||
                store.active_history_references(page))
                return false;
            if (store.host_replica_current(page))
                redundant_pages[pool].push_back(page);
            else
                pages.push_back(page);
        }
        if (!pages.empty())
            migration.add_backups(pool, pages, targets.take(target_ids.histories.at(pool)), true);
    }
    for (const auto image : redundant) {
        if (!directory.states().drop_device_replica(image)) {
            throw std::logic_error("State demotion lost exclusivity");
        }
    }
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool) {
        for (const auto page : redundant_pages[pool]) {
            if (!directory.pool(pool).pages().drop_device_replica(page)) {
                throw std::logic_error("KV demotion lost exclusivity");
            }
        }
    }
    return true;
}

context::ReleasePlan<ContextRelease, ContextReleaseSet>
SessionImpl::plan_releases(std::span<const CheckpointHandle> allowed,
                           std::span<const CheckpointHandle> excluded, const ContextDemand&) const {
    auto facts = physical_facts(*this, allowed, excluded);
    context::ReleasePlan<ContextRelease, ContextReleaseSet> result{{}, facts.inventory};
    if (operation) return result;
    auto selected = facts.inventory->select();
    for (const auto& group : facts.groups) {
        auto probe = selected.probe(group.sources);
        if (!selected.released().empty())
            result.releases.push_back({group.sources, selected.released()});
    }
    return result;
}

ContextReclaimPlan SessionImpl::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                             std::span<const CheckpointHandle> excluded,
                                             const ContextDemand& demand) const {
    auto facts = physical_facts(*this, allowed, excluded);
    ContextReclaimPlan result;
    result.inventory = facts.inventory;
    if (operation) return result;
    const bool needs_device = demand.empty() || demand.uses_memory(context::ContextMemory::Device);
    auto selected           = facts.inventory->select();
    for (const auto& group : facts.groups) {
        auto probe = selected.probe(group.sources);
        if (!selected.released().empty())
            result.releases.push_back({group.sources, selected.released()});
        if (!needs_device || !has_host(*this)) continue;
        if (auto plan = demotion_for_group(*this, group, demand)) {
            result.demotions.push_back(describe_demotion(std::move(plan)));
        }
    }
    return result;
}

runtime::ContextStart SessionImpl::start_demote(const ContextDemotion& quote,
                                                std::span<const CheckpointHandle> victims) {
    if (operation || !quote.impl || quote.impl->owner != this || !has_host(*this) ||
        quote.sources != quote.impl->records || !valid_demotion(*this, *quote.impl)) {
        return runtime::ContextStart::Stale;
    }
    const auto current = describe_demotion(quote.impl);
    if (current.host_bytes != quote.host_bytes || current.released != quote.released) {
        return runtime::ContextStart::Stale;
    }
    for (const auto victim : victims) {
        if (!checkpoints.erasable(victim) ||
            std::find(quote.sources.begin(), quote.sources.end(), victim) != quote.sources.end()) {
            return runtime::ContextStart::Stale;
        }
    }
    ContextDemand demand;
    const auto target_ids = demand.add_host(host_requirements(*quote.impl));
    const auto placement  = directory.assess(demand, footprint(victims, true));
    if (!placement.feasible()) return runtime::ContextStart::Stale;
    if (placement.requires_maintenance() && executor->has_pending_work()) {
        pending_maintenance = placement;
        directory.request_maintenance();
        return runtime::ContextStart::Deferred;
    }
    auto execution = executor->try_reserve(runtime::FrameKind::Context);
    if (!execution) return runtime::ContextStart::Deferred;
    DemoteOperation action{quote.impl, {}, target_ids};
    action.sources.reserve(quote.sources.size());
    for (const auto source : quote.sources) action.sources.push_back(checkpoints.lease(source));
    for (const auto victim : victims) {
        if (!checkpoints.erase(victim))
            throw std::logic_error("Prepared Host victim changed before demotion");
    }
    auto reserved = directory.reserve(demand);
    if (!reserved.reservation)
        throw std::logic_error("Prepared Host placement changed before demotion");
    try {
        if (!reserved.reservation->ready()) {
            auto maintenance = directory.prepare_maintenance(std::move(*reserved.reservation),
                                                             reserved.assessment);
            operation        = std::make_unique<ContextOperation>(
                std::move(maintenance), std::move(action), std::move(*execution));
            operation->preparing_resources = true;
        } else {
            operation = std::make_unique<ContextOperation>(directory, std::move(action),
                                                           std::move(*execution));
            if (!prepare_demotion(operation->migration, *quote.impl, *reserved.reservation,
                                  target_ids)) {
                throw std::logic_error("Prepared demotion lost its destination");
            }
        }
        submit_context();
        return runtime::ContextStart::Started;
    } catch (...) {
        cancel_operation();
        throw;
    }
}

runtime::ContextStart SessionImpl::try_pending_maintenance() {
    if (!pending_maintenance) return runtime::ContextStart::Stale;
    if (operation || compute || executor->has_pending_work())
        return runtime::ContextStart::Deferred;
    auto execution = executor->try_reserve(runtime::FrameKind::Context);
    if (!execution) return runtime::ContextStart::Deferred;
    auto migration = directory.prepare_maintenance(*pending_maintenance);
    if (!migration) {
        pending_maintenance.reset();
        return runtime::ContextStart::Stale;
    }
    operation = std::make_unique<ContextOperation>(std::move(*migration), MaintenanceOperation{},
                                                   std::move(*execution));
    pending_maintenance.reset();
    try {
        submit_context();
    } catch (...) {
        cancel_operation();
        throw;
    }
    return runtime::ContextStart::Started;
}

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {

ContextAssessment Session::assess(const ContextDemand& demand,
                                  const ContextReleaseSet& released) const {
    return impl_->directory.assess(demand, released);
}

bool Session::releases_metadata(const ContextReleaseSet& released,
                                const ContextAssessment& assessment) const {
    return assessment.metadata_contribution(released);
}

std::uint64_t Session::released_bytes(const ContextReleaseSet& released,
                                      const ContextAssessment& assessment) const {
    ContextReleaseSet relevant;
    for (const auto& record : released.records) {
        if (std::any_of(assessment.placement.domains.begin(), assessment.placement.domains.end(),
                        [&](const auto& domain) { return domain.domain == record.domain; })) {
            relevant.records.push_back(record);
        }
    }
    return impl_->directory.released_bytes(relevant);
}

std::uint64_t Session::useful_release_bytes(const ContextReleaseSet& released,
                                            const ContextAssessment& assessment) const {
    return impl_->directory.useful_release_bytes(released, assessment);
}

std::vector<std::uint64_t> Session::release_limits(const ContextAssessment& assessment) const {
    return impl_->directory.release_limits(assessment);
}

runtime::ContextReleaseMeasure Session::release_measure(const ContextReleaseSet& released,
                                                        const ContextAssessment& assessment) const {
    return impl_->directory.release_measure(released, assessment);
}

runtime::ContextStart Session::start_maintenance(const ContextAssessment& assessment) {
    auto& session = *impl_;
    if (session.operation) return runtime::ContextStart::Deferred;
    if (!session.pending_maintenance) {
        const auto current = session.directory.assess(assessment.demand);
        if (!current.requires_maintenance()) return runtime::ContextStart::Stale;
        session.pending_maintenance = current;
        session.directory.request_maintenance();
    }
    return session.try_pending_maintenance();
}

std::size_t Session::host_bytes_released(std::span<const CheckpointHandle> records) const {
    return impl_->host_released(records);
}

context::RecoveryEvaluation<>
Session::evaluate_recovery(std::span<const CheckpointHandle> advertised) const {
    impl_->recovery_index.synchronize(qwen::RecoveryStoreView{impl_->checkpoints});
    return impl_->recovery_index.evaluate(advertised);
}

CheckpointSummary Session::checkpoint_summary(CheckpointHandle handle) const {
    CheckpointSummary result;
    static_cast<CheckpointMetadata&>(result) = checkpoint_metadata(handle);
    result.key                               = checkpoint_key(handle, result.frontier);
    return result;
}

context::ReleasePlan<ContextRelease, ContextReleaseSet>
Session::plan_releases(std::span<const CheckpointHandle> allowed,
                       std::span<const CheckpointHandle> excluded,
                       const ContextDemand& demand) const {
    return impl_->plan_releases(allowed, excluded, demand);
}

ContextReclaimPlan Session::plan_reclaim(std::span<const CheckpointHandle> allowed,
                                         std::span<const CheckpointHandle> excluded,
                                         const ContextDemand& demand) const {
    return impl_->plan_reclaim(allowed, excluded, demand);
}

runtime::ContextStart Session::start_demote(const ContextDemotion& quote,
                                            std::span<const CheckpointHandle> victims) {
    return impl_->start_demote(quote, victims);
}

Session::HostRequirement Session::demotion_host_requirements(const ContextDemotion& quote) const {
    if (!quote.impl || quote.impl->owner != impl_.get()) {
        throw std::invalid_argument("Demotion quote belongs to another Session");
    }
    return impl_->host_requirements(*quote.impl);
}

std::size_t Session::host_write_bytes(const HostRequirement& write) const {
    return impl_->directory.host_write_bytes(write);
}

ContextDemotion Session::combine_demotions(std::span<const ContextDemotion> quotes) const {
    return impl_->combine_demotions(quotes);
}

ContextDemand Session::host_demand(const HostRequirement& write) const {
    return impl_->directory.host_demand(write);
}

bool Session::can_place_host(const HostRequirement& write,
                             const ContextReleaseSet& released) const {
    return impl_->directory.plan_host(write, released).feasible();
}

std::size_t Session::release_redundant_host(std::span<const CheckpointHandle> excluded,
                                            std::optional<SequenceHandle> keep) {
    auto& session = *impl_;
    if (!detail::has_host(session)) return 0;
    const auto protected_images = session.checkpoint_images(excluded);
    std::vector<CheckpointHandle> all;
    for (std::uint32_t i = 0; i < session.checkpoints.size(); ++i) {
        const auto handle = session.checkpoints.handle(i);
        if (session.checkpoints.valid(handle)) all.push_back(handle);
    }
    std::size_t released_bytes = 0;
    for (const auto image : session.checkpoint_images(all)) {
        if (std::find(protected_images.begin(), protected_images.end(), image) !=
                protected_images.end() ||
            (keep && session.valid(*keep) && session.sequence(*keep).continuation.image == image))
            continue;
        const auto records =
            session.directory.state_release(image).replicas(context::ContextMemory::Host);
        const auto bytes = session.directory.released_bytes(records);
        if (session.directory.states().drop_host_replica(image)) released_bytes += bytes;
    }
    for (std::uint16_t pool = 0; pool < session.directory.pool_count(); ++pool) {
        const auto protected_pages = session.checkpoint_pages(excluded, pool);
        auto& pages                = session.directory.pool(pool).pages();
        for (const auto page : session.checkpoint_pages(all, pool)) {
            if (std::find(protected_pages.begin(), protected_pages.end(), page) !=
                protected_pages.end())
                continue;
            const auto records = pages.records(page, context::ContextMemory::Host);
            for (const auto& record : records) {
                const auto size = session.directory.records()
                                      .pool(record.domain)
                                      .resolve(record.allocation)
                                      .bytes;
                if (pages.drop_host_replica(page, record.domain.rank)) released_bytes += size;
            }
        }
    }
    return released_bytes;
}

} // namespace ninfer::models::qwen4_exp
