#include "models/glm5next/session/internal.h"

#include "models/common/context/physical_reclaim.h"

#include <algorithm>
#include <unordered_set>

namespace ninfer::models::glm5next::detail {
namespace {
class RecoveryIdentity final : public runtime::RecoveryIdentity {
public:
    RecoveryIdentity(context::MetadataBudget& budget, const void* owner,
                     std::shared_ptr<const InputIdentity> identity, std::uint32_t frontier,
                     std::uint32_t chunk, std::uint32_t capacity)
        : runtime::RecoveryIdentity(owner, frontier, identity->tag), account_(budget),
          identity_(std::move(identity)), chunk_(chunk), capacity_(capacity) {
        account_.acquire(sizeof(*this));
    }

    std::array<std::uint64_t, 2> digest(std::uint32_t f) const override {
        return identity_->digests.at(f);
    }

    runtime::PrefillWork rebuild(std::uint32_t begin) const override {
        return models::vision::recovery_work(
            identity_->media, [](const auto& media) -> const auto& { return media.item; }, begin,
            frontier, chunk_, capacity_);
    }
private:
    bool exact_prefix(const runtime::RecoveryIdentity& other) const override {
        const auto* value = dynamic_cast<const RecoveryIdentity*>(&other);
        return value && input_prefix_equal(identity_->view(), value->identity_->view(), frontier);
    }

    context::MetadataAccount account_;
    std::shared_ptr<const InputIdentity> identity_;
    std::uint32_t chunk_, capacity_;
};

static_assert(sizeof(RecoveryIdentity) <= runtime::kRecoveryIdentityMetadataLimit);

struct RecoveryStoreView {
    const Checkpoints& store;

    const std::uint64_t& revision() const { return store.revision(); }

    std::size_t capacity() const { return store.metadata_extent(); }

    auto handle(std::uint32_t index) const { return store.handle(index); }

    bool valid(CheckpointHandle handle) const { return store.valid(handle); }

    std::uint32_t frontier(CheckpointHandle handle) const { return store.at(handle).frontier(); }

    context::RecoveryKey key(CheckpointHandle h, std::uint32_t f) const {
        const auto& identity = store.at(h).snapshot.identity();
        return {identity.digests.at(f), identity.tag};
    }

    bool prefix(CheckpointHandle a, CheckpointHandle b) const {
        const auto& left  = store.at(a);
        const auto& right = store.at(b);
        return left.frontier() <= right.frontier() &&
               left.snapshot.identity().tag == right.snapshot.identity().tag &&
               input_prefix_equal(left.snapshot.identity().view(), right.snapshot.identity().view(),
                                  left.frontier());
    }
};

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

runtime::CheckpointEvictableResources
SessionImpl::checkpoint_evictable_resources(CheckpointHandle handle) const {
    runtime::CheckpointEvictableResources out;
    if (!checkpoints.erasable(handle)) return out;
    const std::array handles{handle};
    const auto released = footprint(handles, true);
    out.state_slots = static_cast<std::uint32_t>(released.states.size());
    for (const auto page : released.pages) {
        if (page.pool_id < out.kv_page_groups.size()) ++out.kv_page_groups[page.pool_id];
    }
    // glm5next has no Host context; host_bytes stays zero.
    return out;
}

std::size_t SessionImpl::state_image_bytes() const noexcept {
    return directory.states().image_bytes();
}

std::size_t SessionImpl::kv_page_bytes(std::uint32_t pool) const noexcept {
    return directory.pool(pool).pages().layout().parts.front().record.bytes;
}

std::uint32_t SessionImpl::state_slots_capacity() const noexcept {
    return directory.states().capacity();
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
    auto plan = plan_releases(allowed, excluded, demand);
    return {{}, std::move(plan.releases), std::move(plan.inventory)};
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

} // namespace ninfer::models::glm5next::detail

namespace ninfer::models::glm5next {

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
    if (session.operation || session.pending_maintenance || session.compute ||
        session.executor->has_pending_work())
        return runtime::ContextStart::Deferred;
    const auto current = session.directory.assess(assessment.demand);
    if (!current.requires_maintenance()) return runtime::ContextStart::Stale;
    session.pending_maintenance = current;
    const auto started          = session.try_pending_maintenance();
    // A Deferred public submission belongs to the caller's retry decision. It must not
    // later execute from poll() after that decision has been abandoned or repriced.
    if (started == runtime::ContextStart::Deferred) session.pending_maintenance.reset();
    return started;
}

std::size_t Session::host_bytes_released(std::span<const CheckpointHandle> records) const {
    return 0;
}

runtime::CheckpointEvictableResources
Session::checkpoint_evictable_resources(CheckpointHandle checkpoint) const {
    return impl_->checkpoint_evictable_resources(checkpoint);
}

std::size_t Session::state_image_bytes() const noexcept { return impl_->state_image_bytes(); }

std::size_t Session::kv_page_bytes(std::uint32_t pool) const noexcept {
    return impl_->kv_page_bytes(pool);
}

std::uint32_t Session::state_slots_capacity() const noexcept {
    return impl_->state_slots_capacity();
}

context::RecoveryEvaluation<>
Session::evaluate_recovery(std::span<const CheckpointHandle> advertised) const {
    impl_->recovery_index.synchronize(detail::RecoveryStoreView{impl_->checkpoints});
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
    throw std::invalid_argument("GLM5Next Context has no Host demotion");
}

Session::HostRequirement Session::demotion_host_requirements(const ContextDemotion& quote) const {
    throw std::invalid_argument("GLM5Next Context has no Host demotion");
}

std::size_t Session::host_write_bytes(const HostRequirement& write) const {
    return impl_->directory.host_write_bytes(write);
}

ContextDemotion Session::combine_demotions(std::span<const ContextDemotion> quotes) const {
    if (!quotes.empty()) throw std::invalid_argument("GLM5Next Context has no Host demotion");
    return {};
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
    return 0;
}

} // namespace ninfer::models::glm5next

namespace ninfer::models::glm5next {

context::MetadataBudget& Session::recovery_budget() const {
    return impl_->recovery_metadata.budget();
}

std::vector<Session::RecoveryPoint>
Session::recovery_points(const ContextReleaseSet& removed_device,
                         std::span<const CheckpointHandle> selected) const {
    if (!removed_device.empty()) throw std::invalid_argument("GLM has no Host demotion");
    const auto& p = *impl_;
    std::vector<RecoveryPoint> result;
    for (std::uint32_t i = 0; i < p.checkpoints.size(); ++i) {
        const auto handle = p.checkpoints.handle(i);
        if (!p.checkpoints.valid(handle)) continue;
        if (!selected.empty() &&
            std::find(selected.begin(), selected.end(), handle) == selected.end())
            continue;
        const auto& record = p.checkpoints.at(handle);
        runtime::RecoveryDescription value;
        value.restores.reserve(p.placements.size());
        value.placement = record.snapshot.continuation().metadata.placement;
        value.identity  = record.recovery_identity_cache.lock();
        if (!value.identity) {
            value.identity = std::shared_ptr<detail::RecoveryIdentity>(new detail::RecoveryIdentity(
                recovery_budget(), &p, record.snapshot.identity_owner(), record.frontier(),
                p.shape.prefill_chunk, p.plan.vision_max_features));
            record.recovery_identity_cache = value.identity;
        }
        for (auto target : p.placements) {
            auto transfers =
                target == value.placement
                    ? record.snapshot.restore_requirements(p.directory)
                    : detail::conversion_requirements(p.plan, record.snapshot, target.value);
            if (target == value.placement)
                for (std::uint32_t rank = 0; rank < p.plan.states.size(); ++rank)
                    if (const auto& field = p.plan.states[rank].draft_context) {
                        runtime::ContextTransferRequirement transfer;
                        transfer.direction   = runtime::ContextTransferDirection::DeviceToDevice;
                        transfer.rank        = rank;
                        transfer.image_count = 1;
                        transfer.work        = {field->bytes, 1};
                        transfers.push_back(transfer);
                    }
            auto replay = target == value.placement
                              ? record.snapshot.replay_requirements(p.directory)
                              : std::vector<runtime::ContextTransferRequirement>{};
            value.restores.push_back({target, std::move(transfers), std::move(replay)});
        }
        result.push_back(
            {handle, context::recovery_description(recovery_budget(), std::move(value))});
    }
    return result;
}

std::shared_ptr<const runtime::RecoveryDescription>
Session::recovery_target(const RequestBasePlan& base, std::uint32_t f, runtime::PlacementId target,
                         runtime::CaptureStorage storage) const {
    if (!base.plan_ || !f || f > base.summary().prompt_tokens ||
        target.value >= impl_->plan.states.size() || storage == runtime::CaptureStorage::Host)
        throw std::invalid_argument("invalid GLM recovery target");
    runtime::RecoveryDescription value;
    value.placement = target;
    value.identity  = std::make_shared<detail::RecoveryIdentity>(
        recovery_budget(), impl_.get(), base.plan_->identity, f, impl_->shape.prefill_chunk,
        impl_->plan.vision_max_features);
    value.restores.push_back({target, {}});
    for (std::uint32_t rank = 0; rank < impl_->plan.states.size(); ++rank)
        if (const auto& field = impl_->plan.states[rank].draft_context) {
            runtime::ContextTransferRequirement transfer;
            transfer.direction   = runtime::ContextTransferDirection::DeviceToDevice;
            transfer.rank        = rank;
            transfer.image_count = 1;
            transfer.work        = {field->bytes, 1};
            value.restores.back().transfers.push_back(transfer);
        }
    return context::recovery_description(recovery_budget(), std::move(value));
}

runtime::PauseCostQuote Session::pause_cost(SequenceHandle h, const PauseOption& option,
                                            std::span<const CheckpointHandle> sources,
                                            std::span<const CheckpointHandle> victims) const {
    const auto& p = *impl_;
    const auto& s = p.sequence(h);
    const auto end =
        s.phase == detail::Phase::Replay ? s.replay_target : s.continuation.metadata.main_frontier;
    const auto work = [&](std::uint32_t begin) {
        return models::vision::recovery_work(
            s.identity.media, [](const auto& media) -> const auto& { return media.item; }, begin,
            end, p.shape.prefill_chunk, p.plan.vision_max_features);
    };
    runtime::PauseCostQuote quote;
    quote.work =
        context::maintenance_work(p.directory.assess(option.demand, p.footprint(victims, true)));
    if (option.action.kind == runtime::PauseKind::ReleaseAndReplay) {
        quote.recovery.push_back({work(0), {}});
        for (auto source : sources) {
            if (!p.checkpoints.valid(source)) continue;
            const auto& c  = p.checkpoints.at(source);
            const auto& id = c.snapshot.identity();
            if (c.frontier() > end || id.tag != s.identity.tag ||
                !input_prefix_equal(id.view(), s.identity.view(), c.frontier()))
                continue;
            const auto points = recovery_points({}, {&source, 1});
            for (const auto& restore : points.front().description->restores)
                if (restore.placement == s.continuation.metadata.placement) {
                    auto transfers = restore.transfers;
                    if (c.frontier() < end)
                        transfers.insert(transfers.end(), restore.replay_transfers.begin(),
                                         restore.replay_transfers.end());
                    quote.recovery.push_back({work(c.frontier()), std::move(transfers)});
                }
        }
    } else {
        if (option.action.kind == runtime::PauseKind::Relocate) {
            std::vector<std::uint32_t> frontiers{s.continuation.metadata.main_frontier};
            if (p.pool_count() > 1)
                frontiers.push_back(s.continuation.metadata.mtp_frontier.value());
            auto copies =
                detail::conversion_requirements(p.plan, s.continuation.metadata.placement.value,
                                                option.action.placement.value, frontiers);
            quote.work.insert(quote.work.end(), copies.begin(), copies.end());
        }
        runtime::RecoveryRouteWork route{work(s.continuation.metadata.main_frontier), {}};
        if (option.action.kind == runtime::PauseKind::KeepDevice &&
            !p.directory.states().can_recover_exclusively(
                s.continuation.image,
                context::retired_state_references(p.checkpoints, s.continuation.image, victims), 1,
                s.write_destination)) {
            for (std::uint32_t rank = 0; rank < p.plan.states.size(); ++rank)
                if (const auto& field = p.plan.states[rank].draft_context) {
                    auto transfer = context::state_transfer_requirement(
                        {field->bytes, 1}, runtime::ContextTransferDirection::DeviceToDevice);
                    transfer.rank = rank;
                    route.transfers.push_back(transfer);
                }
        }
        quote.recovery.push_back(std::move(route));
    }
    return quote;
}

runtime::CaptureCostQuote Session::capture_cost(SequenceHandle h, std::uint32_t frontier,
                                                runtime::CaptureStorage storage) const {
    const auto& s = impl_->sequence(h);
    runtime::CaptureCostQuote quote;
    quote.retained = recovery_target(RequestBasePlan(s.request), frontier,
                                     s.continuation.metadata.placement, storage);
    quote.work     = quote.retained->restores.front().transfers;
    return quote;
}

std::vector<runtime::ContextTransferRequirement>
Session::maintenance_work(const ContextAssessment& assessment) const {
    return context::maintenance_work(assessment);
}

std::vector<runtime::ContextTransferRequirement>
Session::demotion_work(const ContextDemotion&, std::span<const CheckpointHandle>) const {
    throw std::invalid_argument("GLM has no Host demotion");
}

} // namespace ninfer::models::glm5next
