#include "models/glm5next/session/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next {

std::span<const runtime::PlacementId> Session::placements(const RequestBasePlan&) const {
    return impl_->placements;
}

runtime::PlacementId Session::placement(SequenceHandle h) const {
    return impl_->sequence(h).continuation.metadata.placement;
}

std::uint64_t Session::load_tokens(SequenceHandle h) const {
    const auto& sequence = impl_->sequence(h);
    return std::max(sequence.request->summary.prompt_tokens,
                    sequence.continuation.metadata.main_frontier);
}

bool detail::SessionImpl::source_matches(const RequestBasePlan& base, CheckpointHandle handle,
                                         const ResumeState* resume) const {
    if (!checkpoints.valid(handle)) return false;
    const auto& checkpoint = checkpoints.at(handle);
    const auto& identity   = checkpoint.snapshot.identity();
    if (resume) {
        if (!resume->state_ || resume->state_->owner != this || !resume->state_->sequence ||
            resume->state_->sequence->request != base.plan_)
            throw std::invalid_argument("resume state belongs to another Session");
        const auto& previous = resume->state_->sequence->identity;
        return checkpoint.frontier() <= resume->frontier() && identity.tag == previous.tag &&
               input_prefix_equal(identity.view(), previous.view(), checkpoint.frontier());
    }
    return base.plan_->reuse && identity.tag == base.plan_->identity->tag &&
           input_prefix_equal(base.plan_->prompt.prefix_view(), identity.view(),
                              checkpoint.frontier());
}

std::optional<SourceCandidate>
Session::inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> h,
                        bool consume, std::span<const CheckpointHandle> carried,
                        std::span<const CheckpointHandle> retired, runtime::PlacementId target,
                        const ResumeState* resume) const {
    const auto& p = *impl_;
    if (target.value >= p.plan.states.size())
        throw std::invalid_argument("invalid GLM5Next placement");
    if (resume && (!resume->state_ || resume->state_->owner != &p || !resume->state_->sequence ||
                   resume->state_->sequence->request != base.plan_))
        throw std::invalid_argument("resume state belongs to another Session");
    if (h && !p.source_matches(base, *h, resume)) return {};
    SourceCandidate out;
    out.checkpoint    = h;
    out.placement     = target;
    out.reused_tokens = h ? p.checkpoints.at(*h).frontier() : 0;
    const auto end    = resume ? resume->frontier() : base.summary().prompt_tokens;
    if (out.reused_tokens > end) return {};
    out.remaining_work = runtime::make_prefill_work(out.reused_tokens, end - out.reused_tokens, 0,
                                                    0, p.shape.prefill_chunk);
    if (h) {
        const auto& c           = p.checkpoints.at(*h);
        const auto image        = c.snapshot.continuation().image;
        const auto& states      = p.directory.states();
        const auto source_owner = c.snapshot.continuation().metadata.placement.value;
        const bool local        = source_owner == target.value;
        out.consume_source      = local && consume && p.checkpoints[h->index].pins == 0 &&
                             states.checkpoint_references(image) == 1 && !states.source_pins(image);
        out.transfers = local ? c.snapshot.restore_requirements(p.directory)
                              : detail::conversion_requirements(p.plan, c.snapshot, target.value);
        out.local_device =
            local && states.device_resident(image) &&
            std::none_of(out.transfers.begin(), out.transfers.end(), [](const auto& transfer) {
                return transfer.direction == runtime::ContextTransferDirection::HostToDevice;
            });
    }
    for (const auto point : carried)
        if ((!h || point != *h || !out.consume_source) && p.source_matches(base, point, resume) &&
            p.checkpoints.at(point).frontier() <= out.reused_tokens)
            out.private_points.push_back(point);
    for (const auto point : retired)
        if ((!h || point != *h) && valid_checkpoint(point)) out.retired_points.push_back(point);
    return out;
}

detail::BindingPlan
detail::SessionImpl::plan_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                  const SourceCandidate& candidate, const ResumeState* resume,
                                  ExecutionUnitKind resume_kind, std::uint32_t tokens) const {
    if (candidate.placement.value >= this->plan.states.size())
        throw std::invalid_argument("invalid GLM5Next placement");
    if (lane.value >= shape.max_concurrency || sequences[lane.value])
        throw std::logic_error("binding requires an empty lane");
    if (resume && (!resume->state_ || resume->state_->owner != this || !resume->state_->sequence ||
                   resume->state_->sequence->request != base.plan_))
        throw std::invalid_argument("binding has an invalid resume owner");
    BindingPlan plan;
    const auto source = candidate.checkpoint;
    if (source && !source_matches(base, *source, resume)) {
        plan.evaluation.source_valid = false;
        return plan;
    }
    const bool own         = source && resume && resume->snapshot_handle() == source;
    const auto* checkpoint = source ? &checkpoints.at(*source) : nullptr;
    const auto frontier    = checkpoint ? checkpoint->frontier() : 0;
    const auto image       = checkpoint ? checkpoint->snapshot.continuation().image : Image{};
    const auto& states     = directory.states();
    const bool convert =
        checkpoint && checkpoint->snapshot.continuation().metadata.placement != candidate.placement;
    const bool resident = checkpoint && states.device_resident(image);
    const bool consume  = !convert && checkpoint && (own || candidate.consume_source) &&
                         checkpoints[source->index].pins == 0 &&
                         states.checkpoint_references(image) == 1 && !states.source_pins(image);
    const auto target = resume ? resume->frontier() : frontier;
    const auto requested_peak =
        resume
            ? std::uint64_t(target) + (resume_kind == ExecutionUnitKind::Decode
                                           ? std::min(shape.draft_tokens + 1, std::max(tokens, 1U))
                                           : tokens)
            : std::min<std::uint64_t>(base.summary().prompt_tokens,
                                      std::uint64_t(frontier) + shape.prefill_chunk);
    auto peak =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(requested_peak, shape.max_context));
    if (resume && resume_kind == ExecutionUnitKind::Prefill)
        peak = std::min<std::uint64_t>(base.summary().prompt_tokens,
                                       std::uint64_t(target) + shape.prefill_chunk);
    std::array<std::vector<const context::HistoryPrefix*>, 2> retired_prefixes;
    for (const auto point : candidate.retired_points) {
        if ((source && point == *source) ||
            std::find(plan.retired.begin(), plan.retired.end(), point) != plan.retired.end())
            continue;
        if (!checkpoints.erasable(point)) {
            plan.evaluation.source_valid = false;
            return plan;
        }
        plan.retired.push_back(point);
        for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool)
            retired_prefixes[pool].push_back(&checkpoints.at(point).snapshot.prefix(pool));
    }
    if (consume)
        for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool)
            retired_prefixes[pool].push_back(&checkpoint->snapshot.prefix(pool));
    auto& action          = plan.action;
    action.lane           = lane.value;
    action.placement      = candidate.placement;
    action.frontier       = frontier;
    action.target         = target;
    action.peak           = peak;
    action.request        = base.plan_;
    action.source         = source;
    action.source_image   = image;
    action.consume_source = consume;
    action.state_action   = !source   ? StateBindingAction::Root
                            : convert ? StateBindingAction::Convert
                            : consume ? StateBindingAction::Move
                                      : StateBindingAction::Fork;
    action.state_target   = plan.demand.add_state(
        {.count         = static_cast<std::uint32_t>(!source || !consume),
           .variant       = candidate.placement.value,
           .logical_count = static_cast<std::uint32_t>(convert || !source || !resident || !consume)});
    for (std::uint32_t pool = 0; pool < pool_count(); ++pool) {
        const auto* prefix = checkpoint ? &checkpoint->snapshot.prefix(pool) : nullptr;
        const auto end     = pool == 0 ? peak : mtp_history_frontier(peak);
        const std::array placement{candidate.placement.value};
        const auto& histories        = directory.pool(pool).histories();
        action.history_targets[pool] = plan.demand.add_history(
            pool, convert ? histories.conversion_demand(*prefix, prefix->frontier(), end, placement)
                          : histories.binding_demand(prefix, prefix ? prefix->frontier() : 0, end,
                                                     retired_prefixes[pool], placement));
    }
    plan.evaluation.assessment = directory.assess(plan.demand, footprint(plan.retired, true));
    plan.evaluation.capacity_possible = !plan.evaluation.assessment.impossible();
    return plan;
}

BindingEvaluation Session::assess_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& candidate,
                                          const ResumeState* resume, ExecutionUnitKind kind,
                                          std::uint32_t tokens) const {
    return impl_->plan_binding(base, lane, candidate, resume, kind, tokens).evaluation;
}

BindingReservation Session::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& candidate, ResumeState* resume,
                                          ExecutionUnitKind resume_kind, std::uint32_t tokens) {
    auto& p = *impl_;
    if (p.operation) throw std::logic_error("binding requires a free context transaction slot");
    auto plan = p.plan_binding(base, lane, candidate, resume, resume_kind, tokens);
    BindingReservation result{.source_valid      = plan.evaluation.source_valid,
                              .capacity_possible = plan.evaluation.capacity_possible,
                              .assessment        = std::move(plan.evaluation.assessment)};
    if (!result.source_valid || !result.assessment.feasible()) return result;
    if (result.assessment.requires_maintenance() && (p.compute || p.executor->has_pending_work())) {
        p.pending_maintenance = result.assessment;
        p.directory.request_maintenance();
        result.assessment.status = context::ContextAvailability::Waiting;
        return result;
    }
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Context);
    if (!reservation) {
        result.assessment.status = context::ContextAvailability::Waiting;
        return result;
    }
    plan.action.resume = resume;
    p.operation = std::make_unique<detail::ContextOperation>(p.directory, std::move(plan.action),
                                                             std::move(*reservation));
    auto& tx    = std::get<detail::BindOperation>(p.operation->action);
    const auto source  = tx.source;
    const bool consume = tx.consume_source;
    try {
        if (source) tx.source_lease.emplace(p.checkpoints.lease(*source));
        tx.retired.reserve(plan.retired.size() + 1);
        tx.carried.reserve(candidate.private_points.size());
        tx.clones.reserve(candidate.private_points.size());
        tx.carried_leases.reserve(candidate.private_points.size());
        for (const auto point : plan.retired) {
            if (!p.checkpoints.erase(point))
                throw std::logic_error("binding retirement changed after its reservation");
            tx.retired.push_back(point);
        }
        auto storage = p.directory.reserve(plan.demand);
        if (!storage)
            throw std::logic_error("binding placement changed after its accepted assessment");
        const bool own = source && resume && resume->snapshot_handle() == source;
        for (const auto point : candidate.private_points) {
            if (!p.checkpoints.valid(point) ||
                (source && point == *source && (tx.consume_source || own)))
                continue;
            if (point != source) tx.carried_leases.push_back(p.checkpoints.lease(point));
            if (candidate.take_private)
                tx.carried.push_back(point);
            else if (const auto clone = p.checkpoints.reserve())
                tx.clones.emplace_back(point, *clone);
        }
        if (!storage.reservation->ready()) {
            p.operation->migration.discard();
            auto maintenance = p.directory.prepare_maintenance(std::move(*storage.reservation),
                                                               storage.assessment);
            auto previous    = std::move(p.operation);
            p.operation      = std::make_unique<detail::ContextOperation>(
                std::move(maintenance), std::move(previous->action),
                std::move(previous->reservation));
            p.operation->preparing_resources = true;
        } else {
            p.prepare_binding_resources(*storage.reservation);
        }
        p.submit_context();
        result.reserved       = true;
        result.retired_points = std::move(plan.retired);
        if (consume) result.consumed_source = source;
        return result;
    } catch (...) {
        p.cancel_operation();
        throw;
    }
}

} // namespace ninfer::models::glm5next

namespace ninfer::models::glm5next::detail {

ResidencyBinding SessionImpl::context_binding(BindOperation& operation) {
    auto binding   = std::make_unique<Sequence>();
    auto& sequence = *binding;
    if (operation.resume) {
        const auto& previous               = *operation.resume->state_->sequence;
        sequence.id                        = previous.id;
        sequence.epoch                     = previous.epoch;
        sequence.request                   = previous.request;
        sequence.identity                  = previous.identity;
        sequence.reused                    = previous.reused;
        sequence.resume_phase              = previous.phase;
        sequence.next_capture              = previous.next_capture;
        sequence.captures                  = previous.captures;
        sequence.last_saved                = previous.last_saved;
        sequence.last_intermediate_attempt = previous.last_intermediate_attempt;
        sequence.speculative               = previous.speculative;
    } else {
        sequence.id           = next_sequence++;
        sequence.epoch        = identity_generation;
        sequence.request      = operation.request;
        sequence.identity     = *operation.request->identity;
        sequence.reused       = operation.frontier;
        sequence.captures     = operation.request->captures;
        sequence.last_saved   = operation.frontier;
        sequence.resume_phase = Phase::Prefill;
    }
    sequence.lane                = operation.lane;
    const auto identity_capacity = std::size_t(sequence.request->summary.prompt_tokens) +
                                   sequence.request->summary.effective_output_tokens;
    sequence.identity.reserve(identity_capacity);
    sequence.binding_generation = ++epochs[operation.lane];
    sequence.phase = operation.target > operation.frontier ? Phase::Replay : sequence.resume_phase;
    sequence.replay_target      = operation.target;
    sequence.recovery_peak      = operation.resume ? operation.peak : 0;
    sequence.permitted_frontier = operation.peak;
    if (operation.source) {
        sequence.continuation.metadata =
            checkpoints.at(*operation.source).snapshot.continuation().metadata;
    }
    sequence.continuation.metadata.placement = operation.placement;
    while (sequence.next_capture < sequence.captures.size() &&
           sequence.captures[sequence.next_capture].frontier <= operation.frontier)
        ++sequence.next_capture;

    const auto generated = std::span<const TokenId>(sequence.identity.tokens)
                               .subspan(sequence.request->summary.prompt_tokens);
    ResidencyBinding result{member(sequence), {generated.begin(), generated.end()}};
    operation.binding = std::move(binding);
    return result;
}

void SessionImpl::prepare_binding_resources(context::ContextReservation& reserved) {
    auto& bind             = std::get<BindOperation>(operation->action);
    auto& migration        = operation->migration;
    auto& states           = directory.states();
    const auto* checkpoint = bind.source ? &checkpoints.at(*bind.source) : nullptr;
    for (std::uint32_t pool = 0; pool < pool_count(); ++pool) {
        const auto* prefix = checkpoint ? &checkpoint->snapshot.prefix(pool) : nullptr;
        const std::array placement{bind.placement.value};
        auto& histories = directory.pool(pool).histories();
        const auto peak = pool ? mtp_history_frontier(bind.peak) : bind.peak;
        if (bind.state_action == StateBindingAction::Convert) {
            const auto from = checkpoint->snapshot.continuation().metadata.placement.value;
            migration.add_binding(
                pool, histories.prepare_conversion(
                          *prefix, prefix->frontier(), peak, bind.lane,
                          reserved.take(bind.history_targets[pool]), placement,
                          history_conversion(plan.caches[pool], from, bind.placement.value,
                                             prefix->frontier() % kPagedKVPageSize)));
        } else {
            migration.add_binding(pool, histories.prepare_binding(
                                            prefix, prefix ? prefix->frontier() : 0, peak,
                                            bind.lane, reserved.take(bind.history_targets[pool]),
                                            bind.consume_source, placement));
        }
    }
    if (bind.state_action == StateBindingAction::Convert) {
        const auto from = checkpoint->snapshot.continuation().metadata.placement.value;
        migration.convert_state(bind.source_image, bind.placement.value,
                                reserved.take(bind.state_target),
                                state_conversion(plan.states, from, bind.placement.value));
        return;
    }
    if (bind.state_action == StateBindingAction::Move) return;
    const auto destination =
        states.reserve_destination(reserved.take(bind.state_target), bind.placement.value);
    if (!destination) throw std::logic_error("binding lost its State metadata reservation");
    migration.own_state(*destination);
    if (bind.state_action == StateBindingAction::Root) { migration.clear_state(); }
}

} // namespace ninfer::models::glm5next::detail

namespace ninfer::models::glm5next::detail {

std::vector<runtime::ContextTransferRequirement>
conversion_requirements(const SessionPlan& plan, const Snapshot& snapshot, std::uint32_t to) {
    const auto from = snapshot.continuation().metadata.placement.value;
    std::vector<runtime::ContextTransferRequirement> out;
    for (const auto& field : state_conversion(plan.states, from, to)) {
        const bool peer = field.source_rank != field.destination_rank;
        runtime::ContextTransferRequirement r;
        r.direction   = peer ? runtime::ContextTransferDirection::PeerDeviceToDevice
                             : runtime::ContextTransferDirection::DeviceToDevice;
        r.image_count = 1;
        r.work        = {field.bytes, 1};
        r.rank        = field.source_rank;
        if (peer) r.peer = runtime::ContextPeerTransfer{field.source_rank, field.destination_rank};
        out.push_back(r);
    }
    for (std::uint16_t pool = 0; pool < plan.caches.size(); ++pool) {
        const auto f          = snapshot.prefix(pool).frontier();
        const auto& page      = plan.caches[pool];
        const auto full_pages = f / page.geometry.page_tokens;
        auto bytes            = full_pages * page.page_bytes;
        const auto ranges     = history_conversion(page, from, to, f % page.geometry.page_tokens);
        for (const auto& range : ranges.partial_page) bytes += range.bytes;
        runtime::ContextTransferRequirement r;
        r.resource    = {runtime::KvPoolId{pool}};
        r.direction   = runtime::ContextTransferDirection::PeerDeviceToDevice;
        r.page_groups = context::kv_pages_for_frontier(f);
        r.work = {bytes, static_cast<std::uint32_t>(full_pages + ranges.partial_page.size())};
        r.rank = from;
        r.peer = runtime::ContextPeerTransfer{from, to};
        out.push_back(r);
    }
    return out;
}

} // namespace ninfer::models::glm5next::detail
