#include "models/qwen4_exp/session/internal.h"
#include "models/common/context/work.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

std::size_t Session::checkpoint_capacity() const noexcept { return impl_->checkpoints.size(); }

std::vector<runtime::ContextTransferRequirement>
Session::checkpoint_restore_requirements(CheckpointHandle h) const {
    return impl_->checkpoints.at(h).snapshot.restore_requirements(impl_->directory);
}

bool Session::valid_checkpoint(CheckpointHandle h) const noexcept {
    return impl_->checkpoints.valid(h);
}

bool Session::release_checkpoint(CheckpointHandle h) noexcept {
    return impl_->checkpoints.erase(h);
}

CheckpointMetadata Session::checkpoint_metadata(CheckpointHandle h) const {
    const auto& p = *impl_;
    const auto& c = p.checkpoints.at(h);
    return {c.frontier(), c.role, p.checkpoints[h.index].pins != 0, c.origin};
}

PrefixShortlistKey Session::checkpoint_key(CheckpointHandle h, std::uint32_t f) const {
    const auto& c = impl_->checkpoints.at(h);
    if (f > c.frontier()) throw std::out_of_range("checkpoint key exceeds its frontier");
    return detail::key(c.snapshot.identity(), f);
}

bool Session::checkpoint_matches(CheckpointHandle h, const RequestBasePlan& base) const {
    if (!valid_checkpoint(h) || !base.plan_->reuse) return false;
    const auto& c        = impl_->checkpoints.at(h);
    const auto& identity = c.snapshot.identity();
    return identity.tag == base.plan_->identity->tag &&
           qwen::prefix_matches(base.plan_->prompt, identity.tokens, identity.positions,
                                c.frontier());
}

std::uint32_t Session::checkpoint_recovery_frontier(CheckpointHandle h, const RequestBasePlan& base,
                                                    std::uint32_t target) const {
    return checkpoint_matches(h, base) && impl_->checkpoints.at(h).frontier() <= target
               ? impl_->checkpoints.at(h).frontier()
               : 0;
}

std::span<const runtime::PlacementId> Session::placements(const RequestBasePlan&) const {
    static constexpr std::array placements{runtime::PlacementId{0}};
    return placements;
}

runtime::PlacementId Session::placement(SequenceHandle h) const {
    (void)impl_->sequence(h);
    return {};
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
        return qwen::recovery_prefix_matches(checkpoint.recovery_identity(),
                                             {resume->frontier(), previous.tag, previous.tokens,
                                              previous.positions, previous.digests, &previous});
    }
    return base.plan_->reuse && identity.tag == base.plan_->identity->tag &&
           qwen::prefix_matches(base.plan_->prompt, identity.tokens, identity.positions,
                                checkpoint.frontier());
}

std::optional<SourceCandidate>
Session::inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> h,
                        bool consume, std::span<const CheckpointHandle> carried,
                        std::span<const CheckpointHandle> retired, runtime::PlacementId target,
                        const ResumeState* resume) const {
    const auto& p = *impl_;
    if (target.value != 0) throw std::invalid_argument("Qwen4Exp has one context placement");
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
    for (const auto& item : base.plan_->prompt.vision_items)
        if (item.token_spans.back().begin + item.token_spans.back().count > out.reused_tokens &&
            item.token_spans.front().begin < end) {
            ++out.remaining_work.vision_items;
            out.remaining_work.vision_patches += item.patch_count;
        }
    if (const auto& vision = base.plan_->vision) {
        out.remaining_work.chunks = 0;
        const vision::Window available;
        for (auto cursor = out.reused_tokens; cursor < end;) {
            cursor = available.limit({}, vision->units, cursor,
                                     std::min(end, cursor + p.shape.prefill_chunk),
                                     p.plan.vision_max_features);
            ++out.remaining_work.chunks;
        }
    }
    if (h) {
        const auto& c      = p.checkpoints.at(*h);
        const auto image   = c.snapshot.continuation().image;
        const auto& states = p.directory.states();
        out.consume_source = consume && p.checkpoints[h->index].pins == 0 &&
                             states.checkpoint_references(image) == 1 && !states.source_pins(image);
        out.transfers = c.snapshot.restore_requirements(p.directory);
        out.local_device =
            states.device_resident(image) &&
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
    if (candidate.placement.value != 0)
        throw std::invalid_argument("Qwen4Exp has one context placement");
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
    const bool resident    = checkpoint && states.device_resident(image);
    const bool consume     = checkpoint && (own || candidate.consume_source) &&
                         checkpoints[source->index].pins == 0 &&
                         states.checkpoint_references(image) == 1 && !states.source_pins(image);
    const bool split = checkpoint && !consume && resident && states.host_resident(image) &&
                       !states.source_pins(image);
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
    if ((!resume || resume_kind == ExecutionUnitKind::Prefill) && base.plan_->vision)
        peak = vision::Window{}.limit({}, base.plan_->vision->units, target, peak,
                                      this->plan.vision_max_features);
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
    action.frontier       = frontier;
    action.target         = target;
    action.peak           = peak;
    action.request        = base.plan_;
    action.source         = source;
    action.source_image   = image;
    action.consume_source = consume;
    action.state_action   = !source     ? StateBindingAction::Root
                            : !resident ? StateBindingAction::Restore
                            : consume   ? StateBindingAction::Move
                            : split     ? StateBindingAction::Split
                                        : StateBindingAction::Fork;
    action.state_target   = plan.demand.add_state(
        {.count         = static_cast<std::uint32_t>(!resident || (!consume && !split)),
           .logical_count = static_cast<std::uint32_t>(!source || !resident || !consume)});
    for (std::uint32_t pool = 0; pool < pool_count(); ++pool) {
        const auto* prefix           = checkpoint ? &checkpoint->snapshot.prefix(pool) : nullptr;
        const auto end               = pool == 0 ? peak : mtp_history_frontier(peak);
        action.history_targets[pool] = plan.demand.add_history(
            pool, directory.pool(pool).histories().binding_demand(
                      prefix, prefix ? prefix->frontier() : 0, end, retired_prefixes[pool]));
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

ContextProgress detail::SessionImpl::complete_context() {
    auto& p = *this;
    if (!p.operation || !p.operation->result)
        throw std::logic_error("context operation has no completed result");
    auto& operation = *p.operation;
    auto& migration = operation.migration;
    ContextProgress result;
    result.kind      = operation.kind();
    result.transfers = std::move(operation.previous_transfers);
    result.transfers.insert(result.transfers.end(), operation.result->transfers.begin(),
                            operation.result->transfers.end());
    result.complete = true;
    if (!operation.final_batch) {
        operation.previous_transfers = std::move(result.transfers);
        operation.result.reset();
        auto execution = p.executor->try_reserve(runtime::FrameKind::Context);
        if (!execution)
            throw std::logic_error("retired Context batch cannot continue its transaction");
        operation.reservation = std::move(*execution);
        p.submit_context();
        result.complete = false;
        return result;
    }
    migration.mark_completed();
    if (auto* pause = std::get_if<detail::PauseOperation>(&operation.action);
        pause && pause->paused->sequence)
        pause->paused->sequence->binding_generation = 0;
    if (migration.cancelled()) {
        p.cancel_operation();
        return result;
    }
    const auto tail_copies = migration.tail_copy_page_groups();
    const auto spill_pages = migration.spill_page_groups();
    try {
        migration.publish();
        if (operation.preparing_resources) {
            auto targets                 = migration.take_reservation();
            operation.previous_transfers = std::move(result.transfers);
            p.resume_context(std::move(targets));
            result.complete = false;
            return result;
        }
        auto& states = p.directory.states();
        if (auto* tx = std::get_if<detail::BindOperation>(&operation.action)) {
            if (!tx->binding) throw std::logic_error("binding operation has no prepared sequence");
            auto& s = *tx->binding;
            for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
                s.histories[pool] = migration.binding(pool)->commit();
            using StateAction = detail::StateBindingAction;
            if (tx->state_action == StateAction::Move) {
                const auto source    = *tx->source;
                s.continuation.image = p.checkpoints.at(source).snapshot.take_mutable_image();
                tx->source_lease.reset();
                if (!p.checkpoints.erase(source))
                    throw std::logic_error("consumed checkpoint still has readers");
                tx->retired.push_back(source);
                tx->source.reset();
                ++result.operations.state_moves;
            } else {
                auto& destination = migration.state_destination();
                if (tx->state_action == StateAction::Fork) {
                    (void)destination.begin_write(tx->source_image);
                    s.continuation.image = tx->source_image;
                    s.write_destination  = destination.release();
                    ++result.operations.state_forks;
                } else {
                    if (tx->state_action == StateAction::Split) {
                        states.split_device_replica_identity(tx->source_image, destination.image());
                        ++result.operations.state_moves;
                    }
                    states.complete_initialization(destination.image());
                    s.continuation.image = destination.release();
                    if (tx->state_action == StateAction::Restore)
                        ++result.operations.state_restores;
                }
            }
            for (const auto& [from, to] : tx->clones) {
                p.checkpoints.publish(to, p.checkpoints.at(from));
                tx->carried.push_back(to);
            }
            tx->source_lease.reset();
            if (tx->consume_source && tx->source) {
                if (!p.checkpoints.erase(*tx->source))
                    throw std::logic_error("consumed checkpoint still has readers");
                tx->retired.push_back(*tx->source);
            }
            if (tx->resume) tx->resume->state_.reset();
            result.operations.partial_tail_cow_pages = tail_copies;
            result.sequence                          = p.handle(s);
            result.residency                         = p.member(s).residency;
            result.replaying                         = s.phase == detail::Phase::Replay;
            result.private_points                    = std::move(tx->carried);
            result.retired_checkpoints               = std::move(tx->retired);
            p.sequences[tx->lane]                    = std::move(tx->binding);
            tx->clones.clear();
        } else if (auto* tx = std::get_if<detail::CaptureOperation>(&operation.action)) {
            auto& s = *p.sequences[tx->lane];
            result.captured_checkpoints.reserve(tx->reservation->records.size());
            for (const auto& [record, role] : tx->reservation->records) {
                p.checkpoints.publish(record,
                                      {*tx->snapshot, role, s.captures[s.next_capture].origin});
                result.captured_checkpoints.push_back(record);
            }
            const auto writer    = context::adopt_capture_writer(states, s.continuation.image,
                                                                 migration.state_destination());
            s.continuation.image = writer.image;
            s.write_destination  = writer.write_destination;
            result.operations.state_forks += writer.write_destination.valid();
            tx->reservation->records.clear();
            s.capture_pending = false;
            s.last_saved      = s.captures[s.next_capture].frontier;
            ++s.next_capture;
        } else if (auto* tx = std::get_if<detail::PauseOperation>(&operation.action)) {
            result.request_speculative = tx->paused->sequence->speculative;
            result.paused.emplace(ResumeState(std::move(tx->paused)));
        } else if (std::holds_alternative<detail::DemoteOperation>(operation.action)) {
            result.operations.pressure_spill_pages = spill_pages;
        }
        result.published = true;
        p.operation.reset();
        return result;
    } catch (...) {
        p.cancel_operation();
        throw;
    }
}

ResourceReservation detail::SessionImpl::reserve_units(std::span<const ExecutionUnit> units) {
    auto& p = *this;
    if (p.directory.maintenance_pending())
        return {.assessment = {.status = context::ContextAvailability::Waiting}};

    struct Need {
        detail::Sequence* sequence;
        std::uint32_t end;
        std::array<std::uint32_t, 2> additional{};
        std::array<context::ContextDemandId, 2> targets;
        std::array<context::RecordReservation, 2> prepared;
    };

    std::vector<Need> needed;
    ContextDemand demand;
    for (const auto& unit : units) {
        auto& sequence = p.sequence(unit.sequence);
        if ((unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) &&
            p.unit_end(sequence, unit.kind, unit.tokens) ==
                sequence.continuation.metadata.main_frontier &&
            sequence.continuation.metadata.main_frontier < sequence.request->summary.prompt_tokens)
            return {.status = runtime::UnitReservationStatus::ExecutionUnavailable};
        if (std::any_of(needed.begin(), needed.end(),
                        [&](const auto& n) { return n.sequence == &sequence; }))
            throw std::logic_error("execution reservation repeats a sequence");
        Need need{&sequence,
                  std::max(sequence.recovery_peak, p.unit_end(sequence, unit.kind, unit.tokens))};
        if (need.end > p.shape.max_context) throw std::out_of_range("execution exceeds context");
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
            const auto& histories = p.directory.pool(pool).histories();
            const auto history    = sequence.histories[pool].handle();
            const auto begin      = pool ? sequence.continuation.metadata.mtp_frontier.value()
                                         : sequence.continuation.metadata.main_frontier;
            const auto end        = pool ? mtp_history_frontier(need.end) : need.end;
            const auto pages      = histories.write_pages(history, begin, end);
            const auto held       = histories.reserved_growth_pages(history);
            need.additional[pool] = pages > held ? pages - held : 0;
            need.targets[pool]    = demand.add_growth(histories, history, need.additional[pool]);
        }
        needed.push_back(std::move(need));
    }
    auto result = p.directory.reserve(demand);
    if (!result || !result.reservation->ready()) return {.assessment = result.assessment};
    for (auto& need : needed)
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
            need.prepared[pool] = result.reservation->take(need.targets[pool]);
            if (need.additional[pool]) {
                p.directory.pool(pool).histories().prepare_growth(
                    need.sequence->histories[pool].handle(), need.prepared[pool]);
            }
        }
    for (std::size_t i = 0; i < units.size(); ++i) {
        auto& sequence = *needed[i].sequence;
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool)
            if (needed[i].additional[pool])
                p.directory.pool(pool).histories().adopt_growth(
                    sequence.histories[pool].handle(), std::move(needed[i].prepared[pool]));
        sequence.permit             = units[i];
        sequence.permitted_frontier = needed[i].end;
    }
    return {.status = runtime::UnitReservationStatus::Granted, .assessment = result.assessment};
}

ResourceReservation Session::reserve_units(std::span<const ExecutionUnit> units) {
    return impl_->reserve_units(units);
}

void Session::release_units(std::span<const SequenceHandle> handles) noexcept {
    try {
        for (const auto h : handles)
            if (impl_->valid(h) && !execution_pending(h)) impl_->settle(impl_->sequence(h));
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::qwen4_exp

namespace ninfer::models::qwen4_exp::detail {

void SessionImpl::cancel_operation() noexcept {
    if (!operation) return;
    try {
        auto& tx = *operation;
        tx.migration.discard();
        if (auto* bind = std::get_if<BindOperation>(&tx.action)) {
            if (bind->binding) clear(*bind->binding);
            bind->source_lease.reset();
            bind->carried_leases.clear();
            for (const auto& [from, to] : bind->clones) {
                if (checkpoints.valid(to)) {
                    if (!checkpoints.erase(to)) std::terminate();
                } else
                    checkpoints.cancel(to);
            }
        } else if (auto* capture = std::get_if<CaptureOperation>(&tx.action)) {
            capture->reservation.reset();
            if (auto& sequence = sequences[capture->lane]) {
                auto& states     = directory.states();
                const auto image = sequence->continuation.image;
                if (capture->snapshot && capture->snapshot->continuation().image == image) {
                    if (sequence->write_destination.valid()) {
                        states.abort_write(image, sequence->write_destination);
                        if (!states.release(sequence->write_destination)) std::terminate();
                        sequence->write_destination = {};
                    }
                    sequence->continuation.image = capture->snapshot->take_mutable_image();
                } else if (!capture->snapshot &&
                           states.role(image) == context::StateRole::CheckpointImmutable &&
                           !states.checkpoint_references(image)) {
                    states.make_mutable(image);
                }
                sequence->capture_pending = false;
            }
        } else if (auto* pause = std::get_if<PauseOperation>(&tx.action)) {
            if (pause->paused && pause->paused->sequence) clear(*pause->paused->sequence);
            if (sequences[pause->lane]) {
                clear(*sequences[pause->lane]);
                sequences[pause->lane].reset();
            }
        }
        operation.reset();
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::qwen4_exp::detail
