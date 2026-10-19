#include "models/qwen4_exp/session/internal.h"
#include "models/common/context/work.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

bool Session::valid_checkpoint(CheckpointHandle h) const noexcept {
    return impl_->checkpoints.valid(h);
}

bool Session::release_checkpoint(CheckpointHandle h) noexcept {
    return impl_->checkpoints.erase(h);
}

CheckpointMetadata Session::checkpoint_metadata(CheckpointHandle h) const {
    const auto& p = *impl_;
    const auto& c = p.checkpoints.at(h);
    return {c.frontier(), c.role, p.checkpoints[h.index].pins != 0};
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

std::optional<SourceCandidate>
Session::inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> h,
                        bool consume, std::span<const CheckpointHandle> carried,
                        std::span<const CheckpointHandle> retired) const {
    const auto& p = *impl_;
    if (h && !checkpoint_matches(*h, base)) return {};
    SourceCandidate out;
    out.checkpoint     = h;
    out.reused_tokens  = h ? p.checkpoints.at(*h).frontier() : 0;
    out.remaining_work = runtime::make_prefill_work(
        out.reused_tokens, base.summary().prompt_tokens - out.reused_tokens, 0, 0,
        p.shape.prefill_chunk);
    for (const auto& item : base.plan_->prompt.vision_items)
        if (item.token_spans.back().begin + item.token_spans.back().count > out.reused_tokens) {
            ++out.remaining_work.vision_items;
            out.remaining_work.vision_patches += item.patch_count;
        }
    if (const auto& vision = base.plan_->vision) {
        out.remaining_work.chunks = 0;
        for (auto cursor = out.reused_tokens; cursor < base.summary().prompt_tokens;) {
            cursor = vision->plan.window_end(
                cursor, std::min(base.summary().prompt_tokens, cursor + p.shape.prefill_chunk));
            ++out.remaining_work.chunks;
        }
    }
    if (h) {
        const auto& c      = p.checkpoints.at(*h);
        const auto image   = c.snapshot.continuation().image;
        const auto& states = p.directory.states();
        out.consume_source = consume && p.checkpoints[h->index].pins == 0 &&
                             states.checkpoint_references(image) == 1 && !states.source_pins(image);
        if (!states.device_resident(image))
            out.transfers.push_back(context::state_transfer_requirement(
                {p.directory.states().image_bytes(), 1},
                runtime::ContextTransferDirection::HostToDevice));
        for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool) {
            const auto& storage     = p.directory.pool(pool);
            const auto& prefix      = c.snapshot.prefix(pool);
            std::uint32_t missing   = 0;
            std::uint32_t run_pages = 0;
            std::optional<context::HostPageReplica> previous;
            TransferWork work;
            const auto finish_run = [&] {
                if (!run_pages) return;
                const auto run = storage.pages().host_transfer_run_work(run_pages);
                work.payload_bytes += run.payload_bytes;
                if (run.copy_operations > UINT32_MAX - work.copy_operations)
                    throw std::overflow_error("Qwen4Exp restore operation count exceeds uint32");
                work.copy_operations += run.copy_operations;
                run_pages = 0;
            };
            for (std::uint32_t i = 0; i < (prefix.frontier() + 63) / 64; ++i) {
                const auto page = storage.histories().logical_page(prefix.history(), i);
                if (storage.pages().device_covers(page,
                                                  std::min(64U, prefix.frontier() - i * 64))) {
                    finish_run();
                    previous.reset();
                    continue;
                }
                const auto& replica = storage.pages().host_replica(page);
                if (previous && (replica.extent != previous->extent ||
                                 replica.page_offset != previous->page_offset + 1))
                    finish_run();
                ++missing;
                ++run_pages;
                previous = replica;
            }
            finish_run();
            if (missing)
                out.transfers.push_back(context::kv_transfer_requirement(
                    {runtime::KvPoolId{pool}}, runtime::ContextTransferDirection::HostToDevice,
                    missing, work));
        }
    }
    for (const auto point : carried)
        if ((!h || point != *h || !out.consume_source) && checkpoint_matches(point, base) &&
            p.checkpoints.at(point).frontier() <= out.reused_tokens)
            out.private_points.push_back(point);
    for (const auto point : retired)
        if ((!h || point != *h) && valid_checkpoint(point)) out.retired_points.push_back(point);
    return out;
}

BindingReservation Session::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                          const SourceCandidate& candidate, ResumeState* resume,
                                          ExecutionUnitKind resume_kind, std::uint32_t tokens) {
    auto& p = *impl_;
    if (p.operation || lane.value >= p.shape.max_concurrency || p.sequences[lane.value])
        throw std::logic_error("binding requires an empty lane and transaction slot");
    if (resume && (!resume->state_ || resume->state_->owner != &p || !resume->state_->sequence))
        throw std::logic_error("binding has an invalid resume owner");
    const bool own    = resume && resume->has_snapshot();
    const auto source = own ? resume->snapshot_handle() : candidate.checkpoint;
    if (source &&
        (!p.checkpoints.valid(*source) ||
         (!own && (!checkpoint_matches(*source, base) ||
                   (resume && p.checkpoints.at(*source).frontier() > resume->frontier())))))
        return {.source_valid = false};
    auto& states           = p.directory.states();
    const auto* checkpoint = source ? &p.checkpoints.at(*source) : nullptr;
    const auto f           = checkpoint ? checkpoint->frontier() : 0;
    const auto image    = checkpoint ? checkpoint->snapshot.continuation().image : detail::Image{};
    const bool resident = checkpoint && states.device_resident(image);
    const bool consume  = checkpoint && (own || candidate.consume_source) &&
                         p.checkpoints[source->index].pins == 0 &&
                         states.checkpoint_references(image) == 1 && !states.source_pins(image);
    const bool split = checkpoint && !consume && resident && states.host_resident(image) &&
                       !states.source_pins(image);
    const auto target = resume ? resume->frontier() : f;
    const auto requested_peak =
        resume ? std::uint64_t(target) +
                     (resume_kind == ExecutionUnitKind::Decode
                          ? std::min(p.shape.draft_tokens + 1, std::max(tokens, 1U))
                          : tokens)
               : std::min<std::uint64_t>(base.summary().prompt_tokens,
                                         std::uint64_t(f) + p.shape.prefill_chunk);
    auto peak =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(requested_peak, p.shape.max_context));
    if (resume && resume_kind == ExecutionUnitKind::Prefill)
        peak = std::min<std::uint64_t>(base.summary().prompt_tokens,
                                       std::uint64_t(target) + p.shape.prefill_chunk);
    if ((!resume || resume_kind == ExecutionUnitKind::Prefill) && base.plan_->vision)
        peak = base.plan_->vision->plan.window_end(target, peak);
    std::vector<CheckpointHandle> retired;
    std::array<std::vector<const context::HistoryPrefix*>, 2> retired_prefixes;
    for (const auto point : candidate.retired_points) {
        if ((source && point == *source) ||
            std::find(retired.begin(), retired.end(), point) != retired.end())
            continue;
        if (!p.checkpoints.erasable(point)) return {.source_valid = false};
        retired.push_back(point);
        for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
            retired_prefixes[pool].push_back(&p.checkpoints.at(point).snapshot.prefix(pool));
    }
    if (consume)
        for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
            retired_prefixes[pool].push_back(&checkpoint->snapshot.prefix(pool));
    const auto released = p.footprint(retired, true);
    BindingReservation result;
    if ((!resident || (!consume && !split)) &&
        states.device_occupied() - released.state_slots == states.device_capacity())
        result.shortage.state_slots = 1;
    result.capacity_possible = true;
    for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool) {
        const auto& storage = p.directory.pool(pool);
        const auto* prefix  = checkpoint ? &checkpoint->snapshot.prefix(pool) : nullptr;
        const auto end      = pool == 0 ? peak : mtp_history_frontier(peak);
        const auto demand   = storage.histories().binding_demand(
            prefix, prefix ? prefix->frontier() : 0, end, retired_prefixes[pool]);
        const auto available =
            storage.allocation().available_pages() + released.kv_page_groups[pool];
        if (demand.device_pages() > available)
            result.shortage.kv_page_groups[pool] = demand.device_pages() - available;
        result.capacity_possible &=
            (end + 63) / 64 + demand.tail_copies <= p.pools[pool].device_page_groups;
    }
    if (result.shortage.state_slots || result.shortage.first_kv_deficit()) return result;
    auto reservation = p.executor->try_reserve(runtime::FrameKind::Context);
    if (!reservation) return result;

    detail::BindOperation action;
    action.lane           = lane.value;
    action.frontier       = f;
    action.target         = target;
    action.peak           = peak;
    action.request        = base.plan_;
    action.resume         = resume;
    action.source         = source;
    action.source_image   = image;
    action.consume_source = consume;
    using StateAction     = detail::StateBindingAction;
    action.state_action   = !source     ? StateAction::Root
                            : !resident ? StateAction::Restore
                            : consume   ? StateAction::Move
                            : split     ? StateAction::Split
                                        : StateAction::Fork;
    p.operation     = std::make_unique<detail::ContextOperation>(p.directory, std::move(action),
                                                                 std::move(*reservation));
    auto& tx        = std::get<detail::BindOperation>(p.operation->action);
    auto& migration = p.operation->migration;
    try {
        if (source) tx.source_lease.emplace(p.checkpoints.lease(*source));
        tx.retired.reserve(retired.size() + 1);
        tx.carried.reserve(candidate.private_points.size());
        tx.clones.reserve(candidate.private_points.size());
        tx.carried_leases.reserve(candidate.private_points.size());
        for (const auto point : retired) {
            if (!p.checkpoints.erase(point))
                throw std::logic_error("binding retirement changed after its reservation");
            tx.retired.push_back(point);
        }
        for (const auto point : candidate.private_points) {
            if (!p.checkpoints.valid(point) || (source && point == *source && (consume || own)))
                continue;
            if (point != source) tx.carried_leases.push_back(p.checkpoints.lease(point));
            if (candidate.take_private)
                tx.carried.push_back(point);
            else if (const auto clone = p.checkpoints.reserve())
                tx.clones.emplace_back(point, *clone);
        }
        for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool) {
            const auto* prefix = checkpoint ? &checkpoint->snapshot.prefix(pool) : nullptr;
            migration.add_binding(pool, p.directory.pool(pool).histories().prepare_binding(
                                            prefix, prefix ? prefix->frontier() : 0,
                                            pool == 0 ? peak : mtp_history_frontier(peak),
                                            lane.value, consume));
        }
        if (tx.state_action != StateAction::Move) {
            const bool device =
                tx.state_action == StateAction::Root || tx.state_action == StateAction::Fork;
            const auto destination =
                device ? states.reserve_destination() : states.reserve_logical_destination();
            if (!destination) throw std::logic_error("binding lost its state reservation");
            migration.own_state(*destination);
            if (tx.state_action == StateAction::Root)
                migration.clear_state();
            else if (tx.state_action == StateAction::Restore) {
                auto transfer = states.reserve_host_to_device(image, *destination);
                if (!transfer) throw std::logic_error("binding lost its Host state source");
                migration.add_state_copy(std::move(*transfer), true);
            }
        }
        p.submit_context();
        result.reserved       = true;
        result.retired_points = tx.retired;
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
    result.transfers = std::move(operation.result->transfers);
    result.complete  = true;
    migration.mark_completed();
    if (auto* pause = std::get_if<detail::PauseOperation>(&operation.action))
        pause->paused->sequence->binding_generation = 0;
    if (migration.cancelled()) {
        p.cancel_operation();
        return result;
    }
    const auto tail_copies = migration.tail_copy_page_groups();
    const auto spill_pages = migration.spill_page_groups();
    try {
        migration.publish();
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
                p.checkpoints.publish(record, {*tx->snapshot, role});
                result.captured_checkpoints.push_back(record);
            }
            auto& destination = migration.state_destination();
            if (states.device_resident(destination.image())) {
                (void)destination.begin_write(s.continuation.image);
                s.write_destination = destination.release();
                ++result.operations.state_forks;
            } else {
                states.split_device_replica_identity(s.continuation.image, destination.image());
                states.complete_initialization(destination.image());
                s.continuation.image = destination.release();
            }
            tx->reservation->records.clear();
            s.capture_pending = false;
            ++s.next_capture;
        } else if (auto* tx = std::get_if<detail::PauseOperation>(&operation.action)) {
            result.request_speculative = tx->paused->sequence->speculative;
            result.paused.emplace(ResumeState(std::move(tx->paused)));
        } else {
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

ResourceReservation Session::reserve_units(std::span<const ExecutionUnit> units) {
    auto& p = *impl_;

    struct Need {
        detail::Sequence* sequence;
        std::uint32_t end;
        std::array<std::uint32_t, 2> pages{};
    };

    std::vector<Need> needed;
    ContextResources additional;
    for (const auto& unit : units) {
        auto& s = p.sequence(unit.sequence);
        if (std::any_of(needed.begin(), needed.end(),
                        [&](const auto& n) { return n.sequence == &s; }))
            throw std::logic_error("execution reservation repeats a sequence");
        Need need{&s, std::max(s.recovery_peak, p.unit_end(s, unit.kind, unit.tokens))};
        if (need.end > p.shape.max_context) throw std::out_of_range("execution exceeds context");
        for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool) {
            const auto& histories = p.directory.pool(pool).histories();
            const auto handle     = s.histories[pool].handle();
            const auto begin      = pool == 0 ? s.continuation.metadata.main_frontier
                                              : s.continuation.metadata.mtp_frontier.value();
            const auto end        = pool == 0 ? need.end : mtp_history_frontier(need.end);
            need.pages[pool]      = histories.write_pages(handle, begin, end);
            const auto reserved   = histories.reserved_growth_pages(handle);
            additional.kv_page_groups[pool] +=
                need.pages[pool] > reserved ? need.pages[pool] - reserved : 0;
        }
        needed.push_back(need);
    }
    ContextResources shortage;
    for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool) {
        const auto free = p.directory.pool(pool).allocation().available_pages();
        if (additional.kv_page_groups[pool] > free)
            shortage.kv_page_groups[pool] = additional.kv_page_groups[pool] - free;
    }
    if (shortage.first_kv_deficit()) return {.shortage = shortage};
    for (std::size_t i = 0; i < units.size(); ++i) {
        auto& s = *needed[i].sequence;
        for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
            p.directory.pool(pool).histories().reserve_growth(s.histories[pool].handle(),
                                                              needed[i].pages[pool]);
        s.permit             = units[i];
        s.permitted_frontier = needed[i].end;
    }
    return {.reserved = true};
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
