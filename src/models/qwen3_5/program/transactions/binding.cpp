#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

void ProgramImpl::initialize_captures(std::uint32_t lane, std::uint32_t from,
                                      std::uint32_t through) {
    auto& request = requests[lane];
    request.capture_groups.clear();
    request.next_capture    = 0;
    request.capture_pending = false;
    for (const auto& group : request.base->capture_groups) {
        if (group.frontier > from && group.frontier <= through) {
            request.capture_groups.push_back(group);
        }
    }
}

void ProgramImpl::initialize_prefill(std::uint32_t lane, std::uint32_t base) {
    auto& request    = requests[lane];
    const auto& plan = *request.base;
    auto& staged     = request.prefill.emplace();
    staged.prompt    = *plan.prompt;
    staged.base = staged.cursor = base;
    staged.prompt_tokens        = plan.summary.prompt_tokens;
    staged.prepare_mtp          = speculative_backend == SpeculativeBackend::Mtp;
    staged.initial_mtp_extent   = initial_mtp_extent(plan);
    staged.reuse                = base ? PrefixReusePath::Checkpoint : PrefixReusePath::Root;
    staged.mtp_bridge           = !staged.prepare_mtp || !base   ? MtpBridgeMode::None
                                  : base == staged.prompt_tokens ? MtpBridgeMode::AfterExactHit
                                                                 : MtpBridgeMode::BeforeSuffix;
    initialize_captures(lane, base, plan.summary.prompt_tokens);
    if (plan.vision_control_plan) {
        auto& vision        = staged.vision_plan.emplace();
        vision.control_plan = plan.vision_control_plan;
        vision.control      = std::make_shared<VisionControl>(
            build_vision_control(staged.prompt, *vision.control_plan, 0));
        for (std::uint32_t i = 0; i < vision.control_plan->items.size(); ++i) {
            const auto& item = vision.control_plan->items[i];
            if (item.token_end <= base) { continue; }
            const auto begin =
                staged.prepare_mtp && item.token_begin ? item.token_begin - 1U : item.token_begin;
            vision.uses.push_back({begin, item.token_end, i, i});
            vision.max_merged_count = std::max(vision.max_merged_count, item.merged_count);
        }
        if (!vision.uses.empty()) {
            staged.vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, staged.prompt, vision, vision_handoff,
                vision_handoff_peak_bytes);
        }
    }
}

BindingReservation ProgramImpl::start_binding(const RequestBasePlan& base, runtime::LaneId lane_id,
                                              const SourceCandidate& candidate, ResumeState* resume,
                                              ExecutionUnitKind resume_kind,
                                              std::uint32_t resume_tokens) {
    const auto lane = lane_id.value;
    if (context_transaction_ || lane >= max_concurrency ||
        requests[lane].lifecycle != Lifecycle::Empty || !base.impl_) {
        throw std::logic_error("binding requires a free lane and context transaction slot");
    }
    const bool own_snapshot = resume && resume->has_snapshot();
    const auto source       = own_snapshot ? resume->impl_->snapshot : candidate.checkpoint;
    if (source &&
        (!valid_checkpoint(*source) ||
         (!own_snapshot && (!checkpoint_matches(*source, base) ||
                            (resume && checkpoint(*source).frontier > resume->frontier()))))) {
        return {.source_valid = false};
    }
    ContextTransaction transaction;
    transaction.kind               = ContextOperationKind::Bind;
    transaction.lane               = lane;
    transaction.epoch              = lane_epochs[lane];
    transaction.base               = base.impl_;
    transaction.resume             = resume;
    transaction.source             = source;
    transaction.reuse_frontier     = source ? checkpoint(*source).frontier : 0;
    transaction.backend_frontier   = source ? checkpoint(*source).backend_frontier : 0;
    transaction.source_tail_hidden = source && checkpoint(*source).tail_hidden_valid;
    transaction.resume_snapshot    = own_snapshot;
    transaction.retired_points.reserve(candidate.retired_points.size() + 1);
    const auto select_source = [&](const SourceCandidate& actual) {
        transaction.take_private   = candidate.take_private;
        transaction.carried_points = actual.private_points;
        // Keep semantic carry intent in the query, so space released by retirement can
        // still preserve the selected input point. An actual Move consumes that point.
        if (actual.move_state) { std::erase(transaction.carried_points, *source); }
        transaction.consume_source =
            actual.consume_source &&
            std::find(transaction.carried_points.begin(), transaction.carried_points.end(),
                      *source) == transaction.carried_points.end();
        transaction.borrow_state = actual.move_state;
        transaction.split_state  = actual.split_state;
        transaction.backup_state = actual.backup_state;
        transaction.borrow_text  = actual.move_history;
        transaction.borrow_backend =
            actual.move_history && checkpoint(*source).kv->backend.has_value();
    };
    if (!own_snapshot && source) {
        const auto actual = inspect_source(base, source, candidate.consume_source,
                                           candidate.private_points, candidate.retired_points);
        if (!actual) { return {.source_valid = false}; }
        select_source(*actual);
    } else if (own_snapshot) {
        transaction.consume_source = true;
        transaction.take_private   = candidate.take_private;
        for (const auto point : candidate.private_points) {
            if (valid_checkpoint(point) &&
                checkpoint(point).frontier <= checkpoint(*source).frontier) {
                transaction.carried_points.push_back(point);
            }
        }
        transaction.borrow_state =
            checkpoints[source->index].pins == 0 &&
            state_store->checkpoint_references(checkpoint(*source).state) == 1 &&
            state_store->source_pins(checkpoint(*source).state) == 0;
        transaction.split_state = !transaction.borrow_state &&
                                  state_store->device_resident(checkpoint(*source).state) &&
                                  state_store->host_resident(checkpoint(*source).state) &&
                                  state_store->source_pins(checkpoint(*source).state) == 0 &&
                                  state_store->device_occupied() == state_store->device_capacity();
        const auto& record = checkpoint(*source);
        const auto movable = [&](const KVAddressSpaceStore& addresses,
                                 const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                 std::uint32_t frontier) {
            if (addresses.active(address) ||
                !addresses.can_destructive_truncate_inactive(address, frontier, true)) {
                return false;
            }
            if (frontier % kPagedKVPageSize) {
                const auto tail =
                    addresses.logical_page(address, kv_pages_for_frontier(frontier) - 1U);
                return pages.address_references(tail) == 1 && pages.source_pins(tail) == 0;
            }
            return true;
        };
        transaction.borrow_text =
            movable(*text_kv_addresses, *text_kv_pages, record.kv->text, record.frontier) &&
            (!record.kv->backend || movable(*backend_kv_addresses, *backend_kv_pages,
                                            *record.kv->backend, record.backend_frontier));
        transaction.borrow_backend = transaction.borrow_text && record.kv->backend.has_value();
    }
    if (!own_snapshot && speculative_backend == SpeculativeBackend::Mtp &&
        transaction.reuse_frontier) {
        transaction.backend_frontier = transaction.reuse_frontier - 1U;
    }
    auto& request = requests[lane];
    UnitDemand first;
    if (!resume) {
        first = prefill_unit(base.summary().prompt_tokens, transaction.reuse_frontier,
                             initial_mtp_extent(*base.impl_));
    } else if (!own_snapshot) {
        first                  = {.kind = ExecutionUnitKind::Replay,
                                  .main_frontier =
                                      std::min(transaction.reuse_frontier + prefill_chunk, resume->frontier())};
        first.backend_frontier = backend_kv_cache() ? first.main_frontier : 0;
    } else {
        first =
            next_unit(resume->impl_->sequence, resume->impl_->control, resume_kind, resume_tokens);
    }
    transaction.first_unit         = first;
    transaction.reservation_demand = first;
    if (resume) {
        const auto& resumed_sequence = resume->impl_->sequence;
        auto coverage =
            next_unit(resumed_sequence, resume->impl_->control, resume_kind, resume_tokens);
        coverage.main_frontier = std::max(coverage.main_frontier, resume->frontier());
        if (backend_kv_cache()) {
            coverage.backend_frontier = std::max(coverage.backend_frontier, resume->frontier());
        }
        transaction.recovery           = RecoveryPermit{.coverage      = coverage,
                                                        .frontier      = resume->frontier(),
                                                        .ledger_tokens = resumed_sequence.ledger.size()};
        transaction.reservation_demand = coverage;
    }
    const auto& coverage      = transaction.reservation_demand;
    const auto main_prefix    = kv_pages_for_frontier(transaction.reuse_frontier);
    const auto backend_prefix = kv_pages_for_frontier(transaction.backend_frontier);
    const auto main_growth    = kv_pages_for_frontier(coverage.main_frontier) > main_prefix
                                    ? kv_pages_for_frontier(coverage.main_frontier) - main_prefix
                                    : 0U;
    const auto backend_growth =
        kv_pages_for_frontier(coverage.backend_frontier) > backend_prefix
            ? kv_pages_for_frontier(coverage.backend_frontier) - backend_prefix
            : 0U;
    const auto collect = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                             KVAddressSpaceHandle address, std::uint32_t count,
                             runtime::ContextResourceClass resource) {
        KVTransfer transfer;
        transfer.pages    = &pages;
        transfer.resource = resource;
        transfer.restore  = true;
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto page = addresses.logical_page(address, i);
            if (!pages.device_resident(page)) { transfer.logical.push_back(page); }
        }
        return transfer;
    };
    std::uint32_t main_missing = 0, backend_missing = 0;
    if (source) {
        const auto& record = checkpoint(*source);
        auto main    = collect(*text_kv_addresses, *text_kv_pages, record.kv->text, main_prefix,
                               runtime::ContextResourceClass::MainKV);
        main_missing = main.logical.size();
        transaction.kv_transfers.push_back(std::move(main));
        if (record.kv->backend) {
            auto backend    = collect(*backend_kv_addresses, *backend_kv_pages, *record.kv->backend,
                                      backend_prefix, runtime::ContextResourceClass::BackendKV);
            backend_missing = backend.logical.size();
            transaction.kv_transfers.push_back(std::move(backend));
        }
    }
    const auto main_tail =
        source && !transaction.borrow_text && transaction.reuse_frontier % kPagedKVPageSize ? 1U
                                                                                            : 0U;
    const auto backend_tail =
        source && !transaction.borrow_backend && transaction.backend_frontier % kPagedKVPageSize
            ? 1U
            : 0U;
    const auto usage = physical_usage();
    runtime::ContextResourceUsage shortage;
    const bool needs_state_slot =
        (!transaction.borrow_state && !transaction.split_state) ||
        (transaction.borrow_state && !state_store->device_resident(checkpoint(*source).state));
    shortage.state_slots =
        needs_state_slot && usage.occupied.state_slots == usage.capacity.state_slots ? 1U : 0U;
    const auto difference = [](std::uint32_t required, std::uint32_t available) {
        return required > available ? required - available : 0U;
    };
    shortage.main_kv_pages = difference(main_missing + main_growth + main_tail,
                                        text_kv_pages->physical_pool().available_pages());
    if (backend_kv_pages) {
        shortage.backend_kv_pages = difference(backend_missing + backend_growth + backend_tail,
                                               backend_kv_pages->physical_pool().available_pages());
    }
    if (!candidate.retired_points.empty() &&
        (shortage.state_slots || shortage.main_kv_pages || shortage.backend_kv_pages)) {
        const auto credit = checkpoint_release_resources(candidate.retired_points, shortage);
        if (!credit) { return {.source_valid = false}; }
        shortage.state_slots      = difference(shortage.state_slots, credit->state_slots);
        shortage.main_kv_pages    = difference(shortage.main_kv_pages, credit->main_kv_pages);
        shortage.backend_kv_pages = difference(shortage.backend_kv_pages, credit->backend_kv_pages);
    }
    if (shortage.state_slots || shortage.main_kv_pages || shortage.backend_kv_pages) {
        const bool capacity_possible =
            main_prefix + main_growth + main_tail <= usage.capacity.main_kv_pages &&
            (!backend_kv_pages ||
             backend_prefix + backend_growth + backend_tail <= usage.capacity.backend_kv_pages);
        return {.capacity_possible = capacity_possible, .shortage = shortage};
    }
    // No ordinary capacity failure follows this point. Prepare allocating request metadata
    // before the retirement/ownership handoff; failed attempts above have no side effects.
    BindingReservation result{.reserved       = true,
                              .retired_points = candidate.retired_points,
                              .consumed_source =
                                  transaction.consume_source ? source : std::nullopt};
    request.base      = base.impl_;
    request.lifecycle = Lifecycle::Prefilling;
    initialize_prefill(lane, transaction.reuse_frontier);
    sequences[lane].lane = lane;
    for (const auto point : candidate.retired_points) {
        if (!release_checkpoint(point)) {
            throw std::logic_error("binding retirement changed after capacity check");
        }
        transaction.retired_points.push_back(point);
    }
    if (source && !own_snapshot && transaction.borrow_state && !candidate.retired_points.empty() &&
        std::find(candidate.private_points.begin(), candidate.private_points.end(), *source) !=
            candidate.private_points.end()) {
        // Retirement may create a contiguous Host destination. Preserve the input image
        // when that now permits a backup/split instead of consuming it for its Device slot.
        const auto actual =
            inspect_source(base, source, candidate.consume_source, candidate.private_points);
        if (!actual) { throw std::logic_error("accepted binding lost its selected source"); }
        select_source(*actual);
        result.consumed_source = transaction.consume_source ? source : std::nullopt;
    }
    transaction.source_history = source ? checkpoint(*source).kv : nullptr;
    context_transaction_.emplace(std::move(transaction));
    auto& tx = *context_transaction_;
    if (source) { ++checkpoints[source->index].pins; }
    try {
        tx.carried_pins.reserve(tx.carried_points.size());
        tx.carried_clones.reserve(tx.carried_points.size());
        for (const auto point : tx.carried_points) {
            if (point != source) {
                ++checkpoints[point.index].pins;
                tx.carried_pins.push_back(point);
            }
            if (!tx.take_private) {
                const auto clone = reserve_checkpoint();
                if (!clone) { continue; }
                tx.carried_clones.emplace_back(point, *clone);
            }
        }
        tx.main_growth = text_kv_pages->physical_pool().reserve(main_growth + main_tail);
        if (backend_kv_pages) {
            tx.backend_growth =
                backend_kv_pages->physical_pool().reserve(backend_growth + backend_tail);
        }
        for (auto& transfer : tx.kv_transfers) {
            transfer.device_reservation =
                transfer.pages->physical_pool().reserve(transfer.logical.size());
            for (const auto page : transfer.logical) {
                transfer.physical.push_back(
                    transfer.pages->reserve_device_replica(page, *transfer.device_reservation));
            }
        }
        auto main = tx.borrow_text ? std::optional(checkpoint(*source).kv->text)
                                   : text_kv_addresses->create_inactive();
        if (!main) { throw std::logic_error("binding exhausted bounded KV descriptors"); }
        tx.reserved_kv = SequenceKVBundle{.text = *main};
        if (backend_kv_addresses) {
            auto backend = tx.borrow_backend ? checkpoint(*source).kv->backend
                                             : backend_kv_addresses->create_inactive();
            if (!backend) { throw std::logic_error("binding exhausted backend descriptors"); }
            tx.reserved_kv->backend = *backend;
        }
        if (tx.borrow_text) {
            tx.binding_history = checkpoint(*source).kv;
        } else {
            tx.binding_history          = std::make_shared<KVHistory>();
            tx.binding_history->text    = tx.reserved_kv->text;
            tx.binding_history->backend = tx.reserved_kv->backend;
        }
        const bool host_state = source && !state_store->device_resident(checkpoint(*source).state);
        tx.reserved_state     = tx.borrow_state ? std::optional(checkpoint(*source).state)
                                : (host_state || tx.split_state)
                                    ? state_store->reserve_logical_destination()
                                    : state_store->reserve_destination();
        if (!tx.reserved_state) { throw std::logic_error("binding lost its state capacity"); }
        context_source_ready_.record(device.stream);
        context_source_ready_.wait(device.transfer_stream);
        if (tx.backup_state) {
            auto backup = state_store->reserve_device_to_host(checkpoint(*source).state);
            if (!backup) {
                throw std::logic_error("source backup lost its checked Host destination");
            }
            tx.state_transfer.emplace(std::move(*backup));
            enqueue_state_backup(tx);
        }
        if (host_state) {
            tx.transfers.push_back(state_transfer_requirement(
                state_images->host_layout(), runtime::ContextTransferDirection::HostToDevice));
            start_context_transfer_timer(runtime::ContextResourceClass::State);
            auto transfer =
                tx.borrow_state
                    ? state_store->begin_host_to_device(*tx.reserved_state, device.transfer_stream)
                    : state_store->begin_host_fork(checkpoint(*source).state, *tx.reserved_state,
                                                   device.transfer_stream);
            if (transfer) { tx.state_transfer.emplace(std::move(*transfer)); }
            if (!tx.state_transfer) {
                throw std::logic_error("binding could not reserve its Host state fork");
            }
            stop_context_transfer_timer(runtime::ContextResourceClass::State);
            ++tx.operations.state_restores;
        }
        enqueue_context_transfers(tx);
        request.lifecycle = Lifecycle::Binding;
        return result;
    } catch (...) {
        abort_context();
        throw;
    }
}

void ProgramImpl::install_binding(ContextTransaction& tx) {
    auto& state        = sequences[tx.lane];
    auto& request      = requests[tx.lane];
    const auto binding = state.state;
    const auto kv      = state.kv;
    if (tx.resume) {
        auto& saved           = *tx.resume->impl_;
        request               = std::move(saved.control);
        state                 = std::move(saved.sequence);
        state.kv              = kv;
        state.state           = binding;
        state.lane            = tx.lane;
        state.mtp_draft_count = 0;
        if (!tx.resume_snapshot) {
            request.resume_lifecycle      = request.lifecycle;
            request.lifecycle             = Lifecycle::Replaying;
            request.replay_target         = saved.frontier;
            request.replay_cursor         = tx.reuse_frontier;
            state.text_kv_valid           = tx.reuse_frontier;
            state.mtp_kv_valid            = tx.backend_frontier;
            state.dflash_context_frontier = tx.reuse_frontier;
            initialize_captures(tx.lane, tx.reuse_frontier, request.replay_target);
        }
        // Reconstruct references into the moved durable prefill object.
        if (request.prefill && request.prefill->vision_plan &&
            !request.prefill->vision_plan->uses.empty()) {
            request.prefill->vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, request.prefill->prompt, *request.prefill->vision_plan,
                vision_handoff, vision_handoff_peak_bytes);
        }
        install_resume_sampling(state, request);
    } else {
        state.ledger = tx.base->prompt->token_ids;
        state.prefix_identity.assign(*tx.base->prompt);
        state.prefix_digests          = tx.base->prefix_digests;
        state.rope_delta              = tx.base->prompt->rope_delta;
        state.text_kv_valid           = tx.reuse_frontier;
        state.mtp_kv_valid            = tx.backend_frontier;
        state.dflash_context_frontier = tx.reuse_frontier;
        state.tail_hidden_valid       = tx.source_tail_hidden;
        request.lifecycle             = Lifecycle::Prefilling;
        request.publish_continuation  = tx.base->summary.publish_continuation;
        install_sampling(state, request, tx.base->sampling);
    }
    refresh_state_views(state);
    request.permit   = tx.first_unit;
    request.recovery = tx.recovery;
}

void ProgramImpl::prepare_binding(ContextTransaction& tx) {
    const auto prepare = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                             KVAddressSpaceHandle destination,
                             std::optional<KVAddressSpaceHandle> source, std::uint32_t prefix,
                             std::uint32_t target, bool borrowed,
                             std::optional<DeviceKVPageReservation>& hold,
                             std::optional<KVPrefixForkReservation>& fork,
                             std::optional<KVActivationReservation>& activation,
                             std::optional<KVActivePrefixViewReservation>& view) {
        const auto count = kv_pages_for_frontier(target), prior = kv_pages_for_frontier(prefix);
        const auto growth = count > prior ? count - prior : 0U;
        hold.reset();
        if (borrowed) {
            activation.emplace(addresses.prepare_activation(destination, growth, tx.lane, prefix));
        } else if (source && prefix && addresses.active(*source)) {
            hold = pages.physical_pool().reserve(growth);
            if (growth && !hold) { throw std::logic_error("active prefix lost reserved growth"); }
            view.emplace(addresses.prepare_active_prefix_view(*source, destination, prefix));
            if (view->needs_tail_copy()) {
                copy_context_tail(tx, pages, addresses.active_prefix_view_tail_source(*view),
                                  addresses.active_prefix_view_tail_destination(*view),
                                  &pages == text_kv_pages.get()
                                      ? runtime::ContextResourceClass::MainKV
                                      : runtime::ContextResourceClass::BackendKV);
            }
        } else if (source && prefix) {
            fork.emplace(
                addresses.prepare_prefix_fork(*source, destination, prefix, growth, tx.lane));
            if (prefix % kPagedKVPageSize) {
                copy_context_tail(tx, pages, addresses.prefix_fork_tail_source(*fork),
                                  addresses.prefix_fork_tail_destination(*fork),
                                  &pages == text_kv_pages.get()
                                      ? runtime::ContextResourceClass::MainKV
                                      : runtime::ContextResourceClass::BackendKV);
            }
        } else {
            activation.emplace(addresses.prepare_activation(destination, growth, tx.lane));
        }
    };
    prepare(*text_kv_addresses, *text_kv_pages, tx.reserved_kv->text,
            tx.source ? std::optional(checkpoint(*tx.source).kv->text) : std::nullopt,
            tx.reuse_frontier, tx.reservation_demand.main_frontier, tx.borrow_text, tx.main_growth,
            tx.text_fork, tx.text_activation, tx.text_view);
    if (tx.reserved_kv->backend) {
        prepare(*backend_kv_addresses, *backend_kv_pages, *tx.reserved_kv->backend,
                tx.source ? checkpoint(*tx.source).kv->backend : std::nullopt, tx.backend_frontier,
                tx.reservation_demand.backend_frontier, tx.borrow_backend, tx.backend_growth,
                tx.backend_fork, tx.backend_activation, tx.backend_view);
    }
    auto& state = sequences[tx.lane];
    state.state = {.read  = tx.split_state ? checkpoint(*tx.source).state : *tx.reserved_state,
                   .write = *tx.reserved_state};
    if (tx.source && !tx.borrow_state && !tx.split_state &&
        state_store->device_resident(checkpoint(*tx.source).state)) {
        const auto from      = checkpoint(*tx.source).state;
        const auto selectors = state_store->begin_fork(from, *tx.reserved_state);
        ++tx.operations.state_forks;
        state.state = {
            .read         = from,
            .write        = *tx.reserved_state,
            .fork_pending = true,
        };
        if (is_masked_draft_backend(speculative_backend)) {
            copy_local_for_context(tx, selectors.source, selectors.destination);
        }
    } else if (!tx.source) {
        state_store->activate_reset(*tx.reserved_state, device.transfer_stream);
    }
    tx.submitted = !tx.source || !tx.transfers.empty();
    if (tx.submitted) { context_completion_.record(device.transfer_stream); }
}

void ProgramImpl::complete_binding(ContextTransaction& tx, ContextProgress& out) {
    // All allocating request installation happens before the irreversible ownership handoff.
    sequences[tx.lane].kv = tx.binding_history;
    install_binding(tx);
    const auto retire_source = [&] {
        if (!tx.consume_source) { return; }
        const auto old = *tx.source;
        auto& slot     = checkpoints[old.index];
        if (slot.pins != 1) {
            throw std::logic_error("consumed checkpoint acquired another lease");
        }
        const auto old_state = slot.value->state;
        if (tx.borrow_state) {
            state_store->release_checkpoint_reference(old_state);
            state_store->move_checkpoint_to_active(old_state);
            ++tx.operations.state_moves;
        } else if (!state_store->release_checkpoint_owner(old_state)) {
            throw std::logic_error("consumed checkpoint state ownership is invalid");
        }
        slot.value.reset();
        slot.reserved = false;
        slot.pins     = 0;
        if (++slot.generation == 0) { ++slot.generation; }
        tx.retired_points.push_back(old);
        tx.source.reset();
        if (tx.resume) { tx.resume->impl_->snapshot.reset(); }
        tx.borrow_state = false;
    };
    if (tx.split_state) {
        state_store->split_device_replica_identity(checkpoint(*tx.source).state,
                                                   *tx.reserved_state);
        sequences[tx.lane].state = {.read = *tx.reserved_state, .write = *tx.reserved_state};
        ++tx.operations.state_moves;
    }
    // Source metadata has been copied; after this point a failure terminates the adopted
    // request.
    tx.adopted = true;
    retire_source();
    if (tx.borrow_text) { refresh_history_requirements(tx.binding_history); }
    const auto prepare_writable = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                      KVAddressSpaceHandle address, std::uint32_t frontier) {
        if (frontier && frontier % kPagedKVPageSize) {
            const auto tail = addresses.logical_page(address, kv_pages_for_frontier(frontier) - 1U);
            if (pages.host_resident(tail)) {
                const std::array<LogicalKVPageHandle, 1> tails{tail};
                if (!host_kv_extents->release_page_replicas(pages, tails)) {
                    throw std::logic_error("private tail Host replica is still in use");
                }
            }
        }
        addresses.destructive_truncate_inactive(address, frontier);
    };
    if (tx.borrow_text) {
        prepare_writable(*text_kv_addresses, *text_kv_pages, tx.binding_history->text,
                         tx.reuse_frontier);
    }
    if (tx.borrow_backend) {
        prepare_writable(*backend_kv_addresses, *backend_kv_pages, *tx.binding_history->backend,
                         tx.backend_frontier);
    }
    const auto activate_view = [&](KVAddressSpaceStore& addresses,
                                   std::optional<KVActivePrefixViewReservation>& view,
                                   KVAddressSpaceHandle destination, std::uint32_t target,
                                   std::optional<DeviceKVPageReservation>& hold,
                                   std::optional<KVActivationReservation>& activation) {
        if (!view) { return; }
        addresses.commit_active_prefix_view(std::move(*view));
        view.reset();
        const auto growth = addresses.growth_pages_for_tokens(destination, target);
        hold.reset();
        activation.emplace(addresses.prepare_activation(destination, growth, tx.lane));
    };
    activate_view(*text_kv_addresses, tx.text_view, tx.binding_history->text,
                  tx.reservation_demand.main_frontier, tx.main_growth, tx.text_activation);
    if (tx.binding_history->backend) {
        activate_view(*backend_kv_addresses, tx.backend_view, *tx.binding_history->backend,
                      tx.reservation_demand.backend_frontier, tx.backend_growth,
                      tx.backend_activation);
    }
    if (tx.text_fork) {
        text_kv_addresses->commit_prefix_fork(std::move(*tx.text_fork), device.stream);
    } else {
        text_kv_addresses->commit_activation(std::move(*tx.text_activation), device.stream);
    }
    if (tx.backend_fork) {
        backend_kv_addresses->commit_prefix_fork(std::move(*tx.backend_fork), device.stream);
    } else if (tx.backend_activation) {
        backend_kv_addresses->commit_activation(std::move(*tx.backend_activation), device.stream);
    }
    tx.binding_history->owner = this;
    tx.reserved_kv.reset();
    // A point follows the new view only when it belongs to the selected computation's
    // history. Equal token prefixes do not permit combining independent state and KV.
    for (const auto& [from, to] : tx.carried_clones) {
        auto point = checkpoint(from);
        if (point.kv == tx.source_history) { point.kv = tx.binding_history; }
        state_store->retain_checkpoint_reference(point.state);
        checkpoints[to.index].value    = std::move(point);
        checkpoints[to.index].reserved = false;
    }
    if (!tx.take_private) {
        tx.carried_points.clear();
        for (const auto& [from, to] : tx.carried_clones) { tx.carried_points.push_back(to); }
    } else {
        for (const auto point : tx.carried_points) {
            auto& kept = checkpoint(point);
            if (kept.kv == tx.source_history) { kept.kv = tx.binding_history; }
        }
    }
    tx.reserved_kv.reset();
    refresh_history_requirements(sequences[tx.lane].kv);
    refresh_state_views(sequences[tx.lane]);
    tx.reserved_state.reset();
    out.private_points      = std::move(tx.carried_points);
    out.retired_checkpoints = std::move(tx.retired_points);
    out.sequence            = sequence_handle(tx.lane);
    out.replaying           = requests[tx.lane].lifecycle == Lifecycle::Replaying;
}

} // namespace ninfer::models::qwen3_5::detail
