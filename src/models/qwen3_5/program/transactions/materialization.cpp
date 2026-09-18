#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

runtime::ContextTransactionReserveStatus
ProgramImpl::reserve_materialization(AdmissionCandidate&& plan, PreparedPromptData&& prompt,
                                     runtime::CancellationFlagView cancellation) {
    if (cancellation.requested()) { return runtime::ContextTransactionReserveStatus::Aborted; }
    const runtime::PreflightStatus preflight = revalidate_materialization(plan, prompt);
    if (preflight != runtime::PreflightStatus::Ready) {
        throw std::logic_error("materialization changed after successful preflight");
    }
    if (has_context_transaction() || pending_transaction_) {
        throw std::logic_error("Program already owns a physical transaction");
    }
    if (plan.impl_ == nullptr) {
        throw std::invalid_argument("materialization reservation is invalid");
    }

    const AdmissionCandidateImpl& details = *plan.impl_;
    const std::uint32_t lane              = details.destination.value;
    if (lane >= max_concurrency || details.destination_epoch != lane_epochs[lane] ||
        requests[lane].lifecycle != Lifecycle::Empty ||
        active_continuations[lane] < continuation_capacity) {
        throw std::logic_error("materialization activation is stale");
    }

    const SequenceState* source_state =
        details.has_source ? &continuation_states[details.source_index] : nullptr;
    const SharedPrefixState* shared_state =
        details.has_shared_source ? &shared_prefix_states[details.shared_source_index] : nullptr;
    MaterializationTransaction transaction;
    transaction.id                  = next_materialization_id_++;
    transaction.destination         = details.destination;
    transaction.has_source          = details.has_source;
    transaction.has_shared_source   = details.has_shared_source;
    transaction.source_mode         = details.source_mode;
    transaction.source_index        = details.has_source ? details.source_index : 0;
    transaction.source_generation   = details.has_source ? details.source_generation : 0;
    transaction.shared_source_index = details.has_shared_source ? details.shared_source_index : 0;
    transaction.shared_source_generation =
        details.has_shared_source ? details.shared_source_generation : 0;
    if (source_state != nullptr) {
        transaction.source_result.emplace();
        transaction.source_result->final_summary.emplace();
        transaction.source_result->final_summary->long_anchors.reserve(
            source_state->long_anchors.size());
    }
    if (shared_state != nullptr) { transaction.shared_source_result.emplace(); }
    const std::size_t victim_count        = details.pressure_options.size();
    const std::size_t shared_victim_count = details.shared_pressure_options.size();
    transaction.victim_count              = victim_count;
    transaction.victim_indices.resize(victim_count);
    transaction.victim_generations.resize(victim_count);
    transaction.victim_released.resize(victim_count, false);
    transaction.pressure.reserve(victim_count);
    transaction.pressure_results.resize(victim_count);
    transaction.shared_victim_count = shared_victim_count;
    transaction.shared_victim_indices.resize(shared_victim_count);
    transaction.shared_victim_generations.resize(shared_victim_count);
    transaction.shared_victim_released.resize(shared_victim_count, false);
    transaction.shared_pressure_results.resize(shared_victim_count);
    transaction.shared_pressure.reserve(shared_victim_count);
    if (victim_count + shared_victim_count > (std::numeric_limits<std::size_t>::max() - 3U) / 3U) {
        throw std::overflow_error("materialization transfer observation capacity overflow");
    }
    transaction.transfer_observations.reserve(3U * (victim_count + shared_victim_count) + 3U);
    const SequenceKVBundle* source_kv =
        source_state != nullptr
            ? (source_state->kv ? &*source_state->kv : nullptr)
            : (shared_state != nullptr && shared_state->kv ? &*shared_state->kv : nullptr);
    if ((source_state != nullptr || shared_state != nullptr) && source_kv == nullptr) {
        throw std::logic_error("materialization source has no KV address space");
    }
    if (source_kv != nullptr) {
        const std::uint32_t text_pages = text_kv_addresses->mapped_pages(source_kv->text);
        transaction.text_restores.reserve(text_pages);
        transaction.text_restore_destinations.reserve(text_pages);
        if (source_kv->backend) {
            const std::uint32_t backend_pages =
                backend_kv_addresses->mapped_pages(*source_kv->backend);
            transaction.backend_restores.reserve(backend_pages);
            transaction.backend_restore_destinations.reserve(backend_pages);
        }
    }
    for (std::size_t victim = 0; victim < victim_count; ++victim) {
        const std::uint32_t index      = details.pressure_indices[victim];
        const std::uint64_t generation = details.pressure_generations[victim];
        if (details.has_source && index == transaction.source_index &&
            generation == transaction.source_generation) {
            throw std::logic_error("materialization source was also selected as a victim");
        }
        for (std::size_t prior = 0; prior < victim; ++prior) {
            if (transaction.victim_indices[prior] == index &&
                transaction.victim_generations[prior] == generation) {
                throw std::logic_error("materialization victim capability is duplicated");
            }
        }
        transaction.victim_indices[victim]         = index;
        transaction.victim_generations[victim]     = generation;
        transaction.pressure_results[victim].owner = details.pressure_owner_ids[victim];
        transaction.pressure_results[victim].final_summary.emplace();
        transaction.pressure_results[victim].final_summary->long_anchors.reserve(
            continuation_states[index].long_anchors.size());
        transaction.pressure.push_back(MaterializationTransaction::PressureWork{
            .option                  = details.pressure_options[victim],
            .continuation_index      = index,
            .continuation_generation = generation,
        });
        prepare_pressure_bookkeeping(transaction.pressure.back());
    }
    for (std::size_t victim = 0; victim < shared_victim_count; ++victim) {
        const std::uint32_t index      = details.shared_pressure_indices[victim];
        const std::uint64_t generation = details.shared_pressure_generations[victim];
        if ((details.has_shared_source && index == transaction.shared_source_index &&
             generation == transaction.shared_source_generation) ||
            shared_prefix_states[index].active_references != 0) {
            throw std::logic_error("materialization shared source was also selected as a victim");
        }
        for (std::size_t prior = 0; prior < victim; ++prior) {
            if (transaction.shared_victim_indices[prior] == index &&
                transaction.shared_victim_generations[prior] == generation) {
                throw std::logic_error("materialization shared victim capability is duplicated");
            }
        }
        transaction.shared_victim_indices[victim]     = index;
        transaction.shared_victim_generations[victim] = generation;
        transaction.shared_pressure_results[victim].owner =
            details.shared_pressure_owner_ids[victim];
        transaction.shared_pressure.push_back(MaterializationTransaction::PressureWork{
            .option                  = details.shared_pressure_options[victim],
            .continuation_index      = index,
            .continuation_generation = generation,
            .shared_owner            = true,
        });
        prepare_pressure_bookkeeping(transaction.shared_pressure.back());
    }
    if (transaction.id == 0) { transaction.id = next_materialization_id_++; }

    if (!details.has_source || details.source_mode == runtime::PrivateSourceMode::Retain) {
        for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
            if (continuation_slots[index].role != ContinuationSlotRole::Free) { continue; }
            transaction.root_continuation_index = index;
            break;
        }
        if (!transaction.root_continuation_index) {
            const auto eviction =
                std::find_if(details.pressure_options.begin(), details.pressure_options.end(),
                             [](const qwen3_5::detail::PressureDecision& option) {
                                 return option.evicts_continuation;
                             });
            if (eviction == details.pressure_options.end()) {
                throw std::logic_error(
                    "preserving materialization has no continuation destination");
            }
            const std::size_t position =
                static_cast<std::size_t>(eviction - details.pressure_options.begin());
            transaction.root_continuation_index = transaction.victim_indices[position];
            transaction.root_waiting_for_victim = true;
        }
    }

    const auto host_started = Clock::now();
    transaction.plan.emplace(std::move(plan));
    AdmissionCandidateImpl& request_plan = *transaction.plan->impl_;
    RequestControl& request              = requests[lane];
    try {
        const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
        if (prompt_tokens != request_plan.summary.prompt_tokens ||
            (request_plan.vision.has_value() && !prompt.has_media())) {
            throw std::invalid_argument("request plan does not describe the prepared prompt");
        }
        if (prompt.identity.rewrite_checkpoint &&
            (prompt.identity.rewrite_checkpoint->frontier == 0 ||
             prompt.identity.rewrite_checkpoint->frontier > prompt_tokens)) {
            throw std::invalid_argument("prepared prompt has an invalid rewrite checkpoint");
        }
        const bool suffix_has_visual = std::any_of(
            prompt.token_types.begin() + static_cast<std::ptrdiff_t>(request_plan.reuse_base),
            prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
        if (suffix_has_visual != request_plan.vision.has_value()) {
            throw std::invalid_argument(
                "request plan does not describe the prompt suffix modality");
        }
        if (((source_state == nullptr && shared_state == nullptr) !=
             (request_plan.reuse == ReusePath::Root))) {
            throw std::logic_error("materialization source does not match the selected reuse path");
        }
        if (source_state != nullptr &&
            !qwen3_5::detail::prefix_matches(prompt, source_state->ledger,
                                             source_state->prefix_identity,
                                             request_plan.reuse_base)) {
            throw std::logic_error("planned resident prefix is no longer reusable");
        }
        if (shared_state != nullptr &&
            (!shared_state->identity || shared_state->identity->prefix_identity() == nullptr ||
             !qwen3_5::detail::prefix_matches(prompt, shared_state->identity->ledger(),
                                              *shared_state->identity->prefix_identity(),
                                              request_plan.reuse_base))) {
            throw std::logic_error("planned shared prefix is no longer reusable");
        }
        if (request_plan.reuse == ReusePath::SharedStablePrefix &&
            (!request_plan.selected_checkpoint ||
             request_plan.selected_checkpoint->kind !=
                 runtime::CheckpointKind::SharedStablePrefix ||
             request_plan.selected_checkpoint->frontier != shared_state->frontier ||
             request_plan.selected_checkpoint->ordinal != 0)) {
            throw std::logic_error("planned shared-prefix checkpoint is unavailable");
        }
        if (is_rewrite_checkpoint_restore(request_plan.reuse) &&
            (!source_state->rewrite_checkpoint.valid ||
             source_state->rewrite_checkpoint.frontier != request_plan.reuse_base ||
             request_plan.reuse != restore_path(source_state->rewrite_checkpoint.kind))) {
            throw std::logic_error("planned rewrite checkpoint is unavailable");
        }
        if (request_plan.reuse == ReusePath::PrivateLongAnchor &&
            (!request_plan.selected_checkpoint ||
             request_plan.selected_checkpoint->kind != runtime::CheckpointKind::LongAnchor ||
             std::none_of(source_state->long_anchors.begin(), source_state->long_anchors.end(),
                          [&](const LongAnchorCheckpoint& anchor) {
                              return anchor.frontier ==
                                         request_plan.selected_checkpoint->frontier &&
                                     anchor.ordinal == request_plan.selected_checkpoint->ordinal &&
                                     state_store->valid(anchor.state);
                          }))) {
            throw std::logic_error("planned long-anchor checkpoint is unavailable");
        }
        if (request_plan.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting &&
            (!prompt.identity.rewrite_checkpoint || source_state == nullptr ||
             !can_retain_rewrite_checkpoint(prompt, *prompt.identity.rewrite_checkpoint,
                                            *source_state, request_plan.reuse,
                                            request_plan.reuse_base))) {
            throw std::logic_error("planned rewrite checkpoint retention is unavailable");
        }
        if (request_plan.rewrite_disposition ==
                RewriteCheckpointDisposition::ReplaceAtCommittedFrontier &&
            (!prompt.identity.rewrite_checkpoint ||
             std::none_of(request_plan.capture_groups.begin(), request_plan.capture_groups.end(),
                          [&](const CaptureGroup& group) {
                              return group.rewrite &&
                                     *group.rewrite == prompt.identity.rewrite_checkpoint->kind &&
                                     group.frontier == prompt.identity.rewrite_checkpoint->frontier;
                          }))) {
            throw std::logic_error("planned rewrite checkpoint capture is invalid");
        }
        for (const CaptureGroup& group : request_plan.capture_groups) {
            const bool base_shared_promotion = group.frontier == request_plan.reuse_base &&
                                               group.shared && !group.rewrite && !group.long_anchor;
            if (!group.identity ||
                (group.frontier <= request_plan.reuse_base && !base_shared_promotion) ||
                group.frontier > prompt_tokens ||
                group.identity->shortlist_key.frontier != group.frontier ||
                group.identity->prefix_identity() == nullptr ||
                !qwen3_5::detail::prefix_matches(prompt, group.identity->ledger(),
                                                 *group.identity->prefix_identity(),
                                                 group.frontier)) {
                throw std::logic_error("planned capture identity is invalid");
            }
        }

        if (request.prefill) {
            throw std::logic_error("free request lane retained prefill bookkeeping");
        }
        if (request_plan.vision) {
            std::vector<bool> used(prompt.media_payloads.size(), false);
            for (const VisionUseSpan& use : request_plan.vision->uses) {
                if (use.prepared_item_index >= used.size()) {
                    throw std::logic_error("Vision plan references a missing media payload");
                }
                used[use.prepared_item_index] = true;
            }
            for (std::size_t index = 0; index < used.size(); ++index) {
                if (!used[index]) {
                    prompt.media_payloads[index].reset();
                    continue;
                }
            }
            VisionPrefillPlan& vision      = *request_plan.vision;
            const std::uint32_t first_item = vision.uses.front().prepared_item_index;
            if (!vision.control_plan) {
                throw std::logic_error("Vision suffix plan has no prepared metadata");
            }
            auto control = std::make_shared<qwen3_5::VisionControl>(
                qwen3_5::build_vision_control(prompt, *vision.control_plan, first_item));
            for (VisionUseSpan& use : vision.uses) {
                if (use.prepared_item_index < first_item) {
                    throw std::logic_error("Vision suffix item order changed during admission");
                }
                use.control_index = use.prepared_item_index - first_item;
                if (use.control_index >= control->items.size()) {
                    throw std::logic_error("Vision suffix control does not cover a planned item");
                }
            }
            vision.control = std::move(control);
            vision.control_plan.reset();
        }
        if (prompt.has_media() && !request_plan.vision) { prompt.release_all_media_payloads(); }

        materialization_ledger_.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        materialization_identity_.assign(prompt);
        materialization_prefix_digests_.assign(prompt);

        const std::uint32_t initial_mtp_extent =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min({draft_window,
                            request_plan.summary.effective_output_tokens > 1
                                ? request_plan.summary.effective_output_tokens - 2
                                : 0U,
                            capacity - prompt_tokens > 0 ? capacity - prompt_tokens - 1 : 0U})
                : 0U;
        RequestControl::Prefill prefill{
            .prompt             = std::move(prompt),
            .vision_plan        = std::move(request_plan.vision),
            .vision             = nullptr,
            .capture_groups     = std::move(request_plan.capture_groups),
            .base               = request_plan.reuse_base,
            .cursor             = request_plan.reuse_base,
            .prompt_tokens      = prompt_tokens,
            .initial_mtp_extent = initial_mtp_extent,
            .elapsed_seconds    = 0.0,
            .prepare_mtp        = request_plan.prepare_mtp,
            .reuse              = request_plan.reuse,
            .mtp_bridge         = request_plan.mtp_bridge,
        };
        request.prefill.emplace(std::move(prefill));
        if (request.prefill->vision_plan) {
            if (!workspace_plan.vision) {
                throw std::logic_error("Vision prefill has no startup workspace plan");
            }
            request.prefill->vision = std::make_unique<execution::VisionPrefillSession>(
                device, parameters,
                DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, request.prefill->prompt, *request.prefill->vision_plan,
                vision_handoff_peak_bytes);
        }
        request.prefill->elapsed_seconds =
            std::chrono::duration<double>(Clock::now() - host_started).count();
        static_assert(std::is_nothrow_move_constructible_v<MaterializationTransaction>);
        if (transaction.root_continuation_index && !transaction.root_waiting_for_victim) {
            ContinuationSlot& destination =
                continuation_slots[*transaction.root_continuation_index];
            if (destination.role != ContinuationSlotRole::Free) {
                throw std::logic_error("materialization continuation destination changed");
            }
            destination.role = ContinuationSlotRole::ReservedMaterialization;
        }
        advance_resource_revision();
        context_transaction_.emplace<MaterializationTransaction>(std::move(transaction));
        return runtime::ContextTransactionReserveStatus::Reserved;
    } catch (...) {
        release_materialization_staging(transaction);
        throw;
    }
}

void ProgramImpl::release_materialization_staging(
    MaterializationTransaction& transaction) noexcept {
    const std::uint32_t lane = transaction.destination.value;
    if (lane < max_concurrency && requests[lane].lifecycle == Lifecycle::Empty) {
        requests[lane].prefill.reset();
    }
    for (std::size_t position = transaction.shared_pressure_cursor;
         position < transaction.shared_pressure.size(); ++position) {
        MaterializationTransaction::PressureWork& work = transaction.shared_pressure[position];
        if (work.submitted) {
            try {
                context_completion_.synchronize();
            } catch (...) { std::terminate(); }
        }
        abort_pressure_work(work);
    }

    for (std::size_t position = transaction.pressure_cursor; position < transaction.pressure.size();
         ++position) {
        MaterializationTransaction::PressureWork& work = transaction.pressure[position];
        if (work.submitted) {
            try {
                context_completion_.synchronize();
            } catch (...) { std::terminate(); }
        }
        abort_pressure_work(work);
    }

    abort_materialization_transfers(transaction);
    transaction.backend_retained_tail_backup.reset();
    transaction.text_retained_tail_backup.reset();
    transaction.backend_retained_tail.reset();
    transaction.text_retained_tail.reset();
    transaction.backend_prefix_fork.reset();
    transaction.text_prefix_fork.reset();
    transaction.backend_source_restore_reservation.reset();
    transaction.text_source_restore_reservation.reset();
    transaction.backend_activation.reset();
    transaction.text_activation.reset();
    if (transaction.root_backend_address && backend_kv_addresses) {
        (void)backend_kv_addresses->release(*transaction.root_backend_address);
        transaction.root_backend_address.reset();
    }
    if (transaction.root_text_address && text_kv_addresses) {
        (void)text_kv_addresses->release(*transaction.root_text_address);
        transaction.root_text_address.reset();
    }
    if (transaction.state_fork_destination) {
        if (state_store) { (void)state_store->release(*transaction.state_fork_destination); }
        transaction.state_fork_destination.reset();
    }
    for (std::size_t index = 0; index < transaction.reserved_state_count; ++index) {
        if (state_store) { (void)state_store->release(transaction.reserved_states[index]); }
        transaction.reserved_states[index] = {};
    }
    transaction.reserved_state_count = 0;

    if (transaction.root_continuation_index) {
        const std::uint32_t index = *transaction.root_continuation_index;
        if (index < continuation_capacity &&
            continuation_slots[index].role == ContinuationSlotRole::ReservedMaterialization) {
            release_continuation_slot_best_effort(index);
        }
        transaction.root_continuation_index.reset();
    }
    transaction.prepared                       = false;
    transaction.prefix_tail_submitted          = false;
    transaction.retained_tail_backup_submitted = false;
    transaction.prefix_forks_ready             = false;
    materialization_ledger_.clear();
    materialization_identity_.clear();
    materialization_prefix_digests_.clear();
}

void ProgramImpl::prepare_consumed_source(MaterializationTransaction& transaction) {
    if (transaction.source_prepared || !transaction.plan || transaction.plan->impl_ == nullptr) {
        throw std::logic_error("materialization source preparation state is invalid");
    }
    transaction.source_prepared           = true;
    const AdmissionCandidateImpl& details = *transaction.plan->impl_;
    if (!transaction.has_source ||
        details.source_mode != runtime::PrivateSourceMode::ConsumeToActive) {
        return;
    }
    if (transaction.source_index >= continuation_capacity ||
        continuation_slots[transaction.source_index].role != ContinuationSlotRole::Catalogued ||
        continuation_slots[transaction.source_index].generation != transaction.source_generation) {
        throw std::logic_error("materialization source changed before dependency release");
    }
    SequenceState& source = continuation_states[transaction.source_index];
    if (!source.kv || details.reuse == ReusePath::Root ||
        details.reuse == ReusePath::SharedStablePrefix) {
        throw std::logic_error("consumed materialization source is incomplete");
    }

    const detail::PhysicalResources before = owner_exclusive_resources(source);
    const auto retained_state              = [&](StateImageHandle handle) {
        if (source.endpoint_valid && source.state.read == handle) { return true; }
        if (source.rewrite_state && *source.rewrite_state == handle) { return true; }
        return std::any_of(
            source.long_anchors.begin(), source.long_anchors.end(),
            [&](const LongAnchorCheckpoint& anchor) { return anchor.state == handle; });
    };
    const auto release_if_unreferenced = [&](StateImageHandle handle) {
        if (!state_store->valid(handle) || retained_state(handle) ||
            state_store->checkpoint_references(handle) != 0) {
            return;
        }
        if (!state_store->release(handle)) {
            throw std::logic_error("superseded source StateImage remained pinned");
        }
    };

    if (source.endpoint_valid && source.execution_frontier > details.reuse_base) {
        const StateImageHandle endpoint = source.state.read;
        source.endpoint_valid           = false;
        source.state                    = {};
        source.tail_hidden              = {};
        source.tail_hidden_valid        = false;
        release_if_unreferenced(endpoint);
    }
    for (std::size_t index = source.long_anchors.size(); index != 0; --index) {
        LongAnchorCheckpoint& anchor = source.long_anchors[index - 1U];
        if (anchor.frontier <= details.reuse_base) { continue; }
        const StateImageHandle state = anchor.state;
        state_store->release_checkpoint_reference(state);
        source.long_anchors.erase(source.long_anchors.begin() +
                                  static_cast<std::ptrdiff_t>(index - 1U));
        release_if_unreferenced(state);
    }
    if (details.reuse == ReusePath::PrivateEndpoint &&
        details.rewrite_disposition != RewriteCheckpointDisposition::RetainExisting &&
        source.rewrite_state) {
        const StateImageHandle rewrite = *source.rewrite_state;
        state_store->release_checkpoint_reference(rewrite);
        source.rewrite_state.reset();
        source.rewrite_checkpoint        = {};
        source.rewrite_checkpoint_hidden = {};
        release_if_unreferenced(rewrite);
    }

    struct TruncateTarget {
        KVAddressSpaceStore* addresses = nullptr;
        LogicalKVPageStore* pages      = nullptr;
        KVAddressSpaceHandle address;
        std::uint32_t frontier        = 0;
        bool prefix_fork              = false;
        bool releases_stale_host_tail = false;
    };

    std::array<TruncateTarget, 2> targets{};
    std::size_t target_count = 0;
    targets[target_count++]  = TruncateTarget{
         .addresses   = text_kv_addresses.get(),
         .pages       = text_kv_pages.get(),
         .address     = source.kv->text,
         .frontier    = details.reuse_base,
         .prefix_fork = details.text_prefix_fork_required,
    };
    if (source.kv->backend) {
        targets[target_count++] = TruncateTarget{
            .addresses   = backend_kv_addresses.get(),
            .pages       = backend_kv_pages.get(),
            .address     = *source.kv->backend,
            .frontier    = backend_frontier_at(speculative_backend, details.reuse_base),
            .prefix_fork = details.backend_prefix_fork_required,
        };
    }

    std::array<HostKVPageReplicaRelease, 2> host_tail_releases{};
    std::size_t host_tail_release_count = 0;
    for (TruncateTarget& target : std::span(targets.data(), target_count)) {
        if (target.prefix_fork) {
            if (!target.addresses->can_truncate_inactive_prefix(target.address, target.frontier)) {
                throw std::logic_error("COW source KV suffix is not releasable");
            }
            continue;
        }
        const std::uint32_t target_pages = kv_pages_for_frontier(target.frontier);
        if (target_pages != 0) {
            const LogicalKVPageHandle tail =
                target.addresses->logical_page(target.address, target_pages - 1U);
            const std::uint32_t columns =
                target.frontier -
                (target_pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            target.releases_stale_host_tail = columns != target.pages->committed_columns(tail) &&
                                              target.pages->host_resident(tail);
            if (target.releases_stale_host_tail) {
                if (!host_kv_extents || host_tail_release_count == host_tail_releases.size()) {
                    throw std::logic_error("stale source Host KV tail is not releasable");
                }
                host_tail_releases[host_tail_release_count++] =
                    HostKVPageReplicaRelease{.pages = target.pages, .page = tail};
            }
        }
        if (!target.addresses->can_destructive_truncate_inactive(target.address, target.frontier,
                                                                 target.releases_stale_host_tail)) {
            throw std::logic_error("consumed source KV is not destructively truncatable");
        }
    }
    if (host_tail_release_count != 0) {
        const std::span<const HostKVPageReplicaRelease> releases(host_tail_releases.data(),
                                                                 host_tail_release_count);
        if (!host_kv_extents->release_page_replicas(releases)) {
            throw std::logic_error("stale source Host KV tails cannot be released atomically");
        }
    }
    for (TruncateTarget& target : std::span(targets.data(), target_count)) {
        if (target.prefix_fork) {
            target.addresses->truncate_inactive_prefix(target.address, target.frontier);
        } else {
            target.addresses->destructive_truncate_inactive(target.address, target.frontier);
        }
        target.addresses->set_checkpoint_requirement(target.address, target.frontier);
    }
    source.text_kv_valid = details.reuse_base;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        source.mtp_kv_valid = backend_frontier_at(speculative_backend, details.reuse_base);
    } else if (is_masked_draft_backend(speculative_backend)) {
        source.dflash_context_frontier = details.reuse_base;
    }
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
    refresh_state_views(source);

    const detail::PhysicalResources after   = owner_exclusive_resources(source);
    const detail::PhysicalResources removed = checked_resource_difference(before, after);
    (void)checked_resource_difference(details.demand.final_removed, removed);
}

void ProgramImpl::prepare_materialization(MaterializationTransaction& transaction) {
    if (transaction.prepared || !transaction.plan ||
        transaction.destination.value >= max_concurrency ||
        !requests[transaction.destination.value].prefill || !transaction.source_prepared) {
        throw std::logic_error("materialization preparation state is invalid");
    }
    for (std::size_t victim = 0; victim < transaction.victim_count; ++victim) {
        if (!transaction.victim_released[victim]) {
            throw std::logic_error("materialization preparation has an unreleased victim");
        }
    }

    const auto prepare_started            = Clock::now();
    const AdmissionCandidateImpl& details = *transaction.plan->impl_;
    const detail::PhysicalDemand& demand  = details.demand;
    const std::uint32_t lane              = transaction.destination.value;
    if (transaction.has_source &&
        (transaction.source_index >= continuation_capacity ||
         continuation_slots[transaction.source_index].role != ContinuationSlotRole::Catalogued ||
         continuation_slots[transaction.source_index].generation !=
             transaction.source_generation)) {
        throw std::logic_error("materialization source changed during capacity preparation");
    }
    if (transaction.has_shared_source &&
        (transaction.shared_source_index >= shared_prefix_capacity ||
         shared_prefix_slots[transaction.shared_source_index].role !=
             SharedPrefixSlotRole::Catalogued ||
         shared_prefix_slots[transaction.shared_source_index].generation !=
             transaction.shared_source_generation)) {
        throw std::logic_error("materialization shared source changed during capacity preparation");
    }
    SequenceState* source_state =
        transaction.has_source ? &continuation_states[transaction.source_index] : nullptr;
    SharedPrefixState* shared_state = transaction.has_shared_source
                                          ? &shared_prefix_states[transaction.shared_source_index]
                                          : nullptr;
    std::uint32_t state_count       = demand.reservation_added.device.state_slots;
    std::optional<StateImageHandle> host_state_restore;
    std::optional<StateImageHandle> host_state_fork_destination;
    if (source_state != nullptr || shared_state != nullptr) {
        const StateImageHandle state =
            source_state != nullptr
                ? selected_state(*source_state, details.reuse, details.selected_checkpoint)
                : shared_state->state;
        const StateReplicaResidency residency = state_store->residency(state);
        // Source existence is a StateImageStore fact. Owner-exclusive resources may be zero for a
        // valid allocation aliased by private and shared checkpoints.
        if (state_store->role(state) != StateImageRole::CheckpointImmutable ||
            residency == StateReplicaResidency::None) {
            throw std::logic_error("materialization source has no published StateImage replica");
        }
        const bool consuming_fork =
            source_state != nullptr &&
            details.source_mode == runtime::PrivateSourceMode::ConsumeToActive &&
            details.state_fork_required;
        if (residency == StateReplicaResidency::HostOnly) {
            host_state_restore = state;
            if (state_count == 0) {
                throw std::logic_error("Host StateImage restore has no Device reservation");
            }
            --state_count;
            if (details.source_mode == runtime::PrivateSourceMode::Retain || consuming_fork) {
                std::optional<StateImageHandle> destination =
                    state_store->reserve_logical_destination();
                if (!destination) { throw std::bad_alloc(); }
                if (consuming_fork) {
                    transaction.state_fork_destination = *destination;
                } else {
                    transaction.reserved_states[transaction.reserved_state_count++] = *destination;
                }
                host_state_fork_destination = *destination;
            }
        } else if (consuming_fork) {
            if (state_count == 0) {
                throw std::logic_error("StateImage Fork has no Device reservation");
            }
            --state_count;
            transaction.state_fork_destination = state_store->reserve_destination();
            if (!transaction.state_fork_destination) { throw std::bad_alloc(); }
        } else if (source_state != nullptr &&
                   details.source_mode == runtime::PrivateSourceMode::Retain &&
                   residency == StateReplicaResidency::Both) {
            if (state_count == 0) {
                throw std::logic_error("Both StateImage split has no active destination");
            }
            --state_count;
            std::optional<StateImageHandle> destination =
                state_store->reserve_logical_destination();
            if (!destination) { throw std::bad_alloc(); }
            transaction.reserved_states[transaction.reserved_state_count++] = *destination;
            transaction.split_state_identity                                = true;
        }
    }
    if (state_count > transaction.reserved_states.size() - transaction.reserved_state_count) {
        throw std::logic_error("materialization state reservation exceeds the active contract");
    }
    for (std::uint32_t index = 0; index < state_count; ++index) {
        std::optional<StateImageHandle> state = state_store->reserve_destination();
        if (!state) { throw std::bad_alloc(); }
        transaction.reserved_states[transaction.reserved_state_count++] = *state;
    }
    if (!transaction.has_source && !transaction.has_shared_source) {
        if (!transaction.root_continuation_index || transaction.root_waiting_for_victim ||
            continuation_slots[*transaction.root_continuation_index].role !=
                ContinuationSlotRole::ReservedMaterialization ||
            transaction.reserved_state_count == 0) {
            throw std::logic_error("root materialization destination is not reserved");
        }
        state_store->activate_reset(transaction.reserved_states[0], device.stream);
    }

    KVAddressSpaceHandle text_address;
    std::optional<KVAddressSpaceHandle> backend_address;
    const bool retained_source = (source_state != nullptr || shared_state != nullptr) &&
                                 details.source_mode == runtime::PrivateSourceMode::Retain;
    if (source_state != nullptr || shared_state != nullptr) {
        const SequenceKVBundle* source_kv = source_state != nullptr
                                                ? (source_state->kv ? &*source_state->kv : nullptr)
                                                : (shared_state->kv ? &*shared_state->kv : nullptr);
        if (source_kv == nullptr) {
            throw std::logic_error("materialization source has no KV address space");
        }
        text_address    = source_kv->text;
        backend_address = source_kv->backend;
        if (retained_source || details.text_prefix_fork_required) {
            transaction.root_text_address = text_kv_addresses->create_inactive();
            if (!transaction.root_text_address) {
                throw std::logic_error("Text KV prefix-fork destination is unavailable");
            }
        }
        if (backend_address && (retained_source || details.backend_prefix_fork_required)) {
            transaction.root_backend_address = backend_kv_addresses->create_inactive();
            if (!transaction.root_backend_address) {
                throw std::logic_error("Backend KV prefix-fork destination is unavailable");
            }
        }
    } else {
        transaction.root_text_address = text_kv_addresses->create_inactive();
        if (!transaction.root_text_address) {
            throw std::logic_error("root Text KV address descriptor is unavailable");
        }
        text_address = *transaction.root_text_address;
        if (details.backend_kv_page_entitlement != 0) {
            if (!backend_kv_addresses) {
                throw std::logic_error("root Backend KV store is unavailable");
            }
            transaction.root_backend_address = backend_kv_addresses->create_inactive();
            if (!transaction.root_backend_address) {
                throw std::logic_error("root Backend KV address descriptor is unavailable");
            }
            backend_address = *transaction.root_backend_address;
        }
    }
    if (details.text_kv_page_entitlement == 0 ||
        backend_address.has_value() != (details.backend_kv_page_entitlement != 0)) {
        throw std::logic_error("materialization KV addresses do not match their entitlements");
    }

    if (source_state != nullptr || shared_state != nullptr) {
        transaction.text_activation_frontier = details.reuse_base;
        if (backend_address) {
            transaction.backend_activation_frontier =
                speculative_backend == SpeculativeBackend::Mtp && details.reuse_base != 0
                    ? details.reuse_base - 1U
                    : details.reuse_base;
        }
    }

    const bool text_prefix_fork =
        (source_state != nullptr || shared_state != nullptr) && details.text_prefix_fork_required;
    const bool backend_prefix_fork = (source_state != nullptr || shared_state != nullptr) &&
                                     details.backend_prefix_fork_required;
    if (text_prefix_fork) {
        transaction.text_source_restore_reservation.emplace(
            text_kv_pages->physical_pool().make_empty_reservation());
    } else {
        const KVAddressSpaceHandle activation_address =
            retained_source ? *transaction.root_text_address : text_address;
        transaction.text_activation.emplace(text_kv_addresses->prepare_activation(
            activation_address, details.text_kv_page_entitlement, static_cast<std::int32_t>(lane),
            transaction.text_activation_frontier));
    }
    if (backend_address && backend_prefix_fork) {
        transaction.backend_source_restore_reservation.emplace(
            backend_kv_pages->physical_pool().make_empty_reservation());
    } else if (backend_address) {
        const KVAddressSpaceHandle activation_address =
            retained_source ? *transaction.root_backend_address : *backend_address;
        transaction.backend_activation.emplace(backend_kv_addresses->prepare_activation(
            activation_address, details.backend_kv_page_entitlement,
            static_cast<std::int32_t>(lane), transaction.backend_activation_frontier));
    }

    const auto prepare_kv_restores =
        [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages, KVAddressSpaceHandle address,
            std::optional<std::uint32_t> activation_frontier, bool source_reservation,
            DeviceKVPageReservation& reservation,
            std::vector<MaterializationTransaction::KVRestorePage>& restores,
            std::vector<DeviceKVPageHandle>& destinations) {
            const std::uint32_t mapped = activation_frontier
                                             ? kv_pages_for_frontier(*activation_frontier)
                                             : addresses.mapped_pages(address);
            if (mapped > addresses.mapped_pages(address)) {
                throw std::logic_error("KV activation frontier exceeds address membership");
            }
            std::uint32_t missing = 0;
            for (std::uint32_t page = 0; page < mapped; ++page) {
                if (!pages.device_resident(addresses.logical_page(address, page))) { ++missing; }
            }
            if (source_reservation) {
                pages.physical_pool().resize_reservation(reservation, missing);
            }
            for (std::uint32_t page = 0; page < mapped; ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.device_resident(logical)) { continue; }
                if (!pages.host_resident(logical) || !host_kv_extents) {
                    throw std::logic_error("checkpoint KV page has no restorable replica");
                }
                const HostKVPageReplica replica = pages.host_replica(logical);
                const DeviceKVPageHandle destination =
                    pages.reserve_device_replica(logical, reservation);
                restores.push_back(MaterializationTransaction::KVRestorePage{
                    .logical     = logical,
                    .extent      = replica.extent,
                    .extent_page = replica.page_offset,
                });
                destinations.push_back(destination);
            }
        };
    DeviceKVPageReservation& text_restore_reservation =
        text_prefix_fork ? *transaction.text_source_restore_reservation
                         : text_kv_addresses->page_reservation(*transaction.text_activation);
    prepare_kv_restores(*text_kv_addresses, *text_kv_pages, text_address,
                        transaction.text_activation_frontier, text_prefix_fork,
                        text_restore_reservation, transaction.text_restores,
                        transaction.text_restore_destinations);
    if (backend_address) {
        DeviceKVPageReservation& backend_restore_reservation =
            backend_prefix_fork
                ? *transaction.backend_source_restore_reservation
                : backend_kv_addresses->page_reservation(*transaction.backend_activation);
        prepare_kv_restores(*backend_kv_addresses, *backend_kv_pages, *backend_address,
                            transaction.backend_activation_frontier, backend_prefix_fork,
                            backend_restore_reservation, transaction.backend_restores,
                            transaction.backend_restore_destinations);
    }
    if (host_state_restore) {
        start_context_transfer_timer(runtime::ContextResourceClass::State);
        std::optional<StateImageTransfer> restore =
            host_state_fork_destination
                ? state_store->begin_host_fork(*host_state_restore, *host_state_fork_destination,
                                               device.transfer_stream)
                : state_store->begin_host_to_device(*host_state_restore, device.transfer_stream);
        if (!restore) { throw std::bad_alloc(); }
        transaction.state_restore.emplace(std::move(*restore));
        stop_context_transfer_timer(runtime::ContextResourceClass::State);
        transaction.transfer_timer_mask |=
            1U << context_resource_index(runtime::ContextResourceClass::State);
    }
    transaction.prepared = true;
    requests[lane].prefill->elapsed_seconds +=
        std::chrono::duration<double>(Clock::now() - prepare_started).count();
}

void ProgramImpl::prepare_prefix_forks(MaterializationTransaction& transaction) {
    if (!transaction.plan || transaction.plan->impl_ == nullptr ||
        (transaction.has_source == transaction.has_shared_source) ||
        (transaction.has_source && transaction.source_index >= continuation_capacity) ||
        (transaction.has_shared_source &&
         transaction.shared_source_index >= shared_prefix_capacity) ||
        transaction.prefix_forks_ready || transaction.prefix_tail_submitted) {
        throw std::logic_error("prefix fork preparation is invalid");
    }
    const AdmissionCandidateImpl& details = *transaction.plan->impl_;
    if ((!details.text_prefix_fork_required && !details.backend_prefix_fork_required) ||
        (details.text_prefix_fork_required &&
         (!transaction.root_text_address || transaction.text_prefix_fork)) ||
        (details.backend_prefix_fork_required &&
         (!transaction.root_backend_address || transaction.backend_prefix_fork))) {
        throw std::logic_error("planned prefix fork destinations are incomplete");
    }
    const SequenceKVBundle* source_kv =
        transaction.has_source ? (continuation_states[transaction.source_index].kv
                                      ? &*continuation_states[transaction.source_index].kv
                                      : nullptr)
                               : (shared_prefix_states[transaction.shared_source_index].kv
                                      ? &*shared_prefix_states[transaction.shared_source_index].kv
                                      : nullptr);
    if (source_kv == nullptr || !transaction.text_activation_frontier) {
        throw std::logic_error("prefix fork source is incomplete");
    }
    if (transaction.text_source_restore_reservation &&
        transaction.text_source_restore_reservation->pages() != 0) {
        throw std::logic_error("retained Text KV restores are incomplete");
    }
    if (transaction.backend_source_restore_reservation &&
        transaction.backend_source_restore_reservation->pages() != 0) {
        throw std::logic_error("retained Backend KV restores are incomplete");
    }
    const auto prepare_retained_tail_backup = [&](KVAddressSpaceStore& addresses,
                                                  LogicalKVPageStore& pages,
                                                  KVPrefixForkReservation& fork, bool staged,
                                                  std::optional<LogicalKVPageHandle>& retained_tail,
                                                  std::optional<HostKVExtentReservation>& backup) {
        if (!staged) { return; }
        const LogicalKVPageHandle tail = addresses.prefix_fork_tail_logical_source(fork);
        if (pages.address_references(tail) != 1 || !pages.device_resident(tail) ||
            pages.writer_references(tail) != 0) {
            throw std::logic_error("retained KV tail changed before staged release");
        }
        retained_tail = tail;
        if (pages.host_resident(tail)) { return; }
        if (host_kv_extents == nullptr) {
            throw std::logic_error("retained KV tail has no Host extent store");
        }
        const std::array membership{tail};
        std::optional<HostKVExtentReservation> reserved =
            host_kv_extents->prepare(pages, membership);
        if (!reserved) { throw std::bad_alloc(); }
        backup.emplace(std::move(*reserved));
    };
    bool copied_tail = false;
    if (details.text_prefix_fork_required) {
        transaction.text_source_restore_reservation.reset();
        transaction.text_prefix_fork.emplace(text_kv_addresses->prepare_prefix_fork(
            source_kv->text, *transaction.root_text_address, *transaction.text_activation_frontier,
            details.text_kv_page_entitlement,
            static_cast<std::int32_t>(transaction.destination.value),
            details.text_retained_tail_release));
        prepare_retained_tail_backup(
            *text_kv_addresses, *text_kv_pages, *transaction.text_prefix_fork,
            details.text_retained_tail_release, transaction.text_retained_tail,
            transaction.text_retained_tail_backup);
        if (*transaction.text_activation_frontier % static_cast<std::uint32_t>(kPagedKVPageSize) !=
            0) {
            start_context_transfer_timer(runtime::ContextResourceClass::MainKV);
            text_kv_pages->physical_pool().copy_page(
                text_kv_addresses->prefix_fork_tail_source(*transaction.text_prefix_fork),
                text_kv_addresses->prefix_fork_tail_destination(*transaction.text_prefix_fork),
                device.transfer_stream);
            stop_context_transfer_timer(runtime::ContextResourceClass::MainKV);
            transaction.transfer_timer_mask |=
                1U << context_resource_index(runtime::ContextResourceClass::MainKV);
            ++transaction.operations.partial_tail_cow_pages;
            copied_tail = true;
        }
    }

    if (details.backend_prefix_fork_required) {
        if (!transaction.root_backend_address || !transaction.backend_activation_frontier) {
            throw std::logic_error("Backend KV prefix-fork destination is incomplete");
        }
        if (!source_kv->backend) {
            throw std::logic_error("Backend KV prefix-fork source is unavailable");
        }
        transaction.backend_source_restore_reservation.reset();
        transaction.backend_prefix_fork.emplace(backend_kv_addresses->prepare_prefix_fork(
            *source_kv->backend, *transaction.root_backend_address,
            *transaction.backend_activation_frontier, details.backend_kv_page_entitlement,
            static_cast<std::int32_t>(transaction.destination.value),
            details.backend_retained_tail_release));
        prepare_retained_tail_backup(
            *backend_kv_addresses, *backend_kv_pages, *transaction.backend_prefix_fork,
            details.backend_retained_tail_release, transaction.backend_retained_tail,
            transaction.backend_retained_tail_backup);
        if (*transaction.backend_activation_frontier %
                static_cast<std::uint32_t>(kPagedKVPageSize) !=
            0) {
            start_context_transfer_timer(runtime::ContextResourceClass::BackendKV);
            backend_kv_pages->physical_pool().copy_page(
                backend_kv_addresses->prefix_fork_tail_source(*transaction.backend_prefix_fork),
                backend_kv_addresses->prefix_fork_tail_destination(
                    *transaction.backend_prefix_fork),
                device.transfer_stream);
            stop_context_transfer_timer(runtime::ContextResourceClass::BackendKV);
            transaction.transfer_timer_mask |=
                1U << context_resource_index(runtime::ContextResourceClass::BackendKV);
            ++transaction.operations.partial_tail_cow_pages;
            copied_tail = true;
        }
    }

    if (copied_tail) {
        context_completion_.record(device.transfer_stream);
        transaction.prefix_tail_submitted = true;
        transaction.transfer_submitted    = true;
    } else {
        transaction.prefix_forks_ready = true;
    }
}

void ProgramImpl::enqueue_materialization_transfers(MaterializationTransaction& transaction) {
    if (!transaction.prepared || transaction.transfer_submitted) {
        throw std::logic_error("materialization transfer batch is not enqueueable");
    }
    const auto enqueue_kv =
        [&](LogicalKVPageStore& pages,
            const std::vector<MaterializationTransaction::KVRestorePage>& restores,
            const std::vector<DeviceKVPageHandle>& destinations,
            runtime::ContextResourceClass resource) {
            if (restores.size() != destinations.size()) {
                throw std::logic_error("KV restore bookkeeping is not row aligned");
            }
            if (restores.empty()) { return; }
            start_context_transfer_timer(resource);
            std::size_t begin = 0;
            while (begin < restores.size()) {
                std::size_t end = begin + 1;
                while (end < restores.size() && restores[end].extent == restores[begin].extent &&
                       restores[end].extent_page == restores[end - 1].extent_page + 1U) {
                    ++end;
                }
                const HostKVAllocationConstView source =
                    host_kv_extents->view(restores[begin].extent)
                        .subview(restores[begin].extent_page,
                                 static_cast<std::uint32_t>(end - begin));
                pages.physical_pool().copy_from_host(
                    source,
                    std::span<const DeviceKVPageHandle>(destinations.data() + begin, end - begin),
                    device.transfer_stream);
                begin = end;
            }
            stop_context_transfer_timer(resource);
            transaction.transfer_timer_mask |= 1U << context_resource_index(resource);
        };
    enqueue_kv(*text_kv_pages, transaction.text_restores, transaction.text_restore_destinations,
               runtime::ContextResourceClass::MainKV);
    if (!transaction.backend_restores.empty()) {
        enqueue_kv(*backend_kv_pages, transaction.backend_restores,
                   transaction.backend_restore_destinations,
                   runtime::ContextResourceClass::BackendKV);
    }
    const bool any = transaction.state_restore.has_value() || !transaction.text_restores.empty() ||
                     !transaction.backend_restores.empty();
    if (any) {
        context_completion_.record(device.transfer_stream);
        transaction.transfer_submitted = true;
    } else if (transaction.plan && transaction.plan->impl_ &&
               (transaction.plan->impl_->text_prefix_fork_required ||
                transaction.plan->impl_->backend_prefix_fork_required)) {
        prepare_prefix_forks(transaction);
    }
}

void ProgramImpl::record_materialization_transfer_observations(
    MaterializationTransaction& transaction) {
    if (!transaction.transfer_submitted || !context_completion_.ready()) {
        throw std::logic_error("materialization transfer observation is not complete");
    }
    const auto record = [&](runtime::ContextResourceClass resource,
                            runtime::ContextTransferDirection direction, TransferWork transfer_work,
                            std::uint32_t pages) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1U << context_resource_index(resource));
        if ((transaction.transfer_timer_mask & bit) == 0) { return; }
        transaction.transfer_observations.push_back(
            context_transfer_observation(resource, direction, transfer_work, pages));
        transaction.transfer_timer_mask &= static_cast<std::uint8_t>(~bit);
    };
    const auto host_layout = [](const LogicalKVPageStore& pages) {
        return plan_host_kv_page_layout(pages.physical_pool().geometry());
    };
    const auto restore_copy_runs = [](const auto& restores, const auto& destinations,
                                      const LogicalKVPageStore& pages) {
        if (restores.size() != destinations.size()) {
            throw std::logic_error("KV restore observation is not row aligned");
        }
        std::uint32_t runs = 0;
        std::size_t begin  = 0;
        while (begin < restores.size()) {
            std::size_t end = begin + 1U;
            while (end < restores.size() && restores[end].extent == restores[begin].extent &&
                   restores[end].extent_page == restores[end - 1U].extent_page + 1U) {
                ++end;
            }
            runs += pages.physical_pool().contiguous_run_count(
                std::span<const DeviceKVPageHandle>(destinations.data() + begin, end - begin));
            begin = end;
        }
        return runs;
    };
    if (transaction.prefix_tail_submitted) {
        if (transaction.text_prefix_fork && transaction.text_prefix_fork->needs_tail_copy()) {
            record(runtime::ContextResourceClass::MainKV,
                   runtime::ContextTransferDirection::DeviceToDevice,
                   plan_device_kv_copy_work(host_layout(*text_kv_pages), 1), 1);
        }
        if (backend_kv_pages && transaction.backend_prefix_fork &&
            transaction.backend_prefix_fork->needs_tail_copy()) {
            record(runtime::ContextResourceClass::BackendKV,
                   runtime::ContextTransferDirection::DeviceToDevice,
                   plan_device_kv_copy_work(host_layout(*backend_kv_pages), 1), 1);
        }
        return;
    }
    if (transaction.retained_tail_backup_submitted) {
        if (transaction.text_retained_tail_backup) {
            record(runtime::ContextResourceClass::MainKV,
                   runtime::ContextTransferDirection::DeviceToHost,
                   plan_host_kv_transfer_work(host_layout(*text_kv_pages), 1, 1), 1);
        }
        if (backend_kv_pages && transaction.backend_retained_tail_backup) {
            record(runtime::ContextResourceClass::BackendKV,
                   runtime::ContextTransferDirection::DeviceToHost,
                   plan_host_kv_transfer_work(host_layout(*backend_kv_pages), 1, 1), 1);
        }
        return;
    }
    if (transaction.state_restore) {
        record(runtime::ContextResourceClass::State,
               runtime::ContextTransferDirection::HostToDevice,
               state_image_transfer_work(host_state_images->layout()), 0);
    }
    record(runtime::ContextResourceClass::MainKV, runtime::ContextTransferDirection::HostToDevice,
           plan_host_kv_transfer_work(host_layout(*text_kv_pages),
                                      static_cast<std::uint32_t>(transaction.text_restores.size()),
                                      restore_copy_runs(transaction.text_restores,
                                                        transaction.text_restore_destinations,
                                                        *text_kv_pages)),
           static_cast<std::uint32_t>(transaction.text_restores.size()));
    if (backend_kv_pages) {
        record(runtime::ContextResourceClass::BackendKV,
               runtime::ContextTransferDirection::HostToDevice,
               plan_host_kv_transfer_work(
                   host_layout(*backend_kv_pages),
                   static_cast<std::uint32_t>(transaction.backend_restores.size()),
                   restore_copy_runs(transaction.backend_restores,
                                     transaction.backend_restore_destinations, *backend_kv_pages)),
               static_cast<std::uint32_t>(transaction.backend_restores.size()));
    }
}

void ProgramImpl::publish_materialization_transfers(MaterializationTransaction& transaction) {
    record_materialization_transfer_observations(transaction);
    const auto enqueue_retained_tail_backups = [&]() {
        bool submitted     = false;
        const auto enqueue = [&](LogicalKVPageStore& pages,
                                 std::optional<HostKVExtentReservation>& backup,
                                 runtime::ContextResourceClass resource) {
            if (!backup) { return; }
            if (host_kv_extents == nullptr || host_kv_extents->page_count(*backup) != 1) {
                throw std::logic_error("retained KV tail Host reservation changed");
            }
            std::array<DeviceKVPageHandle, 1> source{};
            host_kv_extents->device_sources(*backup, source);
            start_context_transfer_timer(resource);
            pages.physical_pool().copy_to_host(source, host_kv_extents->writable_view(*backup),
                                               device.transfer_stream);
            stop_context_transfer_timer(resource);
            transaction.transfer_timer_mask |= 1U << context_resource_index(resource);
            submitted = true;
        };
        enqueue(*text_kv_pages, transaction.text_retained_tail_backup,
                runtime::ContextResourceClass::MainKV);
        if (backend_kv_pages) {
            enqueue(*backend_kv_pages, transaction.backend_retained_tail_backup,
                    runtime::ContextResourceClass::BackendKV);
        }
        if (submitted) {
            context_completion_.record(device.transfer_stream);
            transaction.retained_tail_backup_submitted = true;
            transaction.transfer_submitted             = true;
        }
        return submitted;
    };
    const auto publish_retained_tail_releases = [&]() {
        if (!transaction.plan || transaction.plan->impl_ == nullptr) {
            throw std::logic_error("retained KV tail release lost its admission plan");
        }
        const AdmissionCandidateImpl& details = *transaction.plan->impl_;
        const auto publish = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                 std::optional<KVPrefixForkReservation>& fork, bool staged,
                                 std::optional<LogicalKVPageHandle>& retained_tail,
                                 std::optional<HostKVExtentReservation>& backup) {
            if (!staged) {
                if (retained_tail || backup) {
                    throw std::logic_error("unstaged KV prefix fork owns a retained tail release");
                }
                return;
            }
            if (!fork || !retained_tail ||
                addresses.prefix_fork_tail_logical_source(*fork) != *retained_tail) {
                throw std::logic_error("staged KV prefix-fork tail identity changed");
            }
            if (backup) {
                if (host_kv_extents == nullptr) {
                    throw std::logic_error("retained KV tail Host store disappeared");
                }
                (void)host_kv_extents->publish(std::move(*backup));
                backup.reset();
            }
            addresses.settle_prefix_fork_tail_source(*fork);
            if (!pages.drop_device_replica(*retained_tail)) {
                throw std::logic_error("retained KV tail Device replica is not releasable");
            }
            addresses.complete_prefix_fork_after_tail_release(*fork);
            retained_tail.reset();
        };
        publish(*text_kv_addresses, *text_kv_pages, transaction.text_prefix_fork,
                details.text_retained_tail_release, transaction.text_retained_tail,
                transaction.text_retained_tail_backup);
        if (details.backend_retained_tail_release) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("staged Backend KV tail store is unavailable");
            }
            publish(*backend_kv_addresses, *backend_kv_pages, transaction.backend_prefix_fork, true,
                    transaction.backend_retained_tail, transaction.backend_retained_tail_backup);
        } else if (transaction.backend_retained_tail || transaction.backend_retained_tail_backup) {
            throw std::logic_error("unstaged Backend KV tail release was prepared");
        }
        transaction.prefix_forks_ready = true;
    };
    if (transaction.prefix_tail_submitted) {
        transaction.prefix_tail_submitted = false;
        transaction.transfer_submitted    = false;
        if (enqueue_retained_tail_backups()) { return; }
        publish_retained_tail_releases();
        return;
    }
    if (transaction.retained_tail_backup_submitted) {
        transaction.retained_tail_backup_submitted = false;
        transaction.transfer_submitted             = false;
        publish_retained_tail_releases();
        return;
    }
    if (transaction.state_restore) {
        state_store->publish_transfer(std::move(*transaction.state_restore), true);
        transaction.state_restore.reset();
        transaction.state_restored = true;
    }
    for (const MaterializationTransaction::KVRestorePage& restore : transaction.text_restores) {
        text_kv_pages->publish_device_replica(restore.logical);
    }
    for (const MaterializationTransaction::KVRestorePage& restore : transaction.backend_restores) {
        backend_kv_pages->publish_device_replica(restore.logical);
    }
    transaction.text_restores.clear();
    transaction.text_restore_destinations.clear();
    transaction.backend_restores.clear();
    transaction.backend_restore_destinations.clear();
    transaction.transfer_submitted = false;
    if (transaction.plan && transaction.plan->impl_ &&
        (transaction.plan->impl_->text_prefix_fork_required ||
         transaction.plan->impl_->backend_prefix_fork_required)) {
        prepare_prefix_forks(transaction);
    }
}

void ProgramImpl::abort_materialization_transfers(
    MaterializationTransaction& transaction) noexcept {
    try {
        if (transaction.transfer_submitted) {
            context_completion_.synchronize();
            record_materialization_transfer_observations(transaction);
        }
        if (transaction.state_restore) {
            state_store->abort_transfer(std::move(*transaction.state_restore));
            transaction.state_restore.reset();
        }
        if (transaction.text_activation || transaction.text_source_restore_reservation) {
            DeviceKVPageReservation& reservation =
                transaction.text_source_restore_reservation
                    ? *transaction.text_source_restore_reservation
                    : text_kv_addresses->page_reservation(*transaction.text_activation);
            for (const MaterializationTransaction::KVRestorePage& restore :
                 transaction.text_restores) {
                text_kv_pages->abort_device_replica(restore.logical, reservation);
            }
        }
        if (transaction.backend_activation || transaction.backend_source_restore_reservation) {
            DeviceKVPageReservation& reservation =
                transaction.backend_source_restore_reservation
                    ? *transaction.backend_source_restore_reservation
                    : backend_kv_addresses->page_reservation(*transaction.backend_activation);
            for (const MaterializationTransaction::KVRestorePage& restore :
                 transaction.backend_restores) {
                backend_kv_pages->abort_device_replica(restore.logical, reservation);
            }
        }
    } catch (...) { std::terminate(); }
    transaction.text_restores.clear();
    transaction.text_restore_destinations.clear();
    transaction.backend_restores.clear();
    transaction.backend_restore_destinations.clear();
    transaction.transfer_timer_mask = 0;
    transaction.transfer_submitted  = false;
}

void ProgramImpl::prepare_pressure_bookkeeping(MaterializationTransaction::PressureWork& work) {
    work.state_changes.clear();
    work.main_kv_changes.clear();
    work.backend_kv_changes.clear();
    if (work.option.evicts_continuation) { return; }

    work.state_changes.resize(work.option.state_changes.size());

    const SequenceState* sequence =
        work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
    const SharedPrefixState* shared =
        work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;
    const SequenceKVBundle* kv = sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                                     : (shared->kv ? &*shared->kv : nullptr);
    if (kv == nullptr) { throw std::logic_error("pressure owner has no KV address space"); }

    const auto prepare =
        [&](KVAddressSpaceStore* addresses, LogicalKVPageStore* pages,
            std::optional<KVAddressSpaceHandle> address,
            std::span<const qwen3_5::detail::PressureKVDecision> actions,
            std::vector<MaterializationTransaction::PressureWork::KVChangeWork>& changes) {
            changes.reserve(actions.size());
            for (const qwen3_5::detail::PressureKVDecision& action : actions) {
                changes.emplace_back();
                MaterializationTransaction::PressureWork::KVChangeWork& change = changes.back();
                if (action.kind == qwen3_5::detail::PressureKVDecisionKind::None) {
                    throw std::logic_error("pressure KV action has no operation kind");
                }
                if (addresses == nullptr || pages == nullptr || !address) {
                    throw std::logic_error("pressure KV action has no typed address space");
                }
                const std::uint32_t mapped = addresses->mapped_pages(*address);
                if (action.page_count == 0 || action.begin_page > mapped ||
                    action.page_count > mapped - action.begin_page) {
                    throw std::logic_error("pressure KV action range is invalid");
                }
                change.pages.reserve(action.page_count);
                for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
                    change.pages.push_back(
                        addresses->logical_page(*address, action.begin_page + offset));
                }
                if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DemoteToHost) {
                    change.sources.resize(action.page_count);
                }
            }
        };
    prepare(text_kv_addresses.get(), text_kv_pages.get(), kv->text, work.option.main_kv_changes,
            work.main_kv_changes);
    prepare(backend_kv_addresses.get(), backend_kv_pages.get(), kv->backend,
            work.option.backend_kv_changes, work.backend_kv_changes);
}

void ProgramImpl::publish_pressure_host_releases(MaterializationTransaction::PressureWork& work) {
    detail::PhysicalDelta delta;
    if (work.option.evicts_continuation || work.completed || work.submitted) { return; }
    const bool valid_owner =
        work.shared_owner ? (work.continuation_index < shared_prefix_capacity &&
                             shared_prefix_slots[work.continuation_index].role ==
                                 SharedPrefixSlotRole::Catalogued &&
                             shared_prefix_slots[work.continuation_index].generation ==
                                 work.continuation_generation &&
                             shared_prefix_states[work.continuation_index].active_references == 0)
                          : (work.continuation_index < continuation_capacity &&
                             continuation_slots[work.continuation_index].role ==
                                 ContinuationSlotRole::Catalogued &&
                             continuation_slots[work.continuation_index].generation ==
                                 work.continuation_generation);
    if (!valid_owner || work.option.shared_owner != work.shared_owner) {
        throw std::logic_error("pressure Host release owner changed before publication");
    }
    SequenceState* sequence =
        work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
    SharedPrefixState* shared =
        work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;

    if (!work.option.dropped_checkpoints.empty() && !work.checkpoint_drop_published) {
        if (sequence == nullptr) {
            throw std::logic_error("checkpoint drop targets a shared pressure owner");
        }
        for (const runtime::CheckpointRef checkpoint : work.option.dropped_checkpoints) {
            publish_checkpoint_drop(*sequence, checkpoint);
        }
        delta.removed =
            checked_resource_sum(delta.removed, work.option.checkpoint_drop_effect.removed);
        delta.added = checked_resource_sum(delta.added, work.option.checkpoint_drop_effect.added);
        work.checkpoint_drop_published = true;
        work.mutation_published        = true;
        const bool pure_drop           = work.option.state_changes.empty() &&
                               work.option.main_kv_changes.empty() &&
                               work.option.backend_kv_changes.empty();
        if (pure_drop) {
            work.committed_delta = delta;
            work.completed       = true;
            return;
        }
    }

    if (work.state_changes.size() != work.option.state_changes.size()) {
        throw std::logic_error("pressure State bookkeeping is not action aligned");
    }
    for (std::size_t index = 0; index < work.option.state_changes.size(); ++index) {
        const qwen3_5::detail::PressureStateDecision action = work.option.state_changes[index];
        auto& change                                        = work.state_changes[index];
        if (!pressure_state_drops_host(action) || change.host_released) { continue; }
        const std::optional<StateImageHandle> state =
            pressure_state_source(action, sequence, shared);
        if (!state || !state_store->drop_host_replica(*state)) {
            throw std::logic_error("pressure Host State duplicate is no longer releasable");
        }
        change.host_released    = true;
        work.mutation_published = true;
        ++delta.removed.host.state_slots;
    }

    const SequenceKVBundle* kv = sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                                     : (shared->kv ? &*shared->kv : nullptr);
    if (kv == nullptr) { throw std::logic_error("pressure Host release owner has no KV bundle"); }
    const auto release_kv = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                KVAddressSpaceHandle address,
                                const qwen3_5::detail::PressureKVDecision& action,
                                MaterializationTransaction::PressureWork::KVChangeWork& change) {
        if (action.kind != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate ||
            change.host_released) {
            return;
        }
        const std::uint32_t mapped = addresses.mapped_pages(address);
        if (action.begin_page > mapped || action.page_count > mapped - action.begin_page ||
            change.pages.size() != action.page_count) {
            throw std::logic_error("pressure Host KV release region changed");
        }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            if (change.pages[offset] !=
                addresses.logical_page(address, action.begin_page + offset)) {
                throw std::logic_error("pressure Host KV release membership changed");
            }
        }
        if (!host_kv_extents || !host_kv_extents->release_page_replicas(pages, change.pages)) {
            throw std::logic_error("pressure Host KV duplicates are no longer releasable");
        }
        const std::size_t page_stride =
            &pages == text_kv_pages.get() ? text_host_kv_page_stride : backend_host_kv_page_stride;
        if (action.page_count != 0 &&
            page_stride > std::numeric_limits<std::size_t>::max() / action.page_count) {
            throw std::overflow_error("pressure Host KV release size overflow");
        }
        const std::size_t bytes = page_stride * static_cast<std::size_t>(action.page_count);
        if (bytes > std::numeric_limits<std::size_t>::max() - delta.removed.host.kv_bytes) {
            throw std::overflow_error("pressure Host KV release sum overflow");
        }
        delta.removed.host.kv_bytes += bytes;
        change.host_released    = true;
        work.mutation_published = true;
    };
    if (work.main_kv_changes.size() != work.option.main_kv_changes.size() ||
        work.backend_kv_changes.size() != work.option.backend_kv_changes.size()) {
        throw std::logic_error("pressure KV bookkeeping is not action aligned");
    }
    for (std::size_t index = 0; index < work.option.main_kv_changes.size(); ++index) {
        release_kv(*text_kv_addresses, *text_kv_pages, kv->text, work.option.main_kv_changes[index],
                   work.main_kv_changes[index]);
    }
    if (!work.option.backend_kv_changes.empty()) {
        if (!kv->backend || !backend_kv_addresses || !backend_kv_pages) {
            throw std::logic_error("pressure Host Backend KV release has no typed store");
        }
        for (std::size_t index = 0; index < work.option.backend_kv_changes.size(); ++index) {
            release_kv(*backend_kv_addresses, *backend_kv_pages, *kv->backend,
                       work.option.backend_kv_changes[index], work.backend_kv_changes[index]);
        }
    }
    work.committed_delta.removed =
        checked_resource_sum(work.committed_delta.removed, delta.removed);
    work.committed_delta.added = checked_resource_sum(work.committed_delta.added, delta.added);
}

void ProgramImpl::prepare_pressure_work(MaterializationTransaction::PressureWork& work,
                                        runtime::ContextResourceClass resource) {
    const bool valid_owner =
        work.shared_owner ? (work.continuation_index < shared_prefix_capacity &&
                             shared_prefix_slots[work.continuation_index].role ==
                                 SharedPrefixSlotRole::Catalogued &&
                             shared_prefix_slots[work.continuation_index].generation ==
                                 work.continuation_generation &&
                             shared_prefix_states[work.continuation_index].active_references == 0)
                          : (work.continuation_index < continuation_capacity &&
                             continuation_slots[work.continuation_index].role ==
                                 ContinuationSlotRole::Catalogued &&
                             continuation_slots[work.continuation_index].generation ==
                                 work.continuation_generation);
    if (work.completed || !valid_owner || work.option.shared_owner != work.shared_owner) {
        throw std::logic_error("pressure work source changed before transfer");
    }
    if (work.option.evicts_continuation) { return; }
    SequenceState* sequence =
        work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
    SharedPrefixState* shared =
        work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;
    if (work.state_changes.size() != work.option.state_changes.size()) {
        throw std::logic_error("pressure State bookkeeping is not action aligned");
    }
    if (resource == runtime::ContextResourceClass::State) {
        for (std::size_t index = 0; index < work.option.state_changes.size(); ++index) {
            const qwen3_5::detail::PressureStateDecision action = work.option.state_changes[index];
            auto& change                                        = work.state_changes[index];
            if (!pressure_state_demotes(action)) { continue; }
            if (change.transfer) {
                throw std::logic_error("pressure State transfer was prepared more than once");
            }
            const std::optional<StateImageHandle> source =
                pressure_state_source(action, sequence, shared);
            if (!source) { throw std::logic_error("pressure State transfer has no source"); }
            std::optional<StateImageTransfer> transfer =
                state_store->begin_device_to_host(*source, device.transfer_stream);
            if (!transfer) { throw std::bad_alloc(); }
            change.transfer.emplace(std::move(*transfer));
        }
    }

    const auto prepare_kv = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                KVAddressSpaceHandle address,
                                const qwen3_5::detail::PressureKVDecision& action,
                                MaterializationTransaction::PressureWork::KVChangeWork& change) {
        if (action.page_count == 0) { return; }
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::None) {
            throw std::logic_error("pressure KV action has no operation kind");
        }
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate &&
            change.host_released) {
            return;
        }
        const std::uint32_t mapped = addresses.mapped_pages(address);
        if (action.begin_page > mapped || action.page_count > mapped - action.begin_page ||
            change.pages.size() != action.page_count) {
            throw std::logic_error("pressure KV region changed before transfer");
        }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            const LogicalKVPageHandle logical =
                addresses.logical_page(address, action.begin_page + offset);
            const bool host_resident = pages.host_resident(logical);
            const bool valid_residency =
                action.kind == qwen3_5::detail::PressureKVDecisionKind::DemoteToHost
                    ? !host_resident
                    : host_resident;
            const bool removes_device =
                action.kind != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate;
            if (!pages.device_resident(logical) || pages.writer_references(logical) != 0 ||
                pages.source_pins(logical) != 0 || !valid_residency ||
                (removes_device && addresses.has_active_reference(logical))) {
                throw std::logic_error("pressure KV replica changed before transfer");
            }
            if (change.pages[offset] != logical) {
                throw std::logic_error("pressure KV membership changed before transfer");
            }
        }
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate) {
            if (!host_kv_extents) { throw std::logic_error("Host KV extent store is unavailable"); }
            if (!host_kv_extents->can_release_page_replicas(pages, change.pages)) {
                throw std::logic_error("pressure Host KV replicas are no longer releasable");
            }
            return;
        }
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropDeviceDuplicate) { return; }
        if (!host_kv_extents) { throw std::logic_error("Host KV extent store is unavailable"); }
        std::optional<HostKVExtentReservation> reserved =
            host_kv_extents->prepare(pages, change.pages);
        if (!reserved) { throw std::bad_alloc(); }
        if (change.sources.size() != change.pages.size()) {
            throw std::logic_error("pressure KV source backing was not prepared");
        }
        host_kv_extents->device_sources(*reserved, change.sources);
        pages.physical_pool().copy_to_host(
            change.sources, host_kv_extents->writable_view(*reserved), device.transfer_stream);
        change.backup.emplace(std::move(*reserved));
    };
    const SequenceKVBundle* kv = sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                                     : (shared->kv ? &*shared->kv : nullptr);
    if (kv == nullptr) { throw std::logic_error("pressure owner has no KV address space"); }
    if (resource == runtime::ContextResourceClass::MainKV) {
        if (work.main_kv_changes.size() != work.option.main_kv_changes.size()) {
            throw std::logic_error("pressure Main KV bookkeeping is not action aligned");
        }
        for (std::size_t index = 0; index < work.option.main_kv_changes.size(); ++index) {
            prepare_kv(*text_kv_addresses, *text_kv_pages, kv->text,
                       work.option.main_kv_changes[index], work.main_kv_changes[index]);
        }
    }
    if (resource == runtime::ContextResourceClass::BackendKV &&
        !work.option.backend_kv_changes.empty()) {
        if (!kv->backend || !backend_kv_addresses || !backend_kv_pages) {
            throw std::logic_error("pressure owner has no Backend KV address space");
        }
        if (work.backend_kv_changes.size() != work.option.backend_kv_changes.size()) {
            throw std::logic_error("pressure Backend KV bookkeeping is not action aligned");
        }
        for (std::size_t index = 0; index < work.option.backend_kv_changes.size(); ++index) {
            prepare_kv(*backend_kv_addresses, *backend_kv_pages, *kv->backend,
                       work.option.backend_kv_changes[index], work.backend_kv_changes[index]);
        }
    }
    work.submitted = std::any_of(work.state_changes.begin(), work.state_changes.end(),
                                 [](const auto& change) { return change.transfer.has_value(); }) ||
                     std::any_of(work.main_kv_changes.begin(), work.main_kv_changes.end(),
                                 [](const auto& change) { return change.backup.has_value(); }) ||
                     std::any_of(work.backend_kv_changes.begin(), work.backend_kv_changes.end(),
                                 [](const auto& change) { return change.backup.has_value(); });
}

void ProgramImpl::publish_pressure_work(MaterializationTransaction::PressureWork& work) noexcept {
    try {
        if (work.option.evicts_continuation || work.completed) { std::terminate(); }
        SequenceState* sequence =
            work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
        SharedPrefixState* shared =
            work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;
        if (work.state_changes.size() != work.option.state_changes.size()) { std::terminate(); }
        for (std::size_t index = 0; index < work.option.state_changes.size(); ++index) {
            const qwen3_5::detail::PressureStateDecision action = work.option.state_changes[index];
            auto& change                                        = work.state_changes[index];
            const std::optional<StateImageHandle> source =
                pressure_state_source(action, sequence, shared);
            if (!source) { std::terminate(); }
            if (change.transfer) {
                state_store->publish_transfer(std::move(*change.transfer), false);
                change.transfer.reset();
                work.mutation_published = true;
            } else if (!change.host_released) {
                if (pressure_state_drops_host(action)
                        ? !state_store->drop_host_replica(*source)
                        : !state_store->drop_device_replica(*source)) {
                    std::terminate();
                }
                work.mutation_published = true;
            }
        }

        const auto publish_kv =
            [&](LogicalKVPageStore& pages, const qwen3_5::detail::PressureKVDecision& action,
                MaterializationTransaction::PressureWork::KVChangeWork& change) {
                if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate) {
                    if (change.host_released) { return; }
                    if (!host_kv_extents || change.backup) { std::terminate(); }
                    if (!host_kv_extents->release_page_replicas(pages, change.pages)) {
                        std::terminate();
                    }
                    work.mutation_published = true;
                    return;
                }
                if (change.backup) {
                    if (!host_kv_extents) { std::terminate(); }
                    (void)host_kv_extents->publish(std::move(*change.backup));
                    change.backup.reset();
                }
                for (const LogicalKVPageHandle page : change.pages) {
                    if (!pages.drop_device_replica(page)) { std::terminate(); }
                }
                work.mutation_published = true;
            };
        if (work.main_kv_changes.size() != work.option.main_kv_changes.size() ||
            work.backend_kv_changes.size() != work.option.backend_kv_changes.size()) {
            std::terminate();
        }
        for (std::size_t index = 0; index < work.option.main_kv_changes.size(); ++index) {
            publish_kv(*text_kv_pages, work.option.main_kv_changes[index],
                       work.main_kv_changes[index]);
            if (work.option.main_kv_changes[index].kind ==
                qwen3_5::detail::PressureKVDecisionKind::DemoteToHost) {
                work.spill_pages += work.main_kv_changes[index].pages.size();
            }
        }
        if (!work.option.backend_kv_changes.empty()) {
            if (!backend_kv_pages) { std::terminate(); }
            for (std::size_t index = 0; index < work.option.backend_kv_changes.size(); ++index) {
                publish_kv(*backend_kv_pages, work.option.backend_kv_changes[index],
                           work.backend_kv_changes[index]);
                if (work.option.backend_kv_changes[index].kind ==
                    qwen3_5::detail::PressureKVDecisionKind::DemoteToHost) {
                    work.spill_pages += work.backend_kv_changes[index].pages.size();
                }
            }
        }
        work.submitted = false;
        work.completed = true;
    } catch (...) { std::terminate(); }
}

void ProgramImpl::abort_pressure_work(MaterializationTransaction::PressureWork& work) noexcept {
    try {
        if (work.completed) { return; }
        for (auto& change : work.state_changes) {
            if (change.transfer) {
                state_store->abort_transfer(std::move(*change.transfer));
                change.transfer.reset();
            }
        }
        for (auto& change : work.main_kv_changes) { change.backup.reset(); }
        for (auto& change : work.backend_kv_changes) { change.backup.reset(); }
        work.state_changes.clear();
        work.main_kv_changes.clear();
        work.backend_kv_changes.clear();
        work.submitted = false;
    } catch (...) { std::terminate(); }
}

ProgramImpl::PhysicalReleaseResult
ProgramImpl::release_materialization_victim(MaterializationTransaction& transaction,
                                            std::size_t position) {
    PhysicalReleaseResult out;
    if (position >= transaction.victim_count || transaction.victim_released[position]) {
        return out;
    }
    const std::uint32_t index      = transaction.victim_indices[position];
    const std::uint64_t generation = transaction.victim_generations[position];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Catalogued ||
        continuation_slots[index].generation != generation) {
        return out;
    }
    if (!can_release_continuation_slot_strict(index)) {
        throw std::logic_error("materialization victim is not strictly releasable");
    }

    out.delta.removed = owner_exclusive_resources(continuation_states[index]);
    release_continuation_slot_strict(index);
    if (transaction.root_waiting_for_victim && transaction.root_continuation_index == index) {
        continuation_slots[index].role      = ContinuationSlotRole::ReservedMaterialization;
        transaction.root_waiting_for_victim = false;
    }
    transaction.victim_released[position] = true;
    out.status                            = runtime::ConsumeStatus::Consumed;
    return out;
}

MaterializationResult
ProgramImpl::progress_materialization_transaction(runtime::CancellationFlagView cancellation) {
    MaterializationResult out;
    MaterializationTransaction* transaction_ptr =
        std::get_if<MaterializationTransaction>(&context_transaction_);
    if (transaction_ptr == nullptr || transaction_ptr->terminal) {
        throw std::logic_error("Program has no progressable context transaction");
    }
    MaterializationTransaction& transaction = *transaction_ptr;
    PressureTransition& pressure_transition = transaction.pressure_transition;
    const auto collect_pressure_operations  = [&](MaterializationTransaction::PressureWork& work) {
        if (work.spill_pages > std::numeric_limits<std::uint64_t>::max() -
                                   transaction.operations.pressure_spill_pages) {
            transaction.operations.pressure_spill_pages = std::numeric_limits<std::uint64_t>::max();
        } else {
            transaction.operations.pressure_spill_pages += work.spill_pages;
        }
        work.spill_pages = 0;
    };
    const auto retain_private_result = [&](auto& result, const SequenceState& state) {
        if (!result.final_summary) {
            throw std::logic_error("private acknowledgement backing was not reserved");
        }
        using Result = std::remove_cvref_t<decltype(result)>;
        if constexpr (std::is_same_v<Result, MaterializationSourceResult>) {
            result.mode = runtime::PrivateSourceMode::Retain;
        } else {
            result.disposition = runtime::VictimDisposition::Retained;
        }
        populate_continuation_summary(state, *result.final_summary);
    };
    const auto evict_private_result = [&](MaterializationVictimResult& result) {
        result.disposition        = runtime::VictimDisposition::Evicted;
        result.pressure_committed = true;
        result.final_summary.reset();
    };
    const auto complete_victim_acknowledgement = [&]() {
        for (std::size_t position = 0; position < transaction.victim_count; ++position) {
            if (transaction.victim_released[position]) { continue; }
            const std::uint32_t index      = transaction.victim_indices[position];
            const std::uint64_t generation = transaction.victim_generations[position];
            if (index >= continuation_capacity ||
                continuation_slots[index].role != ContinuationSlotRole::Catalogued ||
                continuation_slots[index].generation != generation) {
                throw std::logic_error("unmodified pressure claim is unavailable");
            }
            retain_private_result(transaction.pressure_results[position],
                                  continuation_states[index]);
            const MaterializationTransaction::PressureWork& work = transaction.pressure[position];
            transaction.pressure_results[position].pressure_committed = work.mutation_published;
        }
        out.victims = std::move(transaction.pressure_results);
    };
    const auto complete_source_acknowledgement = [&](bool published) {
        if (!transaction.has_source) { return; }
        if (published && transaction.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            out.source.emplace(MaterializationSourceResult{
                .mode = runtime::PrivateSourceMode::ConsumeToActive,
            });
            return;
        }
        if (transaction.source_index >= continuation_capacity ||
            continuation_slots[transaction.source_index].role != ContinuationSlotRole::Catalogued ||
            continuation_slots[transaction.source_index].generation !=
                transaction.source_generation) {
            throw std::logic_error("retained materialization source is unavailable");
        }
        SequenceState& source = continuation_states[transaction.source_index];
        if (!transaction.source_result) {
            throw std::logic_error("materialization source backing was not reserved");
        }
        retain_private_result(*transaction.source_result, source);
        out.source.emplace(std::move(*transaction.source_result));
    };
    const auto complete_shared_source_acknowledgement = [&](bool published) {
        if (!transaction.has_shared_source) { return; }
        if (transaction.shared_source_index >= shared_prefix_capacity ||
            shared_prefix_slots[transaction.shared_source_index].role !=
                SharedPrefixSlotRole::Catalogued ||
            shared_prefix_slots[transaction.shared_source_index].generation !=
                transaction.shared_source_generation) {
            throw std::logic_error("retained materialization shared source is unavailable");
        }
        const SharedPrefixState& source = shared_prefix_states[transaction.shared_source_index];
        if (!transaction.shared_source_result) {
            throw std::logic_error("materialization shared-source backing was not reserved");
        }
        transaction.shared_source_result->final_summary = shared_prefix_summary(source);
        out.shared_source.emplace(std::move(*transaction.shared_source_result));
        if (published && out.shared_source->final_summary->active_references == 0) {
            throw std::logic_error("published shared source lost its active reference");
        }
    };
    const auto complete_shared_victim_acknowledgement = [&]() {
        for (std::size_t position = 0; position < transaction.shared_victim_count; ++position) {
            if (transaction.shared_victim_released[position]) { continue; }
            const std::uint32_t index      = transaction.shared_victim_indices[position];
            const std::uint64_t generation = transaction.shared_victim_generations[position];
            if (index >= shared_prefix_capacity ||
                shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
                shared_prefix_slots[index].generation != generation) {
                throw std::logic_error("unmodified shared pressure claim is unavailable");
            }
            transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                .owner              = transaction.shared_pressure_results[position].owner,
                .disposition        = runtime::VictimDisposition::Retained,
                .pressure_committed = transaction.shared_pressure[position].mutation_published,
                .final_summary      = shared_prefix_summary(shared_prefix_states[index]),
            };
        }
        out.shared_victims = std::move(transaction.shared_pressure_results);
    };
    const auto abort_transaction = [&]() {
        release_materialization_staging(transaction);
        transaction.terminal      = true;
        out.status                = runtime::ContextTransactionStatus::Aborted;
        out.transfer_observations = std::move(transaction.transfer_observations);
        out.operations            = transaction.operations;
        complete_source_acknowledgement(false);
        complete_shared_source_acknowledgement(false);
        complete_victim_acknowledgement();
        complete_shared_victim_acknowledgement();
    };

    if (cancellation.requested()) { transaction.cancel_pending = true; }

    if (pressure_transition.phase == PressureTransitionPhase::HostReleases) {
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
        for (std::size_t position = 0; position < transaction.shared_victim_count; ++position) {
            MaterializationTransaction::PressureWork& work = transaction.shared_pressure[position];
            if (work.option.evicts_continuation) {
                const std::uint32_t index      = transaction.shared_victim_indices[position];
                const std::uint64_t generation = transaction.shared_victim_generations[position];
                if (index >= shared_prefix_capacity ||
                    shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
                    shared_prefix_slots[index].generation != generation ||
                    shared_prefix_states[index].active_references != 0) {
                    throw std::logic_error("shared pressure victim changed before release");
                }
                const detail::PhysicalResources exclusive =
                    owner_exclusive_resources(shared_prefix_states[index]);
                if (work.option.effect.added != detail::PhysicalResources{}) {
                    throw std::logic_error("shared pressure eviction changed after reservation");
                }
                if (!can_release_shared_prefix_state(index, SharedPrefixSlotRole::Catalogued)) {
                    throw std::logic_error("shared pressure victim is not strictly releasable");
                }
                const detail::PhysicalResources released =
                    release_shared_prefix_state_strict(index, SharedPrefixSlotRole::Catalogued);
                if (released != exclusive) {
                    throw std::logic_error("shared pressure eviction acknowledgement is invalid");
                }
                work.committed_delta    = detail::PhysicalDelta{.removed = released};
                work.completed          = true;
                work.mutation_published = true;
                transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                    .owner              = transaction.shared_pressure_results[position].owner,
                    .disposition        = runtime::VictimDisposition::Evicted,
                    .pressure_committed = true,
                };
                transaction.shared_victim_released[position] = true;
            } else {
                publish_pressure_host_releases(work);
            }
        }
        for (std::size_t position = 0; position < transaction.victim_count; ++position) {
            MaterializationTransaction::PressureWork& work = transaction.pressure[position];
            if (work.option.evicts_continuation) {
                const PhysicalReleaseResult released =
                    release_materialization_victim(transaction, position);
                if (released.status != runtime::ConsumeStatus::Consumed ||
                    released.delta.added != detail::PhysicalResources{} ||
                    work.option.effect.added != detail::PhysicalResources{}) {
                    throw std::logic_error("materialization eviction changed after reservation");
                }
                work.committed_delta    = released.delta;
                work.completed          = true;
                work.mutation_published = true;
                evict_private_result(transaction.pressure_results[position]);
            } else {
                publish_pressure_host_releases(work);
                if (work.completed) {
                    SequenceState& victim = continuation_states[work.continuation_index];
                    retain_private_result(transaction.pressure_results[position], victim);
                    transaction.pressure_results[position].pressure_committed = true;
                    transaction.victim_released[position]                     = true;
                }
            }
        }
        pressure_transition.phase = PressureTransitionPhase::CopyPreparation;
        if (cancellation.requested()) { transaction.cancel_pending = true; }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
    }

    const auto complete_pressure_delta = [&](MaterializationTransaction::PressureWork& work) {
        (void)checked_resource_difference(work.option.effect.removed, work.committed_delta.removed);
        (void)checked_resource_difference(work.option.effect.added, work.committed_delta.added);
        work.committed_delta = work.option.effect;
    };

    const auto for_each_pending_pressure = [&](auto&& callback) {
        for (MaterializationTransaction::PressureWork& work : transaction.shared_pressure) {
            if (!work.completed) { callback(work); }
        }
        for (MaterializationTransaction::PressureWork& work : transaction.pressure) {
            if (!work.completed) { callback(work); }
        }
    };

    if (pressure_transition.phase == PressureTransitionPhase::CopyPreparation) {
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }

        constexpr std::array pressure_resources{
            runtime::ContextResourceClass::State,
            runtime::ContextResourceClass::MainKV,
            runtime::ContextResourceClass::BackendKV,
        };
        try {
            for (const runtime::ContextResourceClass resource : pressure_resources) {
                bool has_copy = false;
                for_each_pending_pressure(
                    [&](const MaterializationTransaction::PressureWork& work) {
                        has_copy =
                            has_copy ||
                            std::any_of(
                                work.option.transfer_requirements.begin(),
                                work.option.transfer_requirements.end(),
                                [&](const auto& requirement) {
                                    return requirement.resource == resource &&
                                           requirement.direction ==
                                               runtime::ContextTransferDirection::DeviceToHost;
                                });
                    });
                if (has_copy) { start_context_transfer_timer(resource); }
                try {
                    for_each_pending_pressure([&](MaterializationTransaction::PressureWork& work) {
                        prepare_pressure_work(work, resource);
                    });
                } catch (...) {
                    if (has_copy) { stop_context_transfer_timer(resource); }
                    throw;
                }
                if (!has_copy) { continue; }
                stop_context_transfer_timer(resource);
                const std::size_t resource_index = context_resource_index(resource);
                pressure_transition.timer_mask |= static_cast<std::uint8_t>(1U << resource_index);
                for_each_pending_pressure(
                    [&](const MaterializationTransaction::PressureWork& work) {
                        for (const runtime::ContextTransferRequirement& requirement :
                             work.option.transfer_requirements) {
                            if (requirement.resource != resource ||
                                requirement.direction !=
                                    runtime::ContextTransferDirection::DeviceToHost) {
                                continue;
                            }
                            TransferWork& total = pressure_transition.transfer_work[resource_index];
                            total.payload_bytes =
                                requirement.work.payload_bytes >
                                        std::numeric_limits<std::uint64_t>::max() -
                                            total.payload_bytes
                                    ? std::numeric_limits<std::uint64_t>::max()
                                    : total.payload_bytes + requirement.work.payload_bytes;
                            const std::uint64_t operations =
                                static_cast<std::uint64_t>(total.copy_operations) +
                                requirement.work.copy_operations;
                            total.copy_operations =
                                operations > std::numeric_limits<std::uint32_t>::max()
                                    ? std::numeric_limits<std::uint32_t>::max()
                                    : static_cast<std::uint32_t>(operations);
                            const std::uint64_t pages =
                                static_cast<std::uint64_t>(
                                    pressure_transition.transfer_pages[resource_index]) +
                                requirement.page_count;
                            pressure_transition.transfer_pages[resource_index] =
                                pages > std::numeric_limits<std::uint32_t>::max()
                                    ? std::numeric_limits<std::uint32_t>::max()
                                    : static_cast<std::uint32_t>(pages);
                            if (resource == runtime::ContextResourceClass::State) {
                                pressure_transition.state_images =
                                    requirement.units > std::numeric_limits<std::uint64_t>::max() -
                                                            pressure_transition.state_images
                                        ? std::numeric_limits<std::uint64_t>::max()
                                        : pressure_transition.state_images + requirement.units;
                            }
                        }
                    });
            }
        } catch (...) {
            (void)cudaStreamSynchronize(device.transfer_stream);
            for_each_pending_pressure(
                [&](MaterializationTransaction::PressureWork& work) { abort_pressure_work(work); });
            throw;
        }

        bool copies_submitted = false;
        for_each_pending_pressure([&](const MaterializationTransaction::PressureWork& work) {
            copies_submitted = copies_submitted || work.submitted;
        });
        pressure_transition.phase = copies_submitted ? PressureTransitionPhase::CopiesInFlight
                                                     : PressureTransitionPhase::CopyPublication;
        if (copies_submitted) {
            context_completion_.record(device.transfer_stream);
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }

    if (pressure_transition.phase == PressureTransitionPhase::CopiesInFlight) {
        if (!context_completion_.ready()) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
        pressure_transition.phase = PressureTransitionPhase::CopyPublication;
    }
    if (transaction.cancel_pending) {
        // D2H destinations are still private reservations.  Waiting for the stream and aborting
        // them leaves only the already committed PreRelease changes visible.
        abort_transaction();
        return out;
    }

    if (pressure_transition.phase == PressureTransitionPhase::CopyPublication) {
        for (std::size_t position = 0; position < transaction.shared_pressure.size(); ++position) {
            MaterializationTransaction::PressureWork& work = transaction.shared_pressure[position];
            if (work.completed) { continue; }
            publish_pressure_work(work);
            collect_pressure_operations(work);
            const std::uint32_t index = transaction.shared_victim_indices[position];
            transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                .owner              = transaction.shared_pressure_results[position].owner,
                .disposition        = runtime::VictimDisposition::Retained,
                .pressure_committed = true,
                .final_summary      = shared_prefix_summary(shared_prefix_states[index]),
            };
            complete_pressure_delta(work);
            transaction.shared_victim_released[position] = true;
        }
        transaction.shared_pressure_cursor = transaction.shared_pressure.size();

        for (std::size_t position = 0; position < transaction.pressure.size(); ++position) {
            MaterializationTransaction::PressureWork& work = transaction.pressure[position];
            if (work.completed) { continue; }
            publish_pressure_work(work);
            collect_pressure_operations(work);
            retain_private_result(transaction.pressure_results[position],
                                  continuation_states[work.continuation_index]);
            transaction.pressure_results[position].pressure_committed = true;
            complete_pressure_delta(work);
            transaction.victim_released[position] = true;
        }
        transaction.pressure_cursor = transaction.pressure.size();

        constexpr std::array pressure_resources{
            runtime::ContextResourceClass::State,
            runtime::ContextResourceClass::MainKV,
            runtime::ContextResourceClass::BackendKV,
        };
        for (const runtime::ContextResourceClass resource : pressure_resources) {
            const std::size_t index = context_resource_index(resource);
            const std::uint8_t bit  = static_cast<std::uint8_t>(1U << index);
            if ((pressure_transition.timer_mask & bit) == 0) { continue; }
            transaction.transfer_observations.push_back(context_transfer_observation(
                resource, runtime::ContextTransferDirection::DeviceToHost,
                pressure_transition.transfer_work[index], pressure_transition.transfer_pages[index],
                pressure_transition.state_images));
        }
        pressure_transition.timer_mask = 0;
        pressure_transition.phase      = PressureTransitionPhase::Committed;
        if (cancellation.requested()) { transaction.cancel_pending = true; }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
    }
    if (pressure_transition.phase != PressureTransitionPhase::Committed) {
        throw std::logic_error("materialization pressure transition did not reach a stable phase");
    }

    if (!transaction.source_prepared) {
        prepare_consumed_source(transaction);
        if (cancellation.requested()) { transaction.cancel_pending = true; }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
    }

    if (transaction.transfer_submitted) {
        if (!context_completion_.ready()) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
        publish_materialization_transfers(transaction);
        if (transaction.transfer_submitted) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }

    if (transaction.cancel_pending) {
        abort_transaction();
        return out;
    }

    if (!transaction.prepared) {
        prepare_materialization(transaction);
        enqueue_materialization_transfers(transaction);
        if (transaction.transfer_submitted) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }
    if (cancellation.requested()) {
        abort_transaction();
        return out;
    }

    // This is the unique physical publication point. ResourceManager still owns the logical
    // catalog capabilities and adopts them only after validating this terminal result.
    try {
        out.published.emplace(start_request(transaction));
        materialization_ledger_.clear();
        materialization_identity_.clear();
        materialization_prefix_digests_.clear();
    } catch (...) {
        release_materialization_staging(transaction);
        throw;
    }
    transaction.terminal      = true;
    out.status                = runtime::ContextTransactionStatus::Published;
    out.transfer_observations = std::move(transaction.transfer_observations);
    out.operations            = transaction.operations;
    complete_source_acknowledgement(true);
    complete_shared_source_acknowledgement(true);
    complete_victim_acknowledgement();
    complete_shared_victim_acknowledgement();
    return out;
}

ContextTransactionProgress
ProgramImpl::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    const auto terminal_or_pending =
        []<class Result>(Result&& result) -> ContextTransactionProgress {
        if (result.status == runtime::ContextTransactionStatus::InProgress) {
            return runtime::ContextTransactionInProgress{};
        }
        if (result.status != runtime::ContextTransactionStatus::Published &&
            result.status != runtime::ContextTransactionStatus::Aborted) {
            throw std::logic_error("context transaction returned an invalid status");
        }
        return ContextTransactionProgress(std::forward<Result>(result));
    };
    return std::visit(
        [&](auto& transaction) -> ContextTransactionProgress {
            using Transaction = std::decay_t<decltype(transaction)>;
            if constexpr (std::is_same_v<Transaction, std::monostate>) {
                throw std::logic_error("Program has no progressable context transaction");
            } else if constexpr (std::is_same_v<Transaction, MaterializationTransaction>) {
                return terminal_or_pending(progress_materialization_transaction(cancellation));
            } else {
                return terminal_or_pending(progress_active_capture_transaction(cancellation));
            }
        },
        context_transaction_);
}

void ProgramImpl::finalize_context_transaction() noexcept {
    const bool terminal = std::visit(
        [](const auto& transaction) {
            using T = std::decay_t<decltype(transaction)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return false;
            } else if constexpr (std::is_same_v<T, ActiveCaptureTransaction>) {
                return transaction.published;
            } else {
                return transaction.terminal;
            }
        },
        context_transaction_);
    if (terminal) { context_transaction_.emplace<std::monostate>(); }
}

bool ProgramImpl::has_context_transaction() const noexcept {
    return !std::holds_alternative<std::monostate>(context_transaction_);
}


} // namespace ninfer::models::qwen3_5::detail
