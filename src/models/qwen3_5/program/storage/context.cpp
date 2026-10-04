#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::detail {

bool ProgramImpl::valid_sequence(SequenceHandle handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(handle).value;
    if (lane >= max_concurrency || ContractAccess::epoch(handle) != lane_epochs[lane]) {
        return false;
    }
    if (active_continuations[lane] >= continuation_capacity ||
        continuation_slots[active_continuations[lane]].role != ContinuationSlotRole::Active) {
        return false;
    }
    const Lifecycle lifecycle = requests[lane].lifecycle;
    return lifecycle == Lifecycle::Prefilling || lifecycle == Lifecycle::Active ||
           lifecycle == Lifecycle::Pending || lifecycle == Lifecycle::Finishable;
}

bool ProgramImpl::valid_pending(const PendingBatch& pending) const noexcept {
    if (ContractAccess::owner(pending) != this || !pending_transaction_ ||
        ContractAccess::transaction(pending) != pending_transaction_->id) {
        return false;
    }
    const auto rows = ContractAccess::rows(pending);
    if (rows.size() != pending_transaction_->size) { return false; }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (!valid_sequence(rows[row]) ||
            ContractAccess::lane(rows[row]).value != pending_transaction_->lanes[row] ||
            ContractAccess::epoch(rows[row]) != pending_transaction_->epochs[row] ||
            requests[pending_transaction_->lanes[row]].lifecycle != Lifecycle::Pending) {
            return false;
        }
    }
    return true;
}

void ProgramImpl::invalidate_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    ++lane_epochs[lane];
    if (lane_epochs[lane] == 0) { ++lane_epochs[lane]; }
}

SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

const SequenceState& ProgramImpl::active_sequence(std::uint32_t lane) const {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

std::optional<std::uint32_t> ProgramImpl::allocate_continuation_slot() noexcept {
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role == ContinuationSlotRole::Free) {
            continuation_slots[index].role = ContinuationSlotRole::Active;
            return index;
        }
    }
    return std::nullopt;
}

void ProgramImpl::release_continuation_slot_best_effort(std::uint32_t index) noexcept {
    if (index >= continuation_capacity ||
        continuation_slots[index].role == ContinuationSlotRole::Free) {
        return;
    }
    SequenceState& sequence = continuation_states[index];
    release_sequence_kv(sequence);
    release_sequence_state(sequence);
    retire_continuation_slot(index);
}

void ProgramImpl::retire_continuation_slot(std::uint32_t index) noexcept {
    if (index >= continuation_capacity) { std::terminate(); }
    SequenceState& sequence     = continuation_states[index];
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.rope_delta              = 0;
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.endpoint_valid          = false;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] == index) {
            active_continuations[lane] = continuation_capacity;
        }
    }
    ContinuationSlot& slot = continuation_slots[index];
    slot.role              = ContinuationSlotRole::Free;
    if (++slot.generation == 0) { ++slot.generation; }
}

// One request's entitlement fits the Engine when its KV pages fit the pools; the Device state
// capacity always holds one active slot per lane.
bool ProgramImpl::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    if (base.impl_ == nullptr) { return false; }
    const qwen3_5::PagedKVCache* backend = backend_kv_cache();
    return base.impl_->text_kv_page_entitlement <= decoder->text_kv.page_pool().capacity_pages() &&
           base.impl_->backend_kv_page_entitlement <=
               (backend != nullptr ? backend->page_pool().capacity_pages() : 0U);
}

// Once the blocked head's donors finish, the persistent borrowers' entitlements are the only KV
// pages the pools cannot free or evict: each lane's pages are its shared prefix pages plus its
// private reservation, which together equal its entitlement, and every page no lane maps is free
// or evictable cache. The head is proved admissible beside the borrowers and the candidate from
// its root entitlement, so no cache state is assumed to survive until then.
bool ProgramImpl::persistent_backfill_safe(
    const RequestBasePlan& blocked_head, const HybridAdmissionQuote& candidate,
    std::span<const SequenceHandle> persistent_borrowers) const {
    if (blocked_head.impl_ == nullptr || candidate.impl == nullptr ||
        candidate.readiness != runtime::Readiness::Ready ||
        persistent_borrowers.size() + 2U > max_concurrency) {
        return false;
    }
    std::uint64_t text_pages =
        static_cast<std::uint64_t>(blocked_head.impl_->text_kv_page_entitlement) +
        candidate.impl->text_kv_page_entitlement;
    std::uint64_t backend_pages =
        static_cast<std::uint64_t>(blocked_head.impl_->backend_kv_page_entitlement) +
        candidate.impl->backend_kv_page_entitlement;
    std::uint32_t observed_lanes = 0;
    for (const SequenceHandle sequence : persistent_borrowers) {
        if (!valid_sequence(sequence)) {
            throw std::logic_error("persistent backfill proof contains a stale sequence");
        }
        const std::uint32_t lane = ContractAccess::lane(sequence).value;
        const std::uint32_t bit  = 1U << lane;
        if ((observed_lanes & bit) != 0) {
            throw std::logic_error("persistent backfill proof contains a duplicate sequence");
        }
        observed_lanes |= bit;
        text_pages += requests[lane].text_kv_page_entitlement;
        backend_pages += requests[lane].backend_kv_page_entitlement;
    }
    const qwen3_5::PagedKVCache* backend = backend_kv_cache();
    return text_pages <= decoder->text_kv.page_pool().capacity_pages() &&
           backend_pages <= (backend != nullptr ? backend->page_pool().capacity_pages() : 0U);
}

qwen3_5::PhysicalUsageSnapshot ProgramImpl::physical_usage() const noexcept {
    qwen3_5::PhysicalUsageSnapshot out{.resource_revision = resource_revision_};
    if (state_store) { out.device_state_slots = state_store->device_occupied(); }
    if (text_kv_pages) {
        const DeviceKVPagePool& pool = text_kv_pages->physical_pool();
        out.device_main_kv_pages     = pool.allocated_pages() + pool.reserved_pages();
    }
    if (backend_kv_pages) {
        const DeviceKVPagePool& pool = backend_kv_pages->physical_pool();
        out.device_backend_kv_pages  = pool.allocated_pages() + pool.reserved_pages();
    }
    return out;
}

PrefillProgress ProgramImpl::advance_prefill(SequenceHandle sequence,
                                             runtime::ExecutionTiming* failed_timing) {
    if (pending_transaction_ || !valid_sequence(sequence)) {
        throw std::logic_error("prefill sequence capability is invalid");
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    if (requests[lane].lifecycle != Lifecycle::Prefilling) {
        throw std::logic_error("prefill advance requires a prefilling sequence");
    }
    try {
        runtime::PrefillStepResult step = advance_prefill_raw(lane, failed_timing);
        if (failed_timing != nullptr) { *failed_timing += step.timing; }
        return wrap_prefill(lane, std::move(step));
    } catch (...) {
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

bool ProgramImpl::can_clear_lane_strict(const SequenceState& sequence) const {
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence < begin || &sequence >= end || !state_store || !text_kv_addresses ||
        !text_kv_pages || !sequence.kv) {
        return false;
    }
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    if (continuation_slots[continuation].role != ContinuationSlotRole::Active ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses || !backend_kv_pages ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        return false;
    }

    if (!state_store->valid(sequence.state.read) || !state_store->valid(sequence.state.write) ||
        (sequence.state.fork_pending &&
         (!sequence.state.borrows_read() ||
          !state_store->can_abort_fork(sequence.state.read, sequence.state.write)))) {
        return false;
    }
    enum class ForkEndpoint : std::uint8_t { None, Source, Destination };
    const auto validate_state = [&](StateImageHandle handle, bool release_object,
                                    ForkEndpoint fork_endpoint = ForkEndpoint::None) {
        if (!state_store->valid(handle)) { return false; }
        if (!release_object || state_store->checkpoint_references(handle) != 0) { return true; }
        if (fork_endpoint == ForkEndpoint::Source) {
            return state_store->can_release_source_after_fork_abort(sequence.state.read,
                                                                    sequence.state.write, 0);
        }
        if (fork_endpoint == ForkEndpoint::Destination) {
            return state_store->can_release_destination_after_fork_abort(sequence.state.read,
                                                                         sequence.state.write, 0);
        }
        return state_store->can_release_after_checkpoint_references(handle, 0);
    };

    if (!validate_state(sequence.state.read,
                        !sequence.state.read_has_external_owner() ||
                            sequence.state.read == sequence.state.write,
                        sequence.state.fork_pending ? ForkEndpoint::Source : ForkEndpoint::None)) {
        return false;
    }
    if (sequence.state.write != sequence.state.read &&
        !validate_state(sequence.state.write, true,
                        sequence.state.fork_pending ? ForkEndpoint::Destination
                                                    : ForkEndpoint::None)) {
        return false;
    }
    return true;
}

bool ProgramImpl::clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept {
    try {
        if (!can_clear_lane_strict(sequence)) { return false; }
    } catch (...) { return false; }
    const auto* begin                = continuation_states.data();
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    hybrid_release_lane(sequence.lane);
    release_active_sequence_kv_strict(sequence);
    release_active_sequence_state_strict(sequence);
    retire_continuation_slot(continuation);
    request.retire();
    return true;
}

void ProgramImpl::clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept {
    // A concurrent resource transaction may pin or inspect these active owners. Engine-wide
    // cleanup aborts that transaction before releasing lanes, preserving the only safe order.
    if (has_context_transaction()) { return; }
    for (const std::uint32_t lane : lanes) {
        if (lane >= max_concurrency || active_continuations[lane] >= continuation_capacity) {
            continue;
        }
        clear_lane_best_effort(active_sequence(lane), requests[lane]);
        invalidate_lane(lane);
    }
}

void ProgramImpl::clear_lane_best_effort(SequenceState& sequence,
                                         RequestControl& request) noexcept {
    hybrid_release_lane(sequence.lane);
    request.retire();
    const auto* begin            = continuation_states.data();
    const auto* end              = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        release_continuation_slot_best_effort(static_cast<std::uint32_t>(&sequence - begin));
    }
}

StateImageSelectors ProgramImpl::state_selectors(const SequenceState& sequence) const {
    if (!state_store || !state_store->valid(sequence.state.read) ||
        !state_store->valid(sequence.state.write)) {
        throw std::logic_error("sequence has no active StateImage binding");
    }
    return state_store->selectors(sequence.state.read, sequence.state.write);
}

void ProgramImpl::refresh_state_views(SequenceState& sequence) {
    sequence.tail_hidden = {};
    if (state_store->valid(sequence.state.read) && state_store->valid(sequence.state.write)) {
        const StateImageHandle committed =
            sequence.state.fork_pending ? sequence.state.read : sequence.state.write;
        sequence.tail_hidden =
            state_images->continuation_hidden_slot(state_store->physical_slot(committed));
    }
}

void ProgramImpl::settle_state_fork(SequenceState& sequence) {
    if (!sequence.state.fork_pending) { return; }
    if (has_context_transaction()) {
        throw std::logic_error("StateImage Fork settlement overlaps a resource transaction");
    }
    const StateImageHandle source      = sequence.state.read;
    const StateImageHandle destination = sequence.state.write;
    const bool external_source         = sequence.state.read_has_external_owner();
    state_store->commit_fork(source, destination);
    sequence.state = ActiveStateBinding{.read = destination, .write = destination};
    if (!external_source && state_store->checkpoint_references(source) == 0 &&
        !state_store->release(source)) {
        throw std::logic_error("unreferenced StateImage fork source could not be released");
    }
    refresh_state_views(sequence);
}

void ProgramImpl::release_active_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store) { fail(); }
    try {
        if (sequence.state.fork_pending) {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        }
        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        release_if_unreferenced(sequence.state.write, true);
        if (sequence.state.read != sequence.state.write) {
            release_if_unreferenced(sequence.state.read, !sequence.state.read_has_external_owner());
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.endpoint_valid = false;
    sequence.tail_hidden    = {};
}

void ProgramImpl::release_sequence_state(SequenceState& sequence) noexcept {
    if (!state_store) { return; }
    if (sequence.state.fork_pending && state_store->valid(sequence.state.read) &&
        state_store->valid(sequence.state.write)) {
        try {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        } catch (...) {}
    }
    if (state_store->valid(sequence.state.write)) {
        (void)state_store->release(sequence.state.write);
    }
    if (!sequence.state.read_has_external_owner() && sequence.state.read != sequence.state.write &&
        state_store->valid(sequence.state.read)) {
        (void)state_store->release(sequence.state.read);
    }
    sequence.state          = {};
    sequence.endpoint_valid = false;
    sequence.tail_hidden    = {};
}

qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

const qwen3_5::PagedKVCache* ProgramImpl::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (dflash && dflash->full) { return &*dflash->full; }
    return nullptr;
}

std::uint32_t ProgramImpl::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

void ProgramImpl::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv) { throw std::logic_error("KV allocation bundle is unavailable"); }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    const bool text_active = text_kv_addresses->active(sequence.kv->text);
    const bool backend_active =
        sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend);
    if (sequence.kv->backend && text_active != backend_active) {
        throw std::logic_error("KV address-space activation is not bundle-atomic");
    }
    try {
        if (!text_active) {
            text_kv_addresses->activate(sequence.kv->text,
                                        text_kv_addresses->mapped_pages(sequence.kv->text), row);
            if (sequence.kv->backend) {
                backend_kv_addresses->activate(
                    *sequence.kv->backend,
                    backend_kv_addresses->mapped_pages(*sequence.kv->backend), row);
            }
        }
        set_device_i32(io.text_kv_table_row, text_kv_addresses->bound_row(sequence.kv->text));
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                            : 0);
    } catch (...) {
        if (!text_active) {
            if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
                backend_kv_addresses->deactivate(*sequence.kv->backend);
            }
            if (text_kv_addresses->active(sequence.kv->text)) {
                text_kv_addresses->deactivate(sequence.kv->text);
            }
        }
        throw;
    }
}

void ProgramImpl::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
            backend_kv_addresses->deactivate(*sequence.kv->backend);
        }
    } catch (...) {}
    try {
        if (text_kv_addresses->active(sequence.kv->text)) {
            text_kv_addresses->deactivate(sequence.kv->text);
        }
    } catch (...) {}
}

void ProgramImpl::ensure_sequence_kv_mapped(SequenceState& sequence, std::uint32_t main_tokens,
                                            std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    text_kv_addresses->ensure_mapped_to_tokens(sequence.kv->text, main_tokens, device.stream);
    if (backend_tokens != 0) {
        backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, backend_tokens,
                                                      device.stream);
    }
}

void ProgramImpl::commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                     std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity ||
        (backend_tokens != 0 && !sequence.kv->backend)) {
        throw std::logic_error("KV commit request is outside the sequence bundle");
    }
    text_kv_addresses->commit_frontier(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->commit_frontier(*sequence.kv->backend, backend_tokens);
    }
    if (hybrid_) { hybrid_publish_blocks(sequence); }
}

void ProgramImpl::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                   std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    text_kv_addresses->destructive_truncate(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->destructive_truncate(*sequence.kv->backend, backend_tokens);
    }
}

void ProgramImpl::release_active_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        std::terminate();
    }
    if (sequence.kv->backend &&
        !backend_kv_addresses->release_after_deactivate(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release_after_deactivate(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
}

void ProgramImpl::release_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    unbind_sequence_kv(sequence);
    if (sequence.kv->backend && backend_kv_addresses) {
        (void)backend_kv_addresses->release(*sequence.kv->backend);
    }
    if (text_kv_addresses) { (void)text_kv_addresses->release(sequence.kv->text); }
    sequence.kv.reset();
}

qwen3_5::PagedKVCacheView ProgramImpl::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv || !text_kv_addresses->active(sequence.kv->text)) {
        throw std::logic_error("sequence has no active KV execution mapping");
    }
    return decoder->text_kv.execution_view(text_kv_addresses->execution_row(sequence.kv->text));
}

qwen3_5::PagedKVCacheView ProgramImpl::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend ||
        !backend_kv_addresses->active(*sequence.kv->backend)) {
        throw std::logic_error("sequence has no active MTP KV execution mapping");
    }
    return decoder->mtp_cache()->execution_view(
        backend_kv_addresses->execution_row(*sequence.kv->backend));
}

void ProgramImpl::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImpl::ordered_reset(SequenceState& sequence) {
    if (!state_store->valid(sequence.state.write)) {
        throw std::logic_error("pre-reset StateImage reservation is missing");
    } else {
        if (sequence.state.fork_pending || sequence.state.read != sequence.state.write ||
            state_store->role(sequence.state.write) != StateImageRole::ActiveMutable) {
            throw std::logic_error("StateImage reset requires a private mutable destination");
        }
    }
    refresh_state_views(sequence);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}


} // namespace ninfer::models::qwen3_5::detail
