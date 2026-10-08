#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "core/device.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ninfer::models::qwen3_5::detail {
namespace {

bool checkpoint_releasable(const ProgramImpl& program, CheckpointHandle handle) {
    if (!program.valid_checkpoint(handle) || program.checkpoints[handle.index].pins) {
        return false;
    }
    const auto& record      = program.checkpoint(handle);
    const bool last_history = record.kv.use_count() == 1;
    return (!last_history || (program.text_kv_addresses->can_release(record.kv->text) &&
                              (!record.kv->backend ||
                               program.backend_kv_addresses->can_release(*record.kv->backend)))) &&
           program.state_store->can_release_checkpoint_owner(record.state);
}

bool state_evictable(const StateImageStore& states, StateImageHandle state) {
    return states.role(state) == StateImageRole::CheckpointImmutable &&
           states.can_release_after_checkpoint_references(state,
                                                          states.checkpoint_references(state));
}

bool page_evictable(const LogicalKVPageStore& pages, LogicalKVPageHandle page) {
    return pages.device_resident(page) && pages.active_address_references(page) == 0 &&
           pages.source_pins(page) == 0 && pages.can_pin_source(page);
}

} // namespace

KVHistory::~KVHistory() {
    if (!owner) { return; }
    if (backend && !owner->backend_kv_addresses->release_after_deactivate(*backend)) {
        std::terminate();
    }
    if (!owner->text_kv_addresses->release_after_deactivate(text)) { std::terminate(); }
    if (owner->host_kv_extents) { (void)owner->host_kv_extents->release_unreferenced(); }
}

void ProgramImpl::refresh_history_requirements(const std::shared_ptr<KVHistory>& history,
                                               bool trim_unused) {
    if (!history) { return; }
    std::uint32_t main = 0, backend = 0;
    for (const auto& slot : checkpoints) {
        if (!slot.value || slot.value->kv != history) { continue; }
        main    = std::max(main, slot.value->frontier);
        backend = std::max(backend, slot.value->backend_frontier);
    }
    text_kv_addresses->set_checkpoint_requirement(history->text, main);
    if (trim_unused && main && !text_kv_addresses->active(history->text) &&
        main < text_kv_addresses->committed_frontier(history->text)) {
        text_kv_addresses->truncate_inactive_prefix(history->text, main);
    }
    if (history->backend) {
        backend_kv_addresses->set_checkpoint_requirement(*history->backend, backend);
        if (trim_unused && !backend_kv_addresses->active(*history->backend) &&
            backend < backend_kv_addresses->committed_frontier(*history->backend)) {
            backend_kv_addresses->truncate_inactive_prefix(*history->backend, backend);
        }
    }
    if (trim_unused && host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

bool ProgramImpl::valid_checkpoint(CheckpointHandle handle) const noexcept {
    return handle.owner == this && handle.index < checkpoints.size() &&
           checkpoints[handle.index].generation == handle.generation &&
           checkpoints[handle.index].value.has_value();
}

CheckpointState& ProgramImpl::checkpoint(CheckpointHandle handle) {
    if (!valid_checkpoint(handle)) { throw std::logic_error("checkpoint handle is stale"); }
    return *checkpoints[handle.index].value;
}

const CheckpointState& ProgramImpl::checkpoint(CheckpointHandle handle) const {
    if (!valid_checkpoint(handle)) { throw std::logic_error("checkpoint handle is stale"); }
    return *checkpoints[handle.index].value;
}

std::optional<CheckpointHandle> ProgramImpl::reserve_checkpoint() {
    for (std::uint32_t i = 0; i < checkpoints.size(); ++i) {
        auto& slot = checkpoints[i];
        if (!slot.value && !slot.reserved) {
            slot.reserved = true;
            return CheckpointHandle{this, i, slot.generation};
        }
    }
    return std::nullopt;
}

bool ProgramImpl::release_checkpoint(CheckpointHandle handle) noexcept {
    if (!checkpoint_releasable(*this, handle)) { return false; }
    auto& slot   = checkpoints[handle.index];
    auto& record = *slot.value;
    try {
        // Dropping the whole record precedes releasing any last physical replica.
        const auto state = record.state;
        const auto kv    = record.kv;
        slot.value.reset();
        slot.reserved = false;
        if (++slot.generation == 0) { ++slot.generation; }
        if (!state_store->release_checkpoint_owner(state)) { std::terminate(); }
        refresh_history_requirements(kv, true);
        return true;
    } catch (...) { std::terminate(); }
}

bool ProgramImpl::can_release_checkpoint(CheckpointHandle handle) const noexcept {
    return checkpoint_releasable(*this, handle);
}

ResumeStateImpl::~ResumeStateImpl() {
    if (snapshot && owner && !owner->release_checkpoint(*snapshot)) { std::terminate(); }
}

bool ProgramImpl::revoke_snapshot(ResumeState& paused) noexcept {
    if (!paused.impl_ || !paused.impl_->snapshot) { return false; }
    if (!release_checkpoint(*paused.impl_->snapshot)) { return false; }
    paused.impl_->snapshot.reset();
    return true;
}

PhysicalUsageSnapshot ProgramImpl::physical_usage() const noexcept {
    PhysicalUsageSnapshot out;
    if (state_store) {
        out.occupied.state_slots = state_store->device_occupied();
        out.capacity.state_slots = state_store->device_capacity();
    }
    if (text_kv_pages) {
        const auto& pool           = text_kv_pages->physical_pool();
        out.occupied.main_kv_pages = pool.allocated_pages() + pool.reserved_pages();
        out.capacity.main_kv_pages = pool.capacity_pages();
    }
    if (backend_kv_pages) {
        const auto& pool              = backend_kv_pages->physical_pool();
        out.occupied.backend_kv_pages = pool.allocated_pages() + pool.reserved_pages();
        out.capacity.backend_kv_pages = pool.capacity_pages();
    }
    out.host_state_slots = state_store ? state_store->host_occupied() : 0;
    out.host_kv_bytes    = host_kv_arena ? host_kv_arena->occupied_bytes() : 0;
    if (host_context_arena) {
        out.occupied.host_bytes      = host_context_arena->occupied_bytes();
        out.capacity.host_bytes      = host_context_arena->capacity_bytes();
        out.host_reserved_bytes      = host_context_arena->reserved_bytes();
        out.host_peak_occupied_bytes = host_context_arena->peak_occupied_bytes();
    }
    return out;
}

PrefixShortlistKey ProgramImpl::checkpoint_key(CheckpointHandle handle,
                                               std::uint32_t frontier) const {
    const auto& record = checkpoint(handle);
    if (!frontier || frontier > record.frontier) {
        throw std::logic_error("checkpoint key is outside coverage");
    }
    return {record.identity->digests.at(frontier), frontier, record.key.identity_tag};
}

runtime::ContextResourceUsage
ProgramImpl::checkpoint_footprint(std::span<const CheckpointHandle> handles) const {
    runtime::ContextResourceUsage usage;
    std::vector<std::uint8_t> states(state_store->capacity());
    std::vector<std::uint8_t> main(text_kv_pages->capacity());
    std::vector<std::uint8_t> backend(backend_kv_pages ? backend_kv_pages->capacity() : 0);

    struct Coverage {
        std::uint32_t main    = 0;
        std::uint32_t backend = 0;
    };

    std::unordered_map<const KVHistory*, Coverage> histories;
    const auto count = [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                           KVAddressSpaceHandle address, std::uint32_t frontier,
                           std::vector<std::uint8_t>& seen, std::uint32_t& device_pages) {
        for (std::uint32_t index = 0; index < kv_pages_for_frontier(frontier); ++index) {
            const auto page = addresses.logical_page(address, index);
            auto& visited   = seen[pages.descriptor_index(page)];
            if (visited) { continue; }
            visited = 1;
            device_pages += pages.device_resident(page) ? 1U : 0U;
            if (pages.host_resident(page)) {
                usage.host_bytes += (&pages == text_kv_pages.get() ? text_host_kv_page_stride
                                                                   : backend_host_kv_page_stride);
            }
        }
    };
    for (const auto handle : handles) {
        const auto& record = checkpoint(handle);
        auto& visited      = states[state_store->descriptor_index(record.state)];
        if (!visited) {
            visited = 1;
            usage.state_slots += state_store->device_resident(record.state) ? 1U : 0U;
            if (state_store->host_resident(record.state)) {
                usage.host_bytes += state_images->host_layout().image_bytes;
            }
        }
        auto& coverage   = histories[record.kv.get()];
        coverage.main    = std::max(coverage.main, record.frontier);
        coverage.backend = std::max(coverage.backend, record.backend_frontier);
    }
    for (const auto& [history, coverage] : histories) {
        count(*text_kv_addresses, *text_kv_pages, history->text, coverage.main, main,
              usage.main_kv_pages);
        if (history->backend) {
            count(*backend_kv_addresses, *backend_kv_pages, *history->backend, coverage.backend,
                  backend, usage.backend_kv_pages);
        }
    }
    return usage;
}

CheckpointMetadata ProgramImpl::checkpoint_metadata(CheckpointHandle handle) const {
    const auto& record = checkpoint(handle);
    return {.frontier = record.frontier,
            .role     = record.role,
            .leased =
                checkpoints[handle.index].pins != 0 || state_store->source_pins(record.state) != 0};
}

CheckpointSummary ProgramImpl::checkpoint_summary(CheckpointHandle handle) const {
    const auto& record = checkpoint(handle);
    CheckpointSummary out{.key = record.key, .frontier = record.frontier, .role = record.role};
    const bool unpinned   = checkpoints[handle.index].pins == 0;
    out.leased            = checkpoint_metadata(handle).leased;
    const bool releasable = checkpoint_releasable(*this, handle);
    if (unpinned && state_evictable(*state_store, record.state)) {
        out.evictable_resources.state_slots = state_store->device_resident(record.state) ? 1U : 0U;
    }
    if (releasable && state_store->host_resident(record.state) &&
        state_evictable(*state_store, record.state)) {
        out.evictable_resources.host_bytes += state_images->host_layout().image_bytes;
    }
    const auto scan = [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                          KVAddressSpaceHandle address, std::uint32_t frontier,
                          std::uint32_t& evictable) {
        for (std::uint32_t index = 0; index < kv_pages_for_frontier(frontier); ++index) {
            const auto page = addresses.logical_page(address, index);
            if (unpinned && page_evictable(pages, page)) { ++evictable; }
            if (releasable && pages.host_resident(page) && pages.source_pins(page) == 0) {
                out.evictable_resources.host_bytes +=
                    (&pages == text_kv_pages.get() ? text_host_kv_page_stride
                                                   : backend_host_kv_page_stride);
            }
        }
    };
    scan(*text_kv_addresses, *text_kv_pages, record.kv->text, record.frontier,
         out.evictable_resources.main_kv_pages);
    if (record.kv->backend) {
        scan(*backend_kv_addresses, *backend_kv_pages, *record.kv->backend, record.backend_frontier,
             out.evictable_resources.backend_kv_pages);
    }
    return out;
}

runtime::ContextResourceUsage ProgramImpl::snapshot_resources(const ResumeState& paused) const {
    if (!paused.impl_ || paused.impl_->owner != this || !paused.impl_->snapshot ||
        !valid_checkpoint(*paused.impl_->snapshot)) {
        return {};
    }
    return checkpoint_summary(*paused.impl_->snapshot).evictable_resources;
}

std::size_t ProgramImpl::host_bytes_released(std::span<const CheckpointHandle> handles) const {
    struct StateReferences {
        StateImageHandle state;
        std::uint32_t references = 0;
    };

    struct PageReferences {
        LogicalKVPageHandle page;
        std::uint32_t references = 0;
    };

    std::unordered_set<std::uint32_t> records;
    std::unordered_map<const KVHistory*, std::uint32_t> histories;
    std::unordered_map<std::uint32_t, StateReferences> states;
    std::unordered_map<std::uint32_t, PageReferences> main, backend;
    const auto collect = [](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                            KVAddressSpaceHandle address, std::uint32_t retained_frontier,
                            auto& references) {
        for (std::uint32_t i = kv_pages_for_frontier(retained_frontier);
             i < addresses.mapped_pages(address); ++i) {
            const auto page = addresses.logical_page(address, i);
            auto& entry     = references[pages.descriptor_index(page)];
            entry.page      = page;
            ++entry.references;
        }
    };
    for (const auto handle : handles) {
        if (!checkpoint_releasable(*this, handle)) { return 0; }
        if (!records.insert(handle.index).second) { continue; }
        const auto& record = checkpoint(handle);
        auto& state        = states[state_store->descriptor_index(record.state)];
        state.state        = record.state;
        ++state.references;
        ++histories[record.kv.get()];
    }
    for (const auto& [history, selected] : histories) {
        std::uint32_t owners = 0, retained_main = 0, retained_backend = 0;
        long references = 0;
        for (std::uint32_t index = 0; index < checkpoints.size(); ++index) {
            const auto& slot = checkpoints[index];
            if (!slot.value || slot.value->kv.get() != history) { continue; }
            ++owners;
            references = slot.value->kv.use_count();
            if (!records.contains(index)) {
                retained_main    = std::max(retained_main, slot.value->frontier);
                retained_backend = std::max(retained_backend, slot.value->backend_frontier);
            }
        }
        // Active sequences and in-flight history holders keep their directories independently
        // of optional points. Otherwise deletion also trims the unneeded inactive suffix.
        if (references != owners) { continue; }
        collect(*text_kv_addresses, *text_kv_pages, history->text, retained_main, main);
        if (history->backend) {
            collect(*backend_kv_addresses, *backend_kv_pages, *history->backend, retained_backend,
                    backend);
        }
    }
    std::size_t bytes = 0;
    for (const auto& [index, entry] : states) {
        if (entry.references > state_store->checkpoint_references(entry.state)) { return 0; }
        if (entry.references == state_store->checkpoint_references(entry.state) &&
            state_store->can_release_after_checkpoint_references(entry.state, entry.references) &&
            state_store->host_resident(entry.state)) {
            bytes += state_images->host_layout().image_bytes;
        }
    }
    const auto count = [&](const LogicalKVPageStore& pages, const auto& references) {
        for (const auto& [index, entry] : references) {
            if (entry.references != pages.address_references(entry.page)) { continue; }
            // A nonlast reference may retire during a copy; the complete set cannot release
            // its last source until that copy has finished.
            if (pages.source_pins(entry.page) || pages.active_address_references(entry.page)) {
                return false;
            }
            if (pages.host_resident(entry.page)) {
                bytes += (&pages == text_kv_pages.get() ? text_host_kv_page_stride
                                                        : backend_host_kv_page_stride);
            }
        }
        return true;
    };
    if (!count(*text_kv_pages, main) || (backend_kv_pages && !count(*backend_kv_pages, backend))) {
        return 0;
    }
    return bytes;
}

std::optional<std::size_t> ProgramImpl::pause_host_bytes(SequenceHandle handle) const {
    if (context_transaction_ || pending_transaction_ || !valid_sequence(handle) ||
        !host_context_arena) {
        return std::nullopt;
    }
    const auto lane      = ContractAccess::lane(handle).value;
    const auto& control  = requests[lane];
    const auto& sequence = sequences[lane];
    if ((control.lifecycle != Lifecycle::Active && control.lifecycle != Lifecycle::Prefilling) ||
        !sequence.kv || !state_store->valid(sequence.state.read)) {
        return std::nullopt;
    }
    const auto frontier = control.lifecycle == Lifecycle::Prefilling ? control.prefill->cursor
                                                                     : sequence.execution_frontier;
    if (!frontier) { return std::nullopt; }
    if (control.base->vision_control_plan) {
        for (const auto& item : control.base->vision_control_plan->items) {
            if (item.token_begin < frontier && frontier < item.token_end) { return std::nullopt; }
        }
    }
    std::size_t bytes = state_store->host_resident(sequence.state.read)
                            ? 0
                            : state_images->host_layout().image_bytes;
    const auto count  = [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                           KVAddressSpaceHandle address, std::uint32_t target) {
        const auto mapped   = addresses.mapped_pages(address);
        const auto required = kv_pages_for_frontier(target);
        const auto stride   = (&pages == text_kv_pages.get() ? text_host_kv_page_stride
                                                              : backend_host_kv_page_stride);
        for (std::uint32_t i = 0; i < std::min<std::uint32_t>(mapped, required); ++i) {
            const auto page = addresses.logical_page(address, i);
            // Deactivation removes this request's active reference. Other resident readers
            // keep their Device replica, so those pages need no new Host allocation.
            if (!pages.host_resident(page) && pages.active_address_references(page) <= 1) {
                bytes += stride;
            }
        }
        // Masked-draft normalization can append accepted features before snapshot capture.
        if (required > mapped) { bytes += static_cast<std::size_t>(required - mapped) * stride; }
    };
    count(*text_kv_addresses, *text_kv_pages, sequence.kv->text, sequence.text_kv_valid);
    if (sequence.kv->backend) {
        const auto backend = is_masked_draft_backend(speculative_backend)
                                 ? std::max(sequence.execution_frontier, backend_kv_valid(sequence))
                                 : backend_kv_valid(sequence);
        count(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend, backend);
    }
    return bytes;
}

std::size_t ProgramImpl::release_redundant_host(std::span<const CheckpointHandle> excluded,
                                                std::optional<SequenceHandle> pending_backup) {
    if (!host_context_arena) { return 0; }
    std::unordered_set<std::uint32_t> states, main, backend;
    const auto collect = [](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
                            KVAddressSpaceHandle address, auto& seen) {
        for (std::uint32_t i = 0; i < addresses.mapped_pages(address); ++i) {
            seen.insert(pages.descriptor_index(addresses.logical_page(address, i)));
        }
    };
    const auto protect = [&](const CheckpointState& record) {
        states.insert(state_store->descriptor_index(record.state));
        collect(*text_kv_addresses, *text_kv_pages, record.kv->text, main);
        if (record.kv->backend) {
            collect(*backend_kv_addresses, *backend_kv_pages, *record.kv->backend, backend);
        }
    };
    if (pending_backup && valid_sequence(*pending_backup)) {
        const auto& sequence = sequences[ContractAccess::lane(*pending_backup).value];
        states.insert(state_store->descriptor_index(sequence.state.read));
        if (sequence.kv) {
            collect(*text_kv_addresses, *text_kv_pages, sequence.kv->text, main);
            if (sequence.kv->backend) {
                collect(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend, backend);
            }
        }
    }
    for (const auto handle : excluded) {
        if (valid_checkpoint(handle)) { protect(checkpoint(handle)); }
    }
    for (const auto& slot : checkpoints) {
        if (slot.value && slot.pins) { protect(*slot.value); }
    }
    std::vector<HostKVPageReplicaRelease> releases;
    const auto select = [&](LogicalKVPageStore& pages, const KVAddressSpaceStore& addresses,
                            KVAddressSpaceHandle address, auto& seen) {
        for (std::uint32_t i = 0; i < addresses.mapped_pages(address); ++i) {
            const auto page = addresses.logical_page(address, i);
            if (seen.insert(pages.descriptor_index(page)).second && pages.device_resident(page) &&
                host_kv_extents && host_kv_extents->can_release_page_replica(pages, page)) {
                releases.push_back({&pages, page});
            }
        }
    };
    const auto before = host_context_arena->occupied_bytes();
    for (const auto& slot : checkpoints) {
        if (!slot.value) { continue; }
        const auto& record = *slot.value;
        if (states.insert(state_store->descriptor_index(record.state)).second) {
            (void)state_store->drop_host_replica(record.state);
        }
        select(*text_kv_pages, *text_kv_addresses, record.kv->text, main);
        if (record.kv->backend) {
            select(*backend_kv_pages, *backend_kv_addresses, *record.kv->backend, backend);
        }
    }
    if (!releases.empty() && !host_kv_extents->release_page_replicas(releases)) {
        throw std::logic_error("redundant Host replicas changed during release");
    }
    return before - host_context_arena->occupied_bytes();
}

} // namespace ninfer::models::qwen3_5::detail
