#include "models/qwen3_5/session/internal.h"
#include "models/common/context/work.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ninfer::models::qwen3_5::detail {
bool SessionImpl::valid_checkpoint(CheckpointHandle handle) const noexcept {
    return checkpoints.valid(handle);
}

CheckpointState& SessionImpl::checkpoint(CheckpointHandle handle) { return checkpoints.at(handle); }

const CheckpointState& SessionImpl::checkpoint(CheckpointHandle handle) const {
    return checkpoints.at(handle);
}

std::optional<CheckpointHandle> SessionImpl::reserve_checkpoint() { return checkpoints.reserve(); }

bool SessionImpl::release_checkpoint(CheckpointHandle handle) noexcept {
    return checkpoints.erase(handle);
}

bool SessionImpl::can_release_checkpoint(CheckpointHandle handle) const noexcept {
    return checkpoints.erasable(handle);
}

ResumeStateImpl::~ResumeStateImpl() {
    if (snapshot && owner && !owner->release_checkpoint(*snapshot)) { std::terminate(); }
}

bool SessionImpl::revoke_snapshot(ResumeState& paused) noexcept {
    if (!paused.impl_ || !paused.impl_->snapshot) { return false; }
    if (!release_checkpoint(*paused.impl_->snapshot)) { return false; }
    paused.impl_->snapshot.reset();
    return true;
}

PhysicalUsageSnapshot SessionImpl::physical_usage() const noexcept { return directory.usage(); }

PrefixShortlistKey SessionImpl::checkpoint_key(CheckpointHandle handle,
                                               std::uint32_t frontier) const {
    const auto& record = checkpoint(handle);
    if (!frontier || frontier > record.frontier()) {
        throw std::logic_error("checkpoint key is outside coverage");
    }
    return {record.snapshot.identity().digests.at(frontier), frontier, record.key.identity_tag};
}

CheckpointMetadata SessionImpl::checkpoint_metadata(CheckpointHandle handle) const {
    const auto& record = checkpoint(handle);
    return {.frontier = record.frontier(),
            .role     = record.role,
            .leased   = checkpoints[handle.index].pins != 0 ||
                      state_store->source_pins(record.continuation().image) != 0,
            .origin = record.origin};
}

CheckpointSummary SessionImpl::checkpoint_summary(CheckpointHandle handle) const {
    const auto& record = checkpoint(handle);
    CheckpointSummary out{.key = record.key, .frontier = record.frontier(), .role = record.role};
    out.leased = checkpoint_metadata(handle).leased;
    return out;
}

std::size_t SessionImpl::host_bytes_released(std::span<const CheckpointHandle> handles) const {
    auto releases = checkpoint_release_set(handles);
    if (!releases) return 0;
    std::erase_if(releases->records, [](const auto& record) {
        return record.domain.memory != context::ContextMemory::Host;
    });
    return directory.released_bytes(*releases);
}

runtime::CheckpointEvictableResources
SessionImpl::checkpoint_evictable_resources(CheckpointHandle handle) const {
    runtime::CheckpointEvictableResources out;
    if (!can_release_checkpoint(handle)) return out;
    const auto& record = checkpoint(handle);
    const std::array handles{handle};
    const auto released = checkpoint_release_set(handles);
    if (!released) return out;
    out.state_slots = static_cast<std::uint32_t>(released->states.size());
    const std::array<const context::HistoryPrefix*, 1> text_prefixes{&record.text_prefix()};
    out.kv_page_groups[kTextKvPool.index] = static_cast<std::uint32_t>(
        text_kv_histories->released_pages(text_prefixes).size());
    if (record.backend_prefix()) {
        const std::array<const context::HistoryPrefix*, 1> backend_prefixes{
            record.backend_prefix()};
        out.kv_page_groups[kBackendKvPool.index] = static_cast<std::uint32_t>(
            backend_kv_histories->released_pages(backend_prefixes).size());
    }
    out.host_bytes = host_bytes_released(handles);
    return out;
}

std::size_t SessionImpl::state_image_bytes() const noexcept {
    return state_store->image_bytes();
}

std::size_t SessionImpl::kv_page_bytes(std::uint32_t pool) const noexcept {
    return directory.pool(pool).pages().layout().parts.front().record.bytes;
}

std::uint32_t SessionImpl::state_slots_capacity() const noexcept {
    return state_store->capacity();
}

std::optional<context::HostWriteRequirements>
SessionImpl::pause_host_requirements(SequenceHandle handle) const {
    if (context_operation_ || execution || !valid_sequence(handle) || !host_context_enabled) {
        return std::nullopt;
    }
    const auto lane      = this->lane(handle);
    const auto& control  = requests[lane];
    const auto& sequence = sequences[lane];
    if ((control.lifecycle != Lifecycle::Active && control.lifecycle != Lifecycle::Prefilling) ||
        !sequence.kv || !state_store->valid(sequence.continuation.image)) {
        return std::nullopt;
    }
    const auto frontier = control.lifecycle == Lifecycle::Prefilling
                              ? control.prefill->cursor
                              : sequence.continuation.metadata.execution_frontier;
    if (!frontier) { return std::nullopt; }
    if (control.base->vision && !control.base->vision->can_checkpoint(frontier)) return {};
    context::HostWriteRequirements write;
    write.state_images = state_store->host_resident(sequence.continuation.image) ? 0 : 1;
    write.kv_pages.resize(directory.pool_count());
    const auto count = [&](const context::HistoryDirectory& histories,
                           const context::PageDirectory& pages, context::HistoryId address,
                           std::uint32_t target, std::uint32_t& count) {
        const auto mapped   = histories.mapped_pages(address);
        const auto required = context::kv_pages_for_frontier(target);
        for (std::uint32_t i = 0; i < std::min<std::uint32_t>(mapped, required); ++i) {
            const auto page = histories.logical_page(address, i);
            // Deactivation removes this request's active reference. Other resident readers
            // keep their Device replica, so those pages need no new Host allocation.
            const auto columns = static_cast<std::uint32_t>(
                std::min<std::size_t>(kPagedKVPageSize, target - i * kPagedKVPageSize));
            if (!pages.host_covers(page, columns) && pages.active_history_references(page) <= 1) {
                ++count;
            }
        }
        // Masked-draft normalization can append accepted features before snapshot capture.
        if (required > mapped) count += required - mapped;
    };
    count(*text_kv_histories, *text_kv_pages, sequence.kv->text.handle(),
          sequence.continuation.metadata.text_kv_valid, write.kv_pages[kTextKvPool.index]);
    if (sequence.kv->backend) {
        const auto backend = is_masked_draft_backend(speculative_backend)
                                 ? std::max(sequence.continuation.metadata.execution_frontier,
                                            backend_kv_valid(sequence))
                                 : backend_kv_valid(sequence);
        count(*backend_kv_histories, *backend_kv_pages, sequence.kv->backend->handle(), backend,
              write.kv_pages[kBackendKvPool.index]);
    }
    return write;
}

std::size_t SessionImpl::release_redundant_host(std::span<const CheckpointHandle> excluded,
                                                std::optional<SequenceHandle> pending_backup) {
    if (!host_context_enabled) { return 0; }
    std::unordered_set<std::uint32_t> states, main, backend;
    const auto collect = [](const context::HistoryDirectory& histories,
                            const context::PageDirectory& pages, context::HistoryId address,
                            auto& seen) {
        for (std::uint32_t i = 0; i < histories.mapped_pages(address); ++i) {
            seen.insert(pages.descriptor_index(histories.logical_page(address, i)));
        }
    };
    const auto protect = [&](const CheckpointState& record) {
        states.insert(state_store->descriptor_index(record.continuation().image));
        collect(*text_kv_histories, *text_kv_pages, record.history().text, main);
        if (record.history().backend) {
            collect(*backend_kv_histories, *backend_kv_pages, *record.history().backend, backend);
        }
    };
    if (pending_backup && valid_sequence(*pending_backup)) {
        const auto& sequence = sequences[lane(*pending_backup)];
        states.insert(state_store->descriptor_index(sequence.continuation.image));
        if (sequence.kv) {
            collect(*text_kv_histories, *text_kv_pages, sequence.kv->text.handle(), main);
            if (sequence.kv->backend) {
                collect(*backend_kv_histories, *backend_kv_pages, sequence.kv->backend->handle(),
                        backend);
            }
        }
    }
    for (const auto handle : excluded) {
        if (valid_checkpoint(handle)) { protect(checkpoint(handle)); }
    }
    for (const auto& slot : checkpoints) {
        if (slot.value && slot.pins) { protect(*slot.value); }
    }
    const auto select = [&](context::PageDirectory& pages,
                            const context::HistoryDirectory& histories, context::HistoryId address,
                            auto& seen) {
        for (std::uint32_t i = 0; i < histories.mapped_pages(address); ++i) {
            const auto page = histories.logical_page(address, i);
            if (seen.insert(pages.descriptor_index(page)).second && pages.device_resident(page)) {
                (void)pages.drop_host_replica(page, 0);
            }
        }
    };
    const auto before =
        directory.records().pool({0, context::ContextMemory::Host}).usage().live_bytes;
    for (const auto& slot : checkpoints) {
        if (!slot.value) { continue; }
        const auto& record = *slot.value;
        if (states.insert(state_store->descriptor_index(record.continuation().image)).second) {
            (void)state_store->drop_host_replica(record.continuation().image);
        }
        select(*text_kv_pages, *text_kv_histories, record.history().text, main);
        if (record.history().backend) {
            select(*backend_kv_pages, *backend_kv_histories, *record.history().backend, backend);
        }
    }
    return before - directory.records().pool({0, context::ContextMemory::Host}).usage().live_bytes;
}

} // namespace ninfer::models::qwen3_5::detail
