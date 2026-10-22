#include "models/qwen3_5/session/internal.h"
#include "models/common/context/work.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ninfer::models::qwen3_5::detail {
namespace {

bool state_evictable(const context::StateDirectory& states, context::StateId state) {
    return states.role(state) == context::StateRole::CheckpointImmutable &&
           states.can_release_after_checkpoint_references(state,
                                                          states.checkpoint_references(state));
}

bool page_evictable(const context::PageDirectory& pages, context::PageId page) {
    return pages.device_resident(page) && pages.active_history_references(page) == 0 &&
           pages.source_pins(page) == 0 && pages.can_pin_source(page);
}

} // namespace

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

PhysicalUsageSnapshot SessionImpl::physical_usage() const noexcept {
    PhysicalUsageSnapshot out;
    if (state_store) {
        out.occupied.state_slots = state_store->device_occupied();
        out.capacity.state_slots = state_store->device_capacity();
    }
    if (text_kv_pages) {
        const auto& pool = text_kv_pages->allocator();
        out.occupied.kv_page_groups[kTextKvPool.index] =
            pool.allocated_pages() + pool.reserved_pages();
        out.capacity.kv_page_groups[kTextKvPool.index] = pool.capacity_pages();
    }
    if (backend_kv_pages) {
        const auto& pool = backend_kv_pages->allocator();
        out.occupied.kv_page_groups[kBackendKvPool.index] =
            pool.allocated_pages() + pool.reserved_pages();
        out.capacity.kv_page_groups[kBackendKvPool.index] = pool.capacity_pages();
    }
    out.host_state_slots = state_store ? state_store->host_occupied() : 0;
    out.host_kv_bytes    = host_kv_extents ? host_kv_extents->occupied_bytes() : 0;
    if (host_context_arena) {
        out.occupied.host_bytes      = host_context_arena->occupied_bytes();
        out.capacity.host_bytes      = host_context_arena->capacity_bytes();
        out.host_reserved_bytes      = host_context_arena->reserved_bytes();
        out.host_peak_occupied_bytes = host_context_arena->peak_occupied_bytes();
    }
    return out;
}

PrefixShortlistKey SessionImpl::checkpoint_key(CheckpointHandle handle,
                                               std::uint32_t frontier) const {
    const auto& record = checkpoint(handle);
    if (!frontier || frontier > record.frontier()) {
        throw std::logic_error("checkpoint key is outside coverage");
    }
    return {record.snapshot.identity().digests.at(frontier), frontier, record.key.identity_tag};
}

ContextResources
SessionImpl::checkpoint_footprint(std::span<const CheckpointHandle> handles) const {
    ContextResources usage;
    std::vector<bool> states(state_store->capacity()), main(text_kv_pages->capacity()),
        backend(backend_kv_pages ? backend_kv_pages->capacity() : 0);
    const auto count =
        [&](const context::HistoryPrefix& prefix, const context::HistoryDirectory& histories,
            const context::PageDirectory& pages, auto& seen, std::uint32_t& device_pages) {
            for (std::uint32_t i = 0; i < context::kv_pages_for_frontier(prefix.frontier()); ++i) {
                const auto page  = histories.logical_page(prefix.history(), i);
                const auto index = pages.descriptor_index(page);
                if (seen[index]) continue;
                seen[index] = true;
                device_pages += pages.device_resident(page);
                if (pages.host_resident(page))
                    usage.host_bytes += pages.host_page_layout().page_stride;
            }
        };
    for (const auto handle : handles) {
        const auto& record = checkpoint(handle);
        const auto image   = record.continuation().image;
        const auto index   = state_store->descriptor_index(image);
        if (!states[index]) {
            states[index] = true;
            usage.state_slots += state_store->device_resident(image);
            if (state_store->host_resident(image)) usage.host_bytes += state_store->image_bytes();
        }
        count(record.text_prefix(), *text_kv_histories, *text_kv_pages, main,
              usage.kv_page_groups[kTextKvPool.index]);
        if (const auto* prefix = record.backend_prefix())
            count(*prefix, *backend_kv_histories, *backend_kv_pages, backend,
                  usage.kv_page_groups[kBackendKvPool.index]);
    }
    return usage;
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
    const bool unpinned   = checkpoints[handle.index].pins == 0;
    out.leased            = checkpoint_metadata(handle).leased;
    const bool releasable = checkpoints.erasable(handle);
    if (unpinned && state_evictable(*state_store, record.continuation().image)) {
        out.evictable_resources.state_slots =
            state_store->device_resident(record.continuation().image) ? 1U : 0U;
    }
    if (releasable && state_store->host_resident(record.continuation().image) &&
        state_evictable(*state_store, record.continuation().image)) {
        out.evictable_resources.host_bytes += state_store->image_bytes();
    }
    const auto scan = [&](const context::HistoryDirectory& histories,
                          const context::PageDirectory& pages, context::HistoryId address,
                          std::uint32_t frontier, std::uint32_t& evictable) {
        for (std::uint32_t index = 0; index < context::kv_pages_for_frontier(frontier); ++index) {
            const auto page = histories.logical_page(address, index);
            if (unpinned && page_evictable(pages, page)) { ++evictable; }
            if (releasable && pages.host_resident(page) && pages.source_pins(page) == 0) {
                out.evictable_resources.host_bytes += pages.host_page_layout().page_stride;
            }
        }
    };
    scan(*text_kv_histories, *text_kv_pages, record.history().text, record.frontier(),
         out.evictable_resources.kv_page_groups[kTextKvPool.index]);
    if (record.history().backend) {
        scan(*backend_kv_histories, *backend_kv_pages, *record.history().backend,
             record.backend_frontier(),
             out.evictable_resources.kv_page_groups[kBackendKvPool.index]);
    }
    return out;
}

ContextResources SessionImpl::snapshot_resources(const ResumeState& paused) const {
    if (!paused.impl_ || paused.impl_->owner != this || !paused.impl_->snapshot ||
        !valid_checkpoint(*paused.impl_->snapshot)) {
        return {};
    }
    return checkpoint_summary(*paused.impl_->snapshot).evictable_resources;
}

std::size_t SessionImpl::host_bytes_released(std::span<const CheckpointHandle> handles) const {
    std::unordered_set<std::uint32_t> records;

    struct StateOwners {
        context::StateId handle;
        std::uint32_t count = 0;
    };

    std::unordered_map<std::uint32_t, StateOwners> images;
    std::vector<const context::HistoryPrefix*> main, backend;
    for (const auto handle : handles) {
        if (!can_release_checkpoint(handle)) return 0;
        if (!records.insert(handle.index).second) continue;
        const auto& record = checkpoint(handle);
        const auto image   = record.continuation().image;
        auto& entry        = images[state_store->descriptor_index(image)];
        entry.handle       = image;
        ++entry.count;
        main.push_back(&record.text_prefix());
        if (const auto* prefix = record.backend_prefix()) backend.push_back(prefix);
    }
    std::size_t bytes = 0;
    for (const auto& [index, entry] : images)
        if (entry.count == state_store->checkpoint_references(entry.handle) &&
            state_store->can_release_after_checkpoint_references(entry.handle, entry.count) &&
            state_store->host_resident(entry.handle))
            bytes += state_store->image_bytes();
    for (const auto page : text_kv_histories->released_pages(main))
        if (text_kv_pages->host_resident(page))
            bytes += text_kv_pages->host_page_layout().page_stride;
    if (backend_kv_histories)
        for (const auto page : backend_kv_histories->released_pages(backend))
            if (backend_kv_pages->host_resident(page))
                bytes += backend_kv_pages->host_page_layout().page_stride;
    return bytes;
}

std::optional<context::HostWriteRequirements>
SessionImpl::pause_host_requirements(SequenceHandle handle) const {
    if (context_operation_ || execution || !valid_sequence(handle) || !host_context_arena) {
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
    if (control.base->vision_control_plan) {
        for (const auto& item : control.base->vision_control_plan->items) {
            if (item.token_begin < frontier && frontier < item.token_end) { return std::nullopt; }
        }
    }
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
    if (!host_context_arena) { return 0; }
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
    std::vector<context::HostPageRelease> releases;
    const auto select = [&](context::PageDirectory& pages,
                            const context::HistoryDirectory& histories, context::HistoryId address,
                            auto& seen) {
        for (std::uint32_t i = 0; i < histories.mapped_pages(address); ++i) {
            const auto page = histories.logical_page(address, i);
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
        if (states.insert(state_store->descriptor_index(record.continuation().image)).second) {
            (void)state_store->drop_host_replica(record.continuation().image);
        }
        select(*text_kv_pages, *text_kv_histories, record.history().text, main);
        if (record.history().backend) {
            select(*backend_kv_pages, *backend_kv_histories, *record.history().backend, backend);
        }
    }
    if (!releases.empty() && !host_kv_extents->release_page_replicas(releases)) {
        throw std::logic_error("redundant Host replicas changed during release");
    }
    return before - host_context_arena->occupied_bytes();
}

} // namespace ninfer::models::qwen3_5::detail
