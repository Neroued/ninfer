#include "models/common/context/directory/host.h"

#include <algorithm>
#include <exception>

namespace ninfer::models::context {

HostPageReservation::~HostPageReservation() {
    if (owner_ != nullptr) { owner_->abort(*this); }
}

HostDirectory::HostDirectory(HostAllocator& allocator, std::uint32_t descriptor_capacity)
    : allocator_(&allocator), extents_(descriptor_capacity), free_(descriptor_capacity),
      free_count_(descriptor_capacity), memberships_(descriptor_capacity),
      free_memberships_(descriptor_capacity), free_membership_count_(descriptor_capacity),
      release_marks_(descriptor_capacity), extent_marks_(descriptor_capacity) {
    for (std::uint32_t index = 0; index < descriptor_capacity; ++index) {
        free_[index]             = descriptor_capacity - 1U - index;
        free_memberships_[index] = descriptor_capacity - 1U - index;
    }
    affected_extents_.reserve(descriptor_capacity);
    partition_runs_.reserve(descriptor_capacity);
    page_scratch_.reserve(descriptor_capacity);
}

void HostDirectory::restrict_coverage(PageDirectory& pages, PageId handle, std::uint32_t coverage) {
    auto& page = pages.require(handle);
    if (!page.host_replica || page.readers() || page.destination_pinned ||
        coverage > page.host_replica->committed_columns) {
        throw std::logic_error("Host KV coverage cannot be restricted");
    }
    auto& replica = *page.host_replica;
    if (page.host_owner != this || !valid(replica.extent)) {
        throw std::logic_error("Host KV coverage belongs to another store");
    }
    auto& membership = memberships_.at(replica.membership_node);
    if (membership.page != handle || membership.extent != replica.extent.index ||
        membership.offset != replica.page_offset ||
        membership.coverage != replica.committed_columns) {
        throw std::logic_error("Host KV membership changed before restriction");
    }
    membership.coverage       = coverage;
    replica.committed_columns = coverage;
}

const HostKVPageLayout& HostDirectory::page_layout(const PageDirectory& pages) const {
    const HostKVPageLayout& layout = pages.host_page_layout();
    if (layout.page_stride < allocator_->minimum_allocation_bytes()) {
        throw std::invalid_argument("Host KV page geometry is below the allocator minimum");
    }
    return layout;
}

std::optional<HostPageReservation> HostDirectory::prepare(PageDirectory& pages,
                                                          std::span<const PageId> membership) {
    if (membership.empty() || free_count_ == 0 || membership.size() > free_membership_count_) {
        return std::nullopt;
    }
    page_scratch_.clear();
    for (const PageId page : membership) {
        if (!pages.can_pin_source(page) || pages.require(page).host_readers ||
            pages.require(page).host_transfer_pending || pages.host_replica_current(page)) {
            return std::nullopt;
        }
        page_scratch_.push_back(page);
    }
    std::sort(page_scratch_.begin(), page_scratch_.end(),
              [](PageId a, PageId b) { return a.index < b.index; });
    if (std::adjacent_find(page_scratch_.begin(), page_scratch_.end()) != page_scratch_.end()) {
        return std::nullopt;
    }

    const HostKVPageLayout& layout = page_layout(pages);
    if (layout.page_stride > std::numeric_limits<std::size_t>::max() / membership.size()) {
        return std::nullopt;
    }
    std::optional<HostAllocation> allocation =
        allocator_->allocate(layout.page_stride * membership.size());
    if (!allocation) { return std::nullopt; }

    const std::uint32_t descriptor = free_[--free_count_];
    Extent& extent                 = extents_[descriptor];
    if (extent.state != ExtentState::Free) { std::terminate(); }
    extent.state      = ExtentState::Reserved;
    extent.page_store = &pages;
    extent.allocation = std::move(*allocation);
    occupied_bytes_ += extent.allocation.bytes();
    for (const PageId page : membership) {
        const std::uint32_t node = take_membership();
        if (node == kInvalidIndex) { std::terminate(); }
        Membership& entry = memberships_[node];
        entry.page        = page;
        entry.epoch       = pages.content_epoch(page);
        entry.coverage    = pages.required_columns(page);
        entry.extent      = descriptor;
        entry.offset      = extent.page_count;
        entry.next        = kInvalidIndex;
        if (extent.tail == kInvalidIndex) {
            extent.head = node;
        } else {
            memberships_[extent.tail].next = node;
        }
        extent.tail = node;
        ++extent.page_count;
    }
    for (const PageId page : membership) {
        pages.pin_source(page);
        pages.require(page).host_transfer_pending = true;
    }

    HostPageReservation reservation;
    reservation.owner_      = this;
    reservation.descriptor_ = descriptor;
    reservation.generation_ = extent.generation;
    reservation.page_store_ = &pages;
    return reservation;
}

HostPageRange HostDirectory::writable_view(const HostPageReservation& reservation) const {
    validate(reservation);
    const Extent& extent = extents_[reservation.descriptor_];
    return {extent.allocation.extent(), extent.page_count};
}

std::vector<PageSlot> HostDirectory::device_sources(const HostPageReservation& reservation) const {
    validate(reservation);
    std::vector<PageSlot> out;
    out.resize(extents_[reservation.descriptor_].page_count);
    device_sources(reservation, out);
    return out;
}

void HostDirectory::device_sources(const HostPageReservation& reservation,
                                   std::span<PageSlot> out) const {
    validate(reservation);
    const Extent& extent = extents_[reservation.descriptor_];
    if (out.size() != extent.page_count) {
        throw std::invalid_argument("Host KV device-source output has the wrong size");
    }
    std::uint32_t node = extent.head;
    for (std::size_t index = 0; index < out.size(); ++index) {
        if (node == kInvalidIndex) { std::terminate(); }
        out[index] = extent.page_store->physical(memberships_[node].page);
        node       = memberships_[node].next;
    }
    if (node != kInvalidIndex) { std::terminate(); }
}

HostReplicaId HostDirectory::publish(HostPageReservation&& reservation) noexcept {
    if (!valid(reservation)) { std::terminate(); }
    Extent& extent = extents_[reservation.descriptor_];
    page_scratch_.clear();
    begin_release_marks();
    std::uint32_t node = extent.head;
    for (std::uint32_t index = 0; index < extent.page_count; ++index) {
        if (node == kInvalidIndex) { std::terminate(); }
        const Membership& entry = memberships_[node];
        auto& pages             = *extent.page_store;
        if (pages.content_epoch(entry.page) != entry.epoch ||
            !pages.device_covers(entry.page, entry.coverage) ||
            !pages.require(entry.page).host_transfer_pending ||
            pages.require(entry.page).device_readers == 0) {
            std::terminate();
        }
        if (pages.host_resident(entry.page) && !mark_release(pages, entry.page)) {
            std::terminate();
        }
        page_scratch_.push_back(entry.page);
        node = entry.next;
    }
    if (node != kInvalidIndex) { std::terminate(); }
    release_marked_extents();
    extent.allocation.publish();
    extent.state = ExtentState::Published;
    const HostReplicaId capability{allocator_->pool_id(), reservation.descriptor_,
                                   extent.generation};
    node = extent.head;
    for (std::uint32_t index = 0; index < extent.page_count; ++index) {
        Membership& entry = memberships_[node];
        try {
            extent.page_store->attach_host_replica(
                entry.page,
                HostPageReplica{.extent            = capability,
                                .page_offset       = index,
                                .membership_node   = node,
                                .content_epoch     = entry.epoch,
                                .committed_columns = entry.coverage},
                *this);
        } catch (...) { std::terminate(); }
        node = entry.next;
    }
    auto* pages = extent.page_store;
    consume(reservation);
    for (auto page : page_scratch_) {
        pages->require(page).host_transfer_pending = false;
        pages->unpin_source(page);
    }
    return capability;
}

void HostDirectory::abort(HostPageReservation& reservation) noexcept {
    if (!valid(reservation)) {
        consume(reservation);
        return;
    }
    Extent& extent     = extents_[reservation.descriptor_];
    std::uint32_t node = extent.head;
    for (std::uint32_t index = 0; index < extent.page_count; ++index) {
        if (node == kInvalidIndex) { std::terminate(); }
        const PageId page = memberships_[node].page;
        if (!extent.page_store->valid(page) || extent.page_store->source_pins(page) == 0) {
            std::terminate();
        }
        try {
            extent.page_store->require(page).host_transfer_pending = false;
            extent.page_store->unpin_source(page);
        } catch (...) { std::terminate(); }
        node = memberships_[node].next;
    }
    release_descriptor(reservation.descriptor_, extent);
    consume(reservation);
}

bool HostDirectory::can_release_page_replica(PageDirectory& pages, PageId page) const noexcept {
    if (!pages.valid(page) || !pages.host_resident(page) ||
        pages.require(page).host_owner != this || pages.require(page).host_readers != 0 ||
        (!pages.device_resident(page) && pages.history_references(page) != 0)) {
        return false;
    }
    const HostPageReplica replica = pages.host_replica(page);
    if (!valid(replica.extent)) { return false; }
    const Extent& extent     = extents_[replica.extent.index];
    const std::uint32_t node = replica.membership_node;
    if (extent.page_store != &pages || replica.page_offset >= extent.page_count ||
        node >= memberships_.size() || memberships_[node].page != page ||
        memberships_[node].extent != replica.extent.index ||
        memberships_[node].offset != replica.page_offset) {
        return false;
    }
    return true;
}

bool HostDirectory::can_release_page_replicas(
    std::span<const HostPageRelease> releases) const noexcept {
    begin_release_marks();
    for (const HostPageRelease& release : releases) {
        if (release.pages == nullptr || !mark_release(*release.pages, release.page)) {
            return false;
        }
    }
    return true;
}

bool HostDirectory::can_release_page_replicas(PageDirectory& pages,
                                              std::span<const PageId> releases) const noexcept {
    begin_release_marks();
    for (const PageId release : releases) {
        if (!mark_release(pages, release)) { return false; }
    }
    return true;
}

bool HostDirectory::release_page_replicas(std::span<const HostPageRelease> releases) {
    if (!can_release_page_replicas(releases)) { return false; }
    release_marked_extents();
    return true;
}

bool HostDirectory::release_page_replicas(PageDirectory& pages, std::span<const PageId> releases) {
    if (!can_release_page_replicas(pages, releases)) { return false; }
    release_marked_extents();
    return true;
}

void HostDirectory::begin_release_marks() const noexcept {
    affected_extents_.clear();
    ++release_stamp_;
    if (release_stamp_ == 0) {
        std::fill(release_marks_.begin(), release_marks_.end(), 0);
        std::fill(extent_marks_.begin(), extent_marks_.end(), 0);
        release_stamp_ = 1;
    }
}

bool HostDirectory::mark_release(PageDirectory& pages, PageId page) const noexcept {
    if (!can_release_page_replica(pages, page)) { return false; }
    const HostPageReplica replica = pages.host_replica(page);
    if (!valid(replica.extent) || replica.membership_node >= memberships_.size()) { return false; }
    const Membership& membership = memberships_[replica.membership_node];
    if (membership.page != page || membership.extent != replica.extent.index ||
        membership.offset != replica.page_offset ||
        replica.page_offset >= extents_[replica.extent.index].page_count ||
        extents_[replica.extent.index].page_store != &pages) {
        return false;
    }
    if (release_marks_[replica.membership_node] == release_stamp_) { return false; }
    release_marks_[replica.membership_node] = release_stamp_;
    if (extent_marks_[replica.extent.index] != release_stamp_) {
        extent_marks_[replica.extent.index] = release_stamp_;
        affected_extents_.push_back(replica.extent.index);
    }
    return true;
}

void HostDirectory::release_marked_extents() {
    for (const std::uint32_t index : affected_extents_) partition_extent(index);
}

void HostDirectory::partition_extent(std::uint32_t index) {
    if (index >= extents_.size()) {
        throw std::logic_error("Host KV partition extent index is out of range");
    }
    Extent& original = extents_[index];
    if (original.state != ExtentState::Published || original.page_store == nullptr ||
        !original.allocation.valid() || original.page_count == 0) {
        throw std::logic_error("Host KV extent is not partitionable");
    }
    PageDirectory* const pages    = original.page_store;
    const std::size_t page_stride = pages->host_page_layout().page_stride;
    if (original.allocation.bytes() / page_stride != original.page_count ||
        original.allocation.bytes() % page_stride != 0) {
        throw std::logic_error("Host KV allocation does not match its page count");
    }
    const HostReplicaId old{allocator_->pool_id(), index, original.generation};

    partition_runs_.clear();
    std::uint32_t node          = original.head;
    bool any_release            = false;
    std::uint32_t retained_runs = 0;
    for (std::uint32_t offset = 0; offset < original.page_count; ++offset) {
        if (node == kInvalidIndex || node >= memberships_.size()) {
            throw std::logic_error("Host KV extent membership is truncated");
        }
        const Membership& entry = memberships_[node];
        if (!pages->valid(entry.page) || !pages->host_resident(entry.page)) {
            throw std::logic_error("Host KV extent membership is stale");
        }
        const HostPageReplica replica = pages->host_replica(entry.page);
        if (entry.extent != index || entry.offset != offset || replica.extent != old ||
            replica.page_offset != offset || replica.membership_node != node ||
            replica.content_epoch != entry.epoch || replica.committed_columns != entry.coverage) {
            throw std::logic_error("Host KV extent membership disagrees with its replica");
        }
        const bool release = release_marks_[node] == release_stamp_;
        if (release &&
            (pages->require(entry.page).host_readers != 0 ||
             (!pages->device_resident(entry.page) && pages->history_references(entry.page) != 0))) {
            throw std::logic_error("Host KV partition contains an unreleasable page");
        }
        if (partition_runs_.empty() || partition_runs_.back().release != release) {
            partition_runs_.push_back(PartitionRun{
                .begin = offset, .head = node, .tail = node, .count = 1, .release = release});
            if (!release) { ++retained_runs; }
        } else {
            PartitionRun& run = partition_runs_.back();
            run.tail          = node;
            ++run.count;
        }
        if (release) { any_release = true; }
        node = entry.next;
    }
    if (node != kInvalidIndex) {
        throw std::logic_error("Host KV extent membership exceeds its page count");
    }
    if (!any_release) { return; }
    const std::uint32_t additional_extents = retained_runs == 0 ? 0U : retained_runs - 1U;
    if (additional_extents > free_count_) {
        throw std::logic_error("Host KV partition descriptor capacity is exhausted");
    }
    HostAllocation remaining = std::move(original.allocation);
    increment_generation(original.generation);
    bool original_assigned = false;
    for (std::size_t run_index = 0; run_index < partition_runs_.size(); ++run_index) {
        const PartitionRun& run = partition_runs_[run_index];
        HostAllocation allocation;
        if (run_index + 1U == partition_runs_.size()) {
            allocation = std::move(remaining);
        } else {
            auto split = allocator_->split(std::move(remaining), run.count * page_stride);
            allocation = std::move(split.first);
            remaining  = std::move(split.second);
        }
        memberships_[run.tail].next = kInvalidIndex;

        if (run.release) {
            std::uint32_t release_node = run.head;
            for (std::uint32_t offset = 0; offset < run.count; ++offset) {
                if (release_node == kInvalidIndex) { std::terminate(); }
                Membership& entry        = memberships_[release_node];
                const std::uint32_t next = entry.next;
                if (!pages->detach_host_replica(entry.page, old)) { std::terminate(); }
                entry                                       = {};
                free_memberships_[free_membership_count_++] = release_node;
                release_node                                = next;
            }
            const auto bytes = allocation.bytes();
            if (release_node != kInvalidIndex || !allocation.release()) { std::terminate(); }
            occupied_bytes_ -= bytes;
            continue;
        }

        const std::uint32_t target_index = !original_assigned ? index : free_[--free_count_];
        Extent& target                   = extents_[target_index];
        if (target_index != index && target.state != ExtentState::Free) { std::terminate(); }
        original_assigned = true;
        target.state      = ExtentState::Published;
        target.page_store = pages;
        target.allocation = std::move(allocation);
        target.head       = run.head;
        target.tail       = run.tail;
        target.page_count = run.count;
        const HostReplicaId replacement{allocator_->pool_id(), target_index, target.generation};
        std::uint32_t retained_node = run.head;
        for (std::uint32_t offset = 0; offset < run.count; ++offset) {
            if (retained_node == kInvalidIndex) { std::terminate(); }
            Membership& entry        = memberships_[retained_node];
            const std::uint32_t next = entry.next;
            pages->rebind_host_replica(entry.page,
                                       HostPageReplica{.extent            = old,
                                                       .page_offset       = run.begin + offset,
                                                       .membership_node   = retained_node,
                                                       .content_epoch     = entry.epoch,
                                                       .committed_columns = entry.coverage},
                                       HostPageReplica{.extent            = replacement,
                                                       .page_offset       = offset,
                                                       .membership_node   = retained_node,
                                                       .content_epoch     = entry.epoch,
                                                       .committed_columns = entry.coverage});
            entry.extent  = target_index;
            entry.offset  = offset;
            retained_node = next;
        }
        if (retained_node != kInvalidIndex) { std::terminate(); }
    }

    if (!original_assigned) {
        original.state       = ExtentState::Free;
        original.page_store  = nullptr;
        original.allocation  = {};
        original.head        = kInvalidIndex;
        original.tail        = kInvalidIndex;
        original.page_count  = 0;
        free_[free_count_++] = index;
    }
}

void HostDirectory::release_descriptor(std::uint32_t index, Extent& extent) noexcept {
    std::uint32_t node = extent.head;
    for (std::uint32_t offset = 0; offset < extent.page_count; ++offset) {
        if (node == kInvalidIndex) { std::terminate(); }
        const std::uint32_t next                    = memberships_[node].next;
        memberships_[node]                          = {};
        free_memberships_[free_membership_count_++] = node;
        node                                        = next;
    }
    if (node != kInvalidIndex) { std::terminate(); }
    extent.state      = ExtentState::Free;
    extent.page_store = nullptr;
    occupied_bytes_ -= extent.allocation.bytes();
    extent.allocation = {};
    extent.head       = kInvalidIndex;
    extent.tail       = kInvalidIndex;
    extent.page_count = 0;
    increment_generation(extent.generation);
    free_[free_count_++] = index;
}

} // namespace ninfer::models::context
