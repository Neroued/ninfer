#include "models/common/context/directory/pages.h"
#include "models/common/context/directory/host.h"

#include <algorithm>
#include <exception>

namespace ninfer::models::context {
TransferWork KVTransferProfile::host_transfer_run_work(std::uint32_t pages) const {
    if (!pages) return {};
    if (host_operations.size() < 2 || !page_payload_bytes)
        throw std::logic_error("Host KV transfer profile has not been prepared");
    if (page_payload_bytes > UINT64_MAX / pages)
        throw std::overflow_error("Host KV transfer payload overflow");
    return {page_payload_bytes * pages,
            host_operations[std::min<std::size_t>(pages, host_operations.size() - 1)]};
}

TransferWork KVTransferProfile::device_copy_work(std::uint32_t pages) const {
    if (!pages) return {};
    if (!page_payload_bytes || !device_copy_operations)
        throw std::logic_error("Device KV transfer profile has not been prepared");
    if (page_payload_bytes > UINT64_MAX / pages || device_copy_operations > UINT32_MAX / pages)
        throw std::overflow_error("Device KV transfer work overflow");
    return {page_payload_bytes * pages, device_copy_operations * pages};
}

TransferWork PageDirectory::host_transfer_work(std::span<const PageSlot> pages) const {
    TransferWork result;
    for (auto page : pages) (void)allocator_->page_index(page);
    for (std::size_t begin = 0; begin < pages.size();) {
        auto end = begin + 1;
        while (end < pages.size() && pages[end].index == pages[end - 1].index + 1) ++end;
        const auto work = transfer_.host_transfer_run_work(static_cast<std::uint32_t>(end - begin));
        result.payload_bytes += work.payload_bytes;
        if (work.copy_operations > UINT32_MAX - result.copy_operations)
            throw std::overflow_error("Host KV transfer operation count exceeds uint32");
        result.copy_operations += work.copy_operations;
        begin = end;
    }
    return result;
}

namespace {
std::uint32_t update_coverage(std::array<std::uint32_t, kPagedKVPageSize>& counts,
                              std::uint32_t before, std::uint32_t after) {
    if (before > counts.size() || after > counts.size() || (before && counts[before - 1] == 0)) {
        throw std::logic_error("KV coverage reference is invalid");
    }
    if (before) --counts[before - 1];
    if (after) ++counts[after - 1];
    for (std::uint32_t i = counts.size(); i; --i)
        if (counts[i - 1]) return i;
    return 0;
}
} // namespace

PageDirectory::ReadLease::~ReadLease() { reset(); }

PageDirectory::ReadLease::ReadLease(ReadLease&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), page_(other.page_), epoch_(other.epoch_),
      coverage_(other.coverage_), host_(other.host_) {}

PageDirectory::ReadLease& PageDirectory::ReadLease::operator=(ReadLease&& other) noexcept {
    if (this != &other) {
        reset();
        store_    = std::exchange(other.store_, nullptr);
        page_     = other.page_;
        epoch_    = other.epoch_;
        coverage_ = other.coverage_;
        host_     = other.host_;
    }
    return *this;
}

void PageDirectory::ReadLease::reset() noexcept {
    if (!store_) return;
    auto* store = std::exchange(store_, nullptr);
    if (store->content_epoch(page_) != epoch_) std::terminate();
    if (host_) {
        auto& page = store->require(page_);
        if (!page.host_readers) std::terminate();
        --page.host_readers;
        store->retire_unowned(page_);
    } else {
        store->unpin_source(page_);
    }
}

PageDirectory::ReadLease PageDirectory::read_device(PageId page, std::uint32_t coverage) {
    if (!device_covers(page, coverage)) throw std::logic_error("KV Device reader lacks coverage");
    pin_source(page);
    return ReadLease(*this, page, content_epoch(page), coverage, false);
}

PageDirectory::ReadLease PageDirectory::read_host(PageId page, std::uint32_t coverage) {
    auto& object = require(page);
    if (!host_covers(page, coverage) || object.host_transfer_pending ||
        object.readers() == UINT32_MAX)
        throw std::logic_error("KV Host reader lacks an available replica");
    ++object.host_readers;
    return ReadLease(*this, page, content_epoch(page), coverage, true);
}

PageDirectory::Restore::~Restore() {
    if (!store_) return;
    auto& page = store_->require(source_.page());
    if (!page.destination_pinned || page.device_replica) std::terminate();
    page.destination_pinned = false;
    // destination_ and source_ retire in that order after the copy's enclosing fence.
}

PageDirectory::Restore::Restore(Restore&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), source_(std::move(other.source_)),
      destination_(std::move(other.destination_)) {}

void PageDirectory::Restore::publish() {
    if (!store_) throw std::logic_error("KV restore already consumed");
    auto& page = store_->require(source_.page());
    if (page.device_replica || !page.destination_pinned || page.content_epoch != source_.epoch() ||
        !store_->host_covers(source_.page(), source_.coverage()))
        throw std::logic_error("KV source changed before restore publication");
    page.device_replica.emplace(std::move(destination_));
    page.device_columns     = source_.coverage();
    page.destination_pinned = false;
    store_                  = nullptr;
    source_.reset();
}

PageDirectory::PageDirectory(PageAllocator& allocator, std::uint32_t logical_capacity,
                             KVPageGeometry geometry, KVTransferProfile transfer)
    : allocator_(&allocator), host_layout_(plan_host_kv_page_layout(geometry)),
      transfer_(std::move(transfer)), pages_(logical_capacity), free_(pages_.size()),
      free_count_(static_cast<std::uint32_t>(pages_.size())) {
    if (geometry.page_tokens != static_cast<std::uint32_t>(kPagedKVPageSize)) {
        throw std::invalid_argument("Logical KV coverage requires 64-token pages");
    }
    if (pages_.empty() || logical_capacity < allocator.capacity_pages()) {
        throw std::invalid_argument("Logical KV page capacity is inconsistent");
    }
    for (std::uint32_t index = 0; index < pages_.size(); ++index) {
        free_[index] = static_cast<std::uint32_t>(pages_.size()) - 1U - index;
    }
    materialization_scratch_.reserve(allocator.capacity_pages());
}

void PageDirectory::materialize(PageReservation& reservation, std::span<PageId> destinations,
                                std::optional<PageId> preferred_predecessor) {
    if (destinations.empty()) { return; }
    if (destinations.size() > free_count_ ||
        destinations.size() > materialization_scratch_.capacity()) {
        throw std::logic_error("logical KV batch materialization exceeds descriptor capacity");
    }
    for (const PageId destination : destinations) {
        if (destination.valid()) {
            throw std::logic_error("logical KV batch destination is already populated");
        }
    }
    for (std::size_t offset = 0; offset < destinations.size(); ++offset) {
        const std::uint32_t index = free_[free_count_ - 1U - offset];
        if (pages_[index].occupied) {
            throw std::logic_error("logical KV free descriptor is occupied");
        }
    }

    std::optional<PageSlot> physical_predecessor;
    if (preferred_predecessor) { physical_predecessor = physical(*preferred_predecessor); }
    materialization_scratch_.clear();
    allocator_->materialize(reservation, static_cast<std::uint32_t>(destinations.size()),
                            materialization_scratch_, physical_predecessor);
    for (std::size_t offset = 0; offset < destinations.size(); ++offset) {
        const std::uint32_t index = free_[free_count_ - 1U - offset];
        Page& page                = pages_[index];
        page.device_replica.emplace(std::move(materialization_scratch_[offset]));
        page.content_epoch     = next_epoch(page.content_epoch);
        page.committed_columns = 0;
        page.references        = 1;
        page.active_references = 0;
        page.protected_columns = 0;
        page.occupied          = true;
        destinations[offset]   = PageId{allocator_->pool_id(), index, page.generation};
    }
    free_count_ -= static_cast<std::uint32_t>(destinations.size());
    materialization_scratch_.clear();
}

PageId PageDirectory::materialize_transfer_destination(PageReservation& reservation,
                                                       std::uint32_t committed_columns) {
    if (committed_columns == 0 ||
        committed_columns > static_cast<std::uint32_t>(kPagedKVPageSize) || free_count_ == 0) {
        throw std::invalid_argument("logical KV transfer destination is invalid");
    }
    PageLease lease           = allocator_->materialize_one(reservation);
    const std::uint32_t index = free_[--free_count_];
    Page& page                = pages_[index];
    page.device_replica.emplace(std::move(lease));
    page.content_epoch      = next_epoch(page.content_epoch);
    page.committed_columns  = committed_columns;
    page.device_columns     = committed_columns;
    page.references         = 0;
    page.active_references  = 0;
    page.protected_columns  = 0;
    page.destination_pinned = true;
    page.occupied           = true;
    return PageId{allocator_->pool_id(), index, page.generation};
}

void PageDirectory::publish_transfer_destination(PageId handle) noexcept {
    if (!valid(handle)) { std::terminate(); }
    Page& page = pages_[handle.index];
    if (!page.device_replica || page.host_replica || !page.destination_pinned ||
        page.references != 0 || page.writer_identity != 0 || page.readers() != 0) {
        std::terminate();
    }
    page.references         = 1;
    page.destination_pinned = false;
}

void PageDirectory::abort_transfer_destination(PageId handle,
                                               PageReservation& reservation) noexcept {
    if (!valid(handle)) { return; }
    Page& page = pages_[handle.index];
    if (!page.device_replica || page.host_replica || !page.destination_pinned ||
        page.references != 0 || page.writer_identity != 0 || page.readers() != 0) {
        return;
    }
    allocator_->dematerialize_one(reservation, std::move(*page.device_replica));
    page.device_replica.reset();
    release_descriptor(handle, page);
}

bool PageDirectory::valid(PageId handle) const noexcept {
    return handle.valid() && handle.pool_id == allocator_->pool_id() &&
           handle.index < pages_.size() && pages_[handle.index].occupied &&
           pages_[handle.index].generation == handle.generation;
}

PageSlot PageDirectory::physical(PageId handle) const {
    const Page& page = require(handle);
    if (!page.device_replica) { throw std::logic_error("logical KV page has no Device replica"); }
    return page.device_replica->slot();
}

HostExtent PageDirectory::host_extent(PageId handle) const {
    const auto& page = require(handle);
    if (!page.host_owner) throw std::invalid_argument("KV page has no Host owner");
    return page.host_owner->page_extent(*this, handle);
}

const HostPageReplica& PageDirectory::host_replica(PageId handle) const {
    const Page& page = require(handle);
    if (!page.host_replica) { throw std::logic_error("logical KV page has no Host replica"); }
    return *page.host_replica;
}

bool PageDirectory::host_covers(PageId handle, std::uint32_t columns) const {
    const auto& page = require(handle);
    return page.host_replica && page.host_replica->content_epoch == page.content_epoch &&
           page.host_replica->committed_columns >= columns;
}

std::uint32_t PageDirectory::required_references_above(PageId handle, std::uint32_t columns) const {
    const auto& page    = require(handle);
    std::uint32_t count = 0;
    for (auto i = columns; i < page.required_columns; ++i) count += page.required[i];
    return count;
}

void PageDirectory::update_required(PageId handle, std::uint32_t before, std::uint32_t after) {
    auto& page            = require(handle);
    page.required_columns = update_coverage(page.required, before, after);
}

void PageDirectory::update_protection(PageId handle, std::uint32_t before, std::uint32_t after) {
    auto& page = require(handle);
    if (after > page.committed_columns) {
        throw std::logic_error("KV prefix exceeds committed content");
    }
    page.protected_columns = update_coverage(page.protections, before, after);
}

bool PageDirectory::can_write(PageId handle, std::uint64_t writer, std::uint32_t begin,
                              std::uint32_t own_coverage) const {
    const auto& page = require(handle);
    for (auto i = begin; i < page.required_columns; ++i)
        if (page.required[i] > static_cast<std::uint32_t>(own_coverage == i + 1)) return false;
    return writer != 0 && page.device_replica && page.device_columns >= begin &&
           page.protected_columns <= begin && page.readers() == 0 && !page.destination_pinned &&
           (page.writer_identity == 0 || page.writer_identity == writer);
}

void PageDirectory::prepare_write(PageId handle, std::uint64_t writer, std::uint32_t begin,
                                  std::uint32_t own_coverage) {
    if (!can_write(handle, writer, begin, own_coverage)) {
        throw std::logic_error("KV write conflicts with retained content or another writer");
    }
    auto& page = require(handle);
    if (page.host_replica && page.host_replica->committed_columns > begin) {
        page.host_owner->restrict_coverage(*this, handle, begin);
    }
    page.committed_columns = std::min(page.committed_columns, begin);
    page.device_columns    = begin;
    page.writer_identity   = writer;
}

void PageDirectory::release_writer(PageId handle, std::uint64_t writer) {
    auto& page = require(handle);
    if (page.writer_identity == writer) { page.writer_identity = 0; }
}

bool PageDirectory::can_pin_source(PageId handle) const noexcept {
    if (!valid(handle)) { return false; }
    const Page& page = pages_[handle.index];
    return page.device_replica.has_value() && page.writer_identity == 0 &&
           !page.destination_pinned && page.readers() != std::numeric_limits<std::uint32_t>::max();
}

PageDirectory::Restore PageDirectory::reserve_device_replica(PageId handle,
                                                             PageReservation& reservation) {
    auto& page = require(handle);
    if (page.device_replica || page.destination_pinned || !page.host_replica ||
        !host_covers(handle, page.required_columns))
        throw std::logic_error("logical KV Device restore is not reservable");
    auto source             = read_host(handle, page.host_replica->committed_columns);
    auto destination        = allocator_->materialize_one(reservation);
    page.destination_pinned = true;
    return Restore(*this, std::move(source), std::move(destination));
}

bool PageDirectory::drop_device_replica(PageId handle) noexcept {
    if (!can_drop_device_replica(handle)) { return false; }
    Page& page = pages_[handle.index];
    page.device_replica.reset();
    return true;
}

bool PageDirectory::host_replica_current(PageId handle) const noexcept {
    if (!valid(handle)) { return false; }
    const Page& page = pages_[handle.index];
    return page.host_replica && page.host_replica->content_epoch == page.content_epoch &&
           page.host_replica->committed_columns >= page.required_columns;
}

bool PageDirectory::can_drop_device_replica(PageId handle) const noexcept {
    if (!valid(handle)) { return false; }
    const Page& page = pages_[handle.index];
    return page.device_replica && host_replica_current(handle) && page.writer_identity == 0 &&
           page.active_references == 0 && page.readers() == 0 && !page.destination_pinned;
}

void PageDirectory::retain_active_reference(PageId handle) {
    Page& page = require(handle);
    if (page.active_references == std::numeric_limits<std::uint32_t>::max() ||
        page.active_references >= page.references) {
        throw std::logic_error("logical KV active reference is not retainable");
    }
    ++page.active_references;
}

void PageDirectory::release_active_reference(PageId handle) {
    Page& page = require(handle);
    if (page.active_references == 0) {
        throw std::logic_error("logical KV page has no active reference");
    }
    --page.active_references;
}

void PageDirectory::retain_reference(PageId handle) {
    Page& page = require(handle);
    if (page.references == std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("logical KV page reference is not retainable");
    }
    ++page.references;
}

void PageDirectory::pin_source(PageId handle) {
    Page& page = require(handle);
    if (page.readers() == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("logical KV source pin count overflow");
    }
    ++page.device_readers;
}

void PageDirectory::unpin_source(PageId handle) {
    Page& page = require(handle);
    if (page.device_readers == 0) {
        throw std::logic_error("logical KV Device source is not pinned");
    }
    --page.device_readers;
    retire_unowned(handle);
}

void PageDirectory::commit_coverage(PageId handle, std::uint32_t columns) {
    Page& page = require(handle);
    if (!page.device_replica || page.writer_identity == 0 || columns < page.committed_columns ||
        columns > static_cast<std::uint32_t>(kPagedKVPageSize)) {
        throw std::invalid_argument("logical KV committed coverage is not monotonic");
    }
    page.committed_columns = columns;
    page.device_columns    = columns;
}

bool PageDirectory::can_release_reference(PageId handle) const noexcept {
    if (!valid(handle)) { return false; }
    const Page& page = pages_[handle.index];
    return page.references != 0 && page.active_references < page.references;
}

bool PageDirectory::release_reference(PageId handle) noexcept {
    if (!can_release_reference(handle)) return false;
    auto& page = pages_[handle.index];
    if (--page.references == 0 && !page.readers() && !page.destination_pinned) {
        page.device_replica.reset();
        if (!page.host_replica) release_descriptor(handle, page);
    }
    return true;
}

void PageDirectory::retire_unowned(PageId handle) noexcept {
    auto& page = pages_[handle.index];
    if (page.references || page.readers() || page.destination_pinned || page.host_transfer_pending)
        return;
    page.device_replica.reset();
    if (page.host_replica) {
        if (!page.host_owner->release_page_replicas(*this, {&handle, 1})) std::terminate();
    } else {
        release_descriptor(handle, page);
    }
}

bool PageDirectory::can_attach_host_replica(PageId handle, std::uint64_t epoch,
                                            std::uint32_t coverage) const noexcept {
    if (!valid(handle)) { return false; }
    const Page& page = pages_[handle.index];
    return !page.host_replica && epoch == page.content_epoch && coverage <= page.device_columns;
}

void PageDirectory::attach_host_replica(PageId handle, HostPageReplica replica,
                                        HostDirectory& owner) {
    Page& page = require(handle);
    if (!replica.extent.valid() ||
        replica.membership_node == std::numeric_limits<std::uint32_t>::max() ||
        !can_attach_host_replica(handle, replica.content_epoch, replica.committed_columns)) {
        throw std::logic_error("logical KV Host replica is not publishable");
    }
    page.host_replica = replica;
    page.host_owner   = &owner;
}

void PageDirectory::rebind_host_replica(PageId handle, HostPageReplica expected,
                                        HostPageReplica replacement) noexcept {
    if (!valid(handle)) { std::terminate(); }
    Page& page = pages_[handle.index];
    if (!page.host_replica || page.host_replica->extent != expected.extent ||
        page.host_replica->page_offset != expected.page_offset ||
        page.host_replica->membership_node != expected.membership_node ||
        page.host_replica->content_epoch != expected.content_epoch ||
        page.host_replica->committed_columns != expected.committed_columns ||
        expected.content_epoch != page.content_epoch || !replacement.extent.valid() ||
        replacement.membership_node != expected.membership_node ||
        replacement.content_epoch != expected.content_epoch ||
        replacement.committed_columns != expected.committed_columns) {
        std::terminate();
    }
    page.host_replica = replacement;
}

bool PageDirectory::detach_host_replica(PageId handle, HostReplicaId extent) noexcept {
    if (!valid(handle)) { return false; }
    Page& page = pages_[handle.index];
    if (!page.host_replica || page.host_replica->extent != extent || page.host_readers != 0 ||
        (!page.device_replica && page.references != 0)) {
        return false;
    }
    page.host_replica.reset();
    page.host_owner = nullptr;
    if (page.references == 0 && page.readers() == 0 && !page.device_replica) {
        release_descriptor(handle, page);
    }
    return true;
}

void PageDirectory::release_descriptor(PageId handle, Page& page) noexcept {
    if (page.active_references != 0) { std::terminate(); }
    page.committed_columns = 0;
    page.device_columns    = 0;
    page.required_columns  = 0;
    page.writer_identity   = 0;
    page.required          = {};
    page.protections       = {};
    page.references        = 0;
    page.active_references = 0;
    page.protected_columns = 0;
    page.device_readers = page.host_readers = 0;
    page.host_transfer_pending              = false;
    page.destination_pinned                 = false;
    page.occupied                           = false;
    page.host_replica.reset();
    page.host_owner = nullptr;
    if (++page.generation == 0) { ++page.generation; }
    free_[free_count_++] = handle.index;
}
} // namespace ninfer::models::context
