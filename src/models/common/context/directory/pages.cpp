#include "models/common/context/directory/pages.h"

#include <algorithm>
#include <exception>
#include <limits>

namespace ninfer::models::context {
namespace {
std::uint32_t update_coverage(std::array<std::uint32_t, kPagedKVPageSize>& counts,
                              std::uint32_t before, std::uint32_t after) {
    if (before > counts.size() || after > counts.size() || (before && !counts[before - 1]))
        throw std::logic_error("KV coverage reference is invalid");
    if (after && counts[after - 1] == UINT32_MAX)
        throw std::overflow_error("KV coverage reference count overflow");
    if (before) --counts[before - 1];
    if (after) ++counts[after - 1];
    for (std::uint32_t i = counts.size(); i; --i)
        if (counts[i - 1]) return i;
    return 0;
}
} // namespace

std::size_t PageDirectory::AllocationIndex::hash(AllocationId id) noexcept {
    auto value = std::uint64_t(id.pool_id) << 32 | id.index;
    value ^= id.generation + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    return static_cast<std::size_t>(value);
}

std::size_t PageDirectory::AllocationIndex::growth_bytes(std::size_t records) const {
    std::size_t capacity = 16;
    while (capacity / 2 < records) {
        if (capacity > std::numeric_limits<std::size_t>::max() / 2)
            throw std::overflow_error("KV allocation index exceeds addressable metadata");
        capacity *= 2;
    }
    return entries_.size() >= capacity ? 0 : capacity * sizeof(Entry);
}

void PageDirectory::AllocationIndex::reserve(std::size_t records) {
    std::size_t capacity = 16;
    while (capacity / 2 < records) {
        if (capacity > std::numeric_limits<std::size_t>::max() / 2)
            throw std::overflow_error("KV allocation index exceeds addressable metadata");
        capacity *= 2;
    }
    if (entries_.size() >= capacity) return;
    metadata_.acquire(capacity * sizeof(Entry));
    std::vector<Entry> storage;
    try {
        storage.resize(capacity);
    } catch (...) {
        metadata_.release(capacity * sizeof(Entry));
        throw;
    }
    entries_.swap(storage);
    for (const auto& entry : storage)
        if (entry.state == 1) emplace(entry.allocation, entry.page);
    metadata_.release(storage.capacity() * sizeof(Entry));
}

void PageDirectory::AllocationIndex::emplace(AllocationId allocation, PageId page) {
    if (entries_.empty()) throw std::logic_error("KV allocation metadata was not reserved");
    const auto mask = entries_.size() - 1;
    auto index      = hash(allocation) & mask;
    auto reusable   = entries_.size();
    for (std::size_t probes = 0; probes < entries_.size(); ++probes) {
        auto& entry = entries_[index];
        if (entry.state == 1 && entry.allocation == allocation)
            throw std::logic_error("KV allocation already belongs to a page");
        if (entry.state == 2 && reusable == entries_.size()) reusable = index;
        if (!entry.state) {
            if (reusable == entries_.size()) reusable = index;
            break;
        }
        index = (index + 1) & mask;
    }
    if (reusable == entries_.size())
        throw std::logic_error("KV allocation metadata exceeds its reservation");
    entries_[reusable] = {allocation, page, 1};
}

void PageDirectory::AllocationIndex::erase(AllocationId allocation) noexcept {
    if (entries_.empty()) return;
    const auto mask = entries_.size() - 1;
    auto hole       = hash(allocation) & mask;
    while (entries_[hole].state && entries_[hole].allocation != allocation)
        hole = (hole + 1) & mask;
    if (!entries_[hole].state) return;
    auto next = (hole + 1) & mask;
    while (entries_[next].state) {
        const auto home = hash(entries_[next].allocation) & mask;
        if (((next - home) & mask) >= ((next - hole) & mask)) {
            entries_[hole] = entries_[next];
            hole           = next;
        }
        next = (next + 1) & mask;
    }
    entries_[hole] = {};
}

std::optional<PageId> PageDirectory::AllocationIndex::find(AllocationId allocation) const noexcept {
    if (entries_.empty()) return {};
    const auto mask = entries_.size() - 1;
    auto index      = hash(allocation) & mask;
    for (std::size_t probes = 0; probes < entries_.size(); ++probes) {
        const auto& entry = entries_[index];
        if (!entry.state) return {};
        if (entry.state == 1 && entry.allocation == allocation) return entry.page;
        index = (index + 1) & mask;
    }
    return {};
}

std::uint32_t PageDirectory::Page::readers() const noexcept {
    std::uint32_t result = 0;
    for (const auto& local : parts) result += local.device_readers + local.host_readers;
    return result;
}

bool PageDirectory::Page::pending() const noexcept {
    return destination_pinned || std::any_of(parts.begin(), parts.end(), [](const Part& local) {
               return local.restoring || local.backing_up;
           });
}

PageDirectory::ReadLease::ReadLease(PageDirectory& store, PageId page, std::uint32_t rank,
                                    ContextMemory memory, std::uint32_t coverage) noexcept
    : store_(&store), page_(page), epoch_(store.content_epoch(page)), rank_(rank),
      coverage_(coverage), memory_(memory) {}

PageDirectory::ReadLease::~ReadLease() { reset(); }

PageDirectory::ReadLease::ReadLease(ReadLease&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), page_(other.page_), epoch_(other.epoch_),
      rank_(other.rank_), coverage_(other.coverage_), memory_(other.memory_) {}

PageDirectory::ReadLease& PageDirectory::ReadLease::operator=(ReadLease&& other) noexcept {
    if (this != &other) {
        reset();
        store_    = std::exchange(other.store_, nullptr);
        page_     = other.page_;
        epoch_    = other.epoch_;
        rank_     = other.rank_;
        coverage_ = other.coverage_;
        memory_   = other.memory_;
    }
    return *this;
}

ContextRecordBinding PageDirectory::ReadLease::binding() const {
    if (!store_ || rank_ == UINT32_MAX)
        throw std::logic_error("KV reader does not select one rank-local replica");
    return memory_ == ContextMemory::Device ? store_->physical(page_, rank_)
                                            : store_->host_binding(page_, rank_);
}

void PageDirectory::ReadLease::reset() noexcept {
    if (!store_) return;
    auto* store = std::exchange(store_, nullptr);
    if (!store->valid(page_) || store->content_epoch(page_) != epoch_) std::terminate();
    auto& page = store->require(page_);
    for (auto& local : page.parts) {
        if (rank_ != UINT32_MAX && local.rank != rank_) continue;
        auto& readers =
            memory_ == ContextMemory::Device ? local.device_readers : local.host_readers;
        if (!readers) std::terminate();
        --readers;
    }
    store->retire_unowned(page_);
}

PageDirectory::ReadLease PageDirectory::pin_binding(PageId handle, std::uint32_t rank,
                                                    ContextMemory memory) {
    auto& page        = require(handle);
    auto& local       = part(page, rank);
    const bool device = memory == ContextMemory::Device;
    if (!(device ? local.device.has_value() : local.host.has_value()) ||
        page.readers() == UINT32_MAX)
        throw std::logic_error("KV physical binding is not borrowable");
    if (device)
        ++local.device_readers;
    else
        ++local.host_readers;
    return ReadLease(*this, handle, rank, memory,
                     device ? local.device_columns : local.host_columns);
}

PageDirectory::ReadLease PageDirectory::read_device(PageId handle, std::uint32_t coverage) {
    if (!device_covers(handle, coverage)) throw std::logic_error("KV Device reader lacks coverage");
    pin_source(handle);
    return ReadLease(*this, handle, UINT32_MAX, ContextMemory::Device, coverage);
}

PageDirectory::ReadLease PageDirectory::read_device(PageId handle, std::uint32_t rank,
                                                    std::uint32_t coverage) {
    auto& page  = require(handle);
    auto& local = part(page, rank);
    if (!device_covers(handle, rank, coverage) || page.writer_identity || local.restoring ||
        page.destination_pinned || page.readers() == UINT32_MAX)
        throw std::logic_error("KV Device reader lacks an available replica");
    ++local.device_readers;
    return ReadLease(*this, handle, rank, ContextMemory::Device, coverage);
}

PageDirectory::ReadLease PageDirectory::read_host(PageId handle, std::uint32_t rank,
                                                  std::uint32_t coverage) {
    auto& page  = require(handle);
    auto& local = part(page, rank);
    if (!host_covers(handle, rank, coverage) || local.backing_up || page.readers() == UINT32_MAX)
        throw std::logic_error("KV Host reader lacks an available replica");
    ++local.host_readers;
    return ReadLease(*this, handle, rank, ContextMemory::Host, coverage);
}

PageDirectory::ReadLease PageDirectory::read_device_prefix(PageId handle, std::uint32_t rank,
                                                           std::uint32_t coverage) {
    auto& page  = require(handle);
    auto& local = part(page, rank);
    if (!coverage || coverage > page.protected_columns || !device_covers(handle, rank, coverage) ||
        local.restoring || page.destination_pinned || page.readers() == UINT32_MAX)
        throw std::logic_error("KV prefix reader lacks protected Device coverage");
    ++local.device_readers;
    return ReadLease(*this, handle, rank, ContextMemory::Device, coverage);
}

PageDirectory::Restore::Restore(PageDirectory& store, ReadLease source,
                                RecordLease destination) noexcept
    : store_(&store), source_(std::move(source)), destination_(std::move(destination)) {}

PageDirectory::Restore::~Restore() {
    if (!store_) return;
    auto& local = part(store_->require(page()), rank());
    if (!local.restoring || local.device) std::terminate();
    store_->allocation_pages_.erase(destination_.allocation());
    local.restoring_record = {};
    local.restoring        = false;
}

PageDirectory::Restore::Restore(Restore&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), source_(std::move(other.source_)),
      destination_(std::move(other.destination_)) {}

void PageDirectory::Restore::publish() {
    if (!store_) throw std::logic_error("KV restore already consumed");
    auto& page  = store_->require(this->page());
    auto& local = part(page, rank());
    if (local.device || !local.restoring || page.content_epoch != source_.epoch() ||
        !store_->host_covers(this->page(), rank(), source_.coverage()))
        throw std::logic_error("KV source changed before restore publication");
    local.device.emplace(std::move(destination_));
    local.device_columns   = source_.coverage();
    local.restoring_record = {};
    local.restoring        = false;
    store_                 = nullptr;
    source_.reset();
}

PageDirectory::Backup::Backup(PageDirectory& store, ReadLease source,
                              RecordLease destination) noexcept
    : store_(&store), source_(std::move(source)), destination_(std::move(destination)) {}

PageDirectory::Backup::~Backup() {
    if (!store_) return;
    auto& local = part(store_->require(page()), rank());
    if (!local.backing_up) std::terminate();
    store_->allocation_pages_.erase(destination_.allocation());
    local.backing_up = false;
}

PageDirectory::Backup::Backup(Backup&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), source_(std::move(other.source_)),
      destination_(std::move(other.destination_)) {}

void PageDirectory::Backup::publish() {
    if (!store_) throw std::logic_error("KV backup already consumed");
    auto& page  = store_->require(this->page());
    auto& local = part(page, rank());
    if (!local.backing_up || local.host_readers || page.content_epoch != source_.epoch() ||
        !store_->device_covers(this->page(), rank(), source_.coverage()))
        throw std::logic_error("KV source changed before backup publication");
    if (local.host) store_->allocation_pages_.erase(local.host->allocation());
    local.host.emplace(std::move(destination_));
    local.host_columns = source_.coverage();
    local.backing_up   = false;
    store_             = nullptr;
    source_.reset();
}

PageDirectory::PageDirectory(RecordDomains& domains, MetadataBudget& metadata,
                             std::uint32_t pool_id, std::uint32_t layout,
                             std::uint32_t logical_capacity)
    : metadata_(metadata), allocation_pages_(metadata), domains_(&domains), pool_id_(pool_id),
      layout_id_(layout), pages_(logical_capacity, metadata), free_count_(0) {
    if (pool_id == invalid_pool_id || !logical_capacity || domains.layout(layout).parts.empty())
        throw std::invalid_argument("Logical KV directory geometry is invalid");
}

const ContextRecordLayout& PageDirectory::layout() const { return domains_->layout(layout_id_); }

std::uint32_t PageDirectory::capacity() const noexcept { return pages_.limit(); }

std::uint32_t PageDirectory::occupied() const noexcept { return occupied_; }

std::size_t PageDirectory::metadata_bytes(std::uint32_t logical, std::uint32_t ranks,
                                          std::uint32_t rows) {
    return sizeof(PageDirectory) +
           std::size_t(logical) *
               (sizeof(Page) + sizeof(std::uint32_t) + ranks * sizeof(Part) +
                rows * sizeof(PageTableLocation) + AllocationIndex::record_metadata_bytes(ranks));
}

std::size_t PageDirectory::metadata_requirement(std::uint32_t count) const {
    if (!count) return 0;
    if (count > available_descriptors())
        throw std::runtime_error("Logical KV page metadata exhausted");
    const auto needed =
        std::uint64_t(initialized_) + (count > free_count_ ? count - free_count_ : 0);
    const auto target = count > free_count_
                            ? std::min<std::uint64_t>(pages_.limit(), (needed + 255) / 256 * 256)
                            : pages_.size();
    const auto free_bytes =
        target > free_.capacity() ? (target - free_.capacity()) * sizeof(std::uint32_t) : 0;
    std::size_t payload = pages_.growth_bytes(needed);
    std::vector<std::uint32_t> holes;
    const auto free_after = free_count_ + target - initialized_;
    for (auto i = free_after - count; i < free_after; ++i) {
        const auto index = i < free_count_ ? free_[i] : initialized_ + i - free_count_;
        if (index < pages_.size() && !pages_.is_allocated(index) &&
            std::find(holes.begin(), holes.end(), index / 256) == holes.end()) {
            holes.push_back(index / 256);
            payload += pages_.ensure_bytes(index);
        }
        const bool allocated = pages_.is_allocated(index);
        const auto parts     = allocated ? pages_[index].parts.capacity() : 0;
        const auto tables    = allocated ? pages_[index].tables.capacity() : 0;
        if (parts < layout().parts.size())
            payload += (layout().parts.size() - parts) * sizeof(Part);
        if (tables < table_capacity_)
            payload += (table_capacity_ - tables) * sizeof(PageTableLocation);
    }
    const auto reverse = allocation_pages_.growth_bytes(std::size_t(occupied_ + reserved_ + count) *
                                                        layout().parts.size() * 3);
    const auto retained = reverse ? reverse - allocation_pages_.storage_bytes() : 0;
    return free_bytes + std::max(reverse, retained + payload);
}

void PageDirectory::reserve_metadata(std::uint32_t records) {
    if (!records) return;
    allocation_pages_.reserve(std::size_t(occupied_ + reserved_ + records) * layout().parts.size() *
                              3);
    if (records > free_count_) {
        const auto needed = std::uint64_t(initialized_) + records - free_count_;
        if (needed > pages_.limit()) throw std::runtime_error("Logical KV page metadata exhausted");
        const auto target = std::min<std::uint64_t>(pages_.limit(), (needed + 255) / 256 * 256);
        metadata_.reserve(free_, std::max<std::uint64_t>(pages_.size(), target));
        free_.resize(std::max<std::uint64_t>(pages_.size(), target));
        pages_.grow(needed);
        while (initialized_ < pages_.size()) free_[free_count_++] = initialized_++;
    }
    for (auto i = free_count_ - records; i < free_count_; ++i) {
        const auto index = free_[i];
        pages_.ensure(index);
        metadata_.reserve(pages_[index].parts, layout().parts.size());
        metadata_.reserve(pages_[index].tables, table_capacity_);
    }
}

MetadataReservation PageDirectory::reserve_descriptors(std::uint32_t count) {
    reserve_metadata(count);
    std::vector<std::uint32_t> ids;
    ids.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto index = free_[--free_count_];
        if (pages_[index].occupied || pages_[index].reserved) std::terminate();
        pages_[index].reserved = true;
        ids.push_back(index);
    }
    reserved_ += count;
    return MetadataReservation(*this, std::move(ids));
}

void PageDirectory::release_metadata(std::span<const std::uint32_t> ids) noexcept {
    for (auto index : ids) {
        if (!pages_.is_allocated(index)) std::terminate();
        if (pages_[index].occupied && pages_[index].retiring_metadata) {
            pages_[index].retiring_metadata = false;
            continue;
        }
        if (!pages_[index].reserved || pages_[index].occupied) std::terminate();
        pages_[index].reserved = false;
        --reserved_;
        free_[free_count_++] = index;
    }
}

std::size_t PageDirectory::reclaimable_metadata(std::span<const PageId> released) const {
    std::size_t bytes = 0;
    for (std::uint32_t first = 0; first < pages_.size(); first += 256) {
        if (!pages_.is_allocated(first)) continue;
        const auto end      = first + std::min(256U, pages_.size() - first);
        std::size_t segment = 0;
        for (auto i = first; i < end; ++i) {
            const auto& page = pages_[i];
            if (page.reserved ||
                (page.occupied &&
                 std::find(released.begin(), released.end(),
                           PageId{pool_id_, i, page.generation}) == released.end())) {
                segment = 0;
                break;
            }
            segment += sizeof(Page) + page.parts.capacity() * sizeof(Part) +
                       page.tables.capacity() * sizeof(PageTableLocation);
        }
        bytes += segment;
    }
    return bytes;
}

void PageDirectory::reclaim_metadata() noexcept {
    for (std::uint32_t first = 0; first < pages_.size(); first += 256) {
        if (!pages_.is_allocated(first)) continue;
        const auto end = std::min<std::uint64_t>(pages_.size(), std::uint64_t(first) + 256);
        bool empty     = true;
        for (auto index = first; index < end; ++index)
            empty &= !pages_[index].occupied && !pages_[index].reserved;
        if (!empty) continue;
        for (auto index = first; index < end; ++index) {
            metadata_.release(pages_[index].parts.capacity() * sizeof(Part) +
                              pages_[index].tables.capacity() * sizeof(PageTableLocation));
        }
        pages_.release_segment(first);
    }
}

void PageDirectory::reserve_table_capacity(std::uint32_t rows) {
    if (rows <= table_capacity_) return;
    for (std::uint32_t i = 0; i < pages_.size(); ++i)
        if (pages_.is_allocated(i)) metadata_.reserve(pages_[i].tables, rows);
    table_capacity_ = rows;
}

PageDirectory::Part& PageDirectory::part(Page& page, std::uint32_t rank) {
    const auto found = std::find_if(page.parts.begin(), page.parts.end(),
                                    [rank](const Part& local) { return local.rank == rank; });
    if (found == page.parts.end())
        throw std::invalid_argument("KV page is not placed on this rank");
    return *found;
}

const PageDirectory::Part& PageDirectory::part(const Page& page, std::uint32_t rank) {
    const auto found = std::find_if(page.parts.begin(), page.parts.end(),
                                    [rank](const Part& local) { return local.rank == rank; });
    if (found == page.parts.end())
        throw std::invalid_argument("KV page is not placed on this rank");
    return *found;
}

PageId PageDirectory::create(RecordReservation& reservation, std::span<const std::uint32_t> ranks) {
    if (next_generation_ == UINT32_MAX)
        throw std::overflow_error("Logical KV page generation exhausted");
    if (ranks.empty()) ranks = reservation.ranks();
    if (ranks.empty()) throw std::invalid_argument("Logical KV page has no rank-local parts");
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        const auto rank = ranks[i];
        if (std::find(ranks.begin(), ranks.begin() + i, rank) != ranks.begin() + i ||
            !reservation.remaining(rank))
            throw std::invalid_argument("KV materialization requires distinct reserved rank parts");
        if (std::none_of(layout().parts.begin(), layout().parts.end(),
                         [rank](const RecordPartSpec& local) { return local.rank == rank; }))
            throw std::invalid_argument("KV rank is outside its record layout");
    }
    const auto index = reservation.take_metadata(*this);
    if (!pages_.is_allocated(index) || !pages_[index].reserved)
        throw std::logic_error("KV materialization has no owned metadata descriptor");
    auto& page = pages_[index];
    const PageId handle{pool_id_, index, next_generation_};
    try {
        if (page.parts.capacity() < ranks.size())
            throw std::logic_error("KV rank metadata was not reserved before materialization");
        for (auto rank : ranks) {
            Part local;
            local.rank = rank;
            local.device.emplace(reservation.materialize(rank));
            page.parts.push_back(std::move(local));
            const auto& record       = *page.parts.back().device;
            const auto specification = std::find_if(
                layout().parts.begin(), layout().parts.end(),
                [rank](const RecordPartSpec& candidate) { return candidate.rank == rank; });
            if (record.domain().memory != ContextMemory::Device ||
                record.binding().bytes != specification->record.bytes)
                throw std::invalid_argument(
                    "KV materialization received a different record layout");
            allocation_pages_.emplace(record.allocation(), handle);
        }
    } catch (...) {
        for (auto& local : page.parts) {
            allocation_pages_.erase(local.device->allocation());
            reservation.restore(std::move(*local.device));
        }
        page.parts.clear();
        reservation.restore_metadata(*this, index);
        throw;
    }
    page.occupied = true;
    page.reserved = false;
    --reserved_;
    ++occupied_;
    page.generation = next_generation_++;
    page.references = 1;
    if (++page.content_epoch == 0) ++page.content_epoch;
    return handle;
}

void PageDirectory::materialize(RecordReservation& reservation, std::span<PageId> destinations,
                                std::span<const std::uint32_t> ranks) {
    for (auto destination : destinations)
        if (destination.valid()) throw std::logic_error("KV destination is already populated");
    std::size_t created = 0;
    try {
        for (; created < destinations.size(); ++created)
            destinations[created] = create(reservation, ranks);
    } catch (...) {
        for (std::size_t i = 0; i < created; ++i) {
            auto& page = require(destinations[i]);
            for (auto& local : page.parts) {
                allocation_pages_.erase(local.device->allocation());
                reservation.restore(std::move(*local.device));
            }
            release_descriptor(destinations[i], &reservation);
            destinations[i] = {};
        }
        throw;
    }
}

PageId PageDirectory::materialize_transfer_destination(RecordReservation& reservation,
                                                       std::uint32_t columns,
                                                       std::span<const std::uint32_t> ranks) {
    if (!columns || columns > kPagedKVPageSize)
        throw std::invalid_argument("KV transfer destination coverage is invalid");
    const auto handle      = create(reservation, ranks);
    auto& page             = require(handle);
    page.committed_columns = columns;
    for (auto& local : page.parts) local.device_columns = columns;
    page.references         = 0;
    page.destination_pinned = true;
    return handle;
}

PageId PageDirectory::materialize_conversion_destination(RecordReservation& reservation,
                                                         PageId source, std::uint32_t columns,
                                                         std::span<const std::uint32_t> ranks) {
    const auto epoch = content_epoch(source);
    if (columns > committed_columns(source))
        throw std::invalid_argument("KV conversion exceeds committed source coverage");
    const auto destination = materialize_transfer_destination(reservation, columns, ranks);
    require(destination).content_epoch = epoch;
    return destination;
}

void PageDirectory::publish_transfer_destination(PageId handle) noexcept {
    if (!valid(handle)) std::terminate();
    auto& page = pages_[handle.index];
    if (!page.destination_pinned || page.references || page.writer_identity || page.readers())
        std::terminate();
    page.references         = 1;
    page.destination_pinned = false;
}

void PageDirectory::abort_transfer_destination(PageId handle,
                                               RecordReservation& reservation) noexcept {
    if (!valid(handle)) return;
    auto& page = pages_[handle.index];
    if (!page.destination_pinned || page.references || page.writer_identity || page.readers())
        std::terminate();
    try {
        for (auto& local : page.parts) {
            allocation_pages_.erase(local.device->allocation());
            reservation.restore(std::move(*local.device));
        }
    } catch (...) { std::terminate(); }
    release_descriptor(handle, &reservation);
}

bool PageDirectory::valid(PageId handle) const noexcept {
    return handle.valid() && handle.pool_id == pool_id_ && pages_.is_allocated(handle.index) &&
           pages_[handle.index].occupied && pages_[handle.index].generation == handle.generation;
}

PageDirectory::Page& PageDirectory::require(PageId handle) {
    if (!valid(handle)) throw std::invalid_argument("Logical KV page handle is stale");
    return pages_[handle.index];
}

const PageDirectory::Page& PageDirectory::require(PageId handle) const {
    if (!valid(handle)) throw std::invalid_argument("Logical KV page handle is stale");
    return pages_[handle.index];
}

std::uint32_t PageDirectory::descriptor_index(PageId handle) const {
    (void)require(handle);
    return handle.index;
}

std::vector<std::uint32_t> PageDirectory::ranks(PageId handle) const {
    std::vector<std::uint32_t> result;
    for (const auto& local : require(handle).parts) result.push_back(local.rank);
    return result;
}

ContextRecordBinding PageDirectory::physical(PageId handle, std::uint32_t rank) const {
    const auto& local = part(require(handle), rank);
    if (!local.device) throw std::logic_error("KV page has no Device replica on this rank");
    return local.device->binding();
}

ContextRecordBinding PageDirectory::prepared_physical(PageId handle, std::uint32_t rank) const {
    const auto& local = part(require(handle), rank);
    if (local.device) return local.device->binding();
    if (!local.restoring) throw std::logic_error("KV page has no prepared Device replica");
    const ContextDomainId domain{rank, ContextMemory::Device};
    const auto binding = domains_->pool(domain).resolve(local.restoring_record);
    return {domain, binding.allocation, static_cast<std::int64_t>(binding.offset),
            static_cast<std::size_t>(binding.bytes), binding.binding_epoch};
}

std::optional<PageId> PageDirectory::page_for_allocation(AllocationId allocation) const noexcept {
    return allocation_pages_.find(allocation);
}

ContextRecordBinding PageDirectory::host_binding(PageId handle, std::uint32_t rank) const {
    const auto& local = part(require(handle), rank);
    if (!local.host) throw std::logic_error("KV page has no Host replica on this rank");
    return local.host->binding();
}

std::vector<RecordRelease> PageDirectory::records(PageId handle, ContextMemory memory) const {
    std::vector<RecordRelease> result;
    for (const auto& local : require(handle).parts) {
        const auto& record = memory == ContextMemory::Device ? local.device : local.host;
        if (record) result.push_back({record->domain(), record->allocation()});
    }
    return result;
}

bool PageDirectory::device_resident(PageId handle) const {
    const auto& page = require(handle);
    return std::all_of(page.parts.begin(), page.parts.end(),
                       [](const Part& local) { return local.device.has_value(); });
}

bool PageDirectory::host_resident(PageId handle) const {
    const auto& page = require(handle);
    return std::any_of(page.parts.begin(), page.parts.end(),
                       [](const Part& local) { return local.host.has_value(); });
}

bool PageDirectory::device_covers(PageId handle, std::uint32_t columns) const {
    const auto& page = require(handle);
    return std::all_of(page.parts.begin(), page.parts.end(), [columns](const Part& local) {
        return local.device && local.device_columns >= columns;
    });
}

bool PageDirectory::device_covers(PageId handle, std::uint32_t rank, std::uint32_t columns) const {
    const auto& local = part(require(handle), rank);
    return local.device && local.device_columns >= columns;
}

bool PageDirectory::host_covers(PageId handle, std::uint32_t columns) const {
    const auto& page = require(handle);
    return std::all_of(page.parts.begin(), page.parts.end(), [columns](const Part& local) {
        return local.host && local.host_columns >= columns;
    });
}

bool PageDirectory::host_covers(PageId handle, std::uint32_t rank, std::uint32_t columns) const {
    const auto& local = part(require(handle), rank);
    return local.host && local.host_columns >= columns;
}

std::uint32_t PageDirectory::host_coverage(PageId handle, std::uint32_t rank) const {
    const auto& local = part(require(handle), rank);
    return local.host ? local.host_columns : 0;
}

std::uint64_t PageDirectory::content_epoch(PageId handle) const {
    return require(handle).content_epoch;
}

std::uint32_t PageDirectory::committed_columns(PageId handle) const {
    return require(handle).committed_columns;
}

std::uint32_t PageDirectory::required_columns(PageId handle) const {
    return require(handle).required_columns;
}

std::uint64_t PageDirectory::writer_identity(PageId handle) const {
    return require(handle).writer_identity;
}

std::uint32_t PageDirectory::history_references(PageId handle) const {
    return require(handle).references;
}

std::uint32_t PageDirectory::active_history_references(PageId handle) const {
    return require(handle).active_references;
}

std::uint32_t PageDirectory::protected_columns(PageId handle) const {
    return require(handle).protected_columns;
}

std::uint32_t PageDirectory::source_pins(PageId handle) const { return require(handle).readers(); }

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
    if (after > page.committed_columns)
        throw std::logic_error("KV prefix exceeds committed content");
    page.protected_columns = update_coverage(page.protections, before, after);
}

bool PageDirectory::can_write(PageId handle, std::uint64_t writer, std::uint32_t begin,
                              std::uint32_t own_coverage) const {
    const auto& page = require(handle);
    for (auto i = begin; i < page.required_columns; ++i)
        if (page.required[i] > static_cast<std::uint32_t>(own_coverage == i + 1)) return false;
    return writer && device_covers(handle, begin) && page.protected_columns <= begin &&
           !page.readers() && !page.pending() &&
           (!page.writer_identity || page.writer_identity == writer);
}

void PageDirectory::prepare_write(PageId handle, std::uint64_t writer, std::uint32_t begin,
                                  std::uint32_t own_coverage) {
    if (!can_write(handle, writer, begin, own_coverage))
        throw std::logic_error("KV write conflicts with retained content or another writer");
    auto& page             = require(handle);
    page.committed_columns = std::min(page.committed_columns, begin);
    for (auto& local : page.parts) {
        local.device_columns = begin;
        local.host_columns   = std::min(local.host_columns, begin);
    }
    page.writer_identity = writer;
}

void PageDirectory::release_writer(PageId handle, std::uint64_t writer) {
    auto& page = require(handle);
    if (page.writer_identity == writer) page.writer_identity = 0;
}

void PageDirectory::commit_coverage(PageId handle, std::uint32_t columns) {
    auto& page = require(handle);
    if (!device_resident(handle) || !page.writer_identity || columns < page.committed_columns ||
        columns > kPagedKVPageSize)
        throw std::invalid_argument("KV committed coverage is not monotonic");
    page.committed_columns = columns;
    for (auto& local : page.parts) local.device_columns = columns;
}

PageDirectory::Restore PageDirectory::reserve_device_replica(PageId handle, std::uint32_t rank,
                                                             RecordReservation& reservation) {
    auto& page  = require(handle);
    auto& local = part(page, rank);
    if (local.device || local.restoring || page.destination_pinned ||
        !host_covers(handle, rank, page.required_columns))
        throw std::logic_error("KV Device restore is not reservable");
    auto source      = read_host(handle, rank, local.host_columns);
    auto destination = reservation.materialize(rank);
    try {
        if (destination.domain().memory != ContextMemory::Device ||
            destination.binding().bytes != local.host->binding().bytes)
            throw std::invalid_argument("KV restore received a different record layout");
        allocation_pages_.emplace(destination.allocation(), handle);
    } catch (...) {
        reservation.restore(std::move(destination));
        throw;
    }
    local.restoring_record = destination.allocation();
    local.restoring        = true;
    return Restore(*this, std::move(source), std::move(destination));
}

PageDirectory::Backup PageDirectory::prepare_host_replica(PageId handle, std::uint32_t rank,
                                                          RecordLease destination) {
    auto& page  = require(handle);
    auto& local = part(page, rank);
    if (!destination.valid() ||
        destination.domain() != ContextDomainId{rank, ContextMemory::Host} ||
        destination.binding().bytes != physical(handle, rank).bytes || local.backing_up ||
        local.host_readers || host_covers(handle, rank, page.required_columns))
        throw std::logic_error("KV Host backup is not reservable");
    auto source = read_device(handle, rank, page.required_columns);
    allocation_pages_.emplace(destination.allocation(), handle);
    local.backing_up = true;
    return Backup(*this, std::move(source), std::move(destination));
}

bool PageDirectory::can_drop_device_replica(PageId handle, std::uint32_t rank) const noexcept {
    if (!valid(handle)) return false;
    const auto& page = pages_[handle.index];
    const auto found = std::find_if(page.parts.begin(), page.parts.end(),
                                    [rank](const Part& local) { return local.rank == rank; });
    if (found == page.parts.end()) return false;
    return found->device && found->host && found->host_columns >= page.required_columns &&
           !found->device_readers && !found->restoring && !found->backing_up &&
           !page.writer_identity && !page.active_references && !page.destination_pinned;
}

bool PageDirectory::drop_device_replica(PageId handle, std::uint32_t rank) noexcept {
    if (!can_drop_device_replica(handle, rank)) return false;
    auto& local = part(pages_[handle.index], rank);
    allocation_pages_.erase(local.device->allocation());
    local.device.reset();
    local.device_columns = 0;
    return true;
}

bool PageDirectory::drop_device_replica(PageId handle) noexcept {
    if (!valid(handle)) return false;
    auto& page = pages_[handle.index];
    for (const auto& local : page.parts)
        if (local.device && !can_drop_device_replica(handle, local.rank)) return false;
    for (auto& local : page.parts) {
        if (local.device) allocation_pages_.erase(local.device->allocation());
        local.device.reset();
        local.device_columns = 0;
    }
    return true;
}

bool PageDirectory::can_drop_host_replica(PageId handle, std::uint32_t rank) const noexcept {
    if (!valid(handle)) return false;
    const auto& page = pages_[handle.index];
    const auto found = std::find_if(page.parts.begin(), page.parts.end(),
                                    [rank](const Part& local) { return local.rank == rank; });
    if (found == page.parts.end()) return false;
    return found->host && !found->host_readers && !found->backing_up &&
           (page.references == 0 ||
            (found->device && found->device_columns >= page.required_columns));
}

bool PageDirectory::drop_host_replica(PageId handle, std::uint32_t rank) noexcept {
    if (!can_drop_host_replica(handle, rank)) return false;
    auto& local = part(pages_[handle.index], rank);
    allocation_pages_.erase(local.host->allocation());
    local.host.reset();
    local.host_columns = 0;
    retire_unowned(handle);
    return true;
}

bool PageDirectory::host_replica_current(PageId handle) const noexcept {
    return valid(handle) && host_covers(handle, require(handle).required_columns);
}

bool PageDirectory::can_reuse_after_retirement(PageId handle) const noexcept {
    if (!valid(handle)) return false;
    const auto& page = pages_[handle.index];
    return page.references == 1 && !page.active_references && !page.writer_identity &&
           !page.pending() && !page.retiring_metadata && device_resident(handle);
}

void PageDirectory::prepare_returns(RecordReservation& reservation,
                                    std::span<const std::uint32_t> ranks, std::uint32_t count) {
    domains_->prepare_returns(
        reservation, {layout_id_, ContextMemory::Device, count, {ranks.begin(), ranks.end()}});
    reservation.prepare_metadata_returns(*this, count);
}

void PageDirectory::reuse_after_retirement(PageId handle, RecordReservation& reservation) {
    if (!can_reuse_after_retirement(handle))
        throw std::logic_error("KV tail does not exclusively own its retiring records");
    auto& page = require(handle);
    reservation.restore_metadata(*this, handle.index);
    page.retiring_metadata = true;
    for (auto& local : page.parts) reservation.defer_return(*local.device);
}

bool PageDirectory::can_pin_source(PageId handle) const noexcept {
    if (!valid(handle)) return false;
    const auto& page = pages_[handle.index];
    return device_resident(handle) && !page.writer_identity && !page.destination_pinned &&
           page.parts.size() <= UINT32_MAX - page.readers();
}

void PageDirectory::pin_source(PageId handle) {
    auto& page = require(handle);
    if (page.parts.size() > UINT32_MAX - page.readers())
        throw std::overflow_error("KV source pin count overflow");
    for (auto& local : page.parts) ++local.device_readers;
}

void PageDirectory::unpin_source(PageId handle) {
    auto& page = require(handle);
    for (const auto& local : page.parts)
        if (!local.device_readers) throw std::logic_error("KV Device source is not pinned");
    for (auto& local : page.parts) --local.device_readers;
    retire_unowned(handle);
}

void PageDirectory::retain_active_reference(PageId handle) {
    auto& page = require(handle);
    if (page.active_references >= page.references)
        throw std::logic_error("KV active reference is not retainable");
    ++page.active_references;
}

void PageDirectory::release_active_reference(PageId handle) {
    auto& page = require(handle);
    if (!page.active_references) throw std::logic_error("KV page has no active reference");
    --page.active_references;
}

void PageDirectory::retain_reference(PageId handle) {
    auto& page = require(handle);
    if (page.references == UINT32_MAX) throw std::overflow_error("KV reference count overflow");
    ++page.references;
}

bool PageDirectory::can_release_reference(PageId handle) const noexcept {
    if (!valid(handle)) return false;
    const auto& page = pages_[handle.index];
    return page.references && page.active_references < page.references;
}

bool PageDirectory::release_reference(PageId handle) noexcept {
    if (!can_release_reference(handle)) return false;
    --pages_[handle.index].references;
    retire_unowned(handle);
    return true;
}

void PageDirectory::retire_unowned(PageId handle) noexcept {
    auto& page = pages_[handle.index];
    if (page.references || page.readers() || page.pending()) return;
    release_descriptor(handle);
}

void PageDirectory::release_descriptor(PageId handle, RecordReservation* reservation) noexcept {
    auto& page = pages_[handle.index];
    if (page.active_references || !page.tables.empty()) std::terminate();
    for (auto& local : page.parts) {
        if (local.device) allocation_pages_.erase(local.device->allocation());
        if (local.host) allocation_pages_.erase(local.host->allocation());
    }
    const auto epoch             = page.content_epoch;
    const auto retiring_metadata = page.retiring_metadata;
    auto parts                   = std::move(page.parts);
    auto tables                  = std::move(page.tables);
    parts.clear();
    tables.clear();
    page               = Page{};
    page.parts         = std::move(parts);
    page.tables        = std::move(tables);
    page.content_epoch = epoch;
    --occupied_;
    if (retiring_metadata) {
        page.reserved = true;
        ++reserved_;
    } else if (reservation) {
        page.reserved = true;
        ++reserved_;
        try {
            reservation->restore_metadata(*this, handle.index);
        } catch (...) { std::terminate(); }
    } else {
        free_[free_count_++] = handle.index;
    }
}

void PageDirectory::bind_table(PageId handle, PageTableLocation location) {
    auto& locations = require(handle).tables;
    if (!location.generation ||
        std::find(locations.begin(), locations.end(), location) != locations.end())
        throw std::logic_error("KV table binding is invalid or duplicated");
    if (locations.size() == locations.capacity())
        throw std::logic_error("KV table metadata exceeds its active-row reservation");
    locations.push_back(location);
}

void PageDirectory::unbind_table(PageId handle, PageTableLocation location) noexcept {
    if (!valid(handle)) std::terminate();
    auto& locations  = pages_[handle.index].tables;
    const auto found = std::find(locations.begin(), locations.end(), location);
    if (found == locations.end()) std::terminate();
    locations.erase(found);
}

std::span<const PageTableLocation> PageDirectory::table_locations(PageId handle) const {
    return require(handle).tables;
}

} // namespace ninfer::models::context
