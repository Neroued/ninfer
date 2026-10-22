#include "models/common/context/directory/allocation.h"

#include <algorithm>
#include <exception>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace ninfer::models::context {

PageLease::~PageLease() { (void)release(); }

PageLease::PageLease(PageLease&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), slot_(std::exchange(other.slot_, {})) {}

PageLease& PageLease::operator=(PageLease&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_ = std::exchange(other.owner_, nullptr);
    slot_  = std::exchange(other.slot_, {});
    return *this;
}

bool PageLease::belongs_to(const PageAllocator& allocator) const noexcept {
    return owner_ == &allocator;
}

bool PageLease::release() noexcept {
    if (!valid()) { return false; }
    PageAllocator* owner = std::exchange(owner_, nullptr);
    const PageSlot slot  = std::exchange(slot_, {});
    if (!owner->is_live(slot)) { return false; }
    owner->release_page(slot);
    return true;
}

PageReservation::~PageReservation() { release(); }

PageReservation::PageReservation(PageReservation&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), pages_(std::exchange(other.pages_, 0)) {}

PageReservation& PageReservation::operator=(PageReservation&& other) noexcept {
    if (this == &other) { return *this; }
    release();
    owner_ = std::exchange(other.owner_, nullptr);
    pages_ = std::exchange(other.pages_, 0);
    return *this;
}

bool PageReservation::belongs_to(const PageAllocator& allocator) const noexcept {
    return owner_ == &allocator;
}

void PageReservation::clear() noexcept {
    if (!valid()) { return; }
    owner_->release_reservation(pages_);
    pages_ = 0;
}

void PageReservation::release() noexcept {
    clear();
    owner_ = nullptr;
}

PageAllocator::PageAllocator(std::uint32_t pool_id, std::uint32_t capacity_pages)
    : pool_id_(pool_id), capacity_pages_(capacity_pages) {
    if (pool_id == invalid_pool_id) {
        throw std::invalid_argument("Page allocator requires a valid pool ID");
    }
    free_runs_.reserve(capacity_pages);
    generations_.assign(capacity_pages, 1);
    allocated_.assign(capacity_pages, false);
    if (capacity_pages != 0) { free_runs_.push_back({0, capacity_pages}); }
}

bool PageAllocator::is_live(PageSlot slot) const noexcept {
    return slot.pool_id == pool_id_ && slot.index < capacity_pages_ && allocated_[slot.index] &&
           slot.generation == generations_[slot.index];
}

std::uint32_t PageAllocator::page_index(PageSlot slot) const {
    if (!is_live(slot)) { throw std::invalid_argument("Page slot is foreign or stale"); }
    return slot.index;
}

std::optional<PageReservation> PageAllocator::reserve(std::uint32_t pages) noexcept {
    if (pages == 0 || pages > available_pages()) { return std::nullopt; }
    reserved_pages_ += pages;
    return PageReservation(*this, pages);
}

bool PageAllocator::can_resize_reservation(const PageReservation& reservation,
                                           std::uint32_t pages) const noexcept {
    if (!reservation.belongs_to(*this) || reservation.pages_ > reserved_pages_) { return false; }
    const std::uint64_t used = static_cast<std::uint64_t>(allocated_pages_) +
                               static_cast<std::uint64_t>(reserved_pages_ - reservation.pages_) +
                               pages;
    return used <= capacity_pages_;
}

void PageAllocator::resize_reservation(PageReservation& reservation, std::uint32_t pages) {
    if (!can_resize_reservation(reservation, pages)) { throw std::bad_alloc(); }
    reserved_pages_    = reserved_pages_ - reservation.pages_ + pages;
    reservation.pages_ = pages;
}

void PageAllocator::materialize(PageReservation& reservation, std::uint32_t target_page_count,
                                std::vector<PageLease>& destination,
                                std::optional<PageSlot> preferred_predecessor) {
    static_assert(std::is_nothrow_move_constructible_v<PageLease>);
    if (!reservation.belongs_to(*this) || target_page_count < destination.size()) {
        throw std::invalid_argument("Page materialization has an invalid owner or extent");
    }
    const auto old_count      = static_cast<std::uint32_t>(destination.size());
    const std::uint32_t count = target_page_count - old_count;
    if (count > reservation.pages_ || target_page_count > destination.capacity()) {
        throw std::invalid_argument("Page materialization exceeds reserved capacity");
    }
    for (const PageLease& page : destination) {
        if (!page.belongs_to(*this) || !is_live(page.slot())) {
            throw std::invalid_argument("Page materialization destination is foreign or stale");
        }
    }
    if (count == 0) { return; }
    if (count > capacity_pages_ - allocated_pages_) {
        throw std::logic_error("Page reservation invariant was violated");
    }

    std::optional<std::uint32_t> preferred;
    if (preferred_predecessor) {
        preferred = page_index(*preferred_predecessor) + 1;
    } else if (!destination.empty()) {
        preferred = destination.back().slot_.index + 1;
    }
    std::size_t selected         = free_runs_.size();
    std::uint32_t selected_begin = 0;
    if (preferred) {
        const auto upper = std::upper_bound(
            free_runs_.begin(), free_runs_.end(), *preferred,
            [](std::uint32_t page, const FreeRun& run) { return page < run.begin; });
        if (upper != free_runs_.begin()) {
            const auto candidate = upper - 1;
            const std::uint64_t run_end =
                static_cast<std::uint64_t>(candidate->begin) + candidate->count;
            const std::uint64_t requested_end = static_cast<std::uint64_t>(*preferred) + count;
            if (*preferred >= candidate->begin && requested_end <= run_end) {
                selected       = static_cast<std::size_t>(candidate - free_runs_.begin());
                selected_begin = *preferred;
            }
        }
    }
    if (selected == free_runs_.size()) {
        const auto contiguous =
            std::find_if(free_runs_.begin(), free_runs_.end(),
                         [count](const FreeRun& run) { return run.count >= count; });
        if (contiguous != free_runs_.end()) {
            selected       = static_cast<std::size_t>(contiguous - free_runs_.begin());
            selected_begin = contiguous->begin;
        }
    }

    const auto append_page = [&](std::uint32_t index) {
        destination.push_back(PageLease(*this, {pool_id_, index, generations_[index]}));
        allocated_[index] = true;
    };
    if (selected != free_runs_.size()) {
        for (std::uint32_t offset = 0; offset < count; ++offset) {
            append_page(selected_begin + offset);
        }
        consume_free_run(selected, selected_begin, count);
    } else {
        std::uint32_t remaining   = count;
        std::size_t consumed_runs = 0;
        for (std::size_t run_index = 0; remaining != 0; ++run_index) {
            FreeRun& run             = free_runs_[run_index];
            const std::uint32_t take = std::min(remaining, run.count);
            for (std::uint32_t offset = 0; offset < take; ++offset) {
                append_page(run.begin + offset);
            }
            remaining -= take;
            if (take == run.count) {
                ++consumed_runs;
            } else {
                run.begin += take;
                run.count -= take;
            }
        }
        free_runs_.erase(free_runs_.begin(),
                         free_runs_.begin() + static_cast<std::ptrdiff_t>(consumed_runs));
    }
    allocated_pages_ += count;
    reserved_pages_ -= count;
    reservation.pages_ -= count;
}

PageLease PageAllocator::materialize_one(PageReservation& reservation) {
    if (!reservation.belongs_to(*this) || reservation.pages_ == 0) {
        throw std::invalid_argument("Single-page materialization exceeds reservation");
    }
    if (free_runs_.empty()) { throw std::logic_error("Page reservation invariant was violated"); }
    FreeRun& run              = free_runs_.front();
    const std::uint32_t index = run.begin++;
    if (--run.count == 0) { free_runs_.erase(free_runs_.begin()); }
    allocated_[index] = true;
    ++allocated_pages_;
    --reserved_pages_;
    --reservation.pages_;
    return PageLease(*this, {pool_id_, index, generations_[index]});
}

void PageAllocator::dematerialize(PageReservation& reservation, std::uint32_t target_page_count,
                                  std::vector<PageLease>& source) {
    if (!reservation.belongs_to(*this) || target_page_count > source.size()) {
        throw std::invalid_argument("Page dematerialization has an invalid owner or extent");
    }
    for (const PageLease& page : source) {
        if (!page.belongs_to(*this) || !is_live(page.slot())) {
            throw std::invalid_argument("Page dematerialization source is foreign or stale");
        }
    }
    const auto released = static_cast<std::uint32_t>(source.size() - target_page_count);
    if (released > std::numeric_limits<std::uint32_t>::max() - reservation.pages_) {
        throw std::overflow_error("Page reservation size overflow");
    }
    source.resize(target_page_count);
    reserved_pages_ += released;
    reservation.pages_ += released;
}

void PageAllocator::dematerialize_one(PageReservation& reservation, PageLease&& page) {
    if (!reservation.belongs_to(*this) || !page.belongs_to(*this) || !is_live(page.slot()) ||
        reservation.pages_ == std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Single-page dematerialization is invalid");
    }
    if (!page.release()) { throw std::logic_error("Single-page dematerialization lost its lease"); }
    ++reserved_pages_;
    ++reservation.pages_;
}

void PageAllocator::release_page(PageSlot slot) noexcept {
    if (!is_live(slot)) { return; }
    allocated_[slot.index] = false;
    if (++generations_[slot.index] == 0) { ++generations_[slot.index]; }
    release_free_page(slot.index);
    --allocated_pages_;
}

void PageAllocator::release_reservation(std::uint32_t pages) noexcept {
    if (pages > reserved_pages_) { std::terminate(); }
    reserved_pages_ -= pages;
}

void PageAllocator::consume_free_run(std::size_t run_index, std::uint32_t begin,
                                     std::uint32_t count) noexcept {
    if (run_index >= free_runs_.size() || count == 0) { std::terminate(); }
    FreeRun& run                     = free_runs_[run_index];
    const std::uint64_t run_end      = static_cast<std::uint64_t>(run.begin) + run.count;
    const std::uint64_t consumed_end = static_cast<std::uint64_t>(begin) + count;
    if (begin < run.begin || consumed_end > run_end) { std::terminate(); }
    if (begin == run.begin && consumed_end == run_end) {
        free_runs_.erase(free_runs_.begin() + static_cast<std::ptrdiff_t>(run_index));
    } else if (begin == run.begin) {
        run.begin += count;
        run.count -= count;
    } else if (consumed_end == run_end) {
        run.count = begin - run.begin;
    } else {
        const FreeRun right{static_cast<std::uint32_t>(consumed_end),
                            static_cast<std::uint32_t>(run_end - consumed_end)};
        run.count = begin - run.begin;
        free_runs_.insert(free_runs_.begin() + static_cast<std::ptrdiff_t>(run_index + 1), right);
    }
}

void PageAllocator::release_free_page(std::uint32_t index) noexcept {
    const auto next =
        std::lower_bound(free_runs_.begin(), free_runs_.end(), index,
                         [](const FreeRun& run, std::uint32_t page) { return run.begin < page; });
    const bool joins_right = next != free_runs_.end() && index + 1 == next->begin;
    const bool joins_left =
        next != free_runs_.begin() &&
        static_cast<std::uint64_t>((next - 1)->begin) + (next - 1)->count == index;
    if (joins_left && joins_right) {
        auto& left = *(next - 1);
        left.count += 1U + next->count;
        free_runs_.erase(next);
    } else if (joins_left) {
        ++(next - 1)->count;
    } else if (joins_right) {
        next->begin = index;
        ++next->count;
    } else {
        free_runs_.insert(next, {index, 1});
    }
}

std::vector<PageReservation> reserve_page_bundle(std::span<const PageReservationRequest> requests) {
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const auto& request = requests[index];
        if (request.pool == nullptr || request.pages == 0) {
            throw std::invalid_argument("Page bundle reservation is empty");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (requests[previous].pool->pool_id() == request.pool->pool_id()) {
                throw std::invalid_argument("Page bundle names the same pool twice");
            }
        }
        if (request.pages > request.pool->available_pages()) { throw std::bad_alloc(); }
    }
    std::vector<PageReservation> reservations;
    reservations.reserve(requests.size());
    for (const auto& request : requests) {
        auto reservation = request.pool->reserve(request.pages);
        if (!reservation) { throw std::logic_error("Page bundle prevalidation was not stable"); }
        reservations.push_back(std::move(*reservation));
    }
    return reservations;
}

HostAllocation::~HostAllocation() { (void)release(); }

HostAllocation::HostAllocation(HostAllocation&& other) noexcept
    : owner_(other.owner_), extent_(other.extent_), reserved_(other.reserved_) {
    other.disarm();
}

HostAllocation& HostAllocation::operator=(HostAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_    = other.owner_;
    extent_   = other.extent_;
    reserved_ = other.reserved_;
    other.disarm();
    return *this;
}

bool HostAllocation::belongs_to(const HostAllocator& allocator) const noexcept {
    return owner_ == &allocator;
}

void HostAllocation::publish() noexcept {
    if (!reserved()) { return; }
    owner_->reserved_bytes_ -= extent_.bytes;
    reserved_ = false;
}

bool HostAllocation::release() noexcept {
    if (!valid()) { return false; }
    owner_->release(extent_, reserved_);
    disarm();
    return true;
}

void HostAllocation::disarm() noexcept {
    owner_    = nullptr;
    extent_   = {};
    reserved_ = false;
}

HostAllocator::HostAllocator(std::uint32_t pool_id, std::size_t capacity_bytes,
                             std::size_t minimum_allocation_bytes)
    : pool_id_(pool_id), capacity_bytes_(capacity_bytes) {
    if (pool_id == invalid_pool_id) {
        throw std::invalid_argument("Host allocator requires a valid pool ID");
    }
    if (minimum_allocation_bytes == 0 ||
        minimum_allocation_bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::invalid_argument("Host minimum allocation is invalid");
    }
    minimum_bytes_ = (minimum_allocation_bytes + alignment - 1) & ~(alignment - 1);
    if (capacity_bytes == 0) { return; }
    free_extents_.reserve(capacity_bytes / minimum_bytes_ + 1);
    free_extents_.push_back({0, capacity_bytes});
}

std::size_t host_requirement_bytes(std::span<const HostRequest> requests) {
    std::size_t total = 0;
    for (const auto& request : requests) {
        if (request.count &&
            request.item_bytes > (std::numeric_limits<std::size_t>::max() - total) / request.count)
            throw std::overflow_error("Host requirements exceed addressable storage");
        total += request.item_bytes * request.count;
    }
    return total;
}

std::optional<HostPlacement> HostAllocator::plan(std::span<const HostRequest> requests,
                                                 std::span<const HostExtent> released) const {
    auto free = free_extents_;
    free.reserve(free.size() + released.size());
    for (const auto& extent : released) {
        if (extent.pool_id != pool_id_ || !extent.bytes || extent.offset % alignment ||
            extent.bytes % alignment || extent.offset > capacity_bytes_ ||
            extent.bytes > capacity_bytes_ - extent.offset)
            throw std::invalid_argument("invalid projected Host release");
        free.push_back({extent.offset, extent.bytes});
    }
    if (!released.empty()) {
        std::sort(free.begin(), free.end(),
                  [](const auto& a, const auto& b) { return a.offset < b.offset; });
        std::size_t count = 0;
        for (const auto extent : free) {
            if (count && free[count - 1].offset + free[count - 1].bytes > extent.offset)
                throw std::logic_error("projected Host releases overlap free storage");
            if (count && free[count - 1].offset + free[count - 1].bytes == extent.offset)
                free[count - 1].bytes += extent.bytes;
            else
                free[count++] = extent;
        }
        free.resize(count);
    }
    HostPlacement placement;
    placement.owner_ = this;
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const auto& request = requests[index];
        if (!request.count) continue;
        const auto item = allocation_bytes(request.item_bytes);
        if (!item) return {};
        auto left = request.count;
        for (auto& extent : free) {
            const auto count =
                static_cast<std::uint32_t>(std::min<std::size_t>(left, extent.bytes / item));
            if (!count) continue;
            const auto bytes = item * count;
            placement.runs_.push_back({index, count, {pool_id_, extent.offset, bytes}});
            extent.offset += bytes;
            extent.bytes -= bytes;
            left -= count;
            if (!left) break;
        }
        if (left) return {};
    }
    return placement;
}

std::optional<HostReservations> HostAllocator::reserve(const HostPlacement& placement) {
    if (placement.owner_ != this)
        throw std::invalid_argument("Host placement belongs to another allocator");
    const auto find = [&](const HostExtent& extent) {
        return std::find_if(free_extents_.begin(), free_extents_.end(), [&](const auto& free) {
            return free.offset <= extent.offset && extent.offset - free.offset <= free.bytes &&
                   extent.bytes <= free.bytes - (extent.offset - free.offset);
        });
    };
    for (const auto& run : placement.runs_)
        if (find(run.extent) == free_extents_.end()) return {};
    HostReservations out;
    out.reserve(placement.runs_.size());
    for (const auto& run : placement.runs_) {
        const auto index = static_cast<std::size_t>(find(run.extent) - free_extents_.begin());
        consume_free_extent(index, run.extent.offset, run.extent.bytes);
        occupied_bytes_ += run.extent.bytes;
        reserved_bytes_ += run.extent.bytes;
        peak_occupied_bytes_ = std::max(peak_occupied_bytes_, occupied_bytes_);
        ++allocation_count_;
        out.push_back({run.request, run.count, HostAllocation(*this, run.extent)});
    }
    return out;
}

void HostAllocator::consume_free_extent(std::size_t index, std::size_t offset,
                                        std::size_t bytes) noexcept {
    const auto original = free_extents_[index];
    const auto end      = offset + bytes;
    if (offset == original.offset) {
        free_extents_[index] = {end, original.bytes - bytes};
        if (!free_extents_[index].bytes)
            free_extents_.erase(free_extents_.begin() + static_cast<std::ptrdiff_t>(index));
    } else {
        free_extents_[index].bytes = offset - original.offset;
        if (end < original.offset + original.bytes)
            free_extents_.insert(free_extents_.begin() + static_cast<std::ptrdiff_t>(index + 1),
                                 {end, original.offset + original.bytes - end});
    }
}

std::size_t HostAllocator::allocation_bytes(std::size_t bytes) const noexcept {
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        return 0;
    }
    const std::size_t rounded = (bytes + alignment - 1) & ~(alignment - 1);
    return rounded < minimum_bytes_ ? 0 : rounded;
}

std::optional<std::size_t> HostAllocator::find_free_extent(std::size_t bytes) const noexcept {
    for (std::size_t index = 0; index < free_extents_.size(); ++index) {
        if (free_extents_[index].bytes >= bytes) { return index; }
    }
    return std::nullopt;
}

bool HostAllocator::can_allocate(std::size_t bytes) const noexcept {
    const std::size_t rounded = allocation_bytes(bytes);
    return rounded != 0 && find_free_extent(rounded).has_value();
}

std::optional<HostAllocation> HostAllocator::allocate(std::size_t bytes) noexcept {
    const std::size_t rounded = allocation_bytes(bytes);
    if (rounded == 0) { return std::nullopt; }
    const auto free_index = find_free_extent(rounded);
    if (!free_index) { return std::nullopt; }
    const std::size_t offset = free_extents_[*free_index].offset;
    consume_free_extent(*free_index, offset, rounded);
    occupied_bytes_ += rounded;
    reserved_bytes_ += rounded;
    peak_occupied_bytes_ = std::max(peak_occupied_bytes_, occupied_bytes_);
    ++allocation_count_;
    return HostAllocation(*this, {pool_id_, offset, rounded});
}

std::pair<HostAllocation, HostAllocation> HostAllocator::split(HostAllocation&& allocation,
                                                               std::size_t byte_offset) {
    if (!allocation.belongs_to(*this)) {
        throw std::invalid_argument("Cannot split a foreign or released Host allocation");
    }
    const auto extent = allocation.extent_;
    if (byte_offset % alignment != 0 || byte_offset < minimum_bytes_ ||
        byte_offset > extent.bytes || extent.bytes - byte_offset < minimum_bytes_) {
        throw std::out_of_range("Host split must retain two aligned complete geometries");
    }
    HostAllocation left(*this, {pool_id_, extent.offset, byte_offset}, allocation.reserved_);
    HostAllocation right(*this, {pool_id_, extent.offset + byte_offset, extent.bytes - byte_offset},
                         allocation.reserved_);
    allocation.disarm();
    ++allocation_count_;
    return {std::move(left), std::move(right)};
}

void HostAllocator::release(HostExtent extent, bool reserved) noexcept {
    occupied_bytes_ -= extent.bytes;
    if (reserved) { reserved_bytes_ -= extent.bytes; }
    --allocation_count_;
    insert_free_extent({extent.offset, extent.bytes});
}

void HostAllocator::insert_free_extent(FreeExtent extent) noexcept {
    const auto position = std::lower_bound(
        free_extents_.begin(), free_extents_.end(), extent.offset,
        [](const FreeExtent& candidate, std::size_t offset) { return candidate.offset < offset; });
    auto inserted = free_extents_.insert(position, extent);
    if (inserted != free_extents_.begin()) {
        auto previous = inserted - 1;
        if (previous->offset + previous->bytes == inserted->offset) {
            previous->bytes += inserted->bytes;
            free_extents_.erase(inserted);
            inserted = previous;
        }
    }
    const auto next = inserted + 1;
    if (next != free_extents_.end() && inserted->offset + inserted->bytes == next->offset) {
        inserted->bytes += next->bytes;
        free_extents_.erase(next);
    }
}

} // namespace ninfer::models::context
