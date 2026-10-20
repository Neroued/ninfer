#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::models::context {

inline constexpr std::uint32_t invalid_pool_id = std::numeric_limits<std::uint32_t>::max();

struct PageSlot {
    std::uint32_t pool_id    = invalid_pool_id;
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept {
        return pool_id != invalid_pool_id && generation != 0;
    }

    friend bool operator==(PageSlot, PageSlot) = default;
};

class PageAllocator;

class PageLease {
public:
    PageLease() noexcept = default;
    ~PageLease();
    PageLease(const PageLease&)            = delete;
    PageLease& operator=(const PageLease&) = delete;
    PageLease(PageLease&& other) noexcept;
    PageLease& operator=(PageLease&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] PageSlot slot() const noexcept { return slot_; }

    [[nodiscard]] bool belongs_to(const PageAllocator& allocator) const noexcept;
    bool release() noexcept;

private:
    friend class PageAllocator;

    PageLease(PageAllocator& owner, PageSlot slot) noexcept : owner_(&owner), slot_(slot) {}

    PageAllocator* owner_ = nullptr;
    PageSlot slot_;
};

// Capacity has no page identity until materialized. Clearing retains the owning allocator.
class PageReservation {
public:
    PageReservation() noexcept = default;
    ~PageReservation();
    PageReservation(const PageReservation&)            = delete;
    PageReservation& operator=(const PageReservation&) = delete;
    PageReservation(PageReservation&& other) noexcept;
    PageReservation& operator=(PageReservation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::uint32_t pages() const noexcept { return pages_; }

    [[nodiscard]] bool belongs_to(const PageAllocator& allocator) const noexcept;
    void clear() noexcept;
    void release() noexcept;

private:
    friend class PageAllocator;

    PageReservation(PageAllocator& owner, std::uint32_t pages) noexcept
        : owner_(&owner), pages_(pages) {}

    PageAllocator* owner_ = nullptr;
    std::uint32_t pages_  = 0;
};

// Single-owner directory allocator. Pool IDs are unique within a context directory; the allocator
// outlives its leases and reservations. Materialization and release allocate no metadata.
class PageAllocator {
public:
    PageAllocator(std::uint32_t pool_id, std::uint32_t capacity_pages);

    [[nodiscard]] static std::size_t metadata_bytes(std::uint32_t pages) {
        return sizeof(PageAllocator) +
               std::size_t(pages) * (sizeof(FreeRun) + sizeof(std::uint32_t)) +
               ((std::size_t(pages) + 63) / 64) * 8;
    }

    PageAllocator(const PageAllocator&)            = delete;
    PageAllocator& operator=(const PageAllocator&) = delete;
    PageAllocator(PageAllocator&&)                 = delete;
    PageAllocator& operator=(PageAllocator&&)      = delete;

    [[nodiscard]] std::uint32_t pool_id() const noexcept { return pool_id_; }

    [[nodiscard]] std::uint32_t capacity_pages() const noexcept { return capacity_pages_; }

    [[nodiscard]] std::uint32_t allocated_pages() const noexcept { return allocated_pages_; }

    [[nodiscard]] std::uint32_t reserved_pages() const noexcept { return reserved_pages_; }

    [[nodiscard]] std::uint32_t available_pages() const noexcept {
        return capacity_pages_ - allocated_pages_ - reserved_pages_;
    }

    [[nodiscard]] bool is_live(PageSlot slot) const noexcept;
    [[nodiscard]] std::uint32_t page_index(PageSlot slot) const;
    [[nodiscard]] std::optional<PageReservation> reserve(std::uint32_t pages) noexcept;

    [[nodiscard]] PageReservation make_empty_reservation() noexcept {
        return PageReservation(*this, 0);
    }

    [[nodiscard]] bool can_resize_reservation(const PageReservation& reservation,
                                              std::uint32_t pages) const noexcept;
    void resize_reservation(PageReservation& reservation, std::uint32_t pages);

    // Destination capacity is preallocated. Prefer contiguous pages following the predecessor,
    // then any complete free run, then fragmented runs; no second capacity admission is needed.
    void materialize(PageReservation& reservation, std::uint32_t target_page_count,
                     std::vector<PageLease>& destination,
                     std::optional<PageSlot> preferred_predecessor = std::nullopt);
    [[nodiscard]] PageLease materialize_one(PageReservation& reservation);
    void dematerialize(PageReservation& reservation, std::uint32_t target_page_count,
                       std::vector<PageLease>& source);
    void dematerialize_one(PageReservation& reservation, PageLease&& page);

private:
    friend class PageLease;
    friend class PageReservation;

    struct FreeRun {
        std::uint32_t begin = 0;
        std::uint32_t count = 0;
    };

    void release_page(PageSlot slot) noexcept;
    void release_reservation(std::uint32_t pages) noexcept;
    void consume_free_run(std::size_t run_index, std::uint32_t begin, std::uint32_t count) noexcept;
    void release_free_page(std::uint32_t index) noexcept;

    std::uint32_t pool_id_;
    std::uint32_t capacity_pages_;
    std::uint32_t allocated_pages_ = 0;
    std::uint32_t reserved_pages_  = 0;
    std::vector<FreeRun> free_runs_;
    std::vector<std::uint32_t> generations_;
    std::vector<bool> allocated_;
};

struct PageReservationRequest {
    PageAllocator* pool = nullptr;
    std::uint32_t pages = 0;
};

// Reserves all distinct pools or leaves every pool unchanged.
[[nodiscard]] std::vector<PageReservation>
reserve_page_bundle(std::span<const PageReservationRequest> requests);

// Pinned range descriptor, not an ownership capability. Its HostAllocation must outlive consumers.
struct HostExtent {
    std::uint32_t pool_id = invalid_pool_id;
    std::size_t offset    = 0;
    std::size_t bytes     = 0;

    [[nodiscard]] bool valid() const noexcept { return pool_id != invalid_pool_id && bytes != 0; }

    friend bool operator==(HostExtent, HostExtent) = default;
};

class HostAllocator;

// Sole owner of one fixed byte range. Transfers retain this owner until their final consumer.
class HostAllocation {
public:
    HostAllocation() noexcept = default;
    ~HostAllocation();
    HostAllocation(const HostAllocation&)            = delete;
    HostAllocation& operator=(const HostAllocation&) = delete;
    HostAllocation(HostAllocation&& other) noexcept;
    HostAllocation& operator=(HostAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] HostExtent extent() const noexcept { return extent_; }

    [[nodiscard]] std::size_t bytes() const noexcept { return extent_.bytes; }

    [[nodiscard]] bool reserved() const noexcept { return valid() && reserved_; }

    [[nodiscard]] bool belongs_to(const HostAllocator& allocator) const noexcept;
    void publish() noexcept;
    bool release() noexcept;

private:
    friend class HostAllocator;

    HostAllocation(HostAllocator& owner, HostExtent extent, bool reserved = true) noexcept
        : owner_(&owner), extent_(extent), reserved_(reserved) {}

    void disarm() noexcept;

    HostAllocator* owner_ = nullptr;
    HostExtent extent_;
    bool reserved_ = false;
};

// Assigns ranges in a separately owned, startup-fixed backing. The minimum complete geometry
// bounds metadata and every allocation/split piece; released ranges coalesce without allocation.
// The allocator outlives its allocations, and pool IDs are unique within a context directory.
class HostAllocator {
public:
    static constexpr std::size_t alignment = 256;

    HostAllocator(std::uint32_t pool_id, std::size_t capacity_bytes,
                  std::size_t minimum_allocation_bytes);
    HostAllocator(const HostAllocator&)            = delete;
    HostAllocator& operator=(const HostAllocator&) = delete;
    HostAllocator(HostAllocator&&)                 = delete;
    HostAllocator& operator=(HostAllocator&&)      = delete;

    [[nodiscard]] std::uint32_t pool_id() const noexcept { return pool_id_; }

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    [[nodiscard]] std::size_t minimum_allocation_bytes() const noexcept { return minimum_bytes_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    [[nodiscard]] std::size_t live_bytes() const noexcept {
        return occupied_bytes_ - reserved_bytes_;
    }

    [[nodiscard]] std::size_t free_bytes() const noexcept {
        return capacity_bytes_ - occupied_bytes_;
    }

    [[nodiscard]] std::size_t peak_occupied_bytes() const noexcept { return peak_occupied_bytes_; }

    void reset_peak() noexcept { peak_occupied_bytes_ = occupied_bytes_; }

    [[nodiscard]] std::size_t allocation_count() const noexcept { return allocation_count_; }

    [[nodiscard]] bool can_allocate(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<HostAllocation> allocate(std::size_t bytes) noexcept;
    [[nodiscard]] std::pair<HostAllocation, HostAllocation> split(HostAllocation&& allocation,
                                                                  std::size_t byte_offset);

private:
    friend class HostAllocation;

    struct FreeExtent {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    [[nodiscard]] std::size_t allocation_bytes(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<std::size_t> find_free_extent(std::size_t bytes) const noexcept;
    void insert_free_extent(FreeExtent extent) noexcept;
    void release(HostExtent extent, bool reserved) noexcept;

    std::uint32_t pool_id_;
    std::size_t capacity_bytes_;
    std::size_t minimum_bytes_       = 0;
    std::size_t occupied_bytes_      = 0;
    std::size_t reserved_bytes_      = 0;
    std::size_t peak_occupied_bytes_ = 0;
    std::size_t allocation_count_    = 0;
    std::vector<FreeExtent> free_extents_;
};

} // namespace ninfer::models::context
