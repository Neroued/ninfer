#pragma once

#include "models/common/context/directory/allocation.h"
#include "models/common/context/directory/pages.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::context {

class HostDirectory;

struct HostPageRange {
    HostExtent extent;
    std::uint32_t page_count = 0;
};

struct HostPageRelease {
    PageDirectory* pages = nullptr;
    PageId page;
};

class HostPageReservation {
public:
    HostPageReservation() noexcept = default;
    ~HostPageReservation();

    HostPageReservation(HostPageReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), descriptor_(other.descriptor_),
          generation_(other.generation_), page_store_(other.page_store_) {}

    HostPageReservation& operator=(HostPageReservation&&)      = delete;
    HostPageReservation(const HostPageReservation&)            = delete;
    HostPageReservation& operator=(const HostPageReservation&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

private:
    friend class HostDirectory;

    HostDirectory* owner_      = nullptr;
    std::uint32_t descriptor_  = 0;
    std::uint32_t generation_  = 0;
    PageDirectory* page_store_ = nullptr;
};

// Owns Host page membership and byte ranges. Replicas retain their fixed byte offsets when an
// extent is partitioned; physical storage and transfers resolve these ranges outside the directory.
class HostDirectory {
public:
    HostDirectory(HostAllocator& allocator, std::uint32_t descriptor_capacity);

    HostDirectory(const HostDirectory&)            = delete;
    HostDirectory& operator=(const HostDirectory&) = delete;
    HostDirectory(HostDirectory&&)                 = delete;
    HostDirectory& operator=(HostDirectory&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(extents_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    void restrict_coverage(PageDirectory& pages, PageId page, std::uint32_t coverage);

    [[nodiscard]] const HostKVPageLayout& page_layout(const PageDirectory& pages) const;

    [[nodiscard]] std::optional<HostPageReservation> prepare(PageDirectory& pages,
                                                             std::span<const PageId> membership);

    [[nodiscard]] HostPageRange writable_view(const HostPageReservation& reservation) const;

    [[nodiscard]] std::vector<PageSlot>
    device_sources(const HostPageReservation& reservation) const;

    void device_sources(const HostPageReservation& reservation, std::span<PageSlot> out) const;

    [[nodiscard]] std::uint32_t page_count(const HostPageReservation& reservation) const {
        validate(reservation);
        return extents_[reservation.descriptor_].page_count;
    }

    [[nodiscard]] HostReplicaId publish(HostPageReservation&& reservation) noexcept;
    void abort(HostPageReservation& reservation) noexcept;

    [[nodiscard]] bool valid(HostReplicaId replica) const noexcept {
        return replica.valid() && replica.pool_id == allocator_->pool_id() &&
               replica.index < extents_.size() &&
               extents_[replica.index].state == ExtentState::Published &&
               extents_[replica.index].generation == replica.generation;
    }

    [[nodiscard]] HostPageRange view(HostReplicaId replica) const {
        const Extent& extent = require(replica);
        return {extent.allocation.extent(), extent.page_count};
    }

    [[nodiscard]] bool can_release_page_replica(PageDirectory& pages, PageId page) const noexcept;

    [[nodiscard]] bool
    can_release_page_replicas(std::span<const HostPageRelease> releases) const noexcept;

    [[nodiscard]] bool can_release_page_replicas(PageDirectory& pages,
                                                 std::span<const PageId> releases) const noexcept;

    [[nodiscard]] bool release_page_replicas(std::span<const HostPageRelease> releases);

    [[nodiscard]] bool release_page_replicas(PageDirectory& pages,
                                             std::span<const PageId> releases);

private:
    static constexpr std::uint32_t kInvalidIndex = std::numeric_limits<std::uint32_t>::max();

    enum class ExtentState : std::uint8_t {
        Free,
        Reserved,
        Published,
    };

    struct Extent {
        ExtentState state         = ExtentState::Free;
        std::uint32_t generation  = 1;
        PageDirectory* page_store = nullptr;
        HostAllocation allocation;
        std::uint32_t head       = kInvalidIndex;
        std::uint32_t tail       = kInvalidIndex;
        std::uint32_t page_count = 0;
    };

    struct Membership {
        PageId page;
        std::uint64_t epoch    = 0;
        std::uint32_t coverage = 0;
        std::uint32_t extent   = kInvalidIndex;
        std::uint32_t offset   = 0;
        std::uint32_t next     = kInvalidIndex;
    };

    struct PartitionRun {
        std::uint32_t begin = 0;
        std::uint32_t head  = kInvalidIndex;
        std::uint32_t tail  = kInvalidIndex;
        std::uint32_t count = 0;
        bool release        = false;
    };

    [[nodiscard]] bool valid(const HostPageReservation& reservation) const noexcept {
        if (reservation.owner_ != this || reservation.descriptor_ >= extents_.size() ||
            reservation.page_store_ == nullptr) {
            return false;
        }
        const Extent& extent = extents_[reservation.descriptor_];
        return extent.state == ExtentState::Reserved &&
               extent.generation == reservation.generation_ &&
               extent.page_store == reservation.page_store_ && extent.allocation.valid() &&
               extent.page_count != 0 && extent.head != kInvalidIndex &&
               extent.tail != kInvalidIndex;
    }

    void validate(const HostPageReservation& reservation) const {
        if (!valid(reservation)) { throw std::logic_error("Host page reservation is stale"); }
    }

    [[nodiscard]] const Extent& require(HostReplicaId replica) const {
        if (!valid(replica)) { throw std::invalid_argument("Host page replica is stale"); }
        return extents_[replica.index];
    }

    static void increment_generation(std::uint32_t& generation) noexcept {
        ++generation;
        if (generation == 0) { ++generation; }
    }

    [[nodiscard]] std::uint32_t take_membership() noexcept {
        if (free_membership_count_ == 0) { return kInvalidIndex; }
        return free_memberships_[--free_membership_count_];
    }

    void begin_release_marks() const noexcept;
    [[nodiscard]] bool mark_release(PageDirectory& pages, PageId page) const noexcept;
    void release_marked_extents();
    void partition_extent(std::uint32_t index);
    void release_descriptor(std::uint32_t index, Extent& extent) noexcept;

    static void consume(HostPageReservation& reservation) noexcept {
        reservation.owner_      = nullptr;
        reservation.page_store_ = nullptr;
    }

    HostAllocator* allocator_;
    std::vector<Extent> extents_;
    std::vector<std::uint32_t> free_;
    std::uint32_t free_count_   = 0;
    std::size_t occupied_bytes_ = 0;
    std::vector<Membership> memberships_;
    std::vector<std::uint32_t> free_memberships_;
    std::uint32_t free_membership_count_ = 0;
    mutable std::vector<std::uint32_t> release_marks_;
    mutable std::vector<std::uint32_t> extent_marks_;
    mutable std::vector<std::uint32_t> affected_extents_;
    mutable std::uint32_t release_stamp_ = 0;
    std::vector<PartitionRun> partition_runs_;
    std::vector<PageId> page_scratch_;
};

} // namespace ninfer::models::context
