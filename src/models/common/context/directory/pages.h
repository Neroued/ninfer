#pragma once

#include "core/paged_kv_geometry.h"
#include "models/common/context/directory/allocation.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::context {

struct KVTransferProfile {
    std::uint64_t page_payload_bytes     = 0;
    std::uint32_t device_copy_operations = 0;
    // Entry min(run pages, size - 1); the final entry covers all longer runs.
    std::vector<std::uint32_t> host_operations;

    bool operator==(const KVTransferProfile&) const = default;

    [[nodiscard]] TransferWork host_transfer_run_work(std::uint32_t pages) const;
    [[nodiscard]] TransferWork device_copy_work(std::uint32_t pages) const;
};

class PageDirectory;
class HostDirectory;

struct HostReplicaId {
    std::uint32_t pool_id    = invalid_pool_id;
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept {
        return pool_id != invalid_pool_id && generation != 0;
    }

    friend bool operator==(HostReplicaId, HostReplicaId) = default;
};

struct HostPageReplica {
    HostReplicaId extent;
    std::uint32_t page_offset       = 0;
    std::uint32_t membership_node   = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;
};

// Logical page identity in one directory pool; its physical slot may change during migration.
struct PageId {
    std::uint32_t pool_id    = invalid_pool_id;
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept {
        return pool_id != invalid_pool_id && generation != 0;
    }

    friend bool operator==(PageId, PageId) = default;
};

// The allocator and Host directory outlive their page metadata and outstanding read leases.
class PageDirectory {
public:
    class ReadLease {
    public:
        ReadLease() noexcept = default;
        ~ReadLease();
        ReadLease(ReadLease&& other) noexcept;
        ReadLease& operator=(ReadLease&& other) noexcept;
        ReadLease(const ReadLease&)            = delete;
        ReadLease& operator=(const ReadLease&) = delete;

        [[nodiscard]] PageId page() const noexcept { return page_; }

        [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

        [[nodiscard]] std::uint32_t coverage() const noexcept { return coverage_; }

        void reset() noexcept;
    private:
        ReadLease(PageDirectory& store, PageId page, std::uint64_t epoch, std::uint32_t coverage,
                  bool host) noexcept
            : store_(&store), page_(page), epoch_(epoch), coverage_(coverage), host_(host) {}

        PageDirectory* store_ = nullptr;
        PageId page_;
        std::uint64_t epoch_    = 0;
        std::uint32_t coverage_ = 0;
        bool host_              = false;
        friend class PageDirectory;
    };

    class Restore {
    public:
        ~Restore();
        Restore(Restore&& other) noexcept;
        Restore& operator=(Restore&&)      = delete;
        Restore(const Restore&)            = delete;
        Restore& operator=(const Restore&) = delete;

        [[nodiscard]] PageId page() const noexcept { return source_.page(); }

        [[nodiscard]] PageSlot device() const noexcept { return destination_.slot(); }

        void publish();
    private:
        Restore(PageDirectory& store, ReadLease source, PageLease destination) noexcept
            : store_(&store), source_(std::move(source)), destination_(std::move(destination)) {}

        PageDirectory* store_ = nullptr;
        ReadLease source_;
        PageLease destination_;
        friend class PageDirectory;
    };

    [[nodiscard]] ReadLease read_device(PageId page, std::uint32_t coverage);
    [[nodiscard]] ReadLease read_host(PageId page, std::uint32_t coverage);

    PageDirectory(PageAllocator& allocator, std::uint32_t logical_capacity, KVPageGeometry geometry,
                  KVTransferProfile transfer = {});

    [[nodiscard]] static std::size_t metadata_bytes(std::uint32_t logical, std::uint32_t device) {
        return sizeof(PageDirectory) +
               std::size_t(logical) * (sizeof(Page) + sizeof(std::uint32_t)) +
               std::size_t(device) * sizeof(PageLease);
    }

    PageDirectory(const PageDirectory&)            = delete;
    PageDirectory& operator=(const PageDirectory&) = delete;
    PageDirectory(PageDirectory&&)                 = delete;
    PageDirectory& operator=(PageDirectory&&)      = delete;

    [[nodiscard]] PageAllocator& allocator() noexcept { return *allocator_; }

    [[nodiscard]] const PageAllocator& allocator() const noexcept { return *allocator_; }

    [[nodiscard]] const KVTransferProfile& transfer_profile() const noexcept { return transfer_; }

    [[nodiscard]] TransferWork host_transfer_run_work(std::uint32_t pages) const {
        return transfer_.host_transfer_run_work(pages);
    }

    [[nodiscard]] TransferWork device_copy_work(std::uint32_t pages) const {
        return transfer_.device_copy_work(pages);
    }

    [[nodiscard]] TransferWork host_transfer_work(std::span<const PageSlot> pages) const;

    [[nodiscard]] const KVPageGeometry& geometry() const noexcept { return host_layout_.geometry; }

    [[nodiscard]] const HostKVPageLayout& host_page_layout() const noexcept { return host_layout_; }

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(pages_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    void materialize(PageReservation& reservation, std::span<PageId> destinations,
                     std::optional<PageId> preferred_predecessor = std::nullopt);

    [[nodiscard]] PageId materialize_transfer_destination(PageReservation& reservation,
                                                          std::uint32_t committed_columns);

    void publish_transfer_destination(PageId handle) noexcept;

    void abort_transfer_destination(PageId handle, PageReservation& reservation) noexcept;

    [[nodiscard]] bool valid(PageId handle) const noexcept;

    [[nodiscard]] std::uint32_t descriptor_index(PageId handle) const {
        (void)require(handle);
        return handle.index;
    }

    [[nodiscard]] PageSlot physical(PageId handle) const;

    [[nodiscard]] bool device_resident(PageId handle) const {
        return require(handle).device_replica.has_value();
    }

    [[nodiscard]] bool host_resident(PageId handle) const {
        return require(handle).host_replica.has_value();
    }

    [[nodiscard]] const HostPageReplica& host_replica(PageId handle) const;
    [[nodiscard]] HostExtent host_extent(PageId handle) const;

    [[nodiscard]] std::uint64_t content_epoch(PageId handle) const {
        return require(handle).content_epoch;
    }

    [[nodiscard]] std::uint32_t committed_columns(PageId handle) const {
        return require(handle).committed_columns;
    }

    [[nodiscard]] std::uint32_t required_columns(PageId handle) const {
        return require(handle).required_columns;
    }

    [[nodiscard]] bool device_covers(PageId handle, std::uint32_t columns) const {
        const auto& page = require(handle);
        return page.device_replica && page.device_columns >= columns;
    }

    [[nodiscard]] bool host_covers(PageId handle, std::uint32_t columns) const;

    [[nodiscard]] std::uint32_t required_references_above(PageId handle,
                                                          std::uint32_t columns) const;

    // Each history contributes its visible and retained prefix separately. Updating one
    // history cannot remove another history's coverage or protection.
    void update_required(PageId handle, std::uint32_t before, std::uint32_t after);
    void update_protection(PageId handle, std::uint32_t before, std::uint32_t after);

    [[nodiscard]] bool can_write(PageId handle, std::uint64_t writer, std::uint32_t begin,
                                 std::uint32_t own_coverage) const;
    void prepare_write(PageId handle, std::uint64_t writer, std::uint32_t begin,
                       std::uint32_t own_coverage);
    void release_writer(PageId handle, std::uint64_t writer);

    [[nodiscard]] std::uint64_t writer_identity(PageId handle) const {
        return require(handle).writer_identity;
    }

    [[nodiscard]] std::uint32_t history_references(PageId handle) const {
        return require(handle).references;
    }

    [[nodiscard]] std::uint32_t active_history_references(PageId handle) const {
        return require(handle).active_references;
    }

    [[nodiscard]] std::uint32_t protected_columns(PageId handle) const {
        return require(handle).protected_columns;
    }

    [[nodiscard]] std::uint32_t source_pins(PageId handle) const {
        return require(handle).readers();
    }

    [[nodiscard]] bool can_pin_source(PageId handle) const noexcept;

    [[nodiscard]] Restore reserve_device_replica(PageId handle, PageReservation& reservation);

    [[nodiscard]] bool drop_device_replica(PageId handle) noexcept;

    [[nodiscard]] bool host_replica_current(PageId handle) const noexcept;

    [[nodiscard]] bool can_drop_device_replica(PageId handle) const noexcept;

    void retain_active_reference(PageId handle);

    void release_active_reference(PageId handle);

    void retain_reference(PageId handle);

    // The caller owns the ordered producer dependency, including a pending Restore destination.
    void pin_source(PageId handle);

    void unpin_source(PageId handle);

    void commit_coverage(PageId handle, std::uint32_t columns);

    [[nodiscard]] bool can_release_reference(PageId handle) const noexcept;

    // The final history release preserves Host metadata for grouped extent reclamation.
    [[nodiscard]] bool release_reference(PageId handle) noexcept;

private:
    [[nodiscard]] bool can_attach_host_replica(PageId handle, std::uint64_t epoch,
                                               std::uint32_t coverage) const noexcept;

    void attach_host_replica(PageId handle, HostPageReplica replica, HostDirectory& owner);

    void rebind_host_replica(PageId handle, HostPageReplica expected,
                             HostPageReplica replacement) noexcept;

    [[nodiscard]] bool detach_host_replica(PageId handle, HostReplicaId extent) noexcept;

    struct Page {
        std::uint32_t generation        = 1;
        std::uint64_t content_epoch     = 0;
        std::uint32_t committed_columns = 0;
        std::uint32_t references        = 0;
        std::uint32_t active_references = 0;
        std::uint32_t protected_columns = 0;
        std::uint32_t required_columns  = 0;
        std::uint32_t device_columns    = 0;
        std::uint64_t writer_identity   = 0;
        std::array<std::uint32_t, kPagedKVPageSize> required{};
        std::array<std::uint32_t, kPagedKVPageSize> protections{};
        std::uint32_t device_readers = 0;
        std::uint32_t host_readers   = 0;
        bool host_transfer_pending   = false;

        [[nodiscard]] std::uint32_t readers() const noexcept {
            return device_readers + host_readers;
        }

        bool destination_pinned = false;
        bool occupied           = false;
        std::optional<PageLease> device_replica;
        std::optional<HostPageReplica> host_replica;
        HostDirectory* host_owner = nullptr;
    };

    [[nodiscard]] static std::uint64_t next_epoch(std::uint64_t epoch) noexcept {
        ++epoch;
        return epoch == 0 ? 1 : epoch;
    }

    [[nodiscard]] Page& require(PageId handle) {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index];
    }

    [[nodiscard]] const Page& require(PageId handle) const {
        if (!valid(handle)) { throw std::invalid_argument("logical KV page handle is stale"); }
        return pages_[handle.index];
    }

    void release_descriptor(PageId handle, Page& page) noexcept;
    void retire_unowned(PageId handle) noexcept;

    PageAllocator* allocator_ = nullptr;
    HostKVPageLayout host_layout_;
    KVTransferProfile transfer_;
    std::vector<Page> pages_;
    std::vector<std::uint32_t> free_;
    std::vector<PageLease> materialization_scratch_;
    std::uint32_t free_count_ = 0;

    friend class HostDirectory;
};
} // namespace ninfer::models::context
