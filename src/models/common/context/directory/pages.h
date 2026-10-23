#pragma once

#include "core/paged_kv_geometry.h"
#include "models/common/context/directory/records.h"
#include "models/common/context/directory/metadata.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::context {

inline constexpr std::uint32_t invalid_pool_id = UINT32_MAX;

struct PageId {
    std::uint32_t pool_id    = invalid_pool_id;
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept {
        return pool_id != invalid_pool_id && generation != 0;
    }

    friend bool operator==(PageId, PageId) = default;
};

struct PageTableLocation {
    std::uint32_t row        = 0;
    std::uint32_t generation = 0;
    std::uint32_t column     = 0;

    friend bool operator==(PageTableLocation, PageTableLocation) = default;
};

// Logical coverage and references span every required rank-local part. Each physical replica owns
// its record independently; a Host copy is never a member of a separately owned multi-page extent.
class PageDirectory : public MetadataOwner {
public:
    class ReadLease {
    public:
        ReadLease() noexcept = default;
        ~ReadLease();
        ReadLease(ReadLease&&) noexcept;
        ReadLease& operator=(ReadLease&&) noexcept;
        ReadLease(const ReadLease&)            = delete;
        ReadLease& operator=(const ReadLease&) = delete;

        [[nodiscard]] PageId page() const noexcept { return page_; }

        [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

        [[nodiscard]] std::uint32_t coverage() const noexcept { return coverage_; }

        [[nodiscard]] ContextRecordBinding binding() const;
        void reset() noexcept;

    private:
        ReadLease(PageDirectory&, PageId, std::uint32_t rank, ContextMemory,
                  std::uint32_t coverage) noexcept;
        PageDirectory* store_ = nullptr;
        PageId page_;
        std::uint64_t epoch_    = 0;
        std::uint32_t rank_     = UINT32_MAX;
        std::uint32_t coverage_ = 0;
        ContextMemory memory_   = ContextMemory::Device;
        friend class PageDirectory;
    };

    class Restore {
    public:
        ~Restore();
        Restore(Restore&&) noexcept;
        Restore& operator=(Restore&&)      = delete;
        Restore(const Restore&)            = delete;
        Restore& operator=(const Restore&) = delete;

        [[nodiscard]] PageId page() const noexcept { return source_.page(); }

        [[nodiscard]] std::uint32_t rank() const noexcept { return destination_.domain().rank; }

        [[nodiscard]] ContextRecordBinding source() const { return source_.binding(); }

        [[nodiscard]] ContextRecordBinding device() const { return destination_.binding(); }

        void publish();

    private:
        Restore(PageDirectory&, ReadLease, RecordLease) noexcept;
        PageDirectory* store_ = nullptr;
        ReadLease source_;
        RecordLease destination_;
        friend class PageDirectory;
    };

    class Backup {
    public:
        ~Backup();
        Backup(Backup&&) noexcept;
        Backup& operator=(Backup&&)      = delete;
        Backup(const Backup&)            = delete;
        Backup& operator=(const Backup&) = delete;

        [[nodiscard]] PageId page() const noexcept { return source_.page(); }

        [[nodiscard]] std::uint32_t rank() const noexcept { return destination_.domain().rank; }

        [[nodiscard]] ContextRecordBinding source() const { return source_.binding(); }

        [[nodiscard]] ContextRecordBinding host() const { return destination_.binding(); }

        void publish();

    private:
        Backup(PageDirectory&, ReadLease, RecordLease) noexcept;
        PageDirectory* store_ = nullptr;
        ReadLease source_;
        RecordLease destination_;
        friend class PageDirectory;
    };

    PageDirectory(RecordDomains&, MetadataBudget&, std::uint32_t pool_id, std::uint32_t layout,
                  std::uint32_t logical_capacity);
    PageDirectory(const PageDirectory&)            = delete;
    PageDirectory& operator=(const PageDirectory&) = delete;

    [[nodiscard]] RecordDomains& domains() const noexcept { return *domains_; }

    [[nodiscard]] std::uint32_t pool_id() const noexcept { return pool_id_; }

    [[nodiscard]] std::uint32_t layout_id() const noexcept { return layout_id_; }

    [[nodiscard]] const ContextRecordLayout& layout() const;
    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t occupied() const noexcept;
    [[nodiscard]] static std::size_t metadata_bytes(std::uint32_t logical, std::uint32_t ranks,
                                                    std::uint32_t rows = 0);

    [[nodiscard]] std::uint32_t available_descriptors() const noexcept {
        return capacity() - occupied_ - reserved_;
    }

    [[nodiscard]] std::size_t metadata_requirement(std::uint32_t records) const;
    void reserve_metadata(std::uint32_t records);
    [[nodiscard]] MetadataReservation reserve_descriptors(std::uint32_t records);
    void release_metadata(std::span<const std::uint32_t>) noexcept override;
    [[nodiscard]] std::size_t reclaimable_metadata(std::span<const PageId> released = {}) const;
    void reclaim_metadata() noexcept;
    void reserve_table_capacity(std::uint32_t rows);
    void materialize(RecordReservation&, std::span<PageId>,
                     std::span<const std::uint32_t> ranks = {});
    [[nodiscard]] PageId materialize_transfer_destination(RecordReservation&,
                                                          std::uint32_t committed_columns,
                                                          std::span<const std::uint32_t> ranks);
    void publish_transfer_destination(PageId) noexcept;
    void abort_transfer_destination(PageId, RecordReservation&) noexcept;

    [[nodiscard]] bool valid(PageId) const noexcept;
    [[nodiscard]] std::uint32_t descriptor_index(PageId) const;
    [[nodiscard]] std::vector<std::uint32_t> ranks(PageId) const;
    [[nodiscard]] ContextRecordBinding physical(PageId, std::uint32_t rank) const;
    [[nodiscard]] ContextRecordBinding prepared_physical(PageId, std::uint32_t rank) const;
    [[nodiscard]] std::optional<PageId> page_for_allocation(AllocationId) const noexcept;
    [[nodiscard]] ContextRecordBinding host_binding(PageId, std::uint32_t rank) const;
    [[nodiscard]] std::vector<RecordRelease> records(PageId, ContextMemory) const;
    [[nodiscard]] bool device_resident(PageId) const;
    [[nodiscard]] bool host_resident(PageId) const;
    [[nodiscard]] bool device_covers(PageId, std::uint32_t columns) const;
    [[nodiscard]] bool device_covers(PageId, std::uint32_t rank, std::uint32_t columns) const;
    [[nodiscard]] bool host_covers(PageId, std::uint32_t columns) const;
    [[nodiscard]] bool host_covers(PageId, std::uint32_t rank, std::uint32_t columns) const;
    [[nodiscard]] std::uint32_t host_coverage(PageId, std::uint32_t rank) const;
    [[nodiscard]] std::uint64_t content_epoch(PageId) const;
    [[nodiscard]] std::uint32_t committed_columns(PageId) const;
    [[nodiscard]] std::uint32_t required_columns(PageId) const;
    [[nodiscard]] std::uint32_t required_references_above(PageId, std::uint32_t columns) const;

    void update_required(PageId, std::uint32_t before, std::uint32_t after);
    void update_protection(PageId, std::uint32_t before, std::uint32_t after);
    [[nodiscard]] bool can_write(PageId, std::uint64_t writer, std::uint32_t begin,
                                 std::uint32_t own_coverage) const;
    void prepare_write(PageId, std::uint64_t writer, std::uint32_t begin,
                       std::uint32_t own_coverage);
    void release_writer(PageId, std::uint64_t writer);
    void commit_coverage(PageId, std::uint32_t columns);

    [[nodiscard]] ReadLease pin_binding(PageId, std::uint32_t rank, ContextMemory);
    [[nodiscard]] ReadLease read_device(PageId, std::uint32_t coverage);
    [[nodiscard]] ReadLease read_device(PageId, std::uint32_t rank, std::uint32_t coverage);
    [[nodiscard]] ReadLease read_host(PageId, std::uint32_t rank, std::uint32_t coverage);
    [[nodiscard]] Restore reserve_device_replica(PageId, std::uint32_t rank, RecordReservation&);
    [[nodiscard]] Backup prepare_host_replica(PageId, std::uint32_t rank, RecordLease);
    [[nodiscard]] bool can_drop_device_replica(PageId, std::uint32_t rank) const noexcept;
    [[nodiscard]] bool drop_device_replica(PageId, std::uint32_t rank) noexcept;
    [[nodiscard]] bool drop_device_replica(PageId) noexcept;
    [[nodiscard]] bool can_drop_host_replica(PageId, std::uint32_t rank) const noexcept;
    [[nodiscard]] bool drop_host_replica(PageId, std::uint32_t rank) noexcept;
    [[nodiscard]] bool host_replica_current(PageId) const noexcept;

    [[nodiscard]] bool can_reuse_after_retirement(PageId) const noexcept;
    void prepare_returns(RecordReservation&, std::span<const std::uint32_t> ranks,
                         std::uint32_t count);
    void reuse_after_retirement(PageId, RecordReservation&);

    [[nodiscard]] std::uint64_t writer_identity(PageId) const;
    [[nodiscard]] std::uint32_t history_references(PageId) const;
    [[nodiscard]] std::uint32_t active_history_references(PageId) const;
    [[nodiscard]] std::uint32_t protected_columns(PageId) const;
    [[nodiscard]] std::uint32_t source_pins(PageId) const;
    [[nodiscard]] bool can_pin_source(PageId) const noexcept;
    void retain_active_reference(PageId);
    void release_active_reference(PageId);
    void retain_reference(PageId);
    void pin_source(PageId);
    void unpin_source(PageId);
    [[nodiscard]] bool can_release_reference(PageId) const noexcept;
    [[nodiscard]] bool release_reference(PageId) noexcept;

    void bind_table(PageId, PageTableLocation);
    void unbind_table(PageId, PageTableLocation) noexcept;
    [[nodiscard]] std::span<const PageTableLocation> table_locations(PageId) const;

private:
    struct Part {
        std::uint32_t rank = 0;
        std::optional<RecordLease> device;
        std::optional<RecordLease> host;
        std::uint32_t device_columns = 0;
        std::uint32_t host_columns   = 0;
        std::uint32_t device_readers = 0;
        std::uint32_t host_readers   = 0;
        AllocationId restoring_record;
        bool restoring  = false;
        bool backing_up = false;
    };

    struct Page {
        std::uint32_t generation        = 1;
        std::uint64_t content_epoch     = 0;
        std::uint32_t committed_columns = 0;
        std::uint32_t references        = 0;
        std::uint32_t active_references = 0;
        std::uint32_t protected_columns = 0;
        std::uint32_t required_columns  = 0;
        std::uint64_t writer_identity   = 0;
        std::array<std::uint32_t, kPagedKVPageSize> required{};
        std::array<std::uint32_t, kPagedKVPageSize> protections{};
        std::vector<Part> parts;
        std::vector<PageTableLocation> tables;
        bool destination_pinned = false;
        bool occupied           = false;
        bool reserved           = false;
        bool retiring_metadata  = false;

        [[nodiscard]] std::uint32_t readers() const noexcept;
        [[nodiscard]] bool pending() const noexcept;
    };

    [[nodiscard]] Page& require(PageId);
    [[nodiscard]] const Page& require(PageId) const;
    [[nodiscard]] static Part& part(Page&, std::uint32_t rank);
    [[nodiscard]] static const Part& part(const Page&, std::uint32_t rank);
    [[nodiscard]] PageId create(RecordReservation&, std::span<const std::uint32_t> ranks);
    void release_descriptor(PageId, RecordReservation* reservation = nullptr) noexcept;
    void retire_unowned(PageId) noexcept;

    class AllocationIndex {
    public:
        explicit AllocationIndex(MetadataBudget& metadata) : metadata_(metadata) {}

        void reserve(std::size_t records);

        [[nodiscard]] static std::size_t record_metadata_bytes(std::uint32_t ranks) {
            return std::size_t(ranks) * 18 * sizeof(Entry);
        }

        [[nodiscard]] std::size_t growth_bytes(std::size_t records) const;

        [[nodiscard]] std::size_t storage_bytes() const noexcept {
            return entries_.capacity() * sizeof(Entry);
        }

        void emplace(AllocationId, PageId);
        void erase(AllocationId) noexcept;
        [[nodiscard]] std::optional<PageId> find(AllocationId) const noexcept;

    private:
        struct Entry {
            AllocationId allocation;
            PageId page;
            std::uint8_t state = 0;
        };

        [[nodiscard]] static std::size_t hash(AllocationId) noexcept;
        MetadataAccount metadata_;
        std::vector<Entry> entries_;
    };

    MetadataAccount metadata_;
    AllocationIndex allocation_pages_;
    RecordDomains* domains_;
    std::uint32_t pool_id_;
    std::uint32_t layout_id_;
    DirectorySlots<Page> pages_;
    std::uint32_t table_capacity_ = 0;
    std::uint32_t initialized_    = 0;
    std::vector<std::uint32_t> free_;
    std::uint32_t free_count_;
    std::uint32_t occupied_        = 0;
    std::uint32_t reserved_        = 0;
    std::uint32_t next_generation_ = 1;
};

} // namespace ninfer::models::context
