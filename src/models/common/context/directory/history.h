#pragma once

#include "models/common/context/directory/pages.h"

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::context {

class ContextTransaction;
class HistoryDirectory;

struct HistoryId {
    std::uint32_t pool_id    = invalid_pool_id;
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept {
        return pool_id != invalid_pool_id && generation != 0;
    }

    friend bool operator==(HistoryId, HistoryId) = default;
};

struct TableRow {
    std::uint32_t pool_id    = invalid_pool_id;
    std::uint32_t index      = 0;
    std::uint32_t generation = 0;

    [[nodiscard]] bool valid() const noexcept {
        return pool_id != invalid_pool_id && generation != 0;
    }

    friend bool operator==(TableRow, TableRow) = default;
};

class RowLease {
public:
    RowLease() noexcept = default;
    ~RowLease();
    RowLease(RowLease&& other) noexcept;
    RowLease& operator=(RowLease&& other) noexcept;
    RowLease(const RowLease&)            = delete;
    RowLease& operator=(const RowLease&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] TableRow handle() const noexcept { return row_; }

    [[nodiscard]] std::int32_t row_index() const noexcept { return row_.index; }

    void reset() noexcept;

private:
    RowLease(HistoryDirectory& owner, TableRow row) noexcept : owner_(&owner), row_(row) {}

    HistoryDirectory* owner_ = nullptr;
    TableRow row_;
    friend class HistoryDirectory;
};

// The sole mutable owner. Its execution and mapping consumers must retire before reset.
class HistoryOwner {
public:
    HistoryOwner() noexcept = default;
    ~HistoryOwner();
    HistoryOwner(HistoryOwner&& other) noexcept;
    HistoryOwner& operator=(HistoryOwner&& other) noexcept;
    HistoryOwner(const HistoryOwner&)            = delete;
    HistoryOwner& operator=(const HistoryOwner&) = delete;

    [[nodiscard]] HistoryId handle() const noexcept { return handle_; }

    [[nodiscard]] bool valid() const noexcept { return store_ != nullptr; }

    void reset() noexcept;

private:
    HistoryOwner(HistoryDirectory& store, HistoryId handle) noexcept
        : store_(&store), handle_(handle) {}

    HistoryDirectory* store_ = nullptr;
    HistoryId handle_;
    friend class HistoryDirectory;
};

// Retains [0, frontier) in one history, independently of Device residency.
class HistoryPrefix {
public:
    HistoryPrefix() noexcept = default;
    ~HistoryPrefix();
    HistoryPrefix(const HistoryPrefix& other);
    HistoryPrefix& operator=(const HistoryPrefix& other);
    HistoryPrefix(HistoryPrefix&& other) noexcept;
    HistoryPrefix& operator=(HistoryPrefix&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return store_ != nullptr; }

    [[nodiscard]] HistoryId history() const noexcept { return history_; }

    [[nodiscard]] std::uint32_t frontier() const noexcept { return frontier_; }

    void reset() noexcept;

private:
    HistoryPrefix(HistoryDirectory& store, HistoryId history, std::uint32_t frontier) noexcept
        : store_(&store), history_(history), frontier_(frontier) {}

    HistoryDirectory* store_ = nullptr;
    HistoryId history_;
    std::uint32_t frontier_ = 0;
    friend class HistoryDirectory;
};

// Owns all binding reservations. Copies must complete before publishing or releasing a binding.
class HistoryBinding {
public:
    HistoryBinding() noexcept = default;
    ~HistoryBinding();
    HistoryBinding(HistoryBinding&& other) noexcept;
    HistoryBinding& operator=(HistoryBinding&&)      = delete;
    HistoryBinding(const HistoryBinding&)            = delete;
    HistoryBinding& operator=(const HistoryBinding&) = delete;

    [[nodiscard]] HistoryId history() const noexcept { return history_.handle(); }

    [[nodiscard]] std::span<const PageDirectory::Restore> restores() const noexcept {
        return restores_;
    }

    [[nodiscard]] PageDirectory& pages() const noexcept;

    [[nodiscard]] bool needs_tail_copy() const noexcept { return tail_destination_.has_value(); }

    [[nodiscard]] bool blocks_writer(HistoryId history) const noexcept;
    [[nodiscard]] PageSlot tail_source() const;
    [[nodiscard]] PageSlot tail_destination() const;
    void publish_restores();
    void prepare_tail_copy();
    void prepare_mapping(ContextTransaction& transaction);
    [[nodiscard]] HistoryOwner commit();

private:
    HistoryDirectory* store_ = nullptr;
    HistoryOwner history_;
    HistoryPrefix source_;
    PageReservation reservation_;
    std::optional<RowLease> row_;
    std::vector<PageDirectory::Restore> restores_;
    std::optional<PageId> tail_destination_;
    PageId tail_source_;
    bool tail_pinned_        = false;
    bool restores_published_ = false;
    bool write_tail_         = false;
    std::uint32_t frontier_  = 0;
    bool mapping_prepared_   = false;
    friend class HistoryDirectory;
};

struct HistoryBindingDemand {
    std::uint32_t restore_pages = 0;
    std::uint32_t growth_pages  = 0;
    std::uint32_t tail_copies   = 0;
    bool reuse_history          = false;

    [[nodiscard]] std::uint32_t device_pages() const noexcept {
        return restore_pages + growth_pages + tail_copies;
    }
};

class HistoryDirectory {
public:
    HistoryDirectory(PageDirectory& pages, std::uint32_t row_count, std::uint32_t history_capacity,
                     std::uint32_t page_capacity, HostDirectory* host = nullptr);
    ~HistoryDirectory();
    HistoryDirectory(const HistoryDirectory&)            = delete;
    HistoryDirectory& operator=(const HistoryDirectory&) = delete;

    [[nodiscard]] HistoryPrefix retain_prefix(HistoryId history, std::uint32_t frontier);
    [[nodiscard]] bool valid(HistoryId history) const noexcept;
    [[nodiscard]] bool valid(TableRow row) const noexcept;

    [[nodiscard]] std::uint32_t descriptor_index(HistoryId history) const {
        (void)require(history);
        return history.index;
    }

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t occupied() const noexcept;
    [[nodiscard]] bool owned(HistoryId history) const;
    [[nodiscard]] bool active(HistoryId history) const;
    [[nodiscard]] std::uint32_t retained_after(HistoryId history,
                                               std::span<const HistoryPrefix* const> removed) const;
    [[nodiscard]] std::vector<PageId>
    released_pages(std::span<const HistoryPrefix* const> removed) const;

    [[nodiscard]] HistoryBindingDemand
    binding_demand(const HistoryPrefix* source, std::uint32_t frontier, std::uint32_t peak_frontier,
                   std::span<const HistoryPrefix* const> retired = {}) const;
    [[nodiscard]] HistoryBinding prepare_binding(const HistoryPrefix* source,
                                                 std::uint32_t frontier,
                                                 std::uint32_t peak_frontier,
                                                 std::int32_t execution_row,
                                                 bool consume_source = false);
    // These mutations belong to an accepted execution reservation. Their actions must execute
    // through retirement even after cancellation; discarding them does not roll back the directory.
    void deactivate(HistoryId history, ContextTransaction& transaction);
    void reserve_growth(HistoryId history, std::uint32_t pages);
    void release_growth(HistoryId history);
    // Replenishes the requested growth only after trimmed pages retire in this transaction.
    void settle_growth(HistoryOwner& history, std::uint32_t frontier, std::uint32_t retain_through,
                       ContextTransaction& transaction);
    // Prepare retirement of an uncommitted write tail without publishing the accepted frontier.
    // The transaction adopts that frontier only after the execution commit is accepted.
    void prepare_commit(HistoryOwner& history, std::uint32_t frontier, std::uint32_t retain_through,
                        ContextTransaction& transaction);
    [[nodiscard]] std::uint32_t reserved_growth_pages(HistoryId history) const;
    [[nodiscard]] std::uint32_t growth_pages_for_tokens(HistoryId history,
                                                        std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t write_pages(HistoryId history, std::uint32_t begin,
                                            std::uint32_t end) const;
    void prepare_write(HistoryOwner& history, std::uint32_t begin, std::uint32_t end,
                       ContextTransaction& transaction);
    void commit_frontier(HistoryId history, std::uint32_t frontier);
    void trim(HistoryOwner& history, std::uint32_t frontier, ContextTransaction& transaction);

    [[nodiscard]] std::uint32_t committed_frontier(HistoryId history) const;
    [[nodiscard]] std::uint32_t mapped_pages(HistoryId history) const;
    [[nodiscard]] std::int32_t bound_row(HistoryId history) const;
    [[nodiscard]] const RowLease& execution_row(HistoryId history) const;
    [[nodiscard]] PageId logical_page(HistoryId history, std::uint32_t index) const;
    [[nodiscard]] PageSlot physical_page(HistoryId history, std::uint32_t index) const;

    [[nodiscard]] PageDirectory& pages() const noexcept { return *pages_; }

private:
    static constexpr std::uint32_t kChunkPages = 32;

    struct DirectoryNode {
        std::shared_ptr<DirectoryNode> left;
        std::shared_ptr<DirectoryNode> right;
        std::array<PageId, kChunkPages> pages{};
    };

    using Directory = std::shared_ptr<DirectoryNode>;

    struct History {
        Directory directory;
        std::map<std::uint32_t, std::uint32_t> prefixes;
        PageReservation reservation;
        std::optional<RowLease> row;
        std::uint32_t generation   = 1;
        std::uint32_t page_count   = 0;
        std::uint32_t frontier     = 0;
        std::uint32_t retained     = 0;
        std::uint32_t prefix_count = 0;
        bool occupied              = false;
        bool owned                 = false;
    };

    [[nodiscard]] History& require(HistoryId history);
    [[nodiscard]] const History& require(HistoryId history) const;
    [[nodiscard]] History& require_owner(const HistoryOwner& history);
    [[nodiscard]] HistoryId allocate();
    [[nodiscard]] RowLease acquire_row(std::int32_t index);
    void release_row(TableRow row) noexcept;
    [[nodiscard]] HistoryOwner fork_history(const HistoryPrefix& source, std::uint32_t frontier);
    [[nodiscard]] std::uint64_t writer(HistoryId history) const noexcept;
    void add_prefix(HistoryId history, std::uint32_t frontier);
    void remove_prefix(HistoryId history, std::uint32_t frontier) noexcept;
    void release_owner(HistoryId history) noexcept;
    void deactivate_retired(HistoryId history);
    void collect(HistoryId history);
    void update_ranges(History& history, std::uint32_t old_visible, std::uint32_t new_visible,
                       std::uint32_t old_protected, std::uint32_t new_protected);
    void truncate_membership(History& history, std::uint32_t count);
    void abort(HistoryBinding& binding) noexcept;
    [[nodiscard]] HistoryOwner commit(HistoryBinding& binding);
    [[nodiscard]] PageId membership(const History& history, std::uint32_t index) const noexcept;
    [[nodiscard]] PageId& directory_slot(Directory& root, std::uint32_t index);
    [[nodiscard]] static Directory prefix_directory(const Directory& root, std::uint64_t width,
                                                    std::uint32_t count);
    void branch_owner(HistoryOwner& history, std::uint32_t frontier,
                      ContextTransaction& transaction);

    PageDirectory* pages_;
    HostDirectory* host_;
    std::uint32_t page_capacity_;
    std::uint64_t directory_capacity_ = kChunkPages;
    std::vector<std::uint32_t> row_generations_;
    std::vector<bool> row_active_;
    std::vector<History> histories_;
    std::vector<std::uint32_t> free_;
    std::uint32_t free_count_;
    std::vector<PageSlot> publish_;
    std::vector<PageId> materialize_;
    friend class HistoryOwner;
    friend class HistoryPrefix;
    friend class HistoryBinding;
    friend class RowLease;
};

} // namespace ninfer::models::context
