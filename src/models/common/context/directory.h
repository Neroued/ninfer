#pragma once

#include "models/common/context/contract.h"
#include "models/common/context/directory/host.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::context {

struct KVDirectorySpec {
    std::uint32_t pool_id       = 0;
    std::uint32_t device_pages  = 0;
    std::uint32_t logical_pages = 0;
    std::uint32_t histories     = 0;
    std::uint32_t page_capacity = 0;
    std::uint32_t table_rows    = 0;
    KVPageGeometry geometry;
    KVTransferProfile transfer;
};

struct HostDirectorySpec {
    std::uint32_t pool_id      = 0;
    std::size_t bytes          = 0;
    std::size_t minimum_bytes  = 256;
    std::uint32_t extent_count = 0;
};

struct ContextDirectorySpec {
    StateDirectorySpec state;
    HostDirectorySpec host;
    std::vector<KVDirectorySpec> pools;
};

class KVPoolDirectory {
public:
    KVPoolDirectory(const KVDirectorySpec& spec, HostDirectory* host);

    [[nodiscard]] PageAllocator& allocation() noexcept { return allocation_; }

    [[nodiscard]] const PageAllocator& allocation() const noexcept { return allocation_; }

    [[nodiscard]] PageDirectory& pages() noexcept { return pages_; }

    [[nodiscard]] const PageDirectory& pages() const noexcept { return pages_; }

    [[nodiscard]] HistoryDirectory& histories() noexcept { return histories_; }

    [[nodiscard]] const HistoryDirectory& histories() const noexcept { return histories_; }

private:
    PageAllocator allocation_;
    PageDirectory pages_;
    HistoryDirectory histories_;
};

class ContextTransaction;

// Owns the sole logical allocation and reference bookkeeping for one model session.
class ContextDirectory {
public:
    explicit ContextDirectory(ContextDirectorySpec spec);
    ContextDirectory(const ContextDirectory&)            = delete;
    ContextDirectory& operator=(const ContextDirectory&) = delete;

    [[nodiscard]] StateDirectory& states() noexcept { return states_; }

    [[nodiscard]] const StateDirectory& states() const noexcept { return states_; }

    [[nodiscard]] HostAllocator& host_allocation() noexcept { return host_allocation_; }

    [[nodiscard]] const HostAllocator& host_allocation() const noexcept { return host_allocation_; }

    [[nodiscard]] HostDirectory& host() noexcept { return host_; }

    [[nodiscard]] const HostDirectory& host() const noexcept { return host_; }

    [[nodiscard]] KVPoolDirectory& pool(std::uint32_t id);
    [[nodiscard]] const KVPoolDirectory& pool(std::uint32_t id) const;

    [[nodiscard]] std::size_t pool_count() const noexcept { return pools_.size(); }

    [[nodiscard]] ContextTransaction begin_transaction();

private:
    HostAllocator host_allocation_;
    HostDirectory host_;
    StateDirectory states_;
    std::vector<std::unique_ptr<KVPoolDirectory>> pools_;
};

// Owns all control-side reservations and readers through physical completion. Descriptors passed
// to a rank contain values only. Cancelling a submitted transaction changes publication intent;
// the caller reports completion before releasing its storage.
class ContextTransaction {
public:
    explicit ContextTransaction(ContextDirectory& directory);
    ~ContextTransaction();
    ContextTransaction(ContextTransaction&& other) noexcept;
    ContextTransaction& operator=(ContextTransaction&&)      = delete;
    ContextTransaction(const ContextTransaction&)            = delete;
    ContextTransaction& operator=(const ContextTransaction&) = delete;

    void own_state(StateId image);
    [[nodiscard]] StateDirectory::Destination& state_destination();
    void add_state_copy(StateDirectory::Transfer transfer, bool keep_source_replica);
    void add_state_group(StateId source, std::size_t offset, std::size_t bytes);
    void clear_state();
    void add_binding(std::uint32_t pool, HistoryBinding binding);
    [[nodiscard]] HistoryBinding* binding(std::uint32_t pool) noexcept;
    [[nodiscard]] const HistoryBinding* binding(std::uint32_t pool) const noexcept;
    void add_backup(std::uint32_t pool, std::vector<PageId> pages, HostPageReservation destination,
                    bool drop_device);

    // Used by HistoryDirectory for ordinary writes, COW and ordered retirement.
    void retain(PageDirectory& pages, PageId page);
    void retain(StateDirectory& states, StateId image);
    void copy_page(PageDirectory& pages, PageId source, PageId destination);
    void update_mapping(TableRow row, std::uint32_t begin, std::span<const PageSlot> pages);
    void retire(RowLease&& row);
    void reserve_after_retirement(HistoryDirectory& histories, HistoryId history,
                                  std::uint32_t pages);
    void commit_history(HistoryDirectory& histories, HistoryId history, std::uint32_t frontier,
                        std::uint32_t retain_through);

    // Freeze migration copies and mappings before submit. Ordinary write actions can already
    // be present; no descriptor may be added after preparation.
    void prepare_actions();

    [[nodiscard]] const ContextActions& actions() const noexcept { return actions_; }

    void mark_submitted();
    void adopt() noexcept;
    void mark_completed();
    void publish();

    void cancel() noexcept { cancelled_ = true; }

    [[nodiscard]] bool cancelled() const noexcept { return cancelled_; }

    [[nodiscard]] bool in_flight() const noexcept;
    void discard();
    void reset_plan();
    [[nodiscard]] std::uint32_t spill_page_groups() const noexcept;
    [[nodiscard]] std::uint32_t tail_copy_page_groups() const noexcept;

private:
    enum class Stage { Reserved, Prepared, Submitted, Completed, Published, Discarded };

    struct PendingGrowth {
        HistoryDirectory* histories;
        HistoryId history;
        std::uint32_t pages;
    };

    struct PendingCommit {
        HistoryDirectory* histories;
        HistoryId history;
        std::uint32_t frontier;
        std::uint32_t retain_through;
    };

    struct PendingStateCopy {
        StateDirectory::Transfer transfer;
        bool keep_source;
    };

    struct PendingBinding {
        std::uint32_t pool;
        HistoryBinding value;
    };

    struct PendingBackup {
        std::uint32_t pool;
        std::vector<PageId> logical;
        HostPageReservation destination;
        bool drop_device;
    };

    void require_reserved() const;
    void release_reservations();

    ContextDirectory* directory_;
    Stage stage_    = Stage::Reserved;
    bool cancelled_ = false;
    bool adopted_   = false;
    ContextActions actions_;
    std::vector<ContextAction> state_operations_;
    std::optional<StateDirectory::Destination> state_;
    std::vector<PendingStateCopy> state_copies_;
    std::vector<PendingBinding> bindings_;
    std::vector<PendingBackup> backups_;
    std::vector<StateDirectory::ReadLease> state_readers_;
    std::vector<PageDirectory::ReadLease> page_readers_;
    std::vector<RowLease> retired_rows_;
    std::vector<PendingGrowth> growth_;
    std::vector<PendingCommit> commits_;
};

} // namespace ninfer::models::context
