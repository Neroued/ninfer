#pragma once

#include "models/common/context/directory/history.h"
#include "models/common/context/directory/state.h"
#include "ninfer/observability/snapshot.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::context {

struct KVDirectorySpec {
    std::uint32_t pool_id       = 0;
    std::uint32_t logical_pages = 0;
    std::uint32_t histories     = 0;
    std::uint32_t page_capacity = 0;
    std::uint32_t table_rows    = 0;
    std::vector<RecordPartSpec> parts;
    bool operator==(const KVDirectorySpec&) const = default;
};

struct ContextDirectorySpec {
    std::vector<ContextRankSpec> ranks;
    StateDirectorySpec state;
    std::vector<KVDirectorySpec> pools;
    std::size_t metadata_budget_bytes                  = 0;
    bool operator==(const ContextDirectorySpec&) const = default;
};

struct HistoryRecordDemand {
    std::uint32_t pool_id = 0;
    std::uint32_t count   = 0;
    std::vector<std::uint32_t> ranks;
    std::optional<std::uint32_t> logical_pages;
    std::optional<HistoryMetadataDemand> metadata;
};

// Counts refer to semantic objects after the history has resolved reuse, COW and restore.
// A single quote includes every destination needed through the unit's retirement.
struct ContextDemand {
    std::uint32_t new_states    = 0;
    std::uint32_t state_variant = 0;
    std::optional<std::uint32_t> logical_states;
    std::vector<HistoryRecordDemand> histories;
    ContextMemory memory = ContextMemory::Device;
    std::vector<std::uint32_t> state_ranks;
    void add_history(std::uint32_t pool, const HistoryBindingDemand&);
    void add_growth(const HistoryDirectory&, HistoryId, std::uint32_t pages);
    [[nodiscard]] bool empty() const noexcept;
};

struct ContextReleaseSet {
    std::vector<RecordRelease> records;
    std::vector<StateId> states;
    std::vector<PageId> pages;
    std::vector<HistoryId> histories;
    std::vector<MetadataRelease> catalog;
    [[nodiscard]] ContextReleaseSet replicas(ContextMemory) const;
    void normalize();
    void merge(const ContextReleaseSet&);
    void subtract(const ContextReleaseSet&);
    bool operator==(const ContextReleaseSet&) const;

    [[nodiscard]] bool empty() const noexcept {
        return records.empty() && states.empty() && pages.empty() && histories.empty() &&
               catalog.empty();
    }
};

enum class ContextAvailability { Ready, NeedsMaintenance, Waiting, NeedsReclaim, Impossible };

struct ContextAssessment {
    ContextDemand demand;
    ContextAvailability status = ContextAvailability::NeedsReclaim;
    RecordAssessment placement;
    std::size_t metadata_required_bytes = 0;
    bool reclaim_metadata               = false;
    bool metadata_shortage              = false;

    [[nodiscard]] bool metadata_contribution(const ContextReleaseSet& released) const noexcept {
        return metadata_shortage && (!released.states.empty() || !released.pages.empty() ||
                                     !released.histories.empty() || !released.catalog.empty());
    }

    [[nodiscard]] bool ready() const noexcept { return status == ContextAvailability::Ready; }

    [[nodiscard]] bool feasible() const noexcept { return ready() || requires_maintenance(); }

    [[nodiscard]] bool requires_maintenance() const noexcept {
        return status == ContextAvailability::NeedsMaintenance;
    }

    [[nodiscard]] bool waiting() const noexcept { return status == ContextAvailability::Waiting; }

    [[nodiscard]] bool impossible() const noexcept {
        return status == ContextAvailability::Impossible;
    }
};

class ContextReservation {
public:
    [[nodiscard]] RecordReservation take_state();
    [[nodiscard]] RecordReservation take_history(std::uint32_t pool_id);
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
private:
    RecordReservation state_;

    struct History {
        std::uint32_t pool_id;
        RecordReservation records;
    };

    std::vector<History> histories_;
    friend class ContextDirectory;
};

struct ContextReserveResult {
    ContextAssessment assessment;
    std::optional<ContextReservation> reservation;

    explicit operator bool() const noexcept { return reservation.has_value(); }
};

struct HostWriteRequirements {
    std::uint32_t state_images = 0;
    std::vector<std::uint32_t> kv_pages;
    std::uint32_t state_variant = 0;
    [[nodiscard]] bool empty() const noexcept;
};

class KVPoolDirectory {
public:
    KVPoolDirectory(const KVDirectorySpec&, RecordDomains&, MetadataBudget&);

    [[nodiscard]] PageDirectory& pages() noexcept { return pages_; }

    [[nodiscard]] const PageDirectory& pages() const noexcept { return pages_; }

    [[nodiscard]] HistoryDirectory& histories() noexcept { return histories_; }

    [[nodiscard]] const HistoryDirectory& histories() const noexcept { return histories_; }
private:
    PageDirectory pages_;
    HistoryDirectory histories_;
};

class ContextTransaction;

struct ContextUsage {
    std::vector<ninfer::ContextPoolUsage> pools;
    ninfer::ContextMetadataUsage metadata;
    std::uint64_t state_objects   = 0;
    std::uint64_t history_objects = 0;
    std::vector<std::uint64_t> kv_page_groups;
};

class ContextDirectory {
public:
    explicit ContextDirectory(ContextDirectorySpec);
    [[nodiscard]] static std::size_t metadata_bytes(const ContextDirectorySpec&);
    [[nodiscard]] static runtime::ContextMetadataPlan metadata_plan(const ContextDirectorySpec&);
    ContextDirectory(const ContextDirectory&)            = delete;
    ContextDirectory& operator=(const ContextDirectory&) = delete;

    [[nodiscard]] MetadataBudget& metadata_budget() noexcept { return metadata_; }

    [[nodiscard]] StateDirectory& states() noexcept { return states_; }

    [[nodiscard]] const StateDirectory& states() const noexcept { return states_; }

    [[nodiscard]] RecordDomains& records() noexcept { return records_; }

    [[nodiscard]] const RecordDomains& records() const noexcept { return records_; }

    [[nodiscard]] KVPoolDirectory& pool(std::uint32_t);
    [[nodiscard]] const KVPoolDirectory& pool(std::uint32_t) const;

    [[nodiscard]] std::size_t pool_count() const noexcept { return pools_.size(); }

    [[nodiscard]] ContextAssessment assess(const ContextDemand&,
                                           const ContextReleaseSet& = {}) const;
    [[nodiscard]] ContextReserveResult reserve(const ContextDemand&);
    [[nodiscard]] std::size_t released_bytes(const ContextReleaseSet&) const;
    [[nodiscard]] std::size_t useful_release_bytes(const ContextReleaseSet&,
                                                   const ContextAssessment&) const;
    [[nodiscard]] ContextReleaseSet state_release(StateId) const;
    [[nodiscard]] ContextReleaseSet page_release(PageId) const;
    [[nodiscard]] ContextDemand host_demand(const HostWriteRequirements&) const;
    [[nodiscard]] std::size_t host_write_bytes(const HostWriteRequirements&) const;
    [[nodiscard]] ContextAssessment plan_host(const HostWriteRequirements&,
                                              const ContextReleaseSet& = {}) const;
    [[nodiscard]] std::uint32_t available_host_pages(std::uint32_t pool, std::uint32_t limit) const;
    [[nodiscard]] ContextTransaction begin_transaction();
    [[nodiscard]] std::optional<ContextTransaction> prepare_maintenance(const ContextAssessment&);
    [[nodiscard]] ContextTransaction prepare_maintenance(ContextReservation,
                                                         const ContextAssessment&);

    void request_maintenance() noexcept { maintenance_requested_ = true; }

    [[nodiscard]] bool maintenance_pending() const noexcept {
        return maintenance_requested_ || maintenance_active_;
    }

    [[nodiscard]] bool maintenance_active() const noexcept { return maintenance_active_; }

    [[nodiscard]] ContextUsage usage() const;

private:
    [[nodiscard]] std::vector<RecordDemand> record_demand(const ContextDemand&) const;
    ContextDirectorySpec spec_;
    std::size_t core_metadata_budget_ = 0;
    MetadataBudget metadata_;
    RecordDomains records_;
    StateDirectory states_;
    std::vector<std::unique_ptr<KVPoolDirectory>> pools_;
    std::size_t address_borrows_ = 0;
    bool failed_                 = false;
    bool maintenance_active_     = false;
    bool maintenance_requested_  = false;
    friend class ContextTransaction;
};

// Owns control resources through retirement. Rank actions contain no borrowed directory pointers.
class ContextTransaction {
public:
    explicit ContextTransaction(ContextDirectory&);
    ~ContextTransaction();
    ContextTransaction(ContextTransaction&&) noexcept;
    ContextTransaction& operator=(ContextTransaction&&)      = delete;
    ContextTransaction(const ContextTransaction&)            = delete;
    ContextTransaction& operator=(const ContextTransaction&) = delete;

    void own_state(StateId);
    [[nodiscard]] StateDirectory::Destination& state_destination();
    void add_state_copy(StateDirectory::Transfer, bool keep_source_replica);
    void add_state_group(StateId source, std::size_t offset, std::size_t bytes,
                         std::optional<std::uint32_t> rank = {});
    void clear_state();
    void add_binding(std::uint32_t pool, HistoryBinding);
    [[nodiscard]] HistoryBinding* binding(std::uint32_t) noexcept;
    [[nodiscard]] const HistoryBinding* binding(std::uint32_t) const noexcept;
    void add_backup(std::uint32_t pool, PageDirectory::Backup, bool drop_device);
    void add_backups(std::uint32_t pool, std::span<const PageId>, RecordReservation,
                     bool drop_device);
    void retain(PageDirectory&, PageId);
    void retain(StateDirectory&, StateId);
    void copy_page(PageDirectory&, PageId source, PageId destination);
    void update_mapping(PageDirectory&, TableRow, std::uint32_t begin, std::span<const PageId>);
    void retire(RowLease&&);
    void commit_history(HistoryDirectory&, HistoryId, std::uint32_t frontier,
                        std::uint32_t retain_through, RecordReservation = {});
    void prepare_actions();
    [[nodiscard]] const ContextActions& actions(std::uint32_t rank = 0) const;
    void mark_submitted();
    void adopt() noexcept;
    void mark_completed();
    void publish();
    [[nodiscard]] ContextReservation take_reservation();

    void cancel() noexcept { cancelled_ = true; }

    [[nodiscard]] bool cancelled() const noexcept { return cancelled_; }

    [[nodiscard]] bool in_flight() const noexcept;
    void discard();
    void fail_after_drain() noexcept;
    void reset_plan();
    [[nodiscard]] std::uint32_t spill_page_groups() const noexcept;
    [[nodiscard]] std::uint32_t tail_copy_page_groups() const noexcept;

private:
    enum class Stage { Reserved, Prepared, Submitted, Completed, Published, Discarded };

    struct RankActions {
        std::uint32_t rank;
        ContextActions actions;
    };

    struct PendingCommit {
        HistoryDirectory* histories;
        HistoryId history;
        std::uint32_t frontier;
        std::uint32_t retain_through;
        RecordReservation records;
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
        PageDirectory::Backup value;
        bool drop_device;
    };

    void require_reserved() const;
    void release_reservations();
    void release_address_borrow() noexcept;
    ContextActions& mutable_actions(std::uint32_t rank);
    ContextDirectory* directory_;
    Stage stage_            = Stage::Reserved;
    bool cancelled_         = false;
    bool adopted_           = false;
    bool address_borrow_    = true;
    bool maintenance_       = false;
    bool maintenance_begun_ = false;
    std::vector<ContextDomainId> maintenance_domains_;
    std::optional<ContextReservation> maintenance_reservation_;
    std::vector<RankActions> actions_;
    std::optional<StateDirectory::Destination> state_;
    std::vector<PendingStateCopy> state_copies_;
    std::vector<PendingBinding> bindings_;
    std::vector<PendingBackup> backups_;
    std::vector<StateDirectory::ReadLease> state_readers_;
    std::vector<PageDirectory::ReadLease> page_readers_;
    std::vector<RowLease> retired_rows_;
    std::vector<PendingCommit> commits_;
    friend class ContextDirectory;
};

[[nodiscard]] std::optional<StateId> reserve_capture_destination(ContextDirectory&,
                                                                 std::optional<RecordReservation>&,
                                                                 std::uint32_t variant = 0);
void prepare_capture_state(ContextTransaction&, StateDirectory&, StateId,
                           std::optional<RecordReservation>&);

struct CaptureWriter {
    StateId image;
    StateId write_destination;
};

[[nodiscard]] CaptureWriter adopt_capture_writer(StateDirectory&, StateId,
                                                 StateDirectory::Destination&);

} // namespace ninfer::models::context
