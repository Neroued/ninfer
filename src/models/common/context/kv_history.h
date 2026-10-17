#pragma once

#include "models/common/context/logical_kv_store.h"

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::context {

class KVHistoryStore;

class KVHistoryHandle {
public:
    [[nodiscard]] bool valid() const noexcept { return store_ != nullptr; }

    friend bool operator==(KVHistoryHandle, KVHistoryHandle) noexcept = default;

private:
    KVHistoryStore* store_    = nullptr;
    std::uint32_t index_      = 0;
    std::uint32_t generation_ = 0;
    friend class KVHistoryStore;
    friend class KVHistory;
    friend class KVPrefixRef;
};

// The sole mutable owner of a history. Execution must retire before releasing this owner.
class KVHistory {
public:
    KVHistory() noexcept = default;
    ~KVHistory();
    KVHistory(KVHistory&& other) noexcept;
    KVHistory& operator=(KVHistory&& other) noexcept;
    KVHistory(const KVHistory&)            = delete;
    KVHistory& operator=(const KVHistory&) = delete;

    [[nodiscard]] KVHistoryHandle handle() const noexcept { return handle_; }

    [[nodiscard]] bool valid() const noexcept { return handle_.valid(); }

    void reset() noexcept;

private:
    explicit KVHistory(KVHistoryHandle handle) : handle_(handle) {}

    KVHistoryHandle handle_;
    friend class KVHistoryStore;
};

// Retains [0, frontier) in one actual history. It owns logical content, not Device residency.
class KVPrefixRef {
public:
    KVPrefixRef() noexcept = default;
    ~KVPrefixRef();
    KVPrefixRef(const KVPrefixRef& other);
    KVPrefixRef& operator=(const KVPrefixRef& other);
    KVPrefixRef(KVPrefixRef&& other) noexcept;
    KVPrefixRef& operator=(KVPrefixRef&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return history_.valid(); }

    [[nodiscard]] KVHistoryHandle history() const noexcept { return history_; }

    [[nodiscard]] std::uint32_t frontier() const noexcept { return frontier_; }

    void reset() noexcept;

private:
    KVPrefixRef(KVHistoryHandle history, std::uint32_t frontier)
        : history_(history), frontier_(frontier) {}

    KVHistoryHandle history_;
    std::uint32_t frontier_ = 0;
    friend class KVHistoryStore;
};

// Owns the complete binding reservation. The caller owns the transfer completion dependency;
// enqueued copies must finish before commit or destruction.
class KVBinding {
public:
    KVBinding() noexcept = default;
    ~KVBinding();
    KVBinding(KVBinding&& other) noexcept;
    KVBinding& operator=(KVBinding&&)      = delete;
    KVBinding(const KVBinding&)            = delete;
    KVBinding& operator=(const KVBinding&) = delete;

    [[nodiscard]] KVHistoryHandle history() const noexcept { return history_.handle(); }

    [[nodiscard]] std::span<const LogicalKVPageStore::Restore> restores() const noexcept {
        return restores_;
    }

    [[nodiscard]] LogicalKVPageStore& pages() const noexcept;

    [[nodiscard]] bool needs_tail_copy() const noexcept { return tail_destination_.has_value(); }

    [[nodiscard]] bool blocks_writer(KVHistoryHandle history) const noexcept;
    [[nodiscard]] DeviceKVPageHandle tail_source() const;
    [[nodiscard]] DeviceKVPageHandle tail_destination() const;
    void publish_restores();
    void prepare_tail_copy();
    void enqueue_mapping(cudaStream_t stream);
    [[nodiscard]] KVHistory commit();

private:
    KVHistoryStore* store_ = nullptr;
    KVHistory history_;
    KVPrefixRef source_;
    DeviceKVPageReservation reservation_;
    std::optional<KVExecutionRowLease> row_;
    std::vector<LogicalKVPageStore::Restore> restores_;
    std::optional<LogicalKVPageHandle> tail_destination_;
    LogicalKVPageHandle tail_source_;
    bool tail_pinned_        = false;
    bool restores_published_ = false;
    bool write_tail_         = false;
    std::uint32_t frontier_  = 0;
    bool mapping_enqueued_   = false;
    friend class KVHistoryStore;
};

struct KVBindingDemand {
    std::uint32_t restore_pages = 0;
    std::uint32_t growth_pages  = 0;
    std::uint32_t tail_copies   = 0;
    bool reuse_history          = false;

    [[nodiscard]] std::uint32_t device_pages() const noexcept {
        return restore_pages + growth_pages + tail_copies;
    }
};

class KVHistoryStore {
public:
    KVHistoryStore(LogicalKVPageStore& pages, KVExecutionTablePool& tables,
                   std::uint32_t history_capacity, std::uint32_t page_capacity,
                   HostKVExtentStore* host = nullptr);
    ~KVHistoryStore();
    KVHistoryStore(const KVHistoryStore&)            = delete;
    KVHistoryStore& operator=(const KVHistoryStore&) = delete;

    [[nodiscard]] KVPrefixRef retain_prefix(KVHistoryHandle history, std::uint32_t frontier);
    [[nodiscard]] bool valid(KVHistoryHandle history) const noexcept;

    [[nodiscard]] std::uint32_t descriptor_index(KVHistoryHandle history) const {
        (void)require(history);
        return history.index_;
    }

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t occupied() const noexcept;
    [[nodiscard]] bool owned(KVHistoryHandle history) const;
    [[nodiscard]] bool active(KVHistoryHandle history) const;
    [[nodiscard]] std::uint32_t retained_after(KVHistoryHandle history,
                                               std::span<const KVPrefixRef* const> removed) const;
    [[nodiscard]] std::vector<LogicalKVPageHandle>
    released_pages(std::span<const KVPrefixRef* const> removed) const;

    [[nodiscard]] KVBindingDemand
    binding_demand(const KVPrefixRef* source, std::uint32_t frontier, std::uint32_t peak_frontier,
                   std::span<const KVPrefixRef* const> retired = {}) const;
    [[nodiscard]] KVBinding prepare_binding(const KVPrefixRef* source, std::uint32_t frontier,
                                            std::uint32_t peak_frontier, std::int32_t execution_row,
                                            bool consume_source = false);
    void deactivate(KVHistoryHandle history);
    void reserve_growth(KVHistoryHandle history, std::uint32_t pages);
    void release_growth(KVHistoryHandle history);
    void settle_growth(KVHistory& history, std::uint32_t frontier, std::uint32_t retain_through,
                       cudaStream_t stream);
    [[nodiscard]] std::uint32_t reserved_growth_pages(KVHistoryHandle history) const;
    [[nodiscard]] std::uint32_t growth_pages_for_tokens(KVHistoryHandle history,
                                                        std::uint32_t frontier) const;
    [[nodiscard]] std::uint32_t write_pages(KVHistoryHandle history, std::uint32_t begin,
                                            std::uint32_t end) const;
    void prepare_write(KVHistory& history, std::uint32_t begin, std::uint32_t end,
                       cudaStream_t stream);
    void commit_frontier(KVHistoryHandle history, std::uint32_t frontier);
    void trim(KVHistory& history, std::uint32_t frontier, cudaStream_t stream);

    [[nodiscard]] std::uint32_t committed_frontier(KVHistoryHandle history) const;
    [[nodiscard]] std::uint32_t mapped_pages(KVHistoryHandle history) const;
    [[nodiscard]] std::int32_t bound_row(KVHistoryHandle history) const;
    [[nodiscard]] const KVExecutionRowLease& execution_row(KVHistoryHandle history) const;
    [[nodiscard]] LogicalKVPageHandle logical_page(KVHistoryHandle history,
                                                   std::uint32_t index) const;
    [[nodiscard]] DeviceKVPageHandle physical_page(KVHistoryHandle history,
                                                   std::uint32_t index) const;

    [[nodiscard]] LogicalKVPageStore& pages() const noexcept { return *pages_; }

private:
    static constexpr std::uint32_t kChunkPages = 32;

    struct DirectoryNode {
        std::shared_ptr<DirectoryNode> left;
        std::shared_ptr<DirectoryNode> right;
        std::array<LogicalKVPageHandle, kChunkPages> pages{};
    };

    using Directory = std::shared_ptr<DirectoryNode>;

    struct History {
        Directory directory;
        std::map<std::uint32_t, std::uint32_t> prefixes;
        DeviceKVPageReservation reservation;
        std::optional<KVExecutionRowLease> row;
        std::uint32_t generation   = 1;
        std::uint32_t page_count   = 0;
        std::uint32_t frontier     = 0;
        std::uint32_t retained     = 0;
        std::uint32_t prefix_count = 0;
        bool occupied              = false;
        bool owned                 = false;
    };

    [[nodiscard]] History& require(KVHistoryHandle history);
    [[nodiscard]] const History& require(KVHistoryHandle history) const;
    [[nodiscard]] KVHistoryHandle allocate();
    [[nodiscard]] KVHistory fork_history(const KVPrefixRef& source, std::uint32_t frontier);
    [[nodiscard]] std::uint64_t writer(KVHistoryHandle history) const noexcept;
    void add_prefix(KVHistoryHandle history, std::uint32_t frontier);
    void remove_prefix(KVHistoryHandle history, std::uint32_t frontier) noexcept;
    void release_owner(KVHistoryHandle history) noexcept;
    void collect(KVHistoryHandle history);
    void update_ranges(History& history, std::uint32_t old_visible, std::uint32_t new_visible,
                       std::uint32_t old_protected, std::uint32_t new_protected);
    void truncate_membership(History& history, std::uint32_t count);
    void abort(KVBinding& binding) noexcept;
    [[nodiscard]] KVHistory commit(KVBinding& binding);
    [[nodiscard]] LogicalKVPageHandle membership(const History& history,
                                                 std::uint32_t index) const noexcept;
    [[nodiscard]] LogicalKVPageHandle& directory_slot(Directory& root, std::uint32_t index);
    [[nodiscard]] static Directory prefix_directory(const Directory& root, std::uint64_t width,
                                                    std::uint32_t count);
    void branch_owner(KVHistory& history, std::uint32_t frontier, cudaStream_t stream);

    LogicalKVPageStore* pages_;
    KVExecutionTablePool* tables_;
    HostKVExtentStore* host_;
    std::uint32_t page_capacity_;
    std::uint64_t directory_capacity_ = kChunkPages;
    std::vector<History> histories_;
    std::vector<std::uint32_t> free_;
    std::uint32_t free_count_;
    std::vector<DeviceKVPageHandle> publish_;
    std::vector<LogicalKVPageHandle> materialize_;
    friend class KVHistory;
    friend class KVPrefixRef;
    friend class KVBinding;
};
} // namespace ninfer::models::context
