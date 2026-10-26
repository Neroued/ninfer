#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

struct RecordClassId {
    std::uint32_t index = UINT32_MAX;

    [[nodiscard]] bool valid() const noexcept { return index != UINT32_MAX; }

    friend bool operator==(RecordClassId, RecordClassId) = default;
};

struct AllocationId {
    std::uint32_t pool_id    = UINT32_MAX;
    std::uint32_t index      = UINT32_MAX;
    std::uint64_t generation = 0;

    [[nodiscard]] bool valid() const noexcept {
        return pool_id != UINT32_MAX && index != UINT32_MAX && generation != 0;
    }

    friend bool operator==(AllocationId, AllocationId) = default;
};

struct RecordSpec {
    std::size_t bytes                              = 0;
    std::size_t alignment                          = 1;
    friend bool operator==(RecordSpec, RecordSpec) = default;
};

struct RecordRequest {
    RecordClassId record_class;
    std::uint32_t count = 0;
};

struct RecordClassGeometry {
    RecordSpec spec;
    std::size_t stride            = 0;
    std::size_t records_per_block = 0;
};

struct ContextPoolGeometry {
    std::size_t alignment   = 1;
    std::size_t block_bytes = 0;
    std::vector<RecordClassId> spec_classes;
    std::vector<RecordClassGeometry> classes;

    [[nodiscard]] std::size_t required_bytes(std::span<const RecordRequest> demand) const;
    [[nodiscard]] std::size_t record_limit(std::size_t capacity_bytes) const;
    // Upper bound for allocator metadata, including one pressure-planning/maintenance window.
    [[nodiscard]] std::size_t metadata_bytes(std::size_t capacity_bytes) const;
};

[[nodiscard]] ContextPoolGeometry describe_context_pool(std::span<const RecordSpec> specs);

struct RecordBinding {
    AllocationId allocation;
    RecordClassId record_class;
    std::uint64_t offset        = 0;
    std::uint64_t bytes         = 0;
    std::uint64_t binding_epoch = 0;
};

struct RecordMove {
    AllocationId allocation;
    std::uint64_t source_offset      = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes              = 0;
};

enum class PoolAvailability {
    Ready,
    NeedsMaintenance,
    WaitingForRetirement,
    InsufficientCapacity,
    MaintenanceInProgress,
};

struct PoolAssessment {
    PoolAvailability status = PoolAvailability::InsufficientCapacity;
    std::uint64_t revision  = 0;
    std::size_t move_bytes  = 0;
    std::size_t move_count  = 0;
    // Whole-block excess after class packing; bounds reclaim ranking, not admission.
    std::size_t reclaim_bytes = 0;

    [[nodiscard]] bool feasible() const noexcept {
        return status == PoolAvailability::Ready || status == PoolAvailability::NeedsMaintenance;
    }
};

struct RecordClassUsage {
    RecordSpec spec;
    std::size_t stride                     = 0;
    std::size_t records_per_block          = 0;
    std::size_t blocks                     = 0;
    std::size_t live_records               = 0;
    std::size_t reserved_records           = 0;
    std::size_t pending_retirement_records = 0;
    std::size_t future_records             = 0;
    std::size_t free_records               = 0;
};

struct ContextPoolOccupancy {
    std::size_t capacity_bytes           = 0;
    std::size_t block_bytes              = 0;
    std::size_t free_block_bytes         = 0;
    std::size_t live_bytes               = 0;
    std::size_t reserved_bytes           = 0;
    std::size_t pending_retirement_bytes = 0;
    std::size_t relocation_bytes         = 0;
    std::size_t future_reserved_bytes    = 0;
    std::size_t free_class_bytes         = 0;
    std::size_t padding_bytes            = 0;
    std::size_t tail_bytes               = 0;
    std::size_t peak_occupied_bytes      = 0;
    std::size_t metadata_bytes           = 0;
    std::vector<RecordClassUsage> classes;
};

class ContextPool;

// Owns real positions, including positions made available by its pending maintenance or retirement.
// The pool outlives every reservation. Materialized allocations are owned by the caller.
class PoolReservation {
public:
    PoolReservation() noexcept;
    ~PoolReservation();
    PoolReservation(PoolReservation&& other) noexcept;
    PoolReservation& operator=(PoolReservation&& other) noexcept;
    PoolReservation(const PoolReservation&)            = delete;
    PoolReservation& operator=(const PoolReservation&) = delete;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] bool belongs_to(const ContextPool& pool) const noexcept;
    [[nodiscard]] std::size_t remaining(RecordClassId record_class) const;
    [[nodiscard]] std::size_t remaining() const noexcept;
    [[nodiscard]] AllocationId materialize(RecordClassId record_class);
    void prepare_returns(RecordClassId record_class, std::size_t count);
    void shrink(RecordClassId record_class, std::size_t remaining_count);
    void cancel() noexcept;

private:
    friend class ContextPool;
    struct Data;
    explicit PoolReservation(std::unique_ptr<Data> data) noexcept;
    std::unique_ptr<Data> data_;
};

struct PoolReserveResult {
    PoolAssessment assessment;
    std::optional<PoolReservation> reservation;
};

// Single control-thread allocator. No backing, execution tickets or payload operations are owned
// here. The directory protects bindings until consumers retire and excludes users during movement.
class ContextPool {
public:
    ContextPool(std::uint32_t pool_id, std::size_t capacity_bytes,
                std::span<const RecordSpec> specs);
    ~ContextPool();
    ContextPool(const ContextPool&)            = delete;
    ContextPool& operator=(const ContextPool&) = delete;
    ContextPool(ContextPool&&)                 = delete;
    ContextPool& operator=(ContextPool&&)      = delete;

    [[nodiscard]] std::uint32_t pool_id() const noexcept;
    [[nodiscard]] std::size_t capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t alignment() const noexcept;
    [[nodiscard]] std::size_t block_bytes() const noexcept;
    [[nodiscard]] std::size_t record_limit() const noexcept;
    // Stable allocation-index extent after reserving additional records, including ID reuse.
    [[nodiscard]] std::size_t allocation_extent(std::size_t additional_records = 0) const noexcept;
    [[nodiscard]] std::uint64_t revision() const noexcept;
    [[nodiscard]] RecordClassId record_class(std::size_t spec_index) const;
    [[nodiscard]] RecordSpec record_spec(RecordClassId record_class) const;
    [[nodiscard]] ContextPoolOccupancy usage() const;

    [[nodiscard]] PoolAssessment assess(std::span<const RecordRequest> demand,
                                        std::span<const AllocationId> releases = {}) const;
    [[nodiscard]] PoolReserveResult reserve(std::span<const RecordRequest> demand);
    // On failure the original reservation is unchanged. On success it is replaced atomically.
    [[nodiscard]] PoolAssessment resize(PoolReservation& reservation,
                                        std::span<const RecordRequest> demand);

    [[nodiscard]] bool valid(AllocationId allocation) const noexcept;
    [[nodiscard]] RecordBinding resolve(AllocationId allocation) const;
    void release(AllocationId allocation);
    void defer_release(AllocationId allocation);
    void retire(AllocationId allocation);
    void cancel_deferred_release(AllocationId allocation);
    void return_to_reservation(AllocationId allocation, PoolReservation& reservation);
    void defer_to_reservation(AllocationId allocation, PoolReservation& reservation);

    // A prepared plan locks its source/destination blocks. Submit only after the domain's address
    // borrows retire. Completion follows payload copies and execution-table patches. Once begun,
    // cancellation cannot undo movement: complete it, then discard the requesting reservation.
    [[nodiscard]] bool has_maintenance() const noexcept;
    [[nodiscard]] std::span<const RecordMove> maintenance_moves() const noexcept;
    void begin_maintenance();
    void complete_maintenance();
    void cancel_maintenance();
    // The owning instance has failed and all physical consumers have drained. Original sources
    // remain intact; partially patched execution tables must never be submitted again.
    void abandon_maintenance_after_failure();

private:
    friend class PoolReservation;
    friend struct ContextPoolGeometry;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
