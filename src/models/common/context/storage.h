#pragma once

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "models/common/context/contract.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::context {

[[nodiscard]] std::size_t host_context_capacity(std::optional<std::size_t> requested,
                                                std::size_t image_bytes);

// Borrows fixed execution-table storage. Row ownership and logical histories remain in
// the directory; table entries are byte offsets into the rank's Context arena.
class ExecutionTableStorage {
public:
    ExecutionTableStorage(MemorySpan backing, const KVExecutionTableLayout& layout,
                          MemorySpan context_arena, std::size_t record_bytes);

    [[nodiscard]] std::uint32_t logical_page_capacity() const noexcept {
        return spec_.logical_page_capacity;
    }

    [[nodiscard]] std::int32_t row_count() const noexcept { return spec_.table_rows; }

    [[nodiscard]] Tensor row(std::int32_t index) const;

    [[nodiscard]] const Tensor& matrix() const noexcept { return page_offsets_; }

    void validate(std::int32_t row, std::uint32_t logical_begin,
                  std::span<const std::int64_t> offsets) const;

    // Staging is pinned, caller-owned, and remains unchanged until upload completion.
    // A -1 entry is invalid; the caller must ensure that execution cannot access it.
    void publish(std::int32_t row, std::uint32_t logical_begin,
                 std::span<const std::int64_t> offsets, MemorySpan pinned_staging,
                 CudaExecutionView execution) const;

private:
    KVExecutionTableSpec spec_;
    Tensor page_offsets_;
    MemorySpan context_arena_;
    std::size_t record_bytes_;
    friend class ContextStorage;
};

struct ContextTableStorage {
    std::uint32_t pool_id         = 0;
    ExecutionTableStorage* tables = nullptr;
};

// Borrows rank-owned Device/Host arenas and fixed tables. Allocation, logical identity,
// address borrowing, maintenance exclusion and publication belong to the directory.
class ContextStorage {
public:
    ContextStorage(MemorySpan device_arena, MemorySpan host_arena,
                   std::span<const ContextTableStorage> tables, std::uint32_t logical_rank = 0);

    [[nodiscard]] Device device() const noexcept { return device_arena_.device; }

    [[nodiscard]] MemorySpan range(ContextRange range) const;
    [[nodiscard]] const ExecutionTableStorage& tables(std::uint32_t pool_id) const;
    [[nodiscard]] std::size_t mapping_capacity_elements() const noexcept;
    [[nodiscard]] std::size_t copy_segment_capacity() const noexcept;
    [[nodiscard]] static std::size_t mapping_elements(const ContextActions& actions);
    [[nodiscard]] static bool uses_cuda(const ContextActions& actions) noexcept;
    [[nodiscard]] runtime::ContextTransferObservation describe(const ContextAction& action) const;

    void validate(const ContextActions& actions, ConstMemorySpan mapping_staging) const;
    void execute(const ContextActions& actions, MemorySpan mapping_staging,
                 CudaExecutionView execution) const;
    // Returns staging bytes consumed by this action. Call validate before submitting a batch.
    std::size_t execute(const ContextAction& action, MemorySpan mapping_staging,
                        CudaExecutionView execution) const;

private:
    std::uint32_t logical_rank_;
    MemorySpan device_arena_;
    MemorySpan host_arena_;
    std::vector<ContextTableStorage> tables_;
    std::size_t mapping_capacity_ = 0;
};

} // namespace ninfer::models::context
