#pragma once

#include "core/device.h"
#include "core/paged_kv_geometry.h"
#include "core/paged_kv_cache.h"
#include "core/state_image.h"
#include "models/common/context/contract.h"
#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace ninfer::models::context {

// Resolve the target's Host context budget after the native StateImage size is known.
[[nodiscard]] std::size_t host_context_capacity(std::optional<std::size_t> requested,
                                                std::size_t image_stride);

// Borrows fixed rank-local backing. Page ownership and generations belong to the directory.
// Calls are serialized by the rank worker; the caller retains all backing until completion.
class DevicePageStorage {
public:
    DevicePageStorage(MemorySpan backing, const DeviceKVPagePoolLayout& layout);

    [[nodiscard]] Device device() const noexcept { return planes_.front().device; }

    [[nodiscard]] const KVPageGeometry& geometry() const noexcept { return spec_.geometry; }

    [[nodiscard]] const HostKVPageLayout& host_layout() const noexcept { return host_layout_; }

    [[nodiscard]] KVTransferProfile transfer_profile() const {
        return {page_payload_bytes_, static_cast<std::uint32_t>(planes_.size()),
                host_transfer_operations_};
    }

    [[nodiscard]] std::uint32_t capacity_pages() const noexcept { return spec_.page_group_count; }

    [[nodiscard]] std::size_t plane_count() const noexcept { return planes_.size(); }

    [[nodiscard]] const Tensor& plane(std::size_t index) const { return planes_.at(index); }

    void zero_pages(std::span<const std::int32_t> pages, CudaExecutionView execution) const;
    TransferWork copy_page(std::int32_t source, std::int32_t destination,
                           CudaExecutionView execution) const;

    // Host runs are packed records in the canonical layout. Work excludes record padding.
    [[nodiscard]] TransferWork host_transfer_run_work(std::uint32_t pages) const;
    [[nodiscard]] TransferWork host_transfer_work(std::span<const std::int32_t> pages) const;
    [[nodiscard]] TransferWork device_copy_work(std::uint32_t pages) const;

    TransferWork copy_to_host(std::span<const std::int32_t> source, MemorySpan destination,
                              const HostKVPageLayout& layout, CudaExecutionView execution) const;
    TransferWork copy_from_host(ConstMemorySpan source, const HostKVPageLayout& layout,
                                std::span<const std::int32_t> destination,
                                CudaExecutionView execution) const;

private:
    struct HostTransferPlane {
        std::size_t host_offset    = 0;
        std::size_t page_bytes     = 0;
        std::size_t head_bytes     = 0;
        std::uint32_t heads        = 0;
        bool page_copies_supported = false;

        [[nodiscard]] bool copy_by_page(std::size_t pages) const noexcept {
            return page_copies_supported && pages < heads;
        }
    };

    struct HostTransferGroup {
        std::size_t first_plane  = 0;
        std::size_t plane_count  = 1;
        std::size_t device_pitch = 0;
        std::size_t host_pitch   = 0;

        [[nodiscard]] bool copy_by_page(std::size_t pages) const noexcept {
            return pages < plane_count;
        }
    };

    void validate_page(std::int32_t page) const;
    void validate_distinct_pages(std::span<const std::int32_t> pages) const;
    void validate_host_span(ConstMemorySpan memory, const HostKVPageLayout& layout,
                            std::size_t pages) const;
    void initialize_host_transfer_plan();

    template <bool ToHost>
    TransferWork copy_host_pages(std::span<const std::int32_t> pages,
                                 std::conditional_t<ToHost, std::byte*, const std::byte*> host,
                                 CudaExecutionView execution) const;

    DeviceKVPagePoolSpec spec_;
    std::vector<Tensor> planes_;
    HostKVPageLayout host_layout_;
    std::vector<HostTransferPlane> host_transfer_planes_;
    std::vector<HostTransferGroup> host_transfer_groups_;
    // Index min(run pages, size - 1); the final entry also describes all longer runs.
    std::vector<std::uint32_t> host_transfer_operations_;
    std::uint64_t page_payload_bytes_     = 0;
    std::size_t maximum_host_plane_group_ = 1;
    mutable std::vector<std::uint32_t> validation_marks_;
    mutable std::uint32_t validation_stamp_ = 0;
};

// Borrows table backing without allocating rows or retaining a shared Host mapping shadow.
class ExecutionTableStorage {
public:
    ExecutionTableStorage(MemorySpan backing, const KVExecutionTableLayout& layout,
                          const DevicePageStorage& pages);

    [[nodiscard]] std::uint32_t logical_page_capacity() const noexcept {
        return spec_.logical_page_capacity;
    }

    [[nodiscard]] std::int32_t row_count() const noexcept { return spec_.table_rows; }

    [[nodiscard]] Tensor row(std::int32_t index) const;

    [[nodiscard]] const Tensor& matrix() const noexcept { return block_tables_; }

    // Copies IDs into caller-owned pinned staging, then enqueues its upload. The caller keeps
    // that range alive and unchanged until the transfer completes. IDs may alias staging.
    void publish(std::int32_t row, std::uint32_t logical_begin, std::span<const std::int32_t> pages,
                 MemorySpan pinned_staging, CudaExecutionView execution) const;

private:
    KVExecutionTableSpec spec_;
    Tensor block_tables_;
    std::uint32_t page_capacity_ = 0;
};

struct ContextPoolStorage {
    std::uint32_t pool_id         = 0;
    DevicePageStorage* pages      = nullptr;
    ExecutionTableStorage* tables = nullptr;
};

// Combines rank-local physical views. The owning Resources retains all backing; logical IDs,
// reservations, coverage and references remain in the coordinator's ContextDirectory.
class ContextStorage {
public:
    ContextStorage(StateImageStorage& state, std::span<const ContextPoolStorage> pools,
                   std::uint32_t host_pool_id, MemorySpan host_backing);

    [[nodiscard]] StateImageStorage& state() const noexcept { return *state_; }

    [[nodiscard]] const ContextPoolStorage& pool(std::uint32_t id) const;

    [[nodiscard]] std::size_t pool_count() const noexcept { return pools_.size(); }

    [[nodiscard]] MemorySpan host_extent(HostExtent extent) const;
    [[nodiscard]] std::size_t mapping_capacity_elements() const noexcept;
    [[nodiscard]] std::size_t copy_segment_capacity() const noexcept;
    [[nodiscard]] static std::size_t mapping_elements(const ContextActions& actions);
    [[nodiscard]] runtime::ContextTransferObservation describe(const ContextAction& action) const;

    void validate(const ContextActions& actions, ConstMemorySpan mapping_staging) const;
    // mapping_staging remains alive and unchanged until all submitted H2D copies complete.
    void execute(const ContextActions& actions, MemorySpan mapping_staging,
                 CudaExecutionView execution) const;
    // Returns the number of staging bytes consumed by this action.
    std::size_t execute(const ContextAction& action, MemorySpan mapping_staging,
                        CudaExecutionView execution) const;

private:
    [[nodiscard]] std::span<const std::int32_t> page_indices(std::span<const PageSlot> pages) const;
    void validate_slot(PageSlot slot) const;

    StateImageStorage* state_;
    std::vector<ContextPoolStorage> pools_;
    std::uint32_t host_pool_id_;
    MemorySpan host_backing_;
    mutable std::vector<std::int32_t> indices_;
};

} // namespace ninfer::models::context
