#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

inline constexpr std::int32_t kPagedKVPageSize = 64;

/** Non-owning, single-sequence view consumed by growing-cache Ops. */
struct PagedKVLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_table;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;
};

/** Non-owning multi-sequence view consumed by batched growing-cache Ops. */
struct PagedKVBatchLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor block_tables;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;
};

/** Rebinds one checked single-sequence table row as a one-row batched view. */
[[nodiscard]] PagedKVBatchLayerView single_row_paged_kv_batch_view(const PagedKVLayerView& cache);

// A plane is storage-only. Target code assigns K/V/layer meaning to plane indices.
struct KVPlaneGeometry {
    DType dtype                 = DType::BF16;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent    = 0;
    std::size_t alignment       = 256;

    friend bool operator==(const KVPlaneGeometry&, const KVPlaneGeometry&) = default;
};

enum class PagedKVPlaneOrder : std::uint8_t {
    PageMajor,
    HeadMajor,
};

struct KVPageGeometry {
    std::uint32_t page_tokens            = kPagedKVPageSize;
    PagedKVPlaneOrder device_plane_order = PagedKVPlaneOrder::PageMajor;
    std::vector<KVPlaneGeometry> planes;

    friend bool operator==(const KVPageGeometry&, const KVPageGeometry&) = default;
};

struct DeviceKVPagePoolSpec {
    std::uint32_t page_group_count = 0;
    KVPageGeometry geometry;
};

struct KVExecutionTableSpec {
    std::uint32_t logical_page_capacity = 0;
    std::int32_t table_rows             = 0;
};

struct DeviceKVPlaneLayout {
    KVPlaneGeometry geometry;
    TensorRegion storage;
};

struct DeviceKVPagePoolLayout {
    DeviceKVPagePoolSpec spec;
    std::vector<DeviceKVPlaneLayout> planes;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

struct KVExecutionTableLayout {
    KVExecutionTableSpec spec;
    TensorRegion block_tables;

    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
};

[[nodiscard]] DeviceKVPagePoolLayout plan_device_kv_page_pool(LayoutBuilder& builder,
                                                              const DeviceKVPagePoolSpec& spec);

[[nodiscard]] KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                              const KVExecutionTableSpec& spec);

class DeviceKVPagePool;
class KVExecutionTablePool;
class HostKVAllocationView;
class HostKVAllocationConstView;

/** Copyable, non-owning physical-page capability minted by one DeviceKVPagePool. */
class DeviceKVPageHandle {
public:
    DeviceKVPageHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

private:
    friend class DeviceKVPagePool;
    friend class DeviceKVPageLease;
    friend class KVExecutionTablePool;

    DeviceKVPageHandle(const DeviceKVPagePool* owner, std::int32_t index,
                       std::uint32_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    const DeviceKVPagePool* owner_ = nullptr;
    std::int32_t index_            = -1;
    std::uint32_t generation_      = 0;
};

/** Move-only owner of one complete Device page-group replica. */
class DeviceKVPageLease {
public:
    DeviceKVPageLease() noexcept = default;
    ~DeviceKVPageLease();

    DeviceKVPageLease(const DeviceKVPageLease&)            = delete;
    DeviceKVPageLease& operator=(const DeviceKVPageLease&) = delete;
    DeviceKVPageLease(DeviceKVPageLease&& other) noexcept;
    DeviceKVPageLease& operator=(DeviceKVPageLease&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] DeviceKVPageHandle handle() const noexcept;
    [[nodiscard]] bool belongs_to(const DeviceKVPagePool& pool) const noexcept;
    bool release() noexcept;

private:
    friend class DeviceKVPagePool;

    DeviceKVPageLease(DeviceKVPagePool& owner, std::int32_t index,
                      std::uint32_t generation) noexcept
        : owner_(&owner), index_(index), generation_(generation) {}

    DeviceKVPagePool* owner_  = nullptr;
    std::int32_t index_       = -1;
    std::uint32_t generation_ = 0;
};

/** Move-only reservation of capacity not yet associated with physical page IDs. */
class DeviceKVPageReservation {
public:
    DeviceKVPageReservation() noexcept = default;
    ~DeviceKVPageReservation();

    DeviceKVPageReservation(const DeviceKVPageReservation&)            = delete;
    DeviceKVPageReservation& operator=(const DeviceKVPageReservation&) = delete;
    DeviceKVPageReservation(DeviceKVPageReservation&& other) noexcept;
    DeviceKVPageReservation& operator=(DeviceKVPageReservation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::uint32_t pages() const noexcept { return pages_; }

    [[nodiscard]] bool belongs_to(const DeviceKVPagePool& pool) const noexcept;
    void clear() noexcept;
    void release() noexcept;

private:
    friend class DeviceKVPagePool;

    DeviceKVPageReservation(DeviceKVPagePool& owner, std::uint32_t pages) noexcept
        : owner_(&owner), pages_(pages) {}

    DeviceKVPagePool* owner_ = nullptr;
    std::uint32_t pages_     = 0;
};

class DeviceKVPagePool {
public:
    DeviceKVPagePool(DeviceSpan backing, const DeviceKVPagePoolLayout& layout);

    DeviceKVPagePool(const DeviceKVPagePool&)            = delete;
    DeviceKVPagePool& operator=(const DeviceKVPagePool&) = delete;
    DeviceKVPagePool(DeviceKVPagePool&&)                 = delete;
    DeviceKVPagePool& operator=(DeviceKVPagePool&&)      = delete;

    [[nodiscard]] const KVPageGeometry& geometry() const noexcept { return spec_.geometry; }

    [[nodiscard]] std::uint32_t capacity_pages() const noexcept;
    [[nodiscard]] std::uint32_t allocated_pages() const noexcept;
    [[nodiscard]] std::uint32_t reserved_pages() const noexcept;
    [[nodiscard]] std::uint32_t available_pages() const noexcept;
    [[nodiscard]] std::size_t plane_count() const noexcept;
    [[nodiscard]] const Tensor& plane(std::size_t index) const;
    [[nodiscard]] std::uint32_t
    contiguous_run_count(std::span<const DeviceKVPageHandle> pages) const;

    [[nodiscard]] std::optional<DeviceKVPageReservation> reserve(std::uint32_t pages) noexcept;

    [[nodiscard]] DeviceKVPageReservation make_empty_reservation() noexcept {
        return DeviceKVPageReservation(*this, 0);
    }

    [[nodiscard]] bool can_resize_reservation(const DeviceKVPageReservation& reservation,
                                              std::uint32_t new_reserved_pages) const noexcept;
    void resize_reservation(DeviceKVPageReservation& reservation, std::uint32_t new_reserved_pages);

    // Grows destination to target_page_count without host allocation or a second capacity check.
    void materialize(DeviceKVPageReservation& reservation, std::uint32_t target_page_count,
                     std::vector<DeviceKVPageLease>& destination,
                     std::optional<DeviceKVPageHandle> preferred_predecessor = std::nullopt);
    // Single-page forms for fixed-capacity logical stores which do not own a growable lease vector.
    [[nodiscard]] DeviceKVPageLease materialize_one(DeviceKVPageReservation& reservation);
    // Returns trailing leases to the same entitlement instead of releasing their capacity.
    void dematerialize(DeviceKVPageReservation& reservation, std::uint32_t target_page_count,
                       std::vector<DeviceKVPageLease>& source);
    void dematerialize_one(DeviceKVPageReservation& reservation, DeviceKVPageLease&& page);

    void zero_pages(std::span<const DeviceKVPageHandle> pages, cudaStream_t stream = nullptr) const;
    void copy_page(DeviceKVPageHandle source, DeviceKVPageHandle destination,
                   cudaStream_t stream = nullptr) const;

    void copy_to_host(std::span<const DeviceKVPageHandle> source, HostKVAllocationView destination,
                      cudaStream_t stream = nullptr) const;
    void copy_from_host(HostKVAllocationConstView source,
                        std::span<const DeviceKVPageHandle> destination,
                        cudaStream_t stream = nullptr) const;

private:
    friend class DeviceKVPageLease;
    friend class DeviceKVPageReservation;
    friend class KVExecutionTablePool;

    [[nodiscard]] bool valid_handle(DeviceKVPageHandle handle) const noexcept;
    [[nodiscard]] std::int32_t physical_index(DeviceKVPageHandle handle) const;
    void validate_distinct_pages(std::span<const DeviceKVPageHandle> pages,
                                 const char* duplicate_message) const;
    void consume_free_run(std::size_t run_index, std::int32_t begin, std::uint32_t count) noexcept;
    void release_free_page(std::int32_t index) noexcept;
    void release_page(std::int32_t index, std::uint32_t generation) noexcept;
    void release_reservation(std::uint32_t pages) noexcept;

    struct FreePageRun {
        std::int32_t begin  = 0;
        std::uint32_t count = 0;
    };

    DeviceKVPagePoolSpec spec_;
    std::vector<Tensor> planes_;
    std::vector<FreePageRun> free_page_runs_;
    std::vector<std::uint32_t> page_generations_;
    std::vector<bool> page_allocated_;
    mutable std::vector<std::uint32_t> validation_marks_;
    mutable std::uint32_t validation_stamp_ = 0;
    std::uint32_t allocated_pages_          = 0;
    std::uint32_t reserved_pages_           = 0;
};

struct DeviceKVPageReservationRequest {
    DeviceKVPagePool* pool = nullptr;
    std::uint32_t pages    = 0;
};

// Reserves every distinct physical pool or leaves every pool unchanged.
[[nodiscard]] std::vector<DeviceKVPageReservation>
reserve_device_kv_page_bundle(std::span<const DeviceKVPageReservationRequest> requests);

class KVExecutionRowHandle {
public:
    KVExecutionRowHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::int32_t row_index() const noexcept { return row_; }

private:
    friend class KVExecutionTablePool;
    friend class KVExecutionRowLease;

    KVExecutionRowHandle(const KVExecutionTablePool* owner, std::int32_t row,
                         std::uint32_t generation) noexcept
        : owner_(owner), row_(row), generation_(generation) {}

    const KVExecutionTablePool* owner_ = nullptr;
    std::int32_t row_                  = -1;
    std::uint32_t generation_          = 0;
};

class KVExecutionRowLease {
public:
    KVExecutionRowLease() noexcept = default;
    ~KVExecutionRowLease();

    KVExecutionRowLease(const KVExecutionRowLease&)            = delete;
    KVExecutionRowLease& operator=(const KVExecutionRowLease&) = delete;
    KVExecutionRowLease(KVExecutionRowLease&& other) noexcept;
    KVExecutionRowLease& operator=(KVExecutionRowLease&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] KVExecutionRowHandle handle() const noexcept;

    [[nodiscard]] std::int32_t row_index() const noexcept { return row_; }

    [[nodiscard]] bool belongs_to(const KVExecutionTablePool& pool) const noexcept;
    bool release() noexcept;

private:
    friend class KVExecutionTablePool;

    KVExecutionRowLease(KVExecutionTablePool& owner, std::int32_t row,
                        std::uint32_t generation) noexcept
        : owner_(&owner), row_(row), generation_(generation) {}

    KVExecutionTablePool* owner_ = nullptr;
    std::int32_t row_            = -1;
    std::uint32_t generation_    = 0;
};

class KVExecutionTablePool {
public:
    KVExecutionTablePool(DeviceSpan backing, const KVExecutionTableLayout& layout,
                         const DeviceKVPagePool& pages);

    KVExecutionTablePool(const KVExecutionTablePool&)            = delete;
    KVExecutionTablePool& operator=(const KVExecutionTablePool&) = delete;
    KVExecutionTablePool(KVExecutionTablePool&&)                 = delete;
    KVExecutionTablePool& operator=(KVExecutionTablePool&&)      = delete;

    [[nodiscard]] std::uint32_t logical_page_capacity() const noexcept;
    [[nodiscard]] std::int32_t row_count() const noexcept;
    [[nodiscard]] KVExecutionRowLease acquire(std::int32_t row);

    void publish(KVExecutionRowHandle row, std::uint32_t logical_begin,
                 std::span<const DeviceKVPageHandle> pages, cudaStream_t stream = nullptr);
    void publish(KVExecutionRowHandle row, std::uint32_t logical_begin,
                 std::span<const DeviceKVPageLease> pages, cudaStream_t stream = nullptr);
    void publish_repeated(KVExecutionRowHandle row, DeviceKVPageHandle page, std::uint32_t count,
                          cudaStream_t stream = nullptr);

    [[nodiscard]] Tensor row(KVExecutionRowHandle handle) const;

    [[nodiscard]] const Tensor& matrix() const noexcept { return block_tables_; }

private:
    friend class KVExecutionRowLease;

    [[nodiscard]] bool valid_handle(KVExecutionRowHandle handle) const noexcept;
    bool release_row(std::int32_t row, std::uint32_t generation) noexcept;
    void publish_indices(KVExecutionRowHandle row, std::uint32_t logical_begin,
                         std::span<const std::int32_t> indices, cudaStream_t stream);

    KVExecutionTableSpec spec_;
    const DeviceKVPagePool* pages_ = nullptr;
    Tensor block_tables_;
    PinnedHostBuffer host_shadow_;
    std::vector<bool> row_in_use_;
    std::vector<std::uint32_t> row_generations_;
};

} // namespace ninfer
