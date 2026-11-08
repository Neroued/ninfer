#pragma once

#include "core/arena.h"
#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
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
    // Active KV in RAM (--kv-ram-active): the plane lives in pinned, device-mapped Host memory
    // (a UVA pointer kernels dereference over PCIe) instead of the pool's device backing. Page
    // bookkeeping, tables and copies are unchanged; only where the bytes are differs.
    bool host = false;

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

    // Device bytes of the planes (Host planes excluded).
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

struct KVExecutionTableLayout {
    KVExecutionTableSpec spec;
    TensorRegion block_tables;

    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
};

// Host planes (KVPlaneGeometry::host) are planned in `host_builder`, which must then be given.
[[nodiscard]] DeviceKVPagePoolLayout
plan_device_kv_page_pool(LayoutBuilder& builder, const DeviceKVPagePoolSpec& spec,
                         LayoutBuilder* host_builder = nullptr);

[[nodiscard]] KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                              const KVExecutionTableSpec& spec);

class DeviceKVPagePool;
class ElasticArena;
class KVExecutionTablePool;
class HostKVAllocationView;
struct HostKVPageLayout;
class CudaCompletionEvent;
class HostKVAllocationConstView;

/**
 * A second device's replica of one pool, kept byte-identical in *addressing* by repeating every
 * physical-page copy/zero and every execution-table publication onto it. Tensor-parallel rank 1
 * owns its own halved KV planes and tables but no page/address bookkeeping, so rank 0's pool
 * replays each addressing mutation onto rank 1's pool at the same physical indices.
 */
struct DeviceKVMirror {
    int device          = -1;
    cudaStream_t stream = nullptr;
    // Rank 1's Host transfers run here, so a spill or restore does not stall its compute stream.
    // Each one first waits, through `transfer_order`, for everything already issued on `stream`
    // (rank 1's kernels and the replayed page copies); the caller awaits its completion.
    cudaStream_t transfer_stream        = nullptr;
    CudaCompletionEvent* transfer_order = nullptr;
};

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
    // `host` backs the Host planes (pinned, device-mapped memory laid out by the host builder).
    DeviceKVPagePool(DeviceSpan backing, const DeviceKVPagePoolLayout& layout,
                     DeviceSpan host = {});

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

    /**
     * Elastic backing (T64): only pages [0, backed_pages()) have physical memory. Before a page
     * at or beyond it is handed out (materialize, materialize_one), `grow(pages)` is called with
     * the number of leading pages that must then be backed; it backs them and records it with
     * set_backed_pages(), or throws, leaving the pool unchanged. Without a backing every page is
     * backed. allocated_extent() is one past the highest allocated page, so a caller may release
     * the backing of the pages after it.
     */
    void set_backing(std::uint32_t backed_pages, std::function<void(std::uint32_t)> grow);
    void set_backed_pages(std::uint32_t pages);

    [[nodiscard]] std::uint32_t backed_pages() const noexcept { return backed_pages_; }

    [[nodiscard]] std::uint32_t allocated_extent() const noexcept;

    /**
     * Elastic Host planes (active KV in RAM): `arena` (ElasticMemory::Host) is the memory this
     * pool's Host planes were bound to, with one elastic range per Host plane, in plane order,
     * each the plane's whole extent. Only the leading host_backed_pages() pages of every Host
     * plane then have RAM: before a page at or beyond them is handed out (materialize,
     * materialize_one) they grow to the next multiple of `step_pages` (and so does the mirror's,
     * whose own set_host_backing must have been called with the same step), or the call throws
     * and leaves the pool unchanged. trim_host_backing(), between windows, releases the RAM of
     * the pages above the allocated extent (rounded up to the step) on this pool and its mirror,
     * and returns the pages released; it synchronizes the devices whose Host memory shrinks.
     * Without an arena the Host planes' memory is the caller's and fully backed.
     */
    void set_host_backing(ElasticArena& arena, std::uint32_t step_pages,
                          std::uint32_t initial_pages);
    std::uint32_t trim_host_backing();

    [[nodiscard]] std::uint32_t host_backed_pages() const noexcept {
        return host_arena_ != nullptr ? host_backed_pages_ : spec_.page_group_count;
    }

    // Physical bytes of this pool's Host arena (0 without one); the mirror's are its own.
    [[nodiscard]] std::size_t host_physical_bytes() const noexcept;

    /**
     * Compaction of the Device KV backing (v0.6, idle KV to the Host): moves each lease whose page
     * lies above the lowest free page to that free page, in the given order (ascending page order
     * keeps neighbouring pages neighbours), copying the bytes on `stream` and on the mirror's
     * stream (rank 1 keeps the same physical indices) before the old pages are returned. Every
     * destination is taken before any source is returned, so no copy of the batch reads a page
     * another one writes. Allocation and reservation counts are unchanged; only physical indices
     * move, so the caller republishes every execution row that names a moved page before the next
     * launch. Returns the number of leases moved.
     */
    std::uint32_t relocate_lower(std::span<DeviceKVPageLease* const> leases,
                                 cudaStream_t stream = nullptr);

    [[nodiscard]] std::int32_t page_index(DeviceKVPageHandle handle) const {
        return physical_index(handle);
    }

    // Every zero_pages/copy_page issued here is repeated at the same physical indices on `mirror`
    // (another device's pool of identical page capacity) on `where.stream`. Host transfers move
    // both ranks' halves: a Host page record holds host_replicas() copies of the page, rank 0's
    // first, and the mirror's half travels on `where.transfer_stream`.
    void attach_mirror(const DeviceKVPagePool& mirror, DeviceKVMirror where);

    void zero_pages(std::span<const DeviceKVPageHandle> pages, cudaStream_t stream = nullptr) const;
    void copy_page(DeviceKVPageHandle source, DeviceKVPageHandle destination,
                   cudaStream_t stream = nullptr) const;

    [[nodiscard]] std::uint32_t host_replicas() const noexcept {
        return mirror_ != nullptr ? 2U : 1U;
    }

    /**
     * Active KV in RAM: called with the stream of every write this pool issues into page bytes
     * (zero, copy, relocation, Host/disk restore) and, through the execution tables, of every
     * table publication that rebinds an already published logical page. The listener invalidates
     * device caches keyed by logical page (decoder_state.cpp); it must only enqueue on `stream`.
     */
    void set_write_listener(std::function<void(cudaStream_t)> listener) {
        write_listener_ = std::move(listener);
    }

    void notify_write(cudaStream_t stream) const {
        if (write_listener_) { write_listener_(stream); }
    }

    void copy_to_host(std::span<const DeviceKVPageHandle> source, HostKVAllocationView destination,
                      cudaStream_t stream = nullptr) const;
    void copy_from_host(HostKVAllocationConstView source,
                        std::span<const DeviceKVPageHandle> destination,
                        cudaStream_t stream = nullptr) const;
    // The same transfers to/from caller memory laid out as `host` (one record per page, every
    // rank's half), for the disk tier of the context cache (v0.7.2). The memory must stay valid
    // and, for asynchronous copies, page-locked until the caller has synchronized both devices.
    void copy_to_host_buffer(std::span<const DeviceKVPageHandle> source, std::byte* destination,
                             const HostKVPageLayout& host, cudaStream_t stream = nullptr) const;
    void copy_from_host_buffer(const std::byte* source, const HostKVPageLayout& host,
                               std::span<const DeviceKVPageHandle> destination,
                               cudaStream_t stream = nullptr) const;

private:
    friend class DeviceKVPageLease;
    friend class DeviceKVPageReservation;
    friend class KVExecutionTablePool;

    [[nodiscard]] bool valid_handle(DeviceKVPageHandle handle) const noexcept;
    [[nodiscard]] std::int32_t physical_index(DeviceKVPageHandle handle) const;
    void zero_run(std::int32_t first, std::int32_t count, cudaStream_t stream) const;
    void copy_run(std::int32_t source_index, std::int32_t destination_index, cudaStream_t stream,
                  std::int32_t count = 1) const;
    void require_no_mirror(const char* what) const;
    void require_host_replicas(const HostKVPageLayout& host) const;
    void order_mirror_transfer() const;
    void copy_to_host_local(std::span<const DeviceKVPageHandle> source, std::byte* destination,
                            const HostKVPageLayout& host, std::size_t replica_offset,
                            cudaStream_t stream) const;
    void copy_from_host_local(const std::byte* source, const HostKVPageLayout& host,
                              std::size_t replica_offset,
                              std::span<const DeviceKVPageHandle> destination,
                              cudaStream_t stream) const;
    void validate_distinct_pages(std::span<const DeviceKVPageHandle> pages,
                                 const char* duplicate_message) const;
    void consume_free_run(std::size_t run_index, std::int32_t begin, std::uint32_t count) noexcept;
    void release_free_page(std::int32_t index) noexcept;
    void release_page(std::int32_t index, std::uint32_t generation) noexcept;
    void release_reservation(std::uint32_t pages) noexcept;
    void require_backed(std::uint32_t pages);
    // Backs exactly `pages` leading pages of every Host plane of this pool (not the mirror's).
    void map_host_pages(std::uint32_t pages) const;
    void resize_host_backing(std::uint32_t pages);

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
    const DeviceKVPagePool* mirror_         = nullptr;
    DeviceKVMirror mirror_where_;
    std::uint32_t backed_pages_ = 0;
    std::function<void(std::uint32_t)> grow_backing_;
    std::function<void(cudaStream_t)> write_listener_;
    // Elastic Host planes: the arena and its prefixes (mutable: rank 0 resizes its mirror's,
    // like it replays the mirror's page copies, through the const mirror pointer).
    ElasticArena* host_arena_ = nullptr;
    std::vector<std::size_t> host_page_bytes_; // per arena range
    std::uint32_t host_step_pages_           = 0;
    mutable std::uint32_t host_backed_pages_ = 0;
    mutable std::vector<std::size_t> host_mapped_;
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

    // Every acquire/release and every publication into a row here is repeated on the same row of
    // `mirror` (another device's table pool of identical geometry); publications are issued on
    // `where.stream`. While attached, this pool owns the mirror's row leases: the peer reads its
    // row through mirror_row(), never through mirror.acquire(). Neither pool may have a bound row
    // at attach time, and the mirror must outlive this pool.
    void attach_mirror(KVExecutionTablePool& mirror, DeviceKVMirror where);
    // The mirror pool's lease on `row`'s index, held exactly as long as this pool's lease on it.
    [[nodiscard]] const KVExecutionRowLease& mirror_row(KVExecutionRowHandle row) const;

    void publish(KVExecutionRowHandle row, std::uint32_t logical_begin,
                 std::span<const DeviceKVPageHandle> pages, cudaStream_t stream);
    void publish(KVExecutionRowHandle row, std::uint32_t logical_begin,
                 std::span<const DeviceKVPageLease> pages, cudaStream_t stream);
    void publish_repeated(KVExecutionRowHandle row, DeviceKVPageHandle page, std::uint32_t count,
                          cudaStream_t stream);

    [[nodiscard]] Tensor row(KVExecutionRowHandle handle) const;

    [[nodiscard]] const Tensor& matrix() const noexcept { return block_tables_; }

private:
    friend class KVExecutionRowLease;

    [[nodiscard]] bool valid_handle(KVExecutionRowHandle handle) const noexcept;
    bool release_row(std::int32_t row, std::uint32_t generation) noexcept;
    void publish_indices(KVExecutionRowHandle row, std::uint32_t logical_begin,
                         std::span<const std::int32_t> indices, cudaStream_t stream,
                         bool rebinds = false);
    // Starts a publication of `count` entries from `logical_begin` of `row`: records the row's
    // published extent before it in published_before_ (entries below it that change rebind a
    // logical page, for the page pool's write listener), extends it, and returns whether this is
    // the row's first publication since it was acquired.
    bool begin_publication(std::int32_t row, std::uint32_t logical_begin, std::size_t count);

    KVExecutionTableSpec spec_;
    const DeviceKVPagePool* pages_ = nullptr;
    Tensor block_tables_;
    PinnedHostBuffer host_shadow_;
    std::vector<bool> row_in_use_;
    std::vector<std::uint32_t> row_generations_;
    std::vector<std::uint32_t> published_extent_;
    std::uint32_t published_before_ = 0;
    KVExecutionTablePool* mirror_   = nullptr;
    DeviceKVMirror mirror_where_;
    std::vector<std::optional<KVExecutionRowLease>> mirror_rows_;
};

} // namespace ninfer
