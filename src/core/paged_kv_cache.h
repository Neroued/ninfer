#pragma once

#include "core/paged_kv_geometry.h"
#include "core/paged_kv_storage.h"
#include "core/record_view.h"

#include <cstddef>
#include <cstdint>

namespace ninfer {

struct PagedPlaneView {
    PagedRecordView records;
    KVPlaneLayout plane;

    [[nodiscard]] DType dtype() const noexcept { return plane.tensor.dtype; }

    [[nodiscard]] const std::array<std::int32_t, 4>& shape() const noexcept {
        return plane.tensor.shape;
    }

    [[nodiscard]] Device device() const noexcept { return records.storage.device; }

    [[nodiscard]] std::size_t row_stride_bytes() const { return plane.row_stride_bytes(); }

    [[nodiscard]] std::size_t head_stride_bytes() const noexcept { return plane.head_stride_bytes; }

    [[nodiscard]] void* data() const {
        if (records.storage.data == nullptr) { return nullptr; }
        return records.storage.subspan(plane.tensor.region.offset).data;
    }

    [[nodiscard]] Tensor tensor(std::int64_t byte_offset) const {
        if (byte_offset < 0) { throw std::out_of_range("KV record offset is negative"); }
        return plane.tensor.bind(records.storage.subspan(static_cast<std::size_t>(byte_offset)));
    }
};

struct ConstPagedPlaneView {
    ConstPagedRecordView records;
    KVPlaneLayout plane;

    ConstPagedPlaneView() = default;

    ConstPagedPlaneView(const PagedPlaneView& view) : records(view.records), plane(view.plane) {}

    ConstPagedPlaneView(ConstPagedRecordView records, KVPlaneLayout plane)
        : records(records), plane(plane) {}

    [[nodiscard]] DType dtype() const noexcept { return plane.tensor.dtype; }

    [[nodiscard]] const std::array<std::int32_t, 4>& shape() const noexcept {
        return plane.tensor.shape;
    }

    [[nodiscard]] Device device() const noexcept { return records.storage.device; }

    [[nodiscard]] std::size_t row_stride_bytes() const { return plane.row_stride_bytes(); }

    [[nodiscard]] std::size_t head_stride_bytes() const noexcept { return plane.head_stride_bytes; }

    [[nodiscard]] const void* data() const {
        if (records.storage.data == nullptr) return nullptr;
        return records.storage.subspan(plane.tensor.region.offset).data;
    }

    [[nodiscard]] ConstMemorySpan memory(std::int64_t byte_offset) const {
        if (byte_offset < 0) throw std::out_of_range("KV record offset is negative");
        return plane.tensor.region.bind(
            records.storage.subspan(static_cast<std::size_t>(byte_offset)));
    }
};

// Every present plane shares the same arena and I64 page-offset table.
struct PagedKVLayerView {
    PagedPlaneView k_pages;
    PagedPlaneView v_pages;
    PagedPlaneView k_scale_pages;
    PagedPlaneView v_scale_pages;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;
};

struct PagedKVBatchLayerView {
    PagedPlaneView k_pages;
    PagedPlaneView v_pages;
    PagedPlaneView k_scale_pages;
    PagedPlaneView v_scale_pages;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;

    [[nodiscard]] Device device() const;
};

struct ConstPagedKVLayerView {
    ConstPagedPlaneView k_pages;
    ConstPagedPlaneView v_pages;
    ConstPagedPlaneView k_scale_pages;
    ConstPagedPlaneView v_scale_pages;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;

    ConstPagedKVLayerView() = default;

    ConstPagedKVLayerView(const PagedKVLayerView& view)
        : k_pages(view.k_pages), v_pages(view.v_pages), k_scale_pages(view.k_scale_pages),
          v_scale_pages(view.v_scale_pages), head_dim(view.head_dim),
          num_kv_heads(view.num_kv_heads), storage(view.storage) {}
};

struct ConstPagedKVBatchLayerView {
    ConstPagedPlaneView k_pages;
    ConstPagedPlaneView v_pages;
    ConstPagedPlaneView k_scale_pages;
    ConstPagedPlaneView v_scale_pages;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    KvCacheStorage storage    = KvCacheStorage::BFloat16;

    ConstPagedKVBatchLayerView() = default;

    ConstPagedKVBatchLayerView(const PagedKVBatchLayerView& view)
        : k_pages(view.k_pages), v_pages(view.v_pages), k_scale_pages(view.k_scale_pages),
          v_scale_pages(view.v_scale_pages), head_dim(view.head_dim),
          num_kv_heads(view.num_kv_heads), storage(view.storage) {}

    [[nodiscard]] Device device() const;
};

[[nodiscard]] ConstPagedKVBatchLayerView
single_row_paged_kv_batch_view(const ConstPagedKVLayerView& cache);

[[nodiscard]] PagedKVBatchLayerView single_row_paged_kv_batch_view(const PagedKVLayerView& cache);

struct KVExecutionTableSpec {
    std::uint32_t logical_page_capacity = 0;
    std::int32_t table_rows             = 0;
};

struct KVExecutionTableLayout {
    KVExecutionTableSpec spec;
    TensorRegion page_offsets;
};

[[nodiscard]] KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                              const KVExecutionTableSpec& spec);

} // namespace ninfer
