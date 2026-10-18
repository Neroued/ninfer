#pragma once

#include "core/layout.h"
#include "core/paged_kv_storage.h"
#include "core/paged_kv_geometry.h"
#include "core/tensor.h"


#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

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

    [[nodiscard]] Device device() const;
};

/** Rebinds one checked single-sequence table row as a one-row batched view. */
[[nodiscard]] PagedKVBatchLayerView single_row_paged_kv_batch_view(const PagedKVLayerView& cache);

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
};

[[nodiscard]] DeviceKVPagePoolLayout plan_device_kv_page_pool(LayoutBuilder& builder,
                                                              const DeviceKVPagePoolSpec& spec);

[[nodiscard]] KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                              const KVExecutionTableSpec& spec);

} // namespace ninfer
