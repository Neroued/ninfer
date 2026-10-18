#include "core/paged_kv_cache.h"


#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {

Device PagedKVBatchLayerView::device() const {
    const auto location = k_pages.device;
    if (v_pages.device != location || block_tables.device != location ||
        (k_scale_pages.data && k_scale_pages.device != location) ||
        (v_scale_pages.data && v_scale_pages.device != location)) {
        throw std::invalid_argument("paged KV view spans different devices");
    }
    return location;
}

namespace {

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must fit positive int32");
    }
    return static_cast<std::int32_t>(value);
}

std::size_t checked_table_bytes(const KVExecutionTableSpec& spec) {
    const std::size_t logical =
        checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity");
    if (spec.table_rows <= 0) {
        throw std::invalid_argument("Paged KV table row count must be positive");
    }
    const std::size_t rows = static_cast<std::size_t>(spec.table_rows);
    if (logical > std::numeric_limits<std::size_t>::max() / rows / sizeof(std::int32_t)) {
        throw std::overflow_error("Paged KV execution table size overflow");
    }
    return logical * rows * sizeof(std::int32_t);
}

void validate_geometry(const KVPageGeometry& geometry) {
    if (geometry.page_tokens != static_cast<std::uint32_t>(kPagedKVPageSize)) {
        throw std::invalid_argument("Paged KV device geometry requires 64-token pages");
    }
    if (geometry.planes.empty()) { throw std::invalid_argument("Paged KV geometry has no planes"); }
    for (const KVPlaneGeometry& plane : geometry.planes) {
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Paged KV plane extents must be positive");
        }
        (void)kv_plane_rows_per_page(geometry, plane);
    }
}

} // namespace

PagedKVBatchLayerView single_row_paged_kv_batch_view(const PagedKVLayerView& cache) {
    return {
        .k_pages       = cache.k_pages,
        .v_pages       = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages,
        .v_scale_pages = cache.v_scale_pages,
        .block_tables  = cache.block_table.view({cache.block_table.ne[0], 1}),
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .storage       = cache.storage,
    };
}

DeviceKVPagePoolLayout plan_device_kv_page_pool(LayoutBuilder& builder,
                                                const DeviceKVPagePoolSpec& spec) {
    const std::int32_t physical_pages =
        checked_i32(spec.page_group_count, "Paged KV physical page count");
    validate_geometry(spec.geometry);

    DeviceKVPagePoolLayout layout;
    layout.spec = spec;
    layout.planes.reserve(spec.geometry.planes.size());
    for (std::size_t index = 0; index < spec.geometry.planes.size(); ++index) {
        const KVPlaneGeometry& plane = spec.geometry.planes[index];
        const std::int32_t rows      = kv_plane_rows_per_page(spec.geometry, plane);
        DeviceKVPlaneLayout planned;
        planned.geometry        = plane;
        const std::string label = "Paged KV plane " + std::to_string(index);
        if (spec.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
            planned.storage = builder.add_tensor(
                plane.dtype, {plane.leading_extent, rows, plane.head_extent, physical_pages},
                plane.alignment, label);
        } else {
            planned.storage = builder.add_tensor(
                plane.dtype, {plane.leading_extent, rows, physical_pages, plane.head_extent},
                plane.alignment, label);
        }
        layout.planes.push_back(planned);
    }
    return layout;
}

KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                const KVExecutionTableSpec& spec) {
    (void)checked_table_bytes(spec);
    KVExecutionTableLayout layout;
    layout.spec         = spec;
    layout.block_tables = builder.add_tensor(
        DType::I32,
        {checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity"),
         spec.table_rows},
        256, "Paged KV execution tables");
    return layout;
}

std::size_t DeviceKVPagePoolLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const DeviceKVPlaneLayout& plane : planes) { total += plane.storage.region.bytes; }
    return total;
}

} // namespace ninfer
