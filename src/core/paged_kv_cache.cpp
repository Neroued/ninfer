#include "core/paged_kv_cache.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value == 0 || value > static_cast<std::uint32_t>(INT32_MAX)) {
        throw std::invalid_argument(std::string(label) + " must fit positive int32");
    }
    return static_cast<std::int32_t>(value);
}

PagedPlaneView single_row(PagedPlaneView plane) {
    if (plane.records.page_offsets.data != nullptr) {
        const auto& table          = plane.records.page_offsets;
        plane.records.page_offsets = table.view({table.ne[0], 1});
    }
    return plane;
}

} // namespace

Device PagedKVBatchLayerView::device() const {
    const auto location = k_pages.device();
    for (const auto* plane : {&k_pages, &v_pages, &k_scale_pages, &v_scale_pages}) {
        if (plane->records.storage.data != nullptr &&
            (plane->device() != location || plane->records.page_offsets.device != location)) {
            throw std::invalid_argument("Paged KV view spans different devices");
        }
    }
    return location;
}

PagedKVBatchLayerView single_row_paged_kv_batch_view(const PagedKVLayerView& cache) {
    return {
        .k_pages       = single_row(cache.k_pages),
        .v_pages       = single_row(cache.v_pages),
        .k_scale_pages = single_row(cache.k_scale_pages),
        .v_scale_pages = single_row(cache.v_scale_pages),
        .head_dim      = cache.head_dim,
        .num_kv_heads  = cache.num_kv_heads,
        .storage       = cache.storage,
    };
}

Device ConstPagedKVBatchLayerView::device() const {
    const auto location = k_pages.device();
    for (const auto* plane : {&k_pages, &v_pages, &k_scale_pages, &v_scale_pages}) {
        if (plane->records.storage.data != nullptr &&
            (plane->device() != location || plane->records.page_offsets.device != location)) {
            throw std::invalid_argument("Paged KV view spans different devices");
        }
    }
    return location;
}

ConstPagedKVBatchLayerView single_row_paged_kv_batch_view(const ConstPagedKVLayerView& cache) {
    ConstPagedKVBatchLayerView result;
    result.k_pages       = cache.k_pages;
    result.v_pages       = cache.v_pages;
    result.k_scale_pages = cache.k_scale_pages;
    result.v_scale_pages = cache.v_scale_pages;
    result.head_dim      = cache.head_dim;
    result.num_kv_heads  = cache.num_kv_heads;
    result.storage       = cache.storage;
    for (auto* plane :
         {&result.k_pages, &result.v_pages, &result.k_scale_pages, &result.v_scale_pages}) {
        auto& table = plane->records.page_offsets;
        if (table.data != nullptr) table = table.view({table.ne[0], 1});
    }
    return result;
}

KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                const KVExecutionTableSpec& spec) {
    const auto capacity = checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity");
    if (spec.table_rows <= 0) {
        throw std::invalid_argument("Paged KV table row count must be positive");
    }
    return {spec, builder.add_tensor(DType::I64, {capacity, spec.table_rows}, 256,
                                     "Paged KV execution offsets")};
}

} // namespace ninfer
