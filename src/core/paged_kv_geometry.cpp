#include "core/paged_kv_geometry.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {

std::int32_t kv_plane_rows_per_page(const KVPageGeometry& page, const KVPlaneGeometry& plane) {
    const std::uint32_t span = plane.tokens_per_row;
    if (span != 1 && span != 4) {
        throw std::invalid_argument("Paged KV plane token span must be 1 or 4");
    }
    if (page.page_tokens == 0 || page.page_tokens % span != 0) {
        throw std::invalid_argument("Paged KV page token span must be divisible by its row span");
    }
    const auto rows = page.page_tokens / span;
    if (rows > static_cast<std::uint32_t>(INT32_MAX)) {
        throw std::invalid_argument("Paged KV plane rows per page must fit positive int32");
    }
    return static_cast<std::int32_t>(rows);
}

KVPageLayout plan_kv_page_layout(const KVPageGeometry& geometry) {
    if (geometry.page_tokens == 0 || geometry.planes.empty()) {
        throw std::invalid_argument("KV page geometry is empty");
    }
    KVPageLayout out;
    out.geometry = geometry;
    out.planes.reserve(geometry.planes.size());
    LayoutBuilder record;
    for (const auto& plane : geometry.planes) {
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("KV plane geometry must be positive");
        }
        const auto rows = kv_plane_rows_per_page(geometry, plane);
        auto tensor =
            record.add_tensor(plane.dtype, {plane.leading_extent, rows, plane.head_extent},
                              plane.alignment, "KV page plane");
        out.alignment          = std::max(out.alignment, plane.alignment);
        const auto head_stride = tensor.region.bytes / static_cast<std::size_t>(plane.head_extent);
        out.planes.push_back({plane, tensor, head_stride});
    }
    out.page_bytes = record.finish(out.alignment, "KV page record");
    if (out.page_bytes > static_cast<std::size_t>(INT64_MAX)) {
        throw std::overflow_error("KV page record exceeds int64");
    }
    return out;
}

std::size_t KVPageLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const auto& plane : planes) { total += plane.tensor.region.bytes; }
    return total;
}

} // namespace ninfer
