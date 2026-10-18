#include "core/paged_kv_geometry.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kHostKVAlignment = 256;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string(label) + " alignment must be a power of two");
    }
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error(std::string(label) + " alignment overflow");
    }
    return (value + mask) & ~mask;
}

} // namespace

std::int32_t kv_plane_rows_per_page(const KVPageGeometry& page, const KVPlaneGeometry& plane) {
    const std::uint32_t span = plane.tokens_per_row;
    if (span != 1 && span != 4) {
        throw std::invalid_argument("Paged KV plane token span must be 1 or 4");
    }
    if (page.page_tokens == 0 || page.page_tokens % span != 0) {
        throw std::invalid_argument("Paged KV page token span must be divisible by its row span");
    }
    if (span == 4 && page.device_plane_order != PagedKVPlaneOrder::PageMajor) {
        throw std::invalid_argument("Paged KV four-token rows require PageMajor storage");
    }
    const auto rows = page.page_tokens / span;
    if (!rows || rows > static_cast<std::uint32_t>(INT32_MAX))
        throw std::invalid_argument("Paged KV plane rows per page must fit positive int32");
    return static_cast<std::int32_t>(rows);
}

HostKVPageLayout plan_host_kv_page_layout(const KVPageGeometry& geometry) {
    if (geometry.page_tokens == 0 || geometry.planes.empty()) {
        throw std::invalid_argument("Host KV page geometry is empty");
    }

    HostKVPageLayout out;
    out.geometry = geometry;
    out.planes.reserve(geometry.planes.size());
    std::size_t cursor = 0;
    for (const KVPlaneGeometry& plane : geometry.planes) {
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Host KV plane geometry must be positive");
        }
        const auto rows = static_cast<std::size_t>(kv_plane_rows_per_page(geometry, plane));
        // Device slab alignment is not part of the canonical packed Host representation.
        cursor = align_up(cursor, kHostKVAlignment, "Host KV plane");
        const std::size_t head_bytes =
            checked_mul(checked_mul(static_cast<std::size_t>(plane.leading_extent), rows,
                                    "Host KV head payload overflow"),
                        dtype_size(plane.dtype), "Host KV head payload overflow");
        const std::size_t page_bytes =
            checked_mul(head_bytes, static_cast<std::size_t>(plane.head_extent),
                        "Host KV plane payload overflow");
        out.planes.push_back(HostKVPlaneLayout{
            .offset             = cursor,
            .page_payload_bytes = page_bytes,
            .head_payload_bytes = head_bytes,
        });
        cursor = checked_add(cursor, page_bytes, "Host KV page payload overflow");
    }
    out.page_stride = align_up(cursor, kHostKVAlignment, "Host KV page record");
    return out;
}

} // namespace ninfer
