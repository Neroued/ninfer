#pragma once

#include "core/dtype.h"
#include "core/transfer_work.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

inline constexpr std::int32_t kPagedKVPageSize = 64;

// A plane is storage-only. Target code assigns K/V/layer meaning to plane indices.
struct KVPlaneGeometry {
    DType dtype                 = DType::BF16;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent    = 0;
    std::size_t alignment       = 256;
    // Original tokens represented by one complete row. A prefix of c original tokens makes
    // floor(c/tokens_per_row) rows valid; page IDs and coverage remain in the original domain.
    std::uint32_t tokens_per_row = 1;

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

// Shared Device/Host row geometry. Supports spans 1 and 4; span 4 requires PageMajor.
// The page token span must be positive and divisible by tokens_per_row, and the row count
// must fit a positive Tensor extent. Data types and feature/head extents remain plane facts.
[[nodiscard]] std::int32_t kv_plane_rows_per_page(const KVPageGeometry& page,
                                                  const KVPlaneGeometry& plane);

struct HostKVPlaneLayout {
    std::size_t offset             = 0;
    std::size_t page_payload_bytes = 0;
    std::size_t head_payload_bytes = 0;

    friend bool operator==(const HostKVPlaneLayout&, const HostKVPlaneLayout&) = default;
};

struct HostKVPageLayout {
    KVPageGeometry geometry;
    std::vector<HostKVPlaneLayout> planes;
    std::size_t page_stride = 0;

    friend bool operator==(const HostKVPageLayout&, const HostKVPageLayout&) = default;
};

[[nodiscard]] HostKVPageLayout plan_host_kv_page_layout(const KVPageGeometry& geometry);

} // namespace ninfer
