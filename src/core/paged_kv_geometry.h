#pragma once

#include "core/layout.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

inline constexpr std::int32_t kPagedKVPageSize = 64;

// A plane is storage-only. The model assigns K/V/layer meaning to plane indices.
struct KVPlaneGeometry {
    DType dtype                 = DType::BF16;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent    = 0;
    std::size_t alignment       = 256;
    // A prefix of c original tokens exposes floor(c/tokens_per_row) complete rows.
    std::uint32_t tokens_per_row = 1;

    friend bool operator==(const KVPlaneGeometry&, const KVPlaneGeometry&) = default;
};

struct KVPageGeometry {
    std::uint32_t page_tokens = kPagedKVPageSize;
    std::vector<KVPlaneGeometry> planes;

    friend bool operator==(const KVPageGeometry&, const KVPageGeometry&) = default;
};

[[nodiscard]] std::int32_t kv_plane_rows_per_page(const KVPageGeometry& page,
                                                  const KVPlaneGeometry& plane);

// One record-relative plane, dense in [feature,row,head] order on Device and Host alike.
struct KVPlaneLayout {
    KVPlaneGeometry geometry;
    TensorRegion tensor;
    std::size_t head_stride_bytes = 0;

    [[nodiscard]] std::size_t row_stride_bytes() const {
        return static_cast<std::size_t>(tensor.shape[0]) * dtype_size(tensor.dtype);
    }
};

struct KVPageLayout {
    KVPageGeometry geometry;
    std::vector<KVPlaneLayout> planes;
    std::size_t page_bytes = 0;
    std::size_t alignment  = 256;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] KVPageLayout plan_kv_page_layout(const KVPageGeometry& geometry);

} // namespace ninfer
