#pragma once

#include "core/host_kv_arena.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen3_5::detail {

// Host tier geometry of the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache.md
// §5.4). The whole Host budget is one pinned pool of equal slabs. One slab holds one KV block
// bundle: the Main text page packed with its Host page layout, then the backend page when a
// backend pool exists. A StateImage is split over `image_slabs` consecutive-in-image slabs and a
// snapshot tail bundle uses one more slab, so blocks and state images compete for the same bytes
// under the index's eviction policy instead of a fixed split.
struct HybridHostLayout {
    HostKVPageLayout text;
    std::optional<HostKVPageLayout> backend;
    std::size_t backend_offset = 0;
    std::size_t slab_bytes     = 0;
    // Bytes one block bundle actually transfers (the packed page records, without padding).
    std::size_t block_payload_bytes = 0;
    std::size_t image_bytes         = 0;
    std::uint32_t image_slabs       = 0;
};

[[nodiscard]] HybridHostLayout plan_hybrid_host_layout(const KVPageGeometry& text,
                                                       const KVPageGeometry* backend,
                                                       std::size_t image_bytes);

// Slabs a Host budget buys. A zero budget disables the Host tier. A nonzero budget must hold at
// least one complete snapshot (image and tail) and one block, or it is rejected: a smaller tier
// could never back a snapshot.
[[nodiscard]] std::uint32_t hybrid_host_slabs(const HybridHostLayout& layout,
                                              std::uint64_t budget_bytes);

} // namespace ninfer::models::qwen3_5::detail
