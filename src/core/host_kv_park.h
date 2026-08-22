#pragma once

#include "core/host_kv_budget.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// Plane geometry a park/restore needs, lifted out of PagedKVPool so this logic
// is testable without a pool. Mirrors the addressing zero_pages() uses.
struct HostKvPlaneView {
    void* base                 = nullptr;
    std::int64_t page_stride   = 0;  // nb[3] for PageMajor, nb[2] for HeadMajor
    std::int64_t head_stride   = 0;  // nb[3] when HeadMajor, else unused
    std::int64_t head_extent   = 0;  // ne[3] when HeadMajor, else 1
    bool page_major            = true;
    // Physical page count the ids must lie within (0 disables the check).
    // Mirrors the range validation zero_pages() does.
    std::int32_t page_count    = 0;
};

// Bytes one page occupies across every plane. Used to size slabs and to bound
// a park before it starts.
[[nodiscard]] std::size_t host_kv_bytes_per_page(std::span<const HostKvPlaneView> planes);

// Copies the named physical pages, across all planes, device -> slab.
// page_ids need not be sorted; contiguous runs are coalesced into single
// transfers exactly as zero_pages() does. Returns bytes written.
// Throws if the slab cannot hold them, so a mis-sized arena fails loudly.
std::size_t host_kv_park(std::span<const HostKvPlaneView> planes,
                         std::span<const std::int32_t> page_ids, HostKvSlab& slab,
                         cudaStream_t stream);

// Inverse of host_kv_park. page_ids may name different physical pages than the
// park did: kernels reach pages through the block table, so the mapping is
// rewritten by the caller and captured CUDA graphs pick it up without recapture.
//
// May restore a *prefix* of what was parked. A session that rewinds to a
// checkpoint needs only the pages up to that frontier, and 21% of observed
// requests rewind rather than append, so restoring fewer pages than were parked
// is the common case, not an error. page_ids must therefore be no longer than
// the parked run; `parked_pages` states how many pages the park covered so the
// per-plane source offsets can be computed.
void host_kv_restore(std::span<const HostKvPlaneView> planes,
                     std::span<const std::int32_t> page_ids, std::size_t parked_pages,
                     const HostKvSlab& slab, cudaStream_t stream);

}  // namespace ninfer
