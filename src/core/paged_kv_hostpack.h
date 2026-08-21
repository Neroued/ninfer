#pragma once

// Gather/scatter between paged-KV pool planes and a packed per-page staging span.
//
// The packed layout concatenates every plane's page payload (height rows × width bytes,
// row-packed) in plane order; packed_page_bytes(pool) is its size. One kernel launch moves a
// whole batch of physical pages, so host-cache traffic is two copies plus one kernel instead
// of a per-plane launch storm.

#include "core/paged_kv_cache.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <span>

namespace ninfer {

inline constexpr std::size_t kHostpackMaxPlanes = 256;

struct HostpackPlaneGeom {
    const std::byte* base       = nullptr;
    std::int64_t page_offset    = 0; // device byte offset per physical page id
    std::int64_t width          = 0; // packed row bytes
    std::int64_t height         = 0; // rows per page
    std::int64_t pitch          = 0; // device row pitch
    std::int64_t packed_offset  = 0; // offset inside the packed page image
};

struct HostpackGeometry {
    HostpackPlaneGeom planes[kHostpackMaxPlanes];
    std::uint32_t plane_count      = 0;
    std::size_t packed_page_bytes  = 0;
};

[[nodiscard]] HostpackGeometry hostpack_geometry(const PagedKVPool& pool);

// staging holds `page_count` consecutive packed page images; device_page_ids is a device
// pointer to `page_count` physical page ids.
void hostpack_gather(const HostpackGeometry& geometry, const std::int32_t* device_page_ids,
                     std::uint32_t page_count, std::byte* staging, cudaStream_t stream);
void hostpack_scatter(const HostpackGeometry& geometry, const std::int32_t* device_page_ids,
                      std::uint32_t page_count, const std::byte* staging, cudaStream_t stream);

} // namespace ninfer
