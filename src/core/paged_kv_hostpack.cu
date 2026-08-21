#include "core/paged_kv_hostpack.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer {

namespace {

struct DevicePlaneGeom {
    std::uint64_t base;
    std::int64_t page_offset;
    std::int64_t width;
    std::int64_t height;
    std::int64_t pitch;
    std::int64_t packed_offset;
};

__constant__ DevicePlaneGeom g_hostpack_planes[kHostpackMaxPlanes];

// One block per (page, plane); threads stride over the plane's page payload in 16-byte units
// (plane rows are 256-byte aligned by the pool spec, so vector moves are always legal).
template <bool ToPacked>
__global__ void hostpack_kernel(const std::int32_t* page_ids, std::byte* packed,
                                std::int64_t packed_page_bytes) {
    const DevicePlaneGeom geom = g_hostpack_planes[blockIdx.y];
    const std::int32_t page    = page_ids[blockIdx.x];
    std::byte* plane_page =
        reinterpret_cast<std::byte*>(geom.base) + static_cast<std::int64_t>(page) * geom.page_offset;
    std::byte* packed_page = packed + static_cast<std::int64_t>(blockIdx.x) * packed_page_bytes +
                             geom.packed_offset;
    const std::int64_t vectors = (geom.width * geom.height) / 16;
    for (std::int64_t vec = threadIdx.x; vec < vectors; vec += blockDim.x) {
        const std::int64_t byte_index = vec * 16;
        const std::int64_t row        = byte_index / geom.width;
        const std::int64_t column     = byte_index % geom.width;
        auto* device_word =
            reinterpret_cast<uint4*>(plane_page + row * geom.pitch + column);
        auto* packed_word = reinterpret_cast<uint4*>(packed_page + byte_index);
        if constexpr (ToPacked) {
            *packed_word = *device_word;
        } else {
            *device_word = *packed_word;
        }
    }
}

void launch(const HostpackGeometry& geometry, const std::int32_t* device_page_ids,
            std::uint32_t page_count, std::byte* staging, bool to_packed, cudaStream_t stream) {
    if (page_count == 0) { return; }
    if (geometry.plane_count == 0 || geometry.plane_count > kHostpackMaxPlanes) {
        throw std::invalid_argument("hostpack geometry has an unsupported plane count");
    }
    static thread_local const HostpackGeometry* uploaded = nullptr;
    if (uploaded != &geometry) {
        DevicePlaneGeom device_geoms[kHostpackMaxPlanes];
        for (std::uint32_t index = 0; index < geometry.plane_count; ++index) {
            const HostpackPlaneGeom& plane = geometry.planes[index];
            device_geoms[index] =
                DevicePlaneGeom{reinterpret_cast<std::uint64_t>(plane.base), plane.page_offset,
                                plane.width, plane.height, plane.pitch, plane.packed_offset};
        }
        CUDA_CHECK(cudaMemcpyToSymbolAsync(g_hostpack_planes, device_geoms,
                                           sizeof(DevicePlaneGeom) * geometry.plane_count, 0,
                                           cudaMemcpyHostToDevice, stream));
        uploaded = &geometry;
    }
    const dim3 grid(page_count, geometry.plane_count);
    if (to_packed) {
        hostpack_kernel<true><<<grid, 128, 0, stream>>>(
            device_page_ids, staging, static_cast<std::int64_t>(geometry.packed_page_bytes));
    } else {
        hostpack_kernel<false><<<grid, 128, 0, stream>>>(
            device_page_ids, staging, static_cast<std::int64_t>(geometry.packed_page_bytes));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

HostpackGeometry hostpack_geometry(const PagedKVPool& pool) {
    HostpackGeometry out;
    if (pool.plane_count() > kHostpackMaxPlanes) {
        throw std::invalid_argument("paged pool has more planes than hostpack supports");
    }
    std::size_t packed_offset = 0;
    for (std::size_t index = 0; index < pool.plane_count(); ++index) {
        const Tensor& plane      = pool.plane(index);
        HostpackPlaneGeom& geom  = out.planes[out.plane_count++];
        geom.base                = static_cast<const std::byte*>(plane.data);
        geom.packed_offset       = static_cast<std::int64_t>(packed_offset);
        // Mirrors PagedKVPool::zero_pages addressing for both plane orders.
        if (pool.plane_order() == PagedKVPlaneOrder::PageMajor) {
            geom.page_offset = plane.nb[3];
            geom.width       = plane.nb[3];
            geom.height      = 1;
            geom.pitch       = plane.nb[3];
        } else {
            geom.page_offset = plane.nb[2];
            geom.width       = plane.nb[2];
            geom.height      = plane.ne[3];
            geom.pitch       = plane.nb[3];
        }
        if (geom.width % 16 != 0) {
            throw std::logic_error("paged plane rows are not 16-byte aligned");
        }
        packed_offset += static_cast<std::size_t>(geom.width * geom.height);
    }
    out.packed_page_bytes = packed_offset;
    return out;
}

void hostpack_gather(const HostpackGeometry& geometry, const std::int32_t* device_page_ids,
                     std::uint32_t page_count, std::byte* staging, cudaStream_t stream) {
    launch(geometry, device_page_ids, page_count, staging, true, stream);
}

void hostpack_scatter(const HostpackGeometry& geometry, const std::int32_t* device_page_ids,
                      std::uint32_t page_count, const std::byte* staging, cudaStream_t stream) {
    launch(geometry, device_page_ids, page_count, const_cast<std::byte*>(staging), false, stream);
}

} // namespace ninfer
