#include "core/host_kv_park.h"

#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer {
namespace {

// Bytes one page contributes to a single plane.
[[nodiscard]] std::size_t plane_page_bytes(const HostKvPlaneView& plane) {
    if (plane.page_major) { return static_cast<std::size_t>(plane.page_stride); }
    return static_cast<std::size_t>(plane.page_stride) *
           static_cast<std::size_t>(plane.head_extent);
}

// A run of consecutive physical pages, expressed in the logical (block-table)
// order. `logical_start` is the index in the page_ids array, `first_phys` the
// physical id of the run's first page, `count` the run length. The slab stores
// pages in logical order, so the run structure is derived from the logical
// sequence, never from a physical sort.
struct Run {
    std::size_t logical_start;
    std::int32_t first_phys;
    std::int32_t count;
};

// Splits page_ids into runs of consecutive physical pages, preserving the
// logical order. Validates that the physical pages are distinct and, when the
// plane names a page count, that every id lies within it (mirrors the range
// check zero_pages() does).
[[nodiscard]] std::vector<Run> logical_runs(std::span<const std::int32_t> page_ids,
                                            std::int32_t page_count) {
    std::vector<std::int32_t> sorted(page_ids.begin(), page_ids.end());
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::invalid_argument("host KV park requires distinct physical pages");
    }
    if (page_count > 0 &&
        (sorted.front() < 0 || sorted.back() >= page_count)) {
        throw std::out_of_range("host KV page id is out of range");
    }

    std::vector<Run> runs;
    std::size_t begin = 0;
    while (begin < page_ids.size()) {
        std::size_t end = begin + 1;
        while (end < page_ids.size() && page_ids[end] == page_ids[end - 1] + 1) { ++end; }
        runs.push_back(Run{begin, page_ids[begin], static_cast<std::int32_t>(end - begin)});
        begin = end;
    }
    return runs;
}

// Slab offset of a plane's region. The slab is plane-major: every parked page
// of plane 0, then plane 1, and so on. `parked_pages` is the park's extent, so
// the layout is fixed by it and shared by a later (possibly shorter) restore.
[[nodiscard]] std::size_t plane_offset(std::span<const HostKvPlaneView> planes,
                                       std::size_t plane_index, std::size_t parked_pages) {
    std::size_t offset = 0;
    for (std::size_t p = 0; p < plane_index; ++p) {
        offset += plane_page_bytes(planes[p]) * parked_pages;
    }
    return offset;
}

}  // namespace

std::size_t host_kv_bytes_per_page(std::span<const HostKvPlaneView> planes) {
    std::size_t bytes = 0;
    for (const HostKvPlaneView& plane : planes) { bytes += plane_page_bytes(plane); }
    return bytes;
}

std::size_t host_kv_park(std::span<const HostKvPlaneView> planes,
                         std::span<const std::int32_t> page_ids, HostKvSlab& slab,
                         cudaStream_t stream) {
    if (page_ids.empty()) { return 0; }

    const std::size_t parked_pages = page_ids.size();
    const std::size_t required = host_kv_bytes_per_page(planes) * parked_pages;
    if (required > slab.bytes()) {
        throw std::invalid_argument("host KV slab is too small for the requested pages");
    }

    const std::int32_t page_count = planes.empty() ? 0 : planes[0].page_count;
    const auto runs = logical_runs(page_ids, page_count);
    auto* base = static_cast<unsigned char*>(slab.base());

    for (std::size_t p = 0; p < planes.size(); ++p) {
        const HostKvPlaneView& plane = planes[p];
        auto* device                 = static_cast<unsigned char*>(plane.base);
        const std::size_t offset     = plane_offset(planes, p, parked_pages);
        if (plane.page_major) {
            // Slab stores pages in logical order: logical page i at offset + i*page_bytes.
            const std::size_t page_bytes = static_cast<std::size_t>(plane.page_stride);
            for (const Run& run : runs) {
                const std::size_t run_bytes = static_cast<std::size_t>(run.count) * page_bytes;
                const std::size_t slab_off  = offset + run.logical_start * page_bytes;
                const auto* src =
                    device + static_cast<std::int64_t>(run.first_phys) * plane.page_stride;
                CUDA_CHECK(cudaMemcpyAsync(base + slab_off, src, run_bytes,
                                           cudaMemcpyDeviceToHost, stream));
            }
        } else {
            // HeadMajor: the slab stores [head][logical_page] with a fixed row
            // of parked_pages * page_stride, independent of physical runs.
            const std::size_t head_row = parked_pages * static_cast<std::size_t>(plane.page_stride);
            for (const Run& run : runs) {
                const std::size_t run_bytes =
                    static_cast<std::size_t>(run.count) * static_cast<std::size_t>(plane.page_stride);
                const std::size_t slab_off =
                    offset + run.logical_start * static_cast<std::size_t>(plane.page_stride);
                const auto* src = device + static_cast<std::int64_t>(run.first_phys) * plane.page_stride;
                CUDA_CHECK(cudaMemcpy2DAsync(base + slab_off, head_row, src,
                                             static_cast<std::size_t>(plane.head_stride),
                                             run_bytes,
                                             static_cast<std::size_t>(plane.head_extent),
                                             cudaMemcpyDeviceToHost, stream));
            }
        }
    }

    const std::size_t written = host_kv_bytes_per_page(planes) * parked_pages;
    slab.set_filled(written);
    return written;
}

void host_kv_restore(std::span<const HostKvPlaneView> planes,
                     std::span<const std::int32_t> page_ids, std::size_t parked_pages,
                     const HostKvSlab& slab, cudaStream_t stream) {
    if (page_ids.empty()) { return; }
    if (page_ids.size() > parked_pages) {
        throw std::invalid_argument("host KV restore asks for more pages than were parked");
    }
    // Not an equality check against filled(): a slab may carry several regions
    // (text pages, then backend pages, then the hidden and GDN tensors), so
    // this region is only a prefix of it. Bound the read instead.
    if (host_kv_bytes_per_page(planes) * parked_pages > slab.bytes()) {
        throw std::invalid_argument("host KV parked region exceeds the slab");
    }

    const std::int32_t page_count = planes.empty() ? 0 : planes[0].page_count;
    const auto runs = logical_runs(page_ids, page_count);
    const auto* base = static_cast<const unsigned char*>(slab.base());

    for (std::size_t p = 0; p < planes.size(); ++p) {
        const HostKvPlaneView& plane = planes[p];
        auto* device                 = static_cast<unsigned char*>(plane.base);
        const std::size_t offset     = plane_offset(planes, p, parked_pages);
        if (plane.page_major) {
            const std::size_t page_bytes = static_cast<std::size_t>(plane.page_stride);
            for (const Run& run : runs) {
                const std::size_t run_bytes = static_cast<std::size_t>(run.count) * page_bytes;
                const std::size_t slab_off  = offset + run.logical_start * page_bytes;
                auto* dst = device + static_cast<std::int64_t>(run.first_phys) * plane.page_stride;
                CUDA_CHECK(cudaMemcpyAsync(dst, base + slab_off, run_bytes,
                                           cudaMemcpyHostToDevice, stream));
            }
        } else {
            const std::size_t head_row = parked_pages * static_cast<std::size_t>(plane.page_stride);
            for (const Run& run : runs) {
                const std::size_t run_bytes =
                    static_cast<std::size_t>(run.count) * static_cast<std::size_t>(plane.page_stride);
                const std::size_t slab_off =
                    offset + run.logical_start * static_cast<std::size_t>(plane.page_stride);
                auto* dst = device + static_cast<std::int64_t>(run.first_phys) * plane.page_stride;
                CUDA_CHECK(cudaMemcpy2DAsync(dst, static_cast<std::size_t>(plane.head_stride),
                                             base + slab_off, head_row, run_bytes,
                                             static_cast<std::size_t>(plane.head_extent),
                                             cudaMemcpyHostToDevice, stream));
            }
        }
    }
}

}  // namespace ninfer