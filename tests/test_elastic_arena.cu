// GPU test of ElasticArena (core/elastic_arena.h) and the elastic backing of DeviceKVPagePool
// (T64):
//   - granule bookkeeping: a granule is backed exactly when it intersects a fixed byte or a
//     range's mapped prefix (ranges that start or end inside a granule share it with their fixed
//     neighbours); physical_delta() predicts set_mapped()'s change exactly;
//   - addresses never move; backed bytes keep what was written while backed; a granule released
//     and backed again reads zero; fixed bytes survive every change;
//   - DeviceKVPagePool::set_backing: a page past the backed prefix triggers the grow callback with
//     the pages to back, before anything is handed out; a throwing callback leaves the pool
//     unchanged; allocated_extent() tracks the highest allocated page.
//   - Host arenas (ElasticMemory::Host, active KV in RAM): the same bookkeeping on page-locked
//     Host memory the device and the CPU address at one pointer; newly backed bytes read zero
//     from both; the RAM follows the prefixes;
//   - DeviceKVPagePool::set_host_backing: the Host planes' RAM grows in steps before a page past
//     it is handed out, on the pool and its mirror alike, keeps the bytes already written, and
//     trim_host_backing() returns it down to the allocated extent.
// Exit 77 when no GPU is usable.

#include "core/device.h"
#include "core/elastic_arena.h"
#include "core/paged_kv_cache.h"
#include "core/uva_copy.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Linker wrapping injects driver failures without production hooks or exhausting the GPU.
namespace fault {
int stage = -1, remaining = 0;
int handles = 0, mappings = 0;

bool hit(int at) { return stage == at && --remaining == 0; }
} // namespace fault

extern "C" {
CUresult __real_cuMemCreate(CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
                            unsigned long long);
CUresult __real_cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                         unsigned long long);
CUresult __real_cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
CUresult __real_cuMemRelease(CUmemGenericAllocationHandle);
CUresult __real_cuMemUnmap(CUdeviceptr, size_t);

CUresult __wrap_cuMemCreate(CUmemGenericAllocationHandle* h, size_t n, const CUmemAllocationProp* p,
                            unsigned long long flags) {
    if (fault::hit(0)) { return CUDA_ERROR_OUT_OF_MEMORY; }
    const auto result = __real_cuMemCreate(h, n, p, flags);
    if (result == CUDA_SUCCESS) { ++fault::handles; }
    return result;
}

CUresult __wrap_cuMemMap(CUdeviceptr p, size_t n, size_t offset, CUmemGenericAllocationHandle h,
                         unsigned long long flags) {
    if (fault::hit(1)) { return CUDA_ERROR_OUT_OF_MEMORY; }
    const auto result = __real_cuMemMap(p, n, offset, h, flags);
    if (result == CUDA_SUCCESS) { ++fault::mappings; }
    return result;
}

CUresult __wrap_cuMemSetAccess(CUdeviceptr p, size_t n, const CUmemAccessDesc* desc, size_t count) {
    if (fault::hit(2)) { return CUDA_ERROR_OUT_OF_MEMORY; }
    return __real_cuMemSetAccess(p, n, desc, count);
}

CUresult __wrap_cuMemRelease(CUmemGenericAllocationHandle h) {
    const auto result = __real_cuMemRelease(h);
    if (result == CUDA_SUCCESS) { --fault::handles; }
    return result;
}

CUresult __wrap_cuMemUnmap(CUdeviceptr p, size_t n) {
    const auto result = __real_cuMemUnmap(p, n);
    if (result == CUDA_SUCCESS) { --fault::mappings; }
    return result;
}
}

namespace {

using namespace ninfer;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::vector<unsigned char> read(const std::byte* p, std::size_t n) {
    std::vector<unsigned char> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), p, n, cudaMemcpyDeviceToHost));
    return h;
}

bool all(const std::vector<unsigned char>& v, unsigned char b) {
    for (unsigned char x : v) {
        if (x != b) { return false; }
    }
    return true;
}

void test_arena() {
    // Probe the granule with a tiny arena first.
    std::size_t g = 0;
    {
        const ElasticRangeSpec none[1] = {};
        ElasticArena probe(0, 1, std::span<const ElasticRangeSpec>(none, 0));
        g = probe.granule();
        check(probe.physical_bytes() == g, "one fixed byte backs one granule");
    }
    // 11 granules: fixed head [0, g/2+100), range A [g/2+100, 3.5g+100) (granules 1, 2 wholly
    // inside; 0 and 3 shared with fixed bytes), fixed [3.5g+100, 5g), range B [5g, 8g) (granules
    // 5..7), fixed tail [8g, 10g+100).
    const std::size_t bytes          = 10 * g + 100;
    const ElasticRangeSpec ranges[2] = {
        {g / 2 + 100, 3 * g, 0},
        {5 * g, 3 * g, g},
    };
    ElasticArena a(0, bytes, ranges);
    std::byte* base = static_cast<std::byte*>(a.span().data);
    check(a.physical_bytes() == 7 * g, "initial backing: 6 fixed granules + B's first (" +
                                           std::to_string(a.physical_bytes() / g) + ")");
    CUDA_CHECK(cudaMemset(base, 0x11, g / 2 + 100));         // fixed head
    CUDA_CHECK(cudaMemset(base + 5 * g, 0x22, g));           // B's backed granule
    CUDA_CHECK(cudaMemset(base + 8 * g, 0x33, 2 * g + 100)); // fixed tail
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaStream_t s = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    std::size_t want[2] = {3 * g, 0}; // A whole: backs 1, 2 (3 is fixed); B none: releases 5
    check(a.physical_delta(want) == static_cast<std::int64_t>(g), "delta: +2 -1 granules");
    a.set_mapped(want, s);
    check(a.physical_bytes() == 8 * g, "A backed, B released");
    check(a.span().data == base, "the base never moves");
    check(all(read(base + g, 2 * g), 0), "newly backed granules read zero");
    check(all(read(base, g / 2 + 100), 0x11) && all(read(base + 8 * g, 2 * g + 100), 0x33),
          "fixed bytes survive");
    CUDA_CHECK(cudaMemset(base + g / 2 + 100, 0x44, 3 * g));
    CUDA_CHECK(cudaDeviceSynchronize());

    // A shrinks to 1 byte: granule 1 still needed (it holds that byte's granule? no: byte
    // g/2+100 lives in granule 0, which is fixed), so granules 1 and 2 are released.
    std::size_t small[2] = {1, 2 * g + 1};
    check(a.physical_delta(small) == static_cast<std::int64_t>(g), "delta: -2 +3 granules");
    a.set_mapped(small, s);
    check(a.physical_bytes() == 9 * g, "A's whole granules released, B's 3 backed");
    check(all(read(base + 5 * g, 2 * g + 1), 0), "B's re-backed granule reads zero again");
    check(all(read(base + 3 * g, g / 2 + 100), 0x44),
          "A's bytes in the shared fixed granule survive");
    std::size_t again[2] = {3 * g, 0};
    a.set_mapped(again, s);
    check(all(read(base + g, 2 * g), 0), "A's released granules come back zeroed");
    check(a.range(0).mapped == 3 * g && a.range(1).mapped == 0, "prefixes recorded");
    CUDA_CHECK(cudaStreamDestroy(s));
}

void test_growth_failure() {
    std::size_t g = 0;
    {
        ElasticArena probe(0, 1, {});
        g = probe.granule();
    }
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Five new granules in two runs, separated by a shrinking range. Fail each driver
    // operation at every position, including after a whole run has been mapped.
    for (int stage = 0; stage < 3; ++stage) {
        for (int position = 1; position <= 5; ++position) {
            const int handles = fault::handles, mappings = fault::mappings;
            {
                const ElasticRangeSpec ranges[] = {{0, 3 * g, g}, {3 * g, g, g}, {4 * g, 3 * g, 0}};
                ElasticArena arena(0, 8 * g, ranges);
                auto* base = static_cast<std::byte*>(arena.span().data);
                CUDA_CHECK(cudaMemset(base, 0x5a, g));
                CUDA_CHECK(cudaMemset(base + 7 * g, 0xa5, g));
                CUDA_CHECK(cudaDeviceSynchronize());
                const std::size_t want[] = {3 * g, 0, 3 * g};
                // Repeat the same failure: retries must not accumulate hidden backing.
                for (int retry = 0; retry < 2; ++retry) {
                    fault::stage     = stage;
                    fault::remaining = position;
                    bool threw       = false;
                    try {
                        arena.set_mapped(want, stream);
                    } catch (const std::runtime_error&) { threw = true; }
                    fault::stage = -1;
                    check(threw && fault::remaining == 0, "injected driver failure reached");
                    check(arena.range(0).mapped == g && arena.range(1).mapped == 0 &&
                              arena.range(2).mapped == 0,
                          "failed growth retains completed shrink");
                    check(arena.physical_bytes() == 2 * g && fault::handles == handles + 2 &&
                              fault::mappings == mappings + 2,
                          "rollback releases handles and mappings");
                    check(arena.physical_delta(want) == static_cast<std::int64_t>(5 * g),
                          "retry budget includes every failed growth granule");
                    check(all(read(base, g), 0x5a) && all(read(base + 7 * g, g), 0xa5),
                          "rollback preserves elastic and fixed data");
                }
                arena.set_mapped(want, stream);
                check(arena.span().data == base && arena.physical_bytes() == 7 * g &&
                          arena.physical_delta(want) == 0,
                      "retry publishes complete backing");
                check(all(read(base + g, 2 * g), 0) && all(read(base + 4 * g, 3 * g), 0),
                      "all retry growth bytes are zero");
                check(all(read(base, g), 0x5a) && all(read(base + 7 * g, g), 0xa5),
                      "successful retry preserves existing bytes exactly");
                arena.set_mapped(want, stream);
                check(arena.physical_bytes() == 7 * g, "same prefixes are idempotent");
            }
            check(fault::handles == handles && fault::mappings == mappings,
                  "destruction releases all backing after recovery");
        }
        // One granule changes owners without being remapped. On failure neither of the
        // reported prefixes needs it: rollback must release it too.
        const ElasticRangeSpec shared[] = {{0, g / 2, g / 2}, {g / 2, 3 * g - g / 2, 0}};
        ElasticArena arena(0, 3 * g, shared);
        const std::size_t want[] = {0, 3 * g - g / 2};
        fault::stage             = stage;
        fault::remaining         = 2;
        bool threw               = false;
        try {
            arena.set_mapped(want, stream);
        } catch (const std::runtime_error&) { threw = true; }
        fault::stage = -1;
        check(threw && arena.physical_bytes() == 0 && arena.range(0).mapped == 0 &&
                  arena.range(1).mapped == 0,
              "rollback releases an orphaned shared granule");
        arena.set_mapped(want, stream);
        check(all(read(static_cast<std::byte*>(arena.span().data), 3 * g), 0),
              "shared granule retry is fully zeroed");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    check(fault::handles == 0 && fault::mappings == 0, "failure scenarios leak no backing");
}

void test_pool_backing() {
    // A one-plane pool of 64 pages of 64 tokens x 256 BF16, page-major.
    LayoutBuilder builder;
    KVPageGeometry geometry;
    geometry.planes.push_back({DType::BF16, 256, 1, 256});
    const DeviceKVPagePoolLayout layout = plan_device_kv_page_pool(
        builder, DeviceKVPagePoolSpec{.page_group_count = 64, .geometry = geometry});
    const std::size_t bytes = builder.finish(256);
    DeviceArena arena(bytes);
    DeviceKVPagePool pool(arena.alloc_bytes(bytes), layout);
    check(pool.backed_pages() == 64 && pool.allocated_extent() == 0, "default: every page backed");
    std::vector<std::uint32_t> asked;
    bool fail = false;
    pool.set_backing(4, [&](std::uint32_t pages) {
        asked.push_back(pages);
        if (fail) { throw std::runtime_error("no memory"); }
        pool.set_backed_pages((pages + 3) / 4 * 4);
    });
    auto reservation = pool.reserve(20);
    check(reservation.has_value(), "reserve 20 pages");
    std::vector<DeviceKVPageLease> leases;
    leases.reserve(20);
    pool.materialize(*reservation, 3, leases);
    check(asked.empty() && pool.allocated_extent() == 3, "3 pages inside the backing: no call");
    pool.materialize(*reservation, 6, leases);
    check(asked.size() == 1 && asked[0] == 6 && pool.backed_pages() == 8,
          "page 5 needs 6 backed pages (asked " + std::to_string(asked.empty() ? 0 : asked[0]) +
              ")");
    fail       = true;
    bool threw = false;
    try {
        pool.materialize(*reservation, 12, leases);
    } catch (const std::runtime_error&) { threw = true; }
    check(threw && leases.size() == 6 && pool.allocated_extent() == 6 && reservation->pages() == 14,
          "a failing backing leaves the pool unchanged");
    fail = false;
    pool.materialize(*reservation, 12, leases);
    check(pool.backed_pages() == 12 && pool.allocated_extent() == 12, "backed to 12");
    pool.dematerialize(*reservation, 2, leases);
    check(pool.allocated_extent() == 2, "extent follows the release");
    bool rejected = false;
    try {
        pool.set_backed_pages(1);
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "backing cannot drop an allocated page");
    pool.set_backed_pages(2);
    check(pool.backed_pages() == 2, "shrunk to the extent");
}

bool host_all(const std::byte* p, std::size_t n, unsigned char b) {
    for (std::size_t i = 0; i < n; ++i) {
        if (static_cast<unsigned char>(p[i]) != b) { return false; }
    }
    return true;
}

void test_host_arena() {
    std::size_t g = 0;
    {
        const ElasticRangeSpec none[1] = {};
        ElasticArena probe(0, 1, std::span<const ElasticRangeSpec>(none, 0), ElasticMemory::Host);
        g = probe.granule();
        check(probe.memory() == ElasticMemory::Host && probe.physical_bytes() == g,
              "Host arena: one fixed byte backs one granule");
    }
    // Two ranges of 3 granules each, nothing backed but the fixed tail granule.
    const ElasticRangeSpec ranges[2] = {{0, 3 * g, 0}, {3 * g, 3 * g, 0}};
    ElasticArena a(0, 6 * g + 1, ranges, ElasticMemory::Host);
    std::byte* base = static_cast<std::byte*>(a.span().data);
    check(a.physical_bytes() == g, "Host arena: only the fixed tail granule is backed");
    cudaStream_t s = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    std::size_t want[2] = {2 * g, g + 1};
    a.set_mapped(want, s);
    check(a.physical_bytes() == 5 * g, "Host arena: 2 + 2 granules backed");
    check(host_all(base, 2 * g, 0) && host_all(base + 3 * g, 2 * g, 0),
          "Host arena: newly backed granules read zero on the CPU");
    // The device writes, the CPU reads the same pointer.
    uva_fill_async(base + g, 0x5A, g, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    check(host_all(base + g, g, 0x5A) && host_all(base, g, 0),
          "Host arena: a kernel's write is what the CPU reads");
    std::size_t shrink[2] = {g, 0};
    a.set_mapped(shrink, s);
    check(a.physical_bytes() == 2 * g && a.span().data == base, "Host arena: RAM released");
    a.set_mapped(want, s);
    check(host_all(base + g, g, 0), "Host arena: a released granule comes back zeroed");
    CUDA_CHECK(cudaStreamDestroy(s));
}

void test_pool_host_backing(int mirror_device) {
    // Two planes of 64 pages: plane 0 in Host memory (64 x 2048 BF16 = 256 KiB per page), plane 1
    // on the device; the Host plane gets an elastic Host arena, on the origin and on a mirror pool.
    KVPageGeometry geometry;
    geometry.planes.push_back({DType::BF16, 2048, 1, 256, true});
    geometry.planes.push_back({DType::BF16, 32, 1, 256});

    struct Side {
        int device = 0;
        LayoutBuilder builder, host_builder;
        DeviceKVPagePoolLayout layout;
        std::size_t bytes = 0, host_bytes = 0;
        std::unique_ptr<DeviceArena> arena;
        std::unique_ptr<ElasticArena> host;
        std::unique_ptr<DeviceKVPagePool> pool;
        cudaStream_t stream = nullptr;
    };

    Side sides[2];
    for (int i = 0; i < 2; ++i) {
        Side& side  = sides[i];
        side.device = i == 0 ? 0 : mirror_device;
        CUDA_CHECK(cudaSetDevice(side.device));
        side.layout = plan_device_kv_page_pool(
            side.builder, DeviceKVPagePoolSpec{.page_group_count = 64, .geometry = geometry},
            &side.host_builder);
        side.bytes      = side.builder.finish(256);
        side.host_bytes = side.host_builder.finish(2U << 20);
        std::vector<ElasticRangeSpec> ranges;
        for (const DeviceKVPlaneLayout& plane : side.layout.planes) {
            if (plane.geometry.host) {
                ranges.push_back({plane.storage.region.offset, plane.storage.region.bytes, 0});
            }
        }
        side.host  = std::make_unique<ElasticArena>(side.device, side.host_bytes, ranges,
                                                    ElasticMemory::Host);
        side.arena = std::make_unique<DeviceArena>(side.bytes);
        side.pool  = std::make_unique<DeviceKVPagePool>(side.arena->alloc_bytes(side.bytes),
                                                        side.layout, side.host->span());
        side.pool->set_host_backing(*side.host, 4, 4);
        CUDA_CHECK(cudaStreamCreateWithFlags(&side.stream, cudaStreamNonBlocking));
    }
    CUDA_CHECK(cudaSetDevice(0));
    DeviceKVPagePool& pool = *sides[0].pool;
    pool.attach_mirror(*sides[1].pool,
                       DeviceKVMirror{.device = sides[1].device, .stream = sides[1].stream});
    const std::size_t page = 64 * 2048 * 2; // Host plane bytes per page
    const std::size_t g    = sides[0].host->granule();
    check(pool.host_backed_pages() == 4 && sides[1].pool->host_backed_pages() == 4,
          "Host backing: one step backed on both ranks");
    check(pool.host_physical_bytes() == (4 * page + g - 1) / g * g,
          "Host backing: the RAM of one step (" + std::to_string(pool.host_physical_bytes()) + ")");
    auto reservation = pool.reserve(40);
    std::vector<DeviceKVPageLease> leases;
    leases.reserve(40);
    pool.materialize(*reservation, 3, leases);
    // Bytes of page 1 written by a kernel survive later growth.
    std::byte* host0 = static_cast<std::byte*>(pool.plane(0).data);
    uva_fill_async(host0 + page, 0x6B, page, sides[0].stream);
    CUDA_CHECK(cudaStreamSynchronize(sides[0].stream));
    pool.materialize(*reservation, 9, leases);
    check(pool.host_backed_pages() == 12 && sides[1].pool->host_backed_pages() == 12,
          "Host backing: page 8 grows both ranks to 12 pages (" +
              std::to_string(pool.host_backed_pages()) + ", " +
              std::to_string(sides[1].pool->host_backed_pages()) + ")");
    check(host_all(host0 + page, page, 0x6B), "Host backing: written bytes survive growth");
    std::byte* host1 = static_cast<std::byte*>(sides[1].pool->plane(0).data);
    check(host_all(host1 + 8 * page, 4 * page, 0), "Host backing: the mirror's new pages read 0");
    pool.materialize(*reservation, 40, leases);
    check(pool.host_backed_pages() == 40, "Host backing: 40 pages");
    pool.dematerialize(*reservation, 5, leases);
    const std::size_t before     = pool.host_physical_bytes();
    const std::uint32_t released = pool.trim_host_backing();
    check(released == 32 && pool.host_backed_pages() == 8 &&
              sides[1].pool->host_backed_pages() == 8,
          "Host backing: trim to the extent rounded to the step (released " +
              std::to_string(released) + ")");
    check(pool.host_physical_bytes() < before, "Host backing: the RAM goes back");
    check(host_all(host0 + page, page, 0x6B), "Host backing: live bytes survive the trim");
    check(pool.trim_host_backing() == 0, "Host backing: a second trim releases nothing");
    leases.clear();
    reservation.reset();
    for (Side& side : sides) {
        CUDA_CHECK(cudaSetDevice(side.device));
        CUDA_CHECK(cudaStreamDestroy(side.stream));
    }
    CUDA_CHECK(cudaSetDevice(0));
}

} // namespace

// T68: memcpy_device_async between an elastic (VMM) arena on device 1 and a cudaMalloc buffer on
// device 0, eagerly, on either device's stream. Without P2P a plain UVA copy cannot read VMM memory
// of a device other than the stream's (cudaErrorInvalidValue); the helper takes the peer copy.
void test_cross_device_copy() {
    const std::size_t n            = (1u << 20) + 77;
    const ElasticRangeSpec none[1] = {};
    CUDA_CHECK(cudaSetDevice(1));
    ElasticArena a(1, n, std::span<const ElasticRangeSpec>(none, 0));
    std::byte* vmm  = static_cast<std::byte*>(a.span().data);
    cudaStream_t s1 = nullptr;
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaSetDevice(0));
    void* d0 = nullptr;
    CUDA_CHECK(cudaMalloc(&d0, n));
    cudaStream_t s0 = nullptr;
    CUDA_CHECK(cudaStreamCreate(&s0));
    std::vector<unsigned char> ref(n), back(n);
    for (int round = 0; round < 4; ++round) {
        const bool to_d0 = round % 2 == 0; // VMM(1) -> d0, else d0 -> VMM(1)
        const bool on_s0 = round / 2 == 0; // issued on device 0's stream, else device 1's
        for (std::size_t i = 0; i < n; ++i) {
            ref[i] = static_cast<unsigned char>(i * 13 + round * 5 + 1);
        }
        void* src = to_d0 ? static_cast<void*>(vmm) : d0;
        void* dst = to_d0 ? d0 : static_cast<void*>(vmm);
        CUDA_CHECK(cudaSetDevice(to_d0 ? 1 : 0));
        CUDA_CHECK(cudaMemcpy(src, ref.data(), n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaSetDevice(to_d0 ? 0 : 1));
        CUDA_CHECK(cudaMemset(dst, 0, n));
        // Both devices: a cudaMemcpy from pageable memory may return before its DMA lands, and
        // neither stream below waits for the other device's default stream. Without this the copy
        // could read the previous round's source bytes when another process slows the DMA.
        for (const int device : {0, 1}) {
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaSetDevice(on_s0 ? 0 : 1));
        CUDA_CHECK(memcpy_device_async(dst, src, n, on_s0 ? s0 : s1));
        CUDA_CHECK(cudaStreamSynchronize(on_s0 ? s0 : s1));
        CUDA_CHECK(cudaSetDevice(to_d0 ? 0 : 1));
        CUDA_CHECK(cudaMemcpy(back.data(), dst, n, cudaMemcpyDeviceToHost));
        check(back == ref, std::string("cross-device copy ") +
                               (to_d0 ? "VMM(1)->dev0" : "dev0->VMM(1)") +
                               (on_s0 ? " on dev0 stream" : " on dev1 stream"));
    }
    CUDA_CHECK(cudaStreamDestroy(s0));
    CUDA_CHECK(cudaFree(d0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaSetDevice(0));
}

int main() {
    int count             = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver ||
        (err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(err);
    try {
        CUDA_CHECK(cudaSetDevice(0));
        test_arena();
        test_pool_backing();
        test_host_arena();
        test_pool_host_backing(count >= 2 ? 1 : 0);
        test_growth_failure();
        if (count >= 2) { test_cross_device_copy(); }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::printf("elastic arena GPU test: all checks passed\n");
    return 0;
}
