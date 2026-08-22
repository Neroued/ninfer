#include "core/host_kv_budget.h"
#include "core/host_kv_park.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%-62s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) { ++failures; }
}

// A stand-in pool: `pages` page groups, each `page_bytes` of device memory,
// laid out PageMajor as the Qwen3.6 text KV pool is.
struct FakePlane {
    void* device = nullptr;
    std::size_t page_bytes = 0;
    std::uint32_t pages = 0;

    FakePlane(std::size_t bytes, std::uint32_t count) : page_bytes(bytes), pages(count) {
        if (cudaMalloc(&device, bytes * count) != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc failed\n");
            std::exit(1);
        }
    }
    ~FakePlane() { (void)cudaFree(device); }

    [[nodiscard]] ninfer::HostKvPlaneView view() const {
        return ninfer::HostKvPlaneView{.base        = device,
                                       .page_stride = static_cast<std::int64_t>(page_bytes),
                                       .head_stride = 0,
                                       .head_extent = 1,
                                       .page_major  = true};
    }

    void fill(std::uint8_t seed) const {
        std::vector<std::uint8_t> host(page_bytes * pages);
        for (std::size_t i = 0; i < host.size(); ++i) {
            host[i] = static_cast<std::uint8_t>((i * 31 + seed) & 0xFF);
        }
        (void)cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice);
    }

    [[nodiscard]] std::vector<std::uint8_t> read() const {
        std::vector<std::uint8_t> host(page_bytes * pages);
        (void)cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost);
        return host;
    }

    void clobber() const {
        std::vector<std::uint8_t> zero(page_bytes * pages, 0xEE);
        (void)cudaMemcpy(device, zero.data(), zero.size(), cudaMemcpyHostToDevice);
    }
};

std::vector<std::uint8_t> pages_of(const std::vector<std::uint8_t>& buffer, std::size_t page_bytes,
                                   const std::vector<std::int32_t>& ids) {
    std::vector<std::uint8_t> out;
    for (std::int32_t id : ids) {
        const auto* begin = buffer.data() + static_cast<std::size_t>(id) * page_bytes;
        out.insert(out.end(), begin, begin + page_bytes);
    }
    return out;
}

}  // namespace

int main() {
    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }

    constexpr std::size_t kPageBytes = 4096;
    constexpr std::uint32_t kPages   = 64;

    FakePlane plane_a(kPageBytes, kPages);
    FakePlane plane_b(kPageBytes / 2, kPages);
    plane_a.fill(7);
    plane_b.fill(200);

    const std::vector<ninfer::HostKvPlaneView> planes{plane_a.view(), plane_b.view()};

    const std::size_t per_page = ninfer::host_kv_bytes_per_page(planes);
    check(per_page == kPageBytes + kPageBytes / 2, "bytes_per_page sums every plane");

    // A deliberately non-contiguous, unsorted page set: the runs logic must
    // handle both, and restore must tolerate a different physical mapping.
    const std::vector<std::int32_t> parked_ids{9, 10, 11, 40, 3};
    ninfer::HostKvBudget arena(2 * per_page * parked_ids.size());
    check(arena.used_bytes() == 0, "budget starts with the whole budget free");

    ninfer::HostKvSlab* slab = arena.acquire(per_page * parked_ids.size());
    check(slab != nullptr, "acquire returns a slab");
    check(arena.used_bytes() == per_page * parked_ids.size(), "acquire consumes the region");

    const auto before_a = plane_a.read();
    const auto before_b = plane_b.read();

    const std::size_t written = ninfer::host_kv_park(planes, parked_ids, *slab, nullptr);
    (void)cudaStreamSynchronize(nullptr);
    check(written == per_page * parked_ids.size(), "park writes exactly bytes_per_page * pages");
    check(slab->filled() == written, "park records filled bytes on the slab");

    // Destroy the device copy, then restore into *different* physical pages.
    plane_a.clobber();
    plane_b.clobber();
    const std::vector<std::int32_t> restored_ids{20, 21, 22, 50, 1};
    ninfer::host_kv_restore(planes, restored_ids, parked_ids.size(), *slab, nullptr);
    (void)cudaStreamSynchronize(nullptr);

    const auto after_a = plane_a.read();
    const auto after_b = plane_b.read();

    // The slab stores pages in logical (block-table) order: logical page i of
    // the parked set must land on logical page i of the restored set, even
    // when the physical ids are unsorted and the two mappings differ.
    check(pages_of(before_a, kPageBytes, parked_ids) ==
              pages_of(after_a, kPageBytes, restored_ids),
          "plane A round-trips bitwise into remapped pages (logical order)");
    check(pages_of(before_b, kPageBytes / 2, parked_ids) ==
              pages_of(after_b, kPageBytes / 2, restored_ids),
          "plane B round-trips bitwise into remapped pages (logical order)");

    // A page that was never parked must stay clobbered: park copied only what it was told to.
    bool untouched_is_clobbered = true;
    for (std::size_t i = 0; i < kPageBytes; ++i) {
        if (after_a[63 * kPageBytes + i] != 0xEE) { untouched_is_clobbered = false; break; }
    }
    check(untouched_is_clobbered, "pages outside the parked set are not restored");

    // Guard rails.
    bool threw = false;
    try {
        ninfer::HostKvBudget tiny(per_page);            // room for one page only
        ninfer::HostKvSlab* small = tiny.acquire(per_page);
        ninfer::host_kv_park(planes, parked_ids, *small, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "park rejects a slab that cannot hold the pages");

    threw = false;
    try {
        const std::vector<std::int32_t> too_many{1, 2, 3, 4, 5, 6};
        ninfer::host_kv_restore(planes, too_many, parked_ids.size(), *slab, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "restore rejects asking for more pages than were parked");

    threw = false;
    try {
        ninfer::host_kv_restore(planes, restored_ids, parked_ids.size() + 3, *slab, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "restore rejects a parked-page count the slab cannot match");

    // Prefix restore: a session that rewinds to a checkpoint needs only the
    // first pages of what was parked. 21% of observed requests rewind, so this
    // is the common path, not an edge case.
    plane_a.clobber();
    plane_b.clobber();
    const std::vector<std::int32_t> prefix_dest{60, 61, 62};   // 3 of the 5 parked, in range
    ninfer::host_kv_restore(planes, prefix_dest, parked_ids.size(), *slab, nullptr);
    (void)cudaStreamSynchronize(nullptr);

    const auto prefix_a = plane_a.read();
    const auto prefix_b = plane_b.read();
    // The prefix is the first three pages in logical order: {9, 10, 11}.
    const std::vector<std::int32_t> first_three_logical{9, 10, 11};
    check(pages_of(before_a, kPageBytes, first_three_logical) ==
              pages_of(prefix_a, kPageBytes, prefix_dest),
          "prefix restore returns the first parked pages of plane A (logical)");
    check(pages_of(before_b, kPageBytes / 2, first_three_logical) ==
              pages_of(prefix_b, kPageBytes / 2, prefix_dest),
          "prefix restore reads plane B from its own region, not plane A's");

    threw = false;
    try {
        const std::vector<std::int32_t> dupes{5, 5, 6, 7, 8};
        ninfer::host_kv_park(planes, dupes, *slab, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "park rejects duplicate physical pages");

    // HeadMajor: the DFlash full-KV layout.
    {
        constexpr std::size_t kPageDataBytes = 256;  // page_stride (one page group)
        constexpr std::uint32_t kHeads       = 4;    // head_extent
        constexpr std::uint32_t kPages       = 64;   // physical page count
        constexpr std::size_t kHeadStride    = kPageDataBytes * kPages;  // head_stride
        constexpr std::size_t kDevSize       = kHeads * kHeadStride;

        struct FakeHeadPlane {
            void* device = nullptr;
            FakeHeadPlane() {
                if (cudaMalloc(&device, kDevSize) != cudaSuccess) {
                    std::fprintf(stderr, "cudaMalloc (head plane) failed\n");
                    std::exit(1);
                }
            }
            ~FakeHeadPlane() { (void)cudaFree(device); }

            [[nodiscard]] ninfer::HostKvPlaneView view() const {
                return ninfer::HostKvPlaneView{.base        = device,
                                               .page_stride = static_cast<std::int64_t>(kPageDataBytes),
                                               .head_stride = static_cast<std::int64_t>(kHeadStride),
                                               .head_extent  = static_cast<std::int64_t>(kHeads),
                                               .page_major   = false};
            }

            void fill(std::uint8_t seed) const {
                std::vector<std::uint8_t> host(kDevSize, 0);
                for (std::size_t i = 0; i < host.size(); ++i) {
                    host[i] = static_cast<std::uint8_t>((i * 17 + seed) & 0xFF);
                }
                (void)cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice);
            }

            [[nodiscard]] std::vector<std::uint8_t> read() const {
                std::vector<std::uint8_t> host(kDevSize, 0);
                (void)cudaMemcpy(host.data(), device, host.size(), cudaMemcpyDeviceToHost);
                return host;
            }

            void clobber() const {
                std::vector<std::uint8_t> zero(kDevSize, 0xEE);
                (void)cudaMemcpy(device, zero.data(), zero.size(), cudaMemcpyHostToDevice);
            }

            // The device bytes of physical page `phys` at head `h`.
            [[nodiscard]] std::vector<std::uint8_t>
            block_of(const std::vector<std::uint8_t>& buf, std::uint32_t phys, std::uint32_t h) const {
                const auto* begin = buf.data() + static_cast<std::size_t>(phys) * kPageDataBytes +
                                    static_cast<std::size_t>(h) * kHeadStride;
                return std::vector<std::uint8_t>(begin, begin + kPageDataBytes);
            }
        };

        FakeHeadPlane head_plane;
        head_plane.fill(5);
        const std::vector<ninfer::HostKvPlaneView> head_planes{head_plane.view()};

        // Parked set: 5 pages, physical run shape 1+1+3 (non-contiguous).
        const std::vector<std::int32_t> head_parked{7, 20, 30, 31, 32};
        const std::size_t head_needed =
            ninfer::host_kv_bytes_per_page(head_planes) * head_parked.size();
        ninfer::HostKvBudget head_arena(head_needed);
        ninfer::HostKvSlab* head_slab = head_arena.acquire(head_needed);
        const auto head_before = head_plane.read();
        (void)ninfer::host_kv_park(head_planes, head_parked, *head_slab, nullptr);
        (void)cudaStreamSynchronize(nullptr);

        // Restore into a different mapping with a different run shape (one
        // contiguous run of 5).
        head_plane.clobber();
        const std::vector<std::int32_t> head_restored{50, 51, 52, 53, 54};
        ninfer::host_kv_restore(head_planes, head_restored, head_parked.size(), *head_slab,
                                nullptr);
        (void)cudaStreamSynchronize(nullptr);
        const auto head_after = head_plane.read();

        // Logical page i of the parked set must equal logical page i of the
        // restored set, across every head.
        bool head_ok = true;
        for (std::size_t i = 0; i < head_parked.size() && head_ok; ++i) {
            for (std::uint32_t h = 0; h < kHeads; ++h) {
                const auto src = head_plane.block_of(head_before, head_parked[i], h);
                const auto dst = head_plane.block_of(head_after, head_restored[i], h);
                if (src != dst) { head_ok = false; break; }
            }
        }
        check(head_ok, "HeadMajor round-trips bitwise across different run shapes");
    }

    arena.release(slab);
    check(arena.used_bytes() == 0, "release returns the region to the budget");

    std::printf("\n%s\n", failures == 0 ? "all checks passed" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
