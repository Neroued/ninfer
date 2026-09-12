#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

struct PlannedCache {
    ninfer::DeviceKVPagePoolLayout pages;
    ninfer::KVExecutionTableLayout tables;
    std::size_t bytes = 0;
};

PlannedCache plan_cache(std::uint32_t physical_pages, std::uint32_t logical_pages,
                        std::int32_t rows, ninfer::KVPageGeometry geometry) {
    ninfer::LayoutBuilder builder;
    PlannedCache out;
    out.pages = ninfer::plan_device_kv_page_pool(
        builder, {.page_group_count = physical_pages, .geometry = std::move(geometry)});
    out.tables = ninfer::plan_kv_execution_tables(
        builder, {.logical_page_capacity = logical_pages, .table_rows = rows});
    out.bytes = builder.finish(256);
    return out;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, const std::string& label) {
    return expect(actual == expected, label + " expected " + std::to_string(expected) + ", got " +
                                          std::to_string(actual));
}

std::vector<ninfer::DeviceKVPageLease> materialize(ninfer::DeviceKVPagePool& pool,
                                                   std::uint32_t pages) {
    std::optional<ninfer::DeviceKVPageReservation> reservation = pool.reserve(pages);
    if (!reservation) { throw std::bad_alloc(); }
    std::vector<ninfer::DeviceKVPageLease> out;
    out.reserve(pages);
    pool.materialize(*reservation, pages, out);
    return out;
}

std::vector<ninfer::DeviceKVPageHandle> handles(std::span<const ninfer::DeviceKVPageLease> pages) {
    std::vector<ninfer::DeviceKVPageHandle> out;
    out.reserve(pages.size());
    for (const ninfer::DeviceKVPageLease& page : pages) { out.push_back(page.handle()); }
    return out;
}

std::vector<std::int32_t> read_mapping(const ninfer::Tensor& row, std::size_t count) {
    std::vector<std::int32_t> out(count);
    const cudaError_t err =
        cudaMemcpy(out.data(), row.data, out.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("block-table read failed: ") +
                                 cudaGetErrorString(err));
    }
    return out;
}

std::vector<std::vector<unsigned char>> fill_device_pool(ninfer::DeviceKVPagePool& pool,
                                                         cudaStream_t stream) {
    std::vector<std::vector<unsigned char>> bytes;
    bytes.reserve(pool.plane_count());
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const ninfer::Tensor& plane = pool.plane(plane_index);
        std::vector<unsigned char> host(plane.bytes());
        for (std::size_t index = 0; index < host.size(); ++index) {
            host[index] =
                static_cast<unsigned char>((index * 29U + plane_index * 61U + 17U) & 0xffU);
        }
        const cudaError_t err =
            cudaMemcpyAsync(plane.data, host.data(), host.size(), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("device pool fill failed: ") +
                                     cudaGetErrorString(err));
        }
        bytes.push_back(std::move(host));
    }
    return bytes;
}

std::vector<std::byte>
expected_host_records(const ninfer::DeviceKVPagePool& pool,
                      std::span<const std::int32_t> physical_pages,
                      const ninfer::HostKVPageLayout& host_layout,
                      const std::vector<std::vector<unsigned char>>& device_planes) {
    std::vector<std::byte> out(host_layout.page_stride * physical_pages.size(), std::byte{0});
    for (std::size_t logical = 0; logical < physical_pages.size(); ++logical) {
        const std::int32_t physical = physical_pages[logical];
        for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
            const ninfer::Tensor& plane                 = pool.plane(plane_index);
            const ninfer::HostKVPlaneLayout& host_plane = host_layout.planes[plane_index];
            std::byte* destination =
                out.data() + logical * host_layout.page_stride + host_plane.offset;
            if (pool.geometry().device_plane_order == ninfer::PagedKVPlaneOrder::PageMajor) {
                const unsigned char* source = device_planes[plane_index].data() +
                                              static_cast<std::size_t>(physical) * plane.nb[3];
                std::memcpy(destination, source, host_plane.page_payload_bytes);
            } else {
                for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                    const unsigned char* source = device_planes[plane_index].data() +
                                                  static_cast<std::size_t>(head) * plane.nb[3] +
                                                  static_cast<std::size_t>(physical) * plane.nb[2];
                    std::memcpy(destination +
                                    static_cast<std::size_t>(head) * host_plane.head_payload_bytes,
                                source, host_plane.head_payload_bytes);
                }
            }
        }
    }
    return out;
}

bool page_payload_equal(ninfer::HostKVAllocationConstView left_view, std::uint32_t left,
                        ninfer::HostKVAllocationConstView right_view, std::uint32_t right) {
    const ninfer::HostKVPageLayout& layout = left_view.layout();
    if (right_view.layout() != layout) { return false; }
    for (const ninfer::HostKVPlaneLayout& plane : layout.planes) {
        const std::byte* a =
            left_view.data() + static_cast<std::size_t>(left) * layout.page_stride + plane.offset;
        const std::byte* b =
            right_view.data() + static_cast<std::size_t>(right) * layout.page_stride + plane.offset;
        if (std::memcmp(a, b, plane.page_payload_bytes) != 0) { return false; }
    }
    return true;
}

bool page_payload_zero(ninfer::HostKVAllocationConstView view, std::uint32_t page) {
    const ninfer::HostKVPageLayout& layout = view.layout();
    for (const ninfer::HostKVPlaneLayout& plane : layout.planes) {
        const std::byte* data =
            view.data() + static_cast<std::size_t>(page) * layout.page_stride + plane.offset;
        for (std::size_t index = 0; index < plane.page_payload_bytes; ++index) {
            if (data[index] != std::byte{0}) { return false; }
        }
    }
    return true;
}

int exercise_reservation_and_mapping(ninfer::DeviceContext& context) {
    int failures = 0;
    ninfer::KVPageGeometry geometry{
        .planes = {{ninfer::DType::I8, 8, 2, 256}},
    };
    PlannedCache plan = plan_cache(4, 4, 1, geometry);
    ninfer::DeviceArena arena(plan.bytes);
    ninfer::DeviceKVPagePool pool({arena.base(), arena.capacity()}, plan.pages);
    ninfer::KVExecutionTablePool tables({arena.base(), arena.capacity()}, plan.tables, pool);

    std::optional<ninfer::DeviceKVPageReservation> reservation = pool.reserve(3);
    failures += expect(reservation.has_value(), "three-page reservation failed");
    failures += expect_size(pool.reserved_pages(), 3, "reserved pages");
    failures += expect_size(pool.available_pages(), 1, "available after reserve");

    std::vector<ninfer::DeviceKVPageLease> pages;
    pages.reserve(3);
    pool.materialize(*reservation, 2, pages);
    failures += expect_size(pool.allocated_pages(), 2, "allocated after materialize");
    failures += expect_size(pool.reserved_pages(), 1, "remaining reservation");
    failures += expect(!pool.reserve(2).has_value(), "over-capacity reservation succeeded");

    pool.dematerialize(*reservation, 1, pages);
    failures += expect_size(pool.allocated_pages(), 1, "allocated after dematerialize");
    failures += expect_size(pool.reserved_pages(), 2, "reservation after dematerialize");
    failures += expect_size(pool.available_pages(), 1, "entitlement changed after dematerialize");
    pool.materialize(*reservation, 2, pages);

    ninfer::KVExecutionRowLease row = tables.acquire(0);
    tables.publish(row.handle(), 0, pages, context.stream);
    context.synchronize();
    const std::vector<std::int32_t> mapping = read_mapping(tables.row(row.handle()), pages.size());
    failures += expect(mapping == std::vector<std::int32_t>({0, 1}),
                       "execution mapping did not preserve logical order");
    row.release();
    failures += expect_size(pool.allocated_pages(), 2, "row release changed page ownership");

    reservation->clear();
    failures += expect_size(pool.reserved_pages(), 0, "reservation clear");
    const ninfer::DeviceKVPageHandle stale = pages.back().handle();
    pages.back().release();
    bool stale_rejected = false;
    try {
        pool.zero_pages(std::span<const ninfer::DeviceKVPageHandle>(&stale, 1), context.stream);
    } catch (const std::invalid_argument&) { stale_rejected = true; }
    failures += expect(stale_rejected, "released page capability remained usable");

    PlannedCache other_plan = plan_cache(1, 1, 1, geometry);
    ninfer::DeviceArena other_arena(other_plan.bytes);
    ninfer::DeviceKVPagePool other({other_arena.base(), other_arena.capacity()}, other_plan.pages);
    const std::uint32_t before                                = pool.reserved_pages();
    const ninfer::DeviceKVPageReservationRequest impossible[] = {
        {.pool = &pool, .pages = 2},
        {.pool = &other, .pages = 2},
    };
    bool bundle_failed = false;
    try {
        auto unused = ninfer::reserve_device_kv_page_bundle(impossible);
        (void)unused;
    } catch (const std::bad_alloc&) { bundle_failed = true; }
    failures += expect(bundle_failed, "impossible multi-pool reservation succeeded");
    failures += expect_size(pool.reserved_pages(), before, "failed bundle changed the first pool");
    return failures;
}

int exercise_layout_and_transfer(ninfer::DeviceContext& context, ninfer::KVPageGeometry geometry,
                                 const std::string& label) {
    int failures                  = 0;
    PlannedCache source_plan      = plan_cache(10, 8, 2, geometry);
    PlannedCache destination_plan = plan_cache(10, 8, 1, geometry);
    ninfer::DeviceArena source_arena(source_plan.bytes);
    ninfer::DeviceArena destination_arena(destination_plan.bytes);
    ninfer::DeviceKVPagePool source({source_arena.base(), source_arena.capacity()},
                                    source_plan.pages);
    ninfer::KVExecutionTablePool tables({source_arena.base(), source_arena.capacity()},
                                        source_plan.tables, source);
    ninfer::DeviceKVPagePool destination({destination_arena.base(), destination_arena.capacity()},
                                         destination_plan.pages);

    failures += expect_size(source.plane_count(), geometry.planes.size(), label + " plane count");
    const ninfer::Tensor& first_plane = source.plane(0);
    if (geometry.device_plane_order == ninfer::PagedKVPlaneOrder::PageMajor) {
        failures += expect_size(first_plane.ne[2], geometry.planes[0].head_extent,
                                label + " PageMajor heads");
        failures += expect_size(first_plane.ne[3], 10, label + " PageMajor pages");
    } else {
        failures += expect_size(first_plane.ne[2], 10, label + " HeadMajor pages");
        failures += expect_size(first_plane.ne[3], geometry.planes[0].head_extent,
                                label + " HeadMajor heads");
    }

    std::vector<ninfer::DeviceKVPageLease> prefix   = materialize(source, 3);
    std::vector<ninfer::DeviceKVPageLease> blockers = materialize(source, 3);
    prefix.clear();
    std::vector<ninfer::DeviceKVPageLease> fragmented            = materialize(source, 5);
    const std::vector<ninfer::DeviceKVPageHandle> source_handles = handles(fragmented);
    ninfer::KVExecutionRowLease row                              = tables.acquire(1);
    tables.publish(row.handle(), 0, source_handles, context.stream);
    context.synchronize();
    const std::vector<std::int32_t> physical_mapping =
        read_mapping(tables.row(row.handle()), source_handles.size());
    failures += expect(physical_mapping == std::vector<std::int32_t>({0, 1, 2, 6, 7}),
                       label + " execution row differs from logical page order");
    failures += expect(source.contiguous_run_count(source_handles) == 2,
                       label + " physical KV run count missed allocator fragmentation");
    failures +=
        expect(source.contiguous_run_count(std::span<const ninfer::DeviceKVPageHandle>{}) == 0,
               label + " empty physical KV range has a copy run");
    const std::uint32_t allocated_before_row_release = source.allocated_pages();
    row.release();
    failures += expect_size(source.allocated_pages(), allocated_before_row_release,
                            label + " row ownership isolation");

    const std::vector<std::vector<unsigned char>> device_bytes =
        fill_device_pool(source, context.stream);
    context.synchronize();

    const ninfer::HostKVPageLayout host_layout =
        ninfer::plan_host_kv_page_layout(source.geometry());
    const ninfer::HostKVPageLayout layouts[] = {host_layout};
    ninfer::HostKVArena host_arena(host_layout.page_stride * 24,
                                   std::span<const ninfer::HostKVPageLayout>(layouts));
    failures += expect(!host_arena.can_allocate(host_layout, 25),
                       label + " oversized Host extent was reported allocatable");
    std::optional<ninfer::HostKVAllocation> host =
        host_arena.allocate(host_layout, static_cast<std::uint32_t>(source_handles.size()));
    failures += expect(host.has_value(), label + " Host allocation failed");
    ninfer::HostKVAllocationView host_view = host_arena.writable_view(*host);
    std::memset(host_view.data(), 0, host_layout.page_stride * host_view.page_count());
    source.copy_to_host(source_handles, host_view, context.stream);
    context.synchronize();

    const std::vector<std::byte> expected =
        expected_host_records(source, physical_mapping, host_layout, device_bytes);
    failures += expect(std::memcmp(host_view.data(), expected.data(), expected.size()) == 0,
                       label + " D2H canonical page records differ from Device payload");

    std::vector<ninfer::DeviceKVPageLease> restored                = materialize(destination, 5);
    const std::vector<ninfer::DeviceKVPageHandle> restored_handles = handles(restored);
    destination.zero_pages(restored_handles, context.stream);
    destination.copy_from_host(host_arena.view(*host), restored_handles, context.stream);

    const std::array duplicate_destinations{restored[0].handle(), restored[0].handle()};
    bool duplicate_zero_rejected = false;
    try {
        destination.zero_pages(duplicate_destinations, context.stream);
    } catch (const std::invalid_argument&) { duplicate_zero_rejected = true; }
    failures += expect(duplicate_zero_rejected, label + " duplicate zero destination accepted");
    bool duplicate_restore_rejected = false;
    try {
        destination.copy_from_host(host_arena.view(*host).subview(0, 2), duplicate_destinations,
                                   context.stream);
    } catch (const std::invalid_argument&) { duplicate_restore_rejected = true; }
    failures += expect(duplicate_restore_rejected, label + " duplicate H2D destination accepted");

    std::optional<ninfer::HostKVAllocation> roundtrip = host_arena.allocate(host_layout, 5);
    failures += expect(roundtrip.has_value(), label + " roundtrip Host allocation failed");
    ninfer::HostKVAllocationView roundtrip_view = host_arena.writable_view(*roundtrip);
    std::memset(roundtrip_view.data(), 0, host_layout.page_stride * roundtrip_view.page_count());
    destination.copy_to_host(restored_handles, roundtrip_view, context.stream);
    context.synchronize();
    failures += expect(std::memcmp(roundtrip_view.data(), host_view.data(), expected.size()) == 0,
                       label + " Device -> Host -> Device roundtrip changed bytes");

    destination.copy_page(restored[0].handle(), restored[4].handle(), context.stream);
    const ninfer::DeviceKVPageHandle copied[] = {restored[0].handle(), restored[4].handle()};
    std::optional<ninfer::HostKVAllocation> copied_host = host_arena.allocate(host_layout, 2);
    ninfer::HostKVAllocationView copied_view            = host_arena.writable_view(*copied_host);
    std::memset(copied_view.data(), 0, host_layout.page_stride * 2);
    destination.copy_to_host(copied, copied_view, context.stream);
    context.synchronize();
    const ninfer::HostKVAllocationConstView copied_contents = host_arena.view(*copied_host);
    failures += expect(page_payload_equal(copied_contents, 0, copied_contents, 1),
                       label + " D2D copy did not cover the complete page-group");
    failures += expect(page_payload_equal(copied_contents, 0, host_arena.view(*host), 0),
                       label + " D2D copy changed its source page");

    const ninfer::DeviceKVPageHandle zeroed[] = {restored[1].handle()};
    destination.zero_pages(zeroed, context.stream);
    const ninfer::DeviceKVPageHandle zero_observation[] = {restored[0].handle(),
                                                           restored[1].handle()};
    std::optional<ninfer::HostKVAllocation> zero_host   = host_arena.allocate(host_layout, 2);
    ninfer::HostKVAllocationView zero_view              = host_arena.writable_view(*zero_host);
    std::memset(zero_view.data(), 0x5a, host_layout.page_stride * 2);
    destination.copy_to_host(zero_observation, zero_view, context.stream);
    context.synchronize();
    const ninfer::HostKVAllocationConstView zero_contents = host_arena.view(*zero_host);
    failures += expect(page_payload_equal(zero_contents, 0, host_arena.view(*host), 0),
                       label + " selective zero changed an unselected page");
    failures += expect(page_payload_zero(zero_contents, 1),
                       label + " selective zero left page payload bytes");

    std::optional<ninfer::HostKVAllocation> split_source = host_arena.allocate(host_layout, 4);
    failures += expect(split_source.has_value(), label + " split source allocation failed");
    ninfer::HostKVAllocationView stale_view = host_arena.writable_view(*split_source);
    auto [left, right]                      = host_arena.split(std::move(*split_source), 1);
    failures += expect(!stale_view.valid(), label + " split did not invalidate the old capability");
    failures += expect_size(left.page_count(), 1, label + " split left pages");
    failures += expect_size(right.page_count(), 3, label + " split right pages");
    auto [middle, tail] = host_arena.split(std::move(right), 1);
    middle.release();
    std::optional<ninfer::HostKVAllocation> reused = host_arena.allocate(host_layout, 1);
    failures += expect(reused.has_value(), label + " released Host subextent was not reusable");

    ninfer::HostKVArena recipe_arena(host_layout.page_stride * 8,
                                     std::span<const ninfer::HostKVPageLayout>(layouts));
    auto recipe_left   = recipe_arena.allocate(host_layout, 2);
    auto recipe_middle = recipe_arena.allocate(host_layout, 3);
    auto recipe_right  = recipe_arena.allocate(host_layout, 2);
    failures += expect(recipe_left && recipe_middle && recipe_right,
                       label + " release-aware recipe fixture allocation failed");
    const std::array release_handles{recipe_middle->handle(), recipe_right->handle()};
    const std::array target_requests{
        ninfer::HostKVAllocationRequest{.layout = &host_layout, .pages = 5}};
    auto recipe = recipe_arena.plan_after_releases(release_handles, target_requests);
    failures += expect(recipe.has_value(), label + " release-aware Host recipe was not planned");
    auto revision_probe = recipe_arena.allocate(host_layout, 1);
    failures += expect(revision_probe.has_value(), label + " Host recipe revision probe failed");
    std::array<ninfer::HostKVAllocation*, 2> release_allocations{&*recipe_middle, &*recipe_right};
    std::array<ninfer::HostKVAllocation, 1> recipe_targets;
    failures +=
        expect(!recipe_arena.apply_recipe(std::move(*recipe), release_allocations, recipe_targets),
               label + " stale Host recipe changed the arena");
    failures += expect(recipe_middle->valid() && recipe_right->valid(),
                       label + " stale Host recipe consumed a release");
    revision_probe->release();
    recipe = recipe_arena.plan_after_releases(release_handles, target_requests);
    failures += expect(recipe && recipe_arena.apply_recipe(std::move(*recipe), release_allocations,
                                                           recipe_targets),
                       label + " release-aware Host recipe adoption failed");
    failures += expect(recipe_targets[0].page_count() == 5 && !recipe_middle->valid() &&
                           !recipe_right->valid(),
                       label + " release-aware Host recipe published an invalid result");

    ninfer::HostKVArena subrelease_arena(host_layout.page_stride * 8,
                                         std::span<const ninfer::HostKVPageLayout>(layouts));
    auto subrelease_left   = subrelease_arena.allocate(host_layout, 2);
    auto subrelease_middle = subrelease_arena.allocate(host_layout, 4);
    auto subrelease_right  = subrelease_arena.allocate(host_layout, 2);
    failures += expect(subrelease_left && subrelease_middle && subrelease_right,
                       label + " Host suballocation release fixture failed");
    const std::array subrelease{ninfer::HostKVSuballocationRelease{
        .allocation = subrelease_middle->handle(), .begin_page = 1, .page_count = 2}};
    const std::array two_page_target{
        ninfer::HostKVAllocationRequest{.layout = &host_layout, .pages = 2}};
    const std::array three_page_target{
        ninfer::HostKVAllocationRequest{.layout = &host_layout, .pages = 3}};
    failures += expect(!subrelease_arena.can_allocate(host_layout, 1) &&
                           subrelease_arena.can_allocate_after_suballocation_releases(
                               subrelease, two_page_target) &&
                           !subrelease_arena.can_allocate_after_suballocation_releases(
                               subrelease, three_page_target),
                       label + " Host suballocation release feasibility is not extent exact");
    (void)left;
    (void)tail;
    (void)blockers;
    return failures;
}

} // namespace

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || (count_error == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_error) << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext context(0);
        int failures = exercise_reservation_and_mapping(context);
        failures += exercise_layout_and_transfer(
            context,
            ninfer::KVPageGeometry{
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes =
                    {
                        {ninfer::DType::I8, 8, 2, 256},
                        {ninfer::DType::FP16, 1, 2, 256},
                    },
            },
            "PageMajor");
        failures += exercise_layout_and_transfer(
            context,
            ninfer::KVPageGeometry{
                .device_plane_order = ninfer::PagedKVPlaneOrder::HeadMajor,
                .planes =
                    {
                        {ninfer::DType::BF16, 8, 3, 256},
                        {ninfer::DType::FP16, 2, 3, 256},
                    },
            },
            "HeadMajor");
        failures += exercise_layout_and_transfer(
            context,
            ninfer::KVPageGeometry{
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes =
                    {
                        {ninfer::DType::FP8_E4M3FN, 256, 2, 256},
                        {ninfer::DType::U8, 128, 2, 256},
                        {ninfer::DType::FP16, 1, 2, 256},
                        {ninfer::DType::U8, 16, 2, 256},
                    },
            },
            "K8V4 asymmetric PageMajor");
        if (failures != 0) {
            std::cerr << failures << " Paged KV physical-container checks failed\n";
            return 1;
        }
        std::cout << "Paged KV physical-container checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Paged KV physical-container test failed: " << error.what() << '\n';
        return 1;
    }
}
