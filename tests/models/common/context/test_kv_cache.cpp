#include "core/device.h"
#include "core/buffer.h"
#include "models/common/context/storage.h"
#include "models/common/context/directory/allocation.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
namespace context = ninfer::models::context;

struct HostRecords {
    ninfer::MemorySpan memory;
    const ninfer::HostKVPageLayout* geometry;
    std::uint32_t pages;

    std::byte* data() const { return static_cast<std::byte*>(memory.data); }

    const ninfer::HostKVPageLayout& layout() const { return *geometry; }

    std::uint32_t page_count() const { return pages; }

    HostRecords subview(std::uint32_t begin, std::uint32_t count) const {
        if (!count || begin > pages || count > pages - begin)
            throw std::out_of_range("Host record subview");
        return {memory.subspan(begin * geometry->page_stride, count * geometry->page_stride),
                geometry, count};
    }
};

HostRecords records(const context::HostAllocation& allocation, ninfer::PinnedHostBuffer& backing,
                    const ninfer::HostKVPageLayout& layout) {
    const auto extent = allocation.extent();
    return {backing.span().subspan(extent.offset, extent.bytes), &layout,
            static_cast<std::uint32_t>(extent.bytes / layout.page_stride)};
}

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

std::vector<context::PageLease> materialize(context::PageAllocator& pool, std::uint32_t pages) {
    std::optional<context::PageReservation> reservation = pool.reserve(pages);
    if (!reservation) { throw std::bad_alloc(); }
    std::vector<context::PageLease> out;
    out.reserve(pages);
    pool.materialize(*reservation, pages, out);
    return out;
}

std::vector<std::int32_t> handles(std::span<const context::PageLease> pages) {
    std::vector<std::int32_t> out;
    out.reserve(pages.size());
    for (const context::PageLease& page : pages) {
        out.push_back(static_cast<std::int32_t>(page.slot().index));
    }
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

std::vector<std::vector<unsigned char>>
fill_device_pool(context::DevicePageStorage& pool, cudaStream_t stream, unsigned char seed = 0) {
    std::vector<std::vector<unsigned char>> bytes;
    bytes.reserve(pool.plane_count());
    for (std::size_t plane_index = 0; plane_index < pool.plane_count(); ++plane_index) {
        const ninfer::Tensor& plane = pool.plane(plane_index);
        std::vector<unsigned char> host(plane.bytes());
        for (std::size_t index = 0; index < host.size(); ++index) {
            host[index] =
                static_cast<unsigned char>(((index * 29U) ^ (index >> 8U) ^ (index >> 16U) ^
                                            (plane_index * 61U + seed + 17U)) &
                                           0xffU);
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
expected_host_records(const context::DevicePageStorage& pool,
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

bool page_payload_equal(HostRecords left_view, std::uint32_t left, HostRecords right_view,
                        std::uint32_t right) {
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

bool page_payload_zero(HostRecords view, std::uint32_t page) {
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

int exercise_mixed_row_geometry() {
    int failures = 0;
    const ninfer::KVPageGeometry geometry{
        .planes =
            {
                {ninfer::DType::U8, 128, 2, 256},
                {ninfer::DType::U8, 128, 2, 256},
                {ninfer::DType::U8, 16, 2, 256},
                {ninfer::DType::U8, 16, 2, 256},
                {.dtype          = ninfer::DType::BF16,
                 .leading_extent = 128,
                 .head_extent    = 1,
                 .tokens_per_row = 4},
            },
    };
    ninfer::LayoutBuilder builder;
    const auto device =
        ninfer::plan_device_kv_page_pool(builder, {.page_group_count = 3, .geometry = geometry});
    const auto host = ninfer::plan_host_kv_page_layout(geometry);
    // Literal NVFP4-G16 K/V + BF16 compressed-key geometry, independent of the layout helper.
    constexpr std::array<std::array<std::int32_t, 4>, 5> shapes{{
        {128, 64, 2, 3},
        {128, 64, 2, 3},
        {16, 64, 2, 3},
        {16, 64, 2, 3},
        {128, 16, 1, 3},
    }};
    constexpr std::array<std::size_t, 5> page_bytes{16384, 16384, 2048, 2048, 4096};
    constexpr std::array<std::size_t, 5> head_bytes{8192, 8192, 1024, 1024, 4096};
    constexpr std::array<std::size_t, 5> host_offsets{0, 16384, 32768, 34816, 36864};
    constexpr std::array<std::size_t, 5> device_offsets{0, 49152, 98304, 104448, 110592};
    for (std::size_t i = 0; i < geometry.planes.size(); ++i) {
        const auto& plane = device.planes[i].storage;
        failures += expect(plane.shape == shapes[i], "mixed-row Device plane shape");
        failures += expect_size(plane.region.offset, device_offsets[i], "mixed-row Device offset");
        failures += expect_size(plane.region.bytes, 3 * page_bytes[i], "mixed-row Device bytes");
        failures += expect_size(host.planes[i].offset, host_offsets[i], "mixed-row Host offset");
        failures += expect_size(host.planes[i].head_payload_bytes, head_bytes[i],
                                "mixed-row Host head bytes");
        failures += expect_size(host.planes[i].page_payload_bytes, page_bytes[i],
                                "mixed-row Host plane bytes");
    }
    failures += expect_size(host.page_stride, 40960, "mixed-row Host page stride");
    failures += expect_size(device.payload_bytes(), 122880, "mixed-row Device payload");
    failures += expect_size(builder.finish(256), 122880, "mixed-row Device backing");

    // Device slab alignment does not enlarge the canonical packed Host representation.
    auto aligned                    = geometry;
    aligned.planes.back().alignment = 65536;
    ninfer::LayoutBuilder aligned_builder;
    const auto aligned_device = ninfer::plan_device_kv_page_pool(
        aligned_builder, {.page_group_count = 3, .geometry = aligned});
    const auto aligned_host = ninfer::plan_host_kv_page_layout(aligned);
    failures += expect_size(aligned_device.planes.back().storage.region.offset, 131072,
                            "mixed-row aligned Device offset");
    failures += expect_size(aligned_builder.finish(256), 143360, "mixed-row aligned backing");
    failures += expect_size(aligned_device.payload_bytes(), 122880, "mixed-row aligned payload");
    failures += expect_size(aligned_host.page_stride, 40960, "mixed-row aligned Host stride");

    const auto rejected = [&](const ninfer::KVPageGeometry& invalid, const std::string& label) {
        bool device_rejected = false, host_rejected = false;
        try {
            ninfer::LayoutBuilder unused;
            (void)ninfer::plan_device_kv_page_pool(unused,
                                                   {.page_group_count = 1, .geometry = invalid});
        } catch (const std::invalid_argument&) { device_rejected = true; }
        try {
            (void)ninfer::plan_host_kv_page_layout(invalid);
        } catch (const std::invalid_argument&) { host_rejected = true; }
        return expect(device_rejected && host_rejected, label);
    };
    for (const std::uint32_t span : {0U, 2U, 256U}) {
        auto invalid                         = geometry;
        invalid.planes.back().tokens_per_row = span;
        failures += rejected(invalid, "unsupported row span accepted by a layout planner");
    }
    auto invalid        = geometry;
    invalid.page_tokens = 63;
    failures += rejected(invalid, "non-divisible page span accepted by a layout planner");
    invalid                    = geometry;
    invalid.device_plane_order = ninfer::PagedKVPlaneOrder::HeadMajor;
    failures += rejected(invalid, "compressed HeadMajor plane accepted by a layout planner");
    return failures;
}

int exercise_reservation_and_mapping(ninfer::DeviceContext& context) {
    int failures = 0;
    ninfer::KVPageGeometry geometry{
        .planes = {{ninfer::DType::I8, 8, 2, 256}},
    };
    PlannedCache plan = plan_cache(4, 4, 1, geometry);
    ninfer::DeviceBuffer arena(context.device(), plan.bytes);
    context::PageAllocator pool(0, 4);
    context::DevicePageStorage storage(arena.span(), plan.pages);
    context::ExecutionTableStorage tables(arena.span(), plan.tables, storage);

    std::optional<context::PageReservation> reservation = pool.reserve(3);
    failures += expect(reservation.has_value(), "three-page reservation failed");
    failures += expect_size(pool.reserved_pages(), 3, "reserved pages");
    failures += expect_size(pool.available_pages(), 1, "available after reserve");

    std::vector<context::PageLease> pages;
    pages.reserve(3);
    pool.materialize(*reservation, 2, pages);
    failures += expect_size(pool.allocated_pages(), 2, "allocated after materialize");
    failures += expect_size(pool.reserved_pages(), 1, "remaining reservation");
    failures += expect(!pool.reserve(2).has_value(), "over-capacity reservation succeeded");

    pool.dematerialize(*reservation, 1, pages);
    failures += expect_size(pool.allocated_pages(), 1, "allocated after dematerialize");
    failures += expect_size(pool.reserved_pages(), 2, "reservation after dematerialize");
    failures +=
        expect_size(pool.available_pages(), 1, "available capacity changed after dematerialize");
    pool.materialize(*reservation, 2, pages);

    ninfer::PinnedHostBuffer mapping_staging(4 * sizeof(std::int32_t));
    tables.publish(0, 0, handles(pages), mapping_staging.span(), context.execution_view());
    context.synchronize();
    const std::vector<std::int32_t> mapping = read_mapping(tables.row(0), pages.size());
    failures += expect(mapping == std::vector<std::int32_t>({0, 1}),
                       "execution mapping did not preserve logical order");
    failures +=
        expect_size(pool.allocated_pages(), 2, "mapping publication changed page ownership");

    reservation->clear();
    failures += expect_size(pool.reserved_pages(), 0, "reservation clear");
    const auto stale = pages.back().slot();
    pages.back().release();
    bool stale_rejected = false;
    try {
        (void)pool.page_index(stale);
    } catch (const std::invalid_argument&) { stale_rejected = true; }
    failures += expect(stale_rejected, "released page capability remained usable");

    PlannedCache other_plan = plan_cache(1, 1, 1, geometry);
    ninfer::DeviceBuffer other_arena(context.device(), other_plan.bytes);
    context::PageAllocator other(1, 1);
    const std::uint32_t before                         = pool.reserved_pages();
    const context::PageReservationRequest impossible[] = {
        {.pool = &pool, .pages = 2},
        {.pool = &other, .pages = 2},
    };
    bool bundle_failed = false;
    try {
        auto unused = context::reserve_page_bundle(impossible);
        (void)unused;
    } catch (const std::bad_alloc&) { bundle_failed = true; }
    failures += expect(bundle_failed, "impossible multi-pool reservation succeeded");
    failures += expect_size(pool.reserved_pages(), before, "failed bundle changed the first pool");
    return failures;
}

int exercise_layout_and_transfer(ninfer::DeviceContext& context, ninfer::KVPageGeometry geometry,
                                 const std::string& label,
                                 const std::array<std::uint32_t, 4>& run_operations) {
    int failures                  = 0;
    PlannedCache source_plan      = plan_cache(10, 8, 2, geometry);
    PlannedCache destination_plan = plan_cache(10, 8, 1, geometry);
    ninfer::DeviceBuffer source_arena(context.device(), source_plan.bytes);
    ninfer::DeviceBuffer destination_arena(context.device(), destination_plan.bytes);
    context::DevicePageStorage source(source_arena.span(), source_plan.pages);
    context::PageAllocator source_pages(0, 10);
    context::PageAllocator destination_pages(1, 10);
    ninfer::PinnedHostBuffer mapping_staging(8 * sizeof(std::int32_t));
    context::ExecutionTableStorage tables(source_arena.span(), source_plan.tables, source);
    context::DevicePageStorage destination(destination_arena.span(), destination_plan.pages);

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

    std::vector<context::PageLease> prefix   = materialize(source_pages, 3);
    std::vector<context::PageLease> blockers = materialize(source_pages, 3);
    prefix.clear();
    std::vector<context::PageLease> fragmented     = materialize(source_pages, 5);
    const std::vector<std::int32_t> source_handles = handles(fragmented);

    tables.publish(1, 0, source_handles, mapping_staging.span(), context.execution_view());
    context.synchronize();
    const std::vector<std::int32_t> physical_mapping =
        read_mapping(tables.row(1), source_handles.size());
    failures += expect(physical_mapping == std::vector<std::int32_t>({0, 1, 2, 6, 7}),
                       label + " execution row differs from logical page order");


    const std::vector<std::vector<unsigned char>> device_bytes =
        fill_device_pool(source, context.stream());
    context.synchronize();

    const ninfer::HostKVPageLayout host_layout =
        ninfer::plan_host_kv_page_layout(source.geometry());
    ninfer::PinnedHostBuffer host_backing(host_layout.page_stride * 24);
    context::HostAllocator host_arena(0, host_backing.size(), host_layout.page_stride);
    for (std::size_t i = 0; i < geometry.planes.size(); ++i) {
        if (geometry.planes[i].tokens_per_row == 1) { continue; }
        const auto& plane = source.plane(i);
        failures +=
            expect(plane.ne[1] == 16 && plane.nb[1] == plane.ne[0] * plane.nb[0] &&
                       plane.nb[2] == 16 * plane.nb[1] && plane.nb[3] == plane.ne[2] * plane.nb[2],
                   label + " compressed Device strides");
        auto different                     = geometry;
        different.planes[i].tokens_per_row = 1;

        auto stale_layout               = source_plan.pages;
        stale_layout.spec.geometry      = different;
        stale_layout.planes[i].geometry = different.planes[i];
        bool stale_rejected             = false;
        try {
            context::DevicePageStorage stale(source_arena.span(), stale_layout);
        } catch (const std::logic_error&) { stale_rejected = true; }
        failures += expect(stale_rejected, label + " Device binding accepted stale row geometry");
    }
    failures += expect(!host_arena.can_allocate(host_layout.page_stride * 25),
                       label + " oversized Host extent was reported allocatable");
    std::optional<context::HostAllocation> host =
        host_arena.allocate(host_layout.page_stride * source_handles.size());
    failures += expect(host.has_value(), label + " Host allocation failed");
    HostRecords host_view = records(*host, host_backing, host_layout);
    std::memset(host_view.data(), 0, host_layout.page_stride * host_view.page_count());
    std::uint64_t page_payload = 0;
    for (const auto& plane : host_layout.planes) { page_payload += plane.page_payload_bytes; }
    failures += expect(source.host_transfer_run_work(0) == ninfer::TransferWork{},
                       label + " empty run has transfer work");
    for (std::uint32_t pages = 1; pages <= run_operations.size(); ++pages) {
        failures +=
            expect(source.host_transfer_run_work(pages) ==
                           ninfer::TransferWork{page_payload * pages, run_operations[pages - 1]} &&
                       source.transfer_profile().host_transfer_run_work(pages) ==
                           ninfer::TransferWork{page_payload * pages, run_operations[pages - 1]},
                   label + " contiguous-run quote differs from its physical copy work");
    }
    const auto export_work = source.copy_to_host(source_handles, host_view.memory, host_layout,
                                                 context.execution_view());
    failures += expect(export_work == ninfer::TransferWork{5 * page_payload,
                                                           run_operations[2] + run_operations[1]},
                       label + " D2H work missed fragmentation or counts Host padding");
    context.synchronize();

    const std::vector<std::byte> expected =
        expected_host_records(source, physical_mapping, host_layout, device_bytes);
    failures += expect(std::memcmp(host_view.data(), expected.data(), expected.size()) == 0,
                       label + " D2H canonical page records differ from Device payload");

    std::vector<context::PageLease> restored         = materialize(destination_pages, 5);
    const std::vector<std::int32_t> restored_handles = handles(restored);
    destination.zero_pages(restored_handles, context.execution_view());
    const auto restore_work =
        destination.copy_from_host(records(*host, host_backing, host_layout).memory, host_layout,
                                   restored_handles, context.execution_view());
    failures +=
        expect(restore_work == ninfer::TransferWork{5 * page_payload, run_operations.back()},
               label + " H2D work differs from the submitted contiguous run");

    const std::array duplicate_destinations{static_cast<std::int32_t>(restored[0].slot().index),
                                            static_cast<std::int32_t>(restored[0].slot().index)};
    bool duplicate_zero_rejected = false;
    try {
        destination.zero_pages(duplicate_destinations, context.execution_view());
    } catch (const std::invalid_argument&) { duplicate_zero_rejected = true; }
    failures += expect(duplicate_zero_rejected, label + " duplicate zero destination accepted");
    bool duplicate_restore_rejected = false;
    try {
        destination.copy_from_host(records(*host, host_backing, host_layout).subview(0, 2).memory,
                                   host_layout, duplicate_destinations, context.execution_view());
    } catch (const std::invalid_argument&) { duplicate_restore_rejected = true; }
    failures += expect(duplicate_restore_rejected, label + " duplicate H2D destination accepted");

    std::optional<context::HostAllocation> roundtrip =
        host_arena.allocate(host_layout.page_stride * 5);
    failures += expect(roundtrip.has_value(), label + " roundtrip Host allocation failed");
    HostRecords roundtrip_view = records(*roundtrip, host_backing, host_layout);
    std::memset(roundtrip_view.data(), 0, host_layout.page_stride * roundtrip_view.page_count());
    destination.copy_to_host(restored_handles, roundtrip_view.memory, host_layout,
                             context.execution_view());
    context.synchronize();
    failures += expect(std::memcmp(roundtrip_view.data(), host_view.data(), expected.size()) == 0,
                       label + " Device -> Host -> Device roundtrip changed bytes");

    const auto local_work = destination.copy_page(
        static_cast<std::int32_t>(restored[0].slot().index),
        static_cast<std::int32_t>(restored[4].slot().index), context.execution_view());
    const ninfer::TransferWork expected_local{page_payload,
                                              static_cast<std::uint32_t>(geometry.planes.size())};
    failures +=
        expect(local_work == expected_local && destination.device_copy_work(1) == expected_local &&
                   destination.device_copy_work(0) == ninfer::TransferWork{},
               label + " D2D work differs from one full page-group copy");
    failures += expect(destination.copy_page(static_cast<std::int32_t>(restored[0].slot().index),
                                             static_cast<std::int32_t>(restored[0].slot().index),
                                             context.execution_view()) == ninfer::TransferWork{},
                       label + " D2D self-copy must submit no work");
    const std::int32_t copied[] = {static_cast<std::int32_t>(restored[0].slot().index),
                                   static_cast<std::int32_t>(restored[4].slot().index)};
    std::optional<context::HostAllocation> copied_host =
        host_arena.allocate(host_layout.page_stride * 2);
    HostRecords copied_view = records(*copied_host, host_backing, host_layout);
    std::memset(copied_view.data(), 0, host_layout.page_stride * 2);
    destination.copy_to_host(copied, copied_view.memory, host_layout, context.execution_view());
    context.synchronize();
    const HostRecords copied_contents = records(*copied_host, host_backing, host_layout);
    failures += expect(page_payload_equal(copied_contents, 0, copied_contents, 1),
                       label + " D2D copy did not cover the complete page-group");
    failures +=
        expect(page_payload_equal(copied_contents, 0, records(*host, host_backing, host_layout), 0),
               label + " D2D copy changed its source page");

    const std::int32_t zeroed[] = {static_cast<std::int32_t>(restored[1].slot().index)};
    destination.zero_pages(zeroed, context.execution_view());
    const std::int32_t zero_observation[] = {static_cast<std::int32_t>(restored[0].slot().index),
                                             static_cast<std::int32_t>(restored[1].slot().index)};
    std::optional<context::HostAllocation> zero_host =
        host_arena.allocate(host_layout.page_stride * 2);
    HostRecords zero_view = records(*zero_host, host_backing, host_layout);
    std::memset(zero_view.data(), 0x5a, host_layout.page_stride * 2);
    destination.copy_to_host(zero_observation, zero_view.memory, host_layout,
                             context.execution_view());
    context.synchronize();
    const HostRecords zero_contents = records(*zero_host, host_backing, host_layout);
    failures +=
        expect(page_payload_equal(zero_contents, 0, records(*host, host_backing, host_layout), 0),
               label + " selective zero changed an unselected page");
    failures += expect(page_payload_zero(zero_contents, 1),
                       label + " selective zero left page payload bytes");

    {
        context::ExecutionTableStorage destination_tables(destination_arena.span(),
                                                          destination_plan.tables, destination);

        const std::array scattered{static_cast<std::int32_t>(restored[4].slot().index),
                                   static_cast<std::int32_t>(restored[1].slot().index),
                                   static_cast<std::int32_t>(restored[3].slot().index),
                                   static_cast<std::int32_t>(restored[0].slot().index),
                                   static_cast<std::int32_t>(restored[2].slot().index)};
        destination_tables.publish(0, 0, scattered, mapping_staging.span(),
                                   context.execution_view());
        context.synchronize();
        const auto destination_mapping = read_mapping(destination_tables.row(0), scattered.size());

        auto ordered_host = host_arena.allocate(host_layout.page_stride * 7);
        if (!ordered_host) { throw std::bad_alloc(); }
        const auto guarded = records(*ordered_host, host_backing, host_layout);
        std::memset(guarded.data(), 0xa5, host_layout.page_stride * guarded.page_count());
        const auto ordered_view = guarded.subview(1, 5);

        // Both producers precede the transfers. H2D reads pinned Host bytes written by the earlier
        // D2H in this stream; no CPU wait or read separates those dependent operations.
        const auto ordered_source = fill_device_pool(source, context.stream(), 89);
        auto expected_device      = fill_device_pool(destination, context.stream(), 37);
        source.copy_to_host(source_handles, ordered_view.memory, host_layout,
                            context.execution_view());
        const auto scattered_work = destination.copy_from_host(
            records(*ordered_host, host_backing, host_layout).subview(1, 5).memory, host_layout,
            scattered, context.execution_view());
        failures += expect(scattered_work ==
                               ninfer::TransferWork{5 * page_payload, 5 * run_operations.front()},
                           label + " scattered H2D work missed the single-page lowering");
        std::vector<std::vector<unsigned char>> observed_device(destination.plane_count());
        for (std::size_t index = 0; index < destination.plane_count(); ++index) {
            const auto& plane = destination.plane(index);
            auto& observed    = observed_device[index];
            observed.resize(plane.bytes());
            CUDA_CHECK(cudaMemcpyAsync(observed.data(), plane.data, observed.size(),
                                       cudaMemcpyDeviceToHost, context.stream()));
        }
        context.synchronize();

        const auto ordered_expected =
            expected_host_records(source, physical_mapping, host_layout, ordered_source);
        for (std::size_t logical = 0; logical < scattered.size(); ++logical) {
            const auto physical = destination_mapping[logical];
            for (std::size_t index = 0; index < destination.plane_count(); ++index) {
                const auto& plane          = destination.plane(index);
                const auto& plane_geometry = geometry.planes[index];
                const auto& host_plane     = host_layout.planes[index];
                const auto element_bytes   = ninfer::dtype_size(plane_geometry.dtype);
                const auto* canonical =
                    ordered_expected.data() + logical * host_layout.page_stride + host_plane.offset;
                failures +=
                    expect(std::memcmp(ordered_view.data() + logical * host_layout.page_stride +
                                           host_plane.offset,
                                       canonical, host_plane.page_payload_bytes) == 0,
                           label + " stream-ordered D2H payload differs from its producer");
                for (std::int32_t head = 0; head < plane_geometry.head_extent; ++head) {
                    const auto device_head =
                        geometry.device_plane_order == ninfer::PagedKVPlaneOrder::PageMajor
                            ? head * plane.nb[2] + physical * plane.nb[3]
                            : physical * plane.nb[2] + head * plane.nb[3];
                    const auto row_count = geometry.page_tokens / plane_geometry.tokens_per_row;
                    for (std::uint32_t token = 0; token < row_count; ++token) {
                        for (std::int32_t element = 0; element < plane_geometry.leading_extent;
                             ++element) {
                            for (std::size_t byte = 0; byte < element_bytes; ++byte) {
                                const auto host_offset =
                                    head * host_plane.head_payload_bytes +
                                    (token * plane_geometry.leading_extent + element) *
                                        element_bytes +
                                    byte;
                                const auto device_offset = device_head + token * plane.nb[1] +
                                                           element * plane.nb[0] + byte;
                                expected_device[index][device_offset] =
                                    std::to_integer<unsigned char>(canonical[host_offset]);
                            }
                        }
                    }
                }
            }
        }
        failures += expect(observed_device == expected_device,
                           label + " scattered H2D raw Device bytes differ from the scalar oracle");
        const auto is_guard = [&](std::uint32_t page) {
            const auto* first = guarded.data() + page * host_layout.page_stride;
            return std::all_of(first, first + host_layout.page_stride,
                               [](std::byte value) { return value == std::byte{0xa5}; });
        };
        failures += expect(is_guard(0) && is_guard(6),
                           label + " Host transfer overwrote a page outside its subview");
    }

    std::optional<context::HostAllocation> split_source =
        host_arena.allocate(host_layout.page_stride * 4);
    failures += expect(split_source.has_value(), label + " split source allocation failed");
    const auto original_extent = split_source->extent();
    auto [left, right] = host_arena.split(std::move(*split_source), host_layout.page_stride);
    failures += expect(!split_source->valid() && left.extent().offset == original_extent.offset,
                       label + " split did not transfer the old ownership");
    failures += expect_size(left.bytes() / host_layout.page_stride, 1, label + " split left pages");
    failures +=
        expect_size(right.bytes() / host_layout.page_stride, 3, label + " split right pages");
    auto [middle, tail] = host_arena.split(std::move(right), host_layout.page_stride);
    middle.release();
    std::optional<context::HostAllocation> reused =
        host_arena.allocate(host_layout.page_stride * 1);
    failures += expect(reused.has_value(), label + " released Host subextent was not reusable");

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
        int failures = exercise_mixed_row_geometry();
        failures += exercise_reservation_and_mapping(context);
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
            "PageMajor", {2, 2, 2, 2});
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
            "HeadMajor", {2, 4, 6, 6});
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
            "K8V4 asymmetric PageMajor", {4, 4, 4, 4});
        failures += exercise_layout_and_transfer(
            context,
            ninfer::KVPageGeometry{
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes =
                    {
                        {ninfer::DType::I8, 8, 2, 256},
                        {ninfer::DType::I8, 8, 2, 256},
                        {ninfer::DType::FP16, 1, 2, 256},
                        {ninfer::DType::FP16, 1, 2, 256},
                    },
            },
            "Grouped PageMajor", {2, 4, 4, 4});
        failures += exercise_layout_and_transfer(
            context,
            ninfer::KVPageGeometry{
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes =
                    {
                        {ninfer::DType::I8, 1, 3, 256},
                        {ninfer::DType::I8, 1, 3, 256},
                        {ninfer::DType::I8, 1, 3, 256},
                        {ninfer::DType::I8, 1, 3, 256},
                    },
            },
            "Grouped padded PageMajor", {1, 2, 3, 4});
        const ninfer::KVPlaneGeometry index_plane{
            .dtype          = ninfer::DType::BF16,
            .leading_extent = 128,
            .head_extent    = 1,
            .tokens_per_row = 4,
        };
        const ninfer::KVPageGeometry mixed_grouped{
            .planes =
                {
                    {ninfer::DType::BF16, 256, 2, 256},
                    {ninfer::DType::FP16, 256, 2, 256},
                    {ninfer::DType::BF16, 256, 2, 256},
                    {ninfer::DType::FP16, 256, 2, 256},
                    index_plane,
                    index_plane,
                },
        };
        failures += exercise_layout_and_transfer(context, mixed_grouped,
                                                 "Mixed-row grouped PageMajor", {2, 4, 5, 6});
        const ninfer::KVPageGeometry mixed_k8v4{
            .planes =
                {
                    {ninfer::DType::FP8_E4M3FN, 256, 2, 256},
                    {ninfer::DType::U8, 128, 2, 256},
                    {ninfer::DType::FP16, 1, 2, 256},
                    {ninfer::DType::U8, 16, 2, 256},
                    index_plane,
                },
        };
        failures += exercise_layout_and_transfer(context, mixed_k8v4, "Mixed-row K8V4 PageMajor",
                                                 {5, 5, 5, 5});
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
