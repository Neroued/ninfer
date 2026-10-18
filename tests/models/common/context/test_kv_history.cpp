#include "core/device.h"
#include "models/common/context/checkpoint_store.h"
#include "models/common/context/directory.h"
#include "models/common/context/transfer.h"
#include "models/common/context/snapshot.h"
#include "models/common/context/physical_reclaim.h"
#include "models/common/qwen/recovery_store.h"
#include "models/qwen3_5/session/internal.h"
#include "models/qwen4_exp/session/internal.h"

#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using namespace ninfer;
namespace context = ninfer::models::context;

namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

std::vector<std::int32_t> read_block_table(const context::ExecutionTableStorage& tables, int row,
                                           std::size_t count) {
    std::vector<std::int32_t> result(count);
    const auto source = tables.matrix().slice(1, row, 1);
    CUDA_CHECK(cudaMemcpy(result.data(), source.data, result.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return result;
}

struct Fixture {
    DeviceContext& device;
    DeviceBuffer backing;
    PinnedHostBuffer host_backing;
    StateField recurrent, raw;
    std::unique_ptr<StateImageStorage> state;
    std::unique_ptr<context::DevicePageStorage> physical;
    std::unique_ptr<context::ExecutionTableStorage> tables;
    HostKVPageLayout host_layout;
    std::unique_ptr<context::ContextDirectory> directory;
    context::HostAllocator* arena;
    context::PageAllocator* allocation;
    context::PageDirectory* pages;
    context::HostDirectory* host;
    context::HistoryDirectory* histories;
    std::unique_ptr<context::ContextStorage> storage;
    std::unique_ptr<context::ContextMigration> migration;

    Fixture(DeviceContext& device, std::uint32_t physical_capacity = 8,
            std::uint32_t mapping_capacity = 4,
            KVPageGeometry geometry        = {.planes = {{DType::BF16, 8, 2, 256}}})
        : device(device) {
        LayoutBuilder layout;
        StateImageBuilder schema;
        recurrent               = schema.add(DType::FP32, {8}, 1, "recurrent");
        raw                     = schema.add(DType::I32, {3}, 1, "indexer_tail");
        const auto state_layout = schema.finish(layout, 2);
        const auto pool_layout  = plan_device_kv_page_pool(
            layout, {.page_group_count = physical_capacity, .geometry = geometry});
        const auto row_layout = plan_kv_execution_tables(
            layout, {.logical_page_capacity = mapping_capacity, .table_rows = 3});
        backing  = DeviceBuffer(device.device(), layout.finish());
        state    = std::make_unique<StateImageStorage>(backing.span(), state_layout);
        physical = std::make_unique<context::DevicePageStorage>(backing.span(), pool_layout);
        tables =
            std::make_unique<context::ExecutionTableStorage>(backing.span(), row_layout, *physical);
        host_layout           = plan_host_kv_page_layout(geometry);
        const auto host_bytes = host_layout.page_stride * physical_capacity;
        context::ContextDirectorySpec spec{
            .state = {2 + static_cast<std::uint32_t>(host_bytes / state->image_bytes()), 2,
                      state->image_bytes()},
            .host  = {0, host_bytes, state->image_bytes(), physical_capacity * 2},
            .pools = {{0, physical_capacity, physical_capacity * 2, 16, mapping_capacity, 3,
                       geometry, physical->transfer_profile()}}};
        host_backing = PinnedHostBuffer(spec.host.bytes);
        directory    = std::make_unique<context::ContextDirectory>(std::move(spec));
        arena        = &directory->host_allocation();
        allocation   = &directory->pool(0).allocation();
        pages        = &directory->pool(0).pages();
        host         = &directory->host();
        histories    = &directory->pool(0).histories();
        const std::array pools{context::ContextPoolStorage{0, physical.get(), tables.get()}};
        storage = std::make_unique<context::ContextStorage>(*state, pools, 0, host_backing.span());
        migration = std::make_unique<context::ContextMigration>(device, *storage);
    }

    void submit(context::ContextTransaction& transaction) {
        transaction.prepare_actions();
        migration->submit(transaction.actions());
        transaction.mark_submitted();
    }

    void complete(context::ContextTransaction& transaction) {
        while (!migration->ready()) std::this_thread::yield();
        transaction.mark_completed();
        transaction.publish();
        migration->finish();
    }

    context::HistoryBinding transfer(context::HistoryBinding binding) {
        context::ContextTransaction transaction(*directory);
        transaction.add_binding(0, std::move(binding));
        submit(transaction);
        complete(transaction);
        return std::move(*transaction.binding(0));
    }

    context::HistoryOwner bind(const context::HistoryPrefix* source, std::uint32_t frontier,
                               std::uint32_t peak, int row = 0) {
        auto binding = transfer(histories->prepare_binding(source, frontier, peak, row));
        return binding.commit();
    }

    void write(context::HistoryOwner& owner, std::uint32_t begin, std::uint32_t end) {
        context::ContextTransaction transaction(*directory);
        histories->prepare_write(owner, begin, end, transaction);
        submit(transaction);
        complete(transaction);
    }

    void trim(context::HistoryOwner& owner, std::uint32_t frontier) {
        context::ContextTransaction transaction(*directory);
        histories->trim(owner, frontier, transaction);
        submit(transaction);
        complete(transaction);
    }

    void settle(context::HistoryOwner& owner, std::uint32_t frontier, std::uint32_t retained) {
        context::ContextTransaction transaction(*directory);
        histories->settle_growth(owner, frontier, retained, transaction);
        auto pending = std::move(transaction);
        submit(pending);
        complete(pending);
    }

    context::HistoryOwner create(std::uint32_t frontier, int row = 0) {
        auto owner = bind(nullptr, 0, frontier, row);
        write(owner, 0, frontier);
        histories->commit_frontier(owner.handle(), frontier);
        return owner;
    }

    std::vector<context::PageId> logical(context::HistoryId h) {
        std::vector<context::PageId> result;
        for (std::uint32_t i = 0; i < histories->mapped_pages(h); ++i)
            result.push_back(histories->logical_page(h, i));
        return result;
    }

    static std::vector<std::int32_t> indices(std::span<const context::PageSlot> slots) {
        std::vector<std::int32_t> result;
        for (auto slot : slots) result.push_back(slot.index);
        return result;
    }

    MemorySpan host_view(const context::HostAllocation& allocation) const {
        return storage->host_extent(allocation.extent());
    }

    MemorySpan host_view(context::HostPageRange range) const {
        return storage->host_extent(range.extent);
    }

    void copy_to_host(std::span<const context::PageSlot> slots, MemorySpan destination) {
        const auto ids = indices(slots);
        physical->copy_to_host(ids, destination, host_layout, device.execution_view());
    }

    void zero_pages(std::span<const context::PageSlot> slots) {
        const auto ids = indices(slots);
        physical->zero_pages(ids, device.execution_view());
    }

    context::HostReplicaId backup(std::span<const context::PageId> membership) {
        auto reservation = host->prepare(*pages, membership);
        expect(reservation.has_value(), "Host backup reservation failed");
        copy_to_host(host->device_sources(*reservation),
                     host_view(host->writable_view(*reservation)));
        device.synchronize();
        return host->publish(std::move(*reservation));
    }

    void empty() {
        device.synchronize();
        expect(histories->occupied() == 0 && pages->occupied() == 0 &&
                   allocation->allocated_pages() == 0 && allocation->reserved_pages() == 0 &&
                   arena->occupied_bytes() == 0,
               "KV ownership leaked resources");
        auto first = bind(nullptr, 0, 0, 0), second = bind(nullptr, 0, 0, 1),
             third = bind(nullptr, 0, 0, 2);
    }
};

std::vector<std::vector<unsigned char>> fill_device_pool(context::DevicePageStorage& pool,
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

void prefix_retention_and_replicas(DeviceContext& device) {
    const KVPageGeometry geometry{
        .planes = {
            {DType::BF16, 8, 2, 256},
            {.dtype = DType::BF16, .leading_extent = 8, .head_extent = 1, .tokens_per_row = 4}}};
    Fixture fixture(device, 8, 4, geometry);
    auto& histories         = *fixture.histories;
    auto& physical          = *fixture.physical;
    auto& pages             = *fixture.pages;
    auto& host              = *fixture.host;
    auto& arena             = *fixture.arena;
    const auto& host_layout = fixture.host_layout;
    auto writer             = fixture.bind(nullptr, 0, 130, 0);
    fixture.write(writer, 0, 65);
    const std::array original{histories.physical_page(writer.handle(), 0),
                              histories.physical_page(writer.handle(), 1)};
    fixture.zero_pages(original);
    device.synchronize();
    histories.commit_frontier(writer.handle(), 65);
    auto prefix     = histories.retain_prefix(writer.handle(), 65);
    auto shallow    = histories.retain_prefix(writer.handle(), 32);
    const auto tail = histories.logical_page(writer.handle(), 1);
    expect(fixture.allocation->allocated_pages() == 2 && pages.protected_columns(tail) == 1,
           "saving a nonaligned prefix copied its KV tail");
    fixture.write(writer, 65, 70);
    device.synchronize();
    histories.commit_frontier(writer.handle(), 70);
    expect(histories.logical_page(writer.handle(), 1) == tail &&
               fixture.allocation->allocated_pages() == 2,
           "same-writer append changed the retained tail");
    auto branch = fixture.bind(&prefix, 65, 66, 1);
    expect(histories.logical_page(branch.handle(), 1) != tail &&
               fixture.allocation->allocated_pages() == 3,
           "independent writer did not isolate the shared tail");
    fixture.write(branch, 65, 66);
    device.synchronize();
    histories.commit_frontier(branch.handle(), 66);
    branch.reset();
    writer.reset();
    expect(histories.committed_frontier(prefix.history()) == 65 &&
               fixture.allocation->allocated_pages() == 2,
           "retained prefix lost its history after writer release");
    const std::array logical{histories.logical_page(prefix.history(), 0), tail};
    auto backup = host.prepare(pages, logical);
    expect(backup.has_value(), "Host snapshot reservation failed");
    fixture.copy_to_host(host.device_sources(*backup),
                         fixture.host_view(host.writable_view(*backup)));
    device.synchronize();
    (void)host.publish(std::move(*backup));
    writer = fixture.bind(&prefix, 65, 70, 0);
    fixture.write(writer, 65, 70);
    device.synchronize();
    histories.commit_frontier(writer.handle(), 70);
    expect(pages.host_covers(tail, 1) && !pages.host_covers(tail, 6),
           "Device append changed the Host replica's valid coverage");
    writer.reset();
    expect(pages.drop_device_replica(tail),
           "short Host replica could not retain the surviving shorter prefix");
    writer = fixture.bind(&prefix, 65, 70, 0);
    fixture.write(writer, 65, 70);
    device.synchronize();
    histories.commit_frontier(writer.handle(), 70);
    auto longer = histories.retain_prefix(writer.handle(), 70);
    writer.reset();
    expect(!pages.drop_device_replica(tail),
           "short Host replica replaced a required longer Device prefix");
    auto refreshed = host.prepare(pages, std::array{tail});
    expect(refreshed.has_value(), "short Host replica could not be refreshed");
    fixture.copy_to_host(host.device_sources(*refreshed),
                         fixture.host_view(host.writable_view(*refreshed)));
    device.synchronize();
    (void)host.publish(std::move(*refreshed));
    expect(pages.host_covers(tail, 6) && arena.occupied_bytes() == 2 * host_layout.page_stride,
           "Host refresh retained its superseded allocation or lost coverage");
    longer.reset();
    expect(pages.drop_device_replica(logical[0]) && pages.drop_device_replica(tail),
           "refreshed Host replica could not replace Device");
    writer = fixture.bind(&prefix, 65, 66, 0);
    fixture.write(writer, 65, 66);
    device.synchronize();
    histories.commit_frontier(writer.handle(), 66);
    fixture.trim(writer, 16);
    expect(histories.committed_frontier(prefix.history()) == 65 &&
               histories.committed_frontier(writer.handle()) == 16,
           "rewinding the writer changed an older retained history");
    writer.reset();
    prefix.reset();
    expect(histories.committed_frontier(shallow.history()) == 32 &&
               pages.protected_columns(logical[0]) == 32,
           "releasing the deepest prefix did not update local protection");
    shallow.reset();
    fixture.empty();
}

void reservations_and_host(DeviceContext& device) {
    Fixture f(device);
    auto owner = f.bind(nullptr, 0, 130);
    f.write(owner, 0, 65);
    device.synchronize();
    expect(f.histories->committed_frontier(owner.handle()) == 0 &&
               read_block_table(*f.tables, 0, 2) == std::vector<std::int32_t>({0, 1}),
           "mapping published incorrect pages or prematurely committed content");
    f.histories->commit_frontier(owner.handle(), 65);
    f.write(owner, 65, 129);
    f.settle(owner, 65, 130);
    expect(f.allocation->allocated_pages() == 2 && f.allocation->reserved_pages() == 1,
           "settlement lost the retained recovery reservation");
    auto prefix         = f.histories->retain_prefix(owner.handle(), 65);
    const auto original = f.logical(owner.handle());
    owner.reset();
    {
        auto cancelled = f.host->prepare(*f.pages, original);
        expect(cancelled && f.pages->source_pins(original[0]) == 1 &&
                   f.arena->reserved_bytes() == 2 * f.host_layout.page_stride &&
                   f.host->occupied_bytes() == 2 * f.host_layout.page_stride,
               "Host backup did not own its source and destination");
    }
    expect(f.arena->occupied_bytes() == 0 && f.host->occupied_bytes() == 0 &&
               f.pages->source_pins(original[0]) == 0,
           "cancelled Host backup leaked resources");
    auto extent = f.backup(original);
    const std::array first{original[0]}, second{original[1]};
    expect(f.host->release_page_replicas(*f.pages, first) && !f.host->valid(extent) &&
               f.pages->host_replica(original[1]).page_offset == 0 &&
               f.host->occupied_bytes() == f.host_layout.page_stride,
           "partial Host release did not rebind the retained run");
    expect(f.host->release_page_replicas(*f.pages, second) && f.arena->occupied_bytes() == 0 &&
               f.host->occupied_bytes() == 0,
           "partitioned Host allocation leaked");
    (void)f.backup(original);
    for (auto page : original)
        expect(f.pages->drop_device_replica(page), "Host replica could not replace Device");
    expect(!f.host->release_page_replicas(*f.pages, original) &&
               f.arena->can_allocate(6 * f.host_layout.page_stride) &&
               !f.arena->can_allocate(7 * f.host_layout.page_stride),
           "last Host-only copy became reclaimable");
    {
        auto cancelled = f.histories->prepare_binding(&prefix, 65, 65, 0);
        expect(cancelled.restores().size() == 2, "Host bind omitted restores");
    }
    expect(f.allocation->allocated_pages() == 0 && f.allocation->reserved_pages() == 0,
           "cancelled restore leaked pending replicas");
    owner = f.bind(&prefix, 65, 65);
    expect(f.pages->writer_identity(original[1]) != 0 && f.allocation->reserved_pages() == 0,
           "zero-growth binding did not claim its tail");
    expect(f.host->release_page_replicas(*f.pages, original),
           "restored Host replica could not release");
    prefix.reset();
    f.trim(owner, 32);
    auto shallow = f.histories->retain_prefix(owner.handle(), 32);
    owner.reset();
    (void)f.backup(std::array{original[0]});
    shallow.reset();
    f.empty();

    auto occupied = f.create(1);
    auto retained = f.histories->retain_prefix(occupied.handle(), 1);
    occupied.reset();
    auto a = f.bind(nullptr, 0, 0, 0), b = f.bind(nullptr, 0, 0, 1);
    f.histories->reserve_growth(a.handle(), 4);
    bool rejected = false;
    try {
        f.histories->reserve_growth(b.handle(), 4);
    } catch (const std::bad_alloc&) { rejected = true; }
    expect(rejected && f.histories->reserved_growth_pages(a.handle()) == 4 &&
               f.histories->reserved_growth_pages(b.handle()) == 0 &&
               f.allocation->available_pages() == 3,
           "failed growth changed existing reservations");
    f.histories->release_growth(a.handle());
    f.histories->reserve_growth(a.handle(), 3);
    f.histories->reserve_growth(b.handle(), 4);
    expect(f.allocation->available_pages() == 0, "growth retry lost released capacity");
    a.reset();
    b.reset();
    retained.reset();
    f.empty();

    owner = f.bind(nullptr, 0, 130);
    f.write(owner, 0, 63);
    f.histories->commit_frontier(owner.handle(), 63);
    f.write(owner, 63, 71);
    f.write(owner, 63, 64);
    device.synchronize();
    expect(f.histories->mapped_pages(owner.handle()) == 2 && f.allocation->reserved_pages() == 1,
           "a covered write discarded the speculative mapping");
    rejected = false;
    try {
        f.write(owner, 63, 193);
    } catch (const std::logic_error&) { rejected = true; }
    expect(rejected && f.histories->mapped_pages(owner.handle()) == 2,
           "excess write changed mappings");
    f.settle(owner, 64, 64);
    expect(f.allocation->allocated_pages() == 1 && f.allocation->reserved_pages() == 0,
           "terminal accepted prefix retained speculative capacity");
    f.histories->reserve_growth(owner.handle(), f.histories->write_pages(owner.handle(), 64, 65));
    f.write(owner, 64, 65);
    f.settle(owner, 65, 65);
    owner.reset();
    f.empty();

    owner     = f.create(193);
    prefix    = f.histories->retain_prefix(owner.handle(), 193);
    auto four = f.logical(owner.handle());
    owner.reset();
    extent = f.backup(four);
    expect(f.host->release_page_replicas(*f.pages, std::array{four[0], four[2]}) &&
               !f.host->valid(extent) &&
               f.pages->host_replica(four[1]).extent != f.pages->host_replica(four[3]).extent &&
               f.arena->occupied_bytes() == 2 * f.host_layout.page_stride &&
               f.host->occupied_bytes() == 2 * f.host_layout.page_stride,
           "alternating Host runs were not partitioned");
    prefix.reset();
    f.empty();
}

void fork_lifetimes(DeviceContext& device) {
    for (bool cancel : {false, true}) {
        Fixture f(device, 8, 4);
        auto source      = f.create(65);
        const auto bytes = fill_device_pool(*f.physical, device.stream());
        device.synchronize();
        const auto original = read_block_table(*f.tables, 0, 2);
        auto prefix         = f.histories->retain_prefix(source.handle(), 65);
        auto alias          = f.bind(&prefix, 64, 64, 1);
        {
            auto pending = f.transfer(f.histories->prepare_binding(&prefix, 65, 66, 2));
            expect(pending.needs_tail_copy(), "competing writer did not copy tail");
            alias.reset();
            source.reset();
            prefix.reset();
            expect(f.pages->occupied() == 3 && f.histories->occupied() == 2,
                   "binding failed to retain its source after owners disappeared");
            if (!cancel) {
                auto destination = pending.commit();
                device.synchronize();
                auto mapping = read_block_table(*f.tables, 2, 2);
                expect(mapping[0] == original[0] && mapping[1] != original[1],
                       "fork did not share full page and isolate tail");
                auto allocation = f.arena->allocate(2 * f.host_layout.page_stride).value();
                auto output     = f.host_view(allocation);
                std::memset(output.data, 0, 2 * f.host_layout.page_stride);
                const std::array physical{f.histories->physical_page(destination.handle(), 0),
                                          f.histories->physical_page(destination.handle(), 1)};
                f.copy_to_host(physical, output);
                device.synchronize();
                const auto expected =
                    expected_host_records(*f.physical, original, f.host_layout, bytes);
                expect(std::memcmp(output.data, expected.data(), expected.size()) == 0,
                       "fork bytes changed after source release");
                destination.reset();
            }
        }
        f.empty();
    }
    Fixture f(device);
    auto filler          = f.create(193);
    auto retained_filler = f.histories->retain_prefix(filler.handle(), 193);
    filler.reset();
    auto source   = f.create(65);
    auto prefix   = f.histories->retain_prefix(source.handle(), 65);
    bool rejected = false;
    try {
        auto binding = f.histories->prepare_binding(&prefix, 65, 193, 1);
    } catch (const std::bad_alloc&) { rejected = true; }
    expect(rejected && f.allocation->allocated_pages() == 6 &&
               f.allocation->reserved_pages() == 0 && f.histories->occupied() == 2,
           "failed fork changed its source or leaked a reservation");
    retained_filler.reset();
    auto branch = f.bind(&prefix, 65, 193, 1);
    expect(f.allocation->allocated_pages() == 3 && f.allocation->reserved_pages() == 2,
           "retried fork lost tail or growth capacity");
    auto second = f.bind(&prefix, 65, 65, 2);
    expect(f.histories->logical_page(branch.handle(), 1) !=
               f.histories->logical_page(second.handle(), 1),
           "two independent writers share a writable tail");
    source.reset();
    prefix.reset();
    branch.reset();
    second.reset();
    f.empty();

    source        = f.create(129);
    prefix        = f.histories->retain_prefix(source.handle(), 65);
    auto endpoint = f.histories->retain_prefix(source.handle(), 129);
    source.reset();
    const std::array<const context::HistoryPrefix*, 2> retiring{&prefix, &endpoint};
    const auto demand = f.histories->binding_demand(&prefix, 64, 65, retiring);
    expect(demand.reuse_history && demand.tail_copies == 0 &&
               f.histories->committed_frontier(prefix.history()) == 129,
           "prospective consuming rewind changed source or overestimated COW");
    endpoint.reset();
    { auto cancelled = f.histories->prepare_binding(&prefix, 64, 65, 0, true); }
    expect(f.histories->committed_frontier(prefix.history()) == 65,
           "cancelled consuming rewind truncated its source");
    auto binding = f.transfer(f.histories->prepare_binding(&prefix, 64, 65, 0, true));
    prefix.reset();
    source = binding.commit();
    expect(f.histories->committed_frontier(source.handle()) == 64,
           "consuming rewind adopted wrong frontier");
    source.reset();
    f.empty();
}

void shared_tail_writer(DeviceContext& device) {
    Fixture f(device);
    auto original = f.create(128);
    auto prefix   = f.histories->retain_prefix(original.handle(), 65);
    auto endpoint = f.histories->retain_prefix(original.handle(), 128);
    auto writer   = f.bind(&endpoint, 128, 128, 1);
    original.reset();
    endpoint.reset();
    f.trim(writer, 65);
    f.write(writer, 65, 66);
    f.histories->commit_frontier(writer.handle(), 66);
    expect(prefix.history() != writer.handle() && f.histories->logical_page(prefix.history(), 1) ==
                                                      f.histories->logical_page(writer.handle(), 1),
           "shared-tail fixture did not retain a distinct read-only history");
    auto prepared = f.histories->prepare_binding(&prefix, 65, 66, 0);
    expect(prepared.blocks_writer(writer.handle()) && !prepared.blocks_writer(prefix.history()),
           "tail-copy dependency did not identify the physical page's actual writer");
    auto binding = f.transfer(std::move(prepared));
    auto branch  = binding.commit();
    expect(!binding.blocks_writer(writer.handle()),
           "completed copy kept the original writer blocked");
    writer.reset();
    prefix.reset();
    branch.reset();
    f.empty();
}

void shared_directory(DeviceContext& device) {
    constexpr std::uint32_t frontier = 65 * 64 + 1;
    Fixture f(device, 72, 68);
    auto source   = f.create(frontier);
    auto prefix   = f.histories->retain_prefix(source.handle(), frontier);
    auto original = f.logical(source.handle());
    auto branch   = f.bind(&prefix, frontier, frontier + 64, 1);
    for (std::uint32_t i = 0; i < 65; ++i)
        expect(f.histories->logical_page(branch.handle(), i) == original[i] &&
                   f.pages->history_references(original[i]) == 2,
               "persistent directory lost shared full-page identity");
    source.reset();
    prefix.reset();
    expect(!f.pages->valid(original.back()) && f.allocation->allocated_pages() == 66,
           "long branch lost source-release ownership");
    f.write(branch, frontier, frontier + 64);
    f.settle(branch, frontier + 64, frontier + 64);
    prefix       = f.histories->retain_prefix(branch.handle(), frontier + 64);
    auto aligned = f.histories->retain_prefix(branch.handle(), 66 * 64);
    expect(f.allocation->allocated_pages() == 67, "long saved prefix copied a tail");
    branch.reset();
    source = f.bind(&prefix, frontier + 64, frontier + 65);
    f.write(source, frontier + 64, frontier + 65);
    f.histories->commit_frontier(source.handle(), frontier + 65);
    source.reset();
    prefix.reset();
    expect(f.allocation->allocated_pages() == 66,
           "long deepest-prefix release did not trim its directory");
    aligned.reset();
    f.empty();
}

void prefix_bytes(DeviceContext& device, PagedKVPlaneOrder order, bool compressed = false) {
    constexpr std::uint32_t rewrite = 64 + 13, endpoint = 3 * 64 + 11;
    KVPageGeometry geometry{.device_plane_order = order,
                            .planes = {{DType::I8, 8, 2, 256}, {DType::FP16, 1, 2, 256}}};
    if (compressed)
        geometry.planes.push_back(
            {.dtype = DType::BF16, .leading_extent = 128, .head_extent = 1, .tokens_per_row = 4});
    Fixture f(device, 8, 6, geometry);
    auto source      = f.create(endpoint);
    auto saved       = f.histories->retain_prefix(source.handle(), rewrite);
    auto aligned     = f.histories->retain_prefix(source.handle(), 64);
    const auto bytes = fill_device_pool(*f.physical, device.stream());
    device.synchronize();
    const auto original = read_block_table(*f.tables, 0, 4);
    expect(f.allocation->allocated_pages() == 4 && saved.history() == source.handle(),
           "saving prefix changed physical ownership");
    auto output_owner = f.arena->allocate(6 * f.host_layout.page_stride).value();
    auto output       = f.host_view(output_owner);
    std::memset(output.data, 0, 6 * f.host_layout.page_stride);
    const std::array observation{f.histories->physical_page(source.handle(), 0),
                                 f.histories->physical_page(source.handle(), 1),
                                 f.histories->physical_page(source.handle(), 2),
                                 f.histories->physical_page(source.handle(), 3),
                                 f.histories->physical_page(saved.history(), 0),
                                 f.histories->physical_page(saved.history(), 1)};
    f.copy_to_host(observation, output);
    device.synchronize();
    const auto expected_source = expected_host_records(*f.physical, original, f.host_layout, bytes);
    const auto expected_view =
        expected_host_records(*f.physical, std::span(original).first(2), f.host_layout, bytes);
    expect(std::memcmp(output.data, expected_source.data(), expected_source.size()) == 0 &&
               std::memcmp(static_cast<std::byte*>(output.data) + 4 * f.host_layout.page_stride,
                           expected_view.data(), expected_view.size()) == 0,
           "saved prefix differs from independent byte oracle");
    f.histories->reserve_growth(source.handle(), 1);
    f.trim(source, 64 + 5);
    f.write(source, 64 + 5, 64 + 21);
    device.synchronize();
    auto mapping = read_block_table(*f.tables, 0, 2);
    expect(mapping[0] == original[0] && mapping[1] != original[1],
           "rewrite did not isolate protected suffix");
    for (std::size_t i = 0; i < f.physical->plane_count(); ++i) {
        const auto& plane = f.physical->plane(i);
        const auto rows   = 64 / geometry.planes[i].tokens_per_row,
                   first  = 5 / geometry.planes[i].tokens_per_row;
        for (std::uint32_t head = 0; head < geometry.planes[i].head_extent; ++head) {
            const auto offset = order == PagedKVPlaneOrder::PageMajor
                                    ? mapping[1] * plane.nb[3] + head * plane.nb[2]
                                    : head * plane.nb[3] + mapping[1] * plane.nb[2];
            CUDA_CHECK(
                cudaMemsetAsync(static_cast<std::byte*>(plane.data) + offset + first * plane.nb[1],
                                0, (rows - first) * plane.nb[1], device.stream()));
        }
    }
    f.histories->commit_frontier(source.handle(), 64 + 21);
    const std::array exported{f.histories->physical_page(saved.history(), 0),
                              f.histories->physical_page(saved.history(), 1)};
    f.copy_to_host(exported, output.subspan(0, 2 * f.host_layout.page_stride));
    device.synchronize();
    expect(std::memcmp(output.data, expected_view.data(), expected_view.size()) == 0,
           "rewrite changed immutable prefix bytes");
    source.reset();
    saved.reset();
    aligned.reset();
    output_owner = {};
    f.empty();
}

// A deterministic unfinished producer, so cancellation cannot pass just because a tiny DMA
// happened to finish before the Host checked its destination reservation.
class ComputeGate {
public:
    explicit ComputeGate(DeviceContext& device) {
        CUDA_CHECK(cudaLaunchHostFunc(
            device.stream(),
            [](void* opaque) {
                auto& released = static_cast<ComputeGate*>(opaque)->released_;
                while (!released.load(std::memory_order_acquire)) released.wait(false);
            },
            this));
    }

    ~ComputeGate() {
        release();
        (void)cudaDeviceSynchronize();
    }

    void release() {
        released_.store(true, std::memory_order_release);
        released_.notify_all();
    }
private:
    std::atomic<bool> released_{false};
};

void migration_reader_lifetime(DeviceContext& device) {
    for (const bool cancel : {false, true}) {
        Fixture f(device);
        auto& storage    = *f.state;
        auto& states     = f.directory->states();
        const auto image = states.reserve_destination().value();
        storage.clear(states.physical_slot(image), device.execution_view());
        states.complete_initialization(image);
        states.freeze(image);
        states.retain_checkpoint_reference(image);
        auto owner         = f.create(128);
        auto prefix        = f.histories->retain_prefix(owner.handle(), 128);
        const auto logical = f.logical(owner.handle());
        owner.reset();
        context::ContextTransaction batch(*f.directory);
        batch.add_state_copy(states.reserve_device_to_host(image).value(), true);
        batch.add_backup(0, logical, f.host->prepare(*f.pages, logical).value(), true);
        ComputeGate gate(device);
        f.submit(batch);
        expect(!f.migration->ready(), "migration ignored its producing stream");
        prefix.reset();
        expect(states.release_checkpoint_owner(image),
               "last State owner could not exit during DMA");
        if (cancel) batch.cancel();
        expect(f.allocation->allocated_pages() == 2 && states.device_occupied() == 1 &&
                   f.arena->reserved_bytes() ==
                       2 * f.host_layout.page_stride + storage.image_bytes(),
               "last owner or cancellation recycled in-flight storage");
        gate.release();
        while (!f.migration->ready()) std::this_thread::yield();
        batch.mark_completed();
        const auto work = f.migration->observations();
        expect(work.size() == 2 && work[0].image_count == 1 && work[1].page_groups == 2 &&
                   work[0].work.payload_bytes == storage.image_bytes(),
               "completed cancelled DMA lost its work or mixed quantity units");
        if (cancel)
            batch.discard();
        else
            batch.publish();
        f.migration->finish();
        expect(states.occupied() == 0 && states.device_occupied() == 0 &&
                   f.arena->occupied_bytes() == 0,
               "migration left State payload without an owner or reader");
        f.empty();
    }
}

void host_reader_survives_extent_partition(DeviceContext& device) {
    Fixture f(device);
    auto owner       = f.create(192);
    const auto bytes = fill_device_pool(*f.physical, device.stream());
    device.synchronize();
    const auto mapping = read_block_table(*f.tables, 0, 3);
    const auto expected =
        expected_host_records(*f.physical, std::span(mapping).subspan(1, 1), f.host_layout, bytes);
    auto prefix        = f.histories->retain_prefix(owner.handle(), 192);
    const auto logical = f.logical(owner.handle());
    owner.reset();
    {
        auto backup = f.host->prepare(*f.pages, logical).value();
        auto reader = f.pages->read_device(logical[1], 64);
        f.copy_to_host(f.host->device_sources(backup), f.host_view(f.host->writable_view(backup)));
        device.synchronize();
        (void)f.host->publish(std::move(backup));
        expect(f.pages->source_pins(logical[1]) == 1,
               "Host publication consumed another Device reader");
    }
    {
        auto reader = f.pages->read_host(logical[1], 64);
        expect(!f.host->release_page_replicas(*f.pages, logical) &&
                   f.arena->occupied_bytes() == 3 * f.host_layout.page_stride,
               "Host release invalidated a live reader or partially released its extent");
    }
    for (auto page : logical)
        expect(f.pages->drop_device_replica(page), "failed to drop Device source");
    const auto original = f.pages->host_replica(logical[1]);
    auto capacity       = f.allocation->reserve(1).value();
    auto restore        = f.pages->reserve_device_replica(logical[1], capacity);
    const std::array destination{restore.device()};
    ComputeGate gate(device);
    CudaCompletionEvent producer(device), done(device);
    producer.record(device.stream());
    producer.wait(device.transfer_stream());
    f.physical->copy_from_host(
        f.host_view(f.host->view(original.extent))
            .subspan(original.page_offset * f.host_layout.page_stride, f.host_layout.page_stride),
        f.host_layout, f.indices(destination), device.transfer_view());
    done.record(device.transfer_stream());
    prefix.reset();
    expect(!done.ready() && f.arena->occupied_bytes() == f.host_layout.page_stride &&
               f.pages->host_replica(logical[1]).extent != original.extent,
           "unrelated Host members could not retire around a borrowed byte range");
    gate.release();
    done.synchronize();
    auto allocation = f.arena->allocate(1 * f.host_layout.page_stride).value();
    auto output     = f.host_view(allocation);
    std::memset(output.data, 0, f.host_layout.page_stride);
    f.copy_to_host(destination, output);
    device.synchronize();
    expect(std::memcmp(output.data, expected.data(), expected.size()) == 0,
           "Host extent partition changed an in-flight reader's bytes");
    (void)allocation.release();
    restore.publish();
    f.empty();
}

void migration_restore_chain(DeviceContext& device, PagedKVPlaneOrder order) {
    Fixture f(device, 8, 4, {.device_plane_order = order, .planes = {{DType::BF16, 8, 2, 256}}});
    auto owner       = f.create(128);
    const auto bytes = fill_device_pool(*f.physical, device.stream());
    device.synchronize();
    const auto original = read_block_table(*f.tables, 0, 2);
    const auto expected = expected_host_records(*f.physical, original, f.host_layout, bytes);
    auto prefix         = f.histories->retain_prefix(owner.handle(), 65);
    auto endpoint       = f.histories->retain_prefix(owner.handle(), 128);
    const auto logical  = f.logical(owner.handle());
    owner.reset();
    (void)f.backup(logical);
    for (auto page : logical)
        expect(f.pages->drop_device_replica(page), "failed to offload KV source");
    context::ContextTransaction batch(*f.directory);
    batch.add_binding(0, f.histories->prepare_binding(&prefix, 65, 66, 1));
    f.submit(batch);
    while (!f.migration->ready()) std::this_thread::yield();
    batch.mark_completed();
    const auto work = f.migration->observations();
    expect(work.size() == 2 &&
               work[0].direction == runtime::ContextTransferDirection::HostToDevice &&
               work[0].page_groups == 2 &&
               work[1].direction == runtime::ContextTransferDirection::DeviceToDevice &&
               work[1].page_groups == 1,
           "restoration did not execute its pending-page COW in the same batch");
    batch.publish();
    f.migration->finish();
    auto restored      = batch.binding(0)->commit();
    const auto mapping = read_block_table(*f.tables, 1, 2);
    expect(mapping.size() == 2, "batch did not publish its reserved execution row");
    auto allocation = f.arena->allocate(2 * f.host_layout.page_stride).value();
    auto output     = f.host_view(allocation);
    std::memset(output.data, 0, 2 * f.host_layout.page_stride);
    const std::array physical{f.histories->physical_page(restored.handle(), 0),
                              f.histories->physical_page(restored.handle(), 1)};
    f.copy_to_host(physical, output);
    device.synchronize();
    expect(std::memcmp(output.data, expected.data(), expected.size()) == 0,
           "chained restore/COW changed physical KV payload");
    (void)allocation.release();
    prefix.reset();
    endpoint.reset();
    restored.reset();
    batch.discard();
    f.empty();
}

void mixed_snapshot(DeviceContext& device) {
    Fixture f(device, 8, 2,
              {.planes = {{DType::BF16, 8, 2, 256},
                          {.dtype          = DType::BF16,
                           .leading_extent = 128,
                           .head_extent    = 1,
                           .tokens_per_row = 4}}});
    auto& storage        = *f.state;
    auto& states         = f.directory->states();
    const auto recurrent = f.recurrent, raw = f.raw;
    auto image = states.reserve_destination().value();
    storage.clear(states.physical_slot(image), device.execution_view());
    const std::array<std::int32_t, 3> raw_ids{4, 5, 6};
    const std::array<float, 8> state_values{1, 2, 3, 4, 5, 6, 7, 8};
    auto raw_view       = storage.field(raw).slice(1, states.physical_slot(image), 1);
    auto recurrent_view = storage.field(recurrent).slice(1, states.physical_slot(image), 1);
    CUDA_CHECK(cudaMemcpyAsync(raw_view.data, raw_ids.data(), sizeof(raw_ids),
                               cudaMemcpyHostToDevice, device.stream()));
    CUDA_CHECK(cudaMemcpyAsync(recurrent_view.data, state_values.data(), sizeof(state_values),
                               cudaMemcpyHostToDevice, device.stream()));
    device.synchronize();
    states.complete_initialization(image);
    states.freeze(image);
    auto owner = f.bind(nullptr, 0, 65);
    f.write(owner, 0, 65);
    auto membership = f.logical(owner.handle());
    const std::array physical{f.histories->physical_page(owner.handle(), 0),
                              f.histories->physical_page(owner.handle(), 1)};
    f.zero_pages(physical);
    device.synchronize();
    for (auto frontier : {3U, 4U, 6U, 7U}) {
        f.histories->commit_frontier(owner.handle(), frontier);
        expect(f.pages->required_columns(membership[0]) / 4 == frontier / 4,
               "mixed row coverage rounded incomplete block upward");
    }
    auto mapping      = read_block_table(*f.tables, 0, 2);
    const auto& plane = f.physical->plane(1);
    auto* rejected = static_cast<std::byte*>(plane.data) + mapping[0] * plane.nb[3] + plane.nb[1];
    CUDA_CHECK(cudaMemsetAsync(rejected, 0x55, 128 * sizeof(std::uint16_t), device.stream()));
    device.synchronize();

    struct Metadata {
        std::uint32_t frontier;
        std::array<std::int32_t, 3> tail_ids;
    };

    using Snapshot = context::ContextSnapshot<Metadata, std::vector<std::int32_t>>;
    context::CheckpointStore<Snapshot> checkpoints(3);
    const auto point = checkpoints.reserve().value();
    std::vector<context::HistoryPrefix> prefixes;
    prefixes.push_back(f.histories->retain_prefix(owner.handle(), 7));
    checkpoints.publish(point,
                        Snapshot(states, {image, {7, raw_ids}}, std::move(prefixes),
                                 std::make_shared<const std::vector<std::int32_t>>(
                                     std::initializer_list<std::int32_t>{0, 1, 2, 3, 4, 5, 6})));
    auto alias = checkpoints.reserve().value();
    checkpoints.publish(alias, checkpoints.at(point));
    auto lease = checkpoints.lease(point);
    expect(!checkpoints.erase(point) && checkpoints.erase(alias) &&
               states.checkpoint_references(image) == 1,
           "snapshot lease or copied ownership is inconsistent");
    owner.reset();
    expect(f.histories->committed_frontier(checkpoints.at(point).prefix(0).history()) == 7 &&
               f.pages->required_columns(membership[0]) / 4 == 1,
           "rejected index row became visible in saved snapshot");
    {
        context::ContextTransaction backup(*f.directory);
        backup.add_state_copy(states.reserve_device_to_host(image).value(), false);
        const std::vector pages{membership[0]};
        backup.add_backup(0, pages, f.host->prepare(*f.pages, pages).value(), true);
        f.submit(backup);
        f.complete(backup);
    }
    expect(!states.device_resident(image) && !f.pages->device_resident(membership[0]),
           "mixed State/KV batch failed to demote both replicas");
    {
        context::ContextTransaction restore(*f.directory);
        restore.add_state_copy(states.reserve_host_to_device(image).value(), true);
        restore.add_binding(
            0, f.histories->prepare_binding(&checkpoints.at(point).prefix(0), 7, 8, 0));
        f.submit(restore);
        f.complete(restore);
        owner = restore.binding(0)->commit();
    }
    raw_view       = storage.field(raw).slice(1, states.physical_slot(image), 1);
    recurrent_view = storage.field(recurrent).slice(1, states.physical_slot(image), 1);
    std::array<std::int32_t, 3> observed_ids{};
    std::array<float, 8> observed_state{};
    CUDA_CHECK(cudaMemcpy(observed_ids.data(), raw_view.data, sizeof(observed_ids),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(observed_state.data(), recurrent_view.data, sizeof(observed_state),
                          cudaMemcpyDeviceToHost));
    expect(observed_ids == raw_ids && observed_state == state_values &&
               checkpoints.at(point).continuation().metadata.frontier == 7 &&
               checkpoints.at(point).identity() == std::vector<std::int32_t>({0, 1, 2, 3, 4, 5, 6}),
           "snapshot restored mismatched fixed state, tail, metadata or identity");
    lease.reset();
    const auto epoch   = states.content_epoch(image);
    auto mutable_image = checkpoints.at(point).take_mutable_image();
    expect(states.content_epoch(mutable_image) == epoch && checkpoints.erase(point),
           "consuming snapshot changed State identity");
    f.write(owner, 7, 8);
    f.histories->commit_frontier(owner.handle(), 8);
    expect(f.pages->required_columns(f.histories->logical_page(owner.handle(), 0)) / 4 == 2,
           "next token did not complete index block");
    f.trim(owner, 7);
    f.histories->reserve_growth(owner.handle(), 1);
    f.write(owner, 7, 65);
    for (auto frontier : {63U, 64U, 65U}) {
        f.histories->commit_frontier(owner.handle(), frontier);
        const auto full = f.pages->required_columns(f.histories->logical_page(owner.handle(), 0));
        const auto tail = f.pages->required_columns(f.histories->logical_page(owner.handle(), 1));
        expect(full + tail == frontier && full / 4 + tail / 4 == frontier / 4,
               "mixed page boundary exposed incomplete row");
    }
    device.synchronize();
    owner.reset();
    expect(states.release(mutable_image), "mutable image leaked");
    f.empty();
}

template <bool four>
void qwen_reclaim_facts(DeviceContext& device) {
    namespace q35   = models::qwen3_5;
    namespace q4    = models::qwen4_exp;
    namespace qwen  = models::qwen;
    using Store     = std::conditional_t<four, q4::detail::Checkpoints, q35::detail::Checkpoints>;
    using Resources = runtime::ContextResources<2>;
    Fixture main(device), backend(device);
    auto main_owner = main.create(200), backend_owner = backend.create(64);
    StateImageBuilder schema;
    (void)schema.add(DType::BF16, {8}, 1, "test");
    LayoutBuilder builder;
    const auto layout = schema.finish(builder, 16);
    DeviceBuffer backing(device.device(), builder.finish());
    StateImageStorage physical(backing.span(), layout);
    context::StateDirectory states({32, 16, physical.image_bytes()}, nullptr);
    Store store(32);
    const auto add = [&](std::vector<TokenId> tokens, std::uint32_t domain = 0,
                         int position_offset = 0) {
        const auto frontier = static_cast<std::uint32_t>(tokens.size());
        const auto image    = states.reserve_destination();
        expect(image.has_value(), "Qwen reclaim image allocation failed");
        physical.clear(states.physical_slot(*image), device.execution_view());
        states.complete_initialization(*image);
        states.freeze(*image);
        std::vector<context::HistoryPrefix> prefixes;
        prefixes.push_back(main.histories->retain_prefix(main_owner.handle(), frontier));
        prefixes.push_back(backend.histories->retain_prefix(backend_owner.handle(), 64));
        qwen::PreparedPromptData prompt;
        prompt.token_ids = tokens;
        prompt.token_types.resize(frontier);
        prompt.positions.resize(frontier * 3);
        for (unsigned axis = 0; axis < 3; ++axis)
            for (unsigned i = 0; i < frontier; ++i)
                prompt.positions[axis * frontier + i] = i + position_offset;
        const auto handle = *store.reserve();
        if constexpr (four) {
            auto identity    = std::make_shared<q4::detail::Identity>();
            identity->tokens = std::move(tokens);
            identity->tag    = domain;
            identity->positions.assign(prompt);
            identity->digests.assign(prompt);
            store.publish(handle,
                          {q4::detail::Snapshot(
                               states, {*image, {.main_frontier = frontier, .mtp_frontier = 64}},
                               std::move(prefixes), std::move(identity)),
                           runtime::CheckpointRole::SharedPrefix});
        } else {
            auto identity    = std::make_shared<q35::detail::PreparedCaptureBacking>();
            identity->ledger = std::move(tokens);
            identity->prefix_identity.assign(prompt);
            identity->digests.assign(prompt);
            store.publish(handle, {q35::detail::Snapshot(states,
                                                         {*image,
                                                          {.execution_frontier = frontier,
                                                           .text_kv_valid      = frontier,
                                                           .backend_kv_valid   = 64}},
                                                         std::move(prefixes), std::move(identity)),
                                   runtime::CheckpointRole::SharedPrefix,
                                   {.identity_tag = domain}});
        }
        return handle;
    };
    std::vector<TokenId> a(100, 1), b = a;
    const auto root = add(a);
    a.resize(200, 2);
    b.resize(200, 3);
    const auto left = add(a), right = add(b);
    a.resize(150);
    b.resize(150);
    const auto kept_left = add(a), kept_right = add(b);
    const auto domain   = add(std::vector<TokenId>(100, 1), 1);
    const auto position = add(std::vector<TokenId>(100, 1), 0, 1);
    const auto alias    = *store.reserve();
    store.publish(alias, store.at(root));
    device.synchronize();
    const std::array all{root, left, right, kept_left, kept_right, domain, position, alias};
    const std::array removed{root, left, right};
    const std::array kept{kept_left, kept_right};
    context::RecoveryIndex index;
    index.synchronize(qwen::RecoveryStoreView{store});
    auto evaluation = index.evaluate(kept);
    expect(evaluation.loss(removed) == 200 && !evaluation.prefix(domain, left) &&
               !evaluation.prefix(position, left),
           "Native Qwen identity adapter changed branch, domain, or position semantics");
    auto one = index.evaluate(std::span(&kept_left, 1));
    expect(one.loss(removed) == 300, "Native Qwen adapter lost the uncovered branch");
    auto aliases = index.evaluate(std::span(&alias, 1));
    expect(aliases.loss(std::span(&root, 1)) == 0,
           "Native alias was not an exact semantic fallback");

    const std::array pools{
        context::ReclaimPool{main.histories, main.pages, main.host_layout.page_stride},
        context::ReclaimPool{backend.histories, backend.pages, backend.host_layout.page_stride}};
    const Resources shortage{.state_slots = 16, .kv_page_groups = {8, 8}, .host_bytes = 1};
    const auto facts = [&](std::span<const context::CheckpointHandle> excluded = {}) {
        return context::PhysicalReclaim<Resources>(store, states, physical.image_bytes(), pools,
                                                   all, excluded, shortage);
    };
    const auto release = [&](const auto& facts) {
        auto selection = facts.inventory->select();
        auto probe     = selection.probe(all);
        return selection.released();
    };
    expect(release(facts()).kv_page_groups == std::array<std::uint32_t, 2>{0, 0},
           "active Native histories were counted as released capacity");
    auto external            = store.at(root).snapshot;
    const auto main_pages    = main.logical(main_owner.handle());
    const auto backend_pages = backend.logical(backend_owner.handle());
    main_owner.reset();
    backend_owner.reset();
    (void)main.backup(main_pages);
    (void)backend.backup(backend_pages);
    auto limited = release(facts());
    expect(limited.state_slots == 6 &&
               limited.kv_page_groups == std::array<std::uint32_t, 2>{2, 0} &&
               limited.host_bytes == 2 * main.host_layout.page_stride,
           "external snapshot holders were lost or backend frontiers were replaced with Main");
    // Release the independent snapshot without retaining a replacement copy.
    { auto temporary = std::move(external); }
    auto full = release(facts());
    expect(full.state_slots == 7 && full.kv_page_groups == std::array<std::uint32_t, 2>{4, 1} &&
               full.host_bytes ==
                   4 * main.host_layout.page_stride + backend.host_layout.page_stride,
           "Native holder inventory double-counted aliases or omitted a pool");
    {
        auto lease = store.lease(root);
        expect(release(facts()).kv_page_groups == std::array<std::uint32_t, 2>{2, 0},
               "checkpoint reader did not veto its physical cohorts");
    }
    main.pages->pin_source(main_pages.back());
    expect(release(facts()).kv_page_groups == std::array<std::uint32_t, 2>{3, 1},
           "in-flight KV reader was credited as freed storage");
    main.pages->unpin_source(main_pages.back());
    expect(release(facts(std::span(&root, 1))).state_slots == 6,
           "excluded semantic alias lost its independent physical reference");
    expect(evaluation.loss(removed) == 200,
           "physical residency invalidated immutable identity facts");
    store.clear();
    expect(states.occupied() == 0, "Qwen checkpoint fixture leaked State ownership");
    main.empty();
    backend.empty();
}

void accepted_frontier_and_retirement(DeviceContext& device) {
    Fixture f(device);
    auto owner = f.create(65);
    f.histories->reserve_growth(owner.handle(), f.histories->write_pages(owner.handle(), 65, 193));
    f.write(owner, 65, 193);
    const auto unused = f.histories->logical_page(owner.handle(), 3);
    context::ContextTransaction commit(*f.directory);
    f.histories->prepare_commit(owner, 66, 129, commit);
    expect(f.histories->committed_frontier(owner.handle()) == 65,
           "commit preparation published an unaccepted frontier");
    expect(f.pages->valid(unused) && f.pages->source_pins(unused),
           "unaccepted write tail was recycled before retirement");
    f.submit(commit);
    expect(f.histories->committed_frontier(owner.handle()) == 65,
           "physical submission adopted a logical commit");
    commit.adopt();
    expect(f.histories->committed_frontier(owner.handle()) == 66 &&
               f.histories->mapped_pages(owner.handle()) == 2,
           "accepted commit did not adopt its exact prefix");
    f.complete(commit);
    expect(!f.pages->valid(unused) && f.histories->reserved_growth_pages(owner.handle()) == 1,
           "retirement did not release the tail and restore recovery capacity");
    owner.reset();
    f.empty();
}

} // namespace

int main() {
    try {
        int count        = 0;
        const auto error = cudaGetDeviceCount(&count);
        if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || count == 0)
            return 77;
        CUDA_CHECK(error);
        DeviceContext device(0);
        accepted_frontier_and_retirement(device);
        prefix_retention_and_replicas(device);
        reservations_and_host(device);
        fork_lifetimes(device);
        shared_tail_writer(device);
        shared_directory(device);
        prefix_bytes(device, PagedKVPlaneOrder::PageMajor);
        prefix_bytes(device, PagedKVPlaneOrder::HeadMajor);
        prefix_bytes(device, PagedKVPlaneOrder::PageMajor, true);
        migration_reader_lifetime(device);
        host_reader_survives_extent_partition(device);
        migration_restore_chain(device, PagedKVPlaneOrder::PageMajor);
        migration_restore_chain(device, PagedKVPlaneOrder::HeadMajor);
        mixed_snapshot(device);
        qwen_reclaim_facts<false>(device);
        qwen_reclaim_facts<true>(device);
        std::cout << "KV history and snapshot ownership passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
