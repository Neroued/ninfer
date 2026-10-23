#include "core/device.h"
#include "core/state_image.h"
#include "models/common/context/checkpoint_store.h"
#include "models/common/context/directory.h"
#include "models/common/context/physical_reclaim.h"
#include "models/common/context/snapshot.h"
#include "models/common/context/transfer.h"
#include "models/common/qwen/recovery_store.h"
#include "models/qwen3_5/session/internal.h"
#include "models/qwen4_exp/session/internal.h"

#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace ninfer;
namespace context = ninfer::models::context;

namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Fixture {
    DeviceContext& device;
    StateField recurrent, raw;
    StateImageLayout state_layout;
    KVPageLayout page_layout;
    DeviceBuffer backing;
    PinnedHostBuffer host_backing;
    std::vector<DeviceBuffer> table_backings;
    std::unique_ptr<StateImageStorage> state;
    std::vector<std::unique_ptr<context::ExecutionTableStorage>> tables;
    std::unique_ptr<context::ContextDirectory> directory;
    context::PageDirectory* pages;
    context::HistoryDirectory* histories;
    std::unique_ptr<context::ContextStorage> storage;
    std::unique_ptr<context::ContextMigration> migration;

    Fixture(DeviceContext& device, std::uint32_t block_count = 8,
            std::uint32_t mapping_capacity = 4,
            KVPageGeometry geometry        = {.planes = {{DType::BF16, 8, 2, 256}}},
            std::uint32_t pool_count       = 1)
        : device(device) {
        StateImageBuilder schema;
        recurrent              = schema.add(DType::FP32, {8}, 1, "recurrent");
        raw                    = schema.add(DType::I32, {3}, 1, "indexer_tail");
        state_layout           = schema.finish();
        page_layout            = plan_kv_page_layout(geometry);
        const auto block_bytes = std::max(state_layout.image_bytes, page_layout.page_bytes);
        backing                = DeviceBuffer(device.device(), block_count * block_bytes);
        host_backing           = PinnedHostBuffer(block_count * block_bytes);
        state                  = std::make_unique<StateImageStorage>(backing.span(), state_layout);
        context::ContextDirectorySpec spec{
            .ranks = {{0, backing.size(), host_backing.size()}},
            .state = {64, {{0, {{0, state_layout.image_bytes, state_layout.alignment}}}}},
            .metadata_budget_bytes = 16 * 1024 * 1024};
        std::vector<context::ContextTableStorage> bindings;
        for (std::uint32_t pool = 0; pool < pool_count; ++pool) {
            LayoutBuilder layout;
            const auto table_layout = plan_kv_execution_tables(
                layout, {.logical_page_capacity = mapping_capacity, .table_rows = 3});
            table_backings.emplace_back(device.device(), layout.finish());
            tables.push_back(std::make_unique<context::ExecutionTableStorage>(
                table_backings.back().span(), table_layout, backing.span(),
                page_layout.page_bytes));
            bindings.push_back({pool, tables.back().get()});
            spec.pools.push_back({.pool_id       = pool,
                                  .logical_pages = block_count * 2,
                                  .histories     = 32,
                                  .page_capacity = mapping_capacity,
                                  .table_rows    = 3,
                                  .parts = {{0, {page_layout.page_bytes, page_layout.alignment}}}});
        }
        directory = std::make_unique<context::ContextDirectory>(std::move(spec));
        pages     = &directory->pool(0).pages();
        histories = &directory->pool(0).histories();
        storage   = std::make_unique<context::ContextStorage>(backing.span(), host_backing.span(),
                                                              bindings);
        migration = std::make_unique<context::ContextMigration>(device, *storage);
    }

    context::HistoryDirectory& history(std::uint32_t pool = 0) {
        return directory->pool(pool).histories();
    }

    context::PageDirectory& page(std::uint32_t pool = 0) { return directory->pool(pool).pages(); }

    ContextPoolOccupancy
    occupancy(context::ContextMemory memory = context::ContextMemory::Device) const {
        return directory->records().pool({0, memory}).usage();
    }

    std::size_t live_pages() const { return occupancy().live_bytes / page_layout.page_bytes; }

    std::size_t reserved_pages() const {
        return occupancy().reserved_bytes / page_layout.page_bytes;
    }

    std::size_t host_bytes() const {
        const auto usage = occupancy(context::ContextMemory::Host);
        return usage.live_bytes + usage.reserved_bytes + usage.pending_retirement_bytes;
    }

    std::vector<std::int64_t> table(std::int32_t row, std::size_t count,
                                    std::uint32_t pool = 0) const {
        std::vector<std::int64_t> result(count);
        CUDA_CHECK(cudaMemcpy(result.data(), tables.at(pool)->row(row).data,
                              result.size() * sizeof(std::int64_t), cudaMemcpyDeviceToHost));
        return result;
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

    context::RecordReservation reserve_history(const context::HistoryBindingDemand& demand,
                                               std::uint32_t pool) {
        context::ContextDemand whole;
        const auto id    = whole.add_history(pool, demand);
        auto reservation = directory->reserve(whole);
        if (!reservation) throw std::bad_alloc();
        expect(reservation.reservation->ready(), "fixture unexpectedly requires pool maintenance");
        return reservation.reservation->take(id);
    }

    context::HistoryBinding prepare(const context::HistoryPrefix* source, std::uint32_t frontier,
                                    std::uint32_t peak, int row = 0, bool consume = false,
                                    std::uint32_t pool = 0) {
        const std::array<const context::HistoryPrefix*, 1> consumed{source};
        auto demand = history(pool).binding_demand(
            source, frontier, peak,
            consume && source ? std::span<const context::HistoryPrefix* const>(consumed)
                              : std::span<const context::HistoryPrefix* const>{});
        return history(pool).prepare_binding(source, frontier, peak, row,
                                             reserve_history(demand, pool), consume);
    }

    context::HistoryBinding transfer(context::HistoryBinding binding, std::uint32_t pool = 0) {
        context::ContextTransaction transaction(*directory);
        transaction.add_binding(pool, std::move(binding));
        submit(transaction);
        complete(transaction);
        return std::move(*transaction.binding(pool));
    }

    context::HistoryOwner bind(const context::HistoryPrefix* source, std::uint32_t frontier,
                               std::uint32_t peak, int row = 0, std::uint32_t pool = 0) {
        auto binding = transfer(prepare(source, frontier, peak, row, false, pool), pool);
        return binding.commit();
    }

    void grow(context::HistoryId owner, std::uint32_t count) {
        auto& histories = history(owner.pool_id);
        context::ContextDemand demand;
        const auto id    = demand.add_growth(histories, owner, count);
        auto reservation = directory->reserve(demand);
        if (!reservation) throw std::bad_alloc();
        auto records = reservation.reservation->take(id);
        histories.prepare_growth(owner, records);
        histories.adopt_growth(owner, std::move(records));
    }

    void write(context::HistoryOwner& owner, std::uint32_t begin, std::uint32_t end) {
        context::ContextTransaction transaction(*directory);
        history(owner.handle().pool_id).prepare_write(owner, begin, end, transaction);
        submit(transaction);
        complete(transaction);
    }

    void trim(context::HistoryOwner& owner, std::uint32_t frontier) {
        context::ContextTransaction transaction(*directory);
        history(owner.handle().pool_id).trim(owner, frontier, transaction);
        submit(transaction);
        complete(transaction);
    }

    void settle(context::HistoryOwner& owner, std::uint32_t frontier, std::uint32_t retained) {
        context::ContextTransaction transaction(*directory);
        history(owner.handle().pool_id).settle_growth(owner, frontier, retained, transaction);
        submit(transaction);
        complete(transaction);
    }

    context::HistoryOwner create(std::uint32_t frontier, int row = 0, std::uint32_t pool = 0) {
        auto owner = bind(nullptr, 0, frontier, row, pool);
        write(owner, 0, frontier);
        history(pool).commit_frontier(owner.handle(), frontier);
        return owner;
    }

    std::vector<context::PageId> logical(context::HistoryId owner) {
        std::vector<context::PageId> result;
        auto& histories = history(owner.pool_id);
        for (std::uint32_t i = 0; i < histories.mapped_pages(owner); ++i)
            result.push_back(histories.logical_page(owner, i));
        return result;
    }

    context::RecordReservation host_reservation(std::uint32_t count, std::uint32_t pool = 0) {
        context::ContextDemand demand;
        const auto id = demand.add_history(pool, count, context::ContextMemory::Host);
        auto reserved = directory->reserve(demand);
        if (!reserved) throw std::bad_alloc();
        return reserved.reservation->take(id);
    }

    void backup(std::span<const context::PageId> records, bool drop_device = false) {
        if (records.empty()) return;
        const auto pool = records.front().pool_id;
        auto reserved   = host_reservation(records.size(), pool);
        context::ContextTransaction transaction(*directory);
        transaction.add_backups(pool, records, std::move(reserved), drop_device);
        submit(transaction);
        complete(transaction);
    }

    std::vector<std::byte> read(context::ContextRecordBinding binding) const {
        auto source = storage->range({binding.domain.memory, binding.offset, binding.bytes});
        std::vector<std::byte> result(binding.bytes);
        if (binding.domain.memory == context::ContextMemory::Device)
            CUDA_CHECK(
                cudaMemcpy(result.data(), source.data, result.size(), cudaMemcpyDeviceToHost));
        else
            std::memcpy(result.data(), source.data, result.size());
        return result;
    }

    std::vector<std::vector<std::byte>> fill(std::span<const context::PageId> records) {
        std::vector<std::vector<std::byte>> result;
        for (std::size_t p = 0; p < records.size(); ++p) {
            const auto binding = page(records[p].pool_id).physical(records[p], 0);
            auto& bytes        = result.emplace_back(binding.bytes);
            for (std::size_t i = 0; i < bytes.size(); ++i)
                bytes[i] = std::byte((i * 29 + p * 61 + 17) & 255);
            const auto target =
                storage->range({context::ContextMemory::Device, binding.offset, binding.bytes});
            CUDA_CHECK(cudaMemcpy(target.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        }
        return result;
    }

    void empty() {
        device.synchronize();
        for (std::uint32_t i = 0; i < directory->pool_count(); ++i)
            expect(history(i).occupied() == 0 && page(i).occupied() == 0,
                   "KV logical owners leaked");
        const auto usage = occupancy();
        expect(usage.live_bytes == 0 && usage.reserved_bytes == 0 && host_bytes() == 0,
               "Context record owners leaked physical capacity");
        auto a = bind(nullptr, 0, 0, 0), b = bind(nullptr, 0, 0, 1), c = bind(nullptr, 0, 0, 2);
    }
};

void prefix_retention_and_replicas(DeviceContext& device) {
    Fixture f(
        device, 8, 4,
        {.planes = {
             {DType::BF16, 8, 2, 256},
             {.dtype = DType::BF16, .leading_extent = 8, .head_extent = 1, .tokens_per_row = 4}}});
    auto& h     = *f.histories;
    auto& p     = *f.pages;
    auto writer = f.bind(nullptr, 0, 130);
    f.write(writer, 0, 65);
    h.commit_frontier(writer.handle(), 65);
    auto prefix     = h.retain_prefix(writer.handle(), 65);
    auto shallow    = h.retain_prefix(writer.handle(), 32);
    const auto tail = h.logical_page(writer.handle(), 1);
    expect(f.live_pages() == 2 && p.protected_columns(tail) == 1,
           "saving a nonaligned prefix copied its KV tail");
    f.write(writer, 65, 70);
    h.commit_frontier(writer.handle(), 70);
    expect(h.logical_page(writer.handle(), 1) == tail && f.live_pages() == 2,
           "same-writer append changed the retained tail");
    auto branch = f.bind(&prefix, 65, 66, 1);
    expect(h.logical_page(branch.handle(), 1) != tail && f.live_pages() == 3,
           "independent writer failed to isolate its shared tail");
    f.write(branch, 65, 66);
    h.commit_frontier(branch.handle(), 66);
    branch.reset();
    writer.reset();
    expect(h.committed_frontier(prefix.history()) == 65,
           "retained prefix lost its frontier after writer release");
    const std::array logical{h.logical_page(prefix.history(), 0), tail};
    f.backup(logical);
    writer = f.bind(&prefix, 65, 70);
    f.write(writer, 65, 70);
    h.commit_frontier(writer.handle(), 70);
    expect(p.host_covers(tail, 1) && !p.host_covers(tail, 6),
           "Device append invalidated unchanged Host prefix coverage");
    writer.reset();
    expect(p.drop_device_replica(tail),
           "short Host replica cannot preserve the remaining shorter prefix");
    writer = f.bind(&prefix, 65, 70);
    f.write(writer, 65, 70);
    h.commit_frontier(writer.handle(), 70);
    auto longer = h.retain_prefix(writer.handle(), 70);
    writer.reset();
    expect(!p.drop_device_replica(tail),
           "short Host coverage replaced required longer Device content");
    f.backup(std::array{tail});
    expect(p.host_covers(tail, 6) && f.host_bytes() == 2 * f.page_layout.page_bytes,
           "Host refresh leaked its replaced record or lost coverage");
    longer.reset();
    expect(p.drop_device_replica(logical[0]) && p.drop_device_replica(tail),
           "refreshed replicas could not demote Device");
    writer = f.bind(&prefix, 65, 66);
    f.write(writer, 65, 66);
    h.commit_frontier(writer.handle(), 66);
    f.trim(writer, 16);
    expect(h.committed_frontier(prefix.history()) == 65 &&
               h.committed_frontier(writer.handle()) == 16,
           "writer rewind changed a retained history");
    writer.reset();
    prefix.reset();
    expect(h.committed_frontier(shallow.history()) == 32 && p.protected_columns(logical[0]) == 32,
           "deepest-prefix release failed to lower protection");
    shallow.reset();
    f.empty();
}

void reservations_and_host(DeviceContext& device) {
    Fixture f(device);
    auto owner = f.bind(nullptr, 0, 130);
    f.write(owner, 0, 65);
    const auto logical = f.logical(owner.handle());
    expect(f.histories->committed_frontier(owner.handle()) == 0 &&
               f.table(0, 2) == std::vector<std::int64_t>{f.pages->physical(logical[0], 0).offset,
                                                          f.pages->physical(logical[1], 0).offset},
           "mapping upload changed the logical frontier or lost byte offsets");
    f.histories->commit_frontier(owner.handle(), 65);
    f.write(owner, 65, 129);
    f.settle(owner, 65, 130);
    expect(f.live_pages() == 2 && f.reserved_pages() == 1,
           "settlement lost recovery growth capacity");
    auto prefix = f.histories->retain_prefix(owner.handle(), 65);
    owner.reset();
    {
        auto occupied_host  = f.host_reservation(8);
        const auto rejected = f.directory->plan_host({.kv_pages = {2}});
        expect(!rejected.feasible() && f.pages->device_covers(logical[0], 64) &&
                   f.pages->device_covers(logical[1], 1) && !f.pages->host_resident(logical[0]),
               "failed Host placement discarded or modified the retained Device source");
    }
    {
        auto reserved  = f.host_reservation(2);
        auto cancelled = f.pages->prepare_host_replica(logical[0], 0, reserved.materialize(0));
        expect(f.pages->source_pins(logical[0]) == 1 &&
                   f.host_bytes() == 2 * f.page_layout.page_bytes,
               "Host backup failed to own source and complete target capacity");
    }
    expect(f.host_bytes() == 0 && f.pages->source_pins(logical[0]) == 0,
           "cancelled Host backup leaked capacity or read lease");
    f.backup(logical);
    expect(f.pages->drop_host_replica(logical[0], 0) &&
               f.host_bytes() == f.page_layout.page_bytes && f.pages->host_covers(logical[1], 1),
           "releasing one Host record changed an unrelated replica");
    expect(f.pages->drop_host_replica(logical[1], 0),
           "remaining redundant Host record could not release");
    f.backup(logical, true);
    expect(!f.pages->drop_host_replica(logical[0], 0), "last Host-only copy became reclaimable");
    {
        auto cancelled = f.prepare(&prefix, 65, 65);
        expect(cancelled.restores().size() == 2, "Host binding omitted local restores");
    }
    expect(f.live_pages() == 0 && f.reserved_pages() == 0,
           "cancelled restore leaked pending Device records");
    owner = f.bind(&prefix, 65, 65);
    for (const auto page : logical)
        expect(f.pages->drop_host_replica(page, 0),
               "restored redundant Host copy failed to release");
    prefix.reset();
    owner.reset();
    f.empty();

    auto occupied = f.create(1);
    auto retained = f.histories->retain_prefix(occupied.handle(), 1);
    occupied.reset();
    auto a = f.bind(nullptr, 0, 0, 0), b = f.bind(nullptr, 0, 0, 1);
    f.grow(a.handle(), 4);
    bool rejected = false;
    try {
        f.grow(b.handle(), 4);
    } catch (const std::bad_alloc&) { rejected = true; }
    expect(rejected && f.histories->reserved_growth_pages(a.handle()) == 4 &&
               f.histories->reserved_growth_pages(b.handle()) == 0,
           "failed growth mutated an existing permit");
    const context::RecordAssessment pressure{
        {{{0, context::ContextMemory::Device}, {.reclaim_bytes = f.page_layout.page_bytes}}}};
    expect(f.histories->growth_contributes(a.handle(), pressure, false),
           "unused growth permit was absent from pause capacity contribution");
    f.histories->release_growth(a.handle());
    expect(!f.histories->growth_contributes(a.handle(), pressure, false) &&
               !f.histories->growth_contributes(a.handle(), {}, true),
           "released growth still advertised record or metadata contribution");
    f.grow(a.handle(), 3);
    f.grow(b.handle(), 4);
    expect(f.reserved_pages() == 7, "retry lost released shared capacity");
    a.reset();
    b.reset();
    retained.reset();
    f.empty();

    owner = f.bind(nullptr, 0, 130);
    f.write(owner, 0, 63);
    f.histories->commit_frontier(owner.handle(), 63);
    f.write(owner, 63, 71);
    f.write(owner, 63, 64);
    expect(f.histories->mapped_pages(owner.handle()) == 2 && f.reserved_pages() == 1,
           "a covered write discarded the speculative mapping");
    rejected = false;
    try {
        f.write(owner, 63, 193);
    } catch (const std::logic_error&) { rejected = true; }
    expect(rejected && f.histories->mapped_pages(owner.handle()) == 2,
           "an unreserved write changed existing mappings");
    f.settle(owner, 64, 64);
    expect(f.live_pages() == 1 && f.reserved_pages() == 0,
           "terminal accepted prefix retained speculative capacity");
    f.grow(owner.handle(), f.histories->write_pages(owner.handle(), 64, 65));
    f.write(owner, 64, 65);
    f.settle(owner, 65, 65);
    owner.reset();
    f.empty();
}

void fork_lifetimes(DeviceContext& device) {
    for (bool cancel : {false, true}) {
        Fixture f(device);
        auto source         = f.create(65);
        const auto original = f.logical(source.handle());
        const auto bytes    = f.fill(original);
        auto prefix         = f.histories->retain_prefix(source.handle(), 65);
        auto alias          = f.bind(&prefix, 64, 64, 1);
        {
            auto pending = f.transfer(f.prepare(&prefix, 65, 66, 2));
            expect(pending.needs_tail_copy(), "competing writer did not copy its tail");
            alias.reset();
            source.reset();
            prefix.reset();
            expect(f.pages->occupied() == 3 && f.histories->occupied() == 2,
                   "unfinished binding failed to retain source owners");
            if (!cancel) {
                auto destination  = pending.commit();
                const auto copied = f.logical(destination.handle());
                expect(copied[0] == original[0] && copied[1] != original[1],
                       "fork failed to share full pages and isolate tail");
                for (std::size_t i = 0; i < copied.size(); ++i)
                    expect(f.read(f.pages->physical(copied[i], 0)) == bytes[i],
                           "fork changed bytes after source release");
                destination.reset();
            }
        }
        f.empty();
    }
    Fixture f(device);
    auto filler   = f.create(193);
    auto retained = f.histories->retain_prefix(filler.handle(), 193);
    filler.reset();
    auto source   = f.create(65);
    auto prefix   = f.histories->retain_prefix(source.handle(), 65);
    bool rejected = false;
    try {
        auto pending = f.prepare(&prefix, 65, 193, 1);
    } catch (const std::bad_alloc&) { rejected = true; }
    expect(rejected && f.live_pages() == 6 && f.reserved_pages() == 0 &&
               f.histories->occupied() == 2,
           "failed fork changed the source or leaked its partial reservation");
    retained.reset();
    auto first = f.bind(&prefix, 65, 193, 1), second = f.bind(&prefix, 65, 65, 2);
    expect(f.histories->logical_page(first.handle(), 1) !=
               f.histories->logical_page(second.handle(), 1),
           "independent forks share a writable tail");
    source.reset();
    prefix.reset();
    first.reset();
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
           "consuming rewind changed the source or invented COW");
    endpoint.reset();
    { auto pending = f.prepare(&prefix, 64, 65, 0, true); }
    expect(f.histories->committed_frontier(prefix.history()) == 65,
           "cancelled consuming rewind truncated its source");
    auto binding = f.transfer(f.prepare(&prefix, 64, 65, 0, true));
    prefix.reset();
    source = binding.commit();
    expect(f.histories->committed_frontier(source.handle()) == 64,
           "consuming rewind adopted the wrong frontier");
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
    auto prepared = f.prepare(&prefix, 65, 66, 0);
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
    expect(!f.pages->valid(original.back()) && f.live_pages() == 66,
           "long branch lost source-release ownership");
    f.write(branch, frontier, frontier + 64);
    f.settle(branch, frontier + 64, frontier + 64);
    prefix       = f.histories->retain_prefix(branch.handle(), frontier + 64);
    auto aligned = f.histories->retain_prefix(branch.handle(), 66 * 64);
    expect(f.live_pages() == 67, "long saved prefix copied a tail");
    branch.reset();
    source = f.bind(&prefix, frontier + 64, frontier + 65);
    f.write(source, frontier + 64, frontier + 65);
    f.histories->commit_frontier(source.handle(), frontier + 65);
    source.reset();
    prefix.reset();
    expect(f.live_pages() == 66, "long deepest-prefix release did not trim its directory");
    aligned.reset();
    f.empty();
}

void prefix_bytes(DeviceContext& device, bool compressed) {
    constexpr std::uint32_t rewrite = 64 + 13, endpoint = 3 * 64 + 11;
    KVPageGeometry geometry{.planes = {{DType::I8, 8, 2, 256}, {DType::FP16, 1, 2, 256}}};
    if (compressed)
        geometry.planes.push_back(
            {.dtype = DType::BF16, .leading_extent = 128, .head_extent = 1, .tokens_per_row = 4});
    Fixture f(device, 8, 6, geometry);
    auto source         = f.create(endpoint);
    auto saved          = f.histories->retain_prefix(source.handle(), rewrite);
    auto aligned        = f.histories->retain_prefix(source.handle(), 64);
    const auto original = f.logical(source.handle());
    const auto bytes    = f.fill(original);
    expect(f.live_pages() == 4 && saved.history() == source.handle(),
           "prefix capture changed physical ownership");
    f.grow(source.handle(), 1);
    f.trim(source, 64 + 5);
    f.write(source, 64 + 5, 64 + 21);
    const auto copied = f.logical(source.handle());
    expect(copied[0] == original[0] && copied[1] != original[1],
           "rewrite did not isolate the protected suffix");
    const auto destination = f.pages->physical(copied[1], 0);
    for (const auto& plane : f.page_layout.planes) {
        const auto rows      = 64 / plane.geometry.tokens_per_row;
        const auto first     = 5 / plane.geometry.tokens_per_row;
        const auto row_bytes = plane.row_stride_bytes();
        for (std::uint32_t head = 0; head < std::uint32_t(plane.geometry.head_extent); ++head) {
            const auto offset = destination.offset + plane.tensor.region.offset +
                                head * plane.head_stride_bytes + first * row_bytes;
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(f.backing.data()) + offset, 0,
                                       (rows - first) * row_bytes, device.stream()));
        }
    }
    device.synchronize();
    f.histories->commit_frontier(source.handle(), 64 + 21);
    for (std::size_t i = 0; i < 2; ++i)
        expect(f.read(f.pages->physical(original[i], 0)) == bytes[i],
               "rewrite changed immutable prefix bytes");
    source.reset();
    saved.reset();
    aligned.reset();
    f.empty();
}

// Keeps a producer in flight independently of DMA size, making lease cancellation observable.
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
        auto& states     = f.directory->states();
        const auto image = states.reserve_destination().value();
        f.state->clear(states.physical_offset(image), device.execution_view());
        states.complete_initialization(image);
        states.freeze(image);
        states.retain_checkpoint_reference(image);
        auto owner         = f.create(128);
        auto prefix        = f.histories->retain_prefix(owner.handle(), 128);
        const auto logical = f.logical(owner.handle());
        owner.reset();
        context::ContextTransaction batch(*f.directory);
        batch.add_state_copy(states.reserve_device_to_host(image).value(), true);
        batch.add_backups(0, logical, f.host_reservation(logical.size()), true);
        ComputeGate gate(device);
        f.submit(batch);
        expect(!f.migration->ready(), "migration ignored its producing stream");
        prefix.reset();
        expect(states.release_checkpoint_owner(image),
               "last State owner could not exit during DMA");
        if (cancel) batch.cancel();
        expect(f.pages->occupied() == 2 && states.device_occupied() == 1 &&
                   f.host_bytes() == 2 * f.page_layout.page_bytes + f.state_layout.image_bytes,
               "last owner or cancellation recycled in-flight source or destination");
        gate.release();
        while (!f.migration->ready()) std::this_thread::yield();
        batch.mark_completed();
        std::size_t images = 0, pages = 0, bytes = 0;
        for (const auto& work : f.migration->observations()) {
            images += work.image_count;
            pages += work.page_groups;
            bytes += work.work.payload_bytes;
        }
        expect(images == 1 && pages == 2 &&
                   bytes == f.state_layout.image_bytes + 2 * f.page_layout.page_bytes,
               "completed DMA observations lost actual image/page work");
        if (cancel)
            batch.discard();
        else
            batch.publish();
        f.migration->finish();
        expect(states.occupied() == 0 && f.host_bytes() == 0,
               "migration retained content without an owner or reader");
        f.empty();
    }
}

void independent_host_records(DeviceContext& device) {
    Fixture f(device);
    auto owner         = f.create(192);
    const auto logical = f.logical(owner.handle());
    const auto bytes   = f.fill(logical);
    auto prefix        = f.histories->retain_prefix(owner.handle(), 192);
    owner.reset();
    f.backup(logical);
    {
        auto reader = f.pages->read_host(logical[1], 0, 64);
        expect(!f.pages->drop_host_replica(logical[1], 0),
               "Host release invalidated a live reader");
        expect(f.pages->drop_host_replica(logical[0], 0) &&
                   f.pages->drop_host_replica(logical[2], 0),
               "unrelated independently owned Host records could not retire");
    }
    f.backup(std::array{logical[0], logical[2]}, true);
    expect(f.pages->drop_device_replica(logical[1]),
           "Host record could not replace its Device source");
    auto capacity = f.directory->records()
                        .reserve(context::RecordDemand{1, context::ContextMemory::Device, 1})
                        .value();
    auto restore      = f.pages->reserve_device_replica(logical[1], 0, capacity);
    const auto source = restore.source(), destination = restore.device();
    context::ContextActions actions{{context::RecordCopy{
        {context::ContextMemory::Host, source.offset, source.bytes},
        {context::ContextMemory::Device, destination.offset, destination.bytes},
        {runtime::KvPoolId{0}},
        0,
        1}}};
    ComputeGate gate(device);
    f.migration->submit(actions);
    prefix.reset();
    expect(!f.migration->ready() && f.host_bytes() == f.page_layout.page_bytes,
           "unrelated Host owners failed to retire around an in-flight record reader");
    gate.release();
    while (!f.migration->ready()) std::this_thread::yield();
    expect(f.read(destination) == bytes[1],
           "independent Host record release changed the in-flight reader bytes");
    restore.publish();
    f.migration->finish();
    f.empty();
}

void migration_restore_chain(DeviceContext& device) {
    Fixture f(device);
    auto owner         = f.create(128);
    const auto logical = f.logical(owner.handle());
    const auto bytes   = f.fill(logical);
    auto prefix        = f.histories->retain_prefix(owner.handle(), 65);
    auto endpoint      = f.histories->retain_prefix(owner.handle(), 128);
    owner.reset();
    f.backup(logical, true);
    context::ContextTransaction batch(*f.directory);
    batch.add_binding(0, f.prepare(&prefix, 65, 66, 1));
    f.submit(batch);
    while (!f.migration->ready()) std::this_thread::yield();
    batch.mark_completed();
    std::size_t restored_pages = 0, copied_pages = 0;
    for (const auto& work : f.migration->observations()) {
        if (work.direction == runtime::ContextTransferDirection::HostToDevice)
            restored_pages += work.page_groups;
        if (work.direction == runtime::ContextTransferDirection::DeviceToDevice)
            copied_pages += work.page_groups;
    }
    expect(restored_pages == 2 && copied_pages == 1,
           "restore did not execute dependent COW in the same batch");
    batch.publish();
    f.migration->finish();
    auto restored     = batch.binding(0)->commit();
    const auto copied = f.logical(restored.handle());
    for (std::size_t i = 0; i < copied.size(); ++i)
        expect(f.read(f.pages->physical(copied[i], 0)) == bytes[i],
               "chained restore/COW changed packed KV content");
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
    auto& states     = f.directory->states();
    const auto image = states.reserve_destination().value();
    f.state->clear(states.physical_offset(image), device.execution_view());
    const std::array<std::int32_t, 3> raw_ids{4, 5, 6};
    const std::array<float, 8> state_values{1, 2, 3, 4, 5, 6, 7, 8};
    auto raw       = f.state->tensor(f.raw, states.physical_offset(image));
    auto recurrent = f.state->tensor(f.recurrent, states.physical_offset(image));
    CUDA_CHECK(cudaMemcpyAsync(raw.data, raw_ids.data(), sizeof(raw_ids), cudaMemcpyHostToDevice,
                               device.stream()));
    CUDA_CHECK(cudaMemcpyAsync(recurrent.data, state_values.data(), sizeof(state_values),
                               cudaMemcpyHostToDevice, device.stream()));
    device.synchronize();
    states.complete_initialization(image);
    states.freeze(image);
    auto owner = f.bind(nullptr, 0, 65);
    f.write(owner, 0, 65);
    const auto membership = f.logical(owner.handle());
    for (auto frontier : {3U, 4U, 6U, 7U}) {
        f.histories->commit_frontier(owner.handle(), frontier);
        expect(f.pages->required_columns(membership[0]) / 4 == frontier / 4,
               "mixed row coverage rounded an incomplete index block upward");
    }

    struct Metadata {
        std::uint32_t frontier;
        std::array<std::int32_t, 3> tail_ids;
    };

    using Snapshot = context::ContextSnapshot<Metadata, std::vector<std::int32_t>>;
    context::MetadataBudget checkpoint_budget(
        context::CheckpointStore<Snapshot>::metadata_bytes(3));
    context::CheckpointStore<Snapshot> checkpoints(3, checkpoint_budget);
    const auto point = checkpoints.reserve().value();
    std::vector<context::HistoryPrefix> prefixes;
    prefixes.push_back(f.histories->retain_prefix(owner.handle(), 7));
    checkpoints.publish(point,
                        Snapshot(states, {image, {7, raw_ids}}, std::move(prefixes),
                                 std::make_shared<const std::vector<std::int32_t>>(
                                     std::initializer_list<std::int32_t>{0, 1, 2, 3, 4, 5, 6})));
    const auto alias = checkpoints.reserve().value();
    checkpoints.publish(alias, checkpoints.at(point));
    auto lease = checkpoints.lease(point);
    expect(!checkpoints.erase(point) && checkpoints.erase(alias) &&
               states.checkpoint_references(image) == 1,
           "snapshot copy/lease ownership is inconsistent");
    owner.reset();
    expect(f.histories->committed_frontier(checkpoints.at(point).prefix(0).history()) == 7 &&
               f.pages->required_columns(membership[0]) / 4 == 1,
           "rejected index rows became visible in a saved snapshot");
    {
        context::ContextTransaction backup(*f.directory);
        backup.add_state_copy(states.reserve_device_to_host(image).value(), false);
        backup.add_backups(0, std::span(membership).first(1), f.host_reservation(1), true);
        f.submit(backup);
        f.complete(backup);
    }
    expect(!states.device_resident(image) && !f.pages->device_resident(membership[0]),
           "mixed batch failed to demote State and KV");
    {
        context::ContextTransaction restore(*f.directory);
        restore.add_state_copy(states.reserve_host_to_device(image).value(), true);
        restore.add_binding(0, f.prepare(&checkpoints.at(point).prefix(0), 7, 8));
        f.submit(restore);
        f.complete(restore);
        owner = restore.binding(0)->commit();
    }
    raw       = f.state->tensor(f.raw, states.physical_offset(image));
    recurrent = f.state->tensor(f.recurrent, states.physical_offset(image));
    std::array<std::int32_t, 3> observed_ids{};
    std::array<float, 8> observed_state{};
    CUDA_CHECK(
        cudaMemcpy(observed_ids.data(), raw.data, sizeof(observed_ids), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(observed_state.data(), recurrent.data, sizeof(observed_state),
                          cudaMemcpyDeviceToHost));
    expect(observed_ids == raw_ids && observed_state == state_values &&
               checkpoints.at(point).continuation().metadata.frontier == 7 &&
               checkpoints.at(point).identity() == std::vector<std::int32_t>({0, 1, 2, 3, 4, 5, 6}),
           "snapshot restoration mismatched fixed state, indexer tail or identity");
    lease.reset();
    const auto epoch   = states.content_epoch(image);
    auto mutable_image = checkpoints.at(point).take_mutable_image();
    expect(states.content_epoch(mutable_image) == epoch && checkpoints.erase(point),
           "snapshot consumption changed State content identity");
    f.write(owner, 7, 8);
    f.histories->commit_frontier(owner.handle(), 8);
    expect(f.pages->required_columns(f.histories->logical_page(owner.handle(), 0)) / 4 == 2,
           "next token failed to complete the pending index block");
    f.trim(owner, 7);
    f.grow(owner.handle(), 1);
    f.write(owner, 7, 65);
    for (auto frontier : {63U, 64U, 65U}) {
        f.histories->commit_frontier(owner.handle(), frontier);
        const auto full = f.pages->required_columns(f.histories->logical_page(owner.handle(), 0));
        const auto tail = f.pages->required_columns(f.histories->logical_page(owner.handle(), 1));
        expect(full + tail == frontier && full / 4 + tail / 4 == frontier / 4,
               "mixed page boundary exposed incomplete pooled index rows");
    }
    owner.reset();
    expect(states.release(mutable_image), "mutable State image leaked");
    f.empty();
}

template <bool four>
void qwen_reclaim_facts(DeviceContext& device) {
    namespace q35  = models::qwen3_5;
    namespace q4   = models::qwen4_exp;
    namespace qwen = models::qwen;
    using Store    = std::conditional_t<four, q4::detail::Checkpoints, q35::detail::Checkpoints>;
    Fixture f(device, 32, 4, {.planes = {{DType::BF16, 8, 2, 256}}}, 2);
    auto main_owner = f.create(200, 0, 0), backend_owner = f.create(64, 0, 1);
    auto& states = f.directory->states();
    std::vector<context::StateId> images;
    context::MetadataBudget checkpoint_budget(Store::metadata_bytes(32));
    Store store(32, checkpoint_budget);
    const auto add = [&](std::vector<TokenId> tokens, std::uint32_t domain = 0,
                         int position_offset = 0) {
        const auto frontier = static_cast<std::uint32_t>(tokens.size());
        const auto image    = states.reserve_destination();
        expect(image.has_value(), "Qwen reclaim image allocation failed");
        f.state->clear(states.physical_offset(*image), device.execution_view());
        images.push_back(*image);
        states.complete_initialization(*image);
        states.freeze(*image);
        std::vector<context::HistoryPrefix> prefixes;
        prefixes.push_back(f.histories->retain_prefix(main_owner.handle(), frontier));
        prefixes.push_back(f.history(1).retain_prefix(backend_owner.handle(), 64));
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

    const std::array pools{context::ReclaimPool{f.histories, f.pages},
                           context::ReclaimPool{&f.history(1), &f.page(1)}};
    const auto facts = [&](std::span<const context::CheckpointHandle> excluded = {}) {
        return context::PhysicalReclaim(store, *f.directory, pools, all, excluded);
    };
    const auto release = [&](const auto& facts) {
        auto selection = facts.inventory->select();
        auto probe     = selection.probe(all);
        return selection.released();
    };
    const auto main_pages    = f.logical(main_owner.handle());
    const auto backend_pages = f.logical(backend_owner.handle());
    const auto contains      = [](const context::ContextReleaseSet& released,
                             const context::RecordRelease& record) {
        return std::find(released.records.begin(), released.records.end(), record) !=
               released.records.end();
    };
    const auto counts = [&](const context::ContextReleaseSet& released) {
        std::array<std::size_t, 4> result{};
        for (auto image : images)
            for (const auto& record : states.allocations(image))
                result[0] += contains(released, record);
        for (std::uint32_t pool = 0; pool < 2; ++pool)
            for (auto page : pool ? backend_pages : main_pages) {
                for (const auto& record :
                     f.page(pool).records(page, context::ContextMemory::Device))
                    result[pool + 1] += contains(released, record);
                for (const auto& record : f.page(pool).records(page, context::ContextMemory::Host))
                    result[3] += contains(released, record);
            }
        return result;
    };
    auto active = counts(release(facts()));
    expect(active[1] == 0 && active[2] == 0,
           "active Native histories were counted as released capacity");
    auto external = store.at(root).snapshot;
    main_owner.reset();
    backend_owner.reset();
    f.backup(main_pages);
    f.backup(backend_pages);
    auto limited = counts(release(facts()));
    expect(limited == std::array<std::size_t, 4>{6, 2, 0, 2},
           "external snapshot holders were lost or backend frontiers were replaced with Main");
    // Release the independent snapshot without retaining a replacement copy.
    { auto temporary = std::move(external); }
    auto full = counts(release(facts()));
    expect(full == std::array<std::size_t, 4>{7, 4, 1, 5},
           "Native holder inventory double-counted aliases or omitted a pool");
    {
        auto lease = store.lease(root);
        expect(counts(release(facts()))[1] == 2 && counts(release(facts()))[2] == 0,
               "checkpoint reader did not veto its physical cohorts");
    }
    f.pages->pin_source(main_pages.back());
    expect(counts(release(facts()))[1] == 3 && counts(release(facts()))[2] == 1,
           "in-flight KV reader was credited as freed storage");
    f.pages->unpin_source(main_pages.back());
    expect(counts(release(facts(std::span(&root, 1))))[0] == 6,
           "excluded semantic alias lost its independent physical reference");
    expect(evaluation.loss(removed) == 200,
           "physical residency invalidated immutable identity facts");
    store.clear();
    expect(states.occupied() == 0, "Qwen checkpoint fixture leaked State ownership");
    f.empty();
}

void accepted_frontier_and_retirement(DeviceContext& device) {
    Fixture f(device);
    auto owner = f.create(65);
    f.grow(owner.handle(), f.histories->write_pages(owner.handle(), 65, 193));
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
        prefix_bytes(device, false);
        prefix_bytes(device, true);
        migration_reader_lifetime(device);
        independent_host_records(device);
        migration_restore_chain(device);
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
