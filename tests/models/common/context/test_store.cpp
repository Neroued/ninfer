#include "core/device.h"
#include "core/state_image.h"
#include "models/common/context/directory.h"
#include "models/common/context/transfer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace store = ninfer::models::context;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void test_release_set_equality() {
    store::ContextReleaseSet planned{.records   = {{{0, store::ContextMemory::Device}, {0, 5, 1}},
                                                   {{0, store::ContextMemory::Device}, {0, 2, 1}}},
                                     .states    = {{7, 1}, {3, 1}},
                                     .pages     = {{0, 8, 1}, {0, 4, 1}},
                                     .histories = {{0, 9, 1}, {0, 6, 1}}};
    store::ContextReleaseSet observed;
    observed.merge(planned);
    expect(planned == observed, "release proof equality must not depend on traversal order");
    observed.records.push_back(planned.records.front());
    expect(planned == observed, "repeated release facts must not change the proof");
    observed.records.front().allocation.generation++;
    expect(planned != observed, "a changed allocation identity must invalidate the proof");
}

void test_rank_local_placement() {
    store::ContextDirectory directory({.ranks = {{0, 256, 0}, {1, 192, 0}},
                                       .state = {8, {{0, {{0, 64, 16}, {1, 96, 16}}}}},
                                       .metadata_budget_bytes = 16 * 1024 * 1024});
    auto first = directory.records().reserve(
        store::RecordDemand{store::state_record_layout(0), store::ContextMemory::Device, 1, {0}});
    expect(first.has_value(), "rank-local placement fixture reservation");
    const auto image = directory.states().reserve_destination();
    expect(image.has_value() && directory.states().physical_offset(*image, 0) !=
                                    directory.states().physical_offset(*image, 1),
           "one logical State supports independent rank-local physical offsets");
    const auto before_zero = directory.records().pool({0, store::ContextMemory::Device}).usage();
    const auto before_one  = directory.records().pool({1, store::ContextMemory::Device}).usage();
    store::ContextDemand demand;
    demand.add_state({.count = 2});
    const auto rejected = directory.reserve(demand);
    expect(!rejected &&
               directory.records().pool({0, store::ContextMemory::Device}).usage().reserved_bytes ==
                   before_zero.reserved_bytes &&
               directory.records().pool({1, store::ContextMemory::Device}).usage().reserved_bytes ==
                   before_one.reserved_bytes,
           "a rank-local capacity failure left partial reservations on another rank");
    expect(directory.states().release(*image), "rank-local State release");
    first.reset();
}

void test_state_variants_and_metadata() {
    const std::array<store::ContextRankSpec, 2> ranks{{{0, 8192, 4096}, {1, 8192, 4096}}};
    const std::array<store::ContextRecordLayout, 3> layouts{{
        {store::state_record_layout(0), {{0, {768, 256}}, {1, {512, 256}}}},
        {store::state_record_layout(1), {{0, {512, 256}}, {1, {768, 256}}}},
        {1, {{0, {256, 256}}, {1, {256, 256}}}},
    }};
    store::RecordDomains records(ranks, layouts);
    store::MetadataBudget budget(512 << 10);
    {
        store::StateDirectory states(
            {700, {{0, {{0, 768, 256}, {1, 512, 256}}}, {1, {{0, 512, 256}, {1, 768, 256}}}}},
            records, budget);
        const auto first  = states.reserve_destination(0).value();
        const auto second = states.reserve_destination(1).value();
        expect(states.device_binding(first, 0).bytes == 768 &&
                   states.device_binding(first, 1).bytes == 512 &&
                   states.device_binding(second, 0).bytes == 512 &&
                   states.device_binding(second, 1).bytes == 768,
               "State variants preserve both rank shards and the selected owner layout");
        states.complete_initialization(first);
        states.complete_initialization(second);
        states.freeze(second);
        const auto mismatched = states.reserve_destination(0).value();
        bool rejected         = false;
        try {
            (void)states.begin_write(second, mismatched);
        } catch (const std::logic_error&) { rejected = true; }
        expect(rejected && states.source_pins(second) == 0 && states.release(mismatched),
               "cross-variant fork must reject before changing ownership");
        auto backup = states.reserve_device_to_host(second).value();
        expect(backup.description(0).destination_record.bytes == 512 &&
                   backup.description(1).destination_record.bytes == 768,
               "Host transfer uses the source State layout variant");
        backup.publish(false);
        auto restore = states.reserve_host_to_device(second).value();
        restore.publish(false);
        expect(states.variant(second) == 1, "restoring State changed its layout variant");
        states.retain_checkpoint_reference(second);
        auto reader = states.read(second);
        expect(states.release_checkpoint_owner(second), "last State checkpoint owner release");
        states.retain_checkpoint_reference(second);
        reader.reset();
        expect(states.valid(second) && states.checkpoint_references(second) == 1,
               "a live reader can transfer ownership to a new snapshot");
        expect(states.release_checkpoint_owner(second), "promoted State checkpoint release");
        expect(states.release(first) && !states.valid(second), "variant State cleanup");
        expect(states.reclaim_metadata() != 0, "empty State metadata segment was not returned");
        const auto compact = budget.allocated();
        const auto quote   = states.metadata_requirement(1);
        auto descriptor    = states.reserve_descriptors(1);
        expect(budget.allocated() - compact == quote,
               "metadata quote differs from actual descriptor preparation");
        expect(states.reclaim_metadata() == 0,
               "a pending reservation lost its logical descriptor during reclaim");
        auto physical = records
                            .reserve(store::RecordDemand{
                                store::state_record_layout(1), store::ContextMemory::Device, 1, {}})
                            .value();
        physical.attach_metadata(std::move(descriptor));
        const auto adopted = states.reserve_destination(std::move(physical), 1).value();
        expect(states.variant(adopted) == 1 && states.release(adopted),
               "reserved metadata and physical records were not adopted together");
        expect(states.reclaim_metadata() != 0 && budget.allocated() == compact &&
                   !states.valid(first) && !states.valid(second),
               "reclaimed metadata retained its budget or revived a stale State identity");
        auto history =
            records.reserve(store::RecordDemand{1, store::ContextMemory::Device, 1, {1}}).value();
        auto pages = history.materialize();
        expect(pages.size() == 1 && pages.front().domain().rank == 1,
               "owner-only history allocated a replica on another rank");
    }
    expect(budget.allocated() == 0, "State metadata budget leaked after destruction");
}

void test_metadata_admission() {
    store::ContextDirectorySpec spec{.ranks                 = {{0, 8192, 4096}},
                                     .state                 = {700, {{0, {{0, 256, 256}}}}},
                                     .pools                 = {{.pool_id       = 0,
                                                                .logical_pages = 512,
                                                                .histories     = 128,
                                                                .page_capacity = 128,
                                                                .table_rows    = 2,
                                                                .parts         = {{0, {512, 256}}}}},
                                     .metadata_budget_bytes = 16 * 1024 * 1024};
    store::ContextDirectory probe(spec);
    store::ContextDemand demand;
    demand.add_state({.count = 1});
    const auto history_id =
        demand.add_history(0, probe.pool(0).histories().binding_demand(nullptr, 0, 128));
    const auto quote       = probe.assess(demand);
    const auto core_budget = spec.metadata_budget_bytes - probe.metadata_budget().limit();
    spec.metadata_budget_bytes =
        core_budget + probe.metadata_budget().allocated() + quote.metadata_required_bytes;
    {
        store::ContextDirectory exact(spec);
        const auto before = exact.metadata_budget().allocated();
        auto reserved     = exact.reserve(demand);
        expect(reserved && reserved.reservation->ready(),
               "complete metadata quote did not fund State, Page and History preparation");
        expect(exact.metadata_budget().allocated() - before <= quote.metadata_required_bytes,
               "metadata preparation exceeded its quoted peak");
        auto binding = exact.pool(0).histories().prepare_binding(
            nullptr, 0, 128, 0, reserved.reservation->take(history_id));
        expect(binding.history().valid(), "pre-reserved history descriptor was not adopted");
    }
    --spec.metadata_budget_bytes;
    store::ContextDirectory short_budget(spec);
    const auto before = short_budget.records().pool({0, store::ContextMemory::Device}).usage();
    expect(
        !short_budget.reserve(demand) &&
            short_budget.records().pool({0, store::ContextMemory::Device}).usage().reserved_bytes ==
                before.reserved_bytes,
        "metadata shortage partially reserved the physical bundle");

    store::ContextDirectory single({.ranks                 = {{0, 1024, 1024}},
                                    .state                 = {1, {{0, {{0, 256, 256}}}}},
                                    .metadata_budget_bytes = 1024 * 1024});
    auto& states     = single.states();
    const auto image = states.reserve_destination().value();
    states.complete_initialization(image);
    states.freeze(image);
    const auto destroyed = single.state_release(image);
    store::ContextReleaseSet demoted{.records = destroyed.records};
    store::ContextDemand state_demand;
    state_demand.add_state({.count = 1});
    expect(!single.assess(state_demand, demoted).feasible() &&
               single.assess(state_demand, destroyed).feasible(),
           "moving a physical replica was mistaken for freeing a logical State descriptor");
    expect(states.release(image), "metadata admission fixture retained State");
}

void test_mixed_context_demand() {
    store::ContextDirectory directory(
        {.ranks = {{0, 8192, 4096}, {1, 8192, 4096}},
         .state = {16, {{0, {{0, 768, 256}, {1, 512, 256}}}, {1, {{0, 512, 256}, {1, 768, 256}}}}},
         .pools = {{.pool_id       = 0,
                    .logical_pages = 32,
                    .histories     = 8,
                    .page_capacity = 8,
                    .table_rows    = 2,
                    .parts         = {{0, {256, 256}}, {1, {256, 256}}}}},
         .metadata_budget_bytes = 1024 * 1024});
    auto& histories = directory.pool(0).histories();
    const std::array<std::uint32_t, 1> owner_zero{0}, owner_one{1};
    store::ContextDemand demand;
    const auto state_zero = demand.add_state({.count = 1, .variant = 0});
    const auto history_zero =
        demand.add_history(0, histories.binding_demand(nullptr, 0, 64, {}, owner_zero));
    const auto state_one = demand.add_state({.count = 1, .variant = 1});
    const auto history_one =
        demand.add_history(0, histories.binding_demand(nullptr, 0, 128, {}, owner_one));
    const auto host = demand.add_history(0, 2, store::ContextMemory::Host, {1});
    {
        auto reserved = directory.reserve(demand);
        expect(reserved && reserved.reservation->ready(),
               "mixed variants, owners and memory domains must reserve atomically");
        auto& bundle = *reserved.reservation;
        const auto zero =
            directory.states().reserve_destination(bundle.take(state_zero), 0).value();
        const auto one = directory.states().reserve_destination(bundle.take(state_one), 1).value();
        expect(directory.states().device_binding(zero, 0).bytes == 768 &&
                   directory.states().device_binding(one, 0).bytes == 512 &&
                   directory.states().device_binding(zero, 1).bytes == 512 &&
                   directory.states().device_binding(one, 1).bytes == 768,
               "demand identities must preserve State variants");
        auto zero_records = bundle.take(history_zero);
        auto one_records  = bundle.take(history_one);
        expect(zero_records.remaining(0) == 1 && zero_records.remaining(1) == 0 &&
                   one_records.remaining(0) == 0 && one_records.remaining(1) == 2,
               "same-pool history demands must retain independent owner counts");
        const store::RecordAssessment pressure{
            {{{1, store::ContextMemory::Device}, {.reclaim_bytes = 256}}}};
        expect(!zero_records.contributes_to(pressure, false) &&
                   one_records.contributes_to(pressure, false),
               "reservation release must contribute only to its own pressured domain");
        auto split        = one_records.split(1);
        auto materialized = one_records.materialize(1);
        expect(!one_records.contributes_to(pressure, false) &&
                   split.contributes_to(pressure, false),
               "a split reservation must not count another owner's unmaterialized claim");
        one_records.restore(std::move(materialized));
        one_records.merge(std::move(split));
        auto zero_binding = histories.prepare_binding(nullptr, 0, 64, 0, std::move(zero_records),
                                                      false, owner_zero);
        auto one_binding =
            histories.prepare_binding(nullptr, 0, 128, 1, std::move(one_records), false, owner_one);
        expect(zero_binding.history() != one_binding.history(),
               "history control metadata must be reserved separately for each binding");
        auto host_records = bundle.take(host);
        auto host_page    = host_records.materialize(1);
        expect(host_page.domain() == store::ContextDomainId{1, store::ContextMemory::Host} &&
                   host_records.remaining(1) == 1 && host_records.remaining(0) == 0,
               "Host records must retain their demand's memory domain and owner");
        bool consumed = false;
        try {
            (void)bundle.take(history_zero);
        } catch (const std::logic_error&) { consumed = true; }
        expect(consumed && bundle.empty(), "a semantic reservation may only be consumed once");
        expect(directory.states().release(zero) && directory.states().release(one),
               "mixed State destinations must release independently");
    }
    for (const auto rank : {0U, 1U})
        for (const auto memory : {store::ContextMemory::Device, store::ContextMemory::Host}) {
            const auto usage = directory.records().pool({rank, memory}).usage();
            expect(usage.live_bytes == 0 && usage.reserved_bytes == 0,
                   "mixed demand cleanup leaked records");
        }
    demand.add_history(0, 64, store::ContextMemory::Host, {1});
    expect(directory.assess(demand).impossible() && !directory.reserve(demand),
           "Host capacity must be assessed separately from Device capacity");
    for (const auto rank : {0U, 1U})
        for (const auto memory : {store::ContextMemory::Device, store::ContextMemory::Host}) {
            const auto usage = directory.records().pool({rank, memory}).usage();
            expect(usage.reserved_bytes == 0,
                   "one deficient domain must not leave partial reservations in another domain");
        }
}

void test_physical_resource_accounting() {
    store::ContextDirectory directory({.ranks                 = {{0, 8192, 4096}},
                                       .state                 = {16, {{0, {{0, 256, 256}}}}},
                                       .pools                 = {{.pool_id       = 0,
                                                                  .logical_pages = 32,
                                                                  .histories     = 8,
                                                                  .page_capacity = 8,
                                                                  .table_rows    = 2,
                                                                  .parts         = {{0, {256, 256}}}}},
                                       .metadata_budget_bytes = 1024 * 1024});
    const auto usage = [&](ninfer::ContextMemoryDomain memory, bool kv) {
        for (const auto& value : directory.usage().resources)
            if (value.domain == memory && value.history_pool.has_value() == kv) return value;
        throw std::logic_error("missing Context resource observation");
    };
    auto& states     = directory.states();
    const auto state = states.reserve_destination().value();
    states.complete_initialization(state);
    states.freeze(state);
    states.retain_checkpoint_reference(state);
    states.retain_checkpoint_reference(state);
    {
        store::ContextDemand demand;
        const auto id = demand.add_history(0, 1);
        auto reserved = directory.reserve(demand);
        auto backing  = reserved.reservation->take(id);
        std::array<store::PageId, 1> page;
        auto& pages = directory.pool(0).pages();
        pages.materialize(backing, page);
        pages.retain_reference(page[0]);
        const auto s = usage(ninfer::ContextMemoryDomain::Device, false);
        const auto k = usage(ninfer::ContextMemoryDomain::Device, true);
        expect(s.live_records == 1 && s.live_bytes == 256 && k.live_records == 1 &&
                   k.live_bytes == 256,
               "identical allocator geometry must distinguish State from KV without counting "
               "references");
        auto backup = states.reserve_device_to_host(state).value();
        expect(usage(ninfer::ContextMemoryDomain::Host, false).live_bytes == 256,
               "in-flight State copy owns a separate Host replica");
        backup.publish(true);
        expect(pages.release_reference(page[0]) && pages.release_reference(page[0]),
               "shared page release");
    }
    expect(usage(ninfer::ContextMemoryDomain::Device, true).live_records == 0,
           "abandoned history binding leaked its resource count");
    expect(states.release_checkpoint_owner(state), "first shared checkpoint owner release");
    expect(usage(ninfer::ContextMemoryDomain::Device, false).live_records == 1,
           "releasing one shared owner freed its remaining replica");
    auto reader = states.read(state);
    expect(states.release_checkpoint_owner(state), "last shared checkpoint owner release");
    const auto pending = usage(ninfer::ContextMemoryDomain::Device, false);
    expect(pending.live_records == 0 && pending.pending_retirement_records == 1,
           "last-owner release must retain the borrowed physical record until retirement");
    reader.reset();
    for (const auto& resource : directory.usage().resources)
        expect(resource.live_records == 0 && resource.pending_retirement_records == 0,
               "retired physical replica leaked semantic usage");
}

void test_state_store(ninfer::DeviceContext& device) {
    ninfer::StateImageBuilder schema;
    (void)schema.add(ninfer::DType::BF16, {8, 3}, 1, "conv");
    (void)schema.add(ninfer::DType::FP32, {4, 4, 2}, 1, "recurrent");
    const auto hidden_field = schema.add(ninfer::DType::BF16, {8}, 1, "hidden");
    (void)schema.add(ninfer::DType::BF16, {4, 8, 2}, 1, "local_k");
    (void)schema.add(ninfer::DType::FP16, {4, 8, 2}, 1, "local_v");
    const auto layout = schema.finish();
    ninfer::DeviceBuffer backing(device.device(), layout.image_bytes * 4);
    ninfer::StateImageStorage physical(backing.span(), layout);
    ninfer::PinnedHostBuffer host_payload(layout.image_bytes * 2);
    store::ContextDirectory directory(
        {.ranks                 = {{0, backing.size(), host_payload.size()}},
         .state                 = {6, {{0, {{0, layout.image_bytes, layout.alignment}}}}},
         .metadata_budget_bytes = 16 * 1024 * 1024});
    auto& images          = directory.states();
    auto& host_backing    = directory.records().pool({0, store::ContextMemory::Host});
    const auto host_bytes = [&] {
        const auto usage = host_backing.usage();
        return usage.live_bytes + usage.reserved_bytes + usage.pending_retirement_bytes;
    };
    store::ContextStorage storage(backing.span(), host_payload.span(), {});
    const auto enqueue = [&](const store::StateDirectory::Transfer& transfer) {
        const auto copy         = transfer.description();
        const auto& source      = copy.source_record;
        const auto& destination = copy.destination_record;
        storage.execute(
            store::RecordCopy{{source.domain.memory, source.offset, source.bytes},
                              {destination.domain.memory, destination.offset, destination.bytes},
                              {},
                              1,
                              0},
            {}, device.transfer_view());
    };
    const auto hidden_slot = [&](std::int64_t offset) {
        return physical.tensor(hidden_field, offset);
    };

    const auto initialize = [&] {
        auto handle = images.reserve_destination();
        if (handle) {
            physical.clear(images.physical_offset(*handle), device.execution_view());
            images.complete_initialization(*handle);
        }
        return handle;
    };
    const auto source = initialize();
    expect(source.has_value(), "state source allocation");
    const std::int64_t original_slot = images.physical_offset(*source);
    const auto source_epoch          = images.content_epoch(*source);
    images.freeze(*source);
    expect(images.content_epoch(*source) == source_epoch, "freezing retains content identity");
    images.make_mutable(*source);
    expect(images.physical_offset(*source) == original_slot &&
               images.role(*source) == store::StateRole::ActiveMutable,
           "private Move preserves the logical image and physical slot");
    images.freeze(*source);
    const auto destination = images.reserve_destination();
    expect(destination.has_value(), "state destination reservation");
    const store::StateSelectors selectors = images.begin_write(*source, *destination);
    expect(selectors.source >= 0 && selectors.destination >= 0 &&
               selectors.source != selectors.destination,
           "fork resolves distinct physical selectors");
    expect(images.role(*destination) == store::StateRole::Writing,
           "pending write is not readable active content");
    expect(!images.release(*source) && !images.release(*destination),
           "fork pins both logical images");
    const auto concurrent_destination = images.reserve_destination();
    expect(concurrent_destination.has_value(), "concurrent fork destination reservation");
    (void)images.begin_write(*source, *concurrent_destination);
    expect(images.source_pins(*source) == 2 && !images.release(*source),
           "independent forks each retain the source");
    images.abort_write(*source, *concurrent_destination);
    expect(images.release(*concurrent_destination), "concurrent fork destination release");
    expect(images.source_pins(*source) == 1 && !images.release(*source) &&
               !images.release(*destination),
           "aborting one reader preserves the remaining fork's pins");
    images.complete_write(*source, *destination);
    expect(images.role(*source) == store::StateRole::CheckpointImmutable &&
               images.role(*destination) == store::StateRole::ActiveMutable &&
               images.content_epoch(*destination) != source_epoch,
           "fork commit preserves source and activates destination");
    images.retain_checkpoint_reference(*source);
    expect(!images.release(*source), "checkpoint owner keeps the source alive after fork commit");
    expect(images.release_checkpoint_owner(*source) && images.release(*destination) &&
               images.occupied() == 0,
           "checkpoint and active state ownership close after release");
    expect(!images.valid(*source), "released state generation becomes stale");

    const auto reused = initialize();
    expect(reused.has_value() && *reused != *source, "state descriptor reuse advances generation");
    expect(images.release(*reused), "reused state image releases");

    for (const bool cancel_last_reader : {false, true}) {
        const auto shared       = initialize();
        const auto first_reader = images.reserve_destination();
        const auto last_reader  = images.reserve_destination();
        expect(shared && first_reader && last_reader, "shared State reader fixture allocates");
        const auto source_slot = images.physical_offset(*shared);
        const std::array<std::uint16_t, 8> content{1, 3, 5, 7, 9, 11, 13, 15};
        const auto hidden = hidden_slot(source_slot);
        CUDA_CHECK(cudaMemcpyAsync(hidden.data, content.data(), sizeof(content),
                                   cudaMemcpyHostToDevice, device.stream()));
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared); // Cache checkpoint.
        images.retain_checkpoint_reference(*shared); // Paused request snapshot.
        (void)images.begin_write(*shared, *first_reader);
        (void)images.begin_write(*shared, *last_reader);
        expect(images.release_checkpoint_owner(*shared) &&
                   images.release_checkpoint_owner(*shared) && images.valid(*shared) &&
                   images.checkpoint_references(*shared) == 0 && images.source_pins(*shared) == 2,
               "cache eviction and paused cancellation leave execution readers alive");
        expect(!images.can_release_after_checkpoint_references(*shared, 0) &&
                   images.device_occupied() == 3,
               "retired content owner does not advertise pinned State capacity as free");
        physical.copy(source_slot, images.physical_offset(*first_reader), device.execution_view());
        device.synchronize();
        images.complete_write(*shared, *first_reader);
        expect(images.valid(*shared) && images.source_pins(*shared) == 1,
               "one completed reader does not release the remaining reader's State");
        physical.copy(source_slot, images.physical_offset(*last_reader), device.execution_view());
        std::array<std::uint16_t, 8> observed{};
        const auto last_hidden = hidden_slot(images.physical_offset(*last_reader));
        CUDA_CHECK(cudaMemcpyAsync(observed.data(), last_hidden.data, sizeof(observed),
                                   cudaMemcpyDeviceToHost, device.stream()));
        device.synchronize();
        expect(observed == content, "remaining State reader retains the exact source content");
        if (cancel_last_reader) {
            images.abort_write(*shared, *last_reader);
        } else {
            images.complete_write(*shared, *last_reader);
        }
        expect(!images.valid(*shared) && images.device_occupied() == 2,
               "last reader retirement releases the orphaned State automatically");
        const auto replacement = initialize();
        expect(replacement && *replacement != *shared && images.device_occupied() == 3,
               "retired State capacity is reusable under a new logical identity");
        expect(images.release(*replacement) && images.release(*first_reader) &&
                   images.release(*last_reader) && images.occupied() == 0,
               "retired checkpoint and execution readers leave no State allocation behind");
    }

    {
        const auto shared = initialize();
        const auto reader = images.reserve_destination();
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared);
        (void)images.begin_write(*shared, *reader);
        expect(images.release_checkpoint_owner(*shared), "read lease outlives original checkpoint");
        // Pausing an unwritten fork converts its read lease into a new snapshot owner.
        images.retain_checkpoint_reference(*shared);
        images.abort_write(*shared, *reader);
        expect(images.valid(*shared) && images.checkpoint_references(*shared) == 1 &&
                   images.source_pins(*shared) == 0,
               "new snapshot ownership is installed before its read lease retires");
        expect(images.release(*reader) && images.release_checkpoint_owner(*shared) &&
                   images.occupied() == 0,
               "reowned snapshot retires without leaking its previous read lease");
    }

    for (const bool cancel_transfer : {false, true}) {
        const auto shared = initialize();
        device.synchronize();
        images.freeze(*shared);
        images.retain_checkpoint_reference(*shared);
        auto transfer = images.reserve_device_to_host(*shared);
        if (transfer) enqueue(*transfer);
        expect(transfer && images.host_occupied() == 1 &&
                   images.release_checkpoint_owner(*shared) && images.valid(*shared),
               "submitted State transfer retains a retired checkpoint source");
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
        if (cancel_transfer) {
            transfer.reset();
        } else {
            transfer->publish(true);
        }
        expect(!images.valid(*shared) && images.occupied() == 0 && images.host_occupied() == 0 &&
                   host_bytes() == 0,
               "last transfer lease retirement frees both State replicas and reservations");
    }

    {
        const auto first  = initialize();
        const auto second = initialize();
        images.freeze(*first);
        images.freeze(*second);
        const auto first_writer  = images.reserve_destination();
        const auto second_writer = images.reserve_destination();
        (void)images.begin_write(*first, *first_writer);
        (void)images.begin_write(*second, *second_writer);
        bool rejected = false;
        try {
            images.complete_write(*first, *second_writer);
        } catch (const std::logic_error&) { rejected = true; }
        expect(rejected && images.source_pins(*first) == 1 && images.source_pins(*second) == 1,
               "write completion must name its reserved source");
        images.abort_write(*first, *first_writer);
        images.abort_write(*second, *second_writer);
        expect(images.release(*first) && images.release(*second) && images.release(*first_writer) &&
                   images.release(*second_writer),
               "independent writers release their own pins");
    }

    const auto host_source = initialize();
    expect(host_source.has_value(), "Host state source allocation");
    device.synchronize();
    images.freeze(*host_source);
    {
        auto allocation =
            directory.records().reserve(store::RecordDemand{0, store::ContextMemory::Host, 1});
        const auto occupied = host_bytes();
        auto cancelled      = images.reserve_device_to_host(*host_source, std::move(*allocation));
        expect(host_bytes() == occupied && host_backing.usage().classes.front().live_records == 1,
               "transfer consumes the existing Host reservation without reallocating");
        expect(cancelled && images.host_occupied() == 1 && host_bytes() == layout.image_bytes &&
                   !images.host_resident(*host_source) && images.source_pins(*host_source) == 1 &&
                   !images.release(*host_source),
               "State transfer reservation charges destination while its source is pinned");
    }
    expect(host_bytes() == 0 && images.host_occupied() == 0 &&
               images.source_pins(*host_source) == 0 &&
               images.residency(*host_source) == store::StateResidency::DeviceOnly,
           "cancelled State transfer releases its actual Host extent and retains source state");
    auto d2h = images.reserve_device_to_host(*host_source);
    if (d2h) enqueue(*d2h);
    expect(d2h.has_value() && images.host_occupied() == 1 &&
               images.residency(*host_source) == store::StateResidency::DeviceOnly,
           "incomplete State D2H does not publish a Host replica");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    d2h->publish(true);
    expect(images.residency(*host_source) == store::StateResidency::Both &&
               images.host_occupied() == 1 &&
               host_backing.usage().live_bytes == layout.image_bytes &&
               host_backing.usage().reserved_bytes == 0,
           "State D2H backup publishes stable Both residency");

    const auto moved_device = images.reserve_logical_destination();
    expect(moved_device.has_value(), "State replica split destination allocation");
    images.split_device_replica_identity(*host_source, *moved_device);
    expect(images.role(*moved_device) == store::StateRole::ReservedDestination,
           "replica handover does not publish a writer");
    images.complete_initialization(*moved_device);
    expect(images.residency(*host_source) == store::StateResidency::HostOnly &&
               images.host_occupied() == 1 &&
               images.residency(*moved_device) == store::StateResidency::DeviceOnly &&
               images.content_epoch(*moved_device) == images.content_epoch(*host_source),
           "State replica identity split keeps old Host content and moves Device ownership");

    const auto fork_one = images.reserve_logical_destination();
    const auto fork_two = images.reserve_logical_destination();
    expect(fork_one && fork_two, "concurrent Host State forks reserve logical destinations");
    auto h2d_one = images.reserve_host_to_device(*host_source, *fork_one);
    if (h2d_one) enqueue(*h2d_one);
    auto h2d_two = images.reserve_host_to_device(*host_source, *fork_two);
    if (h2d_two) enqueue(*h2d_two);
    expect(h2d_one && h2d_two && images.host_occupied() == 1 &&
               images.source_pins(*host_source) == 2,
           "immutable Host State source supports multiple fork pins");
    auto h2d_source = images.reserve_host_to_device(*host_source);
    if (h2d_source) enqueue(*h2d_source);
    expect(h2d_source && images.source_pins(*host_source) == 3,
           "Host replica remains shared while restoring its own Device replica");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
    bool rejected = false;
    try {
        h2d_source->publish(false);
    } catch (const std::logic_error&) { rejected = true; }
    expect(rejected && h2d_source->valid() && images.host_resident(*host_source) &&
               images.host_occupied() == 1 && !images.device_resident(*host_source) &&
               images.source_pins(*host_source) == 3,
           "restoration cannot retire Host storage retained by other transfers");
    h2d_one->publish(true);
    h2d_two->publish(true);
    expect(images.host_occupied() == 1, "independent Device forks retain one Host source");
    h2d_source->publish(false);
    expect(images.role(*fork_one) == store::StateRole::ReservedDestination &&
               images.host_occupied() == 0,
           "Host materialization is separate from native adoption");
    images.complete_initialization(*fork_one);
    images.complete_initialization(*fork_two);
    expect(images.source_pins(*host_source) == 0 &&
               images.residency(*fork_one) == store::StateResidency::DeviceOnly &&
               images.residency(*fork_two) == store::StateResidency::DeviceOnly &&
               images.content_epoch(*fork_one) == images.content_epoch(*host_source) &&
               images.content_epoch(*fork_two) == images.content_epoch(*host_source),
           "Host State forks publish independent Device destinations");
    expect(images.release(*host_source) && images.release(*moved_device) &&
               images.release(*fork_one) && images.release(*fork_two) &&
               images.host_occupied() == 0,
           "State Host/Device replica ownership closes without leaked slots");
}

void test_migration_state_fields(ninfer::DeviceContext& device) {
    using namespace ninfer;
    StateImageBuilder schema;
    (void)schema.add(DType::FP32, {4}, 1, "recurrent");
    const auto local = schema.add(DType::BF16, {8}, 2, "local_kv");
    (void)schema.add(DType::I32, {3}, 1, "tail");
    const auto group  = schema.group(local, local);
    const auto layout = schema.finish();
    DeviceBuffer backing(device.device(), layout.image_bytes * 2);
    StateImageStorage storage(backing.span(), layout);
    store::ContextDirectory directory(
        {.ranks                 = {{0, backing.size(), 0}},
         .state                 = {2, {{0, {{0, layout.image_bytes, layout.alignment}}}}},
         .metadata_budget_bytes = 16 * 1024 * 1024});
    auto& states = directory.states();
    store::ContextStorage physical(backing.span(), {}, {});
    store::ContextMigration executor(device, physical);
    const auto source = states.reserve_destination().value();
    std::vector<std::byte> words(storage.image_bytes());
    for (std::size_t i = 0; i < words.size(); ++i) words[i] = std::byte((i * 29 + 13) & 0xff);
    CUDA_CHECK(cudaMemcpy(storage.image(states.physical_offset(source)).data, words.data(),
                          words.size(), cudaMemcpyHostToDevice));
    states.complete_initialization(source);
    states.freeze(source);
    for (bool clear : {false, true}) {
        const auto destination = states.reserve_destination().value();
        std::vector<std::byte> expected(words.size(), std::byte{0x55});
        const auto image = storage.image(states.physical_offset(destination));
        CUDA_CHECK(
            cudaMemcpy(image.data, expected.data(), expected.size(), cudaMemcpyHostToDevice));
        store::ContextTransaction migration(directory);
        migration.own_state(destination);
        if (clear) {
            migration.clear_state();
            std::fill(expected.begin(), expected.end(), std::byte{0});
        } else {
            migration.add_state_group(source, group.offset, group.bytes);
            std::copy_n(words.begin() + group.offset, group.bytes, expected.begin() + group.offset);
        }
        migration.prepare_actions();
        executor.submit(migration.actions());
        migration.mark_submitted();
        while (!executor.ready()) std::this_thread::yield();
        migration.mark_completed();
        migration.publish();
        executor.finish();
        std::vector<std::byte> observed(words.size());
        CUDA_CHECK(
            cudaMemcpy(observed.data(), image.data, observed.size(), cudaMemcpyDeviceToHost));
        expect(observed == expected,
               "migration changed bytes outside its State field group or clear extent");
    }
    expect(states.release(source) && states.occupied() == 0,
           "State field migration retained a reservation");
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    try {
        ninfer::DeviceContext device(0);
        test_release_set_equality();
        test_rank_local_placement();
        test_state_variants_and_metadata();
        test_metadata_admission();
        test_mixed_context_demand();
        test_physical_resource_accounting();
        test_state_store(device);
        test_migration_state_fields(device);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
