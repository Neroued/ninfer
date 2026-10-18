#include "core/device.h"
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

void test_state_store(ninfer::DeviceContext& device) {
    ninfer::StateImageBuilder schema;
    (void)schema.add(ninfer::DType::BF16, {8, 3}, 1, "conv");
    (void)schema.add(ninfer::DType::FP32, {4, 4, 2}, 1, "recurrent");
    const auto hidden_field = schema.add(ninfer::DType::BF16, {8}, 1, "hidden");
    (void)schema.add(ninfer::DType::BF16, {4, 8, 2}, 1, "local_k");
    (void)schema.add(ninfer::DType::FP16, {4, 8, 2}, 1, "local_v");
    ninfer::LayoutBuilder builder;
    const auto layout = schema.finish(builder, 4);
    ninfer::DeviceBuffer backing(device.device(), builder.finish());
    ninfer::StateImageStorage physical(backing.span(), layout);
    store::HostAllocator host_backing(0, layout.image_stride * 2, layout.image_stride);
    ninfer::PinnedHostBuffer host_payload(layout.image_stride * 2);
    store::StateDirectory images({6, 4, physical.image_bytes()}, &host_backing);
    store::ContextStorage storage(physical, {}, 0, host_payload.span());
    const auto enqueue = [&](const store::StateDirectory::Transfer& transfer) {
        storage.execute(store::ContextAction{transfer.description()}, {}, device.transfer_view());
    };
    const auto hidden_slot = [&](std::int32_t slot) {
        return physical.field(hidden_field).slice(1, slot, 1).view({8});
    };

    const auto initialize = [&] {
        auto handle = images.reserve_destination();
        if (handle) {
            physical.clear(images.physical_slot(*handle), device.execution_view());
            images.complete_initialization(*handle);
        }
        return handle;
    };
    const auto source = initialize();
    expect(source.has_value(), "state source allocation");
    const std::int32_t original_slot = images.physical_slot(*source);
    const auto source_epoch          = images.content_epoch(*source);
    images.freeze(*source);
    expect(images.content_epoch(*source) == source_epoch, "freezing retains content identity");
    images.make_mutable(*source);
    expect(images.physical_slot(*source) == original_slot &&
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
        const auto source_slot = images.physical_slot(*shared);
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
        physical.copy(source_slot, images.physical_slot(*first_reader), device.execution_view());
        device.synchronize();
        images.complete_write(*shared, *first_reader);
        expect(images.valid(*shared) && images.source_pins(*shared) == 1,
               "one completed reader does not release the remaining reader's State");
        physical.copy(source_slot, images.physical_slot(*last_reader), device.execution_view());
        std::array<std::uint16_t, 8> observed{};
        const auto last_hidden = hidden_slot(images.physical_slot(*last_reader));
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
        expect(replacement && *replacement != *shared &&
                   images.physical_slot(*replacement) == source_slot,
               "retired State storage is reusable under a new generation");
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
                   host_backing.occupied_bytes() == 0,
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
        auto allocation     = host_backing.allocate(layout.image_stride);
        const auto occupied = host_backing.occupied_bytes();
        auto cancelled      = images.reserve_device_to_host(*host_source, std::move(*allocation));
        expect(host_backing.occupied_bytes() == occupied && host_backing.allocation_count() == 1,
               "transfer consumes the existing Host reservation without reallocating");
        expect(cancelled && images.host_occupied() == 1 &&
                   host_backing.reserved_bytes() == layout.image_stride &&
                   host_backing.live_bytes() == 0 && images.source_pins(*host_source) == 1 &&
                   !images.release(*host_source),
               "State transfer reservation charges destination while its source is pinned");
    }
    expect(host_backing.occupied_bytes() == 0 && images.host_occupied() == 0 &&
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
               images.host_occupied() == 1 && host_backing.live_bytes() == layout.image_stride &&
               host_backing.reserved_bytes() == 0,
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
    LayoutBuilder backing_layout;
    StateImageBuilder schema;
    (void)schema.add(DType::FP32, {4}, 1, "recurrent");
    const auto local = schema.add(DType::BF16, {8}, 2, "local_kv");
    (void)schema.add(DType::I32, {3}, 1, "tail");
    const auto group  = schema.group(local, local);
    const auto layout = schema.finish(backing_layout, 2);
    DeviceBuffer backing(device.device(), backing_layout.finish());
    StateImageStorage storage(backing.span(), layout);
    store::ContextDirectory directory({.state = {2, 2, storage.image_bytes()}});
    auto& states = directory.states();
    store::ContextStorage physical(storage, {}, 0, {});
    store::ContextMigration executor(device, physical);
    const auto source = states.reserve_destination().value();
    std::vector<std::byte> words(storage.image_bytes());
    for (std::size_t i = 0; i < words.size(); ++i) words[i] = std::byte((i * 29 + 13) & 0xff);
    CUDA_CHECK(cudaMemcpy(storage.image(states.physical_slot(source)).data, words.data(),
                          words.size(), cudaMemcpyHostToDevice));
    states.complete_initialization(source);
    states.freeze(source);
    for (bool clear : {false, true}) {
        const auto destination = states.reserve_destination().value();
        std::vector<std::byte> expected(words.size(), std::byte{0x55});
        const auto image = storage.image(states.physical_slot(destination));
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
