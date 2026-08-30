#include "core/device.h"
#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"

#include <ninfer/targets/qwen3_6/state_image.h>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <new>
#include <string_view>
#include <vector>

namespace {

namespace q36   = ninfer::targets::qwen3_6;
namespace store = ninfer::targets::qwen3_6::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::vector<std::int32_t> read_block_table(const ninfer::KVExecutionTablePool& tables,
                                           std::int32_t row, std::size_t count) {
    const ninfer::Tensor source = tables.matrix().slice(1, row, 1).view(
        {static_cast<std::int32_t>(tables.logical_page_capacity())});
    std::vector<std::int32_t> values(count);
    CUDA_CHECK(cudaMemcpy(values.data(), source.data, values.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return values;
}

void test_state_store(ninfer::DeviceContext& device) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 1,
                .conv_channels  = 8,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 4,
                .slot_count     = 4,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 8,
        .dflash_local =
            q36::DFlashLocalStateSpec{.layers = 1, .capacity = 8, .kv_heads = 2, .head_dim = 4},
    };
    ninfer::LayoutBuilder builder;
    const q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    ninfer::DeviceArena arena(builder.finish(256));
    q36::StateImageDevicePool physical({arena.base(), arena.capacity()}, layout);
    q36::HostStatePool host(layout.host, 2);
    store::StateImageStore images(
        physical, &host, static_cast<std::uint32_t>(physical.slot_count()) + host.capacity());

    const auto source = images.reserve_reset(device.stream);
    expect(source.has_value(), "state source allocation");
    const std::int32_t original_slot = images.physical_slot(*source);
    images.freeze(*source);
    images.move_checkpoint_to_active(*source);
    expect(images.physical_slot(*source) == original_slot &&
               images.role(*source) == store::StateImageRole::ActiveMutable,
           "private Move preserves the logical image and physical slot");
    images.freeze(*source);
    const auto destination = images.reserve_destination();
    expect(destination.has_value(), "state destination reservation");
    const store::StateImageSelectors selectors = images.begin_fork(*source, *destination);
    expect(selectors.source >= 0 && selectors.destination >= 0 &&
               selectors.source != selectors.destination,
           "fork resolves distinct physical selectors");
    expect(!images.release(*source) && !images.release(*destination),
           "fork pins both logical images");
    images.commit_fork(*source, *destination);
    expect(images.role(*source) == store::StateImageRole::CheckpointImmutable &&
               images.role(*destination) == store::StateImageRole::ActiveMutable,
           "fork commit preserves source and activates destination");
    images.retain_checkpoint_reference(*source);
    const std::uint64_t rewrite_epoch = images.content_epoch(*source);
    images.freeze(*destination);
    const std::uint64_t recycled_epoch = images.recycle_checkpoint_destination(*source);
    const auto rotated_selectors       = images.begin_fork(*destination, *source);
    expect(rotated_selectors.source != rotated_selectors.destination && images.occupied() == 2,
           "rewrite rotation remains within two StateImages");
    images.abort_fork(*destination, *source);
    images.restore_recycled_checkpoint(*source, recycled_epoch);
    images.thaw(*destination);
    expect(images.content_epoch(*source) == rewrite_epoch &&
               images.checkpoint_references(*source) == 1,
           "aborted rewrite rotation restores the prior checkpoint identity");

    images.freeze(*destination);
    (void)images.recycle_checkpoint_destination(*source);
    (void)images.begin_fork(*destination, *source);
    images.commit_fork(*destination, *source);
    images.retain_checkpoint_reference(*destination);
    images.release_checkpoint_reference(*destination);
    expect(images.release(*destination) && images.release(*source) && images.occupied() == 0,
           "rotated state image ownership closes after release");
    expect(!images.valid(*source), "released state generation becomes stale");

    const auto reused = images.reserve_reset(device.stream);
    expect(reused.has_value() && *reused != *source, "state descriptor reuse advances generation");
    expect(images.release(*reused), "reused state image releases");

    const auto host_source = images.reserve_reset(device.stream);
    expect(host_source.has_value(), "Host state source allocation");
    device.synchronize();
    images.freeze(*host_source);
    auto d2h = images.begin_device_to_host(*host_source, device.transfer_stream);
    expect(d2h.has_value() &&
               images.residency(*host_source) == store::StateReplicaResidency::DeviceOnly,
           "incomplete State D2H does not publish a Host replica");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    images.publish_transfer(std::move(*d2h), true);
    expect(images.residency(*host_source) == store::StateReplicaResidency::Both,
           "State D2H backup publishes stable Both residency");

    const auto moved_device = images.reserve_logical_destination();
    expect(moved_device.has_value(), "State replica split destination allocation");
    images.split_device_replica_identity(*host_source, *moved_device);
    expect(images.residency(*host_source) == store::StateReplicaResidency::HostOnly &&
               images.residency(*moved_device) == store::StateReplicaResidency::DeviceOnly,
           "State replica identity split keeps old Host content and moves Device ownership");

    const auto fork_one = images.reserve_logical_destination();
    const auto fork_two = images.reserve_logical_destination();
    expect(fork_one && fork_two, "concurrent Host State forks reserve logical destinations");
    auto h2d_one = images.begin_host_fork(*host_source, *fork_one, device.transfer_stream);
    auto h2d_two = images.begin_host_fork(*host_source, *fork_two, device.transfer_stream);
    expect(h2d_one && h2d_two && images.source_pins(*host_source) == 2,
           "immutable Host State source supports multiple fork pins");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    images.publish_transfer(std::move(*h2d_one), true);
    images.publish_transfer(std::move(*h2d_two), true);
    expect(images.source_pins(*host_source) == 0 &&
               images.residency(*fork_one) == store::StateReplicaResidency::DeviceOnly &&
               images.residency(*fork_two) == store::StateReplicaResidency::DeviceOnly,
           "Host State forks publish independent Device destinations");
    expect(images.release(*host_source) && images.release(*moved_device) &&
               images.release(*fork_one) && images.release(*fork_two) && host.occupied() == 0,
           "State Host/Device replica ownership closes without leaked slots");
}

void test_kv_store(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    ninfer::DeviceKVPagePoolSpec page_spec{
        .page_group_count = 8,
        .geometry =
            {
                .page_tokens        = static_cast<std::uint32_t>(ninfer::kPagedKVPageSize),
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}},
            },
    };
    const ninfer::DeviceKVPagePoolLayout page_layout =
        ninfer::plan_device_kv_page_pool(builder, page_spec);
    const ninfer::KVExecutionTableLayout table_layout =
        ninfer::plan_kv_execution_tables(builder, {.logical_page_capacity = 4, .table_rows = 2});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical_pages(backing, page_layout);
    ninfer::KVExecutionTablePool physical_tables(backing, table_layout, physical_pages);
    const ninfer::HostKVPageLayout host_layout =
        ninfer::plan_host_kv_page_layout(physical_pages.geometry());
    const std::array host_layouts{host_layout};
    ninfer::HostKVArena host_arena(host_layout.page_stride * 8, host_layouts);
    store::LogicalKVPageStore pages(physical_pages, physical_pages.capacity_pages() + 8U);
    store::HostKVExtentStore extents(host_arena, 8);
    store::KVAddressSpaceStore addresses(pages, physical_tables, 4, 4);

    const auto address = addresses.create_active(3, 0);
    expect(address.has_value(), "active KV address allocation");
    addresses.materialize_to_tokens(*address, 65, device.stream);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 2 && addresses.committed_frontier(*address) == 0,
           "physical KV append does not publish canonical coverage before commit");
    expect(read_block_table(physical_tables, 0, 2) == std::vector<std::int32_t>({0, 1}),
           "batched KV materialization publishes the exact execution mapping");
    addresses.commit_frontier(*address, 65);
    expect(addresses.mapped_pages(*address) == 2 && addresses.entitlement(*address) == 3 &&
               addresses.committed_frontier(*address) == 65 && addresses.bound_row(*address) == 0,
           "KV address tracks mapped pages, entitlement, frontier, and execution row");
    expect(pages.active_address_references(addresses.logical_page(*address, 0)) == 1 &&
               pages.active_address_references(addresses.logical_page(*address, 1)) == 1,
           "active KV membership is counted on each logical page");
    addresses.materialize_to_tokens(*address, 129, device.stream);
    device.synchronize();
    expect(read_block_table(physical_tables, 0, 3) == std::vector<std::int32_t>({0, 1, 2}),
           "incremental KV materialization preserves the existing mapping prefix");
    addresses.destructive_truncate(*address, 65);
    expect(addresses.mapped_pages(*address) == 2 && addresses.entitlement(*address) == 3 &&
               addresses.committed_frontier(*address) == 65,
           "same-frontier trim releases uncommitted materialized suffix pages");
    addresses.set_checkpoint_requirement(*address, 65);
    expect(pages.occupied() == 2 && addresses.mapped_pages(*address) == 2,
           "endpoint and rewrite requirements share one ordered page mapping");
    const std::uint64_t old_epoch = addresses.content_epoch(*address, 0);

    addresses.deactivate(*address);
    expect(addresses.bound_row(*address) == -1 && addresses.entitlement(*address) == 2,
           "catalogued KV address owns no execution row or growth reservation");

    const std::array logical_pages{addresses.logical_page(*address, 0),
                                   addresses.logical_page(*address, 1)};
    expect(pages.active_address_references(logical_pages[0]) == 0 &&
               pages.active_address_references(logical_pages[1]) == 0 &&
               !addresses.has_active_reference(logical_pages[0]),
           "KV deactivation clears logical-page active references");
    addresses.set_checkpoint_requirement(*address, 32);
    expect(pages.protected_columns(logical_pages[0]) == 32 &&
               pages.protected_columns(logical_pages[1]) == 0,
           "lowered checkpoint frontier releases stale suffix protection");
    addresses.set_checkpoint_requirement(*address, 65);
    auto host_backup = extents.prepare(pages, logical_pages);
    expect(host_backup.has_value(), "Host KV extent reservation");
    const std::vector<ninfer::DeviceKVPageHandle> sources = extents.device_sources(*host_backup);
    physical_pages.copy_to_host(sources, extents.writable_view(*host_backup),
                                device.transfer_stream);
    expect(!pages.host_resident(logical_pages[0]) && !pages.host_resident(logical_pages[1]),
           "incomplete KV D2H does not publish Host replicas");
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    auto host_extent = extents.publish(std::move(*host_backup));
    expect(pages.host_resident(logical_pages[0]) && pages.host_resident(logical_pages[1]),
           "KV extent publication attaches every logical Host replica");
    const std::array first_host_release{logical_pages[0]};
    expect(extents.release_page_replicas(pages, first_host_release),
           "first Host page release transaction");
    const auto retained_host_extent = pages.host_replica(logical_pages[1]).extent;
    expect(!extents.valid(host_extent) && !pages.host_resident(logical_pages[0]) &&
               extents.valid(retained_host_extent) &&
               pages.host_replica(logical_pages[1]).page_offset == 0,
           "partial Host release partitions an extent and republishes the retained run");
    const std::array second_host_release{logical_pages[1]};
    expect(extents.release_page_replicas(pages, second_host_release),
           "second Host page release transaction");
    expect(!pages.host_resident(logical_pages[1]) && host_arena.occupied_bytes() == 0,
           "partitioned Host replicas release while Device replicas survive");

    auto second_host_backup = extents.prepare(pages, logical_pages);
    expect(second_host_backup.has_value(), "second Host KV extent reservation");
    const std::vector<ninfer::DeviceKVPageHandle> second_sources =
        extents.device_sources(*second_host_backup);
    physical_pages.copy_to_host(second_sources, extents.writable_view(*second_host_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const auto second_host_extent = extents.publish(std::move(*second_host_backup));
    expect(pages.drop_device_replica(logical_pages[0]) &&
               pages.drop_device_replica(logical_pages[1]) && !extents.release(second_host_extent),
           "Host-only KV pages remain valid and cannot lose their last replica");
    const std::array last_reference_release{store::HostKVPageReplicaRelease{
        .pages = &pages,
        .page  = logical_pages[1],
    }};
    const std::array seven_page_request{
        ninfer::HostKVAllocationRequest{.layout = &host_layout, .pages = 7}};
    const std::array eight_page_request{
        ninfer::HostKVAllocationRequest{.layout = &host_layout, .pages = 8}};
    expect(
        extents.can_allocate_after_page_releases({}, last_reference_release, seven_page_request) &&
            !extents.can_allocate_after_page_releases({}, last_reference_release,
                                                      eight_page_request),
        "last address-reference release contributes exact Host allocation credit");
    auto restore_reservation = physical_pages.reserve(2);
    expect(restore_reservation.has_value(), "KV Host restore Device reservation");
    const std::array restore_destinations{
        pages.reserve_device_replica(logical_pages[0], *restore_reservation),
        pages.reserve_device_replica(logical_pages[1], *restore_reservation)};
    physical_pages.copy_from_host(extents.view(second_host_extent), restore_destinations,
                                  device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    pages.publish_device_replica(logical_pages[0]);
    pages.publish_device_replica(logical_pages[1]);
    expect(extents.release(second_host_extent) && host_arena.occupied_bytes() == 0,
           "KV Host restore republishes Device replicas before releasing the extent");
    auto activation = addresses.prepare_activation(*address, 3, 1);
    expect(addresses.bound_row(*address) == -1 && addresses.entitlement(*address) == 2 &&
               physical_pages.reserved_pages() == 1,
           "prepared KV activation preserves the catalogued mapping until publication");
    addresses.commit_activation(std::move(activation), device.stream);
    expect(pages.active_address_references(logical_pages[0]) == 1 &&
               pages.active_address_references(logical_pages[1]) == 1,
           "KV reactivation republishes logical-page active references");
    addresses.destructive_truncate(*address, 32);
    expect(addresses.mapped_pages(*address) == 1 && addresses.entitlement(*address) == 3 &&
               addresses.committed_frontier(*address) == 32 &&
               addresses.content_epoch(*address, 0) != old_epoch,
           "destructive rewrite truncates coverage and advances content epoch");
    addresses.deactivate(*address);
    const std::array final_logical_page{addresses.logical_page(*address, 0)};
    auto final_host_backup = extents.prepare(pages, final_logical_page);
    expect(final_host_backup.has_value(), "final Host KV backup reservation");
    const auto final_sources = extents.device_sources(*final_host_backup);
    physical_pages.copy_to_host(final_sources, extents.writable_view(*final_host_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    (void)extents.publish(std::move(*final_host_backup));
    expect(addresses.release(*address), "catalogued KV address releases");
    expect(extents.release_unreferenced() == host_layout.page_stride &&
               host_arena.occupied_bytes() == 0,
           "address teardown reclaims its zero-reference Host extent");
    expect(!addresses.valid(*address) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "KV release invalidates generations and closes physical ownership");

    const auto snapshot_source      = addresses.create_active(3, 0);
    const auto snapshot_destination = addresses.create_inactive();
    expect(snapshot_source && snapshot_destination, "active KV snapshot endpoints allocate");
    addresses.materialize_to_tokens(*snapshot_source, 65, device.stream);
    addresses.commit_frontier(*snapshot_source, 65);
    const auto snapshot_full = addresses.logical_page(*snapshot_source, 0);
    const auto snapshot_tail = addresses.logical_page(*snapshot_source, 1);
    auto snapshot = addresses.prepare_active_snapshot(*snapshot_source, *snapshot_destination, 65);
    physical_pages.copy_page(addresses.active_snapshot_tail_source(snapshot),
                             addresses.active_snapshot_tail_destination(snapshot),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_active_snapshot(std::move(snapshot), device.stream);
    const auto snapshot_destination_tail = addresses.logical_page(*snapshot_destination, 1);
    expect(addresses.active(*snapshot_destination) && !addresses.active(*snapshot_source) &&
               pages.active_address_references(snapshot_full) == 1 &&
               pages.active_address_references(snapshot_tail) == 0 &&
               pages.active_address_references(snapshot_destination_tail) == 1,
           "active KV snapshot transfers active ownership without double-counting shared pages");
    addresses.deactivate(*snapshot_destination);
    expect(pages.active_address_references(snapshot_full) == 0 &&
               pages.active_address_references(snapshot_destination_tail) == 0 &&
               addresses.release(*snapshot_destination) && addresses.release(*snapshot_source) &&
               pages.occupied() == 0,
           "active KV snapshot references close with both address spaces");

    const auto alternating = addresses.create_active(4, 0);
    expect(alternating.has_value(), "alternating Host release address allocation");
    addresses.materialize_to_tokens(*alternating, 193, device.stream);
    addresses.commit_frontier(*alternating, 193);
    addresses.deactivate(*alternating);
    const std::array alternating_pages{
        addresses.logical_page(*alternating, 0), addresses.logical_page(*alternating, 1),
        addresses.logical_page(*alternating, 2), addresses.logical_page(*alternating, 3)};
    auto alternating_backup = extents.prepare(pages, alternating_pages);
    expect(alternating_backup.has_value(), "alternating Host extent reservation");
    const auto alternating_sources = extents.device_sources(*alternating_backup);
    physical_pages.copy_to_host(alternating_sources, extents.writable_view(*alternating_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const auto alternating_extent = extents.publish(std::move(*alternating_backup));
    const std::array alternating_release{alternating_pages[0], alternating_pages[2]};
    expect(extents.release_page_replicas(pages, alternating_release),
           "alternating Host page release transaction");
    expect(!extents.valid(alternating_extent) && !pages.host_resident(alternating_pages[0]) &&
               pages.host_resident(alternating_pages[1]) &&
               !pages.host_resident(alternating_pages[2]) &&
               pages.host_resident(alternating_pages[3]) &&
               pages.host_replica(alternating_pages[1]).extent !=
                   pages.host_replica(alternating_pages[3]).extent &&
               host_arena.occupied_bytes() == 2U * host_layout.page_stride,
           "one batch partitions alternating Host release and retained runs exactly once");
    const std::array alternating_retained{alternating_pages[1], alternating_pages[3]};
    expect(extents.release_page_replicas(pages, alternating_retained),
           "retained Host run release transaction");
    expect(host_arena.occupied_bytes() == 0 && addresses.release(*alternating) &&
               pages.occupied() == 0,
           "alternating Host extent partitions close without leaked descriptors");

    const auto shared = addresses.create_active(3, 0);
    expect(shared.has_value(), "shared-prefix source address allocation");
    addresses.materialize_to_tokens(*shared, 65, device.stream);
    addresses.commit_frontier(*shared, 65);
    addresses.set_checkpoint_requirement(*shared, 65);
    addresses.deactivate(*shared);
    const auto shared_full = addresses.logical_page(*shared, 0);
    const auto shared_tail = addresses.logical_page(*shared, 1);

    const auto branch_one = addresses.create_inactive();
    expect(branch_one.has_value(), "first shared-prefix branch address allocation");
    auto first_fork = addresses.prepare_prefix_fork(*shared, *branch_one, 65, 3, 1);
    expect(first_fork.needs_tail_copy() && pages.address_references(shared_full) == 1,
           "prefix fork remains unpublished while its tail copy is pending");
    physical_pages.copy_page(addresses.prefix_fork_tail_source(first_fork),
                             addresses.prefix_fork_tail_destination(first_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(first_fork), device.stream);
    const auto branch_one_tail = addresses.logical_page(*branch_one, 1);
    expect(addresses.logical_page(*branch_one, 0) == shared_full &&
               branch_one_tail != shared_tail && pages.address_references(shared_full) == 2 &&
               pages.address_references(shared_tail) == 1,
           "shared branch references full pages and publishes a private partial tail");
    addresses.materialize_to_tokens(*branch_one, 66, device.stream);
    addresses.commit_frontier(*branch_one, 66);
    expect(pages.committed_columns(shared_tail) == 1 &&
               pages.committed_columns(branch_one_tail) == 2,
           "private branch writes do not extend the shared partial tail");
    addresses.deactivate(*branch_one);
    addresses.destructive_truncate_inactive(*branch_one, 65);
    expect(addresses.committed_frontier(*branch_one) == 65 &&
               pages.committed_columns(branch_one_tail) == 1 &&
               pages.address_references(shared_full) == 2,
           "private suffix truncation retains shared full-prefix pages");

    const auto branch_two = addresses.create_inactive();
    expect(branch_two.has_value(), "second shared-prefix branch address allocation");
    auto second_fork = addresses.prepare_prefix_fork(*shared, *branch_two, 65, 3, 1);
    physical_pages.copy_page(addresses.prefix_fork_tail_source(second_fork),
                             addresses.prefix_fork_tail_destination(second_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(second_fork), device.stream);
    const auto branch_two_tail = addresses.logical_page(*branch_two, 1);
    expect(branch_two_tail != shared_tail && branch_two_tail != branch_one_tail &&
               pages.address_references(shared_full) == 3,
           "independent shared branches own distinct partial tails and one shared full page");
    addresses.deactivate(*branch_two);
    expect(addresses.release(*branch_one) && pages.address_references(shared_full) == 2 &&
               addresses.release(*branch_two) && pages.address_references(shared_full) == 1 &&
               addresses.release(*shared) && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0,
           "shared full-page occupancy survives until its final address reference releases");

    const auto mixed_source = addresses.create_active(4, 0);
    expect(mixed_source.has_value(), "mixed snapshot retained-prefix source allocation");
    addresses.materialize_to_tokens(*mixed_source, 65, device.stream);
    addresses.commit_frontier(*mixed_source, 65);
    addresses.set_checkpoint_requirement(*mixed_source, 65);
    addresses.deactivate(*mixed_source);
    const auto mixed_shared_full = addresses.logical_page(*mixed_source, 0);

    const auto mixed_active = addresses.create_inactive();
    expect(mixed_active.has_value(), "mixed snapshot active branch allocation");
    auto mixed_fork = addresses.prepare_prefix_fork(*mixed_source, *mixed_active, 65, 4, 0);
    physical_pages.copy_page(addresses.prefix_fork_tail_source(mixed_fork),
                             addresses.prefix_fork_tail_destination(mixed_fork),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_prefix_fork(std::move(mixed_fork), device.stream);
    addresses.materialize_to_tokens(*mixed_active, 130, device.stream);
    addresses.commit_frontier(*mixed_active, 130);
    const auto mixed_mutable_full = addresses.logical_page(*mixed_active, 1);
    const auto mixed_tail         = addresses.logical_page(*mixed_active, 2);
    const store::KVActiveSnapshotShape first_shape =
        addresses.active_snapshot_shape(*mixed_active, 130);
    expect(first_shape.full_pages == 2 && first_shape.immutable_full_pages == 1 &&
               first_shape.unique_full_pages == 1 && first_shape.tail_columns == 2 &&
               first_shape.copied_pages() == 1,
           "mixed snapshot analysis distinguishes aliased prefix, mutable suffix, and tail");

    const auto aborted_destination = addresses.create_inactive();
    expect(aborted_destination.has_value(), "mixed snapshot abort destination allocation");
    {
        auto aborted = addresses.prepare_active_snapshot(*mixed_active, *aborted_destination, 130);
        expect(pages.source_pins(mixed_shared_full) == 1 &&
                   pages.source_pins(mixed_mutable_full) == 1 && pages.source_pins(mixed_tail) == 1,
               "mixed snapshot preparation pins every source ownership class");
        addresses.abort_active_snapshot(aborted);
    }
    expect(addresses.active(*mixed_active) && pages.source_pins(mixed_shared_full) == 0 &&
               pages.source_pins(mixed_mutable_full) == 0 && pages.source_pins(mixed_tail) == 0 &&
               pages.writer_references(mixed_shared_full) == 0 &&
               pages.writer_references(mixed_mutable_full) == 1 &&
               pages.writer_references(mixed_tail) == 1 && addresses.release(*aborted_destination),
           "aborted mixed snapshot restores pins without changing page ownership");

    const auto first_snapshot_active = addresses.create_inactive();
    expect(first_snapshot_active.has_value(), "mixed snapshot destination allocation");
    auto first_mixed_snapshot =
        addresses.prepare_active_snapshot(*mixed_active, *first_snapshot_active, 130);
    physical_pages.copy_page(addresses.active_snapshot_tail_source(first_mixed_snapshot),
                             addresses.active_snapshot_tail_destination(first_mixed_snapshot),
                             device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    addresses.commit_active_snapshot(std::move(first_mixed_snapshot), device.stream);
    const auto first_snapshot_tail = addresses.logical_page(*first_snapshot_active, 2);
    expect(
        !addresses.active(*mixed_active) && addresses.active(*first_snapshot_active) &&
            addresses.logical_page(*first_snapshot_active, 0) == mixed_shared_full &&
            addresses.logical_page(*first_snapshot_active, 1) == mixed_mutable_full &&
            first_snapshot_tail != mixed_tail && pages.address_references(mixed_shared_full) == 3 &&
            pages.address_references(mixed_mutable_full) == 2 &&
            pages.writer_references(mixed_shared_full) == 0 &&
            pages.writer_references(mixed_mutable_full) == 0 &&
            pages.writer_references(mixed_tail) == 0 &&
            pages.writer_references(first_snapshot_tail) == 1,
        "mixed snapshot reuses immutable full pages, freezes mutable full pages, and copies tail");

    addresses.materialize_to_tokens(*first_snapshot_active, 192, device.stream);
    addresses.commit_frontier(*first_snapshot_active, 192);
    const store::KVActiveSnapshotShape aligned_shape =
        addresses.active_snapshot_shape(*first_snapshot_active, 192);
    expect(aligned_shape.full_pages == 3 && aligned_shape.immutable_full_pages == 2 &&
               aligned_shape.unique_full_pages == 1 && aligned_shape.tail_columns == 0 &&
               aligned_shape.copied_pages() == 0,
           "repeated aligned snapshot analysis preserves the immutable-to-mutable boundary");
    const auto aligned_active = addresses.create_inactive();
    expect(aligned_active.has_value(), "aligned mixed snapshot destination allocation");
    auto aligned_snapshot =
        addresses.prepare_active_snapshot(*first_snapshot_active, *aligned_active, 192);
    expect(!aligned_snapshot.needs_tail_copy(), "aligned mixed snapshot requires no KV copy");
    addresses.commit_active_snapshot(std::move(aligned_snapshot), device.stream);
    addresses.materialize_to_tokens(*aligned_active, 193, device.stream);
    addresses.commit_frontier(*aligned_active, 193);
    expect(addresses.active(*aligned_active) && addresses.mapped_pages(*aligned_active) == 4 &&
               pages.writer_references(addresses.logical_page(*aligned_active, 3)) == 1,
           "aligned snapshot continuation materializes a new mutable suffix page");

    addresses.deactivate(*aligned_active);
    expect(addresses.release(*aligned_active) && addresses.release(*first_snapshot_active) &&
               addresses.release(*mixed_active) && addresses.release(*mixed_source) &&
               pages.occupied() == 0 && physical_pages.allocated_pages() == 0,
           "repeated mixed snapshot ownership closes without leaked logical or physical pages");

    const auto filler = addresses.create_active(4, 0);
    expect(filler.has_value(), "full-capacity staged-fork filler allocation");
    addresses.materialize_to_tokens(*filler, 193, device.stream);
    addresses.commit_frontier(*filler, 193);
    addresses.set_checkpoint_requirement(*filler, 193);
    addresses.deactivate(*filler);

    const auto retained = addresses.create_active(4, 0);
    expect(retained.has_value(), "full-capacity retained source allocation");
    addresses.materialize_to_tokens(*retained, 65, device.stream);
    addresses.commit_frontier(*retained, 65);
    addresses.set_checkpoint_requirement(*retained, 65);
    addresses.deactivate(*retained);
    const auto staged_destination = addresses.create_inactive();
    expect(staged_destination.has_value(), "full-capacity staged-fork destination allocation");
    expect(physical_pages.allocated_pages() == 6 && physical_pages.available_pages() == 2,
           "full-capacity staged-fork fixture leaves only the transient tail headroom");

    bool ordinary_fork_rejected = false;
    try {
        auto ordinary = addresses.prepare_prefix_fork(*retained, *staged_destination, 65, 4, 1);
        (void)ordinary;
    } catch (const std::bad_alloc&) { ordinary_fork_rejected = true; }
    expect(ordinary_fork_rejected,
           "ordinary retained prefix fork cannot reserve the one-page-over capacity shape");

    auto staged = addresses.prepare_prefix_fork(*retained, *staged_destination, 65, 4, 1, true);
    const auto retained_tail = addresses.prefix_fork_tail_logical_source(staged);
    physical_pages.copy_page(addresses.prefix_fork_tail_source(staged),
                             addresses.prefix_fork_tail_destination(staged),
                             device.transfer_stream);
    const std::array retained_tail_membership{retained_tail};
    auto retained_tail_backup = extents.prepare(pages, retained_tail_membership);
    expect(retained_tail_backup.has_value(), "retained tail Host backup reservation");
    const auto retained_tail_sources = extents.device_sources(*retained_tail_backup);
    physical_pages.copy_to_host(retained_tail_sources, extents.writable_view(*retained_tail_backup),
                                device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    (void)extents.publish(std::move(*retained_tail_backup));
    addresses.settle_prefix_fork_tail_source(staged);
    expect(pages.drop_device_replica(retained_tail),
           "retained tail Device replica releases after both copies complete");
    addresses.complete_prefix_fork_after_tail_release(staged);
    addresses.commit_prefix_fork(std::move(staged), device.stream);
    expect(!pages.device_resident(retained_tail) && pages.host_resident(retained_tail) &&
               addresses.entitlement(*staged_destination) == 4 &&
               physical_pages.allocated_pages() == 6 && physical_pages.reserved_pages() == 2,
           "staged retained fork transfers the old tail capacity to the active entitlement");

    addresses.deactivate(*staged_destination);
    expect(addresses.release(*staged_destination) && addresses.release(*retained) &&
               addresses.release(*filler) &&
               extents.release_unreferenced() == host_layout.page_stride && pages.occupied() == 0 &&
               physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "staged retained fork closes Device and Host ownership without leaks");
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
        test_kv_store(device);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
