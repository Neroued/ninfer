#include "core/device.h"
#include "models/qwen3_5/program/storage/host_kv_store.h"
#include "models/qwen3_5/program/storage/kv_store.h"
#include "models/qwen3_5/program/storage/state_store.h"

#include "models/qwen3_5/state/state_image.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <new>
#include <string_view>
#include <vector>

namespace {

namespace q36   = ninfer::models::qwen3_5;
namespace store = ninfer::models::qwen3_5::detail;

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
    expect(images.can_release_source_after_fork_abort(*source, *destination, 0) &&
               images.can_release_destination_after_fork_abort(*source, *destination, 0),
           "fork release preflight accounts for its source and destination pins");
    const auto concurrent_destination = images.reserve_destination();
    expect(concurrent_destination.has_value(), "concurrent fork destination reservation");
    (void)images.begin_fork(*source, *concurrent_destination);
    expect(!images.can_release_source_after_fork_abort(*source, *destination, 0),
           "fork release preflight preserves independent source pins");
    images.abort_fork(*source, *concurrent_destination);
    expect(images.release(*concurrent_destination), "concurrent fork destination release");
    expect(images.can_release_source_after_fork_abort(*source, *destination, 0),
           "fork release preflight accepts the final source pin");
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
    addresses.ensure_mapped_to_tokens(*address, 65, device.stream);
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
    addresses.ensure_mapped_to_tokens(*address, 129, device.stream);
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

    // Verify crosses a page boundary, but the terminal commit consumes only its first column.
    const auto terminal = addresses.create_active(3, 0);
    expect(terminal.has_value(), "terminal boundary KV address allocation");
    addresses.ensure_mapped_to_tokens(*terminal, 63, device.stream);
    addresses.commit_frontier(*terminal, 63);
    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
    device.synchronize();
    const auto terminal_mapping   = read_block_table(physical_tables, 0, 2);
    const auto unchanged_terminal = [&] {
        return addresses.mapped_pages(*terminal) == 2 &&
               addresses.committed_frontier(*terminal) == 63 &&
               addresses.entitlement(*terminal) == 3 && physical_pages.allocated_pages() == 2 &&
               physical_pages.reserved_pages() == 1 && physical_pages.available_pages() == 5 &&
               read_block_table(physical_tables, 0, 2) == terminal_mapping;
    };
    addresses.ensure_mapped_to_tokens(*terminal, 71, device.stream);
    addresses.ensure_mapped_to_tokens(*terminal, 64, device.stream);
    device.synchronize();
    expect(unchanged_terminal(), "covered KV requests preserve speculative mappings and ownership");
    bool exceeded = false;
    try {
        addresses.ensure_mapped_to_tokens(*terminal, 193, device.stream);
    } catch (const std::invalid_argument&) { exceeded = true; }
    expect(exceeded && unchanged_terminal(),
           "KV coverage beyond entitlement fails without mutation");
    addresses.commit_frontier(*terminal, 64);
    addresses.destructive_truncate(*terminal, 64);
    expect(addresses.mapped_pages(*terminal) == 1 &&
               addresses.committed_frontier(*terminal) == 64 &&
               addresses.entitlement(*terminal) == 3 && physical_pages.allocated_pages() == 1 &&
               physical_pages.reserved_pages() == 2 && physical_pages.available_pages() == 5,
           "terminal trim returns the uncommitted page to the same active reservation");
    addresses.deactivate(*terminal);
    expect(addresses.release(*terminal) && physical_pages.allocated_pages() == 0 &&
               physical_pages.reserved_pages() == 0 && physical_pages.available_pages() == 8,
           "terminal settlement releases both mappings and unused growth");

    const auto snapshot_source      = addresses.create_active(3, 0);
    const auto snapshot_destination = addresses.create_inactive();
    expect(snapshot_source && snapshot_destination, "active KV snapshot endpoints allocate");
    addresses.ensure_mapped_to_tokens(*snapshot_source, 65, device.stream);
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
    addresses.ensure_mapped_to_tokens(*alternating, 193, device.stream);
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
    addresses.ensure_mapped_to_tokens(*shared, 65, device.stream);
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
    addresses.ensure_mapped_to_tokens(*branch_one, 66, device.stream);
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
    addresses.ensure_mapped_to_tokens(*mixed_source, 65, device.stream);
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
    addresses.ensure_mapped_to_tokens(*mixed_active, 130, device.stream);
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

    addresses.ensure_mapped_to_tokens(*first_snapshot_active, 192, device.stream);
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
    addresses.ensure_mapped_to_tokens(*aligned_active, 193, device.stream);
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
    addresses.ensure_mapped_to_tokens(*filler, 193, device.stream);
    addresses.commit_frontier(*filler, 193);
    addresses.set_checkpoint_requirement(*filler, 193);
    addresses.deactivate(*filler);

    const auto retained = addresses.create_active(4, 0);
    expect(retained.has_value(), "full-capacity retained source allocation");
    addresses.ensure_mapped_to_tokens(*retained, 65, device.stream);
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

void test_kv_placement(ninfer::DeviceContext& device) {
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
    expect(address.has_value(), "placement KV address allocation");
    addresses.ensure_mapped_to_tokens(*address, 192, device.stream);
    addresses.commit_frontier(*address, 192);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 3, "placement membership covers three pages");

    // Write a distinct pattern into every device page so promote/demote round-trips are
    // observable. Page-major plane [X=8, P=64, H=2, N=8]: page n occupies 1024 half elements
    // at element offset 1024 * n of plane 0.
    const ninfer::Tensor plane = physical_pages.plane(0);
    auto* plane_half          = static_cast<__half*>(plane.data);
    for (std::uint32_t page = 0; page < 3; ++page) {
        std::vector<__half> pattern(1024);
        for (std::size_t element = 0; element < pattern.size(); ++element) {
            pattern[element] = __float2half(
                0.25F * static_cast<float>(page + 1U) +
                0.001F * static_cast<float>(element % 251U));
        }
        CUDA_CHECK(cudaMemcpyAsync(plane_half + 1024ULL * page, pattern.data(),
                                   pattern.size() * sizeof(__half), cudaMemcpyHostToDevice,
                                   device.stream));
    }
    device.synchronize();

    // Establish current Host replicas for every page before any placement change.
    const std::array logical_pages{addresses.logical_page(*address, 0),
                                   addresses.logical_page(*address, 1),
                                   addresses.logical_page(*address, 2)};
    auto backup = extents.prepare(pages, logical_pages, true);
    expect(backup.has_value(), "placement Host backup reservation");
    physical_pages.copy_to_host(extents.device_sources(*backup),
                                extents.writable_view(*backup), device.transfer_stream);
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    (void)extents.publish(std::move(*backup));

    // Corrupt the device copy of page 1 after the backup: a later promote must restore the
    // backed-up bytes, proving the Host replica is the authority for stage-in.
    std::vector<__half> garbage(1024, __float2half(-7.0F));
    CUDA_CHECK(cudaMemcpyAsync(plane_half + 1024ULL, garbage.data(),
                               garbage.size() * sizeof(__half), cudaMemcpyHostToDevice,
                               device.stream));
    device.synchronize();
    const auto initial_table   = read_block_table(physical_tables, 0, 3);
    const std::uint32_t allocated_before = physical_pages.allocated_pages();

    const auto demoted = addresses.apply_device_placement(
        *address, extents, std::array<const std::uint32_t, 2>{0U, 2U}, device.transfer_stream);
    expect(demoted.demoted == 1 && demoted.promoted == 0, "placement demotes the unselected page");
    expect(!pages.device_resident(logical_pages[1]) && pages.host_resident(logical_pages[1]),
           "demoted page keeps only its Host replica");
    expect(pages.device_resident(logical_pages[0]) && pages.device_resident(logical_pages[2]),
           "selected pages stay resident");
    const auto holed = read_block_table(physical_tables, 0, 3);
    expect(holed[0] == initial_table[0] && holed[1] == ninfer::kPagedKVPageHole &&
               holed[2] == initial_table[2],
           "placement republishes holes for demoted pages only");
    expect(physical_pages.allocated_pages() == allocated_before - 1U,
           "demotion releases the outgoing device page");
    expect(addresses.mapped_pages(*address) == 3 &&
               addresses.committed_frontier(*address) == 192,
           "placement changes device residency, not membership or frontier");

    const auto swapped = addresses.apply_device_placement(
        *address, extents, std::array<const std::uint32_t, 1>{1U}, device.transfer_stream);
    expect(swapped.demoted == 2 && swapped.promoted == 1, "reselection swaps the working set");
    const auto swapped_table = read_block_table(physical_tables, 0, 3);
    expect(swapped_table[0] == ninfer::kPagedKVPageHole &&
               swapped_table[2] == ninfer::kPagedKVPageHole && swapped_table[1] >= 0,
           "reselection publishes holes around the promoted page");
    expect(pages.device_resident(logical_pages[1]) && !pages.device_resident(logical_pages[0]) &&
               !pages.device_resident(logical_pages[2]),
           "reselection residency follows the selected set");

    std::vector<__half> restored(1024);
    const std::int64_t promoted_physical = swapped_table[1];
    CUDA_CHECK(cudaMemcpyAsync(restored.data(), plane_half + 1024ULL * promoted_physical,
                               restored.size() * sizeof(__half), cudaMemcpyDeviceToHost,
                               device.stream));
    device.synchronize();
    bool roundtrip = true;
    for (std::size_t element = 0; element < restored.size(); ++element) {
        const __half expected =
            __float2half(0.5F + 0.001F * static_cast<float>(element % 251U));
        if (restored[element] != expected) {
            roundtrip = false;
            break;
        }
    }
    expect(roundtrip, "promoted page restores the backed-up Host bytes over device garbage");

    const auto full = addresses.apply_device_placement(
        *address, extents, std::array<const std::uint32_t, 3>{0U, 1U, 2U}, device.transfer_stream);
    expect(full.promoted == 2 && full.demoted == 0, "full reselection restores every page");
    expect(pages.device_resident(logical_pages[0]) && pages.device_resident(logical_pages[1]) &&
               pages.device_resident(logical_pages[2]),
           "full reselection residency covers the membership");
    const auto full_table = read_block_table(physical_tables, 0, 3);
    expect(full_table[0] >= 0 && full_table[1] >= 0 && full_table[2] >= 0,
           "full reselection publishes no holes");

    addresses.resize_entitlement(*address, 4);
    addresses.ensure_mapped_to_tokens(*address, 256, device.stream);
    addresses.commit_frontier(*address, 256);
    device.synchronize();
    expect(addresses.mapped_pages(*address) == 4, "growth past the working set still maps pages");

    // Sparse activation: shrink the working set to {0, 2}, drop the address, and
    // reactivate with a window-sized entitlement (2 < mapped 4). Only working-set
    // pages must be device-resident; the row republishes holes for Host-only pages
    // and freshly mapped growth pages stay live by actual residency.
    const auto windowed = addresses.apply_device_placement(
        *address, extents, std::array<const std::uint32_t, 2>{0U, 2U}, device.transfer_stream);
    expect(windowed.demoted == 2 && windowed.promoted == 0, "window placement demotes 1 and 3");
    addresses.commit_frontier(*address, 256);
    addresses.deactivate(*address);
    auto sparse_activation = addresses.prepare_activation(*address, 2, 1, 256);
    addresses.commit_activation(std::move(sparse_activation), device.stream);
    device.synchronize();
    expect(addresses.entitlement(*address) == 4 && addresses.bound_row(*address) == 1 &&
               physical_pages.reserved_pages() == 0,
           "sparse activation preserves logical coverage without reserving holes");
    const auto sparse_table = read_block_table(physical_tables, 1, 4);
    expect(sparse_table[0] >= 0 && sparse_table[1] == ninfer::kPagedKVPageHole &&
               sparse_table[2] >= 0 && sparse_table[3] == ninfer::kPagedKVPageHole,
           "sparse activation republishes holes for Host-only pages");
    addresses.deactivate(*address);
    addresses.truncate_inactive_prefix(*address, 64);
    addresses.activate(*address, 3, 0);
    addresses.ensure_mapped_to_tokens(*address, 192, device.stream);
    expect(addresses.device_residency_floor_pages(*address) == 3,
           "inactive truncate followed by growth has no stale or duplicate working-set entries");
    addresses.deactivate(*address);
    expect(addresses.release(*address), "placement address releases");
    expect(physical_pages.allocated_pages() == 0 && physical_pages.reserved_pages() == 0,
           "placement teardown closes physical ownership");
}

void test_sparse_replay_truncate(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    const auto layout = ninfer::plan_device_kv_page_pool(builder, {
        .page_group_count = 6,
        .geometry = {.page_tokens = 64,
                     .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                     .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}}}});
    const auto table_layout = ninfer::plan_kv_execution_tables(
        builder, {.logical_page_capacity = 8, .table_rows = 1});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical(backing, layout);
    ninfer::KVExecutionTablePool tables(backing, table_layout, physical);
    const auto host_layout = ninfer::plan_host_kv_page_layout(physical.geometry());
    const std::array host_layouts{host_layout};
    ninfer::HostKVArena host_arena(host_layout.page_stride * 8, host_layouts);
    store::LogicalKVPageStore pages(physical, 16);
    store::HostKVExtentStore extents(host_arena, 8);
    store::KVAddressSpaceStore addresses(pages, tables, 2, 8);
    addresses.set_sparse_activation_budget(6);
    const auto address = addresses.create_active(6, 0);
    addresses.ensure_mapped_to_tokens(*address, 256, device.stream);
    addresses.commit_frontier(*address, 256);
    device.synchronize();
    addresses.apply_device_placement(*address, extents, std::array{0U, 3U}, device.transfer_stream);
    addresses.apply_device_placement(*address, extents, std::array{0U, 1U, 3U}, device.transfer_stream);
    const auto tail = addresses.logical_page(*address, 1);
    const auto removed = addresses.logical_page(*address, 2);
    expect(pages.host_resident(tail) && pages.device_resident(tail) &&
               !pages.device_resident(removed), "replay fixture has a copied tail and Host-only suffix");
    addresses.truncate_for_replay(*address, 70, extents);
    expect(addresses.mapped_pages(*address) == 2 && addresses.committed_frontier(*address) == 70 &&
               pages.committed_columns(tail) == 6 && !pages.host_resident(tail) &&
               !pages.valid(removed), "replay rewind drops stale Host coverage and suffix ownership");
    addresses.resize_entitlement(*address, 6);
    addresses.ensure_mapped_to_tokens(*address, 256, device.stream);
    addresses.commit_frontier(*address, 256);
    expect(addresses.device_residency_floor_pages(*address) == 4 &&
               pages.committed_columns(tail) == 64, "replayed growth republishes valid coverage");
    addresses.deactivate(*address);
    expect(addresses.release(*address), "replay test releases its address");
    (void)extents.release_unreferenced();
    expect(physical.allocated_pages() == 0 && physical.reserved_pages() == 0 &&
               pages.occupied() == 0, "replay rewind and growth leave no physical or logical leak");
}

void test_sparse_host_credit_shared_aliases(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    const auto layout = ninfer::plan_device_kv_page_pool(builder, {
        .page_group_count = 12,
        .geometry = {.page_tokens = 64,
                     .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                     .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}}}});
    const auto table_layout = ninfer::plan_kv_execution_tables(
        builder, {.logical_page_capacity = 8, .table_rows = 2});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical(backing, layout);
    ninfer::KVExecutionTablePool tables(backing, table_layout, physical);
    const auto host_layout = ninfer::plan_host_kv_page_layout(physical.geometry());
    const std::array host_layouts{host_layout};
    ninfer::HostKVArena host_arena(host_layout.page_stride * 8, host_layouts);
    store::LogicalKVPageStore pages(physical, 20);
    store::HostKVExtentStore extents(host_arena, 8);
    store::KVAddressSpaceStore addresses(pages, tables, 4, 8);
    addresses.set_sparse_activation_budget(6);
    const auto source = addresses.create_active(6, 0);
    addresses.ensure_mapped_to_tokens(*source, 256, device.stream);
    addresses.commit_frontier(*source, 256);
    device.synchronize();
    addresses.apply_device_placement(*source, extents, std::array{0U, 3U}, device.transfer_stream);
    addresses.apply_device_placement(*source, extents, std::array{0U, 1U, 2U, 3U}, device.transfer_stream);
    addresses.deactivate(*source);
    const auto first = addresses.create_inactive();
    auto fork_first = addresses.prepare_prefix_fork(*source, *first, 256, 6, 0);
    addresses.commit_prefix_fork(std::move(fork_first), device.stream);
    const auto second = addresses.create_inactive();
    auto fork_second = addresses.prepare_prefix_fork(*source, *second, 256, 6, 1);
    addresses.commit_prefix_fork(std::move(fork_second), device.stream);
    device.synchronize();
    std::vector<std::uint8_t> seen(pages.capacity());
    std::size_t unique = 0;
    for (const auto address : std::array{*first, *second}) {
        for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
            expect(pages.mark_host_replica_once(addresses.logical_page(address, page), seen, unique),
                   "valid claimed-active Host credit");
        }
    }
    expect(unique == 2 && host_arena.occupied_bytes() == 2 * host_layout.page_stride,
           "two active aliases and inactive source share exactly two actual Host replicas");
    const auto peak = 6 * host_layout.page_stride;
    const std::array two_peaks{peak, peak};
    const auto both = store::sparse_host_budget_occupancy(host_arena.occupied_bytes(),
        unique * host_layout.page_stride, two_peaks);
    expect(both && *both == 2 * peak, "shared actual store replicas credited once, not per address");
    addresses.deactivate(*first);
    expect(addresses.release(*first), "first active alias releases");
    const auto old = addresses.logical_page(*second, 1);
    addresses.deactivate(*second);
    expect(addresses.release(*second), "second active alias releases");
    (void)extents.release_unreferenced();
    const std::array no_peaks{std::size_t{0}};
    const auto retained = store::sparse_host_budget_occupancy(host_arena.occupied_bytes(), 0, no_peaks);
    expect(retained && *retained == 2 * host_layout.page_stride,
           "inactive catalog retains actual replicas after active claims released");
    expect(addresses.release(*source), "inactive catalog source releases");
    (void)extents.release_unreferenced();
    expect(!pages.mark_host_replica_once(old, seen, unique), "stale Host handle fails closed");
    expect(host_arena.occupied_bytes() == 0 && pages.occupied() == 0 &&
               physical.allocated_pages() == 0 && physical.reserved_pages() == 0,
           "shared Host credit test closes all real payload and claims");
}

void test_sparse_shared_reservation(ninfer::DeviceContext& device) {
    ninfer::LayoutBuilder builder;
    const auto layout = ninfer::plan_device_kv_page_pool(builder, {
        .page_group_count = 8,
        .geometry = {.page_tokens = 64,
                     .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                     .planes = {{.dtype = ninfer::DType::BF16, .leading_extent = 8, .head_extent = 2}}}});
    const auto table_layout = ninfer::plan_kv_execution_tables(
        builder, {.logical_page_capacity = 64, .table_rows = 1});
    ninfer::DeviceArena arena(builder.finish(256));
    const ninfer::DeviceSpan backing{arena.base(), arena.capacity()};
    ninfer::DeviceKVPagePool physical(backing, layout);
    ninfer::KVExecutionTablePool tables(backing, table_layout, physical);
    store::LogicalKVPageStore pages(physical, 64);
    store::KVAddressSpaceStore addresses(pages, tables, 4, 64);
    addresses.set_sparse_activation_budget(8);

    auto source = addresses.create_active(8, 0);
    addresses.ensure_mapped_to_tokens(*source, 256, device.stream);
    addresses.commit_frontier(*source, 256);
    addresses.deactivate(*source);
    auto destination = addresses.create_inactive();
    auto fork = addresses.prepare_prefix_fork(*source, *destination, 256, 8, 0);
    addresses.commit_prefix_fork(std::move(fork), device.stream);
    device.synchronize();
    // Four shared pages already consume the physical pool. A large logical request
    // must not add another full window's reservation on top of them.
    addresses.resize_entitlement(*destination, 64);
    expect(physical.allocated_pages() + physical.reserved_pages() == 8,
           "shared prefix and sparse growth together fit one physical window budget");
    addresses.ensure_mapped_to_tokens(*destination, 512, device.stream);
    addresses.commit_frontier(*destination, 512);
    addresses.deactivate(*destination);
    expect(addresses.release(*destination) && addresses.release(*source),
           "sparse shared-prefix test releases both owners");
    expect(physical.allocated_pages() == 0 && physical.reserved_pages() == 0,
           "sparse shared-prefix test leaves no physical claim");
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
        test_kv_placement(device);
        test_sparse_replay_truncate(device);
        test_sparse_shared_reservation(device);
        test_sparse_host_credit_shared_aliases(device);
        device.synchronize();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
