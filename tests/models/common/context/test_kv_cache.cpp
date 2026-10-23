#include "core/buffer.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "models/common/context/storage.h"
#include "models/common/context/transfer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
namespace context = ninfer::models::context;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Function>
void expect_rejected(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::logic_error&) { return; }
    throw std::runtime_error(message);
}

KVPageGeometry mixed_geometry() {
    return {
        .planes = {
            {DType::U8, 128, 2, 256},
            {DType::U8, 128, 2, 256},
            {DType::U8, 16, 2, 256},
            {DType::U8, 16, 2, 256},
            {.dtype = DType::BF16, .leading_extent = 128, .head_extent = 1, .tokens_per_row = 4}}};
}

void check_record_geometry() {
    const auto layout = plan_kv_page_layout(mixed_geometry());
    constexpr std::array<std::size_t, 5> offsets{0, 16384, 32768, 34816, 36864};
    constexpr std::array<std::size_t, 5> bytes{16384, 16384, 2048, 2048, 4096};
    constexpr std::array<std::size_t, 5> head_bytes{8192, 8192, 1024, 1024, 4096};
    constexpr std::array<std::array<std::int32_t, 4>, 5> shapes{
        {{128, 64, 2, 1}, {128, 64, 2, 1}, {16, 64, 2, 1}, {16, 64, 2, 1}, {128, 16, 1, 1}}};
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        expect(layout.planes[i].tensor.shape == shapes[i], "Packed KV record shape differs");
        expect(layout.planes[i].tensor.region.offset == offsets[i],
               "Packed KV record plane offset differs");
        expect(layout.planes[i].tensor.region.bytes == bytes[i],
               "Packed KV record plane span differs");
        expect(layout.planes[i].head_stride_bytes == head_bytes[i],
               "Packed KV record head stride differs");
    }
    expect(layout.page_bytes == 40960 && layout.payload_bytes() == 40960,
           "Packed KV record byte count differs");
    auto aligned                    = mixed_geometry();
    aligned.planes.back().alignment = 65536;
    const auto padded               = plan_kv_page_layout(aligned);
    expect(padded.planes.back().tensor.region.offset == 65536 && padded.page_bytes == 131072 &&
               padded.payload_bytes() == 40960,
           "Packed KV record must preserve plane alignment on both Device and Host");
    for (const std::uint32_t span : {0U, 2U, 256U}) {
        auto invalid                         = mixed_geometry();
        invalid.planes.back().tokens_per_row = span;
        expect_rejected([&] { (void)plan_kv_page_layout(invalid); },
                        "Unsupported KV row span was accepted");
    }
    auto invalid        = mixed_geometry();
    invalid.page_tokens = 63;
    expect_rejected([&] { (void)plan_kv_page_layout(invalid); },
                    "Non-divisible KV page span was accepted");
}

context::ContextRange device_range(std::size_t offset, std::size_t bytes) {
    return {context::ContextMemory::Device, static_cast<std::int64_t>(offset), bytes};
}

context::ContextRange host_range(std::size_t offset, std::size_t bytes) {
    return {context::ContextMemory::Host, static_cast<std::int64_t>(offset), bytes};
}

context::RecordCopy page_copy(context::ContextRange source, context::ContextRange destination) {
    return {source, destination, {runtime::KvPoolId{2}}, 0, 1};
}

void check_action_batches() {
    context::ContextActions plan;
    for (std::size_t i = 0; i < context::kContextActionBatchOperations + 3; ++i) {
        plan.operations.emplace_back(page_copy(device_range((1ULL << 34) + i * 256, 128),
                                               host_range((1ULL << 35) + i * 256, 128)));
    }
    context::MappingUpdate first{2, 3, 5, {}};
    for (std::size_t i = 0; i < context::kContextActionBatchOffsets + 17; ++i) {
        first.offsets.push_back((1LL << 34) + i * 1024);
    }
    plan.operations.emplace_back(first);
    plan.operations.emplace_back(context::RecordClear{device_range(1024, 512)});
    plan.operations.emplace_back(
        context::MappingUpdate{1, 0, 3, std::vector<std::int64_t>(4090, -1)});
    plan.operations.emplace_back(page_copy(device_range(64, 32), device_range(128, 32)));
    context::ContextActionCursor cursor;
    std::size_t expected_operation = 0, expected_mapping = 0;
    while (!cursor.complete(plan)) {
        const auto batch = context::take_action_batch(plan, cursor);
        expect(!batch.empty() && batch.operations.size() <= context::kContextActionBatchOperations,
               "Context batch did not respect its operation limit or make progress");
        std::size_t mapping_count = 0;
        for (const auto& action : batch.operations) {
            const auto& original = plan.operations.at(expected_operation);
            expect(action.index() == original.index(), "Context batching changed operation order");
            if (const auto* mapping = std::get_if<context::MappingUpdate>(&action)) {
                const auto& full = std::get<context::MappingUpdate>(original);
                expect(mapping->pool_id == full.pool_id && mapping->row == full.row &&
                           mapping->logical_begin == full.logical_begin + expected_mapping &&
                           std::equal(mapping->offsets.begin(), mapping->offsets.end(),
                                      full.offsets.begin() + expected_mapping),
                       "Context batching changed a mapping value or logical column");
                expected_mapping += mapping->offsets.size();
                mapping_count += mapping->offsets.size();
                if (expected_mapping == full.offsets.size()) {
                    ++expected_operation;
                    expected_mapping = 0;
                }
            } else if (const auto* copy = std::get_if<context::RecordCopy>(&action)) {
                const auto& full = std::get<context::RecordCopy>(original);
                expect(copy->source == full.source && copy->destination == full.destination &&
                           copy->resource == full.resource &&
                           copy->image_count == full.image_count &&
                           copy->page_groups == full.page_groups,
                       "Context batching changed a copy descriptor");
                ++expected_operation;
            } else {
                expect(std::get<context::RecordClear>(action).destination ==
                           std::get<context::RecordClear>(original).destination,
                       "Context batching changed a clear descriptor");
                ++expected_operation;
            }
        }
        expect(mapping_count <= context::kContextActionBatchOffsets,
               "Context batch exceeded its mapping-offset limit");
    }
    expect(expected_operation == plan.operations.size() && expected_mapping == 0 &&
               context::take_action_batch(plan, cursor).empty(),
           "Context batching lost an action or repeated a completed plan");
}

void check_record_execution(DeviceContext& device) {
    const auto record = plan_kv_page_layout(mixed_geometry());
    const auto bytes  = record.page_bytes;
    DeviceBuffer arena(device.device(), 12 * bytes);
    PinnedHostBuffer host(8 * bytes);
    LayoutBuilder builder;
    const auto table_layout =
        plan_kv_execution_tables(builder, {.logical_page_capacity = 4, .table_rows = 2});
    DeviceBuffer table_backing(device.device(), builder.finish(256));
    context::ExecutionTableStorage tables(table_backing.span(), table_layout, arena.span(), bytes);
    const std::array table_views{context::ContextTableStorage{2, &tables}};
    context::ContextStorage storage(arena.span(), host.span(), table_views);
    context::ContextMigration migration(device, storage);

    std::vector<std::byte> original(arena.size());
    for (std::size_t i = 0; i < original.size(); ++i) {
        original[i] = std::byte((i * 29U + (i >> 9U) + 17U) & 255U);
    }
    arena.copy_from_host(original.data(), original.size());
    std::memset(host.data(), 0xa5, host.size());
    table_backing.fill(0xff);

    const std::array<std::size_t, 3> source_offsets{256, 3 * bytes + 256, 7 * bytes + 256};
    const std::array<std::size_t, 3> host_offsets{256, 2 * bytes + 512, 5 * bytes + 768};
    context::ContextActions export_records;
    for (std::size_t i = 0; i < source_offsets.size(); ++i) {
        export_records.operations.emplace_back(
            page_copy(device_range(source_offsets[i], bytes), host_range(host_offsets[i], bytes)));
    }
    migration.submit(export_records);
    device.synchronize();
    // Transfer uses its own stream; drain is the explicit completion boundary here.
    migration.drain();
    for (std::size_t i = 0; i < source_offsets.size(); ++i) {
        expect(std::memcmp(static_cast<const std::byte*>(host.data()) + host_offsets[i],
                           original.data() + source_offsets[i], bytes) == 0,
               "Device-to-Host transfer changed packed record bytes");
        expect(static_cast<const std::byte*>(host.data())[host_offsets[i] - 1] == std::byte{0xa5} &&
                   static_cast<const std::byte*>(host.data())[host_offsets[i] + bytes] ==
                       std::byte{0xa5},
               "Record transfer touched bytes outside the destination");
    }

    const std::size_t restored = 9 * bytes;
    const std::size_t copied   = 10 * bytes;
    context::ContextActions actions;
    actions.operations.emplace_back(
        page_copy(host_range(host_offsets[1], bytes), device_range(restored, bytes)));
    actions.operations.emplace_back(
        page_copy(device_range(restored, bytes), device_range(copied, bytes)));
    actions.operations.emplace_back(context::RecordClear{device_range(11 * bytes, bytes)});
    actions.operations.emplace_back(context::MappingUpdate{
        2, 1, 1, {static_cast<std::int64_t>(copied), static_cast<std::int64_t>(restored), -1}});
    const auto observed = storage.describe(actions.operations.front());
    expect(observed.work.payload_bytes == bytes && observed.work.copy_operations == 1 &&
               observed.page_groups == 1 &&
               observed.direction == runtime::ContextTransferDirection::HostToDevice,
           "Transfer observation does not describe the actual record copy");
    migration.submit(actions);
    expect_rejected([&] { migration.submit(actions); }, "An occupied migration slot was reused");
    migration.drain();
    std::vector<std::byte> after(arena.size());
    arena.copy_to_host(after.data(), after.size());
    auto expected = original;
    std::copy_n(original.begin() + source_offsets[1], bytes, expected.begin() + restored);
    std::copy_n(original.begin() + source_offsets[1], bytes, expected.begin() + copied);
    std::fill(expected.begin() + 11 * bytes, expected.end(), std::byte{0});
    expect(after == expected, "Ordered record copies/clear changed neighboring bytes");
    std::array<std::int64_t, 8> mapping{};
    table_backing.copy_to_host(mapping.data(), sizeof(mapping));
    const std::array<std::int64_t, 8> expected_mapping{
        -1, -1, -1, -1, -1, static_cast<std::int64_t>(copied), static_cast<std::int64_t>(restored),
        -1};
    expect(mapping == expected_mapping, "I64 execution table changed logical order or other rows");

    context::ContextActions invalid;
    invalid.operations.emplace_back(context::RecordClear{device_range(0, bytes)});
    invalid.operations.emplace_back(
        context::MappingUpdate{2, 0, 0, {static_cast<std::int64_t>(arena.size())}});
    expect_rejected([&] { migration.submit(invalid); }, "An out-of-range mapping was accepted");
    arena.copy_to_host(after.data(), after.size());
    expect(after == expected && !migration.occupied(),
           "A rejected action batch partially changed physical storage");
    invalid.operations = {page_copy(device_range(0, bytes), device_range(256, bytes))};
    expect_rejected([&] { migration.submit(invalid); }, "Overlapping records were copied");

    context::ContextActions host_maintenance;
    host_maintenance.operations.emplace_back(
        page_copy(host_range(host_offsets[0], bytes), host_range(6 * bytes + 1024, bytes)));
    host_maintenance.operations.emplace_back(
        context::RecordClear{host_range(host_offsets[0], bytes)});
    migration.submit(host_maintenance);
    expect(migration.ready(), "Host maintenance did not complete synchronously");
    const auto host_observations = migration.observations();
    expect(host_observations.size() == 1 &&
               host_observations.front().direction ==
                   runtime::ContextTransferDirection::HostToHost &&
               host_observations.front().elapsed_ns > 0,
           "Host maintenance did not report its CPU copy duration");
    migration.finish();
    expect(std::memcmp(static_cast<const std::byte*>(host.data()) + 6 * bytes + 1024,
                       original.data() + source_offsets[0], bytes) == 0,
           "Host record relocation changed bytes");
    expect(std::all_of(static_cast<const std::byte*>(host.data()) + host_offsets[0],
                       static_cast<const std::byte*>(host.data()) + host_offsets[0] + bytes,
                       [](auto value) { return value == std::byte{0}; }),
           "Host clear did not follow Host copy");
}
} // namespace

int main() {
    try {
        check_record_geometry();
        check_action_batches();
        int devices       = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || !devices) {
            std::cout << "CUDA device unavailable\n";
            return 77;
        }
        CUDA_CHECK(status);
        DeviceContext device;
        check_record_execution(device);
        std::cout << "Packed Context records and execution tables passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Context storage test failed: " << error.what() << '\n';
        return 1;
    }
}
