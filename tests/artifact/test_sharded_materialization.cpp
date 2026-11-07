// Multi-device placement of artifact parents.
//
// Without an argument: plan checks, then a one-device upload. A one-device Binder with the
// default resolver must plan exactly the upstream single-device layout (complete parents in
// object order, each at the next aligned offset), and the upload must hold each parent's bytes.
//
// With `--devices 2`: Replicated, Rows, Columns, PrimaryOnly and SingleDevice parents on two
// devices, compared byte for byte with host-applied slices of the parent bytes, per-device
// statistics and views, plus parents larger than one staging slot whose source ranges are read
// by both devices. Skips (77) without two CUDA devices.
//
// Both runs also place two parents HostMapped: one mapped page-locked host copy, outside every
// device capacity, that each device's parent addresses in place.

#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/views.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

using Ranges = std::vector<SliceRange>;

struct Spec {
    std::string id;
    Shape shape;
    std::string format;
    std::string layout;
};

// One parent per layout, split across two files inside the q5 code plane.
const std::array kSpecs = {
    Spec{"bf16", {6, 40}, "bf16", "contiguous_le_v1"},
    Spec{"vector", {10}, "fp32", "contiguous_le_v1"},
    Spec{"q5", {6, 256}, "q5_g64_fp16", "row_split_k128_v1"},
    Spec{"fp8", {6, 40}, "fp8_e4m3fn_row_bf16", "row_scale_v1"},
    Spec{"nvfp4", {256, 128}, "nvfp4", "block_scale_k16_m128x4_v1"},
    Spec{"q8", {4, 200}, "q8_g32_fp16", "row_split_k128_v1"},
    Spec{"shared", {4, 64}, "bf16", "contiguous_le_v1"},
    Spec{"vision", {2, 128}, "bf16", "contiguous_le_v1"},
};
constexpr std::uint64_t kFileSplit = 1001;

// Logical parameters over part of a parent, with their logical shapes.
const std::map<std::string, std::pair<Json, Shape>> kPartBindings = {
    {"bf16_rows", {Json::array({{{"object", "bf16"}, {"range", {80, 240}}}}), {4, 40}}},
    {"vector_pairs", {Json::array({{{"object", "vector"}, {"range", {0, 10}}}}), {5, 2}}},
    {"fp8_rows", {Json::array({{{"object", "fp8"}, {"range", {40, 240}}}}), {5, 40}}},
};

WeightGeometry spec_geometry(const Spec& spec) {
    return weight_geometry(parse_format(spec.format), parse_layout(spec.layout), spec.shape);
}

void write_artifact(Fixture& fixture) {
    Json objects = Json::array();
    Json bindings;
    std::uint64_t offset = 0;
    fixture.payload.clear();
    for (const auto& spec : kSpecs) {
        const auto geometry = spec_geometry(spec);
        offset              = (offset + 255) / 256 * 256;
        objects.push_back({{"id", spec.id},
                           {"kind", "tensor"},
                           {"shape", spec.shape},
                           {"format", spec.format},
                           {"layout", spec.layout},
                           {"offset", offset},
                           {"bytes", geometry.bytes}});
        bindings[spec.id] = {{"object", spec.id}};
        fixture.payload.resize(offset + geometry.bytes);
        for (std::uint64_t i = 0; i < geometry.bytes; ++i) {
            fixture.payload[offset + i] = std::byte((offset + i) * 37 % 251 + 1);
        }
        if (geometry.format == QType::NVFP4) {
            put_word(fixture.payload, offset + geometry.divisor_offset,
                     std::bit_cast<std::uint32_t>(2.0F), 4);
        }
        offset += geometry.bytes;
    }
    for (const auto& [name, binding] : kPartBindings) {
        bindings[name] = {{"parts", binding.first}};
    }
    fixture.root = {
        {"components", {{"text", {{"config", Json::object()}}}}},
        {"objects", objects},
        {"bindings", bindings},
        {"uses", Json::array()},
        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", kFileSplit}},
                               {{"path", "part-1.bin"}, {"payload_bytes", offset - kFileSplit}}})}};
    fixture.write();
}

std::map<std::string, ParameterReference> bind_all(Binder& binder) {
    std::map<std::string, ParameterReference> out;
    for (const auto& spec : kSpecs) { out.emplace(spec.id, binder.parameter(spec.id, spec.shape)); }
    for (const auto& [name, binding] : kPartBindings) {
        out.emplace(name, binder.parameter(name, binding.second));
    }
    return out;
}

Binder::ShardResolver two_device_resolver(const Reader& reader) {
    return [&reader](ObjectHandle object, const WeightGeometry&) {
        const auto& id = reader.directory().tensor(object).id;
        ShardPlacement out;
        const auto split = [&](ShardAxis axis, Ranges first, Ranges second) {
            out.axis             = axis;
            out.device_ranges[0] = std::move(first);
            out.device_ranges[1] = std::move(second);
        };
        if (id == "bf16") { split(ShardAxis::Rows, {{0, 3}}, {{3, 3}}); }
        if (id == "vector") { split(ShardAxis::Rows, {{0, 4}}, {{4, 6}}); }
        if (id == "q5") { split(ShardAxis::Columns, {{0, 128}}, {{128, 128}}); }
        if (id == "fp8") { split(ShardAxis::Columns, {{0, 24}}, {{24, 16}}); }
        if (id == "nvfp4") { split(ShardAxis::Rows, {{0, 128}}, {{128, 128}}); }
        if (id == "q8") { out.axis = ShardAxis::PrimaryOnly; }
        if (id == "vision") {
            out.axis   = ShardAxis::SingleDevice;
            out.device = 1;
        }
        return out;
    };
}

bool same_placement(const DevicePlacement& a, const DevicePlacement& b) {
    return a.object == b.object && a.offset == b.offset && a.bytes == b.bytes &&
           a.alignment == b.alignment && a.device == b.device && a.axis == b.axis &&
           a.ranges == b.ranges && a.copies == b.copies;
}

bool same_plan(const MaterializationPlan& a, const MaterializationPlan& b) {
    return a.device_count == b.device_count && a.device_capacity_bytes == b.device_capacity_bytes &&
           a.per_device_capacity_bytes == b.per_device_capacity_bytes &&
           a.device_objects.size() == b.device_objects.size() &&
           std::equal(a.device_objects.begin(), a.device_objects.end(), b.device_objects.begin(),
                      same_placement);
}

// The single-device layout before multi-device placement: complete parents in object order.
std::vector<DevicePlacement> upstream_layout(const Reader& reader, std::uint64_t& capacity) {
    std::vector<DevicePlacement> out;
    capacity = 0;
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        const auto& geometry = reader.geometry(ObjectHandle{i});
        const auto alignment = std::max<std::uint64_t>(256, geometry.alignment);
        const auto offset    = (capacity + alignment - 1) / alignment * alignment;
        out.push_back({.object    = ObjectHandle{i},
                       .offset    = offset,
                       .bytes     = geometry.bytes,
                       .alignment = alignment});
        capacity = offset + geometry.bytes;
    }
    return out;
}

void single_device_plans(const Fixture& fixture) {
    Reader reader(fixture.entry);
    Binder implicit(reader);
    (void)bind_all(implicit);
    const auto plan        = std::move(implicit).finish();
    std::uint64_t capacity = 0;
    const auto expected    = upstream_layout(reader, capacity);
    require(plan.device_count == 1 && plan.device_capacity_bytes == capacity &&
                plan.per_device_capacity_bytes[0] == capacity &&
                plan.per_device_capacity_bytes[1] == 0 &&
                plan.device_objects.size() == expected.size() &&
                std::equal(expected.begin(), expected.end(), plan.device_objects.begin(),
                           same_placement),
            "one-device plan differs from the complete-parent layout");
    Binder explicit_replicated(reader, 1);
    explicit_replicated.set_shard_resolver(
        [](ObjectHandle, const WeightGeometry&) { return ShardPlacement{}; });
    (void)bind_all(explicit_replicated);
    require(same_plan(std::move(explicit_replicated).finish(), plan),
            "an explicit Replicated resolver changed the one-device plan");

    rejects([&] { (void)Binder(reader, 0); }, "zero-device binder was accepted");
    rejects([&] { (void)Binder(reader, 3); }, "three-device binder was accepted");
    const auto rejected_plan = [&](int devices, Binder::ShardResolver resolver) {
        return [&reader, devices, resolver = std::move(resolver)] {
            Binder binder(reader, devices);
            binder.set_shard_resolver(resolver);
            (void)bind_all(binder);
            (void)std::move(binder).finish();
        };
    };
    rejects(rejected_plan(1, two_device_resolver(reader)),
            "a two-device shard map planned one device");
    rejects(rejected_plan(1,
                          [](ObjectHandle, const WeightGeometry&) {
                              ShardPlacement out;
                              out.axis   = ShardAxis::SingleDevice;
                              out.device = 1;
                              return out;
                          }),
            "single-device placement outside a one-device plan");
}

void two_device_plans(const Fixture& fixture) {
    Reader reader(fixture.entry);
    std::uint64_t capacity = 0;
    const auto single      = upstream_layout(reader, capacity);
    {
        Binder binder(reader, 2);
        (void)bind_all(binder);
        const auto plan = std::move(binder).finish();
        require(plan.device_count == 2 && plan.per_device_capacity_bytes[0] == capacity &&
                    plan.per_device_capacity_bytes[1] == capacity &&
                    plan.device_capacity_bytes == 2 * capacity &&
                    plan.device_objects.size() == 2 * single.size(),
                "default two-device plan does not replicate every parent");
        for (std::size_t i = 0; i < plan.device_objects.size(); ++i) {
            auto expected   = single[i / 2];
            expected.device = static_cast<int>(i % 2);
            require(same_placement(plan.device_objects[i], expected),
                    "replicated parent differs from its one-device placement");
        }
    }
    Binder binder(reader, 2);
    binder.set_shard_resolver(two_device_resolver(reader));
    (void)bind_all(binder);
    const auto plan    = std::move(binder).finish();
    const auto resolve = two_device_resolver(reader);
    std::array<std::uint64_t, 2> end{};
    std::size_t index = 0;
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        const ObjectHandle object{i};
        const auto& geometry = reader.geometry(object);
        const auto placement = resolve(object, geometry);
        for (int device = 0; device < 2; ++device) {
            if ((placement.axis == ShardAxis::PrimaryOnly && device != 0) ||
                (placement.axis == ShardAxis::SingleDevice && device != placement.device)) {
                continue;
            }
            require(index < plan.device_objects.size(), "two-device plan lost a placement");
            const auto& actual = plan.device_objects[index++];
            DevicePlacement expected{.object = object,
                                     .bytes  = geometry.bytes,
                                     .device = device,
                                     .axis   = placement.axis};
            if (is_sharded(placement.axis)) {
                expected.ranges = placement.device_ranges[static_cast<std::size_t>(device)];
                auto slice      = tensor_slice(geometry, placement.axis, expected.ranges);
                expected.bytes  = slice.geometry.bytes;
                expected.copies = std::move(slice.copies);
            }
            auto& cursor    = end[static_cast<std::size_t>(device)];
            expected.offset = (cursor + 255) / 256 * 256;
            cursor          = expected.offset + expected.bytes;
            require(same_placement(actual, expected),
                    "two-device placement differs from its slice or device order");
        }
    }
    require(index == plan.device_objects.size() && plan.per_device_capacity_bytes == end &&
                plan.device_capacity_bytes == end[0] + end[1],
            "two-device plan capacity differs from its placements");

    const auto reject = [&](ShardPlacement placement, const char* message) {
        rejects(
            [&] {
                Binder bad(reader, 2);
                bad.set_shard_resolver(
                    [placement](ObjectHandle, const WeightGeometry&) { return placement; });
                (void)bad.parameter("bf16", {6, 40});
                (void)std::move(bad).finish();
            },
            message);
    };
    ShardPlacement missing;
    missing.axis             = ShardAxis::Rows;
    missing.device_ranges[0] = {{0, 6}};
    reject(missing, "a Rows placement without a range for device 1 was accepted");
    ShardPlacement complete_with_ranges;
    complete_with_ranges.device_ranges[0] = {{0, 3}};
    reject(complete_with_ranges, "a Replicated placement with ranges was accepted");
    ShardPlacement outside;
    outside.axis   = ShardAxis::SingleDevice;
    outside.device = 2;
    reject(outside, "a SingleDevice placement outside the plan was accepted");
    ShardPlacement past_end;
    past_end.axis             = ShardAxis::Columns;
    past_end.device_ranges[0] = {{0, 20}};
    past_end.device_ranges[1] = {{30, 20}};
    reject(past_end, "a Columns placement past the parent was accepted");
}

std::vector<std::byte> download(const WeightParent& parent, int device) {
    std::vector<std::byte> out(parent.geometry.bytes);
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemcpy(out.data(), parent.data, out.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaSetDevice(0));
    return out;
}

std::vector<std::byte> expected_bytes(const Reader& reader, const DevicePlacement& placement) {
    auto parent = reader.read_object(placement.object);
    if (!is_sharded(placement.axis)) { return parent; }
    std::vector<std::byte> out(placement.bytes);
    for (const auto& copy : placement.copies) {
        std::copy_n(parent.begin() + static_cast<std::ptrdiff_t>(copy.source_offset), copy.bytes,
                    out.begin() + static_cast<std::ptrdiff_t>(copy.dest_offset));
    }
    return out;
}

bool same_view(const WeightView& a, const WeightView& b) {
    return a.shape == b.shape && a.parts.size() == b.parts.size() &&
           std::equal(a.parts.begin(), a.parts.end(), b.parts.begin(), [](auto& x, auto& y) {
               return x.parent == y.parent && x.begin == y.begin && x.end == y.end;
           });
}

void single_device_upload(const Fixture& fixture, DeviceContext& device) {
    Reader reader(fixture.entry);
    Binder binder(reader);
    const auto parameters = bind_all(binder);
    auto plan             = std::move(binder).finish();
    const auto expected   = plan;
    rejects(
        [&] {
            Binder two(reader, 2);
            (void)bind_all(two);
            (void)materialize(reader, std::move(two).finish(), device);
        },
        "a two-device plan was uploaded through one DeviceContext");
    auto backing      = materialize(reader, std::move(plan), device);
    const auto& stats = backing.stats();
    std::uint64_t h2d = 0;
    for (const auto& placement : expected.device_objects) {
        const auto& parent = backing.device_parent(placement.object);
        require(download(parent, 0) == reader.read_object(placement.object),
                "one-device upload changed parent bytes");
        require(!backing.device_shard(placement.object).sharded() &&
                    parent.geometry.shape == reader.geometry(placement.object).shape,
                "one-device parent is not complete");
        h2d += placement.bytes;
    }
    require(backing.device_count() == 1 && stats.device_count == 1 && stats.h2d_bytes == h2d &&
                stats.per_device_h2d_bytes[0] == h2d &&
                stats.device_capacity_bytes == expected.device_capacity_bytes &&
                stats.replicated_bytes[0] == h2d && stats.sharded_bytes[0] == 0 &&
                stats.local_bytes[0] == 0 &&
                stats.device_object_count == expected.device_objects.size() &&
                !backing.has_device(reader.find("bf16"), 1),
            "one-device statistics differ from the plan");
    for (const auto& [name, parameter] : parameters) {
        const auto view = bind_view(parameter, backing);
        require(same_view(view, bind_view(parameter, backing, 0)) && view.shape == parameter.shape,
                "device 0 view differs from the default view");
    }
    const auto divisor = backing.device_parent(reader.find("nvfp4")).weight_scale_divisor;
    require(divisor == 2.0F, "one-device NVFP4 divisor changed");

    // The same plan through a one-device ExecutionContext.
    ExecutionContext execution({device.device});
    Binder again(reader);
    (void)bind_all(again);
    auto through_context = materialize(reader, std::move(again).finish(), execution);
    for (const auto& placement : expected.device_objects) {
        require(download(through_context.device_parent(placement.object), 0) ==
                    reader.read_object(placement.object),
                "ExecutionContext upload changed parent bytes");
    }
}

void expect_view(const WeightView& view, Shape shape, const WeightParent& parent,
                 std::uint64_t begin, std::uint64_t end, const char* message) {
    require(view.shape == shape && view.parts.size() == 1 && view.parts[0].parent == &parent &&
                view.parts[0].begin == begin && view.parts[0].end == end,
            message);
}

void two_device_upload(const Fixture& fixture, ExecutionContext& execution) {
    Reader reader(fixture.entry);
    Binder binder(reader, 2);
    binder.set_shard_resolver(two_device_resolver(reader));
    const auto parameters = bind_all(binder);
    auto plan             = std::move(binder).finish();
    const auto expected   = plan;
    rejects(
        [&] {
            auto copy = expected;
            (void)materialize(reader, std::move(copy), execution.primary());
        },
        "a two-device plan was uploaded through one DeviceContext");
    auto backing = materialize(reader, std::move(plan), execution);
    int current  = -1;
    CUDA_CHECK(cudaGetDevice(&current));
    require(current == execution.primary().device, "upload changed the current device");

    std::array<std::uint64_t, 2> h2d{}, sharded{}, replicated{}, local{};
    for (const auto& placement : expected.device_objects) {
        const auto device  = static_cast<std::size_t>(placement.device);
        const auto& id     = reader.directory().tensor(placement.object).id;
        const auto& parent = backing.device_parent(placement.object, placement.device);
        require(download(parent, execution.dev[device]->device) ==
                    expected_bytes(reader, placement),
                ("device bytes differ from the parent slice: " + id).c_str());
        const auto& shard = backing.device_shard(placement.object, placement.device);
        require(shard.axis == placement.axis && shard.ranges == placement.ranges &&
                    shard.source_shape == reader.geometry(placement.object).shape,
                "device shard descriptor differs from its placement");
        if (is_sharded(placement.axis)) {
            sharded[device] += placement.bytes;
            for (const auto& copy : placement.copies) { h2d[device] += copy.bytes; }
        } else {
            (placement.axis == ShardAxis::Replicated ? replicated : local)[device] +=
                placement.bytes;
            h2d[device] += placement.bytes;
        }
    }
    const auto& stats = backing.stats();
    require(stats.device_count == 2 && stats.per_device_h2d_bytes == h2d &&
                stats.h2d_bytes == h2d[0] + h2d[1] && stats.sharded_bytes == sharded &&
                stats.replicated_bytes == replicated && stats.local_bytes == local &&
                stats.per_device_capacity_bytes == expected.per_device_capacity_bytes &&
                stats.device_object_count == kSpecs.size(),
            "two-device statistics differ from the plan");
    require(local[0] == reader.geometry(reader.find("q8")).bytes &&
                local[1] == reader.geometry(reader.find("vision")).bytes &&
                !backing.has_device(reader.find("q8"), 1) &&
                !backing.has_device(reader.find("vision"), 0),
            "device-local parents are held by the wrong device");
    rejects([&] { (void)bind_view(parameters.at("q8"), backing, 1); },
            "a PrimaryOnly parent was bound on device 1");

    const auto parent = [&](std::string_view id, int device) -> const WeightParent& {
        return backing.device_parent(reader.find(id), device);
    };
    // Rows: parent rows 2..5 of bf16; device 0 holds rows 0..2, device 1 rows 3..5.
    expect_view(bind_view(parameters.at("bf16_rows"), backing, 0), {1, 40}, parent("bf16", 0), 80,
                120, "device 0 row view");
    expect_view(bind_view(parameters.at("bf16_rows"), backing, 1), {3, 40}, parent("bf16", 1), 0,
                120, "device 1 row view");
    expect_view(bind_view(parameters.at("bf16"), backing, 1), {3, 40}, parent("bf16", 1), 0, 120,
                "device 1 complete row shard view");
    expect_view(bind_view(parameters.at("vector_pairs"), backing, 0), {2, 2}, parent("vector", 0),
                0, 4, "device 0 reshaped vector view");
    expect_view(bind_view(parameters.at("vector_pairs"), backing, 1), {3, 2}, parent("vector", 1),
                0, 6, "device 1 reshaped vector view");
    // Columns: whole parent rows 1..5 of fp8 at the shard width.
    expect_view(bind_view(parameters.at("fp8_rows"), backing, 0), {5, 24}, parent("fp8", 0), 24,
                144, "device 0 column view");
    expect_view(bind_view(parameters.at("fp8_rows"), backing, 1), {5, 16}, parent("fp8", 1), 16, 96,
                "device 1 column view");
    for (int device = 0; device < 2; ++device) {
        const auto q5 = native_weight(bind_view(parameters.at("q5"), backing, device));
        require(q5.n == 6 && q5.k == 128 && q5.payload == parent("q5", device).data,
                "column shard of a grouped parent is not a complete native weight");
        const auto fp8 = native_weight(bind_view(parameters.at("fp8"), backing, device));
        require(fp8.n == 6 && fp8.k == (device ? 16 : 24), "FP8 column shard native weight");
        const auto nvfp4 = native_weight(bind_view(parameters.at("nvfp4"), backing, device), 1.0F);
        require(nvfp4.n == 128 && nvfp4.k == 128 && nvfp4.weight_scale_divisor == 2.0F,
                "NVFP4 row shard lost its divisor");
        expect_view(bind_view(parameters.at("shared"), backing, device), {4, 64},
                    parent("shared", device), 0, 256, "replicated view");
    }
    expect_view(bind_view(parameters.at("vision"), backing, 1), {2, 128}, parent("vision", 1), 0,
                256, "single-device view");
}

// A parent larger than one 64 MiB staging slot, so ranges of both devices straddle chunks.
// "fp8" and "shared" in one mapped host copy, every other parent Replicated.
ShardPlacement host_mapped_resolver(const Reader& reader, ObjectHandle object) {
    const auto& id = reader.directory().tensor(object).id;
    ShardPlacement out;
    if (id == "fp8" || id == "shared") { out.axis = ShardAxis::HostMapped; }
    return out;
}

// Plans and materializes host_mapped_resolver() on `devices`, then checks the plan layout, the
// bytes each device reads through its own parent, the views and the statistics.
void host_mapped_upload(const Fixture& fixture, std::span<DeviceContext* const> devices) {
    Reader reader(fixture.entry);
    const int count = static_cast<int>(devices.size());
    Binder binder(reader, count);
    binder.set_shard_resolver([&reader](ObjectHandle object, const WeightGeometry&) {
        return host_mapped_resolver(reader, object);
    });
    const auto parameters    = bind_all(binder);
    auto plan                = std::move(binder).finish();
    const auto expected      = plan;
    std::uint64_t mapped_end = 0;
    std::uint64_t mapped     = 0;
    std::uint64_t capacity   = 0;
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        const ObjectHandle object{i};
        const auto& geometry = reader.geometry(object);
        const auto alignment = std::max<std::uint64_t>(256, geometry.alignment);
        if (host_mapped_resolver(reader, object).axis != ShardAxis::HostMapped) {
            capacity = (capacity + alignment - 1) / alignment * alignment + geometry.bytes;
            continue;
        }
        const auto offset = (mapped_end + alignment - 1) / alignment * alignment;
        require(std::any_of(expected.host_mapped_objects.begin(),
                            expected.host_mapped_objects.end(),
                            [&](const HostMappedPlacement& placement) {
                                return placement.object == object && placement.offset == offset &&
                                       placement.bytes == geometry.bytes &&
                                       placement.alignment == alignment;
                            }),
                "host-mapped parent is not planned at the next aligned host offset");
        mapped_end = offset + geometry.bytes;
        mapped += geometry.bytes;
    }
    require(expected.host_mapped_objects.size() == 2 && expected.host_mapped_bytes == mapped_end,
            "host-mapped plan differs from its parents");
    for (int device = 0; device < count; ++device) {
        require(expected.per_device_capacity_bytes[static_cast<std::size_t>(device)] == capacity,
                "a host-mapped parent took device capacity");
    }
    for (const auto& placement : expected.device_objects) {
        require(host_mapped_resolver(reader, placement.object).axis != ShardAxis::HostMapped,
                "a host-mapped parent has a device placement");
    }

    auto backing      = materialize(reader, std::move(plan), devices);
    const auto& stats = backing.stats();
    require(stats.host_mapped_bytes == expected.host_mapped_bytes,
            "host-mapped statistics differ from the plan");
    std::uint64_t all = 0;
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        all += reader.geometry(ObjectHandle{i}).bytes;
    }
    for (int device = 0; device < count; ++device) {
        const auto d = static_cast<std::size_t>(device);
        // Every other parent is a complete device copy; the host-mapped ones are never uploaded.
        require(stats.per_device_host_mapped_bytes[d] == mapped &&
                    stats.per_device_capacity_bytes[d] == capacity &&
                    stats.per_device_h2d_bytes[d] == stats.replicated_bytes[d] &&
                    stats.replicated_bytes[d] + mapped == all && stats.sharded_bytes[d] == 0 &&
                    stats.local_bytes[d] == 0,
                "per-device host-mapped statistics are wrong");
    }
    for (const auto& placement : expected.host_mapped_objects) {
        const auto parent_bytes = reader.read_object(placement.object);
        const std::byte* first  = nullptr;
        for (int device = 0; device < count; ++device) {
            require(backing.has_device(placement.object, device),
                    "a device does not hold its host-mapped parent");
            const auto& parent = backing.device_parent(placement.object, device);
            const auto& shard  = backing.device_shard(placement.object, device);
            require(shard.axis == ShardAxis::HostMapped && !shard.sharded() &&
                        parent.geometry.shape == reader.geometry(placement.object).shape,
                    "host-mapped parent is not complete");
            cudaPointerAttributes attributes{};
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, parent.data));
            require(attributes.type == cudaMemoryTypeHost,
                    "host-mapped parent does not address page-locked host memory");
            if (first == nullptr) { first = parent.data; }
            require(parent.data == first, "devices address different host-mapped copies");
            CUDA_CHECK(cudaSetDevice(devices[static_cast<std::size_t>(device)]->device));
            void* scratch = nullptr;
            CUDA_CHECK(cudaMalloc(&scratch, parent.geometry.bytes));
            std::vector<std::byte> read(parent.geometry.bytes);
            // Through the device: the mapped bytes as this device sees them.
            CUDA_CHECK(cudaMemcpy(scratch, parent.data, read.size(), cudaMemcpyDefault));
            CUDA_CHECK(cudaMemcpy(read.data(), scratch, read.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(scratch));
            CUDA_CHECK(cudaSetDevice(devices[0]->device));
            require(read == parent_bytes, "host-mapped parent bytes differ from the artifact");
        }
    }
    for (const auto& [name, parameter] : parameters) {
        for (int device = 0; device < count; ++device) {
            const auto view = bind_view(parameter, backing, device);
            require(view.shape == parameter.shape && !view.parts.empty(),
                    "a parameter view over a host-mapped plan is incomplete");
        }
    }
    const auto view = bind_view(parameters.at("fp8_rows"), backing, count - 1);
    expect_view(view, {5, 40}, backing.device_parent(reader.find("fp8"), count - 1), 40, 240,
                "partial view of a host-mapped parent");
}

void large_upload(ExecutionContext& execution) {
    constexpr std::uint64_t rows  = 8200;
    constexpr std::uint64_t width = 4096;
    constexpr std::uint64_t bytes = rows * width * 2;
    const auto value              = [](std::uint64_t position) {
        return std::byte((position * 17 + position / 4096 * 13) % 251);
    };
    Fixture fixture;
    fixture.payload.resize(bytes);
    for (std::uint64_t i = 0; i < bytes; ++i) { fixture.payload[i] = value(i); }
    fixture.root = {{"components", {{"text", {{"config", Json::object()}}}}},
                    {"objects", Json::array({{{"id", "large"},
                                              {"kind", "tensor"},
                                              {"shape", {rows, width}},
                                              {"format", "bf16"},
                                              {"layout", "contiguous_le_v1"},
                                              {"offset", 0},
                                              {"bytes", bytes}}})},
                    {"bindings", {{"large", {{"object", "large"}}}}},
                    {"uses", Json::array()},
                    {"files", Json::array({{{"path", nullptr}, {"payload_bytes", bytes}}})}};
    fixture.write();
    Reader reader(fixture.entry);
    for (const bool split : {false, true}) {
        Binder binder(reader, 2);
        if (split) {
            binder.set_shard_resolver([](ObjectHandle, const WeightGeometry&) {
                ShardPlacement out;
                out.axis             = ShardAxis::Rows;
                out.device_ranges[0] = {{0, rows / 2}};
                out.device_ranges[1] = {{rows / 2, rows / 2}};
                return out;
            });
        }
        (void)binder.parameter("large", {rows, width});
        auto backing = materialize(reader, std::move(binder).finish(), execution);
        for (int device = 0; device < 2; ++device) {
            const auto& parent = backing.device_parent(reader.find("large"), device);
            const auto origin  = split && device ? bytes / 2 : 0;
            std::vector<std::byte> chunk(1024 * 1024);
            CUDA_CHECK(cudaSetDevice(execution.dev[static_cast<std::size_t>(device)]->device));
            for (std::uint64_t offset = 0; offset < parent.geometry.bytes; offset += chunk.size()) {
                const auto count =
                    std::min<std::uint64_t>(chunk.size(), parent.geometry.bytes - offset);
                CUDA_CHECK(
                    cudaMemcpy(chunk.data(), parent.data + offset, count, cudaMemcpyDeviceToHost));
                for (std::uint64_t i = 0; i < count; ++i) {
                    require(chunk[i] == value(origin + offset + i),
                            split ? "row shard straddling a staging slot lost bytes"
                                  : "replicated parent straddling a staging slot lost bytes");
                }
            }
            CUDA_CHECK(cudaSetDevice(execution.primary().device));
        }
    }
}

// Exercise the materializer's explicit cap, including mandatory parents and active Host KV.
// Run the same cases on one and two devices; shared host storage must obey one host budget.
void host_mapped_reservation(const Fixture& fixture, std::span<DeviceContext* const> devices) {
    Reader reader(fixture.entry);
    const auto make_plan = [&] {
        Binder binder(reader, static_cast<int>(devices.size()));
        binder.set_shard_resolver([&reader](ObjectHandle object, const WeightGeometry&) {
            return host_mapped_resolver(reader, object);
        });
        (void)bind_all(binder);
        return std::move(binder).finish();
    };
    const auto total = make_plan().host_mapped_bytes;

    struct Case {
        const char* name;
        bool pageable;
        std::uint64_t limit;
        std::uint64_t reserved;
        bool split;
        std::string_view error;
    };

    const Case cases[] = {
        {"mandatory, no cap", false, 0, 0, false, {}},
        {"mandatory, exact cap", false, total, 0, false, {}},
        {"mandatory, spare byte", false, total + 1, 0, false, {}},
        {"mandatory, one byte short", false, total - 1, 0, false, "must stay page-locked"},
        {"mandatory, reservation leaves exact room", false, total + 1, 1, false, {}},
        {"mandatory, reservation leaves insufficient room", false, total, 1, false,
         "must stay page-locked"},
        {"mixed, exact cap", true, total, 0, false, {}},
        {"mixed, one byte short", true, total - 1, 0, true, {}},
        {"mixed, reservation forces split", true, total, 1, true, {}},
        {"mixed, mandatory parent exceeds cap", true, 1, 0, false, "must stay page-locked"},
        {"reservation consumes cap", true, total, total, false, "leaves nothing"},
        {"reservation exceeds cap", true, total, total + 1, false, "leaves nothing"},
    };
    for (const auto& test : cases) {
        const auto check = [&](bool condition, const std::string& detail) {
            require(condition, (std::string(test.name) + ": " + detail).c_str());
        };
        auto plan = make_plan();
        for (auto& placement : plan.host_mapped_objects) {
            if (test.pageable && reader.directory().tensor(placement.object).id == "fp8") {
                placement.pageable_rank = 1;
            }
        }
        plan.host_pinned_limit_bytes    = test.limit;
        plan.host_pinned_reserved_bytes = test.reserved;
        const auto placements           = plan.host_mapped_objects;
        std::string error;
        try {
            auto backing      = materialize(reader, std::move(plan), devices);
            const auto& stats = backing.stats();
            check((stats.host_pageable_objects != 0) == test.split,
                  "incorrect pinned/pageable placement");
            check(stats.host_pinned_limit_bytes == test.limit, "incorrect reported limit");
            check(!stats.host_pinned_alloc_failed,
                  "explicit budget unexpectedly needed allocation fallback");
            if (test.limit != 0) {
                check(stats.host_mapped_bytes <= test.limit - test.reserved,
                      "allocation exceeded the remaining budget");
            }
            // Placement changes storage, never the encoded weights. Check every rank's view.
            for (const auto& placement : placements) {
                const auto expected = reader.read_object(placement.object);
                for (std::size_t rank = 0; rank < devices.size(); ++rank) {
                    const auto& parent =
                        backing.device_parent(placement.object, static_cast<int>(rank));
                    CUDA_CHECK(cudaSetDevice(devices[rank]->device));
                    std::vector<std::byte> actual(expected.size());
                    CUDA_CHECK(
                        cudaMemcpy(actual.data(), parent.data, actual.size(), cudaMemcpyDefault));
                    check(parent.geometry.bytes == expected.size() && actual == expected,
                          "stored weight bytes changed");
                }
            }
            CUDA_CHECK(cudaSetDevice(devices[0]->device));
        } catch (const ArtifactError& exception) { error = exception.what(); }
        check(test.error.empty() ? error.empty() : error.find(test.error) != std::string::npos,
              "unexpected result: " + error);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool two_devices = argc == 3 && std::string_view(argv[1]) == "--devices" &&
                                 std::string_view(argv[2]) == "2";
        if (argc != 1 && !two_devices) { throw std::invalid_argument("expected [--devices 2]"); }
        Fixture fixture;
        write_artifact(fixture);
        single_device_plans(fixture);
        two_device_plans(fixture);
        int count         = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
            (result == cudaSuccess && count == 0)) {
            return 77;
        }
        CUDA_CHECK(result);
        if (!two_devices) {
            DeviceContext device;
            single_device_upload(fixture, device);
            DeviceContext* const one[] = {&device};
            host_mapped_upload(fixture, one);
            host_mapped_reservation(fixture, one);
            std::cout << "one-device artifact placement checks passed\n";
            return 0;
        }
        if (count < 2) {
            std::cout << "two-device placement needs two CUDA devices\n";
            return 77;
        }
        ExecutionContext execution({0, 1});
        two_device_upload(fixture, execution);
        large_upload(execution);
        DeviceContext* const pair[] = {&*execution.dev[0], &*execution.dev[1]};
        host_mapped_upload(fixture, pair);
        host_mapped_reservation(fixture, pair);
        std::cout << "two-device artifact placement checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
