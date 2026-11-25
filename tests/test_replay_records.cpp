#include "core/gdn_replay_records.h"
#include "core/kda_replay_records.h"
#include "core/buffer.h"
#include "core/linear_attention_state.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

int fail(const char* label) {
    std::cerr << label << '\n';
    return 1;
}

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << label << '\n';
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int expect_shape(const ninfer::Tensor& tensor, std::int32_t d0, std::int32_t d1, std::int32_t d2,
                 std::int32_t d3, const char* label) {
    if (tensor.ne[0] == d0 && tensor.ne[1] == d1 && tensor.ne[2] == d2 && tensor.ne[3] == d3) {
        return 0;
    }
    std::cerr << label << " shape differs\n";
    return 1;
}

template <class Fn>
int expect_throw(Fn&& fn, const char* label) {
    try {
        fn();
    } catch (const std::exception&) { return 0; }
    std::cerr << label << " did not throw\n";
    return 1;
}

template <class Layout>
int compact_widths(const Layout& maximum,
                   std::initializer_list<std::pair<ninfer::TensorRegion Layout::*, int>> planes) {
    int failures = 0;
    for (int invalid : {-1, 0, maximum.spec.width + 1})
        failures +=
            expect_throw([&] { (void)maximum.with_width(invalid); }, "outside parent width");
    auto malformed = maximum;
    malformed.conv.region.bytes -= 2;
    failures += expect_throw([&] { (void)malformed.with_width(2); }, "malformed parent layout");
    for (int width = 1; width <= maximum.spec.width; ++width) {
        const auto child = maximum.with_width(width);
        failures += expect(child.spec.width == width &&
                               child.spec.record_capacity == maximum.spec.record_capacity,
                           "compact record geometry");
        for (auto [plane, width_axis] : planes) {
            const auto& a = maximum.*plane;
            const auto& b = child.*plane;
            failures += expect(b.region.offset == a.region.offset && b.region.alignment == 256 &&
                                   b.region.bytes == a.region.bytes / maximum.spec.width * width,
                               "compact plane range");
            auto expected        = a.shape;
            expected[width_axis] = width;
            failures += expect(b.shape == expected && b.dtype == a.dtype, "compact plane shape");
        }
    }
    return failures;
}

using GdnAccess = ninfer::detail::GdnReplayRecordAccess;

int compact_gdn_widths(const ninfer::GdnReplayRecordLayout& maximum) {
    int failures = compact_widths(maximum, {{&ninfer::GdnReplayRecordLayout::conv, 1}});
    const auto [key, delta, alpha] = GdnAccess::planes(maximum);
    for (int width = 1; width <= maximum.spec.width; ++width) {
        const auto child                                 = maximum.with_width(width);
        const auto [child_key, child_delta, child_alpha] = GdnAccess::planes(child);
        failures +=
            expect_size(child.payload_bytes(), maximum.payload_bytes() / maximum.spec.width * width,
                        "compact GDN payload");
        const auto check = [&](const auto& parent, const auto& plane, int axis) {
            auto shape  = parent.shape;
            shape[axis] = width;
            return expect(
                plane.region.offset == parent.region.offset && plane.region.alignment == 256 &&
                    plane.region.bytes == parent.region.bytes / maximum.spec.width * width &&
                    plane.shape == shape && plane.dtype == parent.dtype,
                "compact GDN recurrence plane");
        };
        failures +=
            check(key, child_key, 2) + check(delta, child_delta, 2) + check(alpha, child_alpha, 1);
    }
    return failures;
}

using KdaAccess = ninfer::detail::KdaReplayRecordAccess;

int compact_kda_widths(const ninfer::KdaReplayRecordLayout& maximum) {
    int failures = compact_widths(maximum, {{&ninfer::KdaReplayRecordLayout::conv, 1}});
    const auto [key, delta, alpha] = KdaAccess::planes(maximum);
    for (int width = 1; width <= maximum.spec.width; ++width) {
        const auto child = maximum.with_width(width);
        const auto [child_key, child_delta, child_alpha] = KdaAccess::planes(child);
        failures +=
            expect_size(child.payload_bytes(), maximum.payload_bytes() / maximum.spec.width * width,
                        "compact KDA payload");
        const auto check = [&](const auto& parent, const auto& plane) {
            auto shape = parent.shape;
            shape[2] = width;
            return expect(
                plane.region.offset == parent.region.offset && plane.region.alignment == 256 &&
                    plane.region.bytes == parent.region.bytes / maximum.spec.width * width &&
                    plane.shape == shape && plane.dtype == parent.dtype,
                "compact KDA recurrence plane");
        };
        failures += check(key, child_key) + check(delta, child_delta) + check(alpha, child_alpha);
    }
    return failures;
}

int derived_record_layouts() {
    ninfer::LayoutBuilder builder;
    (void)builder.add(73, 1, "preceding field");
    const auto g = ninfer::plan_gdn_replay_records(builder, {3, 5, 8, 64, 2, 4, 16, 16});
    const auto k = ninfer::plan_kda_replay_records(builder, {3, 5, 8, 64, 2, 4, 16, 16});
    return compact_gdn_widths(g) + compact_kda_widths(k);
}

std::size_t record_bytes(const ninfer::GdnReplayRecordSpec& spec) {
    ninfer::LayoutBuilder builder;
    (void)ninfer::plan_gdn_replay_records(builder, spec);
    return builder.finish(256);
}

int large_record_capacity() {
    int failures = 0;
    for (const int capacity : {9, 64, 128}) {
        constexpr int layers = 3, width = 4, dim = 16, heads = 2;
        const ninfer::GdnReplayRecordSpec gdn{layers, capacity, width, 3 * dim * heads,
                                              heads,  heads,    dim,   dim};
        ninfer::LayoutBuilder gdn_builder;
        const auto gdn_layout = ninfer::plan_gdn_replay_records(gdn_builder, gdn);
        ninfer::HostBuffer gdn_memory(gdn_builder.finish(256));
        const ninfer::GdnReplayRecords gdn_records(gdn_memory.span(), gdn_layout);
        const ninfer::KdaReplayRecordSpec kda{layers, capacity, width, 3 * dim * heads,
                                              heads,  heads,    dim,   dim};
        ninfer::LayoutBuilder kda_builder;
        const auto kda_layout = ninfer::plan_kda_replay_records(kda_builder, kda);
        ninfer::HostBuffer kda_memory(kda_builder.finish(256));
        const ninfer::KdaReplayRecords kda_records(kda_memory.span(), kda_layout);
        const auto gdn_base = GdnAccess::planes(gdn_records);
        const auto kda_base = KdaAccess::planes(kda_records);
        failures += expect(kda_base.key.dtype == ninfer::DType::BF16, "KDA key dtype");
        for (const auto* plane : {&kda_records.conv, &kda_base.key, &kda_base.delta,
                                  &kda_base.alpha})
            failures += expect(reinterpret_cast<std::uintptr_t>(plane->data) % 256 == 0,
                               "KDA record plane alignment");
        for (const int active : {1, 8, capacity}) {
            const auto gdn_layer         = gdn_records.layer(2, active);
            const auto g                 = GdnAccess::planes(gdn_layer.recurrence);
            const auto kda_layer = kda_records.layer(2, active);
            const auto k = KdaAccess::planes(kda_layer.recurrence);
            const std::size_t key_offset = 2ULL * capacity * dim * heads * width * 2;
            failures += expect_size(static_cast<const std::byte*>(g.key.data) -
                                        static_cast<const std::byte*>(gdn_base.key.data),
                                    key_offset, "GDN layer offset depends on active rows");
            failures += expect_size(static_cast<const std::byte*>(k.key.data) -
                                        static_cast<const std::byte*>(kda_base.key.data),
                                    key_offset, "KDA layer offset depends on active rows");
            failures += expect_shape(g.key, dim, heads, width, active, "large GDN layer");
            failures += expect_shape(k.key, dim, heads, width, active, "large KDA layer");
            failures += expect_shape(k.delta, dim, heads, width, active, "large KDA delta");
            failures += expect_shape(k.alpha, dim, heads, width, active, "large KDA alpha");
            for (const auto [plane, base] : {std::pair{&k.delta, &kda_base.delta},
                                             std::pair{&k.alpha, &kda_base.alpha}}) {
                failures += expect(plane->dtype == ninfer::DType::FP32, "KDA transition dtype");
                failures += expect_size(static_cast<const std::byte*>(plane->data) -
                                            static_cast<const std::byte*>(base->data),
                                        2 * key_offset, "KDA transition offset");
            }
        }
        failures += expect_throw([&] { (void)gdn_records.layer(0, capacity + 1); },
                                 "GDN row outside allocated capacity");
        failures += expect_throw([&] { (void)kda_records.layer(0, capacity + 1); },
                                 "KDA row outside allocated capacity");
    }
    return failures;
}

} // namespace

int main() {
    if (derived_record_layouts()) return 1;
    int failures = large_record_capacity();

    const ninfer::GdnReplayRecordSpec spec{
        .layers          = 3,
        .record_capacity = 5,
        .width           = 4,
        .conv_channels   = 256,
        .qk_heads        = 2,
        .value_heads     = 6,
        .key_dim         = 128,
        .value_dim       = 128,
    };
    ninfer::LayoutBuilder builder;
    const auto layout       = ninfer::plan_gdn_replay_records(builder, spec);
    const std::size_t bytes = builder.finish(256);
    ninfer::HostBuffer backing(bytes);
    const ninfer::GdnReplayRecords records(backing.span(), layout);

    const auto [key, delta, alpha] = GdnAccess::planes(records);
    failures += expect_shape(records.conv, 256, 4, 15, 1, "conv plane");
    failures += expect_shape(key, 128, 2, 4, 15, "key plane");
    failures += expect_shape(delta, 128, 6, 4, 15, "delta plane");
    failures += expect_shape(alpha, 6, 4, 15, 1, "alpha plane");
    failures += expect(records.conv.dtype == ninfer::DType::BF16, "conv dtype differs");
    failures += expect(key.dtype == ninfer::DType::BF16, "key dtype differs");
    failures += expect(delta.dtype == ninfer::DType::FP32, "delta dtype differs");
    failures += expect(alpha.dtype == ninfer::DType::FP32, "alpha dtype differs");
    failures += expect(reinterpret_cast<std::uintptr_t>(records.conv.data) % 256 == 0,
                       "conv plane is not aligned");
    failures +=
        expect(reinterpret_cast<std::uintptr_t>(key.data) % 256 == 0, "key plane is not aligned");
    failures += expect(reinterpret_cast<std::uintptr_t>(delta.data) % 256 == 0,
                       "delta plane is not aligned");
    failures += expect(reinterpret_cast<std::uintptr_t>(alpha.data) % 256 == 0,
                       "alpha plane is not aligned");
    failures += expect_size(layout.payload_bytes(),
                            records.conv.bytes() + key.bytes() + delta.bytes() + alpha.bytes(),
                            "record payload bytes");

    const auto layer2 = records.layer(2, 3);
    const auto layer  = GdnAccess::planes(layer2.recurrence);
    failures += expect_shape(layer2.conv, 256, 4, 3, 1, "layer conv slice");
    failures += expect_shape(layer.key, 128, 2, 4, 3, "layer key slice");
    failures += expect_shape(layer.delta, 128, 6, 4, 3, "layer delta slice");
    failures += expect_shape(layer.alpha, 6, 4, 3, 1, "layer alpha slice");
    failures += expect(
        static_cast<std::byte*>(layer2.conv.data) - static_cast<std::byte*>(records.conv.data) ==
            static_cast<std::ptrdiff_t>(2 * spec.record_capacity * records.conv.nb[2]),
        "layer conv slice offset differs");
    failures +=
        expect(static_cast<std::byte*>(layer.key.data) - static_cast<std::byte*>(key.data) ==
                   static_cast<std::ptrdiff_t>(2 * spec.record_capacity * key.nb[3]),
               "layer key slice offset differs");
    failures +=
        expect_size(static_cast<std::byte*>(layer.delta.data) - static_cast<std::byte*>(delta.data),
                    2 * spec.record_capacity * delta.nb[3], "layer delta slice offset");
    failures +=
        expect_size(static_cast<std::byte*>(layer.alpha.data) - static_cast<std::byte*>(alpha.data),
                    2 * spec.record_capacity * alpha.nb[2], "layer alpha slice offset");
    failures += expect_throw([&] { (void)records.layer(-1, 1); }, "negative layer");
    failures += expect_throw([&] { (void)records.layer(3, 1); }, "past-end layer");
    failures += expect_throw([&] { (void)records.layer(0, 0); }, "zero active rows");
    failures += expect_throw([&] { (void)records.layer(0, 6); }, "excess active rows");

    failures += expect_size(record_bytes({.layers          = 48,
                                          .record_capacity = 8,
                                          .width           = 6,
                                          .conv_channels   = 10240,
                                          .qk_heads        = 16,
                                          .value_heads     = 48,
                                          .key_dim         = 128,
                                          .value_dim       = 128}),
                            113688576, "48-layer T6 capacity");
    failures += expect_size(record_bytes({.layers          = 30,
                                          .record_capacity = 8,
                                          .width           = 6,
                                          .conv_channels   = 8192,
                                          .qk_heads        = 16,
                                          .value_heads     = 32,
                                          .key_dim         = 128,
                                          .value_dim       = 128}),
                            53268480, "30-layer T6 capacity");
    failures += expect_size(record_bytes({.layers          = 30,
                                          .record_capacity = 8,
                                          .width           = 16,
                                          .conv_channels   = 8192,
                                          .qk_heads        = 16,
                                          .value_heads     = 32,
                                          .key_dim         = 128,
                                          .value_dim       = 128}),
                            142049280, "30-layer T16 capacity");

    failures += expect_throw(
        [&] {
            ninfer::LayoutBuilder invalid;
            (void)ninfer::plan_gdn_replay_records(invalid, {.layers          = 1,
                                                            .record_capacity = 0,
                                                            .width           = 2,
                                                            .conv_channels   = 1,
                                                            .qk_heads        = 1,
                                                            .value_heads     = 1,
                                                            .key_dim         = 1,
                                                            .value_dim       = 1});
        },
        "zero record capacity");
    failures += expect_throw(
        [&] {
            auto overflow   = spec;
            overflow.layers = std::numeric_limits<std::int32_t>::max();
            ninfer::LayoutBuilder invalid;
            (void)ninfer::plan_gdn_replay_records(invalid, overflow);
        },
        "GDN layer/row product overflow");
    failures += expect_throw(
        [&] {
            ninfer::LayoutBuilder invalid;
            (void)ninfer::plan_kda_replay_records(
                invalid, {std::numeric_limits<std::int32_t>::max(), 2, 1, 3, 1, 1, 1, 1});
        },
        "KDA layer/row product overflow");

    ninfer::StateImageBuilder state_builder;
    const auto state_layout =
        ninfer::plan_linear_attention_state(state_builder, {.layers         = 3,
                                                            .conv_channels  = 256,
                                                            .conv_width     = 3,
                                                            .value_heads    = 6,
                                                            .value_head_dim = 128,
                                                            .key_head_dim   = 128,
                                                            .conv_dtype     = ninfer::DType::BF16});
    const auto images = state_builder.finish();
    ninfer::HostBuffer state_backing(2 * images.image_bytes + 512);
    ninfer::StateImageStorage storage(state_backing.span(), images);
    ninfer::LinearAttentionStateView state(storage, state_layout);
    const auto all            = state.all_layers_view();
    const std::int64_t offset = images.image_bytes + 512;
    failures += expect(all.conv.tensor(offset).data == state.conv_tensor(0, offset).data,
                       "conv record base differs");
    failures += expect(all.recurrent.tensor(offset).data == state.recurrent_tensor(0, offset).data,
                       "recurrent record base differs");
    failures += expect_size(all.conv.field.layer_stride,
                            static_cast<std::byte*>(state.conv_tensor(1, offset).data) -
                                static_cast<std::byte*>(state.conv_tensor(0, offset).data),
                            "conv layer stride");
    failures += expect_size(all.recurrent.field.layer_stride,
                            static_cast<std::byte*>(state.recurrent_tensor(1, offset).data) -
                                static_cast<std::byte*>(state.recurrent_tensor(0, offset).data),
                            "recurrent layer stride");
    failures += expect_size(static_cast<std::size_t>(records.spec.record_capacity), 5,
                            "independent record capacity");

    return failures == 0 ? 0 : fail("Replay record storage test failed");
}
