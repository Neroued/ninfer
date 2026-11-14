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
        for (const int active : {1, 8, capacity}) {
            const auto g                 = gdn_records.layer(2, active);
            const auto k                 = kda_records.layer(2, active);
            const std::size_t key_offset = 2ULL * capacity * dim * heads * width * 2;
            failures += expect_size(static_cast<const std::byte*>(g.key.data) -
                                        static_cast<const std::byte*>(gdn_records.key.data),
                                    key_offset, "GDN layer offset depends on active rows");
            failures += expect_size(static_cast<const std::byte*>(k.key.data) -
                                        static_cast<const std::byte*>(kda_records.key.data),
                                    key_offset, "KDA layer offset depends on active rows");
            failures += expect_shape(g.key, dim, heads, width, active, "large GDN layer");
            failures += expect_shape(k.key, dim, heads, width, active, "large KDA layer");
            failures += expect_shape(k.log_decay, dim, heads, width, active, "large KDA decay");
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

    failures += expect_shape(records.conv, 256, 4, 15, 1, "conv plane");
    failures += expect_shape(records.key, 128, 2, 4, 15, "key plane");
    failures += expect_shape(records.value, 128, 6, 4, 15, "value plane");
    failures += expect_shape(records.gate, 2, 6, 4, 15, "gate plane");
    failures += expect(records.conv.dtype == ninfer::DType::BF16, "conv dtype differs");
    failures += expect(records.key.dtype == ninfer::DType::BF16, "key dtype differs");
    failures += expect(records.value.dtype == ninfer::DType::BF16, "value dtype differs");
    failures += expect(records.gate.dtype == ninfer::DType::FP32, "gate dtype differs");
    failures += expect(reinterpret_cast<std::uintptr_t>(records.conv.data) % 256 == 0,
                       "conv plane is not aligned");
    failures += expect(reinterpret_cast<std::uintptr_t>(records.key.data) % 256 == 0,
                       "key plane is not aligned");
    failures += expect(reinterpret_cast<std::uintptr_t>(records.value.data) % 256 == 0,
                       "value plane is not aligned");
    failures += expect(reinterpret_cast<std::uintptr_t>(records.gate.data) % 256 == 0,
                       "gate plane is not aligned");
    failures += expect_size(layout.payload_bytes(),
                            records.conv.bytes() + records.key.bytes() + records.value.bytes() +
                                records.gate.bytes(),
                            "record payload bytes");

    const auto layer2 = records.layer(2, 3);
    failures += expect_shape(layer2.conv, 256, 4, 3, 1, "layer conv slice");
    failures += expect_shape(layer2.key, 128, 2, 4, 3, "layer key slice");
    failures += expect_shape(layer2.value, 128, 6, 4, 3, "layer value slice");
    failures += expect_shape(layer2.gate, 2, 6, 4, 3, "layer gate slice");
    failures += expect(
        static_cast<std::byte*>(layer2.conv.data) - static_cast<std::byte*>(records.conv.data) ==
            static_cast<std::ptrdiff_t>(2 * spec.record_capacity * records.conv.nb[2]),
        "layer conv slice offset differs");
    failures += expect(
        static_cast<std::byte*>(layer2.key.data) - static_cast<std::byte*>(records.key.data) ==
            static_cast<std::ptrdiff_t>(2 * spec.record_capacity * records.key.nb[3]),
        "layer key slice offset differs");
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
                            85819392, "48-layer T6 capacity");
    failures += expect_size(record_bytes({.layers          = 30,
                                          .record_capacity = 8,
                                          .width           = 6,
                                          .conv_channels   = 8192,
                                          .qk_heads        = 16,
                                          .value_heads     = 32,
                                          .key_dim         = 128,
                                          .value_dim       = 128}),
                            41656320, "30-layer T6 capacity");
    failures += expect_size(record_bytes({.layers          = 30,
                                          .record_capacity = 8,
                                          .width           = 16,
                                          .conv_channels   = 8192,
                                          .qk_heads        = 16,
                                          .value_heads     = 32,
                                          .key_dim         = 128,
                                          .value_dim       = 128}),
                            111083520, "30-layer T16 capacity");

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
