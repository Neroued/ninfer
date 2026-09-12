#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

using AlignedBacking = std::unique_ptr<void, decltype(&std::free)>;

AlignedBacking make_backing(std::size_t bytes) {
    void* data = std::aligned_alloc(256, bytes);
    if (data == nullptr) { throw std::bad_alloc(); }
    return AlignedBacking(data, &std::free);
}

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

} // namespace

int main() {
    int failures = 0;

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
    auto backing            = make_backing(bytes);
    const ninfer::GdnReplayRecords records({backing.get(), bytes}, layout);

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
                                                            .record_capacity = 9,
                                                            .width           = 2,
                                                            .conv_channels   = 1,
                                                            .qk_heads        = 1,
                                                            .value_heads     = 1,
                                                            .key_dim         = 1,
                                                            .value_dim       = 1});
        },
        "record capacity above eight");

    ninfer::LayoutBuilder state_builder;
    const auto state_layout = ninfer::plan_linear_attention_state_pool(
        state_builder, {.layers         = 3,
                        .conv_channels  = 256,
                        .conv_width     = 3,
                        .value_heads    = 6,
                        .value_head_dim = 128,
                        .key_head_dim   = 128,
                        .slot_count     = 7,
                        .conv_dtype     = ninfer::DType::BF16});
    const std::size_t state_bytes = state_builder.finish(256);
    auto state_backing            = make_backing(state_bytes);
    ninfer::LinearAttentionStatePool state({state_backing.get(), state_bytes}, state_layout);
    const auto all = state.all_layers_view();
    failures +=
        expect(all.conv_layer0.data == state.layer_view(0).conv.data, "conv layer-0 base differs");
    failures += expect(all.recurrent_layer0.data == state.layer_view(0).recurrent.data,
                       "recurrent layer-0 base differs");
    failures += expect_size(static_cast<std::size_t>(all.conv_layer_stride_bytes),
                            static_cast<std::byte*>(state.layer_view(1).conv.data) -
                                static_cast<std::byte*>(state.layer_view(0).conv.data),
                            "conv layer stride");
    failures += expect_size(static_cast<std::size_t>(all.recurrent_layer_stride_bytes),
                            static_cast<std::byte*>(state.layer_view(1).recurrent.data) -
                                static_cast<std::byte*>(state.layer_view(0).recurrent.data),
                            "recurrent layer stride");
    failures +=
        expect_size(static_cast<std::size_t>(all.spec.slot_count), 7, "all-layer slot count");
    failures += expect_size(static_cast<std::size_t>(records.spec.record_capacity), 5,
                            "independent record capacity");

    return failures == 0 ? 0 : fail("GDN replay record storage test failed");
}
