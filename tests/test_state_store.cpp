#include "core/device.h"
#include "core/linear_attention_state.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

struct PlannedState {
    ninfer::LinearAttentionStatePoolLayout layout;
    std::size_t bytes = 0;
};

PlannedState plan_state(std::uint32_t layers, std::int32_t conv_channels, std::int32_t conv_width,
                        std::int32_t value_heads, std::int32_t value_head_dim,
                        std::int32_t key_head_dim, std::int32_t slot_count = 1,
                        ninfer::DType conv_dtype = ninfer::DType::BF16) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_linear_attention_state_pool(
        builder, ninfer::LinearAttentionStatePoolSpec{.layers         = layers,
                                                      .conv_channels  = conv_channels,
                                                      .conv_width     = conv_width,
                                                      .value_heads    = value_heads,
                                                      .value_head_dim = value_head_dim,
                                                      .key_head_dim   = key_head_dim,
                                                      .slot_count     = slot_count,
                                                      .conv_dtype     = conv_dtype});
    return PlannedState{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& tensor, const std::int32_t (&expected)[4],
                const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got "
                      << tensor.ne[i] << '\n';
        }
    }
    return failures;
}

int expect_device_byte(const ninfer::Tensor& tensor, unsigned char expected, const char* label) {
    std::vector<unsigned char> host(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    for (unsigned char value : host) {
        if (value != expected) {
            std::cerr << label << " expected byte 0x" << std::hex << static_cast<int>(expected)
                      << ", got 0x" << static_cast<int>(value) << std::dec << '\n';
            return 1;
        }
    }
    return 0;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);
    auto state_plan = plan_state(3, 10, 3, 4, 5, 6);
    ninfer::DeviceArena state_arena(state_plan.bytes);
    CUDA_CHECK(cudaMemset(state_arena.base(), 0x4a, state_arena.capacity()));
    ninfer::LinearAttentionStatePool state({state_arena.base(), state_arena.capacity()},
                                           state_plan.layout);

    failures += expect_size(state.layer_count(), 3, "state.layer_count");
    failures += expect_size(state.slot_count(), 1, "state.slot_count");
    failures += expect_size(state.spec().conv_width, 3, "state.conv_width");
    for (std::size_t layer = 0; layer < state.layer_count(); ++layer) {
        const auto view = state.layer_view(static_cast<std::uint32_t>(layer));
        failures += check_shape(view.conv, {10, 3, 1, 1}, "state.conv");
        failures += check_shape(view.recurrent, {6, 5, 4, 1}, "state.recurrent");
        failures += check_shape(state.conv_slot(static_cast<std::uint32_t>(layer), 0),
                                {10, 3, 1, 1}, "state.conv_slot");
        failures += check_shape(state.recurrent_slot(static_cast<std::uint32_t>(layer), 0),
                                {6, 5, 4, 1}, "state.recurrent_slot");
        if (view.conv.dtype != ninfer::DType::BF16) {
            ++failures;
            std::cerr << "conv dtype is not BF16\n";
        }
        if (view.recurrent.dtype != ninfer::DType::FP32) {
            ++failures;
            std::cerr << "recurrent dtype is not FP32\n";
        }
        if (view.conv.data == view.recurrent.data) {
            ++failures;
            std::cerr << "conv/recurrent alias for layer " << layer << '\n';
        }
        failures += expect_device_byte(view.conv, 0x4a, "constructor-mutated conv");
        failures += expect_device_byte(view.recurrent, 0x4a, "constructor-mutated recurrent");
    }
    if (state.layer_view(0).conv.data == state.layer_view(1).conv.data ||
        state.layer_view(0).recurrent.data == state.layer_view(1).recurrent.data) {
        ++failures;
        std::cerr << "state layers alias\n";
    }

    state.zero_slot(0, ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(state.layer_view(0).conv, 0, "zeroed conv");
    failures += expect_device_byte(state.layer_view(1).recurrent, 0, "zeroed recurrent");

    auto slotted_plan = plan_state(2, 10, 3, 4, 5, 6, 3);
    ninfer::DeviceArena slotted_arena(slotted_plan.bytes);
    CUDA_CHECK(cudaMemset(slotted_arena.base(), 0, slotted_arena.capacity()));
    ninfer::LinearAttentionStatePool slotted({slotted_arena.base(), slotted_arena.capacity()},
                                             slotted_plan.layout);
    failures += expect_size(slotted.slot_count(), 3, "slotted.slot_count");
    failures += check_shape(slotted.layer_view(0).conv, {10, 3, 3, 1}, "slotted.conv");
    failures += check_shape(slotted.layer_view(0).recurrent, {6, 5, 4, 3}, "slotted.recurrent");
    failures += check_shape(slotted.conv_slot(0, 2), {10, 3, 1, 1}, "slotted.conv_slot");
    failures += check_shape(slotted.recurrent_slot(0, 2), {6, 5, 4, 1}, "slotted.recurrent_slot");

    ninfer::Tensor conv0             = slotted.conv_slot(0, 0);
    ninfer::Tensor conv1             = slotted.conv_slot(0, 1);
    ninfer::Tensor recurrent0        = slotted.recurrent_slot(0, 0);
    ninfer::Tensor recurrent1        = slotted.recurrent_slot(0, 1);
    ninfer::Tensor conv1_layer1      = slotted.conv_slot(1, 1);
    ninfer::Tensor recurrent1_layer1 = slotted.recurrent_slot(1, 1);
    CUDA_CHECK(cudaMemset(conv0.data, 0x7a, conv0.bytes()));
    CUDA_CHECK(cudaMemset(conv1.data, 0x6b, conv1.bytes()));
    CUDA_CHECK(cudaMemset(recurrent0.data, 0x5c, recurrent0.bytes()));
    CUDA_CHECK(cudaMemset(recurrent1.data, 0x4d, recurrent1.bytes()));
    CUDA_CHECK(cudaMemset(conv1_layer1.data, 0x3c, conv1_layer1.bytes()));
    CUDA_CHECK(cudaMemset(recurrent1_layer1.data, 0x2d, recurrent1_layer1.bytes()));

    slotted.copy_slot(1, 2, ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(slotted.conv_slot(0, 2), 0x6b, "copied conv slot");
    failures += expect_device_byte(slotted.recurrent_slot(0, 2), 0x4d, "copied recurrent slot");
    failures += expect_device_byte(slotted.conv_slot(1, 2), 0x3c, "copied conv layer1");
    failures += expect_device_byte(slotted.recurrent_slot(1, 2), 0x2d, "copied recurrent layer1");

    slotted.zero_slot(0, ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(slotted.conv_slot(0, 0), 0, "zeroed conv slot0");
    failures += expect_device_byte(slotted.recurrent_slot(0, 0), 0, "zeroed recurrent slot0");
    failures += expect_device_byte(slotted.conv_slot(0, 1), 0x6b, "zero kept conv slot1");
    failures += expect_device_byte(slotted.recurrent_slot(0, 1), 0x4d, "zero kept recurrent slot1");
    failures += expect_device_byte(slotted.conv_slot(0, 2), 0x6b, "zero kept conv slot2");
    failures += expect_device_byte(slotted.recurrent_slot(0, 2), 0x4d, "zero kept recurrent slot2");

    slotted.zero_all(ctx.stream);
    ctx.synchronize();
    failures += expect_device_byte(slotted.layer_view(1).conv, 0, "zeroed all conv");
    failures += expect_device_byte(slotted.layer_view(1).recurrent, 0, "zeroed all recurrent");

    auto fp32_conv_plan = plan_state(1, 7, 2, 2, 3, 4, 2, ninfer::DType::FP32);
    ninfer::DeviceArena fp32_conv_arena(fp32_conv_plan.bytes);
    ninfer::LinearAttentionStatePool fp32_conv({fp32_conv_arena.base(), fp32_conv_arena.capacity()},
                                               fp32_conv_plan.layout);
    if (fp32_conv.layer_view(0).conv.dtype != ninfer::DType::FP32) {
        ++failures;
        std::cerr << "FP32 conv geometry did not retain its dtype\n";
    }
    failures += check_shape(fp32_conv.layer_view(0).conv, {7, 2, 2, 1}, "fp32_conv.conv");
    failures += check_shape(fp32_conv.layer_view(0).recurrent, {4, 3, 2, 2}, "fp32_conv.recurrent");

    return failures == 0 ? 0 : fail("linear attention state pool test failed");
}
