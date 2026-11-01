#include "core/buffer.h"
#include "core/device.h"
#include "core/linear_attention_state.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

namespace {

struct PlannedState {
    ninfer::LinearAttentionStateLayout layout;
    ninfer::StateImageLayout images;
};

PlannedState plan_state(std::uint32_t layers, std::int32_t conv_channels, std::int32_t conv_width,
                        std::int32_t value_heads, std::int32_t value_head_dim,
                        std::int32_t key_head_dim, ninfer::DType conv_dtype = ninfer::DType::BF16) {
    ninfer::StateImageBuilder images;
    auto layout = ninfer::plan_linear_attention_state(images, {.layers         = layers,
                                                               .conv_channels  = conv_channels,
                                                               .conv_width     = conv_width,
                                                               .value_heads    = value_heads,
                                                               .value_head_dim = value_head_dim,
                                                               .key_head_dim   = key_head_dim,
                                                               .conv_dtype     = conv_dtype});
    return {std::move(layout), images.finish()};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) return 0;
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& tensor, const std::int32_t (&expected)[4],
                const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] == expected[i]) continue;
        ++failures;
        std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got "
                  << tensor.ne[i] << '\n';
    }
    return failures;
}

int expect_device_byte(const ninfer::Tensor& tensor, unsigned char expected, const char* label) {
    std::vector<unsigned char> host(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    for (unsigned char value : host) {
        if (value == expected) continue;
        std::cerr << label << " expected byte " << int(expected) << ", got " << int(value) << '\n';
        return 1;
    }
    return 0;
}

int mixed_image_case(ninfer::DeviceContext& device) {
    ninfer::StateImageBuilder builder;
    const auto tail        = builder.add(ninfer::DType::BF16, {128, 3}, 1, "QSA raw tail");
    const auto coordinates = builder.add(ninfer::DType::I32, {3}, 1, "QSA tail coordinates");
    const auto ple         = builder.add(ninfer::DType::BF16, {10240, 9}, 2, "PLE histories");
    const auto hidden      = builder.add(ninfer::DType::BF16, {7}, 1, "continuation hidden");
    const auto tail_group  = builder.group(tail, coordinates);
    const auto layout      = builder.finish();
    const std::array<std::int64_t, 3> offsets{256, std::int64_t(layout.image_bytes + 768),
                                              std::int64_t(3 * layout.image_bytes + 1024)};
    ninfer::DeviceBuffer memory(device.device(), offsets.back() + layout.image_bytes);
    ninfer::StateImageStorage images(memory.span(), layout);
    for (auto offset : offsets) images.clear(offset, device.execution_view());
    const auto fill = [&](ninfer::StateField field, std::uint32_t layer, int value) {
        const auto tensor = images.tensor(field, offsets[0], layer);
        CUDA_CHECK(cudaMemsetAsync(tensor.data, value, tensor.bytes(), device.stream()));
    };
    fill(tail, 0, 0x11);
    fill(coordinates, 0, 0x22);
    fill(ple, 0, 0x33);
    fill(ple, 1, 0x44);
    fill(hidden, 0, 0x55);
    images.copy(offsets[0], offsets[2], device.execution_view());
    images.copy_group(tail_group, offsets[0], offsets[1], device.execution_view());
    device.synchronize();
    int failures = 0;
    failures += expect_device_byte(images.tensor(tail, offsets[1]), 0x11, "tail group");
    failures += expect_device_byte(images.tensor(coordinates, offsets[1]), 0x22, "coordinates");
    failures += expect_device_byte(images.tensor(ple, offsets[1], 0), 0, "local copy isolation");
    failures += expect_device_byte(images.tensor(ple, offsets[2], 0), 0x33, "PLE first layer");
    failures += expect_device_byte(images.tensor(ple, offsets[2], 1), 0x44, "PLE next layer");
    failures += expect_device_byte(images.tensor(hidden, offsets[2]), 0x55, "hidden copy");
    failures +=
        expect_size(images.tensor(coordinates, offsets[0]).bytes(), 3 * 4, "one image field bytes");
    bool rejected = false;
    try {
        (void)images.image(offsets.back() + 256);
    } catch (const std::exception&) { rejected = true; }
    if (!rejected) failures += fail("record extending past arena was accepted");
    rejected = false;
    try {
        ninfer::StateImageStorage short_image(memory.span().subspan(0, layout.image_bytes - 1),
                                              layout);
    } catch (const std::exception&) { rejected = true; }
    if (!rejected) failures += fail("arena smaller than one image was accepted");
    return failures;
}

int recurrent_image_case(ninfer::DeviceContext& device) {
    auto plan = plan_state(3, 10, 3, 4, 5, 6);
    ninfer::DeviceBuffer backing(device.device(), plan.images.image_bytes);
    CUDA_CHECK(cudaMemset(backing.data(), 0x4a, backing.size()));
    ninfer::StateImageStorage images(backing.span(), plan.images);
    ninfer::LinearAttentionStateView state(images, plan.layout);
    int failures = expect_size(state.layer_count(), 3, "layer count");
    failures += expect_size(state.spec().conv_width, 3, "conv width");
    for (std::uint32_t layer = 0; layer < state.layer_count(); ++layer) {
        const auto conv = state.conv_tensor(layer, 0), recurrent = state.recurrent_tensor(layer, 0);
        failures += check_shape(conv, {10, 3, 1, 1}, "conv");
        failures += check_shape(recurrent, {6, 5, 4, 1}, "recurrent");
        if (conv.dtype != ninfer::DType::BF16 || recurrent.dtype != ninfer::DType::FP32)
            failures += fail("linear State field dtype changed");
        if (conv.data == recurrent.data) failures += fail("conv and recurrent fields alias");
        failures += expect_device_byte(conv, 0x4a, "constructor-mutated conv");
        failures += expect_device_byte(recurrent, 0x4a, "constructor-mutated recurrent");
    }
    if (state.conv_tensor(0, 0).data == state.conv_tensor(1, 0).data ||
        state.recurrent_tensor(0, 0).data == state.recurrent_tensor(1, 0).data)
        failures += fail("different layers alias");
    images.clear(0, device.execution_view());
    device.synchronize();
    failures += expect_device_byte(state.conv_tensor(0, 0), 0, "zeroed conv");
    failures += expect_device_byte(state.recurrent_tensor(1, 0), 0, "zeroed recurrent");
    return failures;
}

int scattered_records(ninfer::DeviceContext& device) {
    auto plan = plan_state(2, 10, 3, 4, 5, 6);
    const std::array<std::int64_t, 3> offsets{256, std::int64_t(plan.images.image_bytes * 2 + 768),
                                              std::int64_t(plan.images.image_bytes * 4 + 1024)};
    ninfer::DeviceBuffer backing(device.device(), offsets.back() + plan.images.image_bytes);
    CUDA_CHECK(cudaMemset(backing.data(), 0x42, backing.size()));
    ninfer::StateImageStorage images(backing.span(), plan.images);
    ninfer::LinearAttentionStateView state(images, plan.layout);
    for (auto offset : offsets) images.clear(offset, device.execution_view());
    const auto fill = [&](const ninfer::Tensor& tensor, int value) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, value, tensor.bytes(), device.stream()));
    };
    fill(state.conv_tensor(0, offsets[0]), 0x7a);
    fill(state.conv_tensor(0, offsets[1]), 0x6b);
    fill(state.recurrent_tensor(0, offsets[0]), 0x5c);
    fill(state.recurrent_tensor(0, offsets[1]), 0x4d);
    fill(state.conv_tensor(1, offsets[1]), 0x3c);
    fill(state.recurrent_tensor(1, offsets[1]), 0x2d);
    images.copy(offsets[1], offsets[2], device.execution_view());
    images.clear(offsets[0], device.execution_view());
    device.synchronize();
    int failures = 0;
    failures += expect_device_byte(state.conv_tensor(0, offsets[0]), 0, "cleared source");
    failures += expect_device_byte(state.recurrent_tensor(0, offsets[0]), 0, "cleared recurrent");
    for (auto offset : {offsets[1], offsets[2]}) {
        failures += expect_device_byte(state.conv_tensor(0, offset), 0x6b, "copied conv");
        failures += expect_device_byte(state.recurrent_tensor(0, offset), 0x4d, "copied recurrent");
        failures += expect_device_byte(state.conv_tensor(1, offset), 0x3c, "copied next conv");
        failures +=
            expect_device_byte(state.recurrent_tensor(1, offset), 0x2d, "copied next recurrent");
    }
    std::array<unsigned char, 256> unrelated{};
    CUDA_CHECK(
        cudaMemcpy(unrelated.data(), backing.data(), unrelated.size(), cudaMemcpyDeviceToHost));
    for (auto value : unrelated)
        if (value != 0x42) {
            failures += fail("record clear/copy modified an unowned arena range");
            break;
        }

    auto fp32 = plan_state(1, 7, 2, 2, 3, 4, ninfer::DType::FP32);
    ninfer::DeviceBuffer fp32_backing(device.device(), fp32.images.image_bytes);
    ninfer::StateImageStorage fp32_images(fp32_backing.span(), fp32.images);
    ninfer::LinearAttentionStateView fp32_state(fp32_images, fp32.layout);
    if (fp32_state.conv_tensor(0, 0).dtype != ninfer::DType::FP32)
        failures += fail("FP32 convolution State dtype changed");
    failures += check_shape(fp32_state.conv_tensor(0, 0), {7, 2, 1, 1}, "FP32 conv");
    failures += check_shape(fp32_state.recurrent_tensor(0, 0), {4, 3, 2, 1}, "FP32 recurrent");
    return failures;
}
} // namespace

int main() {
    try {
        int count        = 0;
        const auto error = cudaGetDeviceCount(&count);
        if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || count == 0)
            return 77;
        CUDA_CHECK(error);
        ninfer::DeviceContext device(0);
        const auto failures =
            mixed_image_case(device) + recurrent_image_case(device) + scattered_records(device);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
