#include "core/device.h"
#include "ninfer/ops/position.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int fill_case(std::int32_t count, std::int32_t start) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) { expected[static_cast<std::size_t>(i)] = start + i; }

    GuardedDeviceBuffer output(static_cast<std::size_t>(count) * sizeof(std::int32_t));
    output.fill(0xcd);
    Tensor output_tensor(output.data(), DType::I32, {count});
    ops::fill_i32_positions(output_tensor, start, nullptr);
    cuda_synchronize();

    const std::string label =
        "fill_i32_positions T=" + std::to_string(count) + " start=" + std::to_string(start);
    int failures = verify_exact(
        label.c_str(), from_device<std::int32_t>(output.data(), expected.size()), expected);
    failures += output.verify_guards(label.c_str());
    return failures;
}

int offset_case(std::int32_t count, std::int32_t delta_value, bool in_place) {
    std::vector<std::int32_t> source(static_cast<std::size_t>(count));
    std::vector<std::int32_t> expected(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) {
        source[static_cast<std::size_t>(i)]   = 131072 + 3 * i + (i % 5);
        expected[static_cast<std::size_t>(i)] = source[static_cast<std::size_t>(i)] + delta_value;
    }
    const std::vector<std::int32_t> delta{delta_value};

    GuardedDeviceBuffer device_source(source.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_delta(sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(source.size() * sizeof(std::int32_t));
    device_source.copy_from_host(source.data(), source.size() * sizeof(std::int32_t));
    device_delta.copy_from_host(delta.data(), sizeof(std::int32_t));
    device_output.fill(0xcd);

    Tensor source_tensor(device_source.data(), DType::I32, {count});
    Tensor delta_tensor(device_delta.data(), DType::I32, {1});
    Tensor output_tensor(device_output.data(), DType::I32, {count});
    Tensor& destination = in_place ? source_tensor : output_tensor;
    ops::offset_i32_positions(source_tensor, delta_tensor, destination, nullptr);
    cuda_synchronize();

    const std::string label = "offset_i32_positions T=" + std::to_string(count) +
                              (in_place ? " in-place" : " out-of-place");
    int failures =
        verify_exact(label.c_str(),
                     from_device<std::int32_t>(
                         in_place ? device_source.data() : device_output.data(), expected.size()),
                     expected);
    if (!in_place) {
        failures +=
            verify_exact((label + " preserves source").c_str(),
                         from_device<std::int32_t>(device_source.data(), source.size()), source);
    }
    failures += verify_exact((label + " preserves delta").c_str(),
                             from_device<std::int32_t>(device_delta.data(), delta.size()), delta);
    failures += device_source.verify_guards((label + " source").c_str());
    failures += device_delta.verify_guards((label + " delta").c_str());
    if (!in_place) { failures += device_output.verify_guards((label + " destination").c_str()); }
    return failures;
}

int lane_offset_case(int width, int batch, int axes, bool in_place) {
    const int lane_size = width * axes;
    std::vector<std::int32_t> source(lane_size * batch), delta(batch), expected(source.size());
    for (int b = 0; b < batch; ++b) {
        delta[b] = 512 * b - 37;
        for (int i = 0; i < lane_size; ++i) source[b * lane_size + i] = 131072 + 7 * b + 3 * i;
    }
    GuardedDeviceBuffer input(source.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer offsets(delta.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer output(input.bytes());
    input.copy_from_host(source.data(), input.bytes());
    offsets.copy_from_host(delta.data(), offsets.bytes());
    output.fill(0xcd);
    auto* destination = in_place ? input.data() : output.data();
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_synchronize();
    const auto launch = [&] {
        for (int b = 0; b < batch; ++b) {
            Tensor src(static_cast<std::int32_t*>(input.data()) + b * lane_size, DType::I32, {lane_size});
            Tensor dst(static_cast<std::int32_t*>(destination) + b * lane_size, DType::I32, {lane_size});
            Tensor d(static_cast<std::int32_t*>(offsets.data()) + b, DType::I32, {1});
            ops::offset_i32_positions(src, d, dst, stream);
        }
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (width == 16 && batch == 8) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            for (auto& value : delta) value += 7;
            CUDA_CHECK(cudaMemcpyAsync(input.data(), source.data(), input.bytes(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(offsets.data(), delta.data(), offsets.bytes(), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < lane_size; ++i)
            expected[b * lane_size + i] = source[b * lane_size + i] + delta[b];
    int failures = verify_exact("lane offsets", from_device<std::int32_t>(destination, expected.size()), expected);
    if (!in_place)
        failures += verify_exact("lane source unchanged", from_device<std::int32_t>(input.data(), source.size()), source);
    failures += verify_exact("lane delta unchanged", from_device<std::int32_t>(offsets.data(), delta.size()), delta);
    failures += input.verify_guards("lane source");
    failures += output.verify_guards("lane destination");
    failures += offsets.verify_guards("lane deltas");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    for (int width = 2; width <= 16; ++width)
        for (int batch : {1, 8})
            for (int axes : {1, 3})
                for (bool in_place : {false, true})
                    failures += lane_offset_case(width, batch, axes, in_place);
    failures += fill_case(1, 0);
    failures += fill_case(6, 262144);
    failures += fill_case(1024, 131072);
    failures += offset_case(1, -17, false);
    failures += offset_case(6, 31, true);
    failures += offset_case(1024, -257, false);
    std::cout << (failures ? "FAIL" : "OK") << " position\n";
    return failures ? 1 : 0;
}
