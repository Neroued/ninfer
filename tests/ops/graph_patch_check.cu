#include "ops/graph_patch_check.h"

#include "core/buffer.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using ninfer::CudaExecutionView;
using ninfer::test::CapturedKernelSequence;
using ninfer::test::FlatGraphPatchHarness;

template <int Scale>
__global__ void mix_values(int* output, int scalar) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const int launch_value =
        scalar + 100 * static_cast<int>(gridDim.x) + 10 * static_cast<int>(blockDim.x);
    output[index] = Scale * output[index] + launch_value;
}

template <int Scale>
CapturedKernelSequence capture_sequence(CudaExecutionView execution, int* output, dim3 first_grid,
                                        dim3 first_block, int first_scalar, dim3 second_grid,
                                        dim3 second_block, int second_scalar) {
    return CapturedKernelSequence(execution, [&](CudaExecutionView view) {
        mix_values<Scale><<<first_grid, first_block, 0, view.stream()>>>(output, first_scalar);
        CUDA_CHECK(cudaGetLastError());
        mix_values<Scale><<<second_grid, second_block, 0, view.stream()>>>(output, second_scalar);
        CUDA_CHECK(cudaGetLastError());
    });
}

void check_graph_updates(ninfer::DeviceContext& device) {
    const auto execution = device.execution_view();
    constexpr std::array first_expected{414, 414, 144, 144, 0, 0, 0, 0};
    constexpr std::array second_expected{1041, 1041, 1041, 239, 239, 239, 0, 0};
    ninfer::DeviceBuffer storage(device.device(), sizeof(first_expected));
    auto* output      = static_cast<int*>(storage.data());
    const auto first  = capture_sequence<2>(execution, output, 1, 4, 4, 1, 2, 6);
    const auto second = capture_sequence<3>(execution, output, 2, 3, 9, 3, 1, 14);
    if (first.params().size() != 2 || second.params().size() != 2 ||
        first.params()[0].func == second.params()[0].func ||
        first.params()[0].gridDim.x == second.params()[0].gridDim.x ||
        first.params()[0].blockDim.x == second.params()[0].blockDim.x) {
        throw std::runtime_error(
            "the captured launches must exercise two distinct kernel profiles");
    }
    FlatGraphPatchHarness harness(first);

    const auto one_node = CapturedKernelSequence(execution, [&](CudaExecutionView view) {
        mix_values<2><<<1, 1, 0, view.stream()>>>(output, 0);
        CUDA_CHECK(cudaGetLastError());
    });
    bool rejected       = false;
    try {
        harness.patch_and_launch(one_node, execution);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("node-count mismatch was accepted");

    rejected = false;
    try {
        (void)CapturedKernelSequence(execution, [&](CudaExecutionView view) {
            CUDA_CHECK(cudaMemsetAsync(output, 0, storage.size(), view.stream()));
        });
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("non-kernel graph node was accepted");

    const auto check = [&](auto update, const CapturedKernelSequence& target,
                           const std::array<int, 8>& expected, std::string_view label) {
        CUDA_CHECK(cudaMemsetAsync(output, 0, storage.size(), execution.stream()));
        (harness.*update)(target, execution);
        device.synchronize();
        std::array<int, 8> actual{};
        storage.copy_to_host(actual.data(), sizeof(actual));
        if (actual != expected)
            throw std::runtime_error(std::string(label) + " produced wrong values");
    };

    check(&FlatGraphPatchHarness::patch_and_launch, first, first_expected, "initial profile");
    check(&FlatGraphPatchHarness::patch_and_launch, second, second_expected, "per-node patch");
    check(&FlatGraphPatchHarness::update_whole_graph_and_launch, first, first_expected,
          "whole-graph update to initial profile");
    check(&FlatGraphPatchHarness::update_whole_graph_and_launch, second, second_expected,
          "whole-graph update to new profile");
    check(&FlatGraphPatchHarness::patch_and_launch, first, first_expected,
          "patch after whole-graph update");
}

} // namespace

int main() {
    try {
        int count        = 0;
        const auto error = cudaGetDeviceCount(&count);
        if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
            (error == cudaSuccess && count == 0)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(error);
        ninfer::DeviceContext device(0);
        check_graph_updates(device);
    } catch (const std::exception& error) {
        std::cerr << "graph patch test failed: " << error.what() << '\n';
        return 1;
    }
}
