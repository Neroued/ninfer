#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <exception>
#include <iostream>

namespace {

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_value(void* device, std::uint32_t expected, const char* label) {
    std::uint32_t actual  = 0;
    const cudaError_t err = cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " expected 0x" << std::hex << expected << ", got 0x" << actual << std::dec
              << '\n';
    return 1;
}

int check_external_event(ninfer::DeviceContext& device) {
    ninfer::DeviceArena storage(sizeof(std::uint32_t));
    ninfer::PinnedHostBuffer host(sizeof(std::uint32_t));
    ninfer::CudaCompletionEvent ready(device);
    ninfer::DecodeGraphDefinition definitions[2];
    for (int i = 0; i < 2; ++i) {
        definitions[i].capture(device.stream, [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.base(), i + 1, sizeof(std::uint32_t), device.stream));
            CUDA_CHECK(cudaMemcpyAsync(host.data(), storage.base(), sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, device.stream));
            ready.record_external(device.stream);
            // The handoff exposes the prefix, even though later graph nodes overwrite the source.
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0xff, sizeof(std::uint32_t), device.stream));
        });
    }
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(definitions[0]);
    int failures = 0;
    for (int iteration = 0; iteration < 32; ++iteration) {
        const auto value = static_cast<std::uint32_t>(iteration % 2 + 1) * 0x01010101U;
        executable.update(definitions[iteration % 2]);
        *static_cast<std::uint32_t*>(host.data()) = 0;
        executable.launch(device.stream);
        ready.synchronize();
        if (*static_cast<std::uint32_t*>(host.data()) != value) {
            std::cerr << "external event exposed stale data after replay/update\n";
            ++failures;
        }
        device.synchronize();
        failures += expect_value(storage.base(), 0xffffffffU, "graph suffix");
    }
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        ninfer::DeviceArena storage(sizeof(std::uint32_t));
        CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x33, sizeof(std::uint32_t), device.stream));
        device.synchronize();

        ninfer::DecodeGraphDefinition first;
        first.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x11, sizeof(std::uint32_t), device.stream));
        });
        ninfer::DecodeGraphDefinition second;
        second.capture(device.stream, [&] {
            CUDA_CHECK(cudaMemsetAsync(storage.base(), 0x22, sizeof(std::uint32_t), device.stream));
        });

        int failures = 0;
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(first);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x33333333U, "initial graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "first graph launch");

        executable.update(second);
        executable.upload(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x11111111U, "updated graph upload");

        executable.launch(device.stream);
        device.synchronize();
        failures += expect_value(storage.base(), 0x22222222U, "updated graph launch");
        failures += check_external_event(device);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
