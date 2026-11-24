#include "core/arena.h"
#include "core/buffer.h"
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
    ninfer::DeviceBuffer storage(device.device(), sizeof(std::uint32_t));
    ninfer::PinnedHostBuffer host(sizeof(std::uint32_t));
    ninfer::CudaCompletionEvent ready(device);
    ninfer::DecodeGraphDefinition definitions[2];
    for (int i = 0; i < 2; ++i) {
        definitions[i].capture(device.stream(), [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), i + 1, sizeof(std::uint32_t), device.stream()));
            CUDA_CHECK(cudaMemcpyAsync(host.data(), storage.data(), sizeof(std::uint32_t),
                                       cudaMemcpyDeviceToHost, device.stream()));
            ready.record_external(device.stream());
            // The handoff exposes the prefix, even though later graph nodes overwrite the source.
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), 0xff, sizeof(std::uint32_t), device.stream()));
        });
    }
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(definitions[0]);
    int failures = 0;
    for (int iteration = 0; iteration < 32; ++iteration) {
        const auto value = static_cast<std::uint32_t>(iteration % 2 + 1) * 0x01010101U;
        executable.update(definitions[iteration % 2]);
        *static_cast<std::uint32_t*>(host.data()) = 0;
        executable.launch(device.stream());
        ready.synchronize();
        if (*static_cast<std::uint32_t*>(host.data()) != value) {
            std::cerr << "external event exposed stale data after replay/update\n";
            ++failures;
        }
        device.synchronize();
        failures += expect_value(storage.data(), 0xffffffffU, "graph suffix");
    }
    return failures;
}

int check_child_graph(ninfer::DeviceContext& device) {
    ninfer::DeviceBuffer source(device.device(), sizeof(std::uint32_t));
    ninfer::DeviceBuffer destination(device.device(), sizeof(std::uint32_t));
    ninfer::DecodeGraphDefinition children[2], parent;
    for (int i = 0; i < 2; ++i)
        children[i].capture(device.stream(), [&] {
            CUDA_CHECK(cudaMemsetAsync(source.data(), i + 1, source.size(), device.stream()));
        });
    ninfer::DecodeGraphChild child;
    ninfer::DecodeGraphSection optional;
    parent.capture(device.stream(), [&] {
        CUDA_CHECK(cudaMemsetAsync(source.data(), 0xff, source.size(), device.stream()));
        child = children[0].append_to(device.stream());
        optional.capture(device.stream(), [&] {
            CUDA_CHECK(cudaMemsetAsync(source.data(), 3, source.size(), device.stream()));
        });
        CUDA_CHECK(cudaMemcpyAsync(destination.data(), source.data(), source.size(),
                                   cudaMemcpyDeviceToDevice, device.stream()));
    });
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(parent);
    int failures = 0;
    for (int i = 0; i < 8; ++i) {
        executable.update_child(child, children[i % 2]);
        const bool enabled = (i / 2) % 2 == 0;
        optional.set_enabled(executable, enabled);
        executable.launch(device.stream());
        device.synchronize();
        failures += expect_value(destination.data(), (enabled ? 3 : i % 2 + 1) * 0x01010101U,
                                 "child update, optional section and capture dependencies");
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
        ninfer::DeviceBuffer storage(device.device(), sizeof(std::uint32_t));
        CUDA_CHECK(cudaMemsetAsync(storage.data(), 0x33, sizeof(std::uint32_t), device.stream()));
        device.synchronize();

        ninfer::DecodeGraphDefinition first;
        first.capture(device.stream(), [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), 0x11, sizeof(std::uint32_t), device.stream()));
        });
        ninfer::DecodeGraphDefinition second;
        second.capture(device.stream(), [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), 0x22, sizeof(std::uint32_t), device.stream()));
        });

        int failures = 0;
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(first);
        executable.upload(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x33333333U, "initial graph upload");

        executable.launch(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x11111111U, "first graph launch");

        executable.update(second);
        executable.upload(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x11111111U, "updated graph upload");

        executable.launch(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x22222222U, "updated graph launch");
        try {
            ninfer::DecodeGraphDefinition abandoned;
            abandoned.capture(device.stream(), [&] {
                CUDA_CHECK(
                    cudaMemsetAsync(storage.data(), 0x44, sizeof(std::uint32_t), device.stream()));
                throw ninfer::DeviceAllocationError("startup test");
            });
            ++failures;
        } catch (const ninfer::DeviceAllocationError&) {}
        ninfer::DecodeGraphDefinition recovered;
        recovered.capture(device.stream(), [&] {
            CUDA_CHECK(
                cudaMemsetAsync(storage.data(), 0x55, sizeof(std::uint32_t), device.stream()));
        });
        executable.instantiate(recovered);
        executable.launch(device.stream());
        device.synchronize();
        failures += expect_value(storage.data(), 0x55555555U, "capture after failed preparation");
        failures += check_external_event(device);
        failures += check_child_graph(device);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
