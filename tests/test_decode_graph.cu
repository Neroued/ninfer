#include "core/arena.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
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

__global__ void add_one(unsigned* sum) {
    if (threadIdx.x == 0) atomicAdd(sum, 1U);
}

__global__ void append_digit(unsigned* value, unsigned digit) {
    if (blockIdx.x == 0) *value = *value * 10 + digit;
    atomicAdd(value + 1, 1U);
}

int check_kernel_slice(ninfer::DeviceContext& device, unsigned count) {
    const auto stream    = device.stream();
    const auto secondary = device.transfer_stream();
    ninfer::DeviceBuffer storage(device.device(), 4 * sizeof(unsigned));
    auto* values = static_cast<unsigned*>(storage.data());
    ninfer::CudaCompletionEvent fork(device), join(device), external(device);
    external.record(secondary);
    ninfer::DecodeGraphKernelPatch patch;
    patch.capture(stream, [&] {
        for (unsigned i = 0; i < count; ++i) append_digit<<<i + 2, 1, 0, stream>>>(values, 7 + i);
    });
    ninfer::DecodeGraphKernelSlice slice;
    ninfer::DecodeGraphDefinition parent;
    parent.capture(stream, [&] {
        if (count > 1) {
            fork.record(stream);
            fork.wait(secondary);
            add_one<<<1, 1, 0, secondary>>>(values + 2);
            join.record(secondary);
            external.wait_external(stream);
            add_one<<<1, 1, 0, stream>>>(values + 3);
            join.wait(stream);
            cudaStreamCaptureStatus status;
            cudaGraph_t graph        = nullptr;
            std::size_t dependencies = 0, roots = 0;
            CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph, nullptr, nullptr,
                                                &dependencies));
            CUDA_CHECK(cudaGraphGetRootNodes(graph, nullptr, &roots));
            if (status != cudaStreamCaptureStatusActive || dependencies != 2 || roots != 2)
                throw std::runtime_error("expected a capture with two joined root branches");
        }
        slice.capture(stream, [&] {
            for (unsigned i = 0; i < count; ++i) append_digit<<<1, 1, 0, stream>>>(values, i + 1);
        });
    });
    if (slice.size() != count || patch.params().size() != count)
        throw std::runtime_error("kernel chain length changed");
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(parent);
    int failures = 0;
    for (bool updated : {false, true}) {
        CUDA_CHECK(cudaMemsetAsync(values, 0, storage.size(), stream));
        if (updated) executable.update_slice(slice, patch);
        executable.launch(stream);
        device.synchronize();
        unsigned digits = 0, blocks = 0;
        for (unsigned i = 0; i < count; ++i) {
            digits = digits * 10 + (updated ? 7 + i : i + 1);
            blocks += updated ? i + 2 : 1;
        }
        failures += expect_value(values, digits, "patched launch order and scalar arguments");
        failures += expect_value(values + 1, blocks, "patched launch grid");
        if (count > 1) {
            failures += expect_value(values + 2, 1U, "secondary branch survives slice patch");
            failures += expect_value(values + 3, 1U, "primary branch survives slice patch");
        }
    }
    return failures;
}

int check_invalid_kernel_patch(ninfer::DeviceContext& device) {
    const auto stream = device.stream();
    ninfer::DeviceBuffer storage(device.device(), sizeof(unsigned));
    auto* value = static_cast<unsigned*>(storage.data());
    ninfer::DecodeGraphKernelPatch patch;
    patch.capture(stream, [&] { add_one<<<1, 1, 0, stream>>>(value); });
    for (const std::function<void()>& body : std::array<std::function<void()>, 3>{
             [] {}, [&] { CUDA_CHECK(cudaMemsetAsync(value, 0, storage.size(), stream)); },
             [&] {
                 void* args[]{&value};
                 CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(add_one),
                                                        dim3(1), dim3(1), args, 0, stream));
             }}) {
        try {
            patch.capture(stream, body);
        } catch (const std::logic_error&) {
            if (!patch.ready() && patch.params().empty()) continue;
        }
        std::cerr << "invalid kernel patch survived capture validation\n";
        return 1;
    }
    return 0;
}

int check_cooperative_section(ninfer::DeviceContext& device) {
    ninfer::DeviceBuffer storage(device.device(), 3 * sizeof(unsigned));
    auto* sum = static_cast<unsigned*>(storage.data());
    CUDA_CHECK(cudaMemsetAsync(sum, 0, storage.size(), device.stream()));
    const auto stream = device.stream();
    ninfer::DecodeGraphKernelPatch one, two;
    one.capture(stream, [&] { append_digit<<<2, 1, 0, stream>>>(sum + 1, 3U); });
    two.capture(stream, [&] { append_digit<<<4, 1, 0, stream>>>(sum + 1, 5U); });
    ninfer::DecodeGraphKernelSlice slice;
    ninfer::DecodeGraphSection optional;
    ninfer::DecodeGraphDefinition round;
    round.capture(device.stream(), [&] {
        optional.capture(device.stream(), [&] {
            void* args[]{&sum};
            CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(add_one), dim3(2),
                                                   dim3(512), args, 0, device.stream()));
            slice.capture(stream, [&] { append_digit<<<1, 1, 0, stream>>>(sum + 1, 1U); });
        });
    });
    ninfer::DecodeGraphExecutable graph;
    graph.instantiate(round);
    int failures = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const bool active = i == 0 || i == 3;
        optional.set_enabled(graph, active);
        if (active) graph.update_slice(slice, i ? two : one);
        graph.launch(device.stream());
        device.synchronize();
        failures += expect_value(sum, i == 3 ? 4U : 2U, "cooperative graph section");
        failures += expect_value(sum + 1, i == 3 ? 35U : 3U, "gated graph slice parameters");
        failures += expect_value(sum + 2, i == 3 ? 6U : 2U, "gated graph slice grid");
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
        failures += check_kernel_slice(device, 1);
        failures += check_kernel_slice(device, 3);
        failures += check_invalid_kernel_patch(device);
        failures += check_cooperative_section(device);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
