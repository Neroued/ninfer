#include "core/device.h"
#include "core/buffer.h"
#include "core/transfer.h"

#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_throws_device(int device_id) {
    try {
        ninfer::DeviceContext invalid(device_id);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const ninfer::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream() == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.transfer_stream() == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.compute_capability() <= 0) {
        std::cerr << label << " compute capability is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--invalid-sync") {
        try {
            ninfer::DeviceContext ctx(0);
        } catch (const std::invalid_argument& error) {
            return std::string_view(error.what()).find("NINFER_CUDA_SYNC") != std::string_view::npos
                       ? 0
                       : fail("invalid sync setting has no configuration diagnostic");
        }
        return fail("invalid sync setting did not fail before CUDA initialization");
    }
    const unsigned int expected_flags =
        argc == 2 ? static_cast<unsigned int>(std::stoul(argv[1])) : cudaDeviceScheduleSpin;
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
    unsigned int actual_flags = 0;
    CUDA_CHECK(cudaGetDeviceFlags(&actual_flags));
    if ((actual_flags & cudaDeviceScheduleMask) != expected_flags) {
        return fail("CUDA did not apply the requested synchronization schedule");
    }
    if (ctx.device().index() != 0) {
        ++failures;
        std::cerr << "ctx.device().index() expected 0, got " << ctx.device().index() << '\n';
    }
    failures += check_context(ctx, "ctx");
    // WSL can admit allocations larger than physical VRAM. Exceed addressable storage instead.
    constexpr auto impossible_bytes = std::numeric_limits<std::size_t>::max() / 2 + 1;
    try {
        ninfer::DeviceBuffer impossible(ctx.device(), impossible_bytes);
        failures += fail("oversized Device allocation did not fail");
    } catch (const ninfer::DeviceAllocationError& error) {
        if (error.requested_bytes() != impossible_bytes)
            failures += fail("Device allocation error lost the requested size");
    }
    if (cudaPeekAtLastError() != cudaSuccess)
        failures += fail("handled allocation failure left a CUDA error pending");
    int* device_value = nullptr;
    int* host_value   = nullptr;
    CUDA_CHECK(cudaMalloc(&device_value, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&host_value, sizeof(int)));
    *host_value = 0;
    CUDA_CHECK(cudaMemsetAsync(device_value, 0x5a, sizeof(int), ctx.stream()));
    CUDA_CHECK(cudaMemcpyAsync(host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost,
                               ctx.stream()));
    ctx.synchronize();
    const bool transfer_complete = *host_value == 0x5a5a5a5a;
    CUDA_CHECK(cudaFreeHost(host_value));
    CUDA_CHECK(cudaFree(device_value));
    if (!transfer_complete) {
        return fail("stream synchronization returned before transfer completed");
    }

    const cudaStream_t original_stream = ctx.stream();
    ninfer::DeviceContext moved(std::move(ctx));
    if (ctx.stream() != nullptr || ctx.transfer_stream() != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream() != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    {
        ninfer::PinnedHostBuffer input(64), output(128);
        auto* values = static_cast<std::int32_t*>(input.data());
        for (int i = 0; i < 16; ++i) values[i] = i + 1;
        ninfer::DeviceBuffer source(moved.device(), 64), destination(moved.device(), 128);
        const auto execution = moved.execution_view();
        ninfer::copy_bytes_async(source.span(), input.span(), execution);
        CUDA_CHECK(cudaMemsetAsync(destination.data(), 0x5a, destination.size(), moved.stream()));
        ninfer::copy_rows_async(destination.span().subspan(4), 32, source.span().subspan(8), 16, 8,
                                4, execution);
        ninfer::copy_bytes_async(output.span(), destination.span(), execution);
        moved.synchronize();
        const auto* copied = static_cast<const std::int32_t*>(output.data());
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 8; ++column) {
                const auto expected =
                    column == 1 || column == 2 ? values[row * 4 + column + 1] : 0x5a5a5a5a;
                if (copied[row * 8 + column] != expected)
                    return fail("pitched copy changed a row or its padding");
            }
        try {
            ninfer::copy_rows_async(destination.span(), 32, source.span(), 16, 8, 5, execution);
            return fail("pitched copy accepted an out-of-bounds row");
        } catch (const std::out_of_range&) {}
    }

    failures += expect_throws_device(count);
    failures += expect_throws_device(-1);

    ninfer::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    return failures == 0 ? 0 : fail("device test failed");
}
