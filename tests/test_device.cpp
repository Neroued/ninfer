#include "ninfer/core/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
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
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const ninfer::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.load_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.sm() <= 0) {
        std::cerr << label << " sm is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 0;
    }

    int failures = 0;

    ninfer::DeviceContext ctx(0);
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    CUDA_CHECK(cudaSuccess);
    ctx.synchronize();

    const cudaStream_t original_stream = ctx.stream;
    ninfer::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.load_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    ninfer::DeviceContext assigned(0);
    assigned = std::move(moved);
    if (moved.stream != nullptr || moved.load_stream != nullptr) {
        ++failures;
        std::cerr << "move assignment did not null source streams\n";
    }
    failures += check_context(assigned, "assigned");

    const cudaStream_t self_stream = assigned.stream;
    assigned                       = std::move(assigned);
    if (assigned.stream != self_stream) {
        ++failures;
        std::cerr << "self move assignment changed compute stream\n";
    }

    failures += expect_throws_device(-1);
    failures += expect_throws_device(count);

    ninfer::CudaEventTimer timer(assigned);
    timer.start();
    assigned.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    ninfer::CudaEventTimer timer_a(assigned);
    ninfer::CudaEventTimer timer_b(assigned);
    timer_b = std::move(timer_a);
    timer_b.start();
    assigned.synchronize();
    const float moved_elapsed_ms = timer_b.stop_ms();
    if (moved_elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "move-assigned timer elapsed time was negative\n";
    }

    if (count > 1) {
        CUDA_CHECK(cudaSetDevice(1));
        ninfer::CudaEventTimer drift_timer(assigned);
        drift_timer.start();
        assigned.synchronize();
        const float drift_elapsed_ms = drift_timer.stop_ms();
        if (drift_elapsed_ms < 0.0f) {
            ++failures;
            std::cerr << "current-device drift timer elapsed time was negative\n";
        }
    }

    return failures == 0 ? 0 : fail("device test failed");
}
