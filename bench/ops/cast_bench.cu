#include "ninfer/ops/cast.h"
#include "core/buffer.h"
#include "core/device.h"
#include "op_measurement.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kD = 1536;

void run(std::int32_t patches, bool profile_once) {
    const std::int64_t count = static_cast<std::int64_t>(kD) * patches;
    DeviceBuffer source(ninfer::bench::device_context().device(),
                        static_cast<std::size_t>(count) * sizeof(float));
    DeviceBuffer destination(ninfer::bench::device_context().device(),
                             static_cast<std::size_t>(count) * sizeof(std::uint16_t));
    CUDA_CHECK(fixture::fill_values(static_cast<float*>(source.data()), count, 101U, -2.F, 2.F));
    CUDA_CHECK(cudaMemset(destination.data(), 0, destination.size()));
    Tensor source_tensor(source.span(), DType::FP32, {kD, patches});
    Tensor destination_tensor(destination.span(), DType::BF16, {kD, patches});

    const auto launch = [&](CudaExecutionView cuda) {
        ops::cast_fp32_to_bf16(source_tensor, destination_tensor, cuda);
    };

    if (profile_once) {
        launch(default_execution());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("cast fp32-bf16 profile P=%d\n", patches);
        return;
    }

    const double bytes  = static_cast<double>(count) * 6.0;
    const Result result = bench_loop(launch, bytes);
    char tag[96];
    std::snprintf(tag, sizeof(tag), "cast fp32-bf16 [1536,%d]", patches);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) try {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool profile_once = false;
    std::vector<std::int32_t> patches;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--profile-once")) {
            profile_once = true;
        } else if (!std::strcmp(argv[i], "--patch") && i + 1 < argc) {
            patches.push_back(measurement::integer(argv[++i]));
        } else {
            throw std::invalid_argument("unknown cast benchmark argument");
        }
    }
    if (patches.empty()) {
        patches = {8, 256, 4096, 49152, 65536};
    }
    for (const std::int32_t p : patches) { run(p, profile_once); }
    return 0;
} catch (const std::exception& error) {

    std::fprintf(stderr, "ninfer_cast_bench: %s\n", error.what());
    return 2;
}
