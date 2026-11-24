#include "ninfer/ops/embedding.h"
#include "core/buffer.h"
#include "op_measurement.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kD    = 1152;
constexpr int kRows = 2304;

void run(std::int32_t patches, bool profile_once, measurement::LogicalBenchmark& benchmark) {
    const std::size_t n = static_cast<std::size_t>(kD) * patches;
    DeviceBuffer table  = make_bf16(static_cast<std::size_t>(kD) * kRows, 101U);
    DeviceBuffer x      = make_bf16(n, 103U);
    std::vector<int> indices(static_cast<std::size_t>(patches) * 4);
    std::vector<float> weights(static_cast<std::size_t>(patches) * 4);
    for (int patch = 0; patch < patches; ++patch) {
        const float x          = fixture::uniform(patch, 211U, 0.F, 1.F);
        const float y          = fixture::uniform(patch, 213U, 0.F, 1.F);
        weights[patch * 4]     = (1.F - x) * (1.F - y);
        weights[patch * 4 + 1] = x * (1.F - y);
        weights[patch * 4 + 2] = (1.F - x) * y;
        weights[patch * 4 + 3] = x * y;
        for (int corner = 0; corner < 4; ++corner) {
            indices[static_cast<std::size_t>(patch) * 4 + corner] =
                (patch * 17 + corner * 49) % kRows;
        }
    }
    DeviceBuffer di(ninfer::bench::device_context().device(),
                    indices.size() * sizeof(std::int32_t));
    DeviceBuffer dw(ninfer::bench::device_context().device(), weights.size() * sizeof(float));
    cudaMemcpy(di.data(), indices.data(), di.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw.data(), weights.data(), dw.size(), cudaMemcpyHostToDevice);
    SavedBuffer initial(x);
    const auto restore = [&](CudaExecutionView cuda) { initial.restore(cuda); };
    Tensor ttable(table.span(), DType::BF16, {kD, kRows});
    Tensor tx(x.span(), DType::BF16, {kD, patches});
    Tensor ti(di.span(), DType::I32, {4, patches});
    Tensor tw(dw.span(), DType::FP32, {4, patches});

    const auto launch = [&](CudaExecutionView cuda) {
        ops::embedding_interpolate_add(ttable, ti, tw, tx, cuda);
    };

    if (profile_once) {
        launch(default_execution());
        cudaDeviceSynchronize();
        std::printf("embedding_interpolate_add profile P=%d\n", patches);
        return;
    }

    const std::set<int> unique_rows(indices.begin(), indices.end());
    const std::size_t bytes = n * 4 + unique_rows.size() * kD * 2 + patches * 32;
    benchmark.run(
        {.op = "interpolate-add", .format = "bf16", .n = kD, .k = kRows, .tokens = patches}, bytes,
        launch, restore);
}

} // namespace

int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool profile_once = false;
    std::vector<std::int32_t> patches;
    measurement::Options timing;
    for (int i = 1; i < argc; ++i) {
        const char* flag = argv[i];
        if (!std::strcmp(argv[i], "--profile-once")) {
            profile_once = true;
        } else if (!std::strcmp(argv[i], "--patch") && i + 1 < argc) {
            patches.push_back(measurement::integer(argv[++i]));
        } else if (!std::strcmp(argv[i], "--help")) {
            std::cout << "Position interpolation: --patch P (repeatable) --profile-once\n";
            measurement::help();
            return 0;
        } else if (!measurement::parse(timing, flag, measurement::take(argc, argv, i))) {
            std::fprintf(stderr, "usage: %s [--patch P] [--profile-once]\n", argv[0]);
            return 2;
        }
    }
    measurement::hardware();
    measurement::LogicalBenchmark benchmark(timing);

    if (patches.empty()) {
        patches = {8, 256, 4096, 49152, 65536};
    }
    for (const std::int32_t p : patches) { run(p, profile_once, benchmark); }
    return 0;
} catch (const std::exception& error) {

    std::fprintf(stderr, "ninfer_embedding_interpolate_add_bench: %s\n", error.what());
    return 2;
}
