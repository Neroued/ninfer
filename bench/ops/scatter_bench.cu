#include "ninfer/ops/indexing.h"
#include "core/buffer.h"
#include "op_measurement.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

void run(std::int32_t d, std::int32_t columns, bool profile_once, bool permuted,
         measurement::LogicalBenchmark& benchmark) {
    const std::int32_t destination_columns = permuted ? columns * 2 + 1 : columns + 2;
    const std::size_t n                    = static_cast<std::size_t>(d) * columns;
    DeviceBuffer source                    = make_bf16(n, 101U);
    DeviceBuffer destination = make_bf16(static_cast<std::size_t>(d) * destination_columns, 103U);
    std::vector<std::int32_t> indices(static_cast<std::size_t>(columns));
    std::iota(indices.begin(), indices.end(), 1);
    if (permuted) {
        for (int i = 0; i < columns; ++i) indices[i] = i * 2;
        std::mt19937 generator(101);
        std::shuffle(indices.begin(), indices.end(), generator);
    }
    DeviceBuffer device_indices(ninfer::bench::device_context().device(),
                                indices.size() * sizeof(std::int32_t));
    cudaMemcpy(device_indices.data(), indices.data(), device_indices.size(),
               cudaMemcpyHostToDevice);
    Tensor source_tensor(source.span(), DType::BF16, {d, columns});
    Tensor destination_tensor(destination.span(), DType::BF16, {d, destination_columns});
    Tensor indices_tensor(device_indices.span(), DType::I32, {columns});

    const auto launch = [&](CudaExecutionView cuda) {
        ops::scatter(source_tensor, indices_tensor, destination_tensor, cuda);
    };

    if (profile_once) {
        launch(default_execution());
        cudaDeviceSynchronize();
        std::printf("scatter profile D=%d columns=%d\n", d, columns);
        return;
    }

    const auto bytes = std::size_t(columns) * (d * 4 + sizeof(std::int32_t));
    benchmark.run({.op     = permuted ? "scatter-slots" : "scatter-contiguous",
                   .format = "bf16",
                   .n      = d,
                   .tokens = columns},
                  bytes, launch, [](CudaExecutionView) {});
}

} // namespace

int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    std::int32_t d    = 2048;
    bool profile_once = false, permuted = false;
    measurement::Options timing;
    std::vector<std::int32_t> columns;
    for (int i = 1; i < argc; ++i) {
        const char* flag = argv[i];
        if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            d = measurement::integer(argv[++i]);
        } else if (!std::strcmp(argv[i], "--columns") && i + 1 < argc) {
            columns.push_back(measurement::integer(argv[++i]));
        } else if (!std::strcmp(argv[i], "--profile-once")) {
            profile_once = true;
        } else if (!std::strcmp(flag, "--mapping") && i + 1 < argc) {
            const std::string mapping = argv[++i];
            if (mapping != "contiguous" && mapping != "slots")
                throw std::invalid_argument("mapping: contiguous|slots");
            permuted = mapping == "slots";
        } else if (!std::strcmp(flag, "--help")) {
            std::puts("Scatter: --d D --columns T (repeatable) --mapping contiguous|slots "
                      "--profile-once");
            measurement::help();
            return 0;
        } else if (!measurement::parse(timing, flag, measurement::take(argc, argv, i))) {
            std::fprintf(stderr, "usage: %s [--d D] [--columns T] [--profile-once]\n", argv[0]);
            return 2;
        }
    }
    measurement::hardware();
    measurement::LogicalBenchmark benchmark(timing);
    if (columns.empty()) {
        columns = {1, 2, 64, 1024, 12288, 16384};
    }
    for (const std::int32_t v : columns) { run(d, v, profile_once, permuted, benchmark); }
    return 0;
} catch (const std::exception& error) {

    std::fprintf(stderr, "ninfer_scatter_bench: %s\n", error.what());
    return 2;
}
