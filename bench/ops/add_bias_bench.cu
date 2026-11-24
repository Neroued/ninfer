#include "ninfer/ops/add_bias.h"
#include "core/buffer.h"
#include "op_measurement.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

void run(int d, int columns, measurement::LogicalBenchmark& benchmark) {
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(columns);
    DeviceBuffer x      = make_bf16(n, 101U);
    DeviceBuffer bias   = make_bf16(d, 103U);
    SavedBuffer initial(x);
    const auto restore = [&](CudaExecutionView cuda) { initial.restore(cuda); };
    Tensor tx(x.span(), DType::BF16, {d, columns});
    Tensor tb(bias.span(), DType::BF16, {d});

    benchmark.run(
        {.op = "add-bias", .format = "bf16", .n = d, .tokens = columns}, n * 4 + d * 2,
        [&](CudaExecutionView cuda) { ops::add_bias(tb, tx, cuda); }, restore);
}

void run_matrix(measurement::LogicalBenchmark& benchmark) {
    constexpr std::array<int, 5> patches{8, 256, 4096, 49152, 65536};
    constexpr std::array<int, 5> merged{2, 64, 1024, 12288, 16384};
    for (int d : {1152, 3456, 4304}) {
        for (int p : patches) { run(d, p, benchmark); }
    }
    for (int d : {2048, 2560, 4608}) {
        for (int v : merged) { run(d, v, benchmark); }
    }
}

} // namespace

int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int selected_d       = 0;
    int selected_columns = 0;
    measurement::Options timing;
    for (int i = 1; i < argc; ++i) {
        const char* flag = argv[i];
        if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            selected_d = measurement::integer(argv[++i], 0);
        } else if (!std::strcmp(argv[i], "--columns") && i + 1 < argc) {
            selected_columns = measurement::integer(argv[++i], 0);
        } else if (!std::strcmp(argv[i], "--help")) {
            std::cout << "AddBias: --d D --columns C\n";
            measurement::help();
            return 0;
        } else if (!measurement::parse(timing, flag, measurement::take(argc, argv, i))) {
            std::fprintf(stderr, "usage: %s [--d D --columns C]\n", argv[0]);
            return 2;
        }
    }
    measurement::hardware();
    measurement::LogicalBenchmark benchmark(timing);

    if ((selected_d == 0) != (selected_columns == 0) || selected_d < 0 || selected_columns < 0 ||
        (selected_d > 0 && selected_d % 8 != 0)) {
        std::fprintf(stderr, "d and columns must be supplied together; d must be divisible by 8\n");
        return 2;
    }

    if (selected_d > 0) {
        run(selected_d, selected_columns, benchmark);
        return 0;
    }
    run_matrix(benchmark);
    return 0;
} catch (const std::exception& error) {

    std::fprintf(stderr, "ninfer_add_bias_bench: %s\n", error.what());
    return 2;
}
