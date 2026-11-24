#include "ninfer/ops/gelu.h"
#include "core/buffer.h"
#include "op_measurement.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

void run(ops::GeluMode mode, int d, int columns, measurement::LogicalBenchmark& benchmark) {
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(columns);
    DeviceBuffer x      = make_bf16(n, 101U);
    SavedBuffer initial(x);
    const auto restore = [&](CudaExecutionView cuda) { initial.restore(cuda); };
    Tensor tx(x.span(), DType::BF16, {d, columns});

    benchmark.run(
        {.op     = mode == ops::GeluMode::Tanh ? "gelu-tanh" : "gelu-exact",
         .format = "bf16",
         .n      = d,
         .tokens = columns},
        n * 4, [&](CudaExecutionView cuda) { ops::gelu(tx, mode, cuda); }, restore);
}

} // namespace

int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int selected_columns        = 0;
    ops::GeluMode selected_mode = ops::GeluMode::Tanh;
    bool mode_selected          = false;
    measurement::Options timing;
    for (int i = 1; i < argc; ++i) {
        const char* flag = argv[i];
        if (!std::strcmp(argv[i], "--columns") && i + 1 < argc) {
            selected_columns = measurement::integer(argv[++i], 0);
        } else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char* mode = argv[++i];
            if (!std::strcmp(mode, "tanh")) {
                selected_mode = ops::GeluMode::Tanh;
            } else if (!std::strcmp(mode, "exact")) {
                selected_mode = ops::GeluMode::Exact;
            } else {
                std::fprintf(stderr, "mode must be tanh or exact\n");
                return 2;
            }
            mode_selected = true;
        } else if (!std::strcmp(argv[i], "--help")) {
            std::cout << "GELU: --mode tanh|exact --columns C\n";
            measurement::help();
            return 0;
        } else if (!measurement::parse(timing, flag, measurement::take(argc, argv, i))) {
            std::fprintf(stderr, "usage: %s [--mode tanh|exact --columns C]\n", argv[0]);
            return 2;
        }
    }
    measurement::hardware();
    measurement::LogicalBenchmark benchmark(timing);

    if (selected_columns < 0 || (selected_columns > 0 && !mode_selected)) {
        std::fprintf(stderr, "columns requires an explicit mode and must be positive\n");
        return 2;
    }

    if (selected_columns > 0) {
        const int d = selected_mode == ops::GeluMode::Tanh ? 4304 : 4608;
        run(selected_mode, d, selected_columns, benchmark);
        return 0;
    }
    for (const int patches : {8, 256, 4096, 49152, 65536}) {
        run(ops::GeluMode::Tanh, 4304, patches, benchmark);
    }
    for (const int merged : {2, 64, 1024, 12288, 16384}) {
        run(ops::GeluMode::Exact, 4608, merged, benchmark);
    }
    return 0;
} catch (const std::exception& error) {

    std::fprintf(stderr, "ninfer_gelu_bench: %s\n", error.what());
    return 2;
}
