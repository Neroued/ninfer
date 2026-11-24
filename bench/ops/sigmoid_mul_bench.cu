#include "ninfer/ops/sigmoid_mul.h"
#include "core/buffer.h"
#include "op_measurement.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

void run(int tokens, int heads) {
    const int d         = 256 * heads;
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(tokens);
    DeviceBuffer gate   = make_bf16(n, 101U);
    DeviceBuffer x      = make_bf16(n, 103U);
    SavedBuffer initial(x);
    const auto restore = [&](CudaExecutionView cuda) { initial.restore(cuda); };
    Tensor tgate(gate.span(), DType::BF16, {256, heads, tokens});
    Tensor tx(x.span(), DType::BF16, {256, heads, tokens});

    const Result result = bench_loop_prepared(
        restore, [&](CudaExecutionView cuda) { ops::sigmoid_mul(tgate, tx, cuda); },
        static_cast<double>(n) * 6.0);

    char tag[128];
    std::snprintf(tag, sizeof(tag), "sigmoid_mul [256,%d,%-2d]", heads, tokens);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int heads = 16;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens = measurement::integers(argv[++i]);
        } else if (!std::strcmp(argv[i], "--heads") && i + 1 < argc) {
            heads = measurement::integer(argv[++i]);
            if (heads != 16 && heads != 24) {
                std::fprintf(stderr, "heads must be 16 or 24\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "usage: %s [--heads 16|24] [--tokens T[,T...]]\n", argv[0]);
            return 2;
        }
    }
    for (const int value : tokens) { run(value, heads); }
    return 0;
} catch (const std::exception& error) {

    std::fprintf(stderr, "ninfer_sigmoid_mul_bench: %s\n", error.what());
    return 2;
}
