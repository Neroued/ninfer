// Performance bench for the warp-row L2Norm domain. The registered 35B geometry is
// [D,heads,T]=[128,16,T], with T=1..6 for decode/verification and T=1024 for prefill.
//
// Narrow one-case profiling example:
//   ./ninfer_l2norm_bench --tokens 1024
#include "ninfer/ops/l2norm.h"
#include "core/buffer.h"
#include "op_measurement.cuh"

#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

void run(int d, int heads, int tokens) {
    const int rows   = heads * tokens;
    const auto n     = static_cast<std::size_t>(d) * static_cast<std::size_t>(rows);
    DeviceBuffer x   = make_bf16(n, 101U);
    DeviceBuffer out = make_zeros(n * 2);

    Tensor tx(x.span(), DType::BF16, {d, heads, tokens});
    Tensor tout(out.span(), DType::BF16, {d, heads, tokens});

    const double bytes = 2.0 * static_cast<double>(n) * 2.0;
    const Result result =
        bench_loop([&](CudaExecutionView cuda) { ops::l2norm(tx, 1.0e-6f, tout, cuda); }, bytes);

    char tag[80];
    std::snprintf(tag, sizeof(tag), "l2norm T=%-4d [D=%d,heads=%d]", tokens, d, heads);
    print_result(tag, result);
}

} // namespace

int main(int argc, char** argv) try {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int d               = 128;
    int heads           = 16;
    int selected_tokens = 0;
    bool decode         = false;
    bool prefill        = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            d = measurement::integer(argv[++i]);
        } else if (!std::strcmp(argv[i], "--heads") && i + 1 < argc) {
            heads = measurement::integer(argv[++i]);
        } else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            selected_tokens = measurement::integer(argv[++i], 0);
        } else if (!std::strcmp(argv[i], "--decode")) {
            decode = true;
        } else if (!std::strcmp(argv[i], "--prefill")) {
            prefill = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--decode] [--prefill] [--tokens T] "
                         "[--d D] [--heads H]\n",
                         argv[0]);
            return 2;
        }
    }
    if (d <= 0 || heads <= 0 || selected_tokens < 0) {
        std::fprintf(stderr, "D, heads, and tokens must be positive\n");
        return 2;
    }

    if (selected_tokens > 0) {
        run(d, heads, selected_tokens);
        return 0;
    }
    if (!decode && !prefill) { decode = prefill = true; }
    if (decode) {
        for (int tokens = 1; tokens <= 6; ++tokens) { run(d, heads, tokens); }
    }
    if (prefill) { run(d, heads, 1024); }
    return 0;
} catch (const std::exception& error) {

    std::fprintf(stderr, "ninfer_l2norm_bench: %s\n", error.what());
    return 2;
}
