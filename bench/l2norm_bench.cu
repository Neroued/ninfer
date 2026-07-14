// Performance bench for l2norm at the Qwen3.6-27B attention head shape
// ([head_dim, heads, T] = [128,16,T]). This binary is the ncu/nsys target;
// the GB/s it prints is informational only -- the gate is ncu sustained DRAM %
// (see docs/kernel-development.md §8).
//   ./ninfer_l2norm_bench [--decode] [--prefill]   (default: both)
#include "ninfer/kernels/l2norm.h"
#include "ninfer_bench_common.h"

#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

static void run(int t, const char* tag) {
    constexpr int kHeadDim = 128;
    constexpr int kHeads = 16;
    const auto n = static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kHeads) *
                   static_cast<std::size_t>(t);

    DBuf x = make_bf16(n);
    DBuf out = make_zeros(n * 2);

    Tensor tx(x.p, DType::BF16, {kHeadDim, kHeads, t});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kHeads, t});

    const double bytes = 2.0 * static_cast<double>(n) * 2.0; // read x + write out
    const Result r = bench_loop([&](cudaStream_t s) { kernels::l2norm(tx, 1e-6f, tout, s); },
                                bytes);
    print_result(tag, r);
}

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    bool prefill = false, decode = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--prefill")) prefill = true;
        else if (!std::strcmp(argv[i], "--decode")) decode = true;
    }
    if (!prefill && !decode) { prefill = decode = true; }

    if (decode) run(1, "l2norm decode  [128,16,1]");
    if (prefill) run(4096, "l2norm prefill [128,16,4096]");
    return 0;
}
