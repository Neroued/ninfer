// Performance bench for residual_add at the real Qwen3.6-27B hidden shape
// (hidden_size = 5120). This binary is the ncu/nsys target; the GB/s it prints
// is informational only -- the gate is ncu sustained DRAM % (see
// docs/kernel-development.md §8).
//   ./ninfer_residual_add_bench [--decode] [--prefill]   (default: both)
#include "kernels/residual_add/residual_add.h"
#include "ninfer_bench_common.h"

#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

static void run(int n, const char* tag) {
    DBuf y = make_bf16(static_cast<std::size_t>(n));
    DBuf x = make_bf16(static_cast<std::size_t>(n));
    Tensor ty(y.p, DType::BF16, {n}), tx(x.p, DType::BF16, {n});

    const double bytes = 3.0 * static_cast<double>(n) * 2.0;  // read y + read x + write x
    const Result r = bench_loop([&](cudaStream_t s) { kernels::residual_add(ty, tx, s); }, bytes);
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

    constexpr int kHidden = 5120;  // Qwen3.6-27B hidden_size
    if (decode) run(kHidden * 1, "residual_add decode  [5120,1]");
    if (prefill) run(kHidden * 4096, "residual_add prefill [5120,4096]");
    return 0;
}
