// Performance bench for silu_mul at the real Qwen3.6-27B MLP shape
// (intermediate = 17408). This binary is the ncu/nsys target; the GB/s it
// prints is informational only -- the gate is ncu sustained DRAM %% (see
// docs/kernel-development.md §8).
//   ./ninfer_silu_and_mul_bench [--decode] [--prefill]   (default: both)
#include "kernels/silu_mul/silu_mul.h"
#include "ninfer_bench_common.h"

#include <cstring>

using namespace ninfer;
using namespace ninfer::bench;

static void run(int n, const char* tag) {
    DBuf g   = make_bf16(static_cast<std::size_t>(n));
    DBuf u   = make_bf16(static_cast<std::size_t>(n));
    DBuf out = make_zeros(static_cast<std::size_t>(n) * 2);
    Tensor tg(g.p, DType::BF16, {n}), tu(u.p, DType::BF16, {n}), tout(out.p, DType::BF16, {n});

    const double bytes = 3.0 * static_cast<double>(n) * 2.0; // read gate + read up + write out
    const Result r = bench_loop([&](cudaStream_t s) { kernels::silu_mul(tg, tu, tout, s); }, bytes);
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
        if (!std::strcmp(argv[i], "--prefill"))
            prefill = true;
        else if (!std::strcmp(argv[i], "--decode"))
            decode = true;
    }
    if (!prefill && !decode) { prefill = decode = true; }

    constexpr int kIntermediate = 17408; // Qwen3.6-27B MLP intermediate_size
    if (decode) run(kIntermediate * 1, "silu_mul decode  [17408,1]");
    if (prefill) run(kIntermediate * 4096, "silu_mul prefill [17408,4096]");
    return 0;
}
