#include "qus/kernels/add_bias.h"
#include "qus_bench_common.h"

#include <cstdio>

using namespace qus;
using namespace qus::bench;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    constexpr int d         = 3456;
    constexpr int patches   = 4096;
    constexpr std::size_t n = static_cast<std::size_t>(d) * patches;
    DBuf x                  = make_bf16(n);
    DBuf bias               = make_bf16(d);
    Tensor tx(x.p, DType::BF16, {d, patches});
    Tensor tb(bias.p, DType::BF16, {d});
    const Result result =
        bench_loop([&](cudaStream_t stream) { kernels::add_bias(tb, tx, stream); }, n * 4.0);
    print_result("add_bias [3456,4096]", result);
    return 0;
}
