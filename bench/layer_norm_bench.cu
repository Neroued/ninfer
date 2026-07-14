#include "ninfer/kernels/layer_norm.h"
#include "ninfer_bench_common.h"

#include <cstdio>

using namespace ninfer;
using namespace ninfer::bench;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    constexpr int d         = 1152;
    constexpr int patches   = 4096;
    constexpr std::size_t n = static_cast<std::size_t>(d) * patches;
    DBuf x                  = make_bf16(n);
    DBuf weight             = make_bf16(d);
    DBuf bias               = make_bf16(d);
    DBuf out                = make_zeros(n * 2);
    Tensor tx(x.p, DType::BF16, {d, patches});
    Tensor tw(weight.p, DType::BF16, {d});
    Tensor tb(bias.p, DType::BF16, {d});
    Tensor tout(out.p, DType::BF16, {d, patches});
    const Result result = bench_loop(
        [&](cudaStream_t stream) { kernels::layer_norm(tx, tw, tb, 1.0e-6f, tout, stream); },
        n * 4.0 + d * 4.0);
    print_result("layer_norm [1152,4096]", result);
    return 0;
}
