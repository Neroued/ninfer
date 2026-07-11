#include "qus/kernels/gelu.h"
#include "qus_bench_common.h"

#include <cstdio>

using namespace qus;
using namespace qus::bench;

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    const bool exact        = argc > 1 && std::strcmp(argv[1], "--exact") == 0;
    constexpr int d         = 4304;
    constexpr int patches   = 4096;
    constexpr std::size_t n = static_cast<std::size_t>(d) * patches;
    DBuf x                  = make_bf16(n);
    Tensor tx(x.p, DType::BF16, {d, patches});
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            kernels::gelu(tx, exact ? kernels::GeluMode::Exact : kernels::GeluMode::Tanh, stream);
        },
        n * 4.0);
    print_result(exact ? "gelu exact [4304,4096]" : "gelu tanh [4304,4096]", result);
    return 0;
}
