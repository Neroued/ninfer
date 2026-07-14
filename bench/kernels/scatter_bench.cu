#include "kernels/scatter/scatter.h"
#include "ninfer_bench_common.h"

#include <cstdio>
#include <numeric>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    constexpr int d             = 5120;
    constexpr int vision_tokens = 2040;
    constexpr int prompt_tokens = 4096;
    constexpr std::size_t n     = static_cast<std::size_t>(d) * vision_tokens;
    DBuf src                    = make_bf16(n);
    DBuf dst                    = make_bf16(static_cast<std::size_t>(d) * prompt_tokens);
    std::vector<int> indices(vision_tokens);
    std::iota(indices.begin(), indices.end(), 1024);
    DBuf di(indices.size() * 4);
    cudaMemcpy(di.p, indices.data(), di.bytes, cudaMemcpyHostToDevice);
    Tensor tsrc(src.p, DType::BF16, {d, vision_tokens});
    Tensor tdst(dst.p, DType::BF16, {d, prompt_tokens});
    Tensor ti(di.p, DType::I32, {vision_tokens});
    const Result result =
        bench_loop([&](cudaStream_t stream) { kernels::scatter(tsrc, ti, tdst, stream); }, n * 4.0);
    print_result("scatter [5120,2040]", result);
    return 0;
}
