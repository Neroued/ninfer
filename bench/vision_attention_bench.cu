#include "qus/kernels/vision_attention.h"
#include "qus_bench_common.h"

#include <cstdio>
#include <vector>

using namespace qus;
using namespace qus::bench;

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    int length = 256;
    if (argc > 1 && std::strcmp(argv[1], "--large") == 0) length = 1024;
    if (argc > 1 && std::strcmp(argv[1], "--xlarge") == 0) length = 4096;
    constexpr int d     = 72;
    constexpr int heads = 16;
    const std::size_t n = static_cast<std::size_t>(d) * heads * length;
    DBuf q              = make_bf16(n);
    DBuf k              = make_bf16(n);
    DBuf v              = make_bf16(n);
    DBuf out            = make_zeros(n * 2);
    const std::vector<int> cu{0, length};
    DBuf dcu(cu.size() * 4);
    cudaMemcpy(dcu.p, cu.data(), dcu.bytes, cudaMemcpyHostToDevice);
    Tensor tq(q.p, DType::BF16, {d, heads, length});
    Tensor tk(k.p, DType::BF16, {d, heads, length});
    Tensor tv(v.p, DType::BF16, {d, heads, length});
    Tensor tout(out.p, DType::BF16, {d, heads, length});
    Tensor tcu(dcu.p, DType::I32, {2});
    WorkspaceArena workspace(4096);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            kernels::vision_attention(tq, tk, tv, tcu, workspace, tout, stream);
        },
        n * 8.0, 2, 10, 100);
    const char* label = length == 256    ? "vision_attention L=256"
                        : length == 1024 ? "vision_attention L=1024"
                                         : "vision_attention L=4096";
    print_result(label, result);
    const double seconds     = result.median_us * 1e-6;
    const double math_tflops = 4.0 * length * length * d * heads / seconds / 1.0e12;
    const double mma_tflops  = 2.0 * length * length * heads * (80 + d) / seconds / 1.0e12;
    std::printf("  attention compute: %.1f TFLOP/s mathematical, %.1f TFLOP/s issued MMA\n",
                math_tflops, mma_tflops);
    return 0;
}
