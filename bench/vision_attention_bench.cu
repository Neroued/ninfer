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
    WorkspaceArena workspace(256);
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            kernels::vision_attention(tq, tk, tv, tcu, workspace, tout, stream);
        },
        n * 8.0, 2, 10, 100);
    print_result(length == 256 ? "vision_attention L=256" : "vision_attention L=1024", result);
    return 0;
}
