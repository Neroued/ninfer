#include "ninfer/kernels/vision_pos_embed.h"
#include "ninfer_bench_common.h"

#include <cstdio>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    constexpr int d         = 1152;
    constexpr int rows      = 2304;
    constexpr int patches   = 4096;
    constexpr std::size_t n = static_cast<std::size_t>(d) * patches;
    DBuf table              = make_bf16(static_cast<std::size_t>(d) * rows);
    DBuf x                  = make_bf16(n);
    std::vector<int> indices(static_cast<std::size_t>(patches) * 4);
    std::vector<float> weights(static_cast<std::size_t>(patches) * 4, 0.25f);
    for (int patch = 0; patch < patches; ++patch) {
        for (int corner = 0; corner < 4; ++corner) {
            indices[static_cast<std::size_t>(patch) * 4 + corner] =
                (patch * 17 + corner * 49) % rows;
        }
    }
    DBuf di(indices.size() * 4);
    DBuf dw(weights.size() * 4);
    cudaMemcpy(di.p, indices.data(), di.bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dw.p, weights.data(), dw.bytes, cudaMemcpyHostToDevice);
    Tensor ttable(table.p, DType::BF16, {d, rows});
    Tensor tx(x.p, DType::BF16, {d, patches});
    Tensor ti(di.p, DType::I32, {4, patches});
    Tensor tw(dw.p, DType::FP32, {4, patches});
    const double bytes  = n * 12.0 + patches * 32.0;
    const Result result = bench_loop(
        [&](cudaStream_t stream) { kernels::vision_pos_embed_add(ttable, ti, tw, tx, stream); },
        bytes);
    print_result("vision_pos_embed [1152,4096]", result);
    return 0;
}
