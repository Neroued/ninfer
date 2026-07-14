#include "ninfer/kernels/vision_pos_embed.h"
#include "kernels/op_tester.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    constexpr int d       = 1152;
    constexpr int rows    = 2304;
    constexpr int patches = 256;
    std::vector<float> table(static_cast<std::size_t>(d) * rows);
    std::vector<float> x(static_cast<std::size_t>(d) * patches);
    std::vector<int> indices(static_cast<std::size_t>(patches) * 4);
    std::vector<float> weights(static_cast<std::size_t>(patches) * 4);
    fill_uniform(table, 1u, -2.0f, 2.0f);
    fill_uniform(x, 2u, -8.0f, 8.0f);
    fill_uniform(weights, 3u, 0.0f, 1.0f);
    round_to_bf16(table);
    round_to_bf16(x);
    for (int patch = 0; patch < patches; ++patch) {
        float sum = 0.0f;
        for (int corner = 0; corner < 4; ++corner) {
            indices[static_cast<std::size_t>(patch) * 4 + corner] =
                (patch * 37 + corner * 101) % rows;
            sum += weights[static_cast<std::size_t>(patch) * 4 + corner];
        }
        for (int corner = 0; corner < 4; ++corner) {
            weights[static_cast<std::size_t>(patch) * 4 + corner] /= sum;
        }
    }
    std::vector<double> reference(x.size());
    for (int patch = 0; patch < patches; ++patch) {
        for (int row = 0; row < d; ++row) {
            float position = 0.0f;
            for (int corner = 0; corner < 4; ++corner) {
                const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
                position +=
                    table[static_cast<std::size_t>(indices[control]) * d + row] * weights[control];
            }
            const float rounded = bf16_to_f32(f32_to_bf16(position));
            reference[static_cast<std::size_t>(patch) * d + row] =
                static_cast<double>(x[static_cast<std::size_t>(patch) * d + row]) + rounded;
        }
    }
    DBuf dtable = to_device_bf16(table);
    DBuf dx     = to_device_bf16(x);
    DBuf di     = to_device_i32(indices);
    DBuf dw     = to_device_f32(weights);
    Tensor ttable(dtable.p, DType::BF16, {d, rows});
    Tensor tx(dx.p, DType::BF16, {d, patches});
    Tensor ti(di.p, DType::I32, {4, patches});
    Tensor tw(dw.p, DType::FP32, {4, patches});
    kernels::vision_pos_embed_add(ttable, ti, tw, tx, nullptr);
    cudaDeviceSynchronize();
    const int failures = verify("vision position embedding", from_device_bf16(dx, x.size()),
                                reference, Tolerance::bf16_elementwise());
    std::cout << (failures ? "FAIL" : "OK") << " vision_pos_embed correctness\n";
    return failures ? 1 : 0;
}
