#include "ninfer/ops/vision_pos_embed.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int one_shape(const char* route, int d, int rows, int patches, std::uint32_t seed) {
    std::vector<float> table(static_cast<std::size_t>(d) * rows);
    std::vector<float> x(static_cast<std::size_t>(d) * patches);
    std::vector<int> indices(static_cast<std::size_t>(patches) * 4);
    std::vector<float> weights(static_cast<std::size_t>(patches) * 4);
    fill_uniform(table, seed, -2.0f, 2.0f);
    fill_uniform(x, seed + 1u, -8.0f, 8.0f);
    fill_uniform(weights, seed + 2u, 0.0f, 1.0f);
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
            double position = 0.0;
            for (int corner = 0; corner < 4; ++corner) {
                const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
                position += static_cast<double>(
                                table[static_cast<std::size_t>(indices[control]) * d + row]) *
                            static_cast<double>(weights[control]);
            }
            reference[static_cast<std::size_t>(patch) * d + row] =
                static_cast<double>(x[static_cast<std::size_t>(patch) * d + row]) + position;
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
    ops::vision_pos_embed_add(ttable, ti, tw, tx, nullptr);
    cudaDeviceSynchronize();

    const std::vector<double> got = from_device_bf16(dx, x.size());
    const std::string label       = std::string("vision position embedding ") + route +
                              " D=" + std::to_string(d) + " P=" + std::to_string(patches);
    return verify(label.c_str(), got, reference, Tolerance::bf16_elementwise());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    failures += one_shape("d1152-warp", 1152, 2304, 17, 1u);
    failures += one_shape("d1152-cta", 1152, 2304, 1024, 11u);
    failures += one_shape("generic", 1151, 2304, 37, 21u);
    std::cout << (failures ? "FAIL" : "OK") << " vision_pos_embed correctness\n";
    return failures ? 1 : 0;
}
