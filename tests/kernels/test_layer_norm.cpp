#include "ninfer/kernels/layer_norm.h"
#include "kernels/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int one_shape(std::int32_t tokens, std::uint32_t seed) {
    constexpr std::int32_t d = 1152;
    const std::size_t n      = static_cast<std::size_t>(d) * tokens;
    std::vector<float> x(n), weight(d), bias(d);
    fill_uniform(x, seed, -8.0f, 8.0f);
    fill_uniform(weight, seed + 1, -2.0f, 2.0f);
    fill_uniform(bias, seed + 2, -2.0f, 2.0f);
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(bias);
    std::vector<double> reference(n);
    for (int token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * d;
        double mean            = 0.0;
        for (int row = 0; row < d; ++row) mean += x[base + row];
        mean /= d;
        double variance = 0.0;
        for (int row = 0; row < d; ++row) {
            const double centered = static_cast<double>(x[base + row]) - mean;
            variance += centered * centered;
        }
        variance /= d;
        const double inv = 1.0 / std::sqrt(variance + 1.0e-6);
        for (int row = 0; row < d; ++row) {
            reference[base + row] =
                (static_cast<double>(x[base + row]) - mean) * inv * weight[row] + bias[row];
        }
    }
    DBuf dx = to_device_bf16(x);
    DBuf dw = to_device_bf16(weight);
    DBuf db = to_device_bf16(bias);
    DBuf dout(n * 2);
    Tensor tx(dx.p, DType::BF16, {d, tokens});
    Tensor tw(dw.p, DType::BF16, {d});
    Tensor tb(db.p, DType::BF16, {d});
    Tensor tout(dout.p, DType::BF16, {d, tokens});
    kernels::layer_norm(tx, tw, tb, 1.0e-6f, tout, nullptr);
    cudaDeviceSynchronize();
    return verify("vision layer_norm", from_device_bf16(dout, n), reference,
                  Tolerance::bf16_reduction());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    int failures = 0;
    for (std::uint32_t seed : {1u, 7u, 99u}) {
        failures += one_shape(1, seed);
        failures += one_shape(256, seed + 10);
    }
    failures += one_shape(4096, 2026u);
    std::cout << (failures ? "FAIL" : "OK") << " layer_norm correctness\n";
    return failures ? 1 : 0;
}
