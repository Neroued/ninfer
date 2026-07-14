#include "ninfer/kernels/add_bias.h"
#include "ninfer/kernels/gelu.h"
#include "ninfer/kernels/scatter.h"
#include "kernels/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int test_add_bias(std::int32_t d, std::int32_t tokens, std::uint32_t seed) {
    const std::size_t n = static_cast<std::size_t>(d) * tokens;
    std::vector<float> x(n), bias(d);
    fill_uniform(x, seed, -8.0f, 8.0f);
    fill_uniform(bias, seed + 1, -2.0f, 2.0f);
    round_to_bf16(x);
    round_to_bf16(bias);
    std::vector<double> reference(n);
    for (std::size_t i = 0; i < n; ++i) {
        reference[i] = static_cast<double>(x[i]) + bias[i % static_cast<std::size_t>(d)];
    }
    DBuf dx = to_device_bf16(x);
    DBuf db = to_device_bf16(bias);
    Tensor tx(dx.p, DType::BF16, {d, tokens});
    Tensor tb(db.p, DType::BF16, {d});
    kernels::add_bias(tb, tx, nullptr);
    cudaDeviceSynchronize();
    return verify("vision add_bias", from_device_bf16(dx, n), reference,
                  Tolerance::bf16_elementwise());
}

double gelu_reference(double x, kernels::GeluMode mode) {
    if (mode == kernels::GeluMode::Tanh) {
        constexpr double root = 0.79788456080286535588;
        return 0.5 * x * (1.0 + std::tanh(root * (x + 0.044715 * x * x * x)));
    }
    return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

int test_gelu(kernels::GeluMode mode, std::uint32_t seed) {
    constexpr std::int32_t d      = 4304;
    constexpr std::int32_t tokens = 256;
    const std::size_t n           = static_cast<std::size_t>(d) * tokens;
    std::vector<float> x(n);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(x);
    std::vector<double> reference(n);
    for (std::size_t i = 0; i < n; ++i) reference[i] = gelu_reference(x[i], mode);
    DBuf dx = to_device_bf16(x);
    Tensor tx(dx.p, DType::BF16, {d, tokens});
    kernels::gelu(tx, mode, nullptr);
    cudaDeviceSynchronize();
    return verify(mode == kernels::GeluMode::Tanh ? "vision gelu tanh" : "vision gelu exact",
                  from_device_bf16(dx, n), reference, Tolerance::bf16_elementwise());
}

int test_scatter(std::uint32_t seed) {
    constexpr std::int32_t d = 5120;
    constexpr std::int32_t v = 64;
    constexpr std::int32_t t = 129;
    std::vector<float> src(static_cast<std::size_t>(d) * v);
    std::vector<float> dst(static_cast<std::size_t>(d) * t);
    std::vector<int> indices(v);
    for (int i = 0; i < v; ++i) indices[i] = 1 + i * 2;
    fill_uniform(src, seed, -8.0f, 8.0f);
    fill_uniform(dst, seed + 1, -8.0f, 8.0f);
    round_to_bf16(src);
    round_to_bf16(dst);
    std::vector<double> reference(dst.begin(), dst.end());
    for (int col = 0; col < v; ++col) {
        for (int row = 0; row < d; ++row) {
            reference[static_cast<std::size_t>(indices[col]) * d + row] =
                src[static_cast<std::size_t>(col) * d + row];
        }
    }
    DBuf dsrc = to_device_bf16(src);
    DBuf ddst = to_device_bf16(dst);
    DBuf didx = to_device_i32(indices);
    Tensor tsrc(dsrc.p, DType::BF16, {d, v});
    Tensor tdst(ddst.p, DType::BF16, {d, t});
    Tensor tidx(didx.p, DType::I32, {v});
    kernels::scatter(tsrc, tidx, tdst, nullptr);
    cudaDeviceSynchronize();
    return verify("vision scatter", from_device_bf16(ddst, reference.size()), reference,
                  Tolerance::bf16_elementwise());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    int failures = 0;
    for (std::uint32_t seed : {1u, 7u, 99u}) {
        failures += test_add_bias(1152, 256, seed);
        failures += test_add_bias(3456, 256, seed + 10);
        failures += test_add_bias(5120, 1, seed + 20);
        failures += test_gelu(kernels::GeluMode::Tanh, seed + 30);
        failures += test_gelu(kernels::GeluMode::Exact, seed + 40);
        failures += test_scatter(seed + 50);
    }
    std::cout << (failures ? "FAIL" : "OK") << " vision elementwise correctness\n";
    return failures ? 1 : 0;
}
