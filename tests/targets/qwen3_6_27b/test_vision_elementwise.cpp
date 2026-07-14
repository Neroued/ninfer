#include "kernels/add_bias/add_bias.h"
#include "kernels/gelu/gelu.h"
#include "kernels/scatter/scatter.h"
#include "kernels/op_tester.h"
#include "targets/qwen3_6_27b_rtx5090/impl/config.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/ops.h"

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

int test_shifted_mtp_visual_composition() {
    constexpr std::int32_t d = ninfer::targets::qwen3_6_27b_rtx5090::detail::TextConfig::hidden;
    constexpr std::int32_t t = 4;
    constexpr std::int32_t v = 3;
    std::vector<float> token_embeddings(static_cast<std::size_t>(d) * t);
    std::vector<float> visual_embeddings(static_cast<std::size_t>(d) * v);
    for (int col = 0; col < t; ++col) {
        std::fill_n(token_embeddings.data() + static_cast<std::size_t>(col) * d, d,
                    static_cast<float>(10 + col));
    }
    for (int col = 0; col < v; ++col) {
        std::fill_n(visual_embeddings.data() + static_cast<std::size_t>(col) * d, d,
                    static_cast<float>(100 + col));
    }
    round_to_bf16(token_embeddings);
    round_to_bf16(visual_embeddings);
    DBuf dvisual = to_device_bf16(visual_embeddings);
    Tensor visual(dvisual.p, DType::BF16, {d, v});
    const std::vector<std::int32_t> scatter_indices{2, 4, 7};
    WorkspaceArena work(1ULL << 20);

    auto run = [&](std::int32_t prompt_tokens, const char* label) {
        DBuf dinput = to_device_bf16(token_embeddings);
        Tensor input(dinput.p, DType::BF16, {d, t});
        work.reset();
        ninfer::targets::qwen3_6_27b_rtx5090::detail::schedule::detail::
            scatter_shifted_visual_embeddings(input, visual, scatter_indices, 1, prompt_tokens,
                                              work, nullptr);
        cudaDeviceSynchronize();

        std::vector<double> reference(token_embeddings.begin(), token_embeddings.end());
        for (int row = 0; row < d; ++row) {
            reference[static_cast<std::size_t>(1) * d + row] = visual_embeddings[row];
            if (prompt_tokens > 4) {
                reference[static_cast<std::size_t>(3) * d + row] =
                    visual_embeddings[static_cast<std::size_t>(d) + row];
            }
        }
        return verify(label, from_device_bf16(dinput, reference.size()), reference,
                      Tolerance::bf16_elementwise());
    };

    int failures = 0;
    failures += run(8, "MTP shifted visual composition crosses chunk boundary");
    failures += run(4, "MTP shifted visual composition keeps bonus token embedding");
    return failures;
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
    failures += test_shifted_mtp_visual_composition();
    std::cout << (failures ? "FAIL" : "OK") << " vision elementwise correctness\n";
    return failures ? 1 : 0;
}
