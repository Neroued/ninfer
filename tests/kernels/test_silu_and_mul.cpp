// Correctness + coverage for silu_and_mul, against the frozen op-test standard
// (docs/l1-op-test-standard.md): fp64 golden from bf16-rounded inputs, honest
// input ranges (incl. a large-magnitude stress case that rejects any
// polynomial/"fast" silu approximation), composite tolerance bf16_elementwise.
#include "qus/kernels/silu_and_mul.h"
#include "kernels/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qus;
using namespace qus::test;

// fp64 reference: silu(x) = x / (1 + e^-x), then * up.
static void cpu_silu_and_mul(const std::vector<float>& g, const std::vector<float>& u,
                             std::vector<double>& o) {
    for (std::size_t i = 0; i < g.size(); ++i) {
        const double x = g[i];
        const double s = x / (1.0 + std::exp(-x));
        o[i]           = s * static_cast<double>(u[i]);
    }
}

static int one_shape(const char* tag, int n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> g(n), u(n);
    fill_uniform(g, seed, lo, hi);
    fill_uniform(u, seed + 1000u, lo, hi);
    round_to_bf16(g);
    round_to_bf16(u);

    std::vector<double> ref(n);
    cpu_silu_and_mul(g, u, ref);

    DBuf dg = to_device_bf16(g), du = to_device_bf16(u), dout(static_cast<std::size_t>(n) * 2);
    Tensor tg(dg.p, DType::BF16, {n}), tu(du.p, DType::BF16, {n}), tout(dout.p, DType::BF16, {n});
    kernels::silu_and_mul(tg, tu, tout, nullptr);
    cudaDeviceSynchronize();

    return verify(tag, from_device_bf16(dout, n), ref, Tolerance::bf16_elementwise());
}

static DBuf to_device_bf16_unaligned(const std::vector<float>& h) {
    std::vector<std::uint16_t> b(h.size() + 1);
    b[0] = 0;
    for (std::size_t i = 0; i < h.size(); ++i) b[i + 1] = f32_to_bf16(h[i]);
    DBuf d(b.size() * 2);
    cudaMemcpy(d.p, b.data(), b.size() * 2, cudaMemcpyHostToDevice);
    return d;
}

static int unaligned_data_case() {
    constexpr int n = 255;
    std::vector<float> g(n), u(n);
    fill_uniform(g, 2026u, -8.f, 8.f);
    fill_uniform(u, 3026u, -8.f, 8.f);
    round_to_bf16(g);
    round_to_bf16(u);

    std::vector<double> ref(n);
    cpu_silu_and_mul(g, u, ref);

    DBuf dg = to_device_bf16_unaligned(g), du = to_device_bf16_unaligned(u),
         dout(static_cast<std::size_t>(n + 1) * 2);
    auto* gptr = static_cast<unsigned char*>(dg.p) + 2;
    auto* uptr = static_cast<unsigned char*>(du.p) + 2;
    auto* optr = static_cast<unsigned char*>(dout.p) + 2;
    Tensor tg(gptr, DType::BF16, {n}), tu(uptr, DType::BF16, {n}), tout(optr, DType::BF16, {n});
    kernels::silu_and_mul(tg, tu, tout, nullptr);
    cudaDeviceSynchronize();

    DBuf packed(static_cast<std::size_t>(n) * 2);
    cudaMemcpy(packed.p, optr, static_cast<std::size_t>(n) * 2, cudaMemcpyDeviceToDevice);
    return verify("silu unaligned data", from_device_bf16(packed, n), ref,
                  Tolerance::bf16_elementwise());
}

static int null_validation_case() {
    try {
        Tensor gate(nullptr, DType::BF16, {1});
        Tensor up(nullptr, DType::BF16, {1});
        Tensor out(nullptr, DType::BF16, {1});
        kernels::silu_and_mul(gate, up, out, nullptr);
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << "silu null validation: expected invalid_argument\n";
    return 1;
}

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    int f = 0;
    // Coverage: decode-ish, MLP intermediate, prefill-ish, unaligned; >=3 seeds; honest range.
    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_shape("silu n=1", 1, seed, -8.f, 8.f);
        f += one_shape("silu n=7", 7, seed, -8.f, 8.f);
        f += one_shape("silu n=17408 (intermediate)", 17408, seed, -8.f, 8.f);
        f += one_shape("silu n=1114112 (prefill-ish)", 1114112, seed, -8.f, 8.f);
        f += one_shape("silu n=123457 (unaligned)", 123457, seed, -8.f, 8.f);
    }
    // Stress: large magnitudes break range-limited approximations of silu.
    f += one_shape("silu stress [-30,30]", 65536, 4242u, -30.f, 30.f);
    f += unaligned_data_case();
    f += null_validation_case();

    std::cout << (f ? "FAIL" : "OK") << " silu_and_mul correctness\n";
    return f ? 1 : 0;
}
