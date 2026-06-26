// Correctness + coverage for silu_and_mul, against the frozen op-test standard
// (docs/l1-op-test-standard.md): fp64 golden from bf16-rounded inputs, honest
// input ranges (incl. a large-magnitude stress case that rejects any
// polynomial/"fast" silu approximation), composite tolerance bf16_elementwise.
#include "qus/kernels/silu_and_mul.h"
#include "kernels/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
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

    std::cout << (f ? "FAIL" : "OK") << " silu_and_mul correctness\n";
    return f ? 1 : 0;
}
