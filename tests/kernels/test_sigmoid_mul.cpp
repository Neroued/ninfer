// Correctness + coverage for sigmoid_mul, against the frozen op-test standard
// (docs/kernel-development.md): fp64 golden from bf16-rounded inputs, honest
// input ranges (incl. a large-magnitude stress case), composite tolerance
// bf16_elementwise.
#include "ninfer/kernels/sigmoid_mul.h"
#include "kernels/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

// fp64 reference: x[i] = double(x[i]) * sigmoid(double(gate[i])).
static void cpu_sigmoid_gate_mul(const std::vector<float>& gate, const std::vector<float>& x,
                                 std::vector<double>& o) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double g = static_cast<double>(gate[i]);
        o[i]           = static_cast<double>(x[i]) * (1.0 / (1.0 + std::exp(-g)));
    }
}

static int one_shape(const char* tag, std::int32_t d0, std::int32_t d1, std::uint32_t seed,
                     float lo, float hi) {
    const auto n = static_cast<std::size_t>(d0) * static_cast<std::size_t>(d1);
    std::vector<float> gate(n), x(n);
    fill_uniform(gate, seed, lo, hi);
    fill_uniform(x, seed + 1000u, lo, hi);
    round_to_bf16(gate);
    round_to_bf16(x);

    std::vector<double> ref(n);
    cpu_sigmoid_gate_mul(gate, x, ref);

    DBuf dgate = to_device_bf16(gate), dx = to_device_bf16(x);
    Tensor tgate(dgate.p, DType::BF16, {d0, d1}), tx(dx.p, DType::BF16, {d0, d1});
    kernels::sigmoid_mul(tgate, tx, nullptr);
    cudaDeviceSynchronize();

    return verify(tag, from_device_bf16(dx, n), ref, Tolerance::bf16_elementwise());
}

static int one_shape_1d(const char* tag, std::int32_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> gate(n), x(n);
    fill_uniform(gate, seed, lo, hi);
    fill_uniform(x, seed + 1000u, lo, hi);
    round_to_bf16(gate);
    round_to_bf16(x);

    std::vector<double> ref(n);
    cpu_sigmoid_gate_mul(gate, x, ref);

    DBuf dgate = to_device_bf16(gate), dx = to_device_bf16(x);
    Tensor tgate(dgate.p, DType::BF16, {n}), tx(dx.p, DType::BF16, {n});
    kernels::sigmoid_mul(tgate, tx, nullptr);
    cudaDeviceSynchronize();

    return verify(tag, from_device_bf16(dx, n), ref, Tolerance::bf16_elementwise());
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
    constexpr std::int32_t n = 255;
    std::vector<float> gate(n), x(n);
    fill_uniform(gate, 2026u, -8.f, 8.f);
    fill_uniform(x, 3026u, -8.f, 8.f);
    round_to_bf16(gate);
    round_to_bf16(x);

    std::vector<double> ref(n);
    cpu_sigmoid_gate_mul(gate, x, ref);

    DBuf dgate = to_device_bf16_unaligned(gate), dx = to_device_bf16_unaligned(x);
    auto* gptr = static_cast<unsigned char*>(dgate.p) + 2;
    auto* xptr = static_cast<unsigned char*>(dx.p) + 2;
    Tensor tgate(gptr, DType::BF16, {n}), tx(xptr, DType::BF16, {n});
    kernels::sigmoid_mul(tgate, tx, nullptr);
    cudaDeviceSynchronize();

    DBuf packed(static_cast<std::size_t>(n) * 2);
    cudaMemcpy(packed.p, xptr, static_cast<std::size_t>(n) * 2, cudaMemcpyDeviceToDevice);
    return verify("sigmoid_mul unaligned data", from_device_bf16(packed, n), ref,
                  Tolerance::bf16_elementwise());
}

static int validation_checks() {
    int f = 0;
    Tensor gate(nullptr, DType::BF16, {4});
    Tensor x(nullptr, DType::BF16, {4});

    try {
        Tensor empty_gate(nullptr, DType::BF16, {1});
        Tensor empty_x(nullptr, DType::BF16, {1});
        empty_gate.ne[0] = 0;
        empty_x.ne[0]    = 0;
        kernels::sigmoid_mul(empty_gate, empty_x, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "validation empty: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    try {
        Tensor bad_dtype(nullptr, DType::FP32, {4});
        kernels::sigmoid_mul(bad_dtype, x, nullptr);
        std::cerr << "validation dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_shape = gate;
        bad_shape.ne[2]  = 2;
        kernels::sigmoid_mul(bad_shape, x, nullptr);
        std::cerr << "validation shape: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_stride = x;
        bad_stride.nb[0]  = 4;
        kernels::sigmoid_mul(gate, bad_stride, nullptr);
        std::cerr << "validation contiguous: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        kernels::sigmoid_mul(gate, x, nullptr);
        std::cerr << "validation null data: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    return f;
}

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }
    int f = 0;
    f += validation_checks();

    // Coverage: requested shapes; >=3 seeds; honest range.
    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_shape("sigmoid_mul [6144,1]", 6144, 1, seed, -8.f, 8.f);
        f += one_shape("sigmoid_mul [6144,4096]", 6144, 4096, seed, -8.f, 8.f);
        f += one_shape_1d("sigmoid_mul [255]", 255, seed, -8.f, 8.f);
    }
    // Stress: large magnitudes expose range-limited sigmoid approximations.
    f += one_shape_1d("sigmoid_mul stress [-40,40]", 65536, 4242u, -40.f, 40.f);
    f += unaligned_data_case();

    std::cout << (f ? "FAIL" : "OK") << " sigmoid_mul correctness\n";
    return f ? 1 : 0;
}
