// Correctness + coverage for residual_add, against the frozen op-test standard
// (docs/l1-op-test-standard.md): fp64 golden from bf16-rounded inputs, honest
// input ranges, composite tolerance bf16_elementwise.
#include "qus/kernels/residual_add.h"
#include "kernels/op_tester.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qus;
using namespace qus::test;

// fp64 reference: x[i] = double(x[i]) + double(y[i]).
static void cpu_residual_add(const std::vector<float>& y, const std::vector<float>& x,
                             std::vector<double>& o) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        o[i] = static_cast<double>(x[i]) + static_cast<double>(y[i]);
    }
}

static int one_shape(const char* tag, std::int32_t d0, std::int32_t d1, std::uint32_t seed,
                     float lo, float hi) {
    const auto n = static_cast<std::size_t>(d0) * static_cast<std::size_t>(d1);
    std::vector<float> y(n), x(n);
    fill_uniform(y, seed, lo, hi);
    fill_uniform(x, seed + 1000u, lo, hi);
    round_to_bf16(y);
    round_to_bf16(x);

    std::vector<double> ref(n);
    cpu_residual_add(y, x, ref);

    DBuf dy = to_device_bf16(y), dx = to_device_bf16(x);
    Tensor ty(dy.p, DType::BF16, {d0, d1}), tx(dx.p, DType::BF16, {d0, d1});
    kernels::residual_add(ty, tx, nullptr);
    cudaDeviceSynchronize();

    return verify(tag, from_device_bf16(dx, n), ref, Tolerance::bf16_elementwise());
}

static int one_shape_1d(const char* tag, std::int32_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> y(n), x(n);
    fill_uniform(y, seed, lo, hi);
    fill_uniform(x, seed + 1000u, lo, hi);
    round_to_bf16(y);
    round_to_bf16(x);

    std::vector<double> ref(n);
    cpu_residual_add(y, x, ref);

    DBuf dy = to_device_bf16(y), dx = to_device_bf16(x);
    Tensor ty(dy.p, DType::BF16, {n}), tx(dx.p, DType::BF16, {n});
    kernels::residual_add(ty, tx, nullptr);
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
    std::vector<float> y(n), x(n);
    fill_uniform(y, 2026u, -8.f, 8.f);
    fill_uniform(x, 3026u, -8.f, 8.f);
    round_to_bf16(y);
    round_to_bf16(x);

    std::vector<double> ref(n);
    cpu_residual_add(y, x, ref);

    DBuf dy = to_device_bf16_unaligned(y), dx = to_device_bf16_unaligned(x);
    auto* yptr = static_cast<unsigned char*>(dy.p) + 2;
    auto* xptr = static_cast<unsigned char*>(dx.p) + 2;
    Tensor ty(yptr, DType::BF16, {n}), tx(xptr, DType::BF16, {n});
    kernels::residual_add(ty, tx, nullptr);
    cudaDeviceSynchronize();

    DBuf packed(static_cast<std::size_t>(n) * 2);
    cudaMemcpy(packed.p, xptr, static_cast<std::size_t>(n) * 2, cudaMemcpyDeviceToDevice);
    return verify("residual_add unaligned data", from_device_bf16(packed, n), ref,
                  Tolerance::bf16_elementwise());
}

static int validation_checks() {
    int f = 0;
    Tensor y(nullptr, DType::BF16, {4});
    Tensor x(nullptr, DType::BF16, {4});

    try {
        Tensor empty_y(nullptr, DType::BF16, {1});
        Tensor empty_x(nullptr, DType::BF16, {1});
        empty_y.ne[0] = 0;
        empty_x.ne[0] = 0;
        kernels::residual_add(empty_y, empty_x, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "validation empty: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    try {
        Tensor bad_dtype(nullptr, DType::FP32, {4});
        kernels::residual_add(y, bad_dtype, nullptr);
        std::cerr << "validation dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_shape(nullptr, DType::BF16, {5});
        kernels::residual_add(bad_shape, x, nullptr);
        std::cerr << "validation shape: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_stride = x;
        bad_stride.nb[0]  = 4;
        kernels::residual_add(y, bad_stride, nullptr);
        std::cerr << "validation contiguous: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        kernels::residual_add(y, x, nullptr);
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
        f += one_shape("residual_add [5120,1]", 5120, 1, seed, -8.f, 8.f);
        f += one_shape("residual_add [5120,4096]", 5120, 4096, seed, -8.f, 8.f);
        f += one_shape("residual_add [6144,1]", 6144, 1, seed, -8.f, 8.f);
        f += one_shape_1d("residual_add [123457]", 123457, seed, -8.f, 8.f);
    }
    // Stress: large magnitudes expose rounding and overflow mistakes.
    f += one_shape_1d("residual_add stress [-60,60]", 123457, 4242u, -60.f, 60.f);
    f += unaligned_data_case();

    std::cout << (f ? "FAIL" : "OK") << " residual_add correctness\n";
    return f ? 1 : 0;
}
