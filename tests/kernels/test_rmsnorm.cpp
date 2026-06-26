// Correctness + coverage for rmsnorm, against the frozen op-test standard
// (docs/l1-op-test-standard.md): fp64 golden from bf16-rounded inputs, honest
// input ranges, composite tolerance bf16_reduction.
#include "qus/kernels/rmsnorm.h"
#include "kernels/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qus;
using namespace qus::test;

static void cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& weight,
                        float eps, bool unit_offset, const std::vector<float>* z,
                        std::int32_t d, std::int64_t rows, std::vector<double>& o) {
    for (std::int64_t r = 0; r < rows; ++r) {
        const std::int64_t base = r * d;
        double sumsq = 0.0;
        for (std::int32_t i = 0; i < d; ++i) {
            const double xv = static_cast<double>(x[base + i]);
            sumsq += xv * xv;
        }
        const double inv = 1.0 / std::sqrt(sumsq / static_cast<double>(d) + static_cast<double>(eps));
        for (std::int32_t i = 0; i < d; ++i) {
            const double w = static_cast<double>(weight[i]) + (unit_offset ? 1.0 : 0.0);
            double v = static_cast<double>(x[base + i]) * inv * w;
            if (z != nullptr) {
                const double zv = static_cast<double>((*z)[base + i]);
                v *= zv / (1.0 + std::exp(-zv));
            }
            o[base + i] = v;
        }
    }
}

static Tensor tensor_for_shape(void* data, std::int32_t d0, std::int32_t d1, std::int32_t d2) {
    if (d2 != 1) { return Tensor(data, DType::BF16, {d0, d1, d2}); }
    return Tensor(data, DType::BF16, {d0, d1});
}

static int one_shape(const char* tag, std::int32_t d0, std::int32_t d1, std::int32_t d2,
                     bool unit_offset, bool gated, std::uint32_t seed, float lo, float hi) {
    const auto rows = static_cast<std::int64_t>(d1) * static_cast<std::int64_t>(d2);
    const auto n = static_cast<std::size_t>(d0) * static_cast<std::size_t>(rows);
    std::vector<float> x(n), weight(d0), z(n);
    fill_uniform(x, seed, lo, hi);
    fill_uniform(weight, seed + 1000u, lo, hi);
    if (gated) { fill_uniform(z, seed + 2000u, lo, hi); }
    round_to_bf16(x);
    round_to_bf16(weight);
    if (gated) { round_to_bf16(z); }

    std::vector<double> ref(n);
    cpu_rmsnorm(x, weight, 1e-6f, unit_offset, gated ? &z : nullptr, d0, rows, ref);

    DBuf dx = to_device_bf16(x), dw = to_device_bf16(weight), dout(n * 2);
    DBuf dz = gated ? to_device_bf16(z) : DBuf(0);
    Tensor tx = tensor_for_shape(dx.p, d0, d1, d2);
    Tensor tw(dw.p, DType::BF16, {d0});
    Tensor tout = tensor_for_shape(dout.p, d0, d1, d2);
    Tensor tz;
    const Tensor* tz_ptr = nullptr;
    if (gated) {
        tz = tensor_for_shape(dz.p, d0, d1, d2);
        tz_ptr = &tz;
    }

    kernels::rmsnorm(tx, tw, 1e-6f, unit_offset, tz_ptr, tout, nullptr);
    cudaDeviceSynchronize();

    return verify(tag, from_device_bf16(dout, n), ref, Tolerance::bf16_reduction());
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
    constexpr std::int32_t d0 = 260;
    constexpr std::int32_t d1 = 3;
    constexpr std::int64_t rows = d1;
    constexpr std::size_t n = static_cast<std::size_t>(d0) * d1;
    std::vector<float> x(n), weight(d0);
    fill_uniform(x, 2026u, -8.f, 8.f);
    fill_uniform(weight, 3026u, -8.f, 8.f);
    round_to_bf16(x);
    round_to_bf16(weight);

    std::vector<double> ref(n);
    cpu_rmsnorm(x, weight, 1e-6f, true, nullptr, d0, rows, ref);

    DBuf dx = to_device_bf16_unaligned(x), dw = to_device_bf16_unaligned(weight),
         dout((n + 1) * 2);
    auto* xptr = static_cast<unsigned char*>(dx.p) + 2;
    auto* wptr = static_cast<unsigned char*>(dw.p) + 2;
    auto* optr = static_cast<unsigned char*>(dout.p) + 2;
    Tensor tx(xptr, DType::BF16, {d0, d1});
    Tensor tw(wptr, DType::BF16, {d0});
    Tensor tout(optr, DType::BF16, {d0, d1});
    kernels::rmsnorm(tx, tw, 1e-6f, true, nullptr, tout, nullptr);
    cudaDeviceSynchronize();

    DBuf packed(n * 2);
    cudaMemcpy(packed.p, optr, n * 2, cudaMemcpyDeviceToDevice);
    return verify("rmsnorm unaligned data [260,3]", from_device_bf16(packed, n), ref,
                  Tolerance::bf16_reduction());
}

static int validation_checks() {
    int f = 0;
    Tensor x(nullptr, DType::BF16, {4});
    Tensor weight(nullptr, DType::BF16, {4});
    Tensor out(nullptr, DType::BF16, {4});

    auto make_huge_rows = [] {
        Tensor t(nullptr, DType::BF16, {1});
        t.ne[0] = 1;
        t.ne[1] = 65536;
        t.ne[2] = 65536;
        t.ne[3] = 1;
        t.nb[0] = 2;
        t.nb[1] = 2;
        t.nb[2] = 131072;
        t.nb[3] = 8589934592LL;
        return t;
    };

    try {
        Tensor empty_x(nullptr, DType::BF16, {1});
        Tensor empty_weight(nullptr, DType::BF16, {1});
        Tensor empty_out(nullptr, DType::BF16, {1});
        empty_x.ne[0] = 0;
        empty_weight.ne[0] = 0;
        empty_out.ne[0] = 0;
        kernels::rmsnorm(empty_x, empty_weight, 1e-6f, true, nullptr, empty_out, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "validation empty: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    try {
        Tensor bad_empty_x(nullptr, DType::BF16, {1});
        Tensor bad_empty_weight(nullptr, DType::BF16, {1});
        Tensor bad_empty_out(nullptr, DType::BF16, {1});
        bad_empty_x.ne[0] = 0;
        bad_empty_x.ne[1] = -1;
        bad_empty_weight.ne[0] = 0;
        bad_empty_out.ne[0] = 0;
        bad_empty_out.ne[1] = -1;
        kernels::rmsnorm(bad_empty_x, bad_empty_weight, 1e-6f, true, nullptr, bad_empty_out,
                         nullptr);
        std::cerr << "validation empty negative dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor huge_x = make_huge_rows();
        Tensor huge_weight(nullptr, DType::BF16, {1});
        Tensor huge_out = make_huge_rows();
        kernels::rmsnorm(huge_x, huge_weight, 1e-6f, true, nullptr, huge_out, nullptr);
        std::cerr << "validation huge rows: expected overflow_error\n";
        ++f;
    } catch (const std::overflow_error&) {
    } catch (const std::invalid_argument& e) {
        std::cerr << "validation huge rows: expected overflow_error, got invalid_argument: "
                  << e.what() << '\n';
        ++f;
    }

    try {
        Tensor bad_dtype(nullptr, DType::FP32, {4});
        kernels::rmsnorm(bad_dtype, weight, 1e-6f, true, nullptr, out, nullptr);
        std::cerr << "validation dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_weight(nullptr, DType::BF16, {5});
        kernels::rmsnorm(x, bad_weight, 1e-6f, true, nullptr, out, nullptr);
        std::cerr << "validation weight shape: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_weight_rank = weight;
        bad_weight_rank.ne[1] = 2;
        kernels::rmsnorm(x, bad_weight_rank, 1e-6f, true, nullptr, out, nullptr);
        std::cerr << "validation weight rank: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_z(nullptr, DType::BF16, {4, 2});
        kernels::rmsnorm(x, weight, 1e-6f, true, &bad_z, out, nullptr);
        std::cerr << "validation z shape: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_stride = out;
        bad_stride.nb[0] = 4;
        kernels::rmsnorm(x, weight, 1e-6f, true, nullptr, bad_stride, nullptr);
        std::cerr << "validation contiguous: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        kernels::rmsnorm(x, weight, 0.0f, true, nullptr, out, nullptr);
        std::cerr << "validation eps: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        kernels::rmsnorm(x, weight, 1e-6f, true, nullptr, out, nullptr);
        std::cerr << "validation null data: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor z(nullptr, DType::BF16, {4});
        kernels::rmsnorm(x, weight, 1e-6f, false, &z, out, nullptr);
        std::cerr << "validation z null data: expected invalid_argument\n";
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

    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_shape("rmsnorm layer [5120,1]", 5120, 1, 1, true, false, seed, -8.f, 8.f);
        f += one_shape("rmsnorm layer [5120,4096]", 5120, 4096, 1, true, false, seed, -8.f, 8.f);
        f += one_shape("rmsnorm q-norm [256,24,7]", 256, 24, 7, true, false, seed, -8.f, 8.f);
        f += one_shape("rmsnorm k-norm [256,4,7]", 256, 4, 7, true, false, seed, -8.f, 8.f);
        f += one_shape("rmsnorm gdn gated [128,48,7]", 128, 48, 7, false, true, seed, -8.f, 8.f);
        f += one_shape("rmsnorm unaligned [260,3]", 260, 3, 1, true, false, seed, -8.f, 8.f);
    }
    f += one_shape("rmsnorm stress [-60,60]", 260, 3, 1, true, false, 4242u, -60.f, 60.f);
    f += one_shape("rmsnorm gated stress [-60,60]", 128, 48, 7, false, true, 4343u, -60.f, 60.f);
    f += unaligned_data_case();

    std::cout << (f ? "FAIL" : "OK") << " rmsnorm correctness\n";
    return f ? 1 : 0;
}
