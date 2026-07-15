// Correctness + coverage for l2norm, against the frozen op-test standard
// (docs/op-development.md): fp64 golden from bf16-rounded inputs, honest
// input ranges, composite tolerance bf16_reduction.
#include "ninfer/ops/l2norm.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

static void cpu_l2norm(const std::vector<float>& x, float eps, std::int32_t d, std::int64_t rows,
                       std::vector<double>& o) {
    for (std::int64_t r = 0; r < rows; ++r) {
        const std::int64_t base = r * d;
        double sumsq            = 0.0;
        for (std::int32_t i = 0; i < d; ++i) {
            const double xv = static_cast<double>(x[base + i]);
            sumsq += xv * xv;
        }
        const double inv = 1.0 / std::sqrt(sumsq + static_cast<double>(eps));
        for (std::int32_t i = 0; i < d; ++i) {
            o[base + i] = static_cast<double>(x[base + i]) * inv;
        }
    }
}

static Tensor tensor_for_shape(void* data, std::int32_t d0, std::int32_t d1, std::int32_t d2,
                               bool rank3) {
    if (rank3) { return Tensor(data, DType::BF16, {d0, d1, d2}); }
    return Tensor(data, DType::BF16, {d0, d1});
}

static int one_shape(const char* tag, std::int32_t d0, std::int32_t d1, std::int32_t d2, bool rank3,
                     std::uint32_t seed, float lo, float hi) {
    const auto rows = static_cast<std::int64_t>(d1) * static_cast<std::int64_t>(d2);
    const auto n    = static_cast<std::size_t>(d0) * static_cast<std::size_t>(rows);
    std::vector<float> x(n);
    fill_uniform(x, seed, lo, hi);
    round_to_bf16(x);

    std::vector<double> ref(n);
    cpu_l2norm(x, 1e-6f, d0, rows, ref);

    DBuf dx     = to_device_bf16(x), dout(n * 2);
    Tensor tx   = tensor_for_shape(dx.p, d0, d1, d2, rank3);
    Tensor tout = tensor_for_shape(dout.p, d0, d1, d2, rank3);
    ops::l2norm(tx, 1e-6f, tout, nullptr);
    cudaDeviceSynchronize();

    return verify(tag, from_device_bf16(dout, n), ref, Tolerance::bf16_reduction());
}

static int near_zero_row_case() {
    constexpr std::int32_t d0   = 128;
    constexpr std::int32_t d1   = 16;
    constexpr std::int32_t d2   = 1;
    constexpr std::int64_t rows = static_cast<std::int64_t>(d1) * d2;
    constexpr std::size_t n     = static_cast<std::size_t>(d0) * rows;

    std::vector<float> x(n);
    fill_uniform(x, 20260626u, -8.f, 8.f);
    for (std::int32_t i = 0; i < d0; ++i) { x[i] = (i & 1) ? -1.0e-7f : 1.0e-7f; }
    round_to_bf16(x);

    std::vector<double> ref(n);
    cpu_l2norm(x, 1e-6f, d0, rows, ref);

    DBuf dx = to_device_bf16(x), dout(n * 2);
    Tensor tx(dx.p, DType::BF16, {d0, d1, d2});
    Tensor tout(dout.p, DType::BF16, {d0, d1, d2});
    ops::l2norm(tx, 1e-6f, tout, nullptr);
    cudaDeviceSynchronize();

    return verify("l2norm near-zero row [128,16,1]", from_device_bf16(dout, n), ref,
                  Tolerance::bf16_reduction());
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
    constexpr std::int32_t d0   = 127;
    constexpr std::int32_t d1   = 5;
    constexpr std::int64_t rows = d1;
    constexpr std::size_t n     = static_cast<std::size_t>(d0) * d1;
    std::vector<float> x(n);
    fill_uniform(x, 3026u, -8.f, 8.f);
    round_to_bf16(x);

    std::vector<double> ref(n);
    cpu_l2norm(x, 1e-6f, d0, rows, ref);

    DBuf dx    = to_device_bf16_unaligned(x), dout((n + 1) * 2);
    auto* xptr = static_cast<unsigned char*>(dx.p) + 2;
    auto* optr = static_cast<unsigned char*>(dout.p) + 2;
    Tensor tx(xptr, DType::BF16, {d0, d1});
    Tensor tout(optr, DType::BF16, {d0, d1});
    ops::l2norm(tx, 1e-6f, tout, nullptr);
    cudaDeviceSynchronize();

    DBuf packed(n * 2);
    cudaMemcpy(packed.p, optr, n * 2, cudaMemcpyDeviceToDevice);
    return verify("l2norm unaligned data [127,5]", from_device_bf16(packed, n), ref,
                  Tolerance::bf16_reduction());
}

static int validation_checks() {
    int f = 0;
    Tensor x(nullptr, DType::BF16, {4});
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
        Tensor empty_out(nullptr, DType::BF16, {1});
        empty_x.ne[0]   = 0;
        empty_out.ne[0] = 0;
        ops::l2norm(empty_x, 1e-6f, empty_out, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "validation empty: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    try {
        Tensor bad_empty_x(nullptr, DType::BF16, {1});
        Tensor bad_empty_out(nullptr, DType::BF16, {1});
        bad_empty_x.ne[0]   = 0;
        bad_empty_x.ne[1]   = -1;
        bad_empty_out.ne[0] = 0;
        bad_empty_out.ne[1] = -1;
        ops::l2norm(bad_empty_x, 1e-6f, bad_empty_out, nullptr);
        std::cerr << "validation empty negative dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor huge_x   = make_huge_rows();
        Tensor huge_out = make_huge_rows();
        ops::l2norm(huge_x, 1e-6f, huge_out, nullptr);
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
        ops::l2norm(bad_dtype, 1e-6f, out, nullptr);
        std::cerr << "validation dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_shape(nullptr, DType::BF16, {5});
        ops::l2norm(x, 1e-6f, bad_shape, nullptr);
        std::cerr << "validation shape: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_stride = out;
        bad_stride.nb[0]  = 4;
        ops::l2norm(x, 1e-6f, bad_stride, nullptr);
        std::cerr << "validation contiguous: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        ops::l2norm(x, 0.0f, out, nullptr);
        std::cerr << "validation eps zero: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        ops::l2norm(x, std::numeric_limits<float>::infinity(), out, nullptr);
        std::cerr << "validation eps finite: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        ops::l2norm(x, 1e-6f, out, nullptr);
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

    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_shape("l2norm [128,16,1]", 128, 16, 1, true, seed, -8.f, 8.f);
        f += one_shape("l2norm [128,16,4096]", 128, 16, 4096, true, seed, -8.f, 8.f);
        f += one_shape("l2norm [127,5]", 127, 5, 1, false, seed, -8.f, 8.f);
    }
    f += near_zero_row_case();
    f += unaligned_data_case();

    std::cout << (f ? "FAIL" : "OK") << " l2norm correctness\n";
    return f ? 1 : 0;
}
