// Correctness + coverage for linear dense BF16/FP32, against the frozen
// op-test standard (docs/l1-op-test-standard.md): fp64 golden W @ x from
// bf16-rounded inputs and weights, composite tolerance linear_bf16.
#include "qus/kernels/linear.h"
#include "kernels/op_tester.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

const char* qtype_name(QType qtype) {
    switch (qtype) {
    case QType::BF16_CTRL:
        return "bf16";
    case QType::FP32_CTRL:
        return "fp32";
    default:
        return "unsupported";
    }
}

Weight dense_weight(void* data, QType qtype, std::int32_t n, std::int32_t k) {
    Weight w{};
    w.qtype             = qtype;
    w.layout            = QuantLayout::Contiguous;
    w.q5090_scale_dtype = ScaleDType::None;
    w.payload           = data;
    w.payload_bytes     = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(k) *
                      ((qtype == QType::FP32_CTRL) ? 4u : 2u);
    w.qdata           = data;
    w.scales          = nullptr;
    w.group_size      = 0;
    w.group           = 0;
    w.ndim            = 2;
    w.shape[0]        = n;
    w.shape[1]        = k;
    w.padded_shape[0] = n;
    w.padded_shape[1] = k;
    w.n               = n;
    w.k               = k;
    return w;
}

void cpu_linear_dense(const std::vector<float>& x, const std::vector<float>& weight, std::int32_t n,
                      std::int32_t k, std::int32_t t, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(n) * t, 0.0);
    for (std::int32_t col = 0; col < t; ++col) {
        double* dst = out.data() + static_cast<std::size_t>(col) * n;
        for (std::int32_t row = 0; row < n; ++row) {
            const float* wrow = weight.data() + static_cast<std::size_t>(row) * k;
            for (std::int32_t kk = 0; kk < k; ++kk) {
                const double xv = static_cast<double>(x[static_cast<std::size_t>(col) * k + kk]);
                dst[row] += static_cast<double>(wrow[kk]) * xv;
            }
        }
    }
}

int one_dense_shape(std::int32_t n, std::int32_t k, std::int32_t t, QType qtype,
                    std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(k) * t);
    std::vector<float> weight(static_cast<std::size_t>(n) * k);
    fill_uniform(x, seed, -3.0f, 3.0f);
    fill_uniform(weight, seed + 1000u, -0.125f, 0.125f);
    round_to_bf16(x);
    round_to_bf16(weight);

    std::vector<double> ref;
    cpu_linear_dense(x, weight, n, k, t, ref);

    DBuf dx = to_device_bf16(x);
    DBuf dw = (qtype == QType::FP32_CTRL) ? to_device_f32(weight) : to_device_bf16(weight);
    DBuf dout(static_cast<std::size_t>(n) * t * 2u);

    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor tout(dout.p, DType::BF16, {n, t});
    kernels::linear(tx, dense_weight(dw.p, qtype, n, k), tout, nullptr);
    cudaDeviceSynchronize();

    const std::string label = "linear " + std::string(qtype_name(qtype)) + " [" +
                              std::to_string(n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(t) + " seed=" + std::to_string(seed);
    return verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t), ref,
                  Tolerance::linear_bf16());
}

int unsupported_qtype_validation() {
    int f = 0;
    std::vector<float> x(4, 1.0f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);
    DBuf dout(2 * 2u);
    Tensor tx(dx.p, DType::BF16, {4, 1});
    Tensor tout(dout.p, DType::BF16, {2, 1});

    Weight w = dense_weight(dx.p, QType::BF16_CTRL, 2, 4);
    w.qtype  = QType::W8G128_F16S;
    try {
        kernels::linear(tx, w, tout, nullptr);
        std::cerr << "linear unsupported W8 qtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    w.qtype = QType::Q4G64_F16S;
    try {
        kernels::linear(tx, w, tout, nullptr);
        std::cerr << "linear unsupported Q4 qtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    return f;
}

int dense_metadata_validation() {
    int f = 0;
    std::vector<float> x(4, 1.0f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);
    DBuf dout(2 * 2u);
    Tensor tx(dx.p, DType::BF16, {4, 1});
    Tensor tout(dout.p, DType::BF16, {2, 1});

    Weight w = dense_weight(dx.p, QType::BF16_CTRL, 2, 4);

    try {
        Weight bad            = w;
        bad.q5090_scale_dtype = ScaleDType::FP16;
        kernels::linear(tx, bad, tout, nullptr);
        std::cerr << "linear dense scale dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = w;
        bad.padded_shape[0] = 64;
        kernels::linear(tx, bad, tout, nullptr);
        std::cerr << "linear dense padded N: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = w;
        bad.padded_shape[1] = 64;
        kernels::linear(tx, bad, tout, nullptr);
        std::cerr << "linear dense padded K: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = w;
        bad.padded_shape[2] = 2;
        kernels::linear(tx, bad, tout, nullptr);
        std::cerr << "linear dense padded trailing dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    return f;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int f = 0;
    f += unsupported_qtype_validation();
    f += dense_metadata_validation();

    for (std::uint32_t seed : {1u, 7u, 99u}) {
        for (QType qtype : {QType::BF16_CTRL, QType::FP32_CTRL}) {
            f += one_dense_shape(48, 5120, 1, qtype, seed);
            f += one_dense_shape(48, 5120, 7, qtype, seed);
        }
    }
    for (QType qtype : {QType::BF16_CTRL, QType::FP32_CTRL}) {
        f += one_dense_shape(5120, 6144, 1, qtype, 17u);
        f += one_dense_shape(5120, 6144, 7, qtype, 17u);
    }

    std::cout << (f ? "FAIL" : "OK") << " linear correctness\n";
    return f ? 1 : 0;
}
