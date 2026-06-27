// Correctness + coverage for linear dense BF16/FP32, against the frozen
// op-test standard (docs/l1-op-test-standard.md): fp64 golden W @ x from
// bf16-rounded inputs and weights, composite tolerance linear_bf16.
#include "qus/kernels/linear.h"
#include "kernels/op_tester.h"
#include "kernels/q5090_pack.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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
    case QType::Q4G64_F16S:
        return "q4";
    case QType::Q5G64_F16S:
        return "q5";
    case QType::Q6G64_F16S:
        return "q6";
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

void cpu_linear_dequant(const std::vector<float>& x, const std::vector<float>& weight,
                        std::int32_t n, std::int32_t k, std::int32_t t, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(n) * t, 0.0);
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const std::int32_t n_threads =
        std::min<std::int32_t>(n, static_cast<std::int32_t>(std::min<unsigned>(hw, 16u)));
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(n_threads));

    for (std::int32_t tid = 0; tid < n_threads; ++tid) {
        const std::int32_t row_begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(n) * tid) / n_threads);
        const std::int32_t row_end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(n) * (tid + 1)) / n_threads);
        threads.emplace_back([&, row_begin, row_end] {
            std::unique_ptr<double[]> acc(new double[static_cast<std::size_t>(t)]);
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                std::fill(acc.get(), acc.get() + t, 0.0);
                const float* wrow = weight.data() + static_cast<std::size_t>(row) * k;
                for (std::int32_t kk = 0; kk < k; ++kk) {
                    const double wv = static_cast<double>(wrow[kk]);
                    for (std::int32_t col = 0; col < t; ++col) {
                        acc[col] +=
                            wv * static_cast<double>(x[static_cast<std::size_t>(col) * k + kk]);
                    }
                }
                for (std::int32_t col = 0; col < t; ++col) {
                    out[static_cast<std::size_t>(col) * n + row] = acc[col];
                }
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
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

int one_quant_shape(QType qtype, std::int32_t n, std::int32_t k,
                    const std::vector<std::int32_t>& ts, std::uint32_t seed) {
    const std::int32_t max_t = *std::max_element(ts.begin(), ts.end());
    std::vector<float> source_weight(static_cast<std::size_t>(n) * k);
    fill_uniform(source_weight, seed + 2000u, -0.125f, 0.125f);
    round_to_bf16(source_weight);
    q5090::PackedWeight packed = q5090::pack_tile_lowbit(source_weight, n, k, qtype);
    std::vector<float>().swap(source_weight);

    std::vector<float> x(static_cast<std::size_t>(k) * max_t);
    fill_uniform(x, seed, -3.0f, 3.0f);
    round_to_bf16(x);

    std::vector<double> ref_max_t;
    cpu_linear_dequant(x, packed.dequant, n, k, max_t, ref_max_t);

    DBuf dx = to_device_bf16(x);
    DBuf dweight(packed.payload.size());
    cudaMemcpy(dweight.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);

    int failures = 0;
    for (std::int32_t t : ts) {
        DBuf dout(static_cast<std::size_t>(n) * t * 2u);
        Tensor tx(dx.p, DType::BF16, {k, t});
        Tensor tout(dout.p, DType::BF16, {n, t});
        try {
            kernels::linear(tx, packed.device_weight(dweight.p), tout, nullptr);
            cudaDeviceSynchronize();
        } catch (const std::exception& e) {
            std::cerr << "linear " << qtype_name(qtype) << " [" << n << "," << k << "] T=" << t
                      << " seed=" << seed << ": unexpected exception: " << e.what() << '\n';
            ++failures;
            continue;
        }

        const std::vector<double> ref(ref_max_t.begin(),
                                      ref_max_t.begin() + static_cast<std::size_t>(n) * t);
        const std::string label = "linear " + std::string(qtype_name(qtype)) + " [" +
                                  std::to_string(n) + "," + std::to_string(k) +
                                  "] T=" + std::to_string(t) + " seed=" + std::to_string(seed);
        failures += verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t),
                           ref, Tolerance::linear_bf16());
    }
    return failures;
}

int one_q4_shape(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    return one_quant_shape(QType::Q4G64_F16S, n, k, {1, 2, 7, 64}, seed);
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

    try {
        Tensor empty_tx(nullptr, DType::BF16, {4, 1});
        Tensor empty_out(nullptr, DType::BF16, {2, 1});
        empty_tx.ne[1]  = 0;
        empty_out.ne[1] = 0;
        kernels::linear(empty_tx, w, empty_out, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "linear dense empty T: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    return f;
}

int lowbit_metadata_validation(QType qtype) {
    int f                    = 0;
    constexpr std::int32_t n = 64;
    constexpr std::int32_t k = 64;
    std::vector<float> source(static_cast<std::size_t>(n) * k);
    fill_uniform(source, 123u, -0.125f, 0.125f);
    round_to_bf16(source);
    q5090::PackedWeight packed = q5090::pack_tile_lowbit(source, n, k, qtype);

    std::vector<float> x(k, 1.0f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);
    DBuf dout(static_cast<std::size_t>(n) * 2u);
    Tensor tx(dx.p, DType::BF16, {k, 1});
    Tensor tout(dout.p, DType::BF16, {n, 1});

    try {
        Weight bad        = packed.device_weight(dx.p);
        bad.payload_bytes = 0;
        kernels::linear(tx, bad, tout, nullptr);
        std::cerr << "linear " << qtype_name(qtype) << " payload size: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor empty_tx  = tx;
        Tensor empty_out = tout;
        empty_tx.ne[1]   = 0;
        empty_out.ne[1]  = 0;
        Weight bad       = packed.device_weight(nullptr);
        kernels::linear(empty_tx, bad, empty_out, nullptr);
        std::cerr << "linear " << qtype_name(qtype)
                  << " empty T null payload: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor empty_tx  = tx;
        Tensor empty_out = tout;
        empty_tx.data    = nullptr;
        empty_out.data   = nullptr;
        empty_tx.ne[1]   = 0;
        empty_out.ne[1]  = 0;
        kernels::linear(empty_tx, packed.device_weight(dx.p), empty_out, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "linear " << qtype_name(qtype) << " empty T: expected no throw, got "
                  << e.what() << '\n';
        ++f;
    }

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
    f += lowbit_metadata_validation(QType::Q4G64_F16S);
    f += lowbit_metadata_validation(QType::Q5G64_F16S);
    f += lowbit_metadata_validation(QType::Q6G64_F16S);

    for (std::uint32_t seed : {1u, 7u, 99u}) {
        for (QType qtype : {QType::BF16_CTRL, QType::FP32_CTRL}) {
            f += one_dense_shape(48, 5120, 1, qtype, seed);
            f += one_dense_shape(48, 5120, 7, qtype, seed);
        }
    }
    for (QType qtype : {QType::BF16_CTRL, QType::FP32_CTRL}) {
        f += one_dense_shape(5120, 6144, 1, qtype, 17u);
        f += one_dense_shape(5120, 6144, 7, qtype, 17u);
        f += one_dense_shape(37, 513, 19, qtype, 19u);
    }
    for (auto [n, k] : {std::pair<std::int32_t, std::int32_t>{6144, 5120},
                        std::pair<std::int32_t, std::int32_t>{1024, 5120},
                        std::pair<std::int32_t, std::int32_t>{2048, 5120},
                        std::pair<std::int32_t, std::int32_t>{17408, 5120}}) {
        f += one_q4_shape(n, k, 23u);
    }
    for (auto [n, k] : {std::pair<std::int32_t, std::int32_t>{6144, 5120},
                        std::pair<std::int32_t, std::int32_t>{5120, 6144},
                        std::pair<std::int32_t, std::int32_t>{5120, 17408}}) {
        f += one_quant_shape(QType::Q5G64_F16S, n, k, {1, 7}, 29u);
    }
    f += one_quant_shape(QType::Q6G64_F16S, 4096, 5120, {1, 7}, 31u);

    std::cout << (f ? "FAIL" : "OK") << " linear correctness\n";
    return f ? 1 : 0;
}
