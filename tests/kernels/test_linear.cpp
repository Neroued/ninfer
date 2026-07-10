// Correctness + coverage for linear dense BF16/FP32, against the frozen
// op-test standard (docs/l1-op-test-standard.md): fp64 golden W @ x from
// bf16-rounded inputs and weights, composite tolerance linear_bf16.
#include "qus/kernels/linear.h"
#include "kernels/op_tester.h"
#include "kernels/q5090_pack.h"

#include <algorithm>
#include <cstdlib>
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
    case QType::W8G32_F16S:
        return "w8g32";
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

int one_dense_shape(std::int32_t n, std::int32_t k, std::int32_t t, QType qtype, std::uint32_t seed,
                    float x_abs = 8.0f, float weight_abs = 0.125f,
                    const char* case_name = nullptr) {
    std::vector<float> x(static_cast<std::size_t>(k) * t);
    std::vector<float> weight(static_cast<std::size_t>(n) * k);
    fill_uniform(x, seed, -x_abs, x_abs);
    fill_uniform(weight, seed + 1000u, -weight_abs, weight_abs);
    round_to_bf16(x);
    round_to_bf16(weight);

    std::vector<double> ref;
    cpu_linear_dense(x, weight, n, k, t, ref);

    DBuf dx = to_device_bf16(x);
    DBuf dw = (qtype == QType::FP32_CTRL) ? to_device_f32(weight) : to_device_bf16(weight);
    DBuf dout(static_cast<std::size_t>(n) * t * 2u);

    Tensor tx(dx.p, DType::BF16, {k, t});
    Tensor tout(dout.p, DType::BF16, {n, t});
    WorkspaceArena ws(64ULL << 20);
    kernels::linear(tx, dense_weight(dw.p, qtype, n, k), tout, ws, nullptr);
    cudaDeviceSynchronize();

    const std::string suffix = case_name ? (" " + std::string(case_name)) : std::string();
    const std::string label  = "linear " + std::string(qtype_name(qtype)) + suffix + " [" +
                              std::to_string(n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(t) + " seed=" + std::to_string(seed);
    return verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t), ref,
                  Tolerance::linear_bf16());
}

int fp32_ctrl_first_column_consistency(std::int32_t n, std::int32_t k, std::int32_t t) {
    std::vector<float> x_one(static_cast<std::size_t>(k), 0.0f);
    std::vector<float> x_many(static_cast<std::size_t>(k) * t, 0.0f);
    std::vector<float> weight(static_cast<std::size_t>(n) * k, 0.0f);
    for (std::int32_t kk = 0; kk < std::min<std::int32_t>(k, 16); ++kk) {
        x_one[kk] = 3.0f;
        for (std::int32_t col = 0; col < t; ++col) {
            x_many[static_cast<std::size_t>(col) * k + kk] = 3.0f;
        }
    }
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t kk = 0; kk < std::min<std::int32_t>(k, 16); ++kk) {
            weight[static_cast<std::size_t>(row) * k + kk] = 1.003f;
        }
    }

    DBuf dx_one  = to_device_bf16(x_one);
    DBuf dx_many = to_device_bf16(x_many);
    DBuf dw      = to_device_f32(weight);
    DBuf out_one(static_cast<std::size_t>(n) * 2u);
    DBuf out_many(static_cast<std::size_t>(n) * t * 2u);

    Tensor tx_one(dx_one.p, DType::BF16, {k, 1});
    Tensor tx_many(dx_many.p, DType::BF16, {k, t});
    Tensor tout_one(out_one.p, DType::BF16, {n, 1});
    Tensor tout_many(out_many.p, DType::BF16, {n, t});
    Weight w = dense_weight(dw.p, QType::FP32_CTRL, n, k);
    WorkspaceArena ws(64ULL << 20);

    kernels::linear(tx_one, w, tout_one, ws, nullptr);
    kernels::linear(tx_many, w, tout_many, ws, nullptr);
    cudaDeviceSynchronize();

    const std::vector<double> got_one = from_device_bf16(out_one, n);
    const std::vector<double> got_many =
        from_device_bf16(out_many, static_cast<std::size_t>(n) * t);
    for (std::int32_t row = 0; row < n; ++row) {
        if (got_one[row] != got_many[row]) {
            std::cerr << "linear fp32 first-column consistency [" << n << "," << k << "] T=" << t
                      << ": row " << row << " T=1=" << got_one[row] << " T>1=" << got_many[row]
                      << '\n';
            return 1;
        }
    }
    return 0;
}

int one_quant_shape(QType qtype, std::int32_t n, std::int32_t k,
                    const std::vector<std::int32_t>& ts, std::uint32_t seed, float x_abs = 8.0f,
                    float weight_abs = 0.125f, const char* case_name = nullptr) {
    const std::int32_t max_t = *std::max_element(ts.begin(), ts.end());
    std::vector<float> source_weight(static_cast<std::size_t>(n) * k);
    fill_uniform(source_weight, seed + 2000u, -weight_abs, weight_abs);
    round_to_bf16(source_weight);
    q5090::PackedWeight packed = q5090::pack_row_split_lowbit(source_weight, n, k, qtype);
    std::vector<float>().swap(source_weight);

    std::vector<float> x(static_cast<std::size_t>(k) * max_t);
    fill_uniform(x, seed, -x_abs, x_abs);
    round_to_bf16(x);

    std::vector<double> ref_max_t;
    cpu_linear_dequant(x, packed.dequant, n, k, max_t, ref_max_t);

    DBuf dx = to_device_bf16(x);
    DBuf dweight(packed.payload.size());
    cudaMemcpy(dweight.p, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    WorkspaceArena ws(64ULL << 20);

    int failures = 0;
    for (std::int32_t t : ts) {
        DBuf dout(static_cast<std::size_t>(n) * t * 2u);
        Tensor tx(dx.p, DType::BF16, {k, t});
        Tensor tout(dout.p, DType::BF16, {n, t});
        try {
            kernels::linear(tx, packed.device_weight(dweight.p), tout, ws, nullptr);
            cudaDeviceSynchronize();
        } catch (const std::exception& e) {
            std::cerr << "linear " << qtype_name(qtype) << " [" << n << "," << k << "] T=" << t
                      << " seed=" << seed << ": unexpected exception: " << e.what() << '\n';
            ++failures;
            continue;
        }

        const std::vector<double> ref(ref_max_t.begin(),
                                      ref_max_t.begin() + static_cast<std::size_t>(n) * t);
        const std::string suffix = case_name ? (" " + std::string(case_name)) : std::string();
        const std::string label  = "linear " + std::string(qtype_name(qtype)) + suffix + " [" +
                                  std::to_string(n) + "," + std::to_string(k) +
                                  "] T=" + std::to_string(t) + " seed=" + std::to_string(seed);
        // T > tau (regime_threshold, currently 16) is the LargeT regime -> bf16
        // tensor-core mma GEMM, judged by the normwise linear_tc criterion; T <= tau
        // uses the fp32 multi-step / GEMV path (strict per-element). Keep this bound in
        // sync with detail::regime_threshold in linear_plan.cpp.
        const Tolerance tol = t > 16 ? Tolerance::linear_tc() : Tolerance::linear_bf16();
        failures += verify(label.c_str(), from_device_bf16(dout, static_cast<std::size_t>(n) * t),
                           ref, tol);
    }
    return failures;
}

int one_q4_shape(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    return one_quant_shape(QType::Q4G64_F16S, n, k, {1, 2, 7, 64}, seed);
}

int paired_w8g32_shape(std::int32_t n, std::int32_t k,
                       const std::vector<std::int32_t>& ts, std::uint32_t seed) {
    const std::int32_t max_t = *std::max_element(ts.begin(), ts.end());
    std::vector<float> kw(static_cast<std::size_t>(n) * k);
    std::vector<float> vw(static_cast<std::size_t>(n) * k);
    std::vector<float> x(static_cast<std::size_t>(k) * max_t);
    fill_uniform(kw, seed + 2000u, -0.125f, 0.125f);
    fill_uniform(vw, seed + 3000u, -0.125f, 0.125f);
    fill_uniform(x, seed, -8.0f, 8.0f);
    round_to_bf16(kw);
    round_to_bf16(vw);
    round_to_bf16(x);

    q5090::PackedWeight kpacked =
        q5090::pack_row_split_lowbit(kw, n, k, QType::W8G32_F16S);
    q5090::PackedWeight vpacked =
        q5090::pack_row_split_lowbit(vw, n, k, QType::W8G32_F16S);
    std::vector<double> kref;
    std::vector<double> vref;
    cpu_linear_dequant(x, kpacked.dequant, n, k, max_t, kref);
    cpu_linear_dequant(x, vpacked.dequant, n, k, max_t, vref);

    DBuf dx = to_device_bf16(x);
    DBuf dkw(kpacked.payload.size());
    DBuf dvw(vpacked.payload.size());
    cudaMemcpy(dkw.p, kpacked.payload.data(), kpacked.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dvw.p, vpacked.payload.data(), vpacked.payload.size(), cudaMemcpyHostToDevice);
    WorkspaceArena ws(64ULL << 20);

    int failures = 0;
    for (std::int32_t t : ts) {
        DBuf dkout(static_cast<std::size_t>(n) * t * 2u);
        DBuf dvout(static_cast<std::size_t>(n) * t * 2u);
        Tensor tx(dx.p, DType::BF16, {k, t});
        Tensor tk(dkout.p, DType::BF16, {n, t});
        Tensor tv(dvout.p, DType::BF16, {n, t});
        kernels::linear_w8g32_kv_pair(tx, kpacked.device_weight(dkw.p),
                                      vpacked.device_weight(dvw.p), tk, tv, ws, nullptr);
        cudaDeviceSynchronize();
        const std::size_t count = static_cast<std::size_t>(n) * t;
        const std::vector<double> kr(kref.begin(), kref.begin() + count);
        const std::vector<double> vr(vref.begin(), vref.begin() + count);
        const std::string base = "linear w8g32 paired [" + std::to_string(n) + "," +
                                 std::to_string(k) + "] T=" + std::to_string(t);
        failures += verify((base + " K").c_str(), from_device_bf16(dkout, count), kr,
                           Tolerance::linear_tc());
        failures += verify((base + " V").c_str(), from_device_bf16(dvout, count), vr,
                           Tolerance::linear_tc());
    }
    return failures;
}

int dense_metadata_validation() {
    int f = 0;
    std::vector<float> x(4, 1.0f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);
    DBuf dout(2 * 2u);
    Tensor tx(dx.p, DType::BF16, {4, 1});
    Tensor tout(dout.p, DType::BF16, {2, 1});
    WorkspaceArena ws(64ULL << 20);

    Weight w = dense_weight(dx.p, QType::BF16_CTRL, 2, 4);

    try {
        Weight bad            = w;
        bad.q5090_scale_dtype = ScaleDType::FP16;
        kernels::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear dense scale dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = w;
        bad.padded_shape[0] = 64;
        kernels::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear dense padded N: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = w;
        bad.padded_shape[1] = 64;
        kernels::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear dense padded K: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = w;
        bad.padded_shape[2] = 2;
        kernels::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear dense padded trailing dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor empty_tx(nullptr, DType::BF16, {4, 1});
        Tensor empty_out(nullptr, DType::BF16, {2, 1});
        empty_tx.ne[1]  = 0;
        empty_out.ne[1] = 0;
        kernels::linear(empty_tx, w, empty_out, ws, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "linear dense empty T: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    return f;
}

int dense_alignment_validation() {
    int f                    = 0;
    constexpr std::int32_t n = 2;
    constexpr std::int32_t k = 5;
    DBuf dx(static_cast<std::size_t>(k) * 2u + 16u);
    DBuf dw(static_cast<std::size_t>(n) * k * 2u + 16u);
    DBuf dout(static_cast<std::size_t>(n) * 2u + 16u);
    cudaMemset(dx.p, 0, dx.bytes);
    cudaMemset(dw.p, 0, dw.bytes);
    cudaMemset(dout.p, 0, dout.bytes);

    auto* x_bytes   = static_cast<unsigned char*>(dx.p);
    auto* w_bytes   = static_cast<unsigned char*>(dw.p);
    auto* out_bytes = static_cast<unsigned char*>(dout.p);
    Tensor tx(dx.p, DType::BF16, {k, 1});
    Tensor tout(dout.p, DType::BF16, {n, 1});
    Weight w = dense_weight(dw.p, QType::BF16_CTRL, n, k);
    WorkspaceArena ws(64ULL << 20);

    try {
        Tensor bad_x(x_bytes + 2, DType::BF16, {k, 1});
        kernels::linear(bad_x, w, tout, ws, nullptr);
        cudaDeviceSynchronize();
        std::cerr << "linear dense unaligned x: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad_w = dense_weight(w_bytes + 2, QType::BF16_CTRL, n, k);
        kernels::linear(tx, bad_w, tout, ws, nullptr);
        cudaDeviceSynchronize();
        std::cerr << "linear dense unaligned weight: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_out(out_bytes + 2, DType::BF16, {n, 1});
        kernels::linear(tx, w, bad_out, ws, nullptr);
        cudaDeviceSynchronize();
        std::cerr << "linear dense unaligned out: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    return f;
}

int lowbit_metadata_validation(QType qtype) {
    int f                    = 0;
    constexpr std::int32_t n = 64;
    constexpr std::int32_t k = 64;
    std::vector<float> source(static_cast<std::size_t>(n) * k);
    fill_uniform(source, 123u, -0.125f, 0.125f);
    round_to_bf16(source);
    q5090::PackedWeight packed = q5090::pack_row_split_lowbit(source, n, k, qtype);

    std::vector<float> x(k, 1.0f);
    round_to_bf16(x);
    DBuf dx = to_device_bf16(x);
    DBuf dout(static_cast<std::size_t>(n) * 2u);
    Tensor tx(dx.p, DType::BF16, {k, 1});
    Tensor tout(dout.p, DType::BF16, {n, 1});
    WorkspaceArena ws(64ULL << 20);
    const std::uint32_t expected_group = qtype == QType::W8G32_F16S ? 32u : 64u;

    try {
        Weight bad        = packed.device_weight(dx.p);
        bad.payload_bytes = 0;
        kernels::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear " << qtype_name(qtype) << " payload size: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad    = packed.device_weight(dx.p);
        bad.group     = expected_group == 32u ? 64 : 32;
        bad.group_size = expected_group == 32u ? 64u : 32u;
        kernels::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear " << qtype_name(qtype) << " group size: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad            = packed.device_weight(dx.p);
        bad.q5090_scale_dtype = ScaleDType::None;
        kernels::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear " << qtype_name(qtype) << " scale dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Weight bad          = packed.device_weight(dx.p);
        bad.padded_shape[1] = bad.padded_shape[1] + 128;
        kernels::linear(tx, bad, tout, ws, nullptr);
        std::cerr << "linear " << qtype_name(qtype) << " padded K: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    if (qtype == QType::W8G32_F16S) {
        try {
            Weight bad            = packed.device_weight(dx.p);
            bad.qhigh             = dx.p;
            bad.high_plane_bytes  = 1;
            kernels::linear(tx, bad, tout, ws, nullptr);
            std::cerr << "linear w8g32 high plane: expected invalid_argument\n";
            ++f;
        } catch (const std::invalid_argument&) {}
    }

    try {
        Tensor empty_tx  = tx;
        Tensor empty_out = tout;
        empty_tx.ne[1]   = 0;
        empty_out.ne[1]  = 0;
        Weight bad       = packed.device_weight(nullptr);
        kernels::linear(empty_tx, bad, empty_out, ws, nullptr);
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
        kernels::linear(empty_tx, packed.device_weight(dx.p), empty_out, ws, nullptr);
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

    if (std::getenv("QUS_LINEAR_TEST_W8G32_ONLY") != nullptr) {
        int f = 0;
        f += one_quant_shape(QType::W8G32_F16S, 64, 256, {128}, 119u);
        f += one_quant_shape(QType::W8G32_F16S, 70, 130, {17}, 121u);
        f += paired_w8g32_shape(64, 256, {17, 128}, 127u);
        std::cout << (f ? "FAIL" : "OK") << " linear W8G32 focused correctness\n";
        return f ? 1 : 0;
    }

    int f = 0;
    f += dense_metadata_validation();
    f += dense_alignment_validation();
    f += lowbit_metadata_validation(QType::Q4G64_F16S);
    f += lowbit_metadata_validation(QType::Q5G64_F16S);
    f += lowbit_metadata_validation(QType::Q6G64_F16S);
    f += lowbit_metadata_validation(QType::W8G32_F16S);

    for (std::uint32_t seed : {1u, 7u, 99u}) {
        for (QType qtype : {QType::BF16_CTRL, QType::FP32_CTRL}) {
            f += one_dense_shape(48, 5120, 1, qtype, seed);
            f += one_dense_shape(48, 5120, 7, qtype, seed);
        }
    }
    for (QType qtype : {QType::BF16_CTRL, QType::FP32_CTRL}) {
        if (qtype == QType::BF16_CTRL) { f += one_dense_shape(64, 64, 32, qtype, 11u); }
        f += one_dense_shape(64, 96, 64, qtype, 13u);
        f += one_dense_shape(96, 513, 33, qtype, 15u);
        f += one_dense_shape(5120, 6144, 1, qtype, 17u);
        f += one_dense_shape(5120, 6144, 7, qtype, 17u);
        f += one_dense_shape(37, 513, 19, qtype, 19u);
        f += one_dense_shape(80, 130, 17, qtype, 101u, 8.0f, 1.5f, "stress");
    }
    f += one_dense_shape(96, 128, 512, QType::BF16_CTRL, 53u);
    f += fp32_ctrl_first_column_consistency(48, 64, 7);
    f += fp32_ctrl_first_column_consistency(64, 64, 32);
    f += one_quant_shape(QType::Q4G64_F16S, 70, 130, {1, 5}, 41u);
    f += one_quant_shape(QType::Q4G64_F16S, 70, 130, {1, 9}, 103u, 8.0f, 1.25f, "stress");
    f += one_quant_shape(QType::Q5G64_F16S, 70, 130, {1, 9}, 107u, 8.0f, 1.25f, "stress");
    f += one_quant_shape(QType::Q6G64_F16S, 70, 130, {1, 9}, 109u, 8.0f, 1.25f, "stress");
    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_quant_shape(QType::W8G32_F16S, 70, 130, {1, 2, 5, 17}, seed);
        f += one_quant_shape(QType::W8G32_F16S, 70, 131, {1, 2, 5, 17}, seed);
        for (auto [n, k] : {std::pair<std::int32_t, std::int32_t>{5120, 10240},
                            std::pair<std::int32_t, std::int32_t>{14336, 5120},
                            std::pair<std::int32_t, std::int32_t>{5120, 6144},
                            std::pair<std::int32_t, std::int32_t>{34816, 5120},
                            std::pair<std::int32_t, std::int32_t>{5120, 17408}}) {
            f += one_quant_shape(QType::W8G32_F16S, n, k, {1, 2, 5, 17}, seed);
        }
    }
    f += one_quant_shape(QType::W8G32_F16S, 96, 130, {1, 2, 5, 17}, 113u, 8.0f, 1.25f,
                         "stress");
    f += paired_w8g32_shape(64, 256, {17, 128}, 127u);
    f += one_quant_shape(QType::Q4G64_F16S, 128, 128, {512}, 43u);
    f += one_quant_shape(QType::Q6G64_F16S, 128, 128, {512}, 47u);
    for (auto [n, k] : {std::pair<std::int32_t, std::int32_t>{34816, 5120},
                        std::pair<std::int32_t, std::int32_t>{7168, 5120},
                        std::pair<std::int32_t, std::int32_t>{4096, 5120}}) {
        f += one_quant_shape(QType::Q4G64_F16S, n, k, {1}, 23u);
    }
    for (auto [n, k] : {std::pair<std::int32_t, std::int32_t>{6144, 5120},
                        std::pair<std::int32_t, std::int32_t>{7168, 5120},
                        std::pair<std::int32_t, std::int32_t>{5120, 6144},
                        std::pair<std::int32_t, std::int32_t>{5120, 17408}}) {
        f += one_quant_shape(QType::Q5G64_F16S, n, k, {1, 2, 3, 4, 5, 6, 7}, 29u);
    }
    f += one_quant_shape(QType::Q6G64_F16S, 4096, 5120, {1, 7}, 31u);

    // --- lm_head vocab-projection family (T1 warp-per-row GEMV) ------------------
    // n>=65536, k=5120 routes to LmHeadVocabx5120. The Q6 kernel takes N at runtime
    // (full head 248320 and any draft N), and the Q4 kernel serves the embedded Q4
    // draft head. Cover the exact production N plus a smaller N to guard the
    // runtime-N grid parameterization.
    f += one_quant_shape(QType::Q6G64_F16S, 248320, 5120, {1}, 211u); // full lm_head
    f += one_quant_shape(QType::Q6G64_F16S, 65536, 5120, {1}, 227u);  // runtime-N Q6
    f += one_quant_shape(QType::Q4G64_F16S, 131072, 5120, {1}, 223u); // Q4 draft head

    // --- prefill (T>1) coverage of the multi-step GEMM path ---------------------
    // The fp64 CPU golden is O(N*K*T), so large families run at small/medium T and
    // the long-T (512, 2048) + tile-boundary cases run on small shapes.
    f += one_quant_shape(QType::Q4G64_F16S, 34816, 5120, {2, 8, 64}, 61u);
    f += one_quant_shape(QType::Q5G64_F16S, 5120, 17408, {2, 8, 64}, 67u);
    // Real prefill projection shapes (unregistered families -> multi-step path).
    f += one_quant_shape(QType::Q4G64_F16S, 17408, 5120, {2, 8, 64}, 71u);   // gate/up
    f += one_quant_shape(QType::Q4G64_F16S, 2048, 5120, {2, 8, 64, 512}, 73u); // gdn in_q/in_k
    f += one_quant_shape(QType::Q4G64_F16S, 1024, 5120, {2, 9, 512}, 79u);   // k/v proj
    f += one_quant_shape(QType::Q5G64_F16S, 6144, 5120, {2, 8, 64, 512}, 83u); // q/gate/in_v/in_z
    f += one_quant_shape(QType::Q5G64_F16S, 5120, 6144, {2, 8, 64, 512}, 89u); // o/out proj
    f += one_quant_shape(QType::Q6G64_F16S, 4096, 5120, {2, 8, 64}, 131u);
    // Long T + column-tile boundary (kTt=8) + unaligned N/K on small shapes.
    f += one_quant_shape(QType::Q4G64_F16S, 256, 256, {512, 2048}, 91u);
    f += one_quant_shape(QType::Q5G64_F16S, 130, 256, {7, 8, 9, 65, 2048}, 93u);
    f += one_quant_shape(QType::Q6G64_F16S, 192, 320, {7, 9, 63, 64, 65}, 97u);
    f += one_quant_shape(QType::Q4G64_F16S, 320, 384, {2, 33, 513}, 99u, 8.0f, 1.25f, "stress");
    // LargeT (T > tau) exercises the tensor-core GEMM; cover the big N and big K dims.
    f += one_quant_shape(QType::Q4G64_F16S, 17408, 5120, {128}, 137u); // gate/up (K=5120)
    f += one_quant_shape(QType::Q5G64_F16S, 5120, 17408, {128}, 139u); // mlp_down (K=17408)
    // Regression: k % 8 != 0 at LargeT must fall back to the multi-step GEMV (the mma
    // path streams x with 16B cp.async and needs 16B-aligned token rows; k=130 would
    // otherwise drop the k%8 tail and issue misaligned copies). N=k=130 also stress
    // the N tail; Q5 exercises the high plane.
    f += one_quant_shape(QType::Q5G64_F16S, 130, 130, {64, 512}, 151u);

    std::cout << (f ? "FAIL" : "OK") << " linear correctness\n";
    return f ? 1 : 0;
}
