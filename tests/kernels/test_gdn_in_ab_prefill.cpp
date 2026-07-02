// Correctness coverage for the GDN in_a/in_b prefill path at Qwen3.6-27B
// shapes: two dense BF16 [48,5120] projections followed by gdn_gating.
#include "qus/kernels/gdn_gating.h"
#include "qus/kernels/linear.h"
#include "kernels/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kHeads  = 48;

Weight dense_bf16_weight(void* data) {
    Weight w{};
    w.qtype             = QType::BF16_CTRL;
    w.layout            = QuantLayout::Contiguous;
    w.q5090_scale_dtype = ScaleDType::None;
    w.payload           = data;
    w.payload_bytes     = static_cast<std::uint64_t>(kHeads) * static_cast<std::uint64_t>(kHidden) *
                      2ULL;
    w.qdata           = data;
    w.scales          = nullptr;
    w.group_size      = 0;
    w.group           = 0;
    w.ndim            = 2;
    w.shape[0]        = kHeads;
    w.shape[1]        = kHidden;
    w.padded_shape[0] = kHeads;
    w.padded_shape[1] = kHidden;
    w.n               = kHeads;
    w.k               = kHidden;
    return w;
}

double softplus(double x) {
    return (x > 20.0) ? x : std::log1p(std::exp(x));
}

void cpu_gdn_in_ab_prefill(const std::vector<float>& x, const std::vector<float>& aw,
                           const std::vector<float>& bw, const std::vector<float>& A_log,
                           const std::vector<float>& dt_bias, std::int32_t T,
                           std::vector<double>& g, std::vector<double>& beta) {
    const std::size_t out_elems = static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T);
    g.assign(out_elems, 0.0);
    beta.assign(out_elems, 0.0);

    for (std::int32_t t = 0; t < T; ++t) {
        const float* x_col = x.data() + static_cast<std::size_t>(t) * kHidden;
        for (std::int32_t h = 0; h < kHeads; ++h) {
            const float* aw_row = aw.data() + static_cast<std::size_t>(h) * kHidden;
            const float* bw_row = bw.data() + static_cast<std::size_t>(h) * kHidden;
            float acc_a         = 0.0f;
            float acc_b         = 0.0f;
            for (std::int32_t k = 0; k < kHidden; ++k) {
                acc_a = std::fma(aw_row[k], x_col[k], acc_a);
                acc_b = std::fma(bw_row[k], x_col[k], acc_b);
            }

            const float a = bf16_to_f32(f32_to_bf16(acc_a));
            const float b = bf16_to_f32(f32_to_bf16(acc_b));
            const std::size_t i =
                static_cast<std::size_t>(t) * kHeads + static_cast<std::size_t>(h);
            g[i] = -std::exp(static_cast<double>(A_log[h])) *
                   softplus(static_cast<double>(a) + static_cast<double>(dt_bias[h]));
            beta[i] = 1.0 / (1.0 + std::exp(-static_cast<double>(b)));
        }
    }
}

int one_shape(std::int32_t T, std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(T));
    std::vector<float> aw(static_cast<std::size_t>(kHeads) * kHidden);
    std::vector<float> bw(static_cast<std::size_t>(kHeads) * kHidden);
    std::vector<float> A_log(kHeads), dt_bias(kHeads);
    fill_uniform(x, seed, -1.0f, 1.0f);
    fill_uniform(aw, seed + 1000u, -0.015f, 0.015f);
    fill_uniform(bw, seed + 2000u, -0.015f, 0.015f);
    fill_uniform(A_log, seed + 3000u, -2.0f, 1.0f);
    fill_uniform(dt_bias, seed + 4000u, -1.0f, 1.0f);
    round_to_bf16(x);
    round_to_bf16(aw);
    round_to_bf16(bw);

    std::vector<double> ref_g, ref_beta;
    cpu_gdn_in_ab_prefill(x, aw, bw, A_log, dt_bias, T, ref_g, ref_beta);

    DBuf dx = to_device_bf16(x), daw = to_device_bf16(aw), dbw = to_device_bf16(bw);
    DBuf dA_log = to_device_f32(A_log), ddt_bias = to_device_f32(dt_bias);
    DBuf da(static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T) * 2u);
    DBuf db(static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T) * 2u);
    DBuf dg(static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T) * sizeof(float));
    DBuf dbeta(static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T) * sizeof(float));

    Tensor tx(dx.p, DType::BF16, {kHidden, T});
    Tensor ta(da.p, DType::BF16, {kHeads, T});
    Tensor tb(db.p, DType::BF16, {kHeads, T});
    Tensor tA_log(dA_log.p, DType::FP32, {kHeads});
    Tensor tdt_bias(ddt_bias.p, DType::FP32, {kHeads});
    Tensor tg(dg.p, DType::FP32, {kHeads, T});
    Tensor tbeta(dbeta.p, DType::FP32, {kHeads, T});
    Weight wa = dense_bf16_weight(daw.p);
    Weight wb = dense_bf16_weight(dbw.p);
    WorkspaceArena ws(64ULL << 20);

    kernels::linear(tx, wa, ta, ws, nullptr);
    kernels::linear(tx, wb, tb, ws, nullptr);
    kernels::gdn_gating(ta, tb, tA_log, tdt_bias, tg, tbeta, nullptr);
    cudaDeviceSynchronize();

    const std::size_t n = static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T);
    const std::string label =
        "gdn_in_ab_prefill two-step [48,5120] T=" + std::to_string(T);
    int failures = 0;
    failures +=
        verify((label + " g").c_str(), from_device_f32(dg, n), ref_g, Tolerance::gdn_output_bf16());
    failures += verify((label + " beta").c_str(), from_device_f32(dbeta, n), ref_beta,
                       Tolerance::gdn_output_bf16());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    failures += one_shape(1, 0x101u);
    failures += one_shape(7, 0x202u);
    failures += one_shape(128, 0x303u);

    std::cout << (failures ? "FAIL" : "OK") << " gdn_in_ab_prefill correctness\n";
    return failures ? 1 : 0;
}
