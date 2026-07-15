// Correctness + coverage for gdn_gating, against the frozen op-test standard
// (docs/op-development.md): fp64 golden from bf16-rounded inputs, honest
// GDN gate ranges including the softplus guard, composite tolerance
// fp32_transcendental.
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "core/arena.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

static void cpu_gdn_gating(const std::vector<float>& a, const std::vector<float>& b,
                           const std::vector<float>& A_log, const std::vector<float>& dt_bias,
                           std::int32_t T, std::vector<double>& g, std::vector<double>& beta) {
    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t h = 0; h < 48; ++h) {
            const std::size_t i = static_cast<std::size_t>(t) * 48u + static_cast<std::size_t>(h);
            double sp           = static_cast<double>(a[i]) + static_cast<double>(dt_bias[h]);
            sp                  = (sp > 20.0) ? sp : std::log1p(std::exp(sp));
            g[i]                = -std::exp(static_cast<double>(A_log[h])) * sp;

            const double bv = static_cast<double>(b[i]);
            beta[i]         = 1.0 / (1.0 + std::exp(-bv));
        }
    }
}

static Weight dense_bf16_weight(void* data) {
    Weight w{};
    w.qtype           = QType::BF16_CTRL;
    w.layout          = QuantLayout::Contiguous;
    w.payload         = data;
    w.payload_bytes   = static_cast<std::uint64_t>(48) * 5120u * 2u;
    w.qdata           = data;
    w.ndim            = 2;
    w.shape[0]        = 48;
    w.shape[1]        = 5120;
    w.padded_shape[0] = 48;
    w.padded_shape[1] = 5120;
    w.n               = 48;
    w.k               = 5120;
    return w;
}

static int fused_decode_matches_two_step() {
    std::vector<float> x(5120), aw(48 * 5120), bw(48 * 5120), A_log(48), dt_bias(48);
    fill_uniform(x, 501u, -2.f, 2.f);
    fill_uniform(aw, 502u, -0.05f, 0.05f);
    fill_uniform(bw, 503u, -0.05f, 0.05f);
    fill_uniform(A_log, 504u, -2.f, 1.f);
    fill_uniform(dt_bias, 505u, -1.f, 1.f);
    round_to_bf16(x);
    round_to_bf16(aw);
    round_to_bf16(bw);

    std::vector<float> a(48), b(48);
    for (std::int32_t h = 0; h < 48; ++h) {
        const float* aw_row = aw.data() + static_cast<std::size_t>(h) * 5120u;
        const float* bw_row = bw.data() + static_cast<std::size_t>(h) * 5120u;
        float acc_a         = 0.0f;
        float acc_b         = 0.0f;
        for (std::int32_t k = 0; k < 5120; ++k) {
            acc_a = std::fma(aw_row[k], x[k], acc_a);
            acc_b = std::fma(bw_row[k], x[k], acc_b);
        }
        a[h] = bf16_to_f32(f32_to_bf16(acc_a));
        b[h] = bf16_to_f32(f32_to_bf16(acc_b));
    }
    std::vector<double> ref_g(48), ref_beta(48);
    cpu_gdn_gating(a, b, A_log, dt_bias, 1, ref_g, ref_beta);

    DBuf dx = to_device_bf16(x), daw = to_device_bf16(aw), dbw = to_device_bf16(bw);
    DBuf dA_log = to_device_f32(A_log), ddt_bias = to_device_f32(dt_bias);
    DBuf dg(48 * 4), dbeta(48 * 4);

    Tensor tx(dx.p, DType::BF16, {5120, 1});
    Tensor tA_log(dA_log.p, DType::FP32, {48});
    Tensor tdt_bias(ddt_bias.p, DType::FP32, {48});
    Tensor tg(dg.p, DType::FP32, {48, 1});
    Tensor tbeta(dbeta.p, DType::FP32, {48, 1});
    Weight wa = dense_bf16_weight(daw.p);
    Weight wb = dense_bf16_weight(dbw.p);
    WorkspaceArena ws(4u * 1024u * 1024u);

    ops::gdn_gating_proj(tx, wa, wb, tA_log, tdt_bias, ws, tg, tbeta, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    f += verify("gdn_gating_proj T=1 g", from_device_f32(dg, 48), ref_g,
                Tolerance::fp32_transcendental());
    f += verify("gdn_gating_proj T=1 beta", from_device_f32(dbeta, 48), ref_beta,
                Tolerance::fp32_transcendental());
    return f;
}

static int one_shape(const char* tag, std::int32_t T, std::uint32_t seed, float a_lo, float a_hi) {
    const std::size_t n = static_cast<std::size_t>(48) * static_cast<std::size_t>(T);
    std::vector<float> a(n), b(n), A_log(48), dt_bias(48);
    fill_uniform(a, seed, a_lo, a_hi);
    fill_uniform(b, seed + 1000u, -8.f, 8.f);
    fill_uniform(A_log, seed + 2000u, -2.f, 1.f);
    fill_uniform(dt_bias, seed + 3000u, -1.f, 1.f);
    round_to_bf16(a);
    round_to_bf16(b);

    std::vector<double> ref_g(n), ref_beta(n);
    cpu_gdn_gating(a, b, A_log, dt_bias, T, ref_g, ref_beta);

    DBuf da = to_device_bf16(a), db = to_device_bf16(b);
    DBuf dA_log = to_device_f32(A_log), ddt_bias = to_device_f32(dt_bias);
    DBuf dg(n * sizeof(float)), dbeta(n * sizeof(float));
    Tensor ta(da.p, DType::BF16, {48, T});
    Tensor tb(db.p, DType::BF16, {48, T});
    Tensor tA_log(dA_log.p, DType::FP32, {48});
    Tensor tdt_bias(ddt_bias.p, DType::FP32, {48});
    Tensor tg(dg.p, DType::FP32, {48, T});
    Tensor tbeta(dbeta.p, DType::FP32, {48, T});

    ops::gdn_gating(ta, tb, tA_log, tdt_bias, tg, tbeta, nullptr);
    cudaDeviceSynchronize();

    int f = 0;
    f += verify((std::string(tag) + " g").c_str(), from_device_f32(dg, n), ref_g,
                Tolerance::fp32_transcendental());
    f += verify((std::string(tag) + " beta").c_str(), from_device_f32(dbeta, n), ref_beta,
                Tolerance::fp32_transcendental());
    return f;
}

static int validation_checks() {
    int f = 0;
    Tensor a(nullptr, DType::BF16, {48, 7});
    Tensor b(nullptr, DType::BF16, {48, 7});
    Tensor A_log(nullptr, DType::FP32, {48});
    Tensor dt_bias(nullptr, DType::FP32, {48});
    Tensor g(nullptr, DType::FP32, {48, 7});
    Tensor beta(nullptr, DType::FP32, {48, 7});

    try {
        Tensor empty_a(nullptr, DType::BF16, {48, 1});
        Tensor empty_b(nullptr, DType::BF16, {48, 1});
        Tensor empty_g(nullptr, DType::FP32, {48, 1});
        Tensor empty_beta(nullptr, DType::FP32, {48, 1});
        empty_a.ne[1] = empty_b.ne[1] = empty_g.ne[1] = empty_beta.ne[1] = 0;
        ops::gdn_gating(empty_a, empty_b, A_log, dt_bias, empty_g, empty_beta, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "validation empty T: expected no throw, got " << e.what() << '\n';
        ++f;
    }

    try {
        Tensor bad_empty     = a;
        bad_empty.ne[1]      = 0;
        bad_empty.ne[2]      = -1;
        Tensor bad_empty_out = g;
        bad_empty_out.ne[1]  = 0;
        bad_empty_out.ne[2]  = -1;
        ops::gdn_gating(bad_empty, b, A_log, dt_bias, bad_empty_out, beta, nullptr);
        std::cerr << "validation negative dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor huge_a    = a;
        Tensor huge_b    = b;
        Tensor huge_g    = g;
        Tensor huge_beta = beta;
        for (int d = 0; d < 4; ++d) {
            huge_a.ne[d] = huge_b.ne[d] = huge_g.ne[d] = huge_beta.ne[d] =
                std::numeric_limits<std::int32_t>::max();
        }
        ops::gdn_gating(huge_a, huge_b, A_log, dt_bias, huge_g, huge_beta, nullptr);
        std::cerr << "validation overflow dims: expected overflow_error\n";
        ++f;
    } catch (const std::overflow_error&) {
    } catch (const std::invalid_argument& e) {
        std::cerr << "validation overflow dims: expected overflow_error, got invalid_argument: "
                  << e.what() << '\n';
        ++f;
    }

    try {
        Tensor bad_dtype(nullptr, DType::FP32, {48, 7});
        ops::gdn_gating(bad_dtype, b, A_log, dt_bias, g, beta, nullptr);
        std::cerr << "validation a dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_A(nullptr, DType::BF16, {48});
        ops::gdn_gating(a, b, bad_A, dt_bias, g, beta, nullptr);
        std::cerr << "validation A_log dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_out(nullptr, DType::BF16, {48, 7});
        ops::gdn_gating(a, b, A_log, dt_bias, bad_out, beta, nullptr);
        std::cerr << "validation g dtype: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_heads = a;
        bad_heads.ne[0]  = 47;
        ops::gdn_gating(bad_heads, b, A_log, dt_bias, g, beta, nullptr);
        std::cerr << "validation head dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_T = b;
        bad_T.ne[1]  = 8;
        ops::gdn_gating(a, bad_T, A_log, dt_bias, g, beta, nullptr);
        std::cerr << "validation T dim: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_rank = A_log;
        bad_rank.ne[1]  = 2;
        ops::gdn_gating(a, b, bad_rank, dt_bias, g, beta, nullptr);
        std::cerr << "validation A_log shape: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        Tensor bad_stride = g;
        bad_stride.nb[0]  = 8;
        ops::gdn_gating(a, b, A_log, dt_bias, bad_stride, beta, nullptr);
        std::cerr << "validation contiguous: expected invalid_argument\n";
        ++f;
    } catch (const std::invalid_argument&) {}

    try {
        ops::gdn_gating(a, b, A_log, dt_bias, g, beta, nullptr);
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
    f += fused_decode_matches_two_step();

    for (std::uint32_t seed : {1u, 7u, 99u}) {
        f += one_shape("gdn_gating [48,1]", 1, seed, -8.f, 8.f);
        f += one_shape("gdn_gating [48,7]", 7, seed, -8.f, 8.f);
        f += one_shape("gdn_gating [48,4096]", 4096, seed, -8.f, 8.f);
    }
    f += one_shape("gdn_gating softplus guard [48,4096]", 4096, 4242u, 15.f, 25.f);

    std::cout << (f ? "FAIL" : "OK") << " gdn_gating correctness\n";
    return f ? 1 : 0;
}
