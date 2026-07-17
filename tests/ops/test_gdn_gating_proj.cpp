// Correctness coverage for the exact Qwen3.6-27B two-weight and
// Qwen3.6-35B-A3B contiguous-parent GDN-control projection domains.
#include "ninfer/ops/gdn_gating_proj.h"
#include "core/arena.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden   = 5120;
constexpr std::int32_t kHeads    = 48;
constexpr std::int32_t k35Hidden = 2048;
constexpr std::int32_t k35Heads  = 32;

Weight bf16_weight(void* data, std::int32_t rows, std::int32_t hidden) {
    Weight w{};
    w.qtype         = QType::BF16_CTRL;
    w.layout        = QuantLayout::Contiguous;
    w.payload       = data;
    w.payload_bytes = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(hidden) * 2ULL;
    w.qdata         = data;
    w.scales        = nullptr;
    w.group_size    = 0;
    w.group         = 0;
    w.ndim          = 2;
    w.shape[0]      = rows;
    w.shape[1]      = hidden;
    w.padded_shape[0] = rows;
    w.padded_shape[1] = hidden;
    w.n               = rows;
    w.k               = hidden;
    return w;
}

double softplus_fp64(double x) { return std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x))); }

void cpu_gdn_gating_proj(const std::vector<float>& x, const std::vector<float>& aw,
                         const std::vector<float>& bw, const std::vector<float>& A_log,
                         const std::vector<float>& dt_bias, const std::vector<std::int32_t>& tokens,
                         std::int32_t heads, std::int32_t hidden, std::vector<double>& g,
                         std::vector<double>& beta) {
    const std::size_t out_elems =
        static_cast<std::size_t>(heads) * static_cast<std::size_t>(tokens.size());
    g.assign(out_elems, 0.0);
    beta.assign(out_elems, 0.0);

    for (std::size_t sample = 0; sample < tokens.size(); ++sample) {
        const std::int32_t t = tokens[sample];
        const float* x_col   = x.data() + static_cast<std::size_t>(t) * hidden;
        for (std::int32_t h = 0; h < heads; ++h) {
            const float* aw_row = aw.data() + static_cast<std::size_t>(h) * hidden;
            const float* bw_row = bw.data() + static_cast<std::size_t>(h) * hidden;
            double acc_a        = 0.0;
            double acc_b        = 0.0;
            for (std::int32_t k = 0; k < hidden; ++k) {
                acc_a += static_cast<double>(aw_row[k]) * static_cast<double>(x_col[k]);
                acc_b += static_cast<double>(bw_row[k]) * static_cast<double>(x_col[k]);
            }

            const std::size_t i = sample * heads + static_cast<std::size_t>(h);
            g[i]                = -std::exp(static_cast<double>(A_log[h])) *
                   softplus_fp64(acc_a + static_cast<double>(dt_bias[h]));
            beta[i] = 1.0 / (1.0 + std::exp(-acc_b));
        }
    }
}

int one_shape(std::int32_t T, std::uint32_t seed, std::vector<std::int32_t> sample_tokens = {},
              bool use_graph = false) {
    if (sample_tokens.empty()) {
        sample_tokens.reserve(static_cast<std::size_t>(T));
        for (std::int32_t t = 0; t < T; ++t) { sample_tokens.push_back(t); }
    }
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
    cpu_gdn_gating_proj(x, aw, bw, A_log, dt_bias, sample_tokens, kHeads, kHidden, ref_g, ref_beta);

    DBuf dx = to_device_bf16(x), daw = to_device_bf16(aw), dbw = to_device_bf16(bw);
    DBuf dA_log = to_device_f32(A_log), ddt_bias = to_device_f32(dt_bias);
    DBuf dg(static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T) * sizeof(float));
    DBuf dbeta(static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T) * sizeof(float));

    Tensor tx(dx.p, DType::BF16, {kHidden, T});
    Tensor tA_log(dA_log.p, DType::FP32, {kHeads});
    Tensor tdt_bias(ddt_bias.p, DType::FP32, {kHeads});
    Tensor tg(dg.p, DType::FP32, {kHeads, T});
    Tensor tbeta(dbeta.p, DType::FP32, {kHeads, T});
    Weight wa = bf16_weight(daw.p, kHeads, kHidden);
    Weight wb = bf16_weight(dbw.p, kHeads, kHidden);
    WorkspaceArena ws(64u * 1024u * 1024u);

    if (use_graph) {
        cudaStream_t stream  = nullptr;
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaStreamCreate(&stream);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        ops::gdn_gating_proj(tx, wa, wb, tA_log, tdt_bias, ws, tg, tbeta, stream);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    } else {
        ops::gdn_gating_proj(tx, wa, wb, tA_log, tdt_bias, ws, tg, tbeta, nullptr);
        cudaDeviceSynchronize();
    }

    const std::size_t full_n = static_cast<std::size_t>(kHeads) * static_cast<std::size_t>(T);
    const std::size_t n      = static_cast<std::size_t>(kHeads) * sample_tokens.size();
    const std::vector<double> full_g    = from_device_f32(dg, full_n);
    const std::vector<double> full_beta = from_device_f32(dbeta, full_n);
    std::vector<double> sampled_g(n), sampled_beta(n);
    for (std::size_t sample = 0; sample < sample_tokens.size(); ++sample) {
        const std::size_t source = static_cast<std::size_t>(sample_tokens[sample]) * kHeads;
        for (std::int32_t h = 0; h < kHeads; ++h) {
            sampled_g[sample * kHeads + static_cast<std::size_t>(h)]    = full_g[source + h];
            sampled_beta[sample * kHeads + static_cast<std::size_t>(h)] = full_beta[source + h];
        }
    }
    const std::string label = "gdn_gating_proj [48,5120] T=" + std::to_string(T) +
                              " samples=" + std::to_string(sample_tokens.size());
    const Tolerance tolerance = Tolerance::gdn_control_fp32();
    int failures              = 0;
    failures += verify((label + " g").c_str(), sampled_g, ref_g, tolerance);
    failures += verify((label + " beta").c_str(), sampled_beta, ref_beta, tolerance);
    return failures;
}

int one_shape35(std::int32_t T, std::uint32_t seed, std::vector<std::int32_t> sample_tokens = {},
                bool use_graph = false) {
    if (sample_tokens.empty()) {
        sample_tokens.reserve(static_cast<std::size_t>(T));
        for (std::int32_t t = 0; t < T; ++t) { sample_tokens.push_back(t); }
    }
    std::vector<float> x(static_cast<std::size_t>(k35Hidden) * T);
    std::vector<float> aw(static_cast<std::size_t>(k35Heads) * k35Hidden);
    std::vector<float> bw(static_cast<std::size_t>(k35Heads) * k35Hidden);
    std::vector<float> A_log(k35Heads), dt_bias(k35Heads);
    fill_uniform(x, seed, -1.0f, 1.0f);
    fill_uniform(aw, seed + 1000u, -0.02f, 0.02f);
    fill_uniform(bw, seed + 2000u, -0.02f, 0.02f);
    fill_uniform(A_log, seed + 3000u, -2.0f, 1.0f);
    fill_uniform(dt_bias, seed + 4000u, -1.0f, 1.0f);
    round_to_bf16(x);
    round_to_bf16(aw);
    round_to_bf16(bw);

    std::vector<double> ref_g, ref_beta;
    cpu_gdn_gating_proj(x, aw, bw, A_log, dt_bias, sample_tokens, k35Heads, k35Hidden, ref_g,
                        ref_beta);

    std::vector<float> ab;
    ab.reserve(aw.size() + bw.size());
    ab.insert(ab.end(), aw.begin(), aw.end());
    ab.insert(ab.end(), bw.begin(), bw.end());
    DBuf dx = to_device_bf16(x), dab = to_device_bf16(ab);
    DBuf dA_log = to_device_f32(A_log), ddt_bias = to_device_f32(dt_bias);
    DBuf dg(static_cast<std::size_t>(k35Heads) * T * sizeof(float));
    DBuf dbeta(static_cast<std::size_t>(k35Heads) * T * sizeof(float));

    Tensor tx(dx.p, DType::BF16, {k35Hidden, T});
    Tensor tA_log(dA_log.p, DType::FP32, {k35Heads});
    Tensor tdt_bias(ddt_bias.p, DType::FP32, {k35Heads});
    Tensor tg(dg.p, DType::FP32, {k35Heads, T});
    Tensor tbeta(dbeta.p, DType::FP32, {k35Heads, T});
    Weight parent = bf16_weight(dab.p, 2 * k35Heads, k35Hidden);
    WorkspaceArena ws(ops::gdn_gating_proj_workspace_bytes(T));

    if (use_graph) {
        cudaStream_t stream  = nullptr;
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaStreamCreate(&stream);
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        ops::gdn_gating_proj(tx, parent, tA_log, tdt_bias, ws, tg, tbeta, stream);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    } else {
        ops::gdn_gating_proj(tx, parent, tA_log, tdt_bias, ws, tg, tbeta, nullptr);
        cudaDeviceSynchronize();
    }

    const std::size_t full_n            = static_cast<std::size_t>(k35Heads) * T;
    const std::size_t n                 = static_cast<std::size_t>(k35Heads) * sample_tokens.size();
    const std::vector<double> full_g    = from_device_f32(dg, full_n);
    const std::vector<double> full_beta = from_device_f32(dbeta, full_n);
    std::vector<double> sampled_g(n), sampled_beta(n);
    for (std::size_t sample = 0; sample < sample_tokens.size(); ++sample) {
        const std::size_t source = static_cast<std::size_t>(sample_tokens[sample]) * k35Heads;
        for (std::int32_t h = 0; h < k35Heads; ++h) {
            sampled_g[sample * k35Heads + static_cast<std::size_t>(h)]    = full_g[source + h];
            sampled_beta[sample * k35Heads + static_cast<std::size_t>(h)] = full_beta[source + h];
        }
    }
    const std::string label = "gdn_gating_proj parent [64,2048] T=" + std::to_string(T) +
                              " samples=" + std::to_string(sample_tokens.size());
    const Tolerance tolerance = Tolerance::gdn_control_fp32();
    int failures              = 0;
    failures += verify((label + " g").c_str(), sampled_g, ref_g, tolerance);
    failures += verify((label + " beta").c_str(), sampled_beta, ref_beta, tolerance);
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
    failures += one_shape(2, 0x151u);
    failures += one_shape(6, 0x181u);
    failures += one_shape(7, 0x202u);
    failures += one_shape(8, 0x252u);
    failures += one_shape(128, 0x303u);
    failures += one_shape(512, 0x353u, {0, 257, 511});
    failures += one_shape(1024, 0x404u, {0, 511, 1023}, true);
    failures += one_shape(2048, 0x454u, {0, 1023, 2047});
    failures += one_shape(4096, 0x505u, {0, 2047, 4095});
    failures += one_shape(4097, 0x555u, {0, 2048, 4096});
    failures += one_shape35(1, 0x601u);
    failures += one_shape35(64, 0x651u, {0, 31, 63});
    failures += one_shape35(127, 0x701u, {0, 63, 126});
    failures += one_shape35(128, 0x751u, {0, 63, 127});
    failures += one_shape35(129, 0x801u, {0, 64, 128});
    failures += one_shape35(1024, 0x851u, {0, 511, 1023}, true);
    failures += one_shape35(1025, 0x901u, {0, 512, 1024});
    failures += one_shape35(2049, 0x951u, {0, 1024, 2048});
    failures += one_shape35(4097, 0xa01u, {0, 2048, 4096});

    std::cout << (failures ? "FAIL" : "OK") << " gdn_gating_proj correctness\n";
    return failures ? 1 : 0;
}
