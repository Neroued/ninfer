#include "targets/qwen3_6_27b_rtx5090/impl/config.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/ops.h"
#include "targets/qwen3_6_27b_rtx5090/impl/kernels/sampling/sampling.h"
#include "kernels/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace target = ninfer::targets::qwen3_6_27b_rtx5090::detail;

namespace {

DBuf to_device_i64(const std::vector<std::int64_t>& h) {
    DBuf d(h.size() * sizeof(std::int64_t));
    cudaMemcpy(d.p, h.data(), h.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice);
    return d;
}

DBuf to_device_config(const kernels::SamplingConfig& cfg) {
    DBuf d(sizeof(kernels::SamplingConfig));
    cudaMemcpy(d.p, &cfg, sizeof(cfg), cudaMemcpyHostToDevice);
    return d;
}

const kernels::SamplingConfig* config_ptr(const DBuf& d) {
    return static_cast<const kernels::SamplingConfig*>(d.p);
}

// Zero logits buffer [vocab, cols]; unused by the greedy accept branch but
// required by the wrapper's shape validation.
DBuf zero_logits(int vocab, int cols) {
    return to_device_bf16(std::vector<float>(static_cast<std::size_t>(vocab) * cols, 0.0f));
}

DBuf to_device_u16(const std::vector<std::uint16_t>& h) {
    DBuf d(h.size() * sizeof(std::uint16_t));
    cudaMemcpy(d.p, h.data(), h.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    return d;
}

std::vector<std::int64_t> from_device_i64(const DBuf& d, std::size_t n) {
    std::vector<std::int64_t> out(n);
    cudaMemcpy(out.data(), d.p, n * sizeof(std::int64_t), cudaMemcpyDeviceToHost);
    return out;
}

std::vector<std::uint16_t> from_device_u16(const DBuf& d, std::size_t n) {
    std::vector<std::uint16_t> out(n);
    cudaMemcpy(out.data(), d.p, n * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    return out;
}

template <class T>
int expect_eq(const std::string& label, const std::vector<T>& got, const std::vector<T>& expected) {
    if (got.size() != expected.size()) {
        std::cerr << label << ": size mismatch got=" << got.size()
                  << " expected=" << expected.size() << '\n';
        return 1;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i] != expected[i]) {
            std::cerr << label << ": mismatch at " << i << " got=" << got[i]
                      << " expected=" << expected[i] << '\n';
            return 1;
        }
    }
    return 0;
}

int prepare_verify_case() {
    constexpr int k = 5;
    auto d_token    = to_device_i32({42});
    auto d_drafts   = to_device_i32({101, 102, 103, 104, 105});
    auto d_length   = to_device_i32({11});
    DBuf d_window_base(sizeof(std::int32_t));
    DBuf d_verify((k + 1) * sizeof(std::int32_t));
    DBuf d_positions((k + 1) * sizeof(std::int32_t));

    Tensor token(d_token.p, DType::I32, {1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor window_base(d_window_base.p, DType::I32, {1});
    Tensor verify(d_verify.p, DType::I32, {k + 1});
    Tensor positions(d_positions.p, DType::I32, {k + 1});
    kernels::mtp_prepare_verify_inputs(token, drafts, length, window_base, verify, positions,
                                       nullptr);
    cudaDeviceSynchronize();

    int failures = 0;
    failures += expect_eq("prepare_verify window_base", from_device_i32(d_window_base, 1), {11});
    failures += expect_eq("prepare_verify ids", from_device_i32(d_verify, k + 1),
                          {42, 101, 102, 103, 104, 105});
    failures += expect_eq("prepare_verify positions", from_device_i32(d_positions, k + 1),
                          {11, 12, 13, 14, 15, 16});
    failures += expect_eq("prepare_verify length unchanged", from_device_i32(d_length, 1), {11});
    return failures;
}

int accept_partial_case() {
    constexpr int k = 5;
    auto d_targets  = to_device_i32({7, 8, 9, 10, 11, 12});
    auto d_drafts   = to_device_i32({7, 8, 99, 10, 11});
    auto d_length   = to_device_i32({20});
    auto d_token    = to_device_i32({-1});
    DBuf d_sampled((k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64({5, 7, 3, 4, 100, 200, 300, 400, 500});

    DBuf d_logits = zero_logits(16, k + 1);
    DBuf d_cfg    = to_device_config(kernels::SamplingConfig{}); // greedy
    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {16, k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {target::kStepStatsCounters});
    kernels::mtp_accept_tokens(targets, logits, drafts, length, token, sampled, num, accepted,
                               ar_pos, stats, 16, config_ptr(d_cfg), nullptr);
    cudaDeviceSynchronize();

    int failures = 0;
    failures +=
        expect_eq("accept partial sampled", from_device_i32(d_sampled, k + 1), {7, 8, 9, 0, 0, 0});
    failures += expect_eq("accept partial num", from_device_i32(d_num, 1), {3});
    failures += expect_eq("accept partial accepted", from_device_i32(d_accepted, 1), {2});
    failures += expect_eq("accept partial length", from_device_i32(d_length, 1), {23});
    failures += expect_eq("accept partial token", from_device_i32(d_token, 1), {9});
    failures += expect_eq("accept partial ar_pos", from_device_i32(d_ar_pos, 1), {23});
    failures +=
        expect_eq("accept partial stats", from_device_i64(d_stats, target::kStepStatsCounters),
                  {10, 9, 4, 4, 101, 201, 300, 400, 500});
    return failures;
}

int accept_all_reject_case() {
    constexpr int k = 5;
    auto d_targets  = to_device_i32({7, 8, 9, 10, 11, 12});
    auto d_drafts   = to_device_i32({99, 8, 9, 10, 11});
    auto d_length   = to_device_i32({20});
    auto d_token    = to_device_i32({-1});
    DBuf d_sampled((k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64(std::vector<std::int64_t>(target::kStepStatsCounters, 0));

    DBuf d_logits = zero_logits(16, k + 1);
    DBuf d_cfg    = to_device_config(kernels::SamplingConfig{}); // greedy
    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {16, k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {target::kStepStatsCounters});
    kernels::mtp_accept_tokens(targets, logits, drafts, length, token, sampled, num, accepted,
                               ar_pos, stats, 16, config_ptr(d_cfg), nullptr);
    cudaDeviceSynchronize();

    int failures = 0;
    failures += expect_eq("accept all reject sampled", from_device_i32(d_sampled, k + 1),
                          {7, 0, 0, 0, 0, 0});
    failures += expect_eq("accept all reject num", from_device_i32(d_num, 1), {1});
    failures += expect_eq("accept all reject accepted", from_device_i32(d_accepted, 1), {0});
    failures += expect_eq("accept all reject length", from_device_i32(d_length, 1), {21});
    failures += expect_eq("accept all reject token", from_device_i32(d_token, 1), {7});
    failures += expect_eq("accept all reject ar_pos", from_device_i32(d_ar_pos, 1), {21});
    failures +=
        expect_eq("accept all reject stats", from_device_i64(d_stats, target::kStepStatsCounters),
                  {5, 0, 1, 0, 0, 0, 0, 0, 0});
    return failures;
}

int accept_all_case() {
    constexpr int k = 3;
    auto d_targets  = to_device_i32({1, 2, 3, 77});
    auto d_drafts   = to_device_i32({1, 2, 3});
    auto d_length   = to_device_i32({4});
    auto d_token    = to_device_i32({-1});
    DBuf d_sampled((k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64(std::vector<std::int64_t>(target::kStepStatsCounters, 0));

    DBuf d_logits = zero_logits(16, k + 1);
    DBuf d_cfg    = to_device_config(kernels::SamplingConfig{}); // greedy
    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {16, k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {target::kStepStatsCounters});
    kernels::mtp_accept_tokens(targets, logits, drafts, length, token, sampled, num, accepted,
                               ar_pos, stats, 16, config_ptr(d_cfg), nullptr);
    cudaDeviceSynchronize();

    int failures = 0;
    failures += expect_eq("accept all sampled", from_device_i32(d_sampled, k + 1), {1, 2, 3, 77});
    failures += expect_eq("accept all num", from_device_i32(d_num, 1), {4});
    failures += expect_eq("accept all accepted", from_device_i32(d_accepted, 1), {3});
    failures += expect_eq("accept all length", from_device_i32(d_length, 1), {8});
    failures += expect_eq("accept all token", from_device_i32(d_token, 1), {77});
    failures += expect_eq("accept all stats", from_device_i64(d_stats, target::kStepStatsCounters),
                          {3, 3, 1, 0, 1, 1, 1, 0, 0});
    return failures;
}

int shifted_case() {
    constexpr int k = 5;
    auto d_verify   = to_device_i32({40, 41, 42, 43, 44, 45});
    auto d_token    = to_device_i32({99});
    auto d_accepted = to_device_i32({2});
    auto d_shifted  = to_device_i32({-1, -1, -1, -1, -1, -1});

    Tensor verify(d_verify.p, DType::I32, {k + 1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor shifted(d_shifted.p, DType::I32, {k + 1});
    kernels::mtp_prepare_shifted_ids(verify, token, accepted, shifted, nullptr);
    cudaDeviceSynchronize();

    return expect_eq("shifted partial", from_device_i32(d_shifted, k + 1),
                     {41, 42, 99, 44, 45, -1});
}

int shifted_all_accept_case() {
    constexpr int k = 5;
    auto d_verify   = to_device_i32({40, 41, 42, 43, 44, 45});
    auto d_token    = to_device_i32({99});
    auto d_accepted = to_device_i32({5});
    auto d_shifted  = to_device_i32({-1, -1, -1, -1, -1, -1});

    Tensor verify(d_verify.p, DType::I32, {k + 1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor shifted(d_shifted.p, DType::I32, {k + 1});
    kernels::mtp_prepare_shifted_ids(verify, token, accepted, shifted, nullptr);
    cudaDeviceSynchronize();

    return expect_eq("shifted all accept", from_device_i32(d_shifted, k + 1),
                     {41, 42, 43, 44, 45, 99});
}

int gather_case() {
    constexpr int rows = target::TextConfig::hidden;
    constexpr int cols = 6;
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * cols);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            hidden[static_cast<std::size_t>(col) * rows + row] =
                static_cast<std::uint16_t>((col + 1) * 257 + (row % 251));
        }
    }
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        expected[row] = hidden[static_cast<std::size_t>(3) * rows + row];
    }
    auto d_hidden   = to_device_u16(hidden);
    auto d_accepted = to_device_i32({3});
    DBuf d_out(static_cast<std::size_t>(rows) * sizeof(std::uint16_t));

    Tensor hidden_tensor(d_hidden.p, DType::BF16, {rows, cols});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor out(d_out.p, DType::BF16, {rows, 1});
    kernels::mtp_gather_hidden_row(hidden_tensor, accepted, out, nullptr);
    cudaDeviceSynchronize();

    return expect_eq("gather hidden", from_device_u16(d_out, rows), expected);
}

int increment_case() {
    auto d_value = to_device_i32({23});
    Tensor value(d_value.p, DType::I32, {1});
    kernels::mtp_increment_i32(value, nullptr);
    cudaDeviceSynchronize();
    return expect_eq("increment scalar", from_device_i32(d_value, 1), {24});
}

int fallback_count_case() {
    auto d_stats = to_device_i64({5, 7, 3, 4, 100, 200, 300, 400, 500});
    Tensor stats(d_stats.p, DType::I64, {target::kStepStatsCounters});
    kernels::mtp_count_fallback_step(stats, nullptr);
    cudaDeviceSynchronize();
    return expect_eq("fallback count", from_device_i64(d_stats, target::kStepStatsCounters),
                     {5, 7, 3, 5, 100, 200, 300, 400, 500});
}

int gdn_initial_slot_case() {
    auto d_accepted = to_device_i32({0});
    auto d_slot     = to_device_i32({-1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor slot(d_slot.p, DType::I32, {1});

    int failures = 0;
    for (int value = 0; value <= 5; ++value) {
        cudaMemcpy(d_accepted.p, &value, sizeof(value), cudaMemcpyHostToDevice);
        kernels::mtp_set_gdn_initial_slot_from_accepted(accepted, slot, nullptr);
        cudaDeviceSynchronize();
        failures += expect_eq("gdn initial slot set", from_device_i32(d_slot, 1), {value});
    }

    kernels::mtp_reset_gdn_initial_slot(slot, nullptr);
    cudaDeviceSynchronize();
    failures += expect_eq("gdn initial slot reset", from_device_i32(d_slot, 1), {0});
    return failures;
}

// With a one-hot (greedy) draft, speculative rejection sampling must make the
// committed token at the first decision position distributed exactly as the
// target column-0 distribution p0 -- independent of the draft value and of
// whether that draft is accepted or rejected. Aggregate sampled_out[0] over many
// rounds (each a distinct position => distinct RNG) and compare to p0.
int reject_sampling_distribution_case() {
    constexpr int k     = 1;
    constexpr int vocab = 16;
    std::vector<float> base(vocab);
    fill_uniform(base, 4242u, -2.5f, 2.5f);
    round_to_bf16(base);

    std::vector<float> logits_h(static_cast<std::size_t>(vocab) * (k + 1), 0.0f);
    for (int v = 0; v < vocab; ++v) {
        logits_h[v]         = base[v]; // column 0 -> p0
        logits_h[vocab + v] = base[v]; // column 1 (bonus, not checked here)
    }
    DBuf d_logits = to_device_bf16(logits_h);

    double mmax = base[0];
    for (float x : base) { mmax = std::max(mmax, static_cast<double>(x)); }
    std::vector<double> p0(vocab);
    double sum = 0.0;
    for (int v = 0; v < vocab; ++v) {
        p0[v] = std::exp(static_cast<double>(base[v]) - mmax);
        sum += p0[v];
    }
    for (double& x : p0) { x /= sum; }

    const int d0 = 5; // a mid-probability draft exercises accept and reject
    kernels::SamplingConfig cfg;
    cfg.temperature = 1.0f;
    cfg.seed        = 99u;
    DBuf d_cfg      = to_device_config(cfg);

    const int N = 40000;
    std::vector<int> lengths(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) { lengths[static_cast<std::size_t>(i)] = i; }
    DBuf d_lengths = to_device_i32(lengths);
    DBuf d_scratch(sizeof(std::int32_t));
    DBuf d_collect(static_cast<std::size_t>(N) * sizeof(std::int32_t));

    auto d_targets = to_device_i32({0, 0});
    auto d_drafts  = to_device_i32({d0});
    auto d_token   = to_device_i32({-1});
    DBuf d_sampled((k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64(std::vector<std::int64_t>(target::kStepStatsCounters, 0));

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {vocab, k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_scratch.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {target::kStepStatsCounters});

    for (int r = 0; r < N; ++r) {
        cudaMemcpyAsync(d_scratch.p, static_cast<const std::int32_t*>(d_lengths.p) + r,
                        sizeof(std::int32_t), cudaMemcpyDeviceToDevice, nullptr);
        kernels::mtp_accept_tokens(targets, logits, drafts, length, token, sampled, num, accepted,
                                   ar_pos, stats, vocab, config_ptr(d_cfg), nullptr);
        cudaMemcpyAsync(static_cast<std::int32_t*>(d_collect.p) + r, d_sampled.p,
                        sizeof(std::int32_t), cudaMemcpyDeviceToDevice, nullptr);
    }
    cudaDeviceSynchronize();

    std::vector<int> results = from_device_i32(d_collect, static_cast<std::size_t>(N));
    std::vector<double> freq(vocab, 0.0);
    for (int tok : results) {
        if (tok < 0 || tok >= vocab) {
            std::cerr << "reject distribution: token out of range " << tok << '\n';
            return 1;
        }
        freq[tok] += 1.0;
    }
    for (double& f : freq) { f /= static_cast<double>(N); }

    double max_abs = 0.0;
    for (int v = 0; v < vocab; ++v) { max_abs = std::max(max_abs, std::abs(freq[v] - p0[v])); }
    if (max_abs > 0.015) {
        std::cerr << "reject distribution: committed/target gap " << max_abs << " too large\n";
        for (int v = 0; v < vocab; ++v) {
            std::cerr << "    v=" << v << " freq=" << freq[v] << " p0=" << p0[v] << '\n';
        }
        return 1;
    }
    std::cout << "    reject sampling distribution ok (max abs diff " << max_abs << ")\n";
    return 0;
}

int reject_sampling_reproducible_case() {
    constexpr int k     = 3;
    constexpr int vocab = 32;
    std::vector<float> logits_h(static_cast<std::size_t>(vocab) * (k + 1));
    fill_uniform(logits_h, 7u, -3.0f, 3.0f);
    round_to_bf16(logits_h);
    DBuf d_logits = to_device_bf16(logits_h);

    kernels::SamplingConfig cfg;
    cfg.temperature = 0.8f;
    cfg.top_p       = 0.95f;
    cfg.seed        = 2026u;
    DBuf d_cfg      = to_device_config(cfg);

    auto d_targets = to_device_i32({0, 0, 0, 0});
    auto d_drafts  = to_device_i32({3, 9, 17});
    auto d_token   = to_device_i32({-1});
    DBuf d_sampled((k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64(std::vector<std::int64_t>(target::kStepStatsCounters, 0));

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {vocab, k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {target::kStepStatsCounters});

    auto run_once = [&]() {
        auto d_length = to_device_i32({10});
        Tensor length(d_length.p, DType::I32, {1});
        kernels::mtp_accept_tokens(targets, logits, drafts, length, token, sampled, num, accepted,
                                   ar_pos, stats, vocab, config_ptr(d_cfg), nullptr);
        cudaDeviceSynchronize();
        std::vector<int> out = from_device_i32(d_sampled, k + 1);
        out.push_back(from_device_i32(d_num, 1)[0]);
        return out;
    };

    std::vector<int> a = run_once();
    std::vector<int> b = run_once();
    if (a != b) {
        std::cerr << "reject reproducible: identical seed/length produced different output\n";
        return 1;
    }
    std::cout << "    reject sampling reproducibility ok\n";
    return 0;
}

int reject_sampling_real_shape_distribution_case() {
    constexpr int k     = 1;
    constexpr int vocab = 248320;
    const int ids[]     = {17, 7919, 65537, 200003};
    const float vals[]  = {3.0f, 2.0f, 1.0f, 0.0f};
    std::vector<float> logits_h(static_cast<std::size_t>(vocab) * (k + 1), -20.0f);
    for (int col = 0; col <= k; ++col) {
        for (int i = 0; i < 4; ++i) {
            logits_h[static_cast<std::size_t>(col) * vocab + ids[i]] = vals[i];
        }
    }
    round_to_bf16(logits_h);
    DBuf d_logits = to_device_bf16(logits_h);

    std::vector<double> p0(4);
    double mmax = vals[0];
    double sum  = 0.0;
    for (int i = 0; i < 4; ++i) {
        p0[i] = std::exp(static_cast<double>(vals[i]) - mmax);
        sum += p0[i];
    }
    for (double& x : p0) { x /= sum; }

    kernels::SamplingConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k       = 4;
    cfg.seed        = 7001u;
    DBuf d_cfg      = to_device_config(cfg);

    const int N = 2048;
    std::vector<int> lengths(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) { lengths[static_cast<std::size_t>(i)] = i + 100; }
    DBuf d_lengths = to_device_i32(lengths);
    DBuf d_scratch(sizeof(std::int32_t));
    DBuf d_collect(static_cast<std::size_t>(N) * sizeof(std::int32_t));

    auto d_targets = to_device_i32({0, 0});
    auto d_drafts  = to_device_i32({ids[1]});
    auto d_token   = to_device_i32({-1});
    DBuf d_sampled((k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64(std::vector<std::int64_t>(target::kStepStatsCounters, 0));

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {vocab, k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_scratch.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {target::kStepStatsCounters});

    for (int r = 0; r < N; ++r) {
        cudaMemcpyAsync(d_scratch.p, static_cast<const std::int32_t*>(d_lengths.p) + r,
                        sizeof(std::int32_t), cudaMemcpyDeviceToDevice, nullptr);
        kernels::mtp_accept_tokens(targets, logits, drafts, length, token, sampled, num, accepted,
                                   ar_pos, stats, vocab, config_ptr(d_cfg), nullptr);
        cudaMemcpyAsync(static_cast<std::int32_t*>(d_collect.p) + r, d_sampled.p,
                        sizeof(std::int32_t), cudaMemcpyDeviceToDevice, nullptr);
    }
    cudaDeviceSynchronize();

    std::vector<int> results = from_device_i32(d_collect, static_cast<std::size_t>(N));
    std::vector<double> freq(4, 0.0);
    for (int tok : results) {
        int slot = -1;
        for (int i = 0; i < 4; ++i) {
            if (tok == ids[i]) {
                slot = i;
                break;
            }
        }
        if (slot < 0) {
            std::cerr << "real-shape reject distribution: token out of support " << tok << '\n';
            return 1;
        }
        freq[slot] += 1.0;
    }
    for (double& f : freq) { f /= static_cast<double>(N); }
    double max_abs = 0.0;
    for (int i = 0; i < 4; ++i) { max_abs = std::max(max_abs, std::abs(freq[i] - p0[i])); }
    if (max_abs > 0.05) {
        std::cerr << "real-shape reject distribution: committed/target gap " << max_abs
                  << " too large\n";
        for (int i = 0; i < 4; ++i) {
            std::cerr << "    token=" << ids[i] << " freq=" << freq[i] << " p0=" << p0[i] << '\n';
        }
        return 1;
    }
    std::cout << "    real-shape reject distribution ok (max abs diff " << max_abs << ")\n";
    return 0;
}

std::vector<int> run_real_shape_mtp_sequence(const std::vector<float>& logits_h,
                                             const std::vector<int>& drafts_h,
                                             kernels::SamplingConfig cfg, int rounds) {
    constexpr int vocab = 248320;
    const int k         = static_cast<int>(drafts_h.size());
    DBuf d_logits       = to_device_bf16(logits_h);
    DBuf d_counts(static_cast<std::size_t>(vocab) * sizeof(std::int32_t));
    cudaMemset(d_counts.p, 0, d_counts.bytes);
    cfg.token_counts = static_cast<std::int32_t*>(d_counts.p);
    DBuf d_cfg       = to_device_config(cfg);

    auto d_targets = to_device_i32(std::vector<int>(static_cast<std::size_t>(k + 1), 0));
    auto d_drafts  = to_device_i32(drafts_h);
    auto d_token   = to_device_i32({-1});
    DBuf d_length(sizeof(std::int32_t));
    DBuf d_sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64(std::vector<std::int64_t>(target::kStepStatsCounters, 0));
    DBuf d_collect(static_cast<std::size_t>(rounds) * (k + 2) * sizeof(std::int32_t));

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {vocab, k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {target::kStepStatsCounters});

    for (int r = 0; r < rounds; ++r) {
        const int len = 1000 + r;
        cudaMemcpy(d_length.p, &len, sizeof(len), cudaMemcpyHostToDevice);
        kernels::mtp_accept_tokens(targets, logits, drafts, length, token, sampled, num, accepted,
                                   ar_pos, stats, vocab, config_ptr(d_cfg), nullptr);
        std::int32_t* out = static_cast<std::int32_t*>(d_collect.p) + r * (k + 2);
        cudaMemcpyAsync(out, d_sampled.p, static_cast<std::size_t>(k + 1) * sizeof(std::int32_t),
                        cudaMemcpyDeviceToDevice, nullptr);
        cudaMemcpyAsync(out + k + 1, d_num.p, sizeof(std::int32_t), cudaMemcpyDeviceToDevice,
                        nullptr);
    }
    cudaDeviceSynchronize();
    return from_device_i32(d_collect, static_cast<std::size_t>(rounds) * (k + 2));
}

int reject_sampling_real_shape_reproducible_case() {
    constexpr int k     = 3;
    constexpr int vocab = 248320;
    std::vector<float> logits_h(static_cast<std::size_t>(vocab) * (k + 1), -18.0f);
    std::vector<int> drafts;
    for (int col = 0; col <= k; ++col) {
        for (int i = 0; i < 20; ++i) {
            const int id = (17 + i * 7919 + col * 101) % vocab;
            logits_h[static_cast<std::size_t>(col) * vocab + id] =
                4.0f - 0.08f * static_cast<float>(i);
            if (col == 0 && i < k) { drafts.push_back(id); }
        }
    }
    round_to_bf16(logits_h);

    kernels::SamplingConfig cfg;
    cfg.temperature      = 0.6f;
    cfg.top_k            = 20;
    cfg.top_p            = 0.95f;
    cfg.presence_penalty = 1.0f;
    cfg.seed             = 9001u;

    const int rounds   = 64;
    std::vector<int> a = run_real_shape_mtp_sequence(logits_h, drafts, cfg, rounds);
    std::vector<int> b = run_real_shape_mtp_sequence(logits_h, drafts, cfg, rounds);
    if (a != b) {
        std::cerr << "real-shape reject reproducible: identical seed/count path diverged\n";
        return 1;
    }
    kernels::SamplingConfig cfg2 = cfg;
    cfg2.seed                    = 9002u;
    std::vector<int> c           = run_real_shape_mtp_sequence(logits_h, drafts, cfg2, rounds);
    if (a == c) {
        std::cerr << "real-shape reject reproducible: different seed produced identical stream\n";
        return 1;
    }
    std::cout << "    real-shape reject reproducibility with counts ok\n";
    return 0;
}

int validation_case() {
    try {
        DBuf d(4);
        Tensor scalar(d.p, DType::BF16, {1});
        kernels::mtp_increment_i32(scalar, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "mtp_round validation: expected invalid_argument\n";
    return 1;
}

// Runs a single MTP accept round on dense [vocab, k+1] logits with a zeroed
// (empty round-start) penalty count buffer, so the only counts the penalty can
// see come from the round-local overlay. Returns {sampled_out[0..k], num,
// accepted}.
std::vector<int> run_one_mtp_round(const std::vector<float>& logits_h, int vocab, int k,
                                   const std::vector<int>& drafts_h, kernels::SamplingConfig cfg,
                                   int length, int token_domain = -1) {
    if (token_domain < 0) { token_domain = vocab; }
    DBuf d_logits = to_device_bf16(logits_h);
    DBuf d_counts(static_cast<std::size_t>(token_domain) * sizeof(std::int32_t));
    cudaMemset(d_counts.p, 0, d_counts.bytes);
    cfg.token_counts = static_cast<std::int32_t*>(d_counts.p);
    DBuf d_cfg       = to_device_config(cfg);

    auto d_targets = to_device_i32(std::vector<int>(static_cast<std::size_t>(k + 1), 0));
    auto d_drafts  = to_device_i32(drafts_h);
    auto d_token   = to_device_i32({-1});
    auto d_length  = to_device_i32({length});
    DBuf d_sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64(std::vector<std::int64_t>(target::kStepStatsCounters, 0));

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {vocab, k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length_t(d_length.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {target::kStepStatsCounters});

    kernels::mtp_accept_tokens(targets, logits, drafts, length_t, token, sampled, num, accepted,
                               ar_pos, stats, token_domain, config_ptr(d_cfg), nullptr);
    cudaDeviceSynchronize();

    std::vector<int> out = from_device_i32(d_sampled, k + 1);
    out.push_back(from_device_i32(d_num, 1)[0]);
    out.push_back(from_device_i32(d_accepted, 1)[0]);
    return out;
}

int physical_stride_and_token_domain_case() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int k             = 1;
    constexpr int first         = 17;
    constexpr int bonus         = 200003;
    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    for (int col = 0; col <= k; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * physical_rows;
        logits[base + (col == 0 ? first : bonus)] = 20.0f;
        logits[base + token_domain]                = 100.0f;
        logits[base + physical_rows - 1]           = 200.0f;
    }
    round_to_bf16(logits);

    kernels::SamplingConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k       = 1;
    cfg.seed        = 1234;

    int failures = 0;
    failures += expect_eq(
        "MTP physical stride + token domain bonus",
        run_one_mtp_round(logits, physical_rows, k, {first}, cfg, 10, token_domain),
        {first, bonus, 2, 1});
    failures += expect_eq(
        "MTP physical stride + token domain rejection",
        run_one_mtp_round(logits, physical_rows, k, {7919}, cfg, 10, token_domain),
        {first, 0, 1, 0});
    if (failures == 0) {
        std::cout << "    MTP physical stride + token domain rejection/bonus ok\n";
    }
    return failures;
}

// Intra-round overlay, frequency penalty, small vocab (single-block path).
//
// drafts = [A, A, A]. Logits are shaped so the committed prefix count of A must
// be visible to later columns:
//   col0: A dominates -> accept        (round-local count(A) becomes 1)
//   col1: A dominates after freq*1     -> accept (count(A) becomes 2)
//   col2: freq*2 pushes A out of top-k -> reject, correction = token 2
// With the overlay bug (columns scored from round-start counts only), A would
// dominate col2 as well, be accepted, and the round would emit a col3 bonus.
// Margins are large enough that the accept/reject/correction sequence is exact
// and seed/position independent, so it is a hard oracle rather than a histogram.
int overlay_frequency_repeat_case() {
    constexpr int vocab = 8;
    constexpr int k     = 3; // 4 verify columns
    constexpr int A     = 1;
    std::vector<float> logits(static_cast<std::size_t>(vocab) * (k + 1), -20.0f);
    auto set = [&](int col, int id, float v) {
        logits[static_cast<std::size_t>(col) * vocab + id] = v;
    };
    set(0, A, 100.0f);
    set(1, A, 100.0f);
    set(2, A, 100.0f); // A_raw=100; A_adj at count 2 == 100 - 2*30 == 40 < 42
    set(2, 2, 60.0f);  // dominates the residual -> deterministic correction
    set(2, 3, 42.0f);
    set(2, 4, 42.0f);
    set(2, 5, 42.0f);
    set(3, 6, 100.0f); // bonus marker: only the buggy all-accept path reaches it
    round_to_bf16(logits);

    kernels::SamplingConfig cfg;
    cfg.temperature               = 1.0f;
    cfg.top_k                     = 4;
    cfg.frequency_penalty         = 30.0f;
    const std::vector<int> drafts = {A, A, A};
    // sampled_out = [A, A, 2, 0], num = 3, accepted = 2.
    const std::vector<int> expected = {A, A, 2, 0, 3, 2};

    int failures = 0;
    for (std::uint64_t seed : {1ull, 7ull, 4242ull, 99991ull}) {
        for (int L : {0, 5, 1000, 65536}) {
            cfg.seed             = seed;
            std::vector<int> got = run_one_mtp_round(logits, vocab, k, drafts, cfg, L);
            failures += expect_eq("overlay frequency repeat (seed=" + std::to_string(seed) +
                                      " L=" + std::to_string(L) + ")",
                                  got, expected);
        }
    }
    if (failures == 0) { std::cout << "    overlay frequency-penalty (count reaches 2) ok\n"; }
    return failures;
}

// Intra-round overlay, presence penalty, small vocab (single-block path).
//
// drafts = [A, A]. Round-start counts are empty, so presence can only trigger on
// A at col1 if the accepted col0 A is visible via the overlay:
//   col0: A dominates -> accept (round-local count(A) becomes 1)
//   col1: presence drops A out of top-k -> reject, correction = token 2
// With the overlay bug, col1 sees count(A)==0, presence never fires, A is
// accepted, and the round emits a col2 bonus instead.
int overlay_presence_case() {
    constexpr int vocab = 8;
    constexpr int k     = 2; // 3 verify columns
    constexpr int A     = 1;
    std::vector<float> logits(static_cast<std::size_t>(vocab) * (k + 1), -20.0f);
    auto set = [&](int col, int id, float v) {
        logits[static_cast<std::size_t>(col) * vocab + id] = v;
    };
    set(0, A, 100.0f);
    set(1, A, 100.0f); // A_adj with presence == 100 - 70 == 30 < 42 -> excluded
    set(1, 2, 60.0f);
    set(1, 3, 42.0f);
    set(1, 4, 42.0f);
    set(1, 5, 42.0f);
    set(2, 6, 100.0f); // bonus marker for the buggy accept path
    round_to_bf16(logits);

    kernels::SamplingConfig cfg;
    cfg.temperature               = 1.0f;
    cfg.top_k                     = 4;
    cfg.presence_penalty          = 70.0f;
    const std::vector<int> drafts = {A, A};
    // sampled_out = [A, 2, 0], num = 2, accepted = 1.
    const std::vector<int> expected = {A, 2, 0, 2, 1};

    int failures = 0;
    for (std::uint64_t seed : {2ull, 13ull, 555ull, 271828ull}) {
        for (int L : {0, 9, 4096, 100000}) {
            cfg.seed             = seed;
            std::vector<int> got = run_one_mtp_round(logits, vocab, k, drafts, cfg, L);
            failures += expect_eq("overlay presence (seed=" + std::to_string(seed) +
                                      " L=" + std::to_string(L) + ")",
                                  got, expected);
        }
    }
    if (failures == 0) { std::cout << "    overlay presence-penalty ok\n"; }
    return failures;
}

// Same frequency-penalty overlay contract on the real-vocab multi-block path,
// where per-column distributions are built in parallel by the partial/group
// kernels. Exercises the overlay applied inside mtp_sampling_partial_topk_kernel.
int overlay_frequency_repeat_real_shape_case() {
    constexpr int vocab = 248320;
    constexpr int k     = 3;
    constexpr int A     = 17;
    constexpr int B     = 7919;
    std::vector<float> logits(static_cast<std::size_t>(vocab) * (k + 1), -20.0f);
    auto set = [&](int col, int id, float v) {
        logits[static_cast<std::size_t>(col) * vocab + id] = v;
    };
    set(0, A, 100.0f);
    set(1, A, 100.0f);
    set(2, A, 100.0f); // A_adj at count 2 == 40 < 42 -> out of top-k
    set(2, B, 60.0f);  // dominates the residual
    set(2, 65537, 42.0f);
    set(2, 131072, 42.0f);
    set(2, 200003, 42.0f);
    set(3, 100000, 100.0f);
    round_to_bf16(logits);

    kernels::SamplingConfig cfg;
    cfg.temperature                 = 1.0f;
    cfg.top_k                       = 4;
    cfg.frequency_penalty           = 30.0f;
    const std::vector<int> drafts   = {A, A, A};
    const std::vector<int> expected = {A, A, B, 0, 3, 2};

    int failures = 0;
    for (std::uint64_t seed : {3ull, 88ull, 123457ull}) {
        for (int L : {0, 1000, 250000}) {
            cfg.seed             = seed;
            std::vector<int> got = run_one_mtp_round(logits, vocab, k, drafts, cfg, L);
            failures += expect_eq("overlay frequency real-shape (seed=" + std::to_string(seed) +
                                      " L=" + std::to_string(L) + ")",
                                  got, expected);
        }
    }
    if (failures == 0) {
        std::cout << "    overlay frequency-penalty real-shape (multi-block) ok\n";
    }
    return failures;
}

} // namespace

int main() {
    static_assert(target::kStepStatsCounters == 9);
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    failures += prepare_verify_case();
    failures += accept_partial_case();
    failures += accept_all_reject_case();
    failures += accept_all_case();
    failures += reject_sampling_distribution_case();
    failures += reject_sampling_reproducible_case();
    failures += reject_sampling_real_shape_distribution_case();
    failures += reject_sampling_real_shape_reproducible_case();
    failures += overlay_frequency_repeat_case();
    failures += overlay_presence_case();
    failures += overlay_frequency_repeat_real_shape_case();
    failures += physical_stride_and_token_domain_case();
    failures += shifted_case();
    failures += shifted_all_accept_case();
    failures += gather_case();
    failures += increment_case();
    failures += fallback_count_case();
    failures += gdn_initial_slot_case();
    failures += validation_case();
    std::cout << (failures ? "FAIL" : "OK") << " mtp_round correctness\n";
    return failures ? 1 : 0;
}
