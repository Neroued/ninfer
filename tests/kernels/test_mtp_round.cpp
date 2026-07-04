#include "qus/kernels/mtp_round.h"
#include "qus/model/model.h"
#include "kernels/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qus;
using namespace qus::test;

namespace {

DBuf to_device_i64(const std::vector<std::int64_t>& h) {
    DBuf d(h.size() * sizeof(std::int64_t));
    cudaMemcpy(d.p, h.data(), h.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice);
    return d;
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
int expect_eq(const std::string& label, const std::vector<T>& got,
              const std::vector<T>& expected) {
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
    auto d_token = to_device_i32({42});
    auto d_drafts = to_device_i32({101, 102, 103, 104, 105});
    auto d_length = to_device_i32({11});
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
    auto d_targets = to_device_i32({7, 8, 9, 10, 11, 12});
    auto d_drafts = to_device_i32({7, 8, 99, 10, 11});
    auto d_length = to_device_i32({20});
    auto d_token = to_device_i32({-1});
    DBuf d_sampled((k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64({5, 7, 3, 4, 100, 200, 300, 400, 500});

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {model::kStepStatsCounters});
    kernels::mtp_accept_tokens(targets, drafts, length, token, sampled, num, accepted, ar_pos,
                               stats, nullptr);
    cudaDeviceSynchronize();

    int failures = 0;
    failures += expect_eq("accept partial sampled", from_device_i32(d_sampled, k + 1),
                          {7, 8, 9, 0, 0, 0});
    failures += expect_eq("accept partial num", from_device_i32(d_num, 1), {3});
    failures += expect_eq("accept partial accepted", from_device_i32(d_accepted, 1), {2});
    failures += expect_eq("accept partial length", from_device_i32(d_length, 1), {23});
    failures += expect_eq("accept partial token", from_device_i32(d_token, 1), {9});
    failures += expect_eq("accept partial ar_pos", from_device_i32(d_ar_pos, 1), {23});
    failures += expect_eq("accept partial stats", from_device_i64(d_stats, model::kStepStatsCounters),
                          {10, 9, 4, 4, 101, 201, 300, 400, 500});
    return failures;
}

int accept_all_reject_case() {
    constexpr int k = 5;
    auto d_targets = to_device_i32({7, 8, 9, 10, 11, 12});
    auto d_drafts = to_device_i32({99, 8, 9, 10, 11});
    auto d_length = to_device_i32({20});
    auto d_token = to_device_i32({-1});
    DBuf d_sampled((k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64(std::vector<std::int64_t>(model::kStepStatsCounters, 0));

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {model::kStepStatsCounters});
    kernels::mtp_accept_tokens(targets, drafts, length, token, sampled, num, accepted, ar_pos,
                               stats, nullptr);
    cudaDeviceSynchronize();

    int failures = 0;
    failures += expect_eq("accept all reject sampled", from_device_i32(d_sampled, k + 1),
                          {7, 0, 0, 0, 0, 0});
    failures += expect_eq("accept all reject num", from_device_i32(d_num, 1), {1});
    failures += expect_eq("accept all reject accepted", from_device_i32(d_accepted, 1), {0});
    failures += expect_eq("accept all reject length", from_device_i32(d_length, 1), {21});
    failures += expect_eq("accept all reject token", from_device_i32(d_token, 1), {7});
    failures += expect_eq("accept all reject ar_pos", from_device_i32(d_ar_pos, 1), {21});
    failures += expect_eq("accept all reject stats",
                          from_device_i64(d_stats, model::kStepStatsCounters),
                          {5, 0, 1, 0, 0, 0, 0, 0, 0});
    return failures;
}

int accept_all_case() {
    constexpr int k = 3;
    auto d_targets = to_device_i32({1, 2, 3, 77});
    auto d_drafts = to_device_i32({1, 2, 3});
    auto d_length = to_device_i32({4});
    auto d_token = to_device_i32({-1});
    DBuf d_sampled((k + 1) * sizeof(std::int32_t));
    DBuf d_num(sizeof(std::int32_t));
    DBuf d_accepted(sizeof(std::int32_t));
    DBuf d_ar_pos(sizeof(std::int32_t));
    auto d_stats = to_device_i64(std::vector<std::int64_t>(model::kStepStatsCounters, 0));

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor drafts(d_drafts.p, DType::I32, {k});
    Tensor length(d_length.p, DType::I32, {1});
    Tensor token(d_token.p, DType::I32, {1});
    Tensor sampled(d_sampled.p, DType::I32, {k + 1});
    Tensor num(d_num.p, DType::I32, {1});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor ar_pos(d_ar_pos.p, DType::I32, {1});
    Tensor stats(d_stats.p, DType::I64, {model::kStepStatsCounters});
    kernels::mtp_accept_tokens(targets, drafts, length, token, sampled, num, accepted, ar_pos,
                               stats, nullptr);
    cudaDeviceSynchronize();

    int failures = 0;
    failures += expect_eq("accept all sampled", from_device_i32(d_sampled, k + 1), {1, 2, 3, 77});
    failures += expect_eq("accept all num", from_device_i32(d_num, 1), {4});
    failures += expect_eq("accept all accepted", from_device_i32(d_accepted, 1), {3});
    failures += expect_eq("accept all length", from_device_i32(d_length, 1), {8});
    failures += expect_eq("accept all token", from_device_i32(d_token, 1), {77});
    failures += expect_eq("accept all stats", from_device_i64(d_stats, model::kStepStatsCounters),
                          {3, 3, 1, 0, 1, 1, 1, 0, 0});
    return failures;
}

int shifted_case() {
    constexpr int k = 5;
    auto d_verify = to_device_i32({40, 41, 42, 43, 44, 45});
    auto d_token = to_device_i32({99});
    auto d_accepted = to_device_i32({2});
    auto d_shifted = to_device_i32({-1, -1, -1, -1, -1, -1});

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
    auto d_verify = to_device_i32({40, 41, 42, 43, 44, 45});
    auto d_token = to_device_i32({99});
    auto d_accepted = to_device_i32({5});
    auto d_shifted = to_device_i32({-1, -1, -1, -1, -1, -1});

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
    constexpr int rows = model::kCfg.hidden;
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
    auto d_hidden = to_device_u16(hidden);
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
    Tensor stats(d_stats.p, DType::I64, {model::kStepStatsCounters});
    kernels::mtp_count_fallback_step(stats, nullptr);
    cudaDeviceSynchronize();
    return expect_eq("fallback count", from_device_i64(d_stats, model::kStepStatsCounters),
                     {5, 7, 3, 5, 100, 200, 300, 400, 500});
}

int gdn_initial_slot_case() {
    auto d_accepted = to_device_i32({0});
    auto d_slot = to_device_i32({-1});
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

int validation_case() {
    try {
        DBuf d(4);
        Tensor scalar(d.p, DType::BF16, {1});
        kernels::mtp_increment_i32(scalar, nullptr);
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << "mtp_round validation: expected invalid_argument\n";
    return 1;
}

} // namespace

int main() {
    static_assert(model::kStepStatsCounters == 9);
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    int failures = 0;
    failures += prepare_verify_case();
    failures += accept_partial_case();
    failures += accept_all_reject_case();
    failures += accept_all_case();
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
