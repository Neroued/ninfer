#include "ops/suite.h"
#include "ninfer/ops/indexing.h"
#include "cuda_fixture.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::uint16_t> bit_pattern(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> values(count);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state     = state * 1664525u + 1013904223u;
        values[i] = static_cast<std::uint16_t>((state >> 16) ^ static_cast<std::uint32_t>(i));
    }
    return values;
}

int pack_case(std::int32_t hidden, std::int32_t tokens) {
    const std::int32_t output_rows = 2 * hidden;
    const std::size_t input_count  = static_cast<std::size_t>(hidden) * tokens;
    const auto embedding           = bit_pattern(input_count, 0x1234'5678u);
    const auto hidden_values       = bit_pattern(input_count, 0x8765'4321u);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(output_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < hidden; ++row) {
            expected[static_cast<std::size_t>(token) * output_rows + row] =
                embedding[static_cast<std::size_t>(token) * hidden + row];
            expected[static_cast<std::size_t>(token) * output_rows + hidden + row] =
                hidden_values[static_cast<std::size_t>(token) * hidden + row];
        }
    }

    GuardedDeviceBuffer device_embedding(embedding.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_hidden(hidden_values.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(expected.size() * sizeof(std::uint16_t));
    device_embedding.copy_from_host(embedding.data(), embedding.size() * sizeof(std::uint16_t));
    device_hidden.copy_from_host(hidden_values.data(),
                                 hidden_values.size() * sizeof(std::uint16_t));
    device_output.fill(0xcd);

    Tensor embedding_tensor(device_embedding.span(), DType::BF16, {hidden, tokens});
    Tensor hidden_tensor(device_hidden.span(), DType::BF16, {hidden, tokens});
    Tensor output_tensor(device_output.span(), DType::BF16, {output_rows, tokens});
    ops::concat_bf16_features(embedding_tensor, hidden_tensor, output_tensor,
                              ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label =
        "concat_bf16_features D=" + std::to_string(hidden) + " T=" + std::to_string(tokens);
    int failures = 0;
    failures += verify_exact(
        label.c_str(), from_device<std::uint16_t>(device_output.data(), expected.size()), expected);
    failures += verify_exact((label + " preserves embedding").c_str(),
                             from_device<std::uint16_t>(device_embedding.data(), embedding.size()),
                             embedding);
    failures += verify_exact((label + " preserves hidden").c_str(),
                             from_device<std::uint16_t>(device_hidden.data(), hidden_values.size()),
                             hidden_values);
    failures += device_embedding.verify_guards((label + " embedding").c_str());
    failures += device_hidden.verify_guards((label + " hidden").c_str());
    failures += device_output.verify_guards((label + " output").c_str());
    return failures;
}

int split_case(std::int32_t tokens) {
    constexpr std::int32_t kInputRows = 14336;
    constexpr std::int32_t kQueryRows = 6144;
    constexpr std::int32_t kKvRows    = 1024;
    const auto input = bit_pattern(static_cast<std::size_t>(kInputRows) * tokens, 0x2468'ace0u);
    std::vector<std::uint16_t> expected_query(static_cast<std::size_t>(kQueryRows) * tokens);
    std::vector<std::uint16_t> expected_key(static_cast<std::size_t>(kKvRows) * tokens);
    std::vector<std::uint16_t> expected_gate(static_cast<std::size_t>(kQueryRows) * tokens);
    std::vector<std::uint16_t> expected_value(static_cast<std::size_t>(kKvRows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t input_base = static_cast<std::size_t>(token) * kInputRows;
        for (std::int32_t row = 0; row < kQueryRows; ++row) {
            expected_query[static_cast<std::size_t>(token) * kQueryRows + row] =
                input[input_base + row];
            expected_gate[static_cast<std::size_t>(token) * kQueryRows + row] =
                input[input_base + kQueryRows + kKvRows + row];
        }
        for (std::int32_t row = 0; row < kKvRows; ++row) {
            expected_key[static_cast<std::size_t>(token) * kKvRows + row] =
                input[input_base + kQueryRows + row];
            expected_value[static_cast<std::size_t>(token) * kKvRows + row] =
                input[input_base + kQueryRows + kKvRows + kQueryRows + row];
        }
    }

    GuardedDeviceBuffer device_input(input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_query(expected_query.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_key(expected_key.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_gate(expected_gate.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_value(expected_value.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
    device_query.fill(0xcd);
    device_key.fill(0xcd);
    device_gate.fill(0xcd);
    device_value.fill(0xcd);

    Tensor input_tensor(device_input.span(), DType::BF16, {kInputRows, tokens});
    Tensor query_tensor(device_query.span(), DType::BF16, {256, 24, tokens});
    Tensor key_tensor(device_key.span(), DType::BF16, {256, 4, tokens});
    Tensor gate_tensor(device_gate.span(), DType::BF16, {256, 24, tokens});
    Tensor value_tensor(device_value.span(), DType::BF16, {256, 4, tokens});
    ops::split_bf16_features(input_tensor, query_tensor, key_tensor, gate_tensor, value_tensor,
                             ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label = "split_bf16_features T=" + std::to_string(tokens);
    int failures            = 0;
    failures += verify_exact((label + " query").c_str(),
                             from_device<std::uint16_t>(device_query.data(), expected_query.size()),
                             expected_query);
    failures += verify_exact((label + " key").c_str(),
                             from_device<std::uint16_t>(device_key.data(), expected_key.size()),
                             expected_key);
    failures += verify_exact((label + " gate").c_str(),
                             from_device<std::uint16_t>(device_gate.data(), expected_gate.size()),
                             expected_gate);
    failures += verify_exact((label + " value").c_str(),
                             from_device<std::uint16_t>(device_value.data(), expected_value.size()),
                             expected_value);
    failures += verify_exact((label + " preserves input").c_str(),
                             from_device<std::uint16_t>(device_input.data(), input.size()), input);
    failures += device_input.verify_guards((label + " input").c_str());
    failures += device_query.verify_guards((label + " query").c_str());
    failures += device_key.verify_guards((label + " key").c_str());
    failures += device_gate.verify_guards((label + " gate").c_str());
    failures += device_value.verify_guards((label + " value").c_str());
    return failures;
}

} // namespace

void add_features_cases(ninfer::test::OpSuite& suite) {
    for (int hidden : {5120, 2048})
        for (int tokens : {1, 6, 48})
            suite.add("concat_features/" + std::to_string(hidden) + "/t" + std::to_string(tokens),
                      [=] { return pack_case(hidden, tokens); });
    for (int tokens : {1, 6, 48})
        suite.add("split_features/t" + std::to_string(tokens), [=] { return split_case(tokens); });
}
