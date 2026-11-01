#include "ops/suite.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ninfer/ops/speculative_accept.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include <string_view>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct AcceptExpected {
    std::vector<std::int32_t> sampled;
    std::int32_t num_sampled;
};

AcceptExpected accept_state_oracle(const std::vector<std::int32_t>& drafts, std::int32_t accepted,
                                   std::int32_t terminal_token) {
    const int k = static_cast<int>(drafts.size());
    AcceptExpected expected{
        .sampled     = std::vector<std::int32_t>(static_cast<std::size_t>(k + 1), 0),
        .num_sampled = accepted + 1,
    };
    for (int i = 0; i < accepted; ++i) {
        expected.sampled[static_cast<std::size_t>(i)] = drafts[static_cast<std::size_t>(i)];
    }
    expected.sampled[static_cast<std::size_t>(accepted)] = terminal_token;
    return expected;
}

int execute_accept_case(const std::string& label, const std::vector<std::int32_t>& target_tokens,
                        const std::vector<std::uint16_t>& logits_bits, int physical_rows,
                        const std::vector<std::int32_t>& drafts, std::int32_t initial_length,
                        int token_domain, ops::SamplingConfig config,
                        const std::vector<std::int32_t>& initial_token_counts,
                        const AcceptExpected& expected) {
    const int k              = static_cast<int>(drafts.size());
    DeviceBuffer d_targets   = to_device(target_tokens);
    DeviceBuffer d_logits    = to_device(logits_bits);
    DeviceBuffer d_drafts    = to_device(drafts);
    DeviceBuffer d_counts    = to_device(initial_token_counts);
    config.token_counts      = static_cast<std::int32_t*>(d_counts.data());
    DeviceBuffer d_config    = to_device(std::vector<ops::SamplingConfig>{config});
    const auto config_before = from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig));

    GuardedDeviceBuffer d_length(sizeof(std::int32_t));
    GuardedDeviceBuffer d_sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_num(sizeof(std::int32_t));
    DeviceBuffer d_extent               = to_device<std::int32_t>({k});
    const std::int32_t d_length_initial = initial_length;
    d_length.copy_from_host(&d_length_initial, sizeof(d_length_initial));
    d_sampled.fill(0x9d);
    const std::int32_t d_num_initial = -11;
    d_num.copy_from_host(&d_num_initial, sizeof(d_num_initial));

    Tensor targets(d_targets.span(), DType::I32, {k + 1});
    Tensor logits(d_logits.span(), DType::BF16, {physical_rows, k + 1});
    Tensor draft_tensor(d_drafts.span(), DType::I32, {k});
    Tensor extent(d_extent.span(), DType::I32, {1});
    Tensor length(d_length.span(), DType::I32, {1});
    Tensor sampled(d_sampled.span(), DType::I32, {k + 1});
    Tensor num_sampled(d_num.span(), DType::I32, {1});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, 1, 1);
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(256, workspace_bytes));
    MemoryArena workspace(workspace_storage.span());
    ops::speculative_accept_greedy_drafts(
        targets, logits, draft_tensor, extent, length, sampled, num_sampled, token_domain,
        static_cast<const ops::SamplingConfig*>(d_config.data()), workspace,
        ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures =
        verify_exact((label + " sampled").c_str(),
                     from_device<std::int32_t>(d_sampled.data(), k + 1), expected.sampled);
    failures += verify_exact((label + " num sampled").c_str(),
                             from_device<std::int32_t>(d_num.data(), 1), {expected.num_sampled});
    failures += verify_exact((label + " positions unchanged").c_str(),
                             from_device<std::int32_t>(d_length.data(), 1), {initial_length});

    failures +=
        verify_exact((label + " target tokens unchanged").c_str(),
                     from_device<std::int32_t>(d_targets, target_tokens.size()), target_tokens);
    failures += verify_exact((label + " logits unchanged").c_str(),
                             from_device<std::uint16_t>(d_logits, logits_bits.size()), logits_bits);
    failures += verify_exact((label + " drafts unchanged").c_str(),
                             from_device<std::int32_t>(d_drafts, drafts.size()), drafts);
    failures += verify_exact((label + " config unchanged").c_str(),
                             from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig)),
                             config_before);

    auto expected_counts = initial_token_counts;
    failures +=
        verify_exact((label + " token counts").c_str(),
                     from_device<std::int32_t>(d_counts, expected_counts.size()), expected_counts);

    failures += d_length.verify_guards((label + " length guards").c_str());
    failures += d_sampled.verify_guards((label + " sampled guards").c_str());
    failures += d_num.verify_guards((label + " num guards").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int greedy_accept_case(int k, int accepted_count, int token_domain = 64) {
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k));
    for (int i = 0; i <= k; ++i) {
        targets[static_cast<std::size_t>(i)] = 3 + 2 * i;
        if (i < k) drafts[static_cast<std::size_t>(i)] = targets[static_cast<std::size_t>(i)];
    }
    if (accepted_count < k) {
        drafts[static_cast<std::size_t>(accepted_count)] =
            targets[static_cast<std::size_t>(accepted_count)] + 1;
    }
    const std::int32_t initial_length = 200 + k;
    const auto expected               = accept_state_oracle(drafts, accepted_count,
                                                            targets[static_cast<std::size_t>(accepted_count)]);
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(token_domain) * (k + 1));
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<std::uint16_t>(0x3f00u + (i % 127u));
    }
    std::vector<std::int32_t> token_counts(token_domain);
    for (int i = 0; i < token_domain; ++i) token_counts[static_cast<std::size_t>(i)] = i % 5;
    return execute_accept_case("speculative greedy K=" + std::to_string(k) +
                                   " A=" + std::to_string(accepted_count),
                               targets, logits, token_domain, drafts, initial_length, token_domain,
                               ops::SamplingConfig{}, token_counts, expected);
}

int deterministic_sampling_case() {
    constexpr int physical_rows = 248320;
    constexpr int token_domain  = 248077;
    constexpr int k             = 5;
    constexpr int accepted      = 2;
    const std::vector<std::int32_t> drafts{17, 7919, 65537, 131071, 200003};
    std::vector<std::int32_t> targets(static_cast<std::size_t>(k + 1));
    for (int i = 0; i <= k; ++i) targets[static_cast<std::size_t>(i)] = 101 + i;
    constexpr std::int32_t correction = 150001;

    std::vector<float> logits(static_cast<std::size_t>(physical_rows) * (k + 1), -20.0f);
    for (int col = 0; col <= k; ++col) {
        const std::size_t base = static_cast<std::size_t>(col) * physical_rows;
        const int winner =
            col < accepted ? drafts[static_cast<std::size_t>(col)] : correction + col - accepted;
        logits[base + static_cast<std::size_t>(winner)] = 20.0f;
        logits[base + token_domain]                     = 100.0f;
        logits[base + physical_rows - 1]                = 200.0f;
    }
    round_to_bf16(logits);
    std::vector<std::uint16_t> logits_bits(logits.size());
    for (std::size_t i = 0; i < logits.size(); ++i) logits_bits[i] = f32_to_bf16(logits[i]);

    const std::int32_t initial_length = 4093;
    const auto expected               = accept_state_oracle(drafts, accepted, correction);
    std::vector<std::int32_t> token_counts(token_domain, 0);
    token_counts[static_cast<std::size_t>(drafts[0])]  = 3;
    token_counts[static_cast<std::size_t>(drafts[1])]  = 5;
    token_counts[static_cast<std::size_t>(correction)] = 7;

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    config.top_p       = 0.9f;
    config.min_p       = 0.5f;
    config.seed        = 0x123456789abcdef0ull;
    return execute_accept_case("speculative sampling deterministic support", targets, logits_bits,
                               physical_rows, drafts, initial_length, token_domain, config,
                               token_counts, expected);
}

int grammar_onehot_case(int domain, float temperature) {
    constexpr int k = 3;
    const std::vector<int> drafts{11, 12, 13};
    const std::vector<int> targets(k + 1, 0);
    std::vector<std::uint16_t> logits((domain + 2) * (k + 1), f32_to_bf16(12.0f));
    const int words = (domain + 31) / 32;
    std::vector<std::uint32_t> masks(words * (k + 1), 0);
    for (int col = 0; col <= k; ++col) {
        const int allowed = col < 2 ? drafts[col] : 51;
        masks[col * words + allowed / 32] |= 1u << (allowed % 32);
        logits[col * (domain + 2) + allowed] = f32_to_bf16(-2.0f);
    }
    auto device_masks = to_device(masks);
    ops::SamplingConfig config;
    config.temperature  = temperature;
    config.mask         = {static_cast<const std::uint32_t*>(device_masks.data()), words};
    const auto expected = accept_state_oracle(drafts, 2, 51);
    return execute_accept_case("one-hot grammar mask", targets, logits, domain + 2, drafts, 73,
                               domain, config, std::vector<int>(domain, 0), expected);
}

int greedy_penalty_case(int token_domain) {
    constexpr int k = 2;
    const std::vector<std::int32_t> drafts{1, 1};
    const std::vector<std::int32_t> raw_targets{1, 1, 3};
    std::vector<float> logits(static_cast<std::size_t>(token_domain) * (k + 1), -20.0F);
    logits[1]                    = 5.0F;
    logits[token_domain + 1]     = 5.0F;
    logits[token_domain + 4]     = 4.5F;
    logits[2 * token_domain + 3] = 5.0F;
    std::vector<std::uint16_t> bits(logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index) {
        bits[index] = f32_to_bf16(logits[index]);
    }

    ops::SamplingConfig config{};
    config.temperature      = 0.0F;
    config.presence_penalty = 1.0F;
    const std::vector<std::int32_t> counts(static_cast<std::size_t>(token_domain), 0);
    const auto expected = accept_state_oracle(drafts, 1, 4);
    return execute_accept_case(
        "speculative greedy penalty token-domain=" + std::to_string(token_domain), raw_targets,
        bits, token_domain, drafts, 100, token_domain, config, counts, expected);
}

int batched_sampling_workspace_stride_case() {
    constexpr int physical_rows = 257;
    constexpr int token_domain  = 257;
    constexpr int k             = 3;
    constexpr int batch         = 2;
    constexpr int columns       = k + 1;

    const std::vector<std::int32_t> drafts{10, 11, 12, 30, 31, 32};
    const std::vector<std::int32_t> winners{10, 20, 21, 22, 30, 31, 32, 33};
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns * batch,
                                      f32_to_bf16(-20.0f));
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < columns; ++col) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * columns + col) * physical_rows;
            logits[base + static_cast<std::size_t>(winners[row * columns + col])] =
                f32_to_bf16(20.0f);
        }
    }

    DeviceBuffer d_targets = to_device(winners);
    DeviceBuffer d_logits  = to_device(logits);
    std::vector<int> steps(drafts.size());
    for (int row = 0; row < batch; ++row)
        for (int col = 0; col < k; ++col) steps[col * batch + row] = drafts[row * k + col];
    DeviceBuffer d_drafts  = to_device(steps);
    DeviceBuffer d_extents = to_device<std::int32_t>({k, k});
    DeviceBuffer d_lengths = to_device<std::int32_t>({100, 200});
    DeviceBuffer d_licensed(ninfer::test::cuda_environment().context.device(),
                            static_cast<std::size_t>(columns) * batch * sizeof(std::int32_t));
    DeviceBuffer d_counts(ninfer::test::cuda_environment().context.device(),
                          static_cast<std::size_t>(batch) * sizeof(std::int32_t));

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    const std::vector<ops::SamplingConfig> configs{config, config};
    DeviceBuffer d_configs = to_device(configs);

    Tensor targets(d_targets.span(), DType::I32, {columns, batch});
    Tensor logits_tensor(d_logits.span(), DType::BF16, {physical_rows, columns, batch});
    auto draft_tensor = Tensor(d_drafts.span(), DType::I32, {batch, k}).permute({1, 0, 2, 3});
    Tensor extents(d_extents.span(), DType::I32, {batch});
    Tensor lengths(d_lengths.span(), DType::I32, {batch});
    Tensor licensed(d_licensed.span(), DType::I32, {columns, batch});
    Tensor counts(d_counts.span(), DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, batch,
                                                                       batch);
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           workspace_bytes);
    MemoryArena workspace(workspace_storage.span());
    ops::speculative_accept_greedy_drafts(
        targets, logits_tensor, draft_tensor, extents, lengths, licensed, counts, token_domain,
        static_cast<const ops::SamplingConfig*>(d_configs.data()), workspace,
        ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_exact("speculative sampling B=2 licensed",
                                from_device<std::int32_t>(d_licensed, columns * batch),
                                {10, 20, 0, 0, 30, 31, 32, 33});
    failures += verify_exact("speculative sampling B=2 counts",
                             from_device<std::int32_t>(d_counts, batch), {2, 4});
    failures += verify_exact("speculative sampling B=2 lengths",
                             from_device<std::int32_t>(d_lengths, batch), {100, 200});
    return failures;
}

int capacity_contract() {
    int failures = 0;
    const std::size_t k15 =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 1);
    if (k15 == 0 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 16, 16, 1, 1) != 0 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 1, 16, 1, 1) != k15 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 2) !=
            2 * k15) {
        std::cerr << "speculative accept workspace did not cover its draft interval\n";
        ++failures;
    }
    try {
        (void)ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 0, 15, 1, 1);
        std::cerr << "speculative accept workspace accepted an invalid draft interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}
} // namespace

void add_greedy_accept_cases(ninfer::test::OpSuite& suite) {
    suite.add("accept_greedy/capacity", capacity_contract);
    for (auto [k, accepted, domain] : {std::array{1, 0, 64}, {5, 2, 64}, {5, 5, 64}, {15, 7, 257}})
        suite.add("accept_greedy/k" + std::to_string(k) + "/accepted" + std::to_string(accepted),
                  [=] { return greedy_accept_case(k, accepted, domain); });
    for (int domain : {64, 257, 248077})
        for (float temperature : {0.f, 1.f})
            suite.add("accept_greedy/grammar/v" + std::to_string(domain) + "/temp" +
                          std::to_string(temperature),
                      [=] { return grammar_onehot_case(domain, temperature); });
    for (int domain : {64, 257})
        suite.add("accept_greedy/penalties/v" + std::to_string(domain),
                  [=] { return greedy_penalty_case(domain); });
    suite.add("accept_greedy/deterministic_draw", deterministic_sampling_case);
    suite.add("accept_greedy/batched_workspace", batched_sampling_workspace_stride_case);
}
