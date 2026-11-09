#include "ninfer/ops/speculative_round.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include <string_view>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

using namespace ninfer;
using namespace ninfer::test;

namespace {

template <typename T>
void initialize(GuardedDeviceBuffer& buffer, const std::vector<T>& values) {
    buffer.copy_from_host(values.data(), values.size() * sizeof(T));
}

template <typename T>
std::vector<T> read(const GuardedDeviceBuffer& buffer, std::size_t count) {
    return from_device<T>(buffer.data(), count);
}

DeviceBuffer device_config(const ops::SamplingConfig& config) {
    return to_device(std::vector<ops::SamplingConfig>{config});
}

int prepare_verify_case(int k, int batch) {
    const int width = k + 1;
    std::vector<std::int32_t> anchors(batch), lengths(batch), extents(batch), drafts(k * batch);
    for (int b = 0; b < batch; ++b) {
        anchors[b] = 70000 + 11 * b;
        lengths[b] = 131072 + 1009 * b;
        for (int j = 0; j < k; ++j) drafts[b * k + j] = 37 + 257 * b + 7919 * j;
    }
    DeviceBuffer d_anchors = to_device(anchors), d_lengths = to_device(lengths),
                 d_drafts  = to_device(drafts);
    DeviceBuffer d_extents = to_device(extents);
    GuardedDeviceBuffer d_full(width * batch * sizeof(std::int32_t));
    GuardedDeviceBuffer d_ids(d_full.bytes()), d_positions(d_full.bytes());
    Tensor a(d_anchors.p, DType::I32, {batch}), l(d_lengths.p, DType::I32, {batch});
    Tensor d(d_drafts.p, DType::I32, {k, batch}), e(d_extents.p, DType::I32, {batch});
    Tensor full(d_full.data(), DType::I32, {width, batch}),
        ids(d_ids.data(), DType::I32, {width, batch});
    Tensor positions(d_positions.data(), DType::I32, {width, batch});
    DeviceContext context;
    cuda_synchronize();
    const auto launch = [&] {
        ops::speculative_prepare_verify_inputs(a, d, l, e, full, positions, context.stream);
        ops::speculative_prepare_verify_ids(a, d, e, ids, context.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (k == 15 && batch == 8) {
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase <= k; ++phase) {
        for (int b = 0; b < batch; ++b) extents[b] = (phase + 3 * b) % width;
        CUDA_CHECK(cudaMemcpyAsync(d_extents.p, extents.data(), d_extents.bytes,
                                   cudaMemcpyHostToDevice, context.stream));
        if (graph.ready())
            graph.launch(context.stream);
        else
            launch();
        context.synchronize();
        std::vector<std::int32_t> expected_ids(width * batch), expected_positions(width * batch);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < width; ++j) {
                expected_ids[b * width + j] =
                    j > 0 && j <= extents[b] ? drafts[b * k + j - 1] : anchors[b];
                expected_positions[b * width + j] = lengths[b] + std::min(j, extents[b]);
            }
        failures += verify_exact("verify ids", read<std::int32_t>(d_ids, expected_ids.size()),
                                 expected_ids);
        failures += verify_exact("verify full ids", read<std::int32_t>(d_full, expected_ids.size()),
                                 expected_ids);
        failures += verify_exact("verify positions",
                                 read<std::int32_t>(d_positions, expected_positions.size()),
                                 expected_positions);
        failures +=
            verify_exact("extents unchanged", from_device<std::int32_t>(d_extents, batch), extents);
    }
    failures +=
        verify_exact("anchors unchanged", from_device<std::int32_t>(d_anchors, batch), anchors);
    failures +=
        verify_exact("lengths unchanged", from_device<std::int32_t>(d_lengths, batch), lengths);
    failures += verify_exact("drafts unchanged", from_device<std::int32_t>(d_drafts, drafts.size()),
                             drafts);
    failures += d_full.verify_guards("verify full ids");
    failures += d_ids.verify_guards("verify ids");
    failures += d_positions.verify_guards("verify positions");
    return failures;
}

struct AcceptExpected {
    std::vector<std::int32_t> sampled;
    std::int32_t num_sampled;
    std::int32_t accepted;
    std::int32_t length;
    std::int32_t token;
};

constexpr int kSparsePhysicalRows = 248320;
constexpr int kSparseTokenDomain  = 248077;
constexpr int kSparseCandidates   = 16;
constexpr int kTargetCandidateCap = 20;

struct SparseAcceptSuite {
    const int kSparseDrafts;
    const int kSparseColumns;
    const int kSparseBatch;
    std::size_t observed_workspace = 0;
    // Optional token masks, [K+1] positions of mask_words per row (upstream 08902f73).
    std::vector<std::uint32_t> masks;
    static constexpr int mask_words = (kSparseTokenDomain + 31) / 32;

    SparseAcceptSuite(int drafts, int batch)
        : kSparseDrafts(drafts), kSparseColumns(drafts + 1), kSparseBatch(batch) {}

    std::size_t sparse_logit_index(int row, int column, int token) {
        return (static_cast<std::size_t>(row) * kSparseColumns + column) * kSparsePhysicalRows +
               token;
    }

    std::size_t sparse_candidate_index(int row, int draft, int candidate) {
        return (static_cast<std::size_t>(row) * kSparseDrafts + draft) * kSparseCandidates +
               candidate;
    }

    std::uint64_t oracle_splitmix64(std::uint64_t value) {
        value += 0x9E3779B97F4A7C15ull;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
        return value ^ (value >> 31);
    }

    float oracle_uniform(std::uint64_t seed, int position, int purpose) {
        std::uint64_t key = seed;
        key               = oracle_splitmix64(key ^
                                              (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                                 0xD1B54A32D192ED03ull));
        key               = oracle_splitmix64(
            key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(purpose)) << 21));
        const std::uint32_t bits = static_cast<std::uint32_t>(key >> 40);
        return static_cast<float>(bits) * (1.0f / 16777216.0f);
    }

    std::uint64_t find_seed_for_uniform(int position, int purpose, float lower, float upper) {
        for (std::uint64_t seed = 1; seed < 1000000; ++seed) {
            const float value = oracle_uniform(seed, position, purpose);
            if (value > lower && value < upper) { return seed; }
        }
        throw std::runtime_error("failed to find deterministic sparse-accept seed");
    }

    struct TargetDistribution {
        std::vector<std::int32_t> ids;
        std::vector<double> probabilities;
    };

    TargetDistribution sparse_target_distribution(const std::vector<std::uint16_t>& logits, int row,
                                                  int column, const ops::SamplingConfig& config,
                                                  const std::vector<std::int32_t>& token_counts,
                                                  const std::vector<std::int32_t>& drafts) {
        const auto adjusted = [&](int token) {
            if (!masks.empty() &&
                !(masks[(static_cast<std::size_t>(row) * kSparseColumns + column) * mask_words +
                        token / 32] &
                  (1u << (token % 32))))
                return -std::numeric_limits<double>::infinity();
            double value = bf16_to_f32(logits[sparse_logit_index(row, column, token)]);
            int count    = token_counts[static_cast<std::size_t>(row) * kSparseTokenDomain + token];
            for (int previous = 0; previous < column; ++previous) {
                if (drafts[static_cast<std::size_t>(row) * kSparseDrafts + previous] == token) {
                    ++count;
                }
            }
            if (count > 0) value -= config.presence_penalty;
            value -= config.frequency_penalty * static_cast<double>(count);
            return value;
        };
        const auto better = [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
        };

        if (!(config.temperature > 0.0f)) {
            std::pair<double, std::int32_t> best{-std::numeric_limits<double>::infinity(), 0};
            for (int token = 0; token < kSparseTokenDomain; ++token) {
                const std::pair<double, std::int32_t> candidate{adjusted(token), token};
                if (better(candidate, best)) best = candidate;
            }
            return {{best.second}, {1.0}};
        }

        int cap = kTargetCandidateCap;
        if (config.top_k > 0 && config.top_k < cap) cap = config.top_k;
        cap = std::min(cap, kSparseTokenDomain);
        std::vector<std::pair<double, std::int32_t>> ranked(
            static_cast<std::size_t>(kSparseTokenDomain));
        for (int token = 0; token < kSparseTokenDomain; ++token) {
            ranked[static_cast<std::size_t>(token)] = {adjusted(token), token};
        }
        std::partial_sort(ranked.begin(), ranked.begin() + cap, ranked.end(), better);
        ranked.resize(static_cast<std::size_t>(cap));

        std::vector<double> weights(static_cast<std::size_t>(cap));
        const double inverse_temperature = 1.0 / static_cast<double>(config.temperature);
        const double maximum             = ranked.front().first * inverse_temperature;
        double total                     = 0.0;
        for (int item = 0; item < cap; ++item) {
            weights[static_cast<std::size_t>(item)] = std::exp(
                ranked[static_cast<std::size_t>(item)].first * inverse_temperature - maximum);
            total += weights[static_cast<std::size_t>(item)];
        }

        const double min_p_threshold = config.min_p > 0.0f ? config.min_p * weights.front() : -1.0;
        const bool top_p_active      = config.top_p < 1.0f;
        const double top_p_target    = config.top_p * total;
        double cumulative            = 0.0;
        int support                  = 0;
        for (int item = 0; item < cap; ++item) {
            if (min_p_threshold >= 0.0 &&
                weights[static_cast<std::size_t>(item)] < min_p_threshold) {
                break;
            }
            cumulative += weights[static_cast<std::size_t>(item)];
            support = item + 1;
            if (top_p_active && cumulative >= top_p_target) break;
        }
        support            = std::max(1, support);
        double support_sum = 0.0;
        for (int item = 0; item < support; ++item) {
            support_sum += weights[static_cast<std::size_t>(item)];
        }

        TargetDistribution distribution;
        distribution.ids.resize(static_cast<std::size_t>(support));
        distribution.probabilities.resize(static_cast<std::size_t>(support));
        for (int item = 0; item < support; ++item) {
            distribution.ids[static_cast<std::size_t>(item)] =
                ranked[static_cast<std::size_t>(item)].second;
            distribution.probabilities[static_cast<std::size_t>(item)] =
                weights[static_cast<std::size_t>(item)] / support_sum;
        }
        return distribution;
    }

    double distribution_probability(const TargetDistribution& distribution, int token) {
        for (std::size_t item = 0; item < distribution.ids.size(); ++item) {
            if (distribution.ids[item] == token) return distribution.probabilities[item];
        }
        return 0.0;
    }

    double sparse_proposal_probability(const std::vector<std::int32_t>& candidate_ids,
                                       const std::vector<float>& proposal_q, int row, int draft,
                                       int token) {
        for (int candidate = 0; candidate < kSparseCandidates; ++candidate) {
            const std::size_t index = sparse_candidate_index(row, draft, candidate);
            if (candidate_ids[index] == token) return proposal_q[index];
        }
        return 0.0;
    }

    int sample_target_distribution(const TargetDistribution& distribution, double uniform) {
        double cumulative = 0.0;
        for (std::size_t item = 0; item < distribution.ids.size(); ++item) {
            cumulative += distribution.probabilities[item];
            if (uniform < cumulative) return distribution.ids[item];
        }
        return distribution.ids.back();
    }

    int sample_sparse_residual(const TargetDistribution& target,
                               const std::vector<std::int32_t>& candidate_ids,
                               const std::vector<float>& proposal_q, int row, int draft,
                               double uniform) {
        double mass = 0.0;
        for (std::size_t item = 0; item < target.ids.size(); ++item) {
            const double q = sparse_proposal_probability(candidate_ids, proposal_q, row, draft,
                                                         target.ids[item]);
            mass += std::max(target.probabilities[item] - q, 0.0);
        }
        if (!(mass > 0.0)) return target.ids.front();
        const double goal = uniform * mass;
        double cumulative = 0.0;
        int selected      = target.ids.back();
        for (std::size_t item = 0; item < target.ids.size(); ++item) {
            const double q = sparse_proposal_probability(candidate_ids, proposal_q, row, draft,
                                                         target.ids[item]);
            const double residual = std::max(target.probabilities[item] - q, 0.0);
            if (!(residual > 0.0)) continue;
            cumulative += residual;
            selected = target.ids[item];
            if (goal < cumulative) return selected;
        }
        return selected;
    }

    struct SparseExpected {
        std::vector<std::int32_t> licensed_tokens;
        std::vector<std::int32_t> licensed_counts;
        std::vector<std::int32_t> accepted;
        std::vector<std::int32_t> lengths;
        std::vector<std::int32_t> anchors;
    };

    SparseExpected sparse_accept_oracle(const std::vector<std::uint16_t>& logits,
                                        const std::vector<std::int32_t>& drafts,
                                        const std::vector<std::int32_t>& candidate_ids,
                                        const std::vector<float>& proposal_q,
                                        const std::vector<std::int32_t>& extents,
                                        const std::vector<std::int32_t>& initial_lengths,
                                        const std::vector<ops::SamplingConfig>& configs,
                                        const std::vector<std::int32_t>& token_counts) {
        SparseExpected expected{
            .licensed_tokens = std::vector<std::int32_t>(kSparseColumns * kSparseBatch, 0),
            .licensed_counts = std::vector<std::int32_t>(kSparseBatch),
            .accepted        = std::vector<std::int32_t>(kSparseBatch),
            .lengths         = initial_lengths,
            .anchors         = std::vector<std::int32_t>(kSparseBatch),
        };
        for (int row = 0; row < kSparseBatch; ++row) {
            const int extent = std::clamp(extents[static_cast<std::size_t>(row)], 0, kSparseDrafts);
            int accepted_count = 0;
            int terminal_token = 0;
            for (int column = 0; column <= extent; ++column) {
                const TargetDistribution target = sparse_target_distribution(
                    logits, row, column, configs[static_cast<std::size_t>(row)], token_counts,
                    drafts);
                if (column == extent) {
                    const double uniform =
                        oracle_uniform(configs[static_cast<std::size_t>(row)].seed,
                                       initial_lengths[static_cast<std::size_t>(row)] + extent + 1,
                                       ops::kSamplePurposeSpeculativeBonus);
                    terminal_token = sample_target_distribution(target, uniform);
                    break;
                }

                const int draft = drafts[static_cast<std::size_t>(row) * kSparseDrafts + column];
                if (!(configs[static_cast<std::size_t>(row)].temperature > 0.0f)) {
                    if (target.ids.front() == draft) {
                        ++accepted_count;
                        continue;
                    }
                    terminal_token = target.ids.front();
                    break;
                }

                const double p = distribution_probability(target, draft);
                const double q =
                    sparse_proposal_probability(candidate_ids, proposal_q, row, column, draft);
                const double uniform =
                    oracle_uniform(configs[static_cast<std::size_t>(row)].seed,
                                   initial_lengths[static_cast<std::size_t>(row)] + column + 1,
                                   ops::kSamplePurposeSpeculativeAccept);
                if (p >= q || uniform * q < p) {
                    ++accepted_count;
                    continue;
                }
                const double correction_uniform =
                    oracle_uniform(configs[static_cast<std::size_t>(row)].seed,
                                   initial_lengths[static_cast<std::size_t>(row)] + column + 1,
                                   ops::kSamplePurposeSpeculativeCorrection);
                terminal_token = sample_sparse_residual(target, candidate_ids, proposal_q, row,
                                                        column, correction_uniform);
                break;
            }

            const std::size_t row_base   = static_cast<std::size_t>(row) * kSparseColumns;
            const std::size_t draft_base = static_cast<std::size_t>(row) * kSparseDrafts;
            for (int item = 0; item < accepted_count; ++item) {
                expected.licensed_tokens[row_base + item] = drafts[draft_base + item];
            }
            expected.licensed_tokens[row_base + accepted_count]     = terminal_token;
            expected.licensed_counts[static_cast<std::size_t>(row)] = accepted_count + 1;
            expected.accepted[static_cast<std::size_t>(row)]        = accepted_count;
            expected.lengths[static_cast<std::size_t>(row)] += accepted_count + 1;
            expected.anchors[static_cast<std::size_t>(row)] = terminal_token;
        }
        return expected;
    }

    int execute_sparse_accept_case(
        const std::string& label, const std::vector<std::int32_t>& target_tokens,
        const std::vector<std::uint16_t>& logits, const std::vector<std::int32_t>& drafts,
        const std::vector<std::int32_t>& candidate_ids, const std::vector<float>& proposal_q,
        const std::vector<std::int32_t>& extents, const std::vector<std::int32_t>& initial_lengths,
        const std::vector<std::int32_t>& initial_anchors,
        const std::vector<ops::SamplingConfig>& host_configs,
        const std::vector<std::int32_t>& token_counts,
        ops::SpeculativeAcceptExecutionEnvelope envelope) {
        const SparseExpected expected =
            sparse_accept_oracle(logits, drafts, candidate_ids, proposal_q, extents,
                                 initial_lengths, host_configs, token_counts);

        DeviceBuffer d_target_tokens                    = to_device(target_tokens);
        DeviceBuffer d_logits                           = to_device(logits);
        DeviceBuffer d_drafts                           = to_device(drafts);
        DeviceBuffer d_candidate_ids                    = to_device(candidate_ids);
        DeviceBuffer d_proposal_q                       = to_device(proposal_q);
        DeviceBuffer d_extents                          = to_device(extents);
        DeviceBuffer d_token_counts                     = to_device(token_counts);
        std::vector<ops::SamplingConfig> device_configs = host_configs;
        for (int row = 0; row < kSparseBatch; ++row) {
            device_configs[static_cast<std::size_t>(row)].token_counts =
                static_cast<std::int32_t*>(d_token_counts.p) +
                static_cast<std::size_t>(row) * kSparseTokenDomain;
        }
        std::optional<DeviceBuffer> d_masks;
        if (!masks.empty()) {
            d_masks.emplace(to_device(masks));
            for (int row = 0; row < kSparseBatch; ++row) {
                device_configs[static_cast<std::size_t>(row)].mask = {
                    static_cast<const std::uint32_t*>(d_masks->p) +
                        static_cast<std::size_t>(row) * kSparseColumns * mask_words,
                    mask_words};
            }
        }
        DeviceBuffer d_configs = to_device(device_configs);

        GuardedDeviceBuffer d_lengths(kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_anchors(kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_licensed(kSparseColumns * kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_licensed_counts(kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_accepted(kSparseBatch * sizeof(std::int32_t));
        initialize(d_lengths, initial_lengths);
        initialize(d_anchors, initial_anchors);
        d_licensed.fill(0xcd);
        d_licensed_counts.fill(0xef);
        d_accepted.fill(0xab);

        Tensor target_tensor(d_target_tokens.p, DType::I32, {kSparseColumns, kSparseBatch});
        Tensor logits_tensor(d_logits.p, DType::BF16,
                             {kSparsePhysicalRows, kSparseColumns, kSparseBatch});
        Tensor drafts_tensor(d_drafts.p, DType::I32, {kSparseDrafts, kSparseBatch});
        Tensor candidate_tensor(d_candidate_ids.p, DType::I32,
                                {kSparseCandidates, kSparseDrafts, kSparseBatch});
        Tensor q_tensor(d_proposal_q.p, DType::FP32,
                        {kSparseCandidates, kSparseDrafts, kSparseBatch});
        Tensor extent_tensor(d_extents.p, DType::I32, {kSparseBatch});
        Tensor lengths_tensor(d_lengths.data(), DType::I32, {kSparseBatch});
        Tensor anchors_tensor(d_anchors.data(), DType::I32, {kSparseBatch});
        Tensor licensed_tensor(d_licensed.data(), DType::I32, {kSparseColumns, kSparseBatch});
        Tensor licensed_counts_tensor(d_licensed_counts.data(), DType::I32, {kSparseBatch});
        Tensor accepted_tensor(d_accepted.data(), DType::I32, {kSparseBatch});
        const std::size_t workspace_bytes =
            ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                kSparseTokenDomain, envelope, kSparseDrafts, kSparseDrafts, kSparseBatch,
                kSparseBatch);
        GuardedDeviceBuffer scratch(std::max<std::size_t>(workspace_bytes, 1));
        WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
        const auto launch = [&](cudaStream_t stream) {
            ops::speculative_accept_sparse_drafts(
                target_tensor, logits_tensor, drafts_tensor, candidate_tensor, q_tensor,
                extent_tensor, lengths_tensor, anchors_tensor, licensed_tensor,
                licensed_counts_tensor, accepted_tensor, kSparseTokenDomain,
                static_cast<const ops::SamplingConfig*>(d_configs.p), envelope, workspace, stream);
        };
        launch(nullptr);
        cuda_synchronize();

        const auto check_result = [&](const SparseExpected& wanted, const std::string& phase) {
            int checks = verify_exact((label + phase + " licensed tokens").c_str(),
                                      read<std::int32_t>(d_licensed, wanted.licensed_tokens.size()),
                                      wanted.licensed_tokens);
            checks += verify_exact((label + phase + " licensed counts").c_str(),
                                   read<std::int32_t>(d_licensed_counts, kSparseBatch),
                                   wanted.licensed_counts);
            checks += verify_exact((label + phase + " accepted drafts").c_str(),
                                   read<std::int32_t>(d_accepted, kSparseBatch), wanted.accepted);
            checks += verify_exact((label + phase + " provisional lengths").c_str(),
                                   read<std::int32_t>(d_lengths, kSparseBatch), wanted.lengths);
            checks += verify_exact((label + phase + " provisional anchors").c_str(),
                                   read<std::int32_t>(d_anchors, kSparseBatch), wanted.anchors);
            return checks;
        };
        int failures = check_result(expected, " eager");
        if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) ++failures;
        failures += scratch.verify_guards(label + " workspace");
        observed_workspace = std::max(observed_workspace, workspace.peak_used());
        if (kSparseBatch == 1 || kSparseBatch == 8) {
            cudaStream_t stream = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "sparse graph stream");
            {
                DecodeGraphDefinition definition;
                DecodeGraphExecutable executable;
                definition.capture(stream, [&] { launch(stream); });
                executable.instantiate(definition);
                for (int pass = 0; pass < 3; ++pass) {
                    auto next_extents = extents;
                    auto next_lengths = initial_lengths;
                    if (pass == 1)
                        for (int row = 0; row < kSparseBatch; ++row) {
                            next_extents[row] = (row + kSparseDrafts / 2) % (kSparseDrafts + 1);
                            next_lengths[row] += 4096;
                        }
                    const bool change_inputs =
                        !envelope.all_rows_greedy_without_penalties && kSparseBatch == 8 &&
                        (kSparseDrafts == 1 || kSparseDrafts == 7 || kSparseDrafts == 15);
                    std::vector<std::uint16_t> replay_logits;
                    std::vector<float> replay_q;
                    if (change_inputs) {
                        replay_logits = logits;
                        replay_q      = proposal_q;
                        if (pass == 1) {
                            for (auto& value : replay_logits)
                                value = f32_to_bf16(2.0f * bf16_to_f32(value));
                            for (int row = 0; row < kSparseBatch; ++row)
                                if (host_configs[row].temperature > 0.0f)
                                    for (int column = 0; column < kSparseDrafts; ++column) {
                                        const int base = sparse_candidate_index(row, column, 0);
                                        int rank       = 0;
                                        while (rank < 16 &&
                                               candidate_ids[base + rank] !=
                                                   drafts[row * kSparseDrafts + column])
                                            ++rank;
                                        if (rank == 16)
                                            throw std::runtime_error("invalid proposal fixture");
                                        std::fill_n(replay_q.begin() + base, 16, 0.0f);
                                        replay_q[base + rank]            = 0.75f;
                                        replay_q[base + (rank + 1) % 16] = 0.25f;
                                    }
                        }
                        d_logits.copy_from_host(replay_logits.data(),
                                                replay_logits.size() * sizeof(std::uint16_t));
                        d_proposal_q.copy_from_host(replay_q.data(),
                                                    replay_q.size() * sizeof(float));
                    }
                    const auto& current_logits = change_inputs ? replay_logits : logits;
                    const auto& current_q      = change_inputs ? replay_q : proposal_q;
                    initialize(d_lengths, next_lengths);
                    initialize(d_anchors, initial_anchors);
                    d_extents.copy_from_host(next_extents.data(),
                                             next_extents.size() * sizeof(int));
                    cuda_synchronize();
                    executable.launch(stream);
                    cuda_synchronize(stream);
                    const auto wanted = sparse_accept_oracle(current_logits, drafts, candidate_ids,
                                                             current_q, next_extents, next_lengths,
                                                             host_configs, token_counts);
                    failures += check_result(wanted, " replay " + std::to_string(pass));
                    if (change_inputs) {
                        failures += verify_exact(
                            "replay logits readonly",
                            from_device<std::uint16_t>(d_logits, current_logits.size()),
                            current_logits);
                        failures += verify_exact("replay q readonly",
                                                 from_device<float>(d_proposal_q, current_q.size()),
                                                 current_q);
                    }
                    failures += verify_exact("sparse extent readonly",
                                             from_device<int>(d_extents, next_extents.size()),
                                             next_extents);
                    failures += scratch.verify_guards(label + " replay workspace");
                    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes)
                        ++failures;
                }
            }
            cuda_check(cudaStreamDestroy(stream), "destroy sparse graph stream");
        }
        failures += verify_exact("sparse logits readonly",
                                 from_device<std::uint16_t>(d_logits, logits.size()), logits);
        failures +=
            verify_exact("sparse targets readonly",
                         from_device<int>(d_target_tokens, target_tokens.size()), target_tokens);
        failures += verify_exact("sparse drafts readonly",
                                 from_device<int>(d_drafts, drafts.size()), drafts);
        failures +=
            verify_exact("sparse candidate ids readonly",
                         from_device<int>(d_candidate_ids, candidate_ids.size()), candidate_ids);
        failures += verify_exact("sparse actual q readonly",
                                 from_device<float>(d_proposal_q, proposal_q.size()), proposal_q);
        const auto* config_bytes = reinterpret_cast<const std::uint8_t*>(device_configs.data());
        failures += verify_exact(
            "sparse configs readonly", from_device<std::uint8_t>(d_configs, d_configs.bytes),
            std::vector<std::uint8_t>(config_bytes, config_bytes + d_configs.bytes));
        failures += verify_exact((label + " token counts read-only").c_str(),
                                 from_device<std::int32_t>(d_token_counts, token_counts.size()),
                                 token_counts);
        failures += d_lengths.verify_guards((label + " lengths guards").c_str());
        failures += d_anchors.verify_guards((label + " anchors guards").c_str());
        failures += d_licensed.verify_guards((label + " licensed guards").c_str());
        failures += d_licensed_counts.verify_guards((label + " licensed-count guards").c_str());
        failures += d_accepted.verify_guards((label + " accepted guards").c_str());
        return failures;
    }

    int sparse_greedy_direct_case(int pattern = 0, bool general = false) {
        std::vector<std::int32_t> targets(kSparseColumns * kSparseBatch),
            drafts(kSparseDrafts * kSparseBatch);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(-20.0f));
        std::vector<int> ids(kSparseCandidates * kSparseDrafts * kSparseBatch),
            extents(kSparseBatch), lengths(kSparseBatch), anchors(kSparseBatch, -1);
        std::vector<float> q(ids.size(), 0.0f);
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<int> history(kSparseTokenDomain * kSparseBatch, 3);
        for (int row = 0; row < kSparseBatch; ++row) {
            const int kind   = (row + pattern) % 7;
            extents[row]     = kind == 0   ? kSparseDrafts
                               : kind == 1 ? 0
                               : kind == 5 ? -2
                               : kind == 6 ? kSparseDrafts + 3
                                           : kSparseDrafts;
            const int extent = std::clamp(extents[row], 0, kSparseDrafts);
            const int reject = kind == 2   ? 0
                               : kind == 3 ? kSparseDrafts / 2
                               : kind == 4 ? kSparseDrafts - 1
                                           : extent;
            lengths[row]     = 4096 + row * 13;
            for (int col = 0; col < kSparseColumns; ++col) {
                const int target                             = 100000 + row * 128 + col;
                targets[row * kSparseColumns + col]          = target;
                logits[sparse_logit_index(row, col, target)] = f32_to_bf16(20.0f);
                for (int v = kSparseTokenDomain; v < kSparsePhysicalRows; ++v)
                    logits[sparse_logit_index(row, col, v)] = f32_to_bf16(100.0f);
            }
            for (int col = 0; col < kSparseDrafts; ++col) {
                const int base = sparse_candidate_index(row, col, 0);
                for (int rank = 0; rank < kSparseCandidates; ++rank)
                    ids[base + rank] = 30000 + row * 1024 + col * 16 + rank;
                ids[base]                         = targets[row * kSparseColumns + col];
                const int rank                    = col == reject && reject < extent ? 15 : 0;
                drafts[row * kSparseDrafts + col] = ids[base + rank];
                q[base + rank]                    = 1.0f;
            }
        }
        return execute_sparse_accept_case("sparse greedy K=" + std::to_string(kSparseDrafts) +
                                              " B=" + std::to_string(kSparseBatch),
                                          targets, logits, drafts, ids, q, extents, lengths,
                                          anchors, configs, history, {!general});
    }

    int generated_general_case() {
        std::vector<int> targets(kSparseColumns * kSparseBatch),
            drafts(kSparseDrafts * kSparseBatch);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(-20.0f));
        std::vector<int> ids(kSparseCandidates * kSparseDrafts * kSparseBatch),
            extents(kSparseBatch), lengths(kSparseBatch), anchors(kSparseBatch, -1);
        std::vector<float> q(ids.size(), 0.0f);
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<int> history(kSparseTokenDomain * kSparseBatch, 0);
        for (int row = 0; row < kSparseBatch; ++row) {
            const int kind        = (row + kSparseDrafts) % 4;
            auto& cfg             = configs[row];
            cfg.temperature       = (kind == 1 || kind == 2) ? 0.0f : 0.75f;
            cfg.top_k             = kind == 3 ? 13 : 20;
            cfg.top_p             = kind == 3 ? 0.78f : 1.0f;
            cfg.min_p             = kind == 3 ? 0.3f : 0.0f;
            cfg.presence_penalty  = kind == 1 || kind == 3 ? 0.5f : 0.0f;
            cfg.frequency_penalty = kind == 1 || kind == 3 ? 0.125f : 0.0f;
            cfg.seed              = 10007 + 31 * row + 17 * kSparseDrafts;
            lengths[row]          = 9000 + 37 * row;
            const int extent      = row == kSparseBatch - 1 ? kSparseDrafts
                                    : row % 4 == 0          ? 0
                                                            : std::max(1, kSparseDrafts - row % 3);
            extents[row]          = extent;
            const int style       = (2 * row + kSparseDrafts) % 5;
            const int reject      = style == 0 || style == 4 ? extent
                                    : style == 1             ? 0
                                    : style == 2             ? extent / 2
                                                             : std::max(0, extent - 1);
            for (int col = 0; col < kSparseDrafts; ++col) {
                const int base = sparse_candidate_index(row, col, 0);
                for (int rank = 0; rank < 16; ++rank) {
                    ids[base + rank] = 10000 + row * 4096 + col * 32 + rank;
                    q[base + rank]   = cfg.temperature > 0.0f ? 1.0f / 16.0f
                                       : rank == 0            ? 1.0f
                                                              : 0.0f;
                    history[row * kSparseTokenDomain + ids[base + rank]] = (col + rank) % 3;
                }
                drafts[row * kSparseDrafts + col] = ids[base];
                if (kind == 0 && col == reject && reject < extent)
                    drafts[row * kSparseDrafts + col] = ids[base + 9];
            }
            for (int col = 0; col < kSparseColumns; ++col) {
                const int base = 10000 + row * 4096 + col * 32;
                if (col < reject && col < kSparseDrafts) {
                    targets[row * kSparseColumns + col] = drafts[row * kSparseDrafts + col];
                    logits[sparse_logit_index(row, col, targets[row * kSparseColumns + col])] =
                        f32_to_bf16(16.0f);
                } else {
                    for (int rank = 0; rank < 20; ++rank)
                        logits[sparse_logit_index(row, col, base + 1 + rank)] =
                            f32_to_bf16(2.0f - rank * 0.0625f);
                    targets[row * kSparseColumns + col] = base + 1;
                }
                for (int v = kSparseTokenDomain; v < kSparsePhysicalRows; ++v)
                    logits[sparse_logit_index(row, col, v)] = f32_to_bf16(100.0f);
            }
        }
        return execute_sparse_accept_case("sparse general K=" + std::to_string(kSparseDrafts) +
                                              " B=" + std::to_string(kSparseBatch),
                                          targets, logits, drafts, ids, q, extents, lengths,
                                          anchors, configs, history, {false});
    }

    // Port of upstream 08902f73 grammar_mask_case: only a=V-2 and b=V-1 are legal at every
    // column, both far below the illegal 12.0 plateau (x=42 is the raw target of every column).
    // p(a)=p(b)=.5, q(a)=q(x)=.5: a accepts; the illegal x is rejected (p=0) and corrects to b.
    int grammar_mask_case() {
        std::vector<int> targets(kSparseColumns * kSparseBatch, 42),
            drafts(kSparseDrafts * kSparseBatch),
            ids(kSparseCandidates * kSparseDrafts * kSparseBatch),
            extents(kSparseBatch, kSparseDrafts), lengths(kSparseBatch, 70),
            anchors(kSparseBatch, 0);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(12.0f));
        std::vector<float> q(ids.size(), 0.0f);
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<int> history(static_cast<std::size_t>(kSparseTokenDomain) * kSparseBatch, 0);
        masks.assign(static_cast<std::size_t>(kSparseColumns) * kSparseBatch * mask_words, 0);
        const int a = kSparseTokenDomain - 2, b = kSparseTokenDomain - 1, x = 42;
        for (int row = 0; row < kSparseBatch; ++row) {
            configs[row].temperature = row < kSparseBatch / 2 ? 0.0f : 1.0f;
            configs[row].seed        = 8123 + row;
            for (int col = 0; col < kSparseColumns; ++col) {
                for (int token : {a, b}) {
                    masks[(static_cast<std::size_t>(row) * kSparseColumns + col) * mask_words +
                          token / 32] |= 1u << (token % 32);
                    logits[sparse_logit_index(row, col, token)] = f32_to_bf16(0.0f);
                }
                if (col < kSparseDrafts) {
                    const int base = sparse_candidate_index(row, col, 0);
                    for (int c = 0; c < 16; ++c) ids[base + c] = c + 100;
                    ids[base]     = a;
                    ids[base + 1] = x;
                    q[base] = q[base + 1]             = 0.5f;
                    drafts[row * kSparseDrafts + col] = (row + col) % 2 ? x : a;
                }
            }
        }
        return execute_sparse_accept_case("sparse grammar mask K=" + std::to_string(kSparseDrafts),
                                          targets, logits, drafts, ids, q, extents, lengths,
                                          anchors, configs, history, {false});
    }

    int repeated_history_case(bool stochastic) {
        std::vector<int> targets(kSparseColumns * kSparseBatch, 100),
            drafts(kSparseDrafts * kSparseBatch, 100);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(-20.0f));
        std::vector<int> ids(kSparseCandidates * kSparseDrafts * kSparseBatch),
            extents(kSparseBatch, kSparseDrafts), lengths(kSparseBatch, 7000),
            anchors(kSparseBatch, -1);
        std::vector<float> q(ids.size(), 0.0f);
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<int> history(kSparseTokenDomain * kSparseBatch, 0);
        for (int row = 0; row < kSparseBatch; ++row) {
            auto& cfg                               = configs[row];
            cfg.temperature                         = stochastic ? 1.0f : 0.0f;
            cfg.top_k                               = 2;
            cfg.frequency_penalty                   = 1.0f;
            cfg.seed                                = 523 + row;
            history[row * kSparseTokenDomain + 100] = 2;
            for (int col = 0; col < kSparseDrafts; ++col) {
                const int base = sparse_candidate_index(row, col, 0);
                for (int rank = 0; rank < 16; ++rank) ids[base + rank] = 100 + rank;
                q[base] = 1.0f;
            }
            for (int col = 0; col < kSparseColumns; ++col) {
                logits[sparse_logit_index(row, col, 100)] = f32_to_bf16(5.0f);
                logits[sparse_logit_index(row, col, 101)] = f32_to_bf16(0.5f);
            }
        }
        return execute_sparse_accept_case(
            "sparse repeated-history K=" + std::to_string(kSparseDrafts) +
                (stochastic ? " stochastic" : " greedy"),
            targets, logits, drafts, ids, q, extents, lengths, anchors, configs, history, {false});
    }

    int sparse_general_mixed_case() {
        std::vector<std::int32_t> targets(kSparseColumns * kSparseBatch);
        std::vector<std::uint16_t> logits(static_cast<std::size_t>(kSparsePhysicalRows) *
                                              kSparseColumns * kSparseBatch,
                                          f32_to_bf16(-20.0f));
        std::vector<std::int32_t> drafts(kSparseDrafts * kSparseBatch);
        std::vector<std::int32_t> candidate_ids(kSparseCandidates * kSparseDrafts * kSparseBatch);
        std::vector<float> proposal_q(candidate_ids.size(), 0.0f);
        std::vector<std::int32_t> extents{2, 1, 1, kSparseDrafts, 0, 3, 2, 4};
        for (auto& extent : extents) extent = std::min(extent, kSparseDrafts);
        std::vector<std::int32_t> lengths(kSparseBatch);
        std::vector<std::int32_t> anchors(kSparseBatch, -1);
        std::vector<ops::SamplingConfig> configs(kSparseBatch);
        std::vector<std::int32_t> token_counts(
            static_cast<std::size_t>(kSparseTokenDomain) * kSparseBatch, 0);

        for (int row = 0; row < kSparseBatch; ++row) {
            lengths[static_cast<std::size_t>(row)]             = 4000 + 31 * row;
            configs[static_cast<std::size_t>(row)].temperature = 1.0f;
            configs[static_cast<std::size_t>(row)].top_k       = 1;
            configs[static_cast<std::size_t>(row)].top_p       = 1.0f;
            configs[static_cast<std::size_t>(row)].seed        = 100 + row;
            for (int draft = 0; draft < kSparseDrafts; ++draft) {
                for (int candidate = 0; candidate < kSparseCandidates; ++candidate) {
                    candidate_ids[sparse_candidate_index(row, draft, candidate)] =
                        1000 + row * 2048 + draft * kSparseCandidates + candidate;
                }
                const std::size_t base = sparse_candidate_index(row, draft, 0);
                drafts[static_cast<std::size_t>(row) * kSparseDrafts + draft] = candidate_ids[base];
                proposal_q[base]                                              = 0.5f;
                proposal_q[base + 1]                                          = 0.5f;
            }

            const int extent = extents[static_cast<std::size_t>(row)];
            for (int column = 0; column < kSparseColumns; ++column) {
                const int token =
                    column < extent ? drafts[static_cast<std::size_t>(row) * kSparseDrafts + column]
                                    : 200000 + row * 8 + column;
                targets[static_cast<std::size_t>(row) * kSparseColumns + column] = token;
                logits[sparse_logit_index(row, column, token)] = f32_to_bf16(12.0f);
                if (column > extent) {
                    const int poison = 240000 + row * kSparseColumns + column;
                    logits[sparse_logit_index(row, column, poison)] = f32_to_bf16(24.0f);
                    targets[static_cast<std::size_t>(row) * kSparseColumns + column] = poison;
                }
            }
        }

        // A committed-history penalty changes row 0's first target argmax.
        configs[0].temperature      = 0.0f;
        configs[0].presence_penalty = 2.0f;
        const int row0_draft        = drafts[0];
        const int row0_alternative  = candidate_ids[sparse_candidate_index(0, 0, 1)];
        logits[sparse_logit_index(0, 0, row0_draft)]       = f32_to_bf16(5.0f);
        logits[sparse_logit_index(0, 0, row0_alternative)] = f32_to_bf16(4.0f);
        targets[0]                                         = row0_draft;
        token_counts[static_cast<std::size_t>(row0_draft)] = 1;

        // p(d)=1/4, q(d)=1/2, and u is inside (p,p/q): row 1 must accept by p/q.
        configs[1].top_k         = 4;
        const int previous_draft = candidate_ids[sparse_candidate_index(1, 0, 0)];
        logits[sparse_logit_index(1, 0, previous_draft)] = f32_to_bf16(-20.0f);
        candidate_ids[sparse_candidate_index(1, 0, 0)]   = kSparseTokenDomain - 1;
        drafts[kSparseDrafts]                            = kSparseTokenDomain - 1;
        targets[kSparseColumns] = candidate_ids[sparse_candidate_index(1, 0, 1)];
        for (int candidate = 0; candidate < 4; ++candidate) {
            const int token = candidate_ids[sparse_candidate_index(1, 0, candidate)];
            logits[sparse_logit_index(1, 0, token)] = f32_to_bf16(0.0f);
        }
        configs[1].seed = find_seed_for_uniform(lengths[1] + 1,
                                                ops::kSamplePurposeSpeculativeAccept, 0.30f, 0.45f);

        // Row 2's draft has p=0. Sparse q removes the two larger target masses, so candidate 3 is
        // the only positive residual correction.
        configs[2].top_k     = 3;
        const int row2_draft = candidate_ids[sparse_candidate_index(2, 0, 0)];
        const int row2_a     = candidate_ids[sparse_candidate_index(2, 0, 1)];
        const int row2_b     = candidate_ids[sparse_candidate_index(2, 0, 2)];
        const int row2_c     = candidate_ids[sparse_candidate_index(2, 0, 3)];
        logits[sparse_logit_index(2, 0, row2_draft)] = f32_to_bf16(-20.0f);
        logits[sparse_logit_index(2, 0, row2_a)]     = f32_to_bf16(3.0f);
        logits[sparse_logit_index(2, 0, row2_b)]     = f32_to_bf16(2.0f);
        logits[sparse_logit_index(2, 0, row2_c)]     = f32_to_bf16(1.0f);
        targets[2 * kSparseColumns]                  = row2_a;
        const std::size_t row2_q                     = sparse_candidate_index(2, 0, 0);
        std::fill_n(proposal_q.begin() + static_cast<std::ptrdiff_t>(row2_q), kSparseCandidates,
                    0.0f);
        proposal_q[row2_q]     = 0.05f;
        proposal_q[row2_q + 1] = 0.70f;
        proposal_q[row2_q + 2] = 0.25f;

        // Row 5 is raw greedy inside the general mixed-batch route.
        configs[5].temperature = 0.0f;

        return execute_sparse_accept_case(
            "sparse speculative general mixed B=8", targets, logits, drafts, candidate_ids,
            proposal_q, extents, lengths, anchors, configs, token_counts,
            ops::SpeculativeAcceptExecutionEnvelope{.all_rows_greedy_without_penalties = false});
    }
};

AcceptExpected accept_state_oracle(const std::vector<std::int32_t>& drafts, std::int32_t accepted,
                                   std::int32_t terminal_token, std::int32_t initial_length) {
    const int k = static_cast<int>(drafts.size());
    AcceptExpected expected{
        .sampled     = std::vector<std::int32_t>(static_cast<std::size_t>(k + 1), 0),
        .num_sampled = accepted + 1,
        .accepted    = accepted,
        .length      = initial_length + accepted + 1,
        .token       = terminal_token,
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
    config.token_counts      = static_cast<std::int32_t*>(d_counts.p);
    DeviceBuffer d_config    = device_config(config);
    const auto config_before = from_device<std::uint8_t>(d_config, sizeof(ops::SamplingConfig));

    GuardedDeviceBuffer d_length(sizeof(std::int32_t));
    GuardedDeviceBuffer d_token(sizeof(std::int32_t));
    GuardedDeviceBuffer d_sampled(static_cast<std::size_t>(k + 1) * sizeof(std::int32_t));
    GuardedDeviceBuffer d_num(sizeof(std::int32_t));
    GuardedDeviceBuffer d_accepted(sizeof(std::int32_t));
    DeviceBuffer d_extent = to_device<std::int32_t>({k});
    initialize(d_length, std::vector<std::int32_t>{initial_length});
    initialize(d_token, std::vector<std::int32_t>{-1234567});
    d_sampled.fill(0x9d);
    initialize(d_num, std::vector<std::int32_t>{-11});
    initialize(d_accepted, std::vector<std::int32_t>{-13});

    Tensor targets(d_targets.p, DType::I32, {k + 1});
    Tensor logits(d_logits.p, DType::BF16, {physical_rows, k + 1});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k});
    Tensor extent(d_extent.p, DType::I32, {1});
    Tensor length(d_length.data(), DType::I32, {1});
    Tensor token(d_token.data(), DType::I32, {1});
    Tensor sampled(d_sampled.data(), DType::I32, {k + 1});
    Tensor num_sampled(d_num.data(), DType::I32, {1});
    Tensor accepted(d_accepted.data(), DType::I32, {1});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, 1, 1);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::speculative_accept_greedy_drafts(
        targets, logits, draft_tensor, extent, length, token, sampled, num_sampled, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_config.p), workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact((label + " sampled").c_str(), read<std::int32_t>(d_sampled, k + 1),
                                expected.sampled);
    failures += verify_exact((label + " num sampled").c_str(), read<std::int32_t>(d_num, 1),
                             {expected.num_sampled});
    failures += verify_exact((label + " accepted").c_str(), read<std::int32_t>(d_accepted, 1),
                             {expected.accepted});
    failures += verify_exact((label + " length").c_str(), read<std::int32_t>(d_length, 1),
                             {expected.length});
    failures +=
        verify_exact((label + " token").c_str(), read<std::int32_t>(d_token, 1), {expected.token});

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
    for (int i = 0; i < expected.num_sampled; ++i) {
        ++expected_counts[static_cast<std::size_t>(expected.sampled[static_cast<std::size_t>(i)])];
    }
    failures +=
        verify_exact((label + " token counts").c_str(),
                     from_device<std::int32_t>(d_counts, expected_counts.size()), expected_counts);

    failures += d_length.verify_guards((label + " length guards").c_str());
    failures += d_token.verify_guards((label + " token guards").c_str());
    failures += d_sampled.verify_guards((label + " sampled guards").c_str());
    failures += d_num.verify_guards((label + " num guards").c_str());
    failures += d_accepted.verify_guards((label + " accepted guards").c_str());
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
    const auto expected               = accept_state_oracle(
        drafts, accepted_count, targets[static_cast<std::size_t>(accepted_count)], initial_length);
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
    const auto expected = accept_state_oracle(drafts, accepted, correction, initial_length);
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
    const auto expected = accept_state_oracle(drafts, 1, 4, 100);
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
    DeviceBuffer d_drafts  = to_device(drafts);
    DeviceBuffer d_extents = to_device<std::int32_t>({k, k});
    DeviceBuffer d_lengths = to_device<std::int32_t>({100, 200});
    DeviceBuffer d_anchors = to_device<std::int32_t>({-1, -1});
    DeviceBuffer d_licensed(static_cast<std::size_t>(columns) * batch * sizeof(std::int32_t));
    DeviceBuffer d_counts(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    DeviceBuffer d_accepted(static_cast<std::size_t>(batch) * sizeof(std::int32_t));

    ops::SamplingConfig config{};
    config.temperature = 1.0f;
    config.top_k       = 1;
    const std::vector<ops::SamplingConfig> configs{config, config};
    DeviceBuffer d_configs = to_device(configs);

    Tensor targets(d_targets.p, DType::I32, {columns, batch});
    Tensor logits_tensor(d_logits.p, DType::BF16, {physical_rows, columns, batch});
    Tensor draft_tensor(d_drafts.p, DType::I32, {k, batch});
    Tensor extents(d_extents.p, DType::I32, {batch});
    Tensor lengths(d_lengths.p, DType::I32, {batch});
    Tensor anchors(d_anchors.p, DType::I32, {batch});
    Tensor licensed(d_licensed.p, DType::I32, {columns, batch});
    Tensor counts(d_counts.p, DType::I32, {batch});
    Tensor accepted(d_accepted.p, DType::I32, {batch});
    const std::size_t workspace_bytes =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(token_domain, k, k, batch,
                                                                       batch);
    WorkspaceArena workspace(workspace_bytes);
    ops::speculative_accept_greedy_drafts(
        targets, logits_tensor, draft_tensor, extents, lengths, anchors, licensed, counts, accepted,
        token_domain, static_cast<const ops::SamplingConfig*>(d_configs.p), workspace, nullptr);
    cuda_synchronize();

    int failures = verify_exact("speculative sampling B=2 licensed",
                                from_device<std::int32_t>(d_licensed, columns * batch),
                                {10, 20, 0, 0, 30, 31, 32, 33});
    failures += verify_exact("speculative sampling B=2 counts",
                             from_device<std::int32_t>(d_counts, batch), {2, 4});
    failures += verify_exact("speculative sampling B=2 accepted",
                             from_device<std::int32_t>(d_accepted, batch), {1, 3});
    failures += verify_exact("speculative sampling B=2 lengths",
                             from_device<std::int32_t>(d_lengths, batch), {102, 204});
    failures += verify_exact("speculative sampling B=2 anchors",
                             from_device<std::int32_t>(d_anchors, batch), {20, 33});
    return failures;
}

int select_hidden_case(int rows, int columns, int accepted_value) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * columns);
    for (int col = 0; col < columns; ++col) {
        for (int row = 0; row < rows; ++row) {
            hidden[static_cast<std::size_t>(col) * rows + row] =
                static_cast<std::uint16_t>(0x0100u + ((col * 257 + row * 13) & 0x7fffu));
        }
    }
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows));
    std::copy_n(hidden.begin() + static_cast<std::ptrdiff_t>(accepted_value) * rows, rows,
                expected.begin());

    DeviceBuffer d_hidden   = to_device(hidden);
    DeviceBuffer d_accepted = to_device<std::int32_t>({accepted_value});
    GuardedDeviceBuffer d_out(static_cast<std::size_t>(rows) * sizeof(std::uint16_t));
    d_out.fill(0xcd);
    Tensor hidden_tensor(d_hidden.p, DType::BF16, {rows, columns});
    Tensor accepted(d_accepted.p, DType::I32, {1});
    Tensor out(d_out.data(), DType::BF16, {rows, 1});
    ops::speculative_select_accepted_hidden(hidden_tensor, accepted, out, nullptr);
    cuda_synchronize();

    const std::string label =
        "speculative select D=" + std::to_string(rows) + " A=" + std::to_string(accepted_value);
    int failures = verify_exact((label + " output").c_str(),
                                read<std::uint16_t>(d_out, expected.size()), expected);
    failures += verify_exact((label + " hidden unchanged").c_str(),
                             from_device<std::uint16_t>(d_hidden, hidden.size()), hidden);
    failures += verify_exact((label + " accepted unchanged").c_str(),
                             from_device<std::int32_t>(d_accepted, 1), {accepted_value});
    failures += d_out.verify_guards((label + " output guards").c_str());
    return failures;
}

int batched_select_hidden_case(int width, int batch) {
    constexpr int rows = 5120;
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(rows) * width * batch);
    for (std::size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = static_cast<std::uint16_t>(i * 257 + i / rows * 6113);
    std::vector<std::int32_t> selectors(batch);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows) * batch);
    DeviceBuffer input = to_device(hidden), indices = to_device(selectors);
    GuardedDeviceBuffer output(expected.size() * sizeof(std::uint16_t));
    output.fill(0xcd);
    Tensor h(input.p, DType::BF16, {rows, width, batch}), i(indices.p, DType::I32, {batch});
    Tensor o(output.data(), DType::BF16, {rows, batch});
    DeviceContext context;
    cuda_synchronize();
    const auto launch = [&] { ops::speculative_select_accepted_hidden(h, i, o, context.stream); };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (width == 16 && batch == 8) {
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    // Final commit N is positive here; a zero commit never issues this gather with selector -1.
    for (int phase = 0; phase < width; ++phase) {
        for (int b = 0; b < batch; ++b) {
            selectors[b] = (phase + 3 * b) % width;
            std::copy_n(hidden.begin() + static_cast<std::size_t>(b * width + selectors[b]) * rows,
                        rows, expected.begin() + b * rows);
        }
        CUDA_CHECK(cudaMemcpyAsync(indices.p, selectors.data(), indices.bytes,
                                   cudaMemcpyHostToDevice, context.stream));
        if (graph.ready())
            graph.launch(context.stream);
        else
            launch();
        context.synchronize();
        failures += verify_exact("committed hidden N-1",
                                 read<std::uint16_t>(output, expected.size()), expected);
        failures += verify_exact("selectors unchanged", from_device<std::int32_t>(indices, batch),
                                 selectors);
    }
    failures +=
        verify_exact("hidden unchanged", from_device<std::uint16_t>(input, hidden.size()), hidden);
    failures += output.verify_guards("committed hidden");
    return failures;
}

int remap_case(int token_count) {
    constexpr int map_size = 131072;
    std::vector<std::int32_t> id_map(map_size);
    for (int i = 0; i < map_size; ++i) {
        const auto value                    = 65537u * static_cast<std::uint32_t>(i) + 17u;
        id_map[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(value & (map_size - 1u));
    }
    std::vector<std::int32_t> proposals(static_cast<std::size_t>(token_count));
    for (int i = 0; i < token_count; ++i) {
        proposals[static_cast<std::size_t>(i)] =
            i == 0 ? 0 : (i == token_count - 1 ? map_size - 1 : (7919 * i) & (map_size - 1));
    }
    std::vector<std::int32_t> expected(proposals.size());
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        expected[i] = id_map[static_cast<std::size_t>(proposals[i])];
    }

    DeviceBuffer d_map = to_device(id_map);
    GuardedDeviceBuffer d_proposals(proposals.size() * sizeof(std::int32_t));
    initialize(d_proposals, proposals);
    Tensor proposal_tensor(d_proposals.data(), DType::I32, {token_count});
    ops::proposal_remap_token_ids(proposal_tensor, static_cast<const std::int32_t*>(d_map.p),
                                  map_size, nullptr);
    cuda_synchronize();

    const std::string label = "proposal remap T=" + std::to_string(token_count);
    int failures            = verify_exact((label + " in-place output").c_str(),
                                           read<std::int32_t>(d_proposals, proposals.size()), expected);
    failures += verify_exact((label + " map unchanged").c_str(),
                             from_device<std::int32_t>(d_map, id_map.size()), id_map);
    failures += d_proposals.verify_guards((label + " guards").c_str());
    return failures;
}

int transforms_conformance() {
    int failures = 0;
    for (int k = 1; k <= 15; ++k)
        for (int batch : {1, 8}) failures += prepare_verify_case(k, batch);
    for (int width = 2; width <= 16; ++width)
        for (int batch : {1, 8}) failures += batched_select_hidden_case(width, batch);
    failures += select_hidden_case(5120, 6, 0);
    failures += select_hidden_case(5120, 6, 5);
    failures += select_hidden_case(2048, 16, 7);
    failures += remap_case(1);
    failures += remap_case(15);
    failures += remap_case(120);
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Token masks in speculative_accept_greedy_drafts (MTP: one-hot greedy drafts). Column i of a
// masked row reads mask position i. Routes: the single-block kernel with the small builder
// (domain <= 256), the multi-block partial/group route (domain > 256, K+1 <= 16) and the
// single-block fast fallback (domain > 256, K+1 > 16).

struct MaskedAcceptBatch {
    int domain   = 0;
    int physical = 0;
    int k        = 0;
    int batch    = 0;
    std::vector<float> logits;                     // [physical, K+1, B], BF16-representable
    std::vector<std::int32_t> targets;             // [K+1, B] raw argmax input of the fast path
    std::vector<std::int32_t> drafts;              // [K, B]
    std::vector<std::int32_t> extents;             // [B]
    std::vector<std::int32_t> lengths;             // [B]
    std::vector<ops::SamplingConfig> configs;      // [B]; mask and token_counts are bound here
    std::vector<std::vector<std::uint32_t>> masks; // per row: empty = unmasked, else (K+1)*words
    std::vector<std::int32_t> counts;              // [domain, B] committed counts, or empty = null

    MaskedAcceptBatch(int token_domain, int physical_rows, int drafts_per_row, int rows)
        : domain(token_domain), physical(physical_rows), k(drafts_per_row), batch(rows),
          logits(static_cast<std::size_t>(physical_rows) * (drafts_per_row + 1) * rows, -20.0f),
          targets(static_cast<std::size_t>(drafts_per_row + 1) * rows, 0),
          drafts(static_cast<std::size_t>(drafts_per_row) * rows, 0), extents(rows, drafts_per_row),
          lengths(rows, 500), configs(rows), masks(rows) {}

    int words() const { return (domain + 31) / 32; }

    float& logit(int row, int col, int token) {
        return logits[(static_cast<std::size_t>(row) * (k + 1) + col) * physical + token];
    }

    float logit_at(int row, int col, int token) const {
        return logits[(static_cast<std::size_t>(row) * (k + 1) + col) * physical + token];
    }

    void set_legal(int row, int col, const std::vector<int>& legal) {
        auto& mask = masks[static_cast<std::size_t>(row)];
        if (mask.empty()) { mask.assign(static_cast<std::size_t>(k + 1) * words(), 0u); }
        auto* position = mask.data() + static_cast<std::size_t>(col) * words();
        std::fill_n(position, words(), 0u);
        for (int token : legal) { position[token / 32] |= 1u << (token % 32); }
    }

    bool legal(int row, int col, int token) const {
        const auto& mask = masks[static_cast<std::size_t>(row)];
        if (mask.empty()) { return true; }
        return (mask[static_cast<std::size_t>(col) * words() + token / 32] >> (token % 32)) & 1u;
    }
};

struct MaskedAcceptResult {
    std::vector<std::int32_t> licensed, counts, accepted, lengths, anchors;
};

class MaskedAcceptRunner {
public:
    explicit MaskedAcceptRunner(const MaskedAcceptBatch& b)
        : b_(b), logits_bits_(b.logits.size()), configs_(b.configs) {
        for (std::size_t i = 0; i < b.logits.size(); ++i) {
            logits_bits_[i] = f32_to_bf16(b.logits[i]);
        }
        d_logits_  = to_device(logits_bits_);
        d_targets_ = to_device(b.targets);
        d_drafts_  = to_device(b.drafts);
        d_extents_ = to_device(b.extents);
        std::vector<std::uint32_t> all;
        std::vector<std::ptrdiff_t> offsets(static_cast<std::size_t>(b.batch), -1);
        for (int row = 0; row < b.batch; ++row) {
            const auto& mask = b.masks[static_cast<std::size_t>(row)];
            if (mask.empty()) { continue; }
            offsets[static_cast<std::size_t>(row)] = static_cast<std::ptrdiff_t>(all.size());
            all.insert(all.end(), mask.begin(), mask.end());
        }
        all_masks_ = all;
        if (!all.empty()) { d_masks_ = to_device(all); }
        if (!b.counts.empty()) {
            d_counts_pristine_ = to_device(b.counts);
            d_counts_          = to_device(b.counts);
        }
        for (int row = 0; row < b.batch; ++row) {
            auto& config = configs_[static_cast<std::size_t>(row)];
            config.mask  = {};
            if (offsets[static_cast<std::size_t>(row)] >= 0) {
                config.mask = {static_cast<const std::uint32_t*>(d_masks_.p) +
                                   offsets[static_cast<std::size_t>(row)],
                               b.words()};
            }
            config.token_counts = b.counts.empty() ? nullptr
                                                   : static_cast<std::int32_t*>(d_counts_.p) +
                                                         static_cast<std::size_t>(row) * b.domain;
        }
        d_configs_ = to_device(configs_);
        d_lengths_ = to_device(b.lengths);
        d_anchors_ = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(b.batch), -1));
        d_licensed_ =
            to_device(std::vector<std::int32_t>(static_cast<std::size_t>(b.k + 1) * b.batch, -9));
        d_count_out_ = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(b.batch), -9));
        d_accepted_  = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(b.batch), -9));
        workspace_bytes_ = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            b.domain, b.k, b.k, b.batch, b.batch);
        workspace_.emplace(std::max<std::size_t>(256, workspace_bytes_));
    }

    // One launch; `seeds` (optional, one per row) replaces the configured seeds.
    MaskedAcceptResult run(const std::vector<unsigned long long>& seeds = {}) {
        if (!seeds.empty()) {
            for (int row = 0; row < b_.batch; ++row) {
                configs_[static_cast<std::size_t>(row)].seed = seeds[static_cast<std::size_t>(row)];
            }
            CUDA_CHECK(cudaMemcpy(d_configs_.p, configs_.data(),
                                  configs_.size() * sizeof(ops::SamplingConfig),
                                  cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_lengths_.p, b_.lengths.data(),
                              b_.lengths.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));
        if (!b_.counts.empty()) {
            CUDA_CHECK(cudaMemcpy(d_counts_.p, d_counts_pristine_.p, d_counts_.bytes,
                                  cudaMemcpyDeviceToDevice));
        }
        const int cols = b_.k + 1;
        Tensor targets(d_targets_.p, DType::I32, {cols, b_.batch});
        Tensor logits(d_logits_.p, DType::BF16, {b_.physical, cols, b_.batch});
        Tensor drafts(d_drafts_.p, DType::I32, {b_.k, b_.batch});
        Tensor extents(d_extents_.p, DType::I32, {b_.batch});
        Tensor lengths(d_lengths_.p, DType::I32, {b_.batch});
        Tensor anchors(d_anchors_.p, DType::I32, {b_.batch});
        Tensor licensed(d_licensed_.p, DType::I32, {cols, b_.batch});
        Tensor counts(d_count_out_.p, DType::I32, {b_.batch});
        Tensor accepted(d_accepted_.p, DType::I32, {b_.batch});
        ops::speculative_accept_greedy_drafts(
            targets, logits, drafts, extents, lengths, anchors, licensed, counts, accepted,
            b_.domain, static_cast<const ops::SamplingConfig*>(d_configs_.p), *workspace_, nullptr);
        cuda_synchronize();
        MaskedAcceptResult out;
        out.licensed =
            from_device<std::int32_t>(d_licensed_, static_cast<std::size_t>(cols) * b_.batch);
        out.counts   = from_device<std::int32_t>(d_count_out_, b_.batch);
        out.accepted = from_device<std::int32_t>(d_accepted_, b_.batch);
        out.lengths  = from_device<std::int32_t>(d_lengths_, b_.batch);
        out.anchors  = from_device<std::int32_t>(d_anchors_, b_.batch);
        return out;
    }

    // Read-only operands must be untouched and the workspace query must match the launch.
    int integrity(const std::string& label) const {
        int failures =
            verify_exact((label + " logits read-only").c_str(),
                         from_device<std::uint16_t>(d_logits_, logits_bits_.size()), logits_bits_);
        failures +=
            verify_exact((label + " targets read-only").c_str(),
                         from_device<std::int32_t>(d_targets_, b_.targets.size()), b_.targets);
        failures += verify_exact((label + " drafts read-only").c_str(),
                                 from_device<std::int32_t>(d_drafts_, b_.drafts.size()), b_.drafts);
        if (!all_masks_.empty()) {
            failures +=
                verify_exact((label + " masks read-only").c_str(),
                             from_device<std::uint32_t>(d_masks_, all_masks_.size()), all_masks_);
        }
        if (workspace_->used() != 0 || workspace_->peak_used() != workspace_bytes_) {
            std::cerr << label << ": workspace query/execution high-water mismatch\n";
            ++failures;
        }
        return failures;
    }

private:
    const MaskedAcceptBatch& b_;
    std::vector<std::uint16_t> logits_bits_;
    std::vector<ops::SamplingConfig> configs_;
    std::vector<std::uint32_t> all_masks_;
    DeviceBuffer d_logits_, d_targets_, d_drafts_, d_extents_, d_masks_, d_counts_pristine_,
        d_counts_, d_configs_, d_lengths_, d_anchors_, d_licensed_, d_count_out_, d_accepted_;
    std::size_t workspace_bytes_ = 0;
    std::optional<WorkspaceArena> workspace_;
};

// FP64 target distribution of verification column `col` of `row`: legal tokens only, committed
// counts plus the round overlay drafts[0..col-1], penalties, top-k (cap 20), min-p, top-p and
// renormalization; a greedy row is the point mass on the penalty-adjusted legal argmax.
std::vector<std::pair<int, double>> masked_column_distribution(const MaskedAcceptBatch& b, int row,
                                                               int col) {
    const auto& config = b.configs[static_cast<std::size_t>(row)];
    std::vector<std::pair<double, int>> ranked;
    for (int token = 0; token < b.domain; ++token) {
        if (!b.legal(row, col, token)) { continue; }
        int count =
            b.counts.empty() ? 0 : b.counts[static_cast<std::size_t>(row) * b.domain + token];
        for (int previous = 0; previous < col; ++previous) {
            if (b.drafts[static_cast<std::size_t>(row) * b.k + previous] == token) { ++count; }
        }
        double value = static_cast<double>(b.logit_at(row, col, token));
        if (count > 0) { value -= static_cast<double>(config.presence_penalty); }
        value -= static_cast<double>(config.frequency_penalty) * static_cast<double>(count);
        ranked.emplace_back(value, token);
    }
    if (ranked.empty()) { throw std::runtime_error("masked oracle: empty mask at a used column"); }
    const auto better = [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
    };
    std::sort(ranked.begin(), ranked.end(), better);
    if (!(config.temperature > 0.0f)) { return {{ranked.front().second, 1.0}}; }
    int cap = kTargetCandidateCap;
    if (config.top_k > 0 && config.top_k < cap) { cap = config.top_k; }
    cap = std::min(cap, static_cast<int>(ranked.size()));
    std::vector<double> weights(static_cast<std::size_t>(cap));
    double total = 0.0;
    for (int j = 0; j < cap; ++j) {
        weights[static_cast<std::size_t>(j)] =
            std::exp((ranked[static_cast<std::size_t>(j)].first - ranked.front().first) /
                     static_cast<double>(config.temperature));
        total += weights[static_cast<std::size_t>(j)];
    }
    const double min_weight = config.min_p > 0.0f ? config.min_p * weights.front() : -1.0;
    double cumulative       = 0.0;
    int support             = 0;
    for (int j = 0; j < cap; ++j) {
        if (min_weight >= 0.0 && weights[static_cast<std::size_t>(j)] < min_weight) { break; }
        cumulative += weights[static_cast<std::size_t>(j)];
        support = j + 1;
        if (config.top_p < 1.0f && cumulative >= config.top_p * total) { break; }
    }
    support     = std::max(support, 1);
    double kept = 0.0;
    for (int j = 0; j < support; ++j) { kept += weights[static_cast<std::size_t>(j)]; }
    std::vector<std::pair<int, double>> out;
    for (int j = 0; j < support; ++j) {
        out.emplace_back(ranked[static_cast<std::size_t>(j)].second,
                         weights[static_cast<std::size_t>(j)] / kept);
    }
    return out;
}

// Exact law of (accepted count A, terminal token) for one row. With one-hot drafts the rejection
// test accepts draft d_i with p_i(d_i) and corrects from p_i without d_i, so
// P(A=j, t) = prod_{i<j} p_i(d_i) * p_j(t) [t != d_j] for j < extent, and the bonus column
// gives P(A=extent, t) = prod_{i<extent} p_i(d_i) * p_extent(t). An illegal draft has p=0.
std::map<std::pair<int, int>, double> masked_accept_law(const MaskedAcceptBatch& b, int row) {
    const int extent = std::clamp(b.extents[static_cast<std::size_t>(row)], 0, b.k);
    std::map<std::pair<int, int>, double> law;
    double prefix = 1.0;
    for (int j = 0; j <= extent && prefix > 0.0; ++j) {
        const auto distribution = masked_column_distribution(b, row, j);
        const int draft = j < extent ? b.drafts[static_cast<std::size_t>(row) * b.k + j] : -1;
        double p_draft  = 0.0;
        for (const auto& [token, p] : distribution) {
            if (token == draft) {
                p_draft = p;
                continue;
            }
            if (p > 0.0) { law[{j, token}] += prefix * p; }
        }
        prefix *= p_draft;
    }
    return law;
}

// Checks the full effect of one row against a point-mass law (deterministic fixtures).
int verify_masked_row(const std::string& label, const MaskedAcceptBatch& b,
                      const MaskedAcceptResult& got, int row) {
    const auto law = masked_accept_law(b, row);
    if (law.size() != 1 || std::abs(law.begin()->second - 1.0) > 1e-12) {
        std::cerr << label << ": fixture row " << row << " is not deterministic\n";
        return 1;
    }
    const auto [accepted_count, terminal] = law.begin()->first;
    const int cols                        = b.k + 1;
    std::vector<std::int32_t> licensed(static_cast<std::size_t>(cols), 0);
    for (int i = 0; i < accepted_count; ++i) {
        licensed[static_cast<std::size_t>(i)] = b.drafts[static_cast<std::size_t>(row) * b.k + i];
    }
    licensed[static_cast<std::size_t>(accepted_count)] = terminal;
    const std::string row_label                        = label + " row " + std::to_string(row);
    int failures =
        verify_exact((row_label + " licensed").c_str(),
                     std::vector<std::int32_t>(
                         got.licensed.begin() + static_cast<std::ptrdiff_t>(row) * cols,
                         got.licensed.begin() + static_cast<std::ptrdiff_t>(row + 1) * cols),
                     licensed);
    failures += verify_exact<std::int32_t>((row_label + " count").c_str(),
                                           {got.counts[static_cast<std::size_t>(row)]},
                                           {accepted_count + 1});
    failures +=
        verify_exact<std::int32_t>((row_label + " accepted").c_str(),
                                   {got.accepted[static_cast<std::size_t>(row)]}, {accepted_count});
    failures += verify_exact<std::int32_t>(
        (row_label + " length").c_str(), {got.lengths[static_cast<std::size_t>(row)]},
        {b.lengths[static_cast<std::size_t>(row)] + accepted_count + 1});
    failures += verify_exact<std::int32_t>(
        (row_label + " anchor").c_str(), {got.anchors[static_cast<std::size_t>(row)]}, {terminal});
    return failures;
}

struct AcceptRoute {
    int domain;
    int physical;
    int k;
};

// Small builder, multi-block (the MTP production shape: 248077 public ids, K=3), fast fallback.
constexpr AcceptRoute kAcceptRoutes[] = {{64, 64, 3},         {64, 66, 5},         {257, 257, 3},
                                         {248077, 248320, 3}, {248077, 248320, 5}, {257, 259, 16}};

std::string accept_label(const char* what, const AcceptRoute& route, float temperature) {
    return std::string(what) + " domain=" + std::to_string(route.domain) +
           " K=" + std::to_string(route.k) + " T=" + std::to_string(temperature);
}

// Upstream 08902f73 grammar_onehot_case: one legal token per column, the legal drafts of
// columns 0 and 1 accepted, the draft of column 2 illegal, so column 2's only legal token is the
// correction, whatever its logit (-2 against a 12.0 illegal plateau).
int masked_onehot_case(const AcceptRoute& route, float temperature) {
    MaskedAcceptBatch b(route.domain, route.physical, route.k, 1);
    std::fill(b.logits.begin(), b.logits.end(), 12.0f);
    for (int col = 0; col <= route.k; ++col) {
        const int allowed = col < 2 ? 11 + col : 51;
        if (col < route.k) { b.drafts[static_cast<std::size_t>(col)] = 11 + col; }
        b.set_legal(0, col, {allowed});
        b.logit(0, col, allowed) = -2.0f;
    }
    b.configs[0].temperature = temperature;
    b.configs[0].seed        = 73;
    b.lengths[0]             = 73;
    MaskedAcceptRunner runner(b);
    const std::string label = accept_label("masked one-hot", route, temperature);
    const auto result       = runner.run();
    return verify_masked_row(label, b, result, 0) + runner.integrity(label);
}

// An illegal draft is rejected at every position j (the oracle law is a point mass because each
// column has one legal token), and with every draft legal the bonus comes from column K's mask.
// Columns after the rejection carry the safe filler {0}; masks of columns past the extent are
// left empty and must not be read. Greedy rows also get raw targets that are illegal and would
// be accepted by the unconstrained fast path.
int masked_reject_every_position_case(const AcceptRoute& route, float temperature) {
    int failures = 0;
    for (int reject = 0; reject <= route.k; ++reject) {
        for (const bool short_extent : {false, true}) {
            if (short_extent && reject + 1 >= route.k) { continue; }
            MaskedAcceptBatch b(route.domain, route.physical, route.k, 1);
            const int extent         = short_extent ? reject + 1 : route.k;
            b.extents[0]             = extent;
            b.lengths[0]             = 1000 + reject;
            b.configs[0].temperature = temperature;
            b.configs[0].top_k       = 20;
            b.configs[0].seed        = 4242 + reject;
            for (int col = 0; col <= route.k; ++col) {
                const int draft = 20 + 3 * col; // distinct ids below 64
                if (col < route.k) { b.drafts[static_cast<std::size_t>(col)] = draft; }
                b.targets[static_cast<std::size_t>(col)] = draft; // raw argmax = the draft
                b.logit(0, col, draft)                   = 15.0f;
                b.logit(0, col, draft + 1)               = -5.0f;
                if (col > extent) { continue; } // never consumed: mask stays all-zero
                if (col < reject) {
                    b.set_legal(0, col, {draft});
                } else if (col == reject) {
                    b.set_legal(0, col, {draft + 1}); // the draft (if any) is illegal here
                } else {
                    b.set_legal(0, col, {0}); // safe filler after the rejection
                }
            }
            MaskedAcceptRunner runner(b);
            const std::string label = accept_label("masked reject", route, temperature) +
                                      " at=" + std::to_string(reject) +
                                      " extent=" + std::to_string(extent);
            const auto result = runner.run();
            failures += verify_masked_row(label, b, result, 0) + runner.integrity(label);
        }
    }
    return failures;
}

// Rows 0 and 2 masked, rows 1 and 3 not, greedy and stochastic, with penalties: unmasked rows
// equal an all-unmasked launch exactly, masked rows equal their solo launch.
int masked_mixed_batch_case(const AcceptRoute& route) {
    constexpr int batch = 4;
    MaskedAcceptBatch b(route.domain, route.physical, route.k, batch);
    b.counts.assign(static_cast<std::size_t>(route.domain) * batch, 0);
    for (int row = 0; row < batch; ++row) {
        auto& config                             = b.configs[static_cast<std::size_t>(row)];
        config.temperature                       = row < 2 ? 0.9f : 0.0f;
        config.top_k                             = 8;
        config.presence_penalty                  = row == 1 || row == 2 ? 0.5f : 0.0f;
        config.seed                              = 900 + row;
        b.lengths[static_cast<std::size_t>(row)] = 3000 + 17 * row;
        for (int col = 0; col <= route.k; ++col) {
            for (int token = 0; token < 40; ++token) {
                b.logit(row, col, token) =
                    static_cast<float>((token * 7 + col * 5 + row * 3) % 16) * 0.25f;
            }
            if (col < route.k) {
                const int draft                                         = (col * 11 + row) % 40;
                b.drafts[static_cast<std::size_t>(row) * route.k + col] = draft;
                b.targets[static_cast<std::size_t>(row) * (route.k + 1) + col] = draft;
            }
        }
        b.counts[static_cast<std::size_t>(row) * route.domain + 7] = 2;
    }
    for (int row : {0, 2}) {
        for (int col = 0; col <= route.k; ++col) {
            std::vector<int> legal;
            for (int token = col % 2; token < 40; token += 2) { legal.push_back(token); }
            b.set_legal(row, col, legal);
        }
    }
    MaskedAcceptBatch plain = b;
    for (auto& mask : plain.masks) { mask.clear(); }
    MaskedAcceptRunner mixed_runner(b), plain_runner(plain);
    const auto mixed        = mixed_runner.run();
    const auto unmask       = plain_runner.run();
    const std::string label = "masked mixed batch domain=" + std::to_string(route.domain) +
                              " K=" + std::to_string(route.k);
    int failures         = mixed_runner.integrity(label) + plain_runner.integrity(label + " plain");
    const int cols       = route.k + 1;
    const auto row_slice = [&](const std::vector<std::int32_t>& v, int row) {
        return std::vector<std::int32_t>(v.begin() + static_cast<std::ptrdiff_t>(row) * cols,
                                         v.begin() + static_cast<std::ptrdiff_t>(row + 1) * cols);
    };
    for (int row : {1, 3}) {
        const std::string row_label = label + " unmasked row " + std::to_string(row);
        failures += verify_exact((row_label + " licensed").c_str(), row_slice(mixed.licensed, row),
                                 row_slice(unmask.licensed, row));
        failures += verify_exact<std::int32_t>((row_label + " accepted").c_str(),
                                               {mixed.accepted[static_cast<std::size_t>(row)]},
                                               {unmask.accepted[static_cast<std::size_t>(row)]});
    }
    for (int row : {0, 2}) {
        MaskedAcceptBatch solo(route.domain, route.physical, route.k, 1);
        const auto r = static_cast<std::size_t>(row);
        std::copy_n(b.logits.begin() + static_cast<std::ptrdiff_t>(row) * cols * route.physical,
                    static_cast<std::size_t>(cols) * route.physical, solo.logits.begin());
        std::copy_n(b.targets.begin() + static_cast<std::ptrdiff_t>(row) * cols, cols,
                    solo.targets.begin());
        std::copy_n(b.drafts.begin() + static_cast<std::ptrdiff_t>(row) * route.k, route.k,
                    solo.drafts.begin());
        solo.extents[0] = b.extents[r];
        solo.lengths[0] = b.lengths[r];
        solo.configs[0] = b.configs[r];
        solo.masks[0]   = b.masks[r];
        solo.counts.assign(b.counts.begin() + static_cast<std::ptrdiff_t>(row) * route.domain,
                           b.counts.begin() + static_cast<std::ptrdiff_t>(row + 1) * route.domain);
        MaskedAcceptRunner solo_runner(solo);
        const auto alone            = solo_runner.run();
        const std::string row_label = label + " masked row " + std::to_string(row);
        failures += verify_exact((row_label + " licensed").c_str(), row_slice(mixed.licensed, row),
                                 alone.licensed);
        const int accepted = mixed.accepted[r];
        for (int i = 0; i <= accepted; ++i) {
            const int token = mixed.licensed[r * cols + static_cast<std::size_t>(i)];
            if (!b.legal(row, i, token)) {
                std::cerr << row_label << ": illegal token " << token << " at column " << i << '\n';
                ++failures;
            }
        }
        if (b.configs[r].temperature == 0.0f) {
            failures += verify_masked_row(label, b, mixed, row);
        }
    }
    return failures;
}

// Mask + penalties (committed counts and the round overlay) + top-k/top-p/min-p against the FP64
// law of (A, terminal) over many seeds. Two draft patterns: A has every draft legal (bonus
// reachable), B has an illegal draft at column 2.
int masked_accept_distribution_case(const AcceptRoute& route, bool illegal_draft) {
    constexpr int batch  = 8;
    constexpr int ranked = 12;
    MaskedAcceptBatch b(route.domain, route.physical, route.k, batch);
    b.counts.assign(static_cast<std::size_t>(route.domain) * batch, 0);
    const int spacing   = std::max(1, (route.domain - 4) / ranked);
    const auto token_of = [&](int j) { return 2 + j * spacing; };
    for (int row = 0; row < batch; ++row) {
        auto& config                             = b.configs[static_cast<std::size_t>(row)];
        config.temperature                       = 0.8f;
        config.top_k                             = 6;
        config.top_p                             = 0.97f;
        config.min_p                             = 0.01f;
        config.presence_penalty                  = 0.5f;
        config.frequency_penalty                 = 0.25f;
        b.lengths[static_cast<std::size_t>(row)] = 777;
        b.counts[static_cast<std::size_t>(row) * route.domain + token_of(2)] = 1;
        for (int col = 0; col <= route.k; ++col) {
            std::vector<int> legal;
            for (int j = 0; j < ranked; ++j) {
                const int rank                 = (j + col) % ranked;
                b.logit(row, col, token_of(j)) = 3.0f - 0.25f * static_cast<float>(rank);
                if (rank % 3 != 0) { legal.push_back(token_of(j)); }
            }
            for (int token = 1; token < route.domain; token += 5) {
                if ((token - 2) % spacing != 0 || token > token_of(ranked - 1)) {
                    legal.push_back(token);
                }
            }
            b.set_legal(row, col, legal);
            if (col < route.k) {
                // Pattern A: the best legal token of the column (rank 1). Pattern B: the
                // second-best legal at column 1 and an illegal rank-0 token at column 2.
                int rank = 1;
                if (illegal_draft && col == 1) { rank = 2; }
                if (illegal_draft && col == 2) { rank = 0; }
                const int j = ((rank - col) % ranked + ranked) % ranked;
                b.drafts[static_cast<std::size_t>(row) * route.k + col] = token_of(j);
            }
        }
    }
    std::map<std::pair<int, int>, double> law = masked_accept_law(b, 0);
    MaskedAcceptRunner runner(b);
    std::map<std::pair<int, int>, int> observed;
    const int launches      = 4096 / batch;
    int failures            = 0;
    const std::string label = std::string("masked accept law ") + (illegal_draft ? "B" : "A") +
                              " domain=" + std::to_string(route.domain) +
                              " K=" + std::to_string(route.k);
    for (int launch = 0; launch < launches; ++launch) {
        std::vector<unsigned long long> seeds(batch);
        for (int row = 0; row < batch; ++row) {
            seeds[static_cast<std::size_t>(row)] = 0x51ed0000ull + launch * batch + row;
        }
        const auto result = runner.run(seeds);
        for (int row = 0; row < batch; ++row) {
            const auto r = static_cast<std::size_t>(row);
            const int a  = result.accepted[r];
            const int t  = result.licensed[r * (route.k + 1) + static_cast<std::size_t>(a)];
            if (result.anchors[r] != t || result.counts[r] != a + 1 ||
                result.lengths[r] != b.lengths[r] + a + 1) {
                std::cerr << label << ": inconsistent row state\n";
                ++failures;
            }
            ++observed[{a, t}];
        }
    }
    failures += runner.integrity(label);
    const double n = static_cast<double>(launches * batch);
    double max_z   = 0.0;
    for (const auto& [outcome, count] : observed) {
        if (law.find(outcome) == law.end()) {
            std::cerr << label << ": outcome A=" << outcome.first << " t=" << outcome.second
                      << " has zero probability\n";
            ++failures;
        }
    }
    for (const auto& [outcome, probability] : law) {
        const auto it          = observed.find(outcome);
        const double frequency = it == observed.end() ? 0.0 : it->second / n;
        const double sigma     = std::sqrt(probability * (1.0 - probability) / n);
        const double gap       = std::abs(frequency - probability);
        if (gap > 7.0 * sigma + 2.0 / n) {
            std::cerr << label << ": A=" << outcome.first << " t=" << outcome.second
                      << " frequency=" << frequency << " law=" << probability << '\n';
            ++failures;
        }
        if (sigma > 0.0) { max_z = std::max(max_z, gap / sigma); }
    }
    if (!illegal_draft) {
        bool bonus = false;
        for (const auto& [outcome, probability] : law) bonus |= outcome.first == route.k;
        if (!bonus) {
            std::cerr << label << ": pattern A cannot reach the bonus column\n";
            ++failures;
        }
    } else {
        for (const auto& [outcome, probability] : law) {
            if (outcome.first > 2) {
                std::cerr << label << ": accepted past the illegal draft\n";
                ++failures;
            }
        }
    }
    std::cout << "    " << label << " FP64 law match over " << law.size()
              << " outcomes (max z=" << max_z << ")\n";
    return failures;
}

// Empty mask at column 0 (always consumed) must trap, in a child process (the trap poisons
// the CUDA context).
int run_accept_trap_case(std::string_view name) {
    AcceptRoute route{64, 64, 3};
    float temperature = 0.0f;
    if (name == "accept-greedy-small") {
    } else if (name == "accept-stochastic-small") {
        temperature = 1.0f;
    } else if (name == "accept-greedy-multiblock") {
        route = {248077, 248320, 3};
    } else if (name == "accept-stochastic-multiblock") {
        route       = {248077, 248320, 3};
        temperature = 1.0f;
    } else if (name == "accept-greedy-fast") {
        route = {257, 257, 16};
    } else if (name == "accept-stochastic-fast") {
        route       = {257, 257, 16};
        temperature = 1.0f;
    } else {
        std::cerr << "unknown trap case " << name << '\n';
        return 2;
    }
    MaskedAcceptBatch b(route.domain, route.physical, route.k, 2);
    for (int row = 0; row < 2; ++row) {
        b.configs[static_cast<std::size_t>(row)].temperature = temperature;
    }
    for (int col = 0; col <= route.k; ++col) b.set_legal(1, col, {});
    MaskedAcceptRunner runner(b);
    try {
        (void)runner.run();
    } catch (const std::exception& error) {
        std::cout << "trap case " << name << ": " << error.what() << std::endl;
        // Skip the destructors: freeing on a trapped context only prints cleanup errors.
        std::_Exit(0);
    }
    std::cout << "trap case " << name << ": no error\n";
    return 3;
}

// Absolute path of this test binary (for re-running it as a child: inside std::system the
// /proc/self/exe link names the shell).
std::string self_executable() {
    char path[4096];
    const ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0) { throw std::runtime_error("cannot resolve /proc/self/exe"); }
    path[n] = '\0';
    return path;
}

int masked_accept_suite() {
    int failures = 0;
    for (const AcceptRoute& route : kAcceptRoutes) {
        for (const float temperature : {0.0f, 1.0f}) {
            failures += masked_onehot_case(route, temperature);
            failures += masked_reject_every_position_case(route, temperature);
        }
        failures += masked_mixed_batch_case(route);
    }
    for (const AcceptRoute& route : {AcceptRoute{64, 64, 3}, AcceptRoute{257, 257, 3},
                                     AcceptRoute{248077, 248320, 3}, AcceptRoute{257, 257, 16}}) {
        failures += masked_accept_distribution_case(route, false);
        failures += masked_accept_distribution_case(route, true);
    }
    for (const char* name :
         {"accept-greedy-small", "accept-stochastic-small", "accept-greedy-multiblock",
          "accept-stochastic-multiblock", "accept-greedy-fast", "accept-stochastic-fast"}) {
        const std::string command = "\"" + self_executable() + "\" --trap-case " + name;
        const int status          = std::system(command.c_str());
        if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            std::cerr << "empty mask " << name << " did not trap (status " << status << ")\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "speculative_round: SKIP (CUDA unavailable)\n";
        return 77;
    }

    if (argc == 3 && std::string_view(argv[1]) == "--trap-case") {
        return run_accept_trap_case(argv[2]);
    }
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--transforms-only")) {
        std::cerr
            << "usage: ninfer_speculative_round_test [--transforms-only | --trap-case NAME]\n";
        return 2;
    }
    int failures = transforms_conformance();
    if (argc == 2) {
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " speculative transforms\n";
        return failures == 0 ? 0 : 1;
    }
    const std::size_t k15 =
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 1);
    if (k15 == 0 || k15 != ops::sampling_workspace_capacity_bytes(257, 16, 16) ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 16, 16, 1, 1) != 0 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 1, 16, 1, 1) != k15 ||
        ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 15, 15, 1, 2) !=
            2 * k15) {
        std::cerr << "speculative accept workspace did not close over K+1 sampling columns\n";
        ++failures;
    }
    try {
        (void)ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(257, 0, 15, 1, 1);
        std::cerr << "speculative accept workspace accepted an invalid draft interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += greedy_accept_case(1, 0);
    failures += greedy_accept_case(5, 2);
    failures += greedy_accept_case(5, 5);
    failures += greedy_accept_case(15, 7, 257);
    failures += greedy_penalty_case(64);
    failures += greedy_penalty_case(257);
    failures += deterministic_sampling_case();
    failures += batched_sampling_workspace_stride_case();
    std::size_t sparse_peak = 0;
    for (int k = 1; k <= 15; ++k) {
        for (int batch = 1; batch <= 8; ++batch) {
            SparseAcceptSuite suite(k, batch);
            failures += suite.sparse_greedy_direct_case();
            failures += suite.generated_general_case();
            sparse_peak = std::max(sparse_peak, suite.observed_workspace);
            if (batch == 1)
                for (int pattern = 1; pattern < 7; ++pattern)
                    failures += suite.sparse_greedy_direct_case(pattern);
        }
    }
    if (ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
            kSparseTokenDomain, {false}, 1, 15, 1, 8) != sparse_peak ||
        ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(kSparseTokenDomain, {true},
                                                                       1, 15, 1, 8) != 0) {
        std::cerr << "sparse interval workspace does not cover the observed domain\n";
        ++failures;
    }
    for (bool raw : {false, true}) {
        try {
            (void)ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                kSparseTokenDomain, {raw}, 1, 16, 1, 8);
            std::cerr << "sparse query admitted K=16\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    for (int k : {1, 7, 15}) {
        SparseAcceptSuite suite(k, 8);
        failures += suite.sparse_general_mixed_case();
        failures += SparseAcceptSuite(k, 8).grammar_mask_case();
        failures += suite.sparse_greedy_direct_case(0, true);
        failures += SparseAcceptSuite(k, 1).repeated_history_case(false);
        failures += SparseAcceptSuite(k, 1).repeated_history_case(true);
    }
    failures += masked_accept_suite();

    if (failures != 0) {
        std::cerr << "speculative_round failures=" << failures << '\n';
        return 1;
    }
    std::cout << "speculative_round: PASS\n";
    return 0;
}
