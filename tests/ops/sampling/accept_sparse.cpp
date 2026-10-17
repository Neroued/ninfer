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

constexpr int kSparsePhysicalRows = 248320;
constexpr int kSparseTokenDomain  = 248077;
constexpr int kSparseCandidates   = 16;
constexpr int kTargetCandidateCap = 20;

struct SparseAcceptSuite {
    const int kSparseDrafts;
    const int kSparseColumns;
    const int kSparseBatch;
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
                !(masks[(row * kSparseColumns + column) * mask_words + token / 32] &
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
                static_cast<std::int32_t*>(d_token_counts.data()) +
                static_cast<std::size_t>(row) * kSparseTokenDomain;
        }
        std::optional<DeviceBuffer> d_masks;
        if (!masks.empty()) {
            d_masks.emplace(to_device(masks));
            for (int row = 0; row < kSparseBatch; ++row) {
                device_configs[row].mask = {static_cast<const std::uint32_t*>(d_masks->data()) +
                                                row * kSparseColumns * mask_words,
                                            mask_words};
            }
        }
        DeviceBuffer d_configs = to_device(device_configs);

        GuardedDeviceBuffer d_lengths(kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_anchors(kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_licensed(kSparseColumns * kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_licensed_counts(kSparseBatch * sizeof(std::int32_t));
        GuardedDeviceBuffer d_accepted(kSparseBatch * sizeof(std::int32_t));
        d_lengths.copy_from_host(initial_lengths.data(),
                                 initial_lengths.size() * sizeof(initial_lengths[0]));
        d_anchors.copy_from_host(initial_anchors.data(),
                                 initial_anchors.size() * sizeof(initial_anchors[0]));
        d_licensed.fill(0xcd);
        d_licensed_counts.fill(0xef);
        d_accepted.fill(0xab);

        Tensor target_tensor(d_target_tokens.span(), DType::I32, {kSparseColumns, kSparseBatch});
        Tensor logits_tensor(d_logits.span(), DType::BF16,
                             {kSparsePhysicalRows, kSparseColumns, kSparseBatch});
        Tensor drafts_tensor(d_drafts.span(), DType::I32, {kSparseDrafts, kSparseBatch});
        Tensor candidate_tensor(d_candidate_ids.span(), DType::I32,
                                {kSparseCandidates, kSparseDrafts, kSparseBatch});
        Tensor q_tensor(d_proposal_q.span(), DType::FP32,
                        {kSparseCandidates, kSparseDrafts, kSparseBatch});
        Tensor extent_tensor(d_extents.span(), DType::I32, {kSparseBatch});
        Tensor lengths_tensor(d_lengths.span(), DType::I32, {kSparseBatch});
        Tensor anchors_tensor(d_anchors.span(), DType::I32, {kSparseBatch});
        Tensor licensed_tensor(d_licensed.span(), DType::I32, {kSparseColumns, kSparseBatch});
        Tensor licensed_counts_tensor(d_licensed_counts.span(), DType::I32, {kSparseBatch});
        Tensor accepted_tensor(d_accepted.span(), DType::I32, {kSparseBatch});
        const std::size_t workspace_bytes =
            ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                kSparseTokenDomain, envelope, kSparseDrafts, kSparseDrafts, kSparseBatch,
                kSparseBatch);
        GuardedDeviceBuffer scratch(std::max<std::size_t>(workspace_bytes, 1));
        MemoryArena workspace(scratch.span());
        const auto launch = [&](CudaExecutionView cuda) {
            ops::speculative_accept_sparse_drafts(
                target_tensor, logits_tensor, drafts_tensor, candidate_tensor, q_tensor,
                extent_tensor, lengths_tensor, anchors_tensor, licensed_tensor,
                licensed_counts_tensor, accepted_tensor, kSparseTokenDomain,
                static_cast<const ops::SamplingConfig*>(d_configs.data()), envelope, workspace,
                cuda);
        };
        launch(ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();

        const auto check_result = [&](const SparseExpected& wanted, const std::string& phase) {
            int checks = verify_exact(
                (label + phase + " licensed tokens").c_str(),
                from_device<std::int32_t>(d_licensed.data(), wanted.licensed_tokens.size()),
                wanted.licensed_tokens);
            checks +=
                verify_exact((label + phase + " licensed counts").c_str(),
                             from_device<std::int32_t>(d_licensed_counts.data(), kSparseBatch),
                             wanted.licensed_counts);
            checks += verify_exact((label + phase + " accepted drafts").c_str(),
                                   from_device<std::int32_t>(d_accepted.data(), kSparseBatch),
                                   wanted.accepted);
            checks += verify_exact((label + phase + " provisional lengths").c_str(),
                                   from_device<std::int32_t>(d_lengths.data(), kSparseBatch),
                                   wanted.lengths);
            checks += verify_exact((label + phase + " provisional anchors").c_str(),
                                   from_device<std::int32_t>(d_anchors.data(), kSparseBatch),
                                   wanted.anchors);
            return checks;
        };
        int failures = check_result(expected, " eager");
        if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) ++failures;
        failures += scratch.verify_guards(label + " workspace");
        if (kSparseBatch == 1 || kSparseBatch == 8) {
            cudaStream_t stream = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "sparse graph stream");
            const auto stream_execution =
                ninfer::test::cuda_environment().context.execution_view(stream);
            {
                DecodeGraphDefinition definition;
                DecodeGraphExecutable executable;
                definition.capture(stream, [&] { launch(stream_execution); });
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
                    d_lengths.copy_from_host(next_lengths.data(),
                                             next_lengths.size() * sizeof(next_lengths[0]));
                    d_anchors.copy_from_host(initial_anchors.data(),
                                             initial_anchors.size() * sizeof(initial_anchors[0]));
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
            "sparse configs readonly", from_device<std::uint8_t>(d_configs, d_configs.size()),
            std::vector<std::uint8_t>(config_bytes, config_bytes + d_configs.size()));
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
        std::vector<int> history(kSparseTokenDomain * kSparseBatch, 0);
        masks.assign(kSparseColumns * kSparseBatch * mask_words, 0);
        const int a = kSparseTokenDomain - 2, b = kSparseTokenDomain - 1, x = 42;
        for (int row = 0; row < kSparseBatch; ++row) {
            configs[row].temperature = row < kSparseBatch / 2 ? 0.0f : 1.0f;
            configs[row].seed        = 8123 + row;
            for (int col = 0; col < kSparseColumns; ++col) {
                for (int token : {a, b}) {
                    masks[(row * kSparseColumns + col) * mask_words + token / 32] |=
                        1u << (token % 32);
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
        // p(a)=p(b)=.5, q(a)=q(x)=.5: a accepts, invalid x corrects to b with probability 1.
        return execute_sparse_accept_case("sparse grammar mask", targets, logits, drafts, ids, q,
                                          extents, lengths, anchors, configs, history, {false});
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

int capacity_contract() {
    std::size_t peak = 0;
    for (int k = 1; k <= 15; ++k)
        for (int batch = 1; batch <= 8; ++batch)
            peak = std::max(peak, ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                                      kSparseTokenDomain, {false}, k, k, batch, batch));
    int failures = ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                       kSparseTokenDomain, {false}, 1, 15, 1, 8) != peak ||
                   ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                       kSparseTokenDomain, {true}, 1, 15, 1, 8) != 0;
    for (bool raw : {false, true}) {
        try {
            (void)ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                kSparseTokenDomain, {raw}, 1, 16, 1, 8);
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    return failures;
}
} // namespace

void add_sparse_accept_cases(ninfer::test::OpSuite& suite) {
    suite.add("accept_sparse/capacity", capacity_contract);
    for (int k = 1; k <= 15; ++k)
        for (int batch = 1; batch <= 8; ++batch) {
            const auto id = "accept_sparse/k" + std::to_string(k) + "/b" + std::to_string(batch);
            suite.add(id + "/direct",
                      [=] { return SparseAcceptSuite(k, batch).sparse_greedy_direct_case(); },
                      {{"execution", batch == 1 || batch == 8 ? "graph" : "eager"}});
            suite.add(id + "/general",
                      [=] { return SparseAcceptSuite(k, batch).generated_general_case(); },
                      {{"execution", batch == 1 || batch == 8 ? "graph" : "eager"}});
            if (batch == 1)
                for (int pattern = 1; pattern < 7; ++pattern)
                    suite.add(id + "/pattern" + std::to_string(pattern),
                              [=] {
                                  return SparseAcceptSuite(k, batch).sparse_greedy_direct_case(
                                      pattern);
                              },
                              {{"execution", "graph"}});
        }
    for (int k : {1, 7, 15}) {
        const auto id = "accept_sparse/k" + std::to_string(k);
        suite.add(id + "/mixed",
                  [=] { return SparseAcceptSuite(k, 8).sparse_general_mixed_case(); },
                  {{"execution", "graph"}});
        suite.add(id + "/grammar", [=] { return SparseAcceptSuite(k, 8).grammar_mask_case(); },
                  {{"execution", "graph"}});
        suite.add(id + "/greedy/general_envelope",
                  [=] { return SparseAcceptSuite(k, 8).sparse_greedy_direct_case(0, true); },
                  {{"execution", "graph"}});
        for (bool stochastic : {false, true})
            suite.add(id + (stochastic ? "/history_stochastic" : "/history_greedy"),
                      [=] { return SparseAcceptSuite(k, 1).repeated_history_case(stochastic); },
                      {{"execution", "graph"}});
    }
}
