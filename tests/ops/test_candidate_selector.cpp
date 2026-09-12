#include "ninfer/ops/candidate_selector.h"

#include "ops/op_tester.h"
#include "core/decode_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kCandidates   = 16;
int kSteps                           = 7;
constexpr std::int32_t kRank         = 256;
constexpr std::int32_t kCodebookRows = 248320;
constexpr std::int32_t kTokenDomain  = 248077;
constexpr std::int32_t kMaxBatch     = 8;

constexpr PointwiseCriterion kProbabilityCriterion{/*absolute=*/2.0e-6,
                                                   /*relative=*/2.0e-5};

std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float represented_pattern(std::uint32_t first, std::uint32_t second, std::uint32_t seed,
                          float scale) {
    const std::uint32_t mixed = mix32(first * 0x9e3779b9U ^ second * 0x85ebca6bU ^ seed);
    int centered              = static_cast<int>((mixed >> 8) & 0xffU) - 128;
    if (centered == 0) { centered = ((first + second) & 1U) == 0 ? 1 : -1; }
    return bf16_to_f32(f32_to_bf16(static_cast<float>(centered) * scale));
}

float predecessor_value(std::int32_t token, std::int32_t rank) {
    return represented_pattern(static_cast<std::uint32_t>(token), static_cast<std::uint32_t>(rank),
                               101U, 1.0F / 256.0F);
}

float successor_value(std::int32_t token, std::int32_t rank) {
    return represented_pattern(static_cast<std::uint32_t>(token), static_cast<std::uint32_t>(rank),
                               211U, 1.0F / 256.0F);
}

std::size_t candidate_offset(std::int32_t batch, std::int32_t step, std::int32_t candidate) {
    return (static_cast<std::size_t>(batch) * kSteps + step) * kCandidates + candidate;
}

std::size_t lattice_offset(std::int32_t batch, std::int32_t step, std::int32_t predecessor_rank,
                           std::int32_t candidate) {
    return ((static_cast<std::size_t>(batch) * kSteps + step) * kCandidates + predecessor_rank) *
               kCandidates +
           candidate;
}

std::vector<std::int32_t> make_candidate_ids() {
    std::vector<std::int32_t> result(static_cast<std::size_t>(kMaxBatch) * kSteps * kCandidates);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        for (std::int32_t step = 0; step < kSteps; ++step) {
            for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                const std::int32_t token = 1000 + batch * 20000 + step * 257 + candidate;
                if (token >= kTokenDomain) { throw std::logic_error("candidate fixture overflow"); }
                result[candidate_offset(batch, step, candidate)] = token;
            }
        }
    }
    return result;
}

std::vector<float> make_unary_scores(std::uint32_t salt = 0) {
    std::vector<float> result(static_cast<std::size_t>(kMaxBatch) * kSteps * kCandidates);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        for (std::int32_t step = 0; step < kSteps; ++step) {
            for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                const std::uint32_t mixed = mix32(static_cast<std::uint32_t>(
                    (batch * kSteps + step) * kCandidates + candidate + 307 + salt * 7919));
                const float variation =
                    static_cast<float>(static_cast<int>((mixed >> 10) & 0xffU) - 128) / 512.0F;
                result[candidate_offset(batch, step, candidate)] =
                    variation + static_cast<float>(kCandidates - candidate) * 0.0078125F;
            }
        }
    }
    return result;
}

std::vector<std::uint16_t> make_projected_hidden() {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(kMaxBatch) * kSteps * kRank);
    for (std::int32_t column = 0; column < kMaxBatch * kSteps; ++column) {
        for (std::int32_t rank = 0; rank < kRank; ++rank) {
            result[static_cast<std::size_t>(column) * kRank + rank] = f32_to_bf16(
                represented_pattern(static_cast<std::uint32_t>(column),
                                    static_cast<std::uint32_t>(rank), 401U, 1.0F / 512.0F));
        }
    }
    return result;
}

std::vector<std::int32_t> make_anchors() {
    std::vector<std::int32_t> result(kMaxBatch);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        result[batch] = 220000 + batch * 101;
    }
    return result;
}

std::vector<std::int32_t> make_base_positions() {
    std::vector<std::int32_t> result(kMaxBatch);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) { result[batch] = 1000 + batch * 97; }
    return result;
}

std::vector<std::int32_t> accessed_tokens(const std::vector<std::int32_t>& candidate_ids,
                                          const std::vector<std::int32_t>& anchors) {
    std::set<std::int32_t> tokens(anchors.begin(), anchors.end());
    tokens.insert(candidate_ids.begin(), candidate_ids.end());
    return {tokens.begin(), tokens.end()};
}

void populate_codebook(DeviceBuffer& device, std::span<const std::int32_t> tokens,
                       bool predecessor) {
    std::vector<std::uint16_t> row(kRank);
    for (const std::int32_t token : tokens) {
        for (std::int32_t rank = 0; rank < kRank; ++rank) {
            const float value =
                predecessor ? predecessor_value(token, rank) : successor_value(token, rank);
            row[rank] = f32_to_bf16(value);
        }
        device.copy_from_host(row.data(), row.size() * sizeof(std::uint16_t),
                              static_cast<std::size_t>(token) * kRank * sizeof(std::uint16_t));
    }
}

std::vector<double> build_lattice(const std::vector<std::int32_t>& candidate_ids,
                                  const std::vector<float>& unary_scores,
                                  const std::vector<std::uint16_t>& projected_hidden,
                                  const std::vector<std::int32_t>& anchors) {
    std::vector<double> lattice(static_cast<std::size_t>(kMaxBatch) * kSteps * kCandidates *
                                kCandidates);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        for (std::int32_t step = 0; step < kSteps; ++step) {
            for (std::int32_t predecessor_rank = 0; predecessor_rank < kCandidates;
                 ++predecessor_rank) {
                const std::int32_t predecessor =
                    step == 0 ? anchors[batch]
                              : candidate_ids[candidate_offset(batch, step - 1, predecessor_rank)];
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    const std::int32_t successor =
                        candidate_ids[candidate_offset(batch, step, candidate)];
                    double edge = unary_scores[candidate_offset(batch, step, candidate)];
                    const std::size_t hidden_base =
                        static_cast<std::size_t>(batch * kSteps + step) * kRank;
                    for (std::int32_t rank = 0; rank < kRank; ++rank) {
                        edge +=
                            static_cast<double>(predecessor_value(predecessor, rank)) *
                            static_cast<double>(bf16_to_f32(projected_hidden[hidden_base + rank])) *
                            static_cast<double>(successor_value(successor, rank));
                    }
                    lattice[lattice_offset(batch, step, predecessor_rank, candidate)] = edge;
                }
            }
        }
    }
    return lattice;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

float oracle_uniform(std::uint64_t seed, std::int32_t position) {
    std::uint64_t key = seed;
    key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                            0xD1B54A32D192ED03ULL));
    key = splitmix64(key ^ (static_cast<std::uint64_t>(ops::kSamplePurposeDFlash2Proposal) << 21));
    const std::uint32_t bits = static_cast<std::uint32_t>(key >> 40);
    return static_cast<float>(bits) * (1.0F / 16777216.0F);
}

struct WalkResult {
    std::vector<std::int32_t> drafts;
    std::vector<double> probabilities;
    std::array<double, kMaxBatch> minimum_decision_margin;
};

WalkResult walk_oracle(const std::vector<std::int32_t>& candidate_ids,
                       const std::vector<double>& lattice,
                       const std::vector<ops::SamplingConfig>& configs,
                       const std::vector<std::int32_t>& base_positions) {
    WalkResult result;
    result.drafts.resize(static_cast<std::size_t>(kMaxBatch) * kSteps);
    result.probabilities.resize(static_cast<std::size_t>(kMaxBatch) * kSteps * kCandidates);
    result.minimum_decision_margin.fill(std::numeric_limits<double>::infinity());
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        std::int32_t predecessor_rank = 0;
        for (std::int32_t step = 0; step < kSteps; ++step) {
            const std::size_t edge_base = lattice_offset(batch, step, predecessor_rank, 0);
            std::int32_t selected       = 0;
            if (configs[batch].temperature > 0.0F) {
                double maximum = lattice[edge_base];
                for (std::int32_t candidate = 1; candidate < kCandidates; ++candidate) {
                    maximum = std::max(maximum, lattice[edge_base + candidate]);
                }
                double sum = 0.0;
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    const double probability = std::exp((lattice[edge_base + candidate] - maximum) /
                                                        configs[batch].temperature);
                    result.probabilities[candidate_offset(batch, step, candidate)] = probability;
                    sum += probability;
                }
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    result.probabilities[candidate_offset(batch, step, candidate)] /= sum;
                }
                const double uniform =
                    oracle_uniform(configs[batch].seed, base_positions[batch] + step);
                double lower      = 0.0;
                double cumulative = 0.0;
                selected          = kCandidates - 1;
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    cumulative += result.probabilities[candidate_offset(batch, step, candidate)];
                    if (uniform < cumulative) {
                        selected = candidate;
                        result.minimum_decision_margin[batch] =
                            std::min(result.minimum_decision_margin[batch],
                                     std::min(uniform - lower, cumulative - uniform));
                        break;
                    }
                    lower = cumulative;
                }
            } else {
                double best   = lattice[edge_base];
                double second = -std::numeric_limits<double>::infinity();
                for (std::int32_t candidate = 1; candidate < kCandidates; ++candidate) {
                    const double edge = lattice[edge_base + candidate];
                    if (edge > best) {
                        second   = best;
                        best     = edge;
                        selected = candidate;
                    } else {
                        second = std::max(second, edge);
                    }
                }
                result.minimum_decision_margin[batch] =
                    std::min(result.minimum_decision_margin[batch], best - second);
                for (std::int32_t candidate = 0; candidate < kCandidates; ++candidate) {
                    result.probabilities[candidate_offset(batch, step, candidate)] =
                        candidate == selected ? 1.0 : 0.0;
                }
            }
            result.drafts[static_cast<std::size_t>(batch) * kSteps + step] =
                candidate_ids[candidate_offset(batch, step, selected)];
            predecessor_rank = selected;
        }
    }
    return result;
}

std::vector<ops::SamplingConfig> choose_configs(const std::vector<std::int32_t>& candidate_ids,
                                                const std::vector<double>& lattice,
                                                const std::vector<std::int32_t>& base_positions) {
    std::vector<ops::SamplingConfig> configs(kMaxBatch);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        if ((batch & 1) != 0) {
            configs[batch].temperature = 0.0F;
            configs[batch].seed        = 1000 + batch;
            continue;
        }
        configs[batch].temperature = 0.75F + 0.1F * static_cast<float>(batch % 3);
        bool found                 = false;
        for (std::uint64_t seed = 1; seed < 100000; ++seed) {
            configs[batch].seed = seed;
            const WalkResult candidate =
                walk_oracle(candidate_ids, lattice, configs, base_positions);
            if (candidate.minimum_decision_margin[batch] > 5.0e-3) {
                found = true;
                break;
            }
        }
        if (!found) { throw std::runtime_error("failed to choose selector RNG fixture"); }
    }
    const WalkResult final = walk_oracle(candidate_ids, lattice, configs, base_positions);
    for (std::int32_t batch = 0; batch < kMaxBatch; ++batch) {
        const double minimum_margin = configs[batch].temperature > 0.0F ? 5.0e-3 : 5.0e-4;
        if (final.minimum_decision_margin[batch] <= minimum_margin) {
            throw std::runtime_error(
                "selector fixture has an unstable decision boundary at B=" + std::to_string(batch) +
                ": " + std::to_string(final.minimum_decision_margin[batch]));
        }
    }
    return configs;
}

int verify_draws(const std::string& label, int batch, const std::vector<int>& ids,
                 const std::vector<ops::SamplingConfig>& configs, const std::vector<int>& positions,
                 const std::vector<int>& drafts, const std::vector<float>& q) {
    int failures = 0;
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < kSteps; ++i) {
            const auto base = candidate_offset(b, i, 0);
            double mass     = 0;
            for (int c = 0; c < kCandidates; ++c) {
                const float value = q[base + c];
                if (!std::isfinite(value) || value < 0) ++failures;
                mass += value;
                if (configs[b].temperature <= 0 && value != 0 && value != 1) ++failures;
            }
            if (std::abs(mass - 1.0) > 2.0e-6) ++failures;
            const float uniform = configs[b].temperature > 0
                                      ? oracle_uniform(configs[b].seed, positions[b] + i)
                                      : 0.0F;
            float cumulative    = 0;
            int selected        = kCandidates - 1;
            for (int c = 0; c < kCandidates; ++c) {
                cumulative += q[base + c];
                if (uniform < cumulative) {
                    selected = c;
                    break;
                }
            }
            if (drafts[b * kSteps + i] != ids[base + selected]) ++failures;
        }
    if (failures) std::cerr << label << " output q does not describe the actual draw\n";
    return failures;
}

int run(bool ties = false, bool dependent = false) {

    std::vector<std::int32_t> candidate_ids = make_candidate_ids();
    std::vector<float> unary_scores;
    std::vector<std::uint16_t> projected_hidden    = make_projected_hidden();
    const std::vector<std::int32_t> anchors        = make_anchors();
    const std::vector<std::int32_t> base_positions = make_base_positions();
    std::vector<double> lattice;
    bool stable = false;
    const std::vector<ops::SamplingConfig> greedy_probe(kMaxBatch);
    for (std::uint32_t salt = 0; salt < 64; ++salt) {
        unary_scores     = make_unary_scores(salt);
        lattice          = build_lattice(candidate_ids, unary_scores, projected_hidden, anchors);
        const auto probe = walk_oracle(candidate_ids, lattice, greedy_probe, base_positions);
        stable           = true;
        for (int b = 1; b < kMaxBatch; b += 2) stable &= probe.minimum_decision_margin[b] > 5.0e-4;
        if (stable) break;
    }
    if (!stable) throw std::runtime_error("could not construct stable selector fixture");
    if (ties) {
        std::fill(unary_scores.begin(), unary_scores.end(), 0.0F);
        std::fill(projected_hidden.begin(), projected_hidden.end(), 0);
        for (std::size_t first = 0; first < candidate_ids.size(); first += kCandidates)
            std::reverse(candidate_ids.begin() + first,
                         candidate_ids.begin() + first + kCandidates);
        lattice = build_lattice(candidate_ids, unary_scores, projected_hidden, anchors);
    }
    if (dependent) {
        std::fill(unary_scores.begin(), unary_scores.end(), 0.0F);
        lattice = build_lattice(candidate_ids, unary_scores, projected_hidden, anchors);
        for (int b = 0; b < kMaxBatch; ++b) {
            for (int c = 0; c < kCandidates; ++c)
                unary_scores[candidate_offset(b, 0, c)] = c == 5 ? 1000.0F : -1000.0F;
            int left = 0, right = 1;
            double largest = 0, midpoint = 0;
            for (int c = 0; c < kCandidates; ++c)
                for (int d = 0; d < kCandidates; ++d) {
                    const double actual =
                        lattice[lattice_offset(b, 1, 5, c)] - lattice[lattice_offset(b, 1, 5, d)];
                    const double other =
                        lattice[lattice_offset(b, 1, 0, c)] - lattice[lattice_offset(b, 1, 0, d)];
                    if (actual - other > largest) {
                        largest  = actual - other;
                        left     = c;
                        right    = d;
                        midpoint = (actual + other) / 2;
                    }
                }
            if (largest < 0.05) throw std::runtime_error("weak predecessor-sensitive fixture");
            for (int c = 0; c < kCandidates; ++c)
                unary_scores[candidate_offset(b, 1, c)] = -1000.0F;
            unary_scores[candidate_offset(b, 1, left)]  = static_cast<float>(-midpoint);
            unary_scores[candidate_offset(b, 1, right)] = 0;
        }
        for (int b = 0; b < kMaxBatch; ++b)
            for (int i = 2; i < kSteps; ++i)
                for (int c = 0; c < kCandidates; ++c)
                    unary_scores[candidate_offset(b, i, c)] = c == 0 ? 1000.0F : -1000.0F;
        lattice = build_lattice(candidate_ids, unary_scores, projected_hidden, anchors);
    }
    std::vector<ops::SamplingConfig> configs;
    if (ties || dependent) {
        configs.resize(kMaxBatch);
        for (int b = 0; b < kMaxBatch; ++b) {
            configs[b].temperature = !dependent && (b % 2 == 0) ? 0.75F : 0.0F;
            configs[b].seed        = 1234 + b;
        }
    } else
        configs = choose_configs(candidate_ids, lattice, base_positions);
    std::vector<int> host_counts(kTokenDomain, 7);
    DeviceBuffer counts = to_device(host_counts);
    for (auto& config : configs) {
        config.top_k             = 1;
        config.top_p             = 0.01F;
        config.min_p             = 0.9F;
        config.presence_penalty  = 2.0F;
        config.frequency_penalty = 3.0F;
        config.token_counts      = static_cast<int*>(counts.p);
    }
    const WalkResult expected = walk_oracle(candidate_ids, lattice, configs, base_positions);

    DeviceBuffer candidate_device = to_device(candidate_ids);
    DeviceBuffer unary_device     = to_device(unary_scores);
    DeviceBuffer hidden_device    = to_device(projected_hidden);
    DeviceBuffer anchor_device    = to_device(anchors);
    DeviceBuffer position_device  = to_device(base_positions);
    DeviceBuffer config_device    = to_device(configs);
    DeviceBuffer predecessor_device(static_cast<std::size_t>(kRank) * kCodebookRows *
                                    sizeof(std::uint16_t));
    DeviceBuffer successor_device(static_cast<std::size_t>(kRank) * kCodebookRows *
                                  sizeof(std::uint16_t));
    predecessor_device.fill();
    successor_device.fill();
    const std::vector<std::int32_t> tokens = accessed_tokens(candidate_ids, anchors);
    populate_codebook(predecessor_device, tokens, true);
    populate_codebook(successor_device, tokens, false);

    Tensor predecessor(predecessor_device.p, DType::BF16, {kRank, kCodebookRows});
    Tensor successor(successor_device.p, DType::BF16, {kRank, kCodebookRows});

    int failures = 0;
    for (int batch_size = 1; batch_size <= kMaxBatch; ++batch_size) {
        const std::size_t draft_count = static_cast<std::size_t>(batch_size) * kSteps;
        const std::size_t q_count     = draft_count * kCandidates;
        GuardedDeviceBuffer draft_device(draft_count * sizeof(std::int32_t));
        GuardedDeviceBuffer q_device(q_count * sizeof(float));
        draft_device.fill(0xff);
        q_device.fill(0xff);
        Tensor ids(candidate_device.p, DType::I32, {kCandidates, kSteps, batch_size});
        Tensor unary(unary_device.p, DType::FP32, {kCandidates, kSteps, batch_size});
        Tensor hidden(hidden_device.p, DType::BF16, {kRank, kSteps, batch_size});
        Tensor anchor(anchor_device.p, DType::I32, {batch_size});
        Tensor positions(position_device.p, DType::I32, {batch_size});
        Tensor drafts(draft_device.data(), DType::I32, {kSteps, batch_size});
        Tensor q(q_device.data(), DType::FP32, {kCandidates, kSteps, batch_size});
        const auto capacity = ops::candidate_selector_path_workspace_capacity_bytes(
            kSteps, kSteps, batch_size, batch_size);
        GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
        WorkspaceArena workspace(DeviceSpan{scratch.data(), scratch.bytes()});
        const auto launch = [&](cudaStream_t stream) {
            ops::candidate_selector_path(ids, unary, hidden, anchor, predecessor, successor,
                                         positions,
                                         static_cast<const ops::SamplingConfig*>(config_device.p),
                                         drafts, q, workspace, stream);
        };
        launch(nullptr);
        cuda_synchronize();

        const std::string label = "candidate_selector_path K=" + std::to_string(kSteps) +
                                  " B=" + std::to_string(batch_size);
        failures += verify_exact((label + " drafts").c_str(),
                                 from_device<std::int32_t>(draft_device.data(), draft_count),
                                 std::vector<std::int32_t>(expected.drafts.begin(),
                                                           expected.drafts.begin() + draft_count));
        const std::vector<float> actual_q = from_device<float>(q_device.data(), q_count);
        std::vector<double> actual_q_double(actual_q.begin(), actual_q.end());
        failures += verify_pointwise(
            label + " proposal_q", actual_q_double,
            std::span<const double>(expected.probabilities.data(), q_count), kProbabilityCriterion);
        failures += verify_draws(label, batch_size, candidate_ids, configs, base_positions,
                                 from_device<int>(draft_device.data(), draft_count), actual_q);
        failures += draft_device.verify_guards(label + " drafts");
        failures += q_device.verify_guards(label + " proposal_q");
        failures += scratch.verify_guards(label + " workspace");
        if (workspace.used() != 0 || workspace.peak_used() != capacity) {
            std::cerr << label << " workspace scope/peak\n";
            ++failures;
        }
        if (batch_size == 1 || batch_size == 8) {
            cudaStream_t stream = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                       "selector graph stream");
            {
                DecodeGraphDefinition definition;
                DecodeGraphExecutable executable;
                definition.capture(stream, [&] { launch(stream); });
                executable.instantiate(definition);
                executable.launch(stream);
                executable.launch(stream);
                cuda_synchronize(stream);
                if (!ties && !dependent && batch_size == 8 &&
                    (kSteps == 1 || kSteps == 7 || kSteps == 15)) {
                    auto next_positions = base_positions;
                    for (auto& position : next_positions) position += 4096;
                    const auto next_configs =
                        choose_configs(candidate_ids, lattice, next_positions);
                    const auto next_expected =
                        walk_oracle(candidate_ids, lattice, next_configs, next_positions);
                    cuda_check(cudaMemcpyAsync(position_device.p, next_positions.data(),
                                               next_positions.size() * sizeof(int),
                                               cudaMemcpyHostToDevice, stream),
                               "update graph positions");
                    cuda_check(cudaMemcpyAsync(config_device.p, next_configs.data(),
                                               next_configs.size() * sizeof(ops::SamplingConfig),
                                               cudaMemcpyHostToDevice, stream),
                               "update graph configs");
                    executable.launch(stream);
                    cuda_synchronize(stream);
                    const auto next_drafts = from_device<int>(draft_device.data(), draft_count);
                    const auto next_q      = from_device<float>(q_device.data(), q_count);
                    failures += verify_exact("selector graph updated RNG", next_drafts,
                                             next_expected.drafts);
                    failures +=
                        verify_pointwise("selector graph updated conditional q",
                                         std::vector<double>(next_q.begin(), next_q.end()),
                                         next_expected.probabilities, kProbabilityCriterion);
                    failures += verify_draws("selector graph updated q", batch_size, candidate_ids,
                                             next_configs, next_positions, next_drafts, next_q);
                    cuda_check(cudaMemcpyAsync(position_device.p, base_positions.data(),
                                               base_positions.size() * sizeof(int),
                                               cudaMemcpyHostToDevice, stream),
                               "restore graph positions");
                    cuda_check(cudaMemcpyAsync(config_device.p, configs.data(),
                                               configs.size() * sizeof(ops::SamplingConfig),
                                               cudaMemcpyHostToDevice, stream),
                               "restore graph configs");
                    executable.launch(stream);
                    cuda_synchronize(stream);
                }
            }
            cuda_check(cudaStreamDestroy(stream), "destroy selector stream");
            failures += verify_exact(
                (label + " graph drafts").c_str(),
                from_device<int>(draft_device.data(), draft_count),
                std::vector<int>(expected.drafts.begin(), expected.drafts.begin() + draft_count));
            failures += verify_exact((label + " graph q").c_str(),
                                     from_device<float>(q_device.data(), q_count), actual_q);
            failures += scratch.verify_guards(label + " graph workspace");
        }
    }
    if (!ties && !dependent && (kSteps == 1 || kSteps == 7 || kSteps == 15)) {
        constexpr std::array<int, 3> order{7, 0, 4};
        const auto pack = [&]<class T>(const std::vector<T>& values, int stride) {
            std::vector<T> out;
            for (int b : order)
                out.insert(out.end(), values.begin() + b * stride,
                           values.begin() + (b + 1) * stride);
            return out;
        };
        const auto pi = pack(candidate_ids, kSteps * kCandidates);
        const auto pu = pack(unary_scores, kSteps * kCandidates);
        const auto ph = pack(projected_hidden, kSteps * kRank);
        const auto pa = pack(anchors, 1), pp = pack(base_positions, 1);
        const auto pc = pack(configs, 1);
        const auto ed = pack(expected.drafts, kSteps);
        const auto eq = pack(expected.probabilities, kSteps * kCandidates);
        auto di = to_device(pi), du = to_device(pu), dh = to_device(ph), da = to_device(pa),
             dp = to_device(pp), dc = to_device(pc);
        GuardedDeviceBuffer dd(3 * kSteps * 4), dq(3 * kSteps * kCandidates * 4);
        const auto capacity =
            ops::candidate_selector_path_workspace_capacity_bytes(kSteps, kSteps, 3, 3);
        GuardedDeviceBuffer sw(std::max<std::size_t>(capacity, 1));
        WorkspaceArena workspace(DeviceSpan{sw.data(), sw.bytes()});
        Tensor ti(di.p, DType::I32, {kCandidates, kSteps, 3}),
            tu(du.p, DType::FP32, {kCandidates, kSteps, 3});
        Tensor th(dh.p, DType::BF16, {kRank, kSteps, 3}), ta(da.p, DType::I32, {3}),
            tp(dp.p, DType::I32, {3});
        Tensor td(dd.data(), DType::I32, {kSteps, 3}),
            tq(dq.data(), DType::FP32, {kCandidates, kSteps, 3});
        ops::candidate_selector_path(ti, tu, th, ta, predecessor, successor, tp,
                                     static_cast<const ops::SamplingConfig*>(dc.p), td, tq,
                                     workspace, nullptr);
        cuda_synchronize();
        const auto actual_d = from_device<int>(dd.data(), 3 * kSteps);
        const auto actual_q = from_device<float>(dq.data(), 3 * kSteps * kCandidates);
        failures += verify_exact("selector compact row RNG", actual_d, ed);
        failures += verify_pointwise("selector compact conditional q",
                                     std::vector<double>(actual_q.begin(), actual_q.end()), eq,
                                     kProbabilityCriterion);
        failures += verify_draws("selector compact", 3, pi, pc, pp, actual_d, actual_q);
        failures += dd.verify_guards("compact drafts") + dq.verify_guards("compact q") +
                    sw.verify_guards("compact workspace");
    }
    failures +=
        verify_exact("selector preserves candidate ids",
                     from_device<int>(candidate_device.p, candidate_ids.size()), candidate_ids);
    failures += verify_exact("selector preserves unary",
                             from_device<float>(unary_device.p, unary_scores.size()), unary_scores);
    failures += verify_exact("selector preserves hidden",
                             from_device<std::uint16_t>(hidden_device.p, projected_hidden.size()),
                             projected_hidden);
    failures += verify_exact("selector ignores token counts",
                             from_device<int>(counts.p, host_counts.size()), host_counts);
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        int failures = 0;
        for (kSteps = 1; kSteps <= 15; ++kSteps) failures += run();
        {
            kSteps = 1;
            failures += run(true);
            kSteps = 15;
            failures += run(true);
            kSteps = 2;
            failures += run(false, true);
            kSteps = 15;
            failures += run(false, true);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " candidate_selector_path\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "candidate_selector_path test: " << error.what() << '\n';
        return 1;
    }
}
