#include "ops/suite.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "../../../tools/support/gathered_selector_fixture.h"

#include <numeric>
#include <array>
#include <limits>
#include <span>
#include <tuple>

using namespace ninfer;
using namespace ninfer::test;
using F = tool::GatheredSelectorFixture;

namespace {
constexpr PointwiseCriterion kQCriterion{2.e-6, 2.e-5};

std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

double uniform(std::uint64_t seed, int position) {
    auto k = mix64(seed ^ (std::uint64_t(std::uint32_t(position)) * 0xD1B54A32D192ED03ULL));
    k      = mix64(k ^ (std::uint64_t(ops::kSamplePurposeDFlash2Proposal) << 21));
    return double(std::uint32_t(k >> 40)) / 16777216.;
}

int verify_math(const F& f, const std::vector<int>& drafts, const std::vector<float>& q,
                std::span<const int> live_rows = {}) {
    std::vector<double> reference(q.size());
    int failures = 0;
    for (int b = 0; b < f.batch; ++b) {
        if (!live_rows.empty() && !live_rows[b]) {
            for (int step = 0; step < f.steps; ++step) failures += drafts[b * f.steps + step] != 0;
            continue;
        }
        int predecessor = f.anchors[b];
        for (int step = 0; step < f.steps; ++step) {
            const int col = b * f.steps + step;
            std::array<double, 16> edge{};
            int live = 0;
            for (int c = 0; c < 16; ++c) {
                const int token = f.ids[col * 16 + c];
                if (token == INT_MAX) {
                    edge[c] = -INFINITY;
                    continue;
                }
                ++live;
                double sum = f.unary[col * 16 + c];
                for (int r = 0; r < 256; ++r)
                    sum += double(F::value(F::weight(predecessor, r, true))) *
                           double(F::value(f.hidden_bits[col * 256 + r])) *
                           double(F::value(F::weight(token, r, false)));
                edge[c] = sum;
            }
            const int maximum = int(std::max_element(edge.begin(), edge.end()) - edge.begin());
            const double temperature = f.sampling[b].temperature;
            if (temperature <= 0)
                reference[col * 16 + maximum] = 1;
            else {
                double mass = 0;
                for (int c = 0; c < live; ++c) {
                    reference[col * 16 + c] = std::exp((edge[c] - edge[maximum]) / temperature);
                    mass += reference[col * 16 + c];
                }
                for (int c = 0; c < live; ++c) reference[col * 16 + c] /= mass;
            }
            double mass = 0;
            for (int c = 0; c < 16; ++c) {
                mass += q[col * 16 + c];
                if (!std::isfinite(q[col * 16 + c]) || q[col * 16 + c] < 0 ||
                    (c >= live && q[col * 16 + c] != 0))
                    ++failures;
            }
            failures += std::abs(mass - 1) > 2.e-6;
            const double u =
                temperature <= 0 ? 0 : uniform(f.sampling[b].seed, f.positions[b] + step);
            double cumulative = 0;
            int selected      = live - 1;
            for (int c = 0; c < live; ++c) {
                cumulative += q[col * 16 + c];
                if (u < cumulative) {
                    selected = c;
                    break;
                }
            }
            if (drafts[col] != f.ids[col * 16 + selected] || q[col * 16 + selected] <= 0) {
                std::cerr << "gathered selector draw/q mismatch d=" << step << " b=" << b << '\n';
                ++failures;
            }
            predecessor = drafts[col];
        }
    }
    return failures + verify_pointwise("gathered selector independent conditional q",
                                       std::vector<double>(q.begin(), q.end()), reference,
                                       kQCriterion);
}

int masked_rows(ops::CandidateSelectorInput input, int steps, int batch) {
    const auto e = cuda_environment().context.execution_view();
    F f(e.device(), steps, batch);
    auto candidate_input = to_device(f.ids), unary = to_device(f.unary);
    GuardedDeviceBuffer ids_buffer(f.ids.size() * sizeof(int)),
        drafts_buffer(std::size_t(steps) * batch * sizeof(int)),
        q_buffer(f.ids.size() * sizeof(float));
    Tensor ids(ids_buffer.span(), DType::I32, {16, steps, batch}),
        drafts(drafts_buffer.span(), DType::I32, {steps, batch}),
        q(q_buffer.span(), DType::FP32, {16, steps, batch});
    std::vector<int> live(batch, 1);
    auto device_live = to_device(live);
    Tensor validity(device_live.span(), DType::I32, {batch});
    const auto capacity =
        ops::candidate_selector_path_workspace_capacity_bytes(input, steps, steps, batch, batch);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(1, capacity));
    MemoryArena arena(scratch.span());
    const auto launch = [&] {
        const auto* configs = static_cast<const ops::SamplingConfig*>(f.configs.data());
        if (input == ops::CandidateSelectorInput::Candidates) {
            const Tensor candidates(candidate_input.span(), DType::I32, {16, steps, batch}),
                scores(unary.span(), DType::FP32, {16, steps, batch});
            ops::candidate_selector_path(candidates, scores, f.hidden_tensor(), f.anchor_tensor(),
                                         f.predecessor_tensor(), f.successor_tensor(), F::Domain,
                                         f.position_tensor(), configs, drafts, q, validity, arena,
                                         e);
        } else {
            ops::candidate_selector_path(f.key_tensor(), f.hidden_tensor(), f.anchor_tensor(),
                                         f.predecessor_tensor(), f.successor_tensor(), F::Domain,
                                         f.position_tensor(), configs, ids, drafts, q, validity,
                                         arena, e);
        }
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    cuda_synchronize();
    definition.capture(e.stream(), launch);
    executable.instantiate(definition);
    int failures = 0;
    for (int replay = 0; replay < 3; ++replay) {
        auto anchors = f.anchors, positions = f.positions, candidates = f.ids;
        auto hidden       = f.hidden_bits;
        auto keys         = f.key_words;
        auto configs      = f.sampling;
        auto scores       = f.unary;
        auto expected_ids = f.ids;
        for (int b = 0; b < batch; ++b) {
            live[b] = replay < 2 && b % 2 == replay;
            if (live[b]) continue;
            anchors[b] = positions[b] = INT_MIN;
            configs[b].temperature    = std::numeric_limits<float>::quiet_NaN();
            for (int step = 0; step < steps; ++step) {
                const auto column = std::size_t(b) * steps + step;
                std::fill_n(hidden.begin() + column * 256, 256, std::uint16_t{0x7fc0});
                for (int c = 0; c < 16; ++c) {
                    candidates[column * 16 + c]   = INT_MIN;
                    scores[column * 16 + c]       = std::numeric_limits<float>::quiet_NaN();
                    expected_ids[column * 16 + c] = 0;
                    for (int rank = 0; rank < 2; ++rank)
                        keys[(rank * std::size_t(steps) * batch + column) * 16 + c] = ULLONG_MAX;
                }
            }
        }
        device_live.copy_from_host(live.data(), device_live.size());
        f.anchor.copy_from_host(anchors.data(), f.anchor.size());
        f.position.copy_from_host(positions.data(), f.position.size());
        f.hidden.copy_from_host(hidden.data(), f.hidden.size());
        f.keys.copy_from_host(keys.data(), f.keys.size());
        f.configs.copy_from_host(configs.data(), f.configs.size());
        candidate_input.copy_from_host(candidates.data(), candidate_input.size());
        unary.copy_from_host(scores.data(), unary.size());
        ids_buffer.fill(0xff);
        drafts_buffer.fill(0xff);
        q_buffer.fill(0xff);
        cuda_synchronize();
        executable.launch(e.stream());
        cuda_synchronize(e.stream());
        failures += verify_math(f, from_device<int>(drafts_buffer.data(), steps * batch),
                                from_device<float>(q_buffer.data(), f.ids.size()), live);
        if (input == ops::CandidateSelectorInput::GatheredKeys)
            failures +=
                verify_exact("masked merged IDs", from_device<int>(ids_buffer.data(), f.ids.size()),
                             expected_ids);
        else
            failures +=
                verify_exact("masked candidate input preserved",
                             from_device<int>(candidate_input, candidates.size()), candidates);
        failures += verify_exact("selector validity preserved",
                                 from_device<int>(device_live, live.size()), live);
        failures +=
            ids_buffer.verify_guards("masked IDs") + drafts_buffer.verify_guards("masked drafts") +
            q_buffer.verify_guards("masked q") + scratch.verify_guards("masked selector workspace");
        failures += arena.used() != 0 || arena.peak_used() > capacity;
    }
    return failures;
}

int run(int steps, int batch, F::Pattern pattern, bool graph) {
    const auto e = cuda_environment().context.execution_view();
    F f(e.device(), steps, batch, pattern);
    GuardedDeviceBuffer id_buffer(16ULL * steps * batch * 4), draft_buffer(steps * batch * 4),
        q_buffer(16ULL * steps * batch * 4);
    Tensor ids(id_buffer.span(), DType::I32, {16, steps, batch}),
        drafts(draft_buffer.span(), DType::I32, {steps, batch}),
        q(q_buffer.span(), DType::FP32, {16, steps, batch});
    const auto capacity = ops::candidate_selector_path_workspace_capacity_bytes(
        ops::CandidateSelectorInput::GatheredKeys, steps, steps, batch, batch);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    MemoryArena arena(scratch.span());
    const auto launch = [&] {
        ops::candidate_selector_path(f.key_tensor(), f.hidden_tensor(), f.anchor_tensor(),
                                     f.predecessor_tensor(), f.successor_tensor(), F::Domain,
                                     f.position_tensor(),
                                     static_cast<const ops::SamplingConfig*>(f.configs.data()), ids,
                                     drafts, q, {}, arena, e);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    cuda_synchronize();
    if (graph) {
        definition.capture(e.stream(), launch);
        executable.instantiate(definition);
    }
    int failures = 0;
    for (int replay = 0; replay < (graph ? 2 : 1); ++replay) {
        if (replay) f.refresh(replay);
        for (int mode : {0, 1, 2}) {
            if (mode == 2 && batch == 1) continue;
            f.set_mode(mode);
            id_buffer.fill(0xff);
            draft_buffer.fill(0xff);
            q_buffer.fill(0xff);
            scratch.fill(0xa5);
            cuda_synchronize();
            if (graph)
                executable.launch(e.stream());
            else
                launch();
            cuda_synchronize(e.stream());
            const auto actual_ids    = from_device<int>(id_buffer.data(), f.ids.size());
            const auto actual_drafts = from_device<int>(draft_buffer.data(), steps * batch);
            const auto actual_q      = from_device<float>(q_buffer.data(), f.ids.size());
            failures += verify_exact("lossless gathered top16 IDs", actual_ids, f.ids);
            failures += verify_math(f, actual_drafts, actual_q);
            failures += id_buffer.verify_guards("merged IDs") +
                        draft_buffer.verify_guards("drafts") + q_buffer.verify_guards("q") +
                        scratch.verify_guards("selector workspace");
            failures += arena.used() != 0 || arena.peak_used() > capacity;
            // Swapping rank lists preserves the global input set and all RNG inputs.
            if (graph) {
                auto swapped = f.key_words;
                std::rotate(swapped.begin(), swapped.begin() + swapped.size() / 2, swapped.end());
                f.keys.copy_from_host(swapped.data(), swapped.size() * 8);
                cuda_synchronize();
                executable.launch(e.stream());
                cuda_synchronize(e.stream());
                failures +=
                    verify_exact("rank invariant IDs",
                                 from_device<int>(id_buffer.data(), f.ids.size()), actual_ids);
                failures += verify_exact("rank invariant drafts",
                                         from_device<int>(draft_buffer.data(), steps * batch),
                                         actual_drafts);
                failures +=
                    verify_exact("rank invariant FP32 q",
                                 from_device<float>(q_buffer.data(), f.ids.size()), actual_q);
                f.keys.copy_from_host(f.key_words.data(), f.key_words.size() * 8);
            }
        }
        failures +=
            verify_exact("selector keys unchanged",
                         from_device<unsigned long long>(f.keys, f.key_words.size()), f.key_words);
        failures +=
            verify_exact("selector hidden unchanged",
                         from_device<std::uint16_t>(f.hidden, f.hidden_bits.size()), f.hidden_bits);
        failures += verify_exact("selector anchors unchanged", from_device<int>(f.anchor, batch),
                                 f.anchors);
        failures += verify_exact("selector positions unchanged",
                                 from_device<int>(f.position, batch), f.positions);
        for (int token : f.touched)
            for (bool pred : {true, false}) {
                std::array<std::uint16_t, 256> row;
                (pred ? f.predecessor : f.successor)
                    .copy_to_host(row.data(), 512, std::size_t(token) * 512);
                for (int r = 0; r < 256; ++r)
                    if (row[r] != F::weight(token, r, pred)) ++failures;
            }
    }
    return failures;
}
} // namespace

void add_gathered_selector_cases(OpSuite& suite) {
    for (auto [input, steps, batch] :
         {std::tuple{ops::CandidateSelectorInput::Candidates, 3, 9},
          std::tuple{ops::CandidateSelectorInput::Candidates, 7, 17},
          std::tuple{ops::CandidateSelectorInput::GatheredKeys, 7, 2},
          std::tuple{ops::CandidateSelectorInput::GatheredKeys, 7, 17}}) {
        const auto form =
            input == ops::CandidateSelectorInput::Candidates ? "candidates" : "gathered";
        suite.add("selector/masked/" + std::string(form) + "/k" + std::to_string(steps) + "/b" +
                      std::to_string(batch),
                  [=] { return masked_rows(input, steps, batch); },
                  {{"suite", "selector"}, {"suite", "concurrency"}, {"execution", "graph"}});
    }
    suite.add("gathered_selector/contracts",
              [] {
                  const auto e = cuda_environment().context.execution_view();
                  F f(e.device(), 1, 1);
                  DeviceBuffer outputs(e.device(), 256), scratch(e.device(), 256);
                  Tensor ids(outputs.span().subspan(0, 64), DType::I32, {16, 1, 1}),
                      drafts(outputs.span().subspan(64, 16), DType::I32, {1, 1}),
                      q(outputs.span().subspan(80, 64), DType::FP32, {16, 1, 1});
                  MemoryArena arena(scratch.span());
                  int failures        = 0;
                  const auto rejected = [&](const Tensor& keys, const Tensor& successor, int domain,
                                            const Tensor& qview) {
                      try {
                          auto out = qview;
                          ops::candidate_selector_path(
                              keys, f.hidden_tensor(), f.anchor_tensor(), f.predecessor_tensor(),
                              successor, domain, f.position_tensor(),
                              static_cast<const ops::SamplingConfig*>(f.configs.data()), ids,
                              drafts, out, {}, arena, e);
                          ++failures;
                      } catch (const std::invalid_argument&) {}
                  };
                  for (int domain : {0, -1, F::Rows + 1})
                      rejected(f.key_tensor(), f.successor_tensor(), domain, q);
                  auto wrong_rows = f.successor_tensor();
                  --wrong_rows.ne[1];
                  rejected(f.key_tensor(), wrong_rows, F::Domain, q);
                  auto wrong_ranks  = f.key_tensor();
                  wrong_ranks.ne[3] = 1;
                  rejected(wrong_ranks, f.successor_tensor(), F::Domain, q);
                  rejected(f.key_tensor(), f.successor_tensor(), F::Domain,
                           Tensor(ids.data, e.device(), DType::FP32, {16, 1, 1}));
                  for (auto input : {ops::CandidateSelectorInput::Candidates,
                                     ops::CandidateSelectorInput::GatheredKeys}) {
                      std::size_t peak = 0;
                      for (int d = 1; d <= 15; ++d)
                          for (int b : {1, 2, 3, 7, 8, 9, 17, 128})
                              peak = std::max(peak,
                                              ops::candidate_selector_path_workspace_capacity_bytes(
                                                  input, d, d, b, b));
                      failures += peak != ops::candidate_selector_path_workspace_capacity_bytes(
                                              input, 1, 15, 1, 128);
                  }
                  return failures;
              },
              {{"suite", "selector"}});
    for (int d : {1, 3, 5, 6, 7, 9, 10, 15})
        for (int b : {1, 2, 3, 8})
            suite.add("gathered_selector/d" + std::to_string(d) + "/b" + std::to_string(b),
                      [=] {
                          return run(d, b, F::Pattern::Regular,
                                     (d == 1 || d == 7 || d == 15) && (b == 1 || b == 8));
                      },
                      {{"suite", "selector"}});
    for (auto [name, pattern] : {std::pair{"ties", F::Pattern::Ties},
                                 {"precision", F::Pattern::Precision},
                                 {"empty_rank", F::Pattern::EmptyRank},
                                 {"missing", F::Pattern::Missing}})
        for (int d : {1, 7, 15})
            suite.add("gathered_selector/" + std::string(name) + "/d" + std::to_string(d),
                      [=] { return run(d, 8, pattern, true); }, {{"suite", "selector"}});
    for (auto [name, pattern] :
         {std::pair{"empty_rank", F::Pattern::EmptyRank}, {"missing", F::Pattern::Missing}})
        for (int d : {7, 15})
            suite.add("gathered_selector/" + std::string(name) + "/d" + std::to_string(d) + "/b1",
                      [=] { return run(d, 1, pattern, true); }, {{"suite", "selector"}});
}
