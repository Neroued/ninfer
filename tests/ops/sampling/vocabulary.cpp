#include "ops/suite.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/vocabulary.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>

namespace {
using namespace ninfer;
using namespace ninfer::test;

std::uint64_t key(float score, int id) {
    if (score == 0) score = 0;
    const auto bits    = std::bit_cast<std::uint32_t>(score);
    const auto ordered = bits >> 31 ? ~bits : bits ^ 0x80000000U;
    return (std::uint64_t(ordered) << 32) | (UINT32_MAX - std::uint32_t(id));
}

struct Fixture {
    int v, t, b, r, k, domain, mask_stride, steps;
    std::vector<std::uint16_t> logits;
    std::vector<int> counts, valid, drafts;
    std::vector<std::uint32_t> masks;
    std::vector<ops::SamplingConfig> configs;
    GuardedDeviceBuffer d_logits, d_counts, d_valid, d_drafts, d_masks, d_configs, d_keys, d_work;
    Tensor keys, valid_view, draft_view;
    MemoryArena workspace;

    Fixture(int rows, int tokens, int batch, int ranks, int cap, bool penalties, bool masked,
            int proposal_columns = -1)
        : v(rows), t(tokens), b(batch), r(ranks), k(cap), domain(v * r - 3),
          mask_stride((domain + 31) / 32), steps(proposal_columns < 0 ? t - 1 : proposal_columns),
          logits(std::size_t(v) * t * b * r), counts(std::size_t(domain) * b), valid(b, t),
          drafts(std::max(1, steps) * b), masks(std::size_t(mask_stride) * t * b, UINT32_MAX),
          configs(b), d_logits(logits.size() * 2), d_counts(counts.size() * 4), d_valid(b * 4),
          d_drafts(drafts.size() * 4), d_masks(masks.size() * 4),
          d_configs(b * sizeof(ops::SamplingConfig)), d_keys(std::size_t(k) * t * b * r * 8),
          d_work(std::max<std::size_t>(
              256, ops::vocabulary_candidates_workspace_capacity_bytes(v, t * b, k))),
          keys(d_keys.span(), DType::I64, {k, t, b, r}),
          valid_view(d_valid.span(), DType::I32, {b}), workspace(d_work.span()) {
        if (steps) draft_view = Tensor(d_drafts.span(), DType::I32, {steps, b});
        for (int row = 0; row < b; ++row) {
            auto& c             = configs[row];
            c.temperature       = k == 1 || row % 3 == 1 ? 0 : 0.75F;
            c.top_k             = row % 3 == 2 ? 7 : 20;
            c.presence_penalty  = penalties ? .25F : 0;
            c.frequency_penalty = penalties ? .125F : 0;
            c.token_counts      = static_cast<int*>(d_counts.data()) + row * domain;
            if (masked)
                c.mask = {static_cast<const std::uint32_t*>(d_masks.data()) + row * t * mask_stride,
                          mask_stride};
            for (int id = 0; id < domain; ++id) counts[row * domain + id] = (id * 7 + row) % 4;
            for (int col = 0; col < steps; ++col)
                drafts[row * steps + col] = (col * 3 + row) % domain;
        }
        for (int rank = 0; rank < r; ++rank)
            for (int row = 0; row < b; ++row)
                for (int col = 0; col < t; ++col)
                    for (int local = 0; local < v; ++local) {
                        const int id = rank * v + local;
                        const float value =
                            float((id * 17 + col * 7 + row * 3) % 257 - 128) * .0625F;
                        logits[((std::size_t(rank) * b + row) * t + col) * v + local] =
                            f32_to_bf16(value);
                    }
        // Exact score ties, signed zero and excluded high-valued physical padding.
        for (int rank = 0; rank < r; ++rank) {
            logits[std::size_t(rank) * b * t * v] = rank % 2 ? 0x8000 : 0;
            for (int local = 0; local < v; ++local)
                if (rank * v + local >= domain)
                    for (int col = 0; col < t * b; ++col)
                        logits[(std::size_t(rank) * b * t + col) * v + local] = f32_to_bf16(10000);
        }
        reset(0);
    }

    void reset(int phase) {
        for (int row = 0; row < b; ++row) {
            const int limit = std::min(t, steps + 1);
            valid[row]      = phase && row == b - 1 && b > 1 ? 0 : limit - row % limit;
            if (!valid[row]) {
                configs[row].temperature  = std::numeric_limits<float>::quiet_NaN();
                configs[row].token_counts = reinterpret_cast<const int*>(std::uintptr_t(0x1000));
                configs[row].mask.words =
                    reinterpret_cast<const std::uint32_t*>(std::uintptr_t(0x2000));
            }
            for (int col = 0; col < t; ++col)
                for (int id = 0; id < domain; ++id) {
                    const auto at  = (row * t + col) * mask_stride + id / 32;
                    const auto bit = 1U << (id % 32);
                    // Phase zero leaves rank zero entirely empty when masks are enabled.
                    if ((phase == 0 && r > 1 && id < v) || (id + col + phase) % 5 == 0)
                        masks[at] &= ~bit;
                    else
                        masks[at] |= bit;
                }
        }
        d_logits.copy_from_host(logits.data(), logits.size() * 2);
        d_counts.copy_from_host(counts.data(), counts.size() * 4);
        d_valid.copy_from_host(valid.data(), b * 4);
        d_drafts.copy_from_host(drafts.data(), drafts.size() * 4);
        d_masks.copy_from_host(masks.data(), masks.size() * 4);
        d_configs.copy_from_host(configs.data(), configs.size() * sizeof(ops::SamplingConfig));
        CUDA_CHECK(cudaMemset(d_keys.data(), 0xa5, d_keys.bytes()));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    void launch(CudaExecutionView cuda, Tensor* records = nullptr,
                MemoryArena* confidence_workspace = nullptr) {
        for (int rank = 0; rank < r; ++rank) {
            Tensor input(d_logits.span().subspan(std::size_t(rank) * v * t * b * 2,
                                                 std::size_t(v) * t * b * 2),
                         DType::BF16, {v, t, b});
            auto out        = (records ? *records : keys).slice(3, rank, 1);
            const auto* cfg = static_cast<const ops::SamplingConfig*>(d_configs.data());
            if (records)
                ops::vocabulary_candidates_with_lse(input, rank * v, domain, cfg, valid_view,
                                                    draft_view, out, *confidence_workspace, cuda);
            else
                ops::vocabulary_candidates(input, rank * v, domain, cfg, valid_view, draft_view,
                                           out, workspace, cuda);
        }
    }

    std::vector<std::uint64_t> oracle(std::vector<double>* relative_lse = nullptr) const {
        std::vector<std::uint64_t> result(std::size_t(k) * t * b * r);
        if (relative_lse) relative_lse->assign(std::size_t(t) * b * r, 0.0);
        for (int rank = 0; rank < r; ++rank)
            for (int row = 0; row < b; ++row)
                for (int col = 0; col < valid[row]; ++col) {
                    const auto& c = configs[row];
                    std::vector<std::pair<double, int>> candidates;
                    for (int local = 0; local < v && rank * v + local < domain; ++local) {
                        const int id = rank * v + local;
                        if (c.mask.words &&
                            !(masks[(row * t + col) * mask_stride + id / 32] & (1U << (id % 32))))
                            continue;
                        int count = counts[row * domain + id];
                        for (int j = 0; j < col; ++j) count += drafts[row * steps + j] == id;
                        double score = bf16_to_f32(
                            logits[((std::size_t(rank) * b + row) * t + col) * v + local]);
                        score -= double(c.presence_penalty) * (count > 0) +
                                 double(c.frequency_penalty) * count;
                        candidates.emplace_back(score, id);
                    }
                    const int cap =
                        c.temperature > 0 ? (c.top_k > 0 && c.top_k < 20 ? c.top_k : 20) : 1;
                    const int n = std::min(cap, int(candidates.size()));
                    std::partial_sort(candidates.begin(), candidates.begin() + n, candidates.end(),
                                      [](auto a, auto b) {
                                          return a.first > b.first ||
                                                 (a.first == b.first && a.second < b.second);
                                      });
                    const auto index = (std::size_t(rank) * b + row) * t + col;
                    for (int i = 0; i < n; ++i)
                        result[index * k + i] =
                            key(float(candidates[i].first), candidates[i].second);
                    if (relative_lse && n) {
                        double mass      = 0.0;
                        const double tau = c.temperature > 0.0f ? c.temperature : 1.0;
                        for (const auto& [score, id] : candidates)
                            mass += std::exp((score - candidates.front().first) / tau);
                        (*relative_lse)[index] = std::log(mass);
                    }
                }
        return result;
    }
};

int check_records(Fixture& f, const GuardedDeviceBuffer& records) {
    std::vector<double> lse;
    const auto reference = f.oracle(&lse);
    const auto ordinary  = from_device<std::uint64_t>(f.d_keys.data(), reference.size());
    const auto actual    = from_device<std::uint64_t>(records.data(), lse.size() * (f.k + 1));
    std::vector<std::uint64_t> keys;
    int failures = 0;
    for (std::size_t i = 0; i < lse.size(); ++i) {
        const auto start = i * (f.k + 1);
        keys.insert(keys.end(), actual.begin() + start, actual.begin() + start + f.k);
        const auto word      = actual[start + f.k];
        const float log_mass = std::bit_cast<float>(static_cast<std::uint32_t>(word));
        if (!(word <= UINT32_MAX && (reference[i * f.k] != 0 || word == 0) && log_mass >= 0.0f &&
              std::abs(log_mass - lse[i]) <= 3e-6)) {
            std::cerr << "vocabulary relative LSE at column " << i << " is " << log_mass
                      << ", expected " << lse[i] << '\n';
            ++failures;
        }
    }
    return failures + verify_exact("confidence record keys", keys, reference) +
           verify_exact("ordinary vocabulary keys", keys, ordinary) +
           records.verify_guards("vocabulary LSE records");
}

int relative_lse_overlay() {
    Fixture f(4097, 4, 3, 2, 20, true, true);
    const auto e = cuda_environment().context.execution_view();
    GuardedDeviceBuffer records(std::size_t(21) * f.t * f.b * f.r * 8);
    Tensor output(records.span(), DType::I64, {21, f.t, f.b, f.r});
    GuardedDeviceBuffer scratch(
        ops::vocabulary_candidates_with_lse_workspace_capacity_bytes(f.v, f.t * f.b, f.k));
    MemoryArena arena(scratch.span());
    f.launch(e);
    f.launch(e, &output, &arena);
    cuda_synchronize(e.stream());
    return check_records(f, records) + scratch.verify_guards("vocabulary overlay scratch");
}

int probability_readout(int rows, int ranks, int cap, int batch) {
    Fixture f(rows, 1, batch, ranks, cap, true, true);
    auto& device       = cuda_environment().context;
    const auto e       = device.execution_view();
    const int physical = rows * ranks;
    GuardedDeviceBuffer records(std::size_t(cap + 1) * batch * ranks * 8);
    Tensor gathered(records.span(), DType::I64, {cap + 1, 1, batch, ranks});
    const auto bytes =
        ops::vocabulary_candidates_with_lse_workspace_capacity_bytes(rows, batch, cap);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(1, bytes));
    MemoryArena arena(scratch.span());
    std::vector<int> live(batch, 1), positions(batch);
    std::iota(positions.begin(), positions.end(), 17);
    auto enabled = to_device(live), position_buffer = to_device(positions);
    const Tensor live_rows(enabled.span(), DType::I32, {batch});
    const Tensor pos(position_buffer.span(), DType::I32, {batch});
    const auto* cfg = static_cast<const ops::SamplingConfig*>(f.d_configs.data());
    GuardedDeviceBuffer sampled(batch * 4), legacy(batch * 4), confidence(batch * 4),
        single_ids(batch * 4), single_p(batch * 4), unsharded(std::size_t(physical) * batch * 2);
    Tensor y(sampled.span(), DType::I32, {batch}), old(legacy.span(), DType::I32, {batch}),
        p(confidence.span(), DType::FP32, {batch}), local_y(single_ids.span(), DType::I32, {batch}),
        local_p(single_p.span(), DType::FP32, {batch}),
        local_x(unsharded.span(), DType::BF16, {physical, batch});
    GuardedDeviceBuffer single_work(
        std::max({std::size_t(1),
                  ops::sampling_with_probability_workspace_capacity_bytes(f.domain, 1, batch),
                  ops::argmax_with_probability_workspace_capacity_bytes(f.domain, 1)}));
    MemoryArena local_arena(single_work.span());
    DecodeGraphDefinition definition;
    definition.capture(e.stream(), [&] {
        f.launch(e, &gathered, &arena);
        ops::sample_vocabulary_candidates(gathered, cfg, pos, ops::kSamplePurposeDecode, y, p,
                                          live_rows, e);
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int failures = 0;
    for (int phase = 0; phase < 2; ++phase) {
        f.reset(phase);
        f.configs[0] = {.temperature = cap == 1 ? 0.0f : 1e-6f, .top_k = 1};
        // A tied, large shift that cannot be represented in an absolute LSE.
        f.logits[0] = f.logits[1] = f32_to_bf16(0x1p120f);
        if (ranks > 1) f.logits[std::size_t(ranks - 1) * batch * rows] = f32_to_bf16(0x1p120f);
        if (phase) f.logits[1] = f32_to_bf16(-0x1p120f);
        if (cap == 20) {
            f.configs[2].top_p = .8f;
            f.configs[2].min_p = .25f;
        }
        const int empty = batch - 2, inactive = batch - 1;
        std::fill_n(f.masks.begin() + empty * f.mask_stride, f.mask_stride, 0);
        f.valid[inactive] = 0;
        live[inactive]    = 0;
        f.d_masks.copy_from_host(f.masks.data(), f.masks.size() * 4);
        f.d_valid.copy_from_host(f.valid.data(), batch * 4);
        f.d_logits.copy_from_host(f.logits.data(), f.logits.size() * 2);
        f.d_configs.copy_from_host(f.configs.data(), f.d_configs.bytes());
        enabled.copy_from_host(live.data(), batch * 4);
        std::vector<std::uint16_t> logits(f.logits.size());
        for (int rank = 0; rank < ranks; ++rank)
            for (int b = 0; b < batch; ++b)
                std::copy_n(f.logits.begin() + (rank * batch + b) * rows, rows,
                            logits.begin() + b * physical + rank * rows);
        CUDA_CHECK(
            cudaMemcpy(unsharded.data(), logits.data(), logits.size() * 2, cudaMemcpyHostToDevice));

        graph.launch(e.stream());
        f.launch(e);
        ops::sample_vocabulary_candidates(f.keys, cfg, pos, ops::kSamplePurposeDecode, old,
                                          live_rows, e);
        device.synchronize();
        failures += check_records(f, records);
        const auto ids = from_device<int>(sampled.data(), batch);
        const auto got = from_device<float>(confidence.data(), batch);
        failures +=
            verify_exact("vocabulary confidence draw", ids, from_device<int>(legacy.data(), batch));
        failures += ids[inactive] != 0 || got[inactive] != 0.0f;
        failures += ids[empty] != -1 || got[empty] != 0.0f;

        live[empty] = 0;
        enabled.copy_from_host(live.data(), batch * 4);
        ops::sample(local_x, local_y, local_p, f.domain, cfg, pos, ops::kSamplePurposeDecode,
                    live_rows, local_arena, e);
        device.synchronize();
        const auto local_ids        = from_device<int>(single_ids.data(), batch);
        const auto local_confidence = from_device<float>(single_p.data(), batch);
        for (int b = 0; b < empty; ++b) {
            const auto& c = f.configs[b];
            std::vector<double> scores;
            for (int v = 0; v < f.domain; ++v) {
                if (c.mask.words && !(f.masks[b * f.mask_stride + v / 32] & (1U << (v % 32))))
                    continue;
                const int count = f.counts[b * f.domain + v];
                scores.push_back(double(bf16_to_f32(logits[b * physical + v])) -
                                 double(c.presence_penalty) * (count > 0) -
                                 double(c.frequency_penalty) * count);
            }
            const double max = *std::max_element(scores.begin(), scores.end());
            const double tau = c.temperature > 0.0f ? c.temperature : 1.0;
            double mass      = 0.0;
            for (const double score : scores) mass += std::exp((score - max) / tau);
            if (ids[b] != local_ids[b] ||
                !(got[b] > 0.0f && got[b] <= 1.0f && std::abs(got[b] * mass - 1.0) <= 5e-6 &&
                  std::abs(got[b] - local_confidence[b]) <= 5e-6 / mass)) {
                std::cerr << "vocabulary confidence b=" << b << " got " << got[b] << " expected "
                          << 1.0 / mass << '\n';
                ++failures;
            }
        }
        live[empty] = 1;
        if (cap == 1) {
            auto x  = local_x.slice(1, 0, 1);
            auto y0 = local_y.slice(0, 0, 1), p0 = local_p.slice(0, 0, 1);
            ops::argmax_with_probability(x, y0, p0, f.domain, local_arena, e);
            device.synchronize();
            failures += from_device<int>(y0.data, 1)[0] != ids[0];
            failures += !(std::abs(from_device<float>(p0.data, 1)[0] - got[0]) <= 2e-6);
        }
    }
    return failures + scratch.verify_guards("vocabulary confidence workspace") +
           confidence.verify_guards("vocabulary confidence output");
}

int extraction(int rows, int tokens, int batch, int ranks, int cap, bool controls, int steps = -1) {
    Fixture f(rows, tokens, batch, ranks, cap, controls, controls, steps);
    auto& device = cuda_environment().context;
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] { f.launch(device.execution_view()); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int errors = 0;
    for (int phase = 0; phase < 2; ++phase) {
        f.reset(phase);
        graph.launch(device.stream());
        device.synchronize();
        errors += verify_exact(
            "local candidates",
            from_device<std::uint64_t>(f.d_keys.data(), std::size_t(cap) * tokens * batch * ranks),
            f.oracle());
        errors += verify_exact("counts preserved",
                               from_device<int>(f.d_counts.data(), f.counts.size()), f.counts);
        errors += f.d_keys.verify_guards("keys") + f.d_work.verify_guards("workspace") +
                  f.d_logits.verify_guards("logits");
        errors += f.workspace.used() != 0;
    }
    return errors;
}

int long_columns(int cap) {
    constexpr int rows = 1025, tokens = 32769, batch = 2;
    std::vector<std::uint16_t> logits(std::size_t(rows) * tokens * batch);
    for (std::size_t i = 0; i < logits.size(); ++i)
        logits[i] = f32_to_bf16(float(int(i % rows) - 1000));
    auto input = to_device(logits);
    std::vector<ops::SamplingConfig> cfg(batch);
    for (auto& c : cfg) c.temperature = cap == 1 ? 0 : 1;
    auto configs = to_device(cfg), valid = to_device(std::vector<int>{tokens, tokens - 1});
    GuardedDeviceBuffer output(std::size_t(cap) * tokens * batch * 8);
    GuardedDeviceBuffer scratch(ops::vocabulary_candidates_workspace_capacity_bytes(
        rows, std::int64_t(tokens) * batch, cap));
    MemoryArena arena(scratch.span());
    Tensor x(input.span(), DType::BF16, {rows, tokens, batch});
    Tensor n(valid.span(), DType::I32, {batch}), y(output.span(), DType::I64, {cap, tokens, batch});
    auto& device = cuda_environment().context;
    ops::vocabulary_candidates(x, 0, rows, static_cast<const ops::SamplingConfig*>(configs.data()),
                               n, {}, y, arena, device.execution_view());
    device.synchronize();
    std::vector<std::uint64_t> expected(std::size_t(cap) * tokens * batch);
    for (int col = 0; col < tokens * batch - 1; ++col)
        for (int j = 0; j < cap; ++j) expected[std::size_t(col) * cap + j] = key(24 - j, 1024 - j);
    return verify_exact("large token extent",
                        from_device<std::uint64_t>(output.data(), expected.size()), expected) +
           output.verify_guards("keys") + scratch.verify_guards("workspace");
}

int sample_decisions(bool stochastic) {
    // The exact candidate input has masses 1, 1/2, 1/4, 1/8 before truncation.
    // min_p excludes the last candidate; top_p=.8 keeps the first two => p=(2/3,1/3).
    constexpr int batch = 8, ranks = 4;
    const int cap = stochastic ? 20 : 1;
    std::vector<std::uint64_t> keys(cap * batch * ranks);
    std::vector<ops::SamplingConfig> configs(batch);
    std::vector<int> positions(batch);
    for (int b = 0; b < batch; ++b) {
        configs[b].temperature = stochastic ? 1.0F : 0;
        configs[b].min_p       = .2F;
        configs[b].top_p       = .8F;
        configs[b].seed        = 991 + b * 71;
        for (int rank = 0; rank < ranks; ++rank)
            keys[(rank * batch + b) * cap] = key(-float(rank) * std::log(2.0F), 11 + rank);
    }
    GuardedDeviceBuffer d_keys(keys.size() * 8),
        d_cfg(configs.size() * sizeof(ops::SamplingConfig)), d_pos(batch * 4), d_out(batch * 4);
    d_keys.copy_from_host(keys.data(), keys.size() * 8);
    d_cfg.copy_from_host(configs.data(), configs.size() * sizeof(ops::SamplingConfig));
    Tensor in(d_keys.span(), DType::I64, {cap, 1, batch, ranks}),
        pos(d_pos.span(), DType::I32, {batch}), out(d_out.span(), DType::I32, {batch});
    auto& device = cuda_environment().context;
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] {
        ops::sample_vocabulary_candidates(in, static_cast<const ops::SamplingConfig*>(d_cfg.data()),
                                          pos, ops::kSamplePurposeDecode, out, {},
                                          device.execution_view());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int errors = 0, first = 0, draws = 0;
    for (int round = 0; round < (stochastic ? 1024 : 2); ++round) {
        std::iota(positions.begin(), positions.end(), round * batch + 17);
        d_pos.copy_from_host(positions.data(), batch * 4);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        graph.launch(device.stream());
        device.synchronize();
        const auto values = from_device<int>(d_out.data(), batch);
        for (int x : values) {
            errors += x != 11 && (x != 12 || !stochastic);
            first += x == 11;
            ++draws;
        }
    }
    if (stochastic) errors += std::abs(double(first) / draws - 2.0 / 3.0) > .03;
    // Global empty support is represented explicitly rather than a rank-local failure.
    CUDA_CHECK(cudaMemset(d_keys.data(), 0, d_keys.bytes()));
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    graph.launch(device.stream());
    device.synchronize();
    errors += verify_exact("empty global candidates", from_device<int>(d_out.data(), batch),
                           std::vector<int>(batch, -1));
    return errors + d_keys.verify_guards("candidate input") + d_out.verify_guards("decision");
}

int acceptance(bool stochastic, int d = 5) {
    constexpr int batch = 8, ranks = 2;
    const int cap = stochastic ? 20 : 1;
    std::vector<std::uint64_t> keys(cap * (d + 1) * batch * ranks);
    std::vector<int> drafts(d * batch), extents(batch), lengths(batch, 30),
        expected((d + 1) * batch), accepted(batch), licensed(batch);
    std::vector<ops::SamplingConfig> configs(batch);
    for (int b = 0; b < batch; ++b) {
        configs[b].temperature = stochastic ? .7F : 0;
        configs[b].seed        = 733 + b;
        extents[b]             = b == 7 ? d : b % (d + 1);
        accepted[b]            = b == 7 ? d : std::min(b % 4, extents[b]);
        licensed[b]            = accepted[b] + 1;
        for (int col = 0; col < d; ++col) drafts[b * d + col] = 100 + col;
        for (int col = 0; col <= extents[b]; ++col) {
            const int token = col == accepted[b] ? 200 + col : 100 + col;
            keys[(((col % ranks) * batch + b) * (d + 1) + col) * cap] = key(.25F, token);
        }
        for (int col = 0; col < accepted[b]; ++col) expected[b * (d + 1) + col] = 100 + col;
        expected[b * (d + 1) + accepted[b]] = 200 + accepted[b];
    }
    std::vector<int> step_major(drafts.size());
    for (int row = 0; row < batch; ++row)
        for (int col = 0; col < d; ++col) step_major[col * batch + row] = drafts[row * d + col];
    auto dk = to_device(keys), dd = to_device(step_major), de = to_device(extents),
         dc  = to_device(configs);
    auto dl  = to_device(std::vector<int>(batch, 30)),
         doo = to_device(std::vector<int>((d + 1) * batch, -1)),
         dn  = to_device(std::vector<int>(batch, -1));
    Tensor kt(dk.span(), DType::I64, {cap, d + 1, batch, ranks}),
        dt(Tensor(dd.span(), DType::I32, {batch, d}).permute({1, 0, 2, 3})),
        et(de.span(), DType::I32, {batch}), lt(dl.span(), DType::I32, {batch}),
        ot(doo.span(), DType::I32, {d + 1, batch}), nt(dn.span(), DType::I32, {batch});
    auto& device = cuda_environment().context;
    ops::accept_vocabulary_candidates(kt, static_cast<const ops::SamplingConfig*>(dc.data()), dt,
                                      et, lt, ot, nt, {}, device.execution_view());
    device.synchronize();
    int alias_failure = 1;
    try {
        ops::accept_vocabulary_candidates(kt, static_cast<const ops::SamplingConfig*>(dc.data()),
                                          dt, et, lt, ot, lt, {}, device.execution_view());
        device.synchronize();
    } catch (const std::invalid_argument&) { alias_failure = 0; }
    return alias_failure +
           verify_exact("licensed", from_device<int>(doo.data(), expected.size()), expected) +
           verify_exact("lengths", from_device<int>(dl.data(), batch), lengths) +
           verify_exact("counts", from_device<int>(dn.data(), batch), licensed);
}

int merged_support() {
    constexpr int batch = 8;
    std::vector<std::uint64_t> keys(20 * batch * 4);
    std::vector<ops::SamplingConfig> configs(batch);
    for (int b = 0; b < batch; ++b) {
        configs[b].temperature = 1;
        configs[b].seed        = 139 + b * 29;
        for (int rank = 0; rank < 4; ++rank)
            for (int j = 0; j < 20; ++j) keys[(rank * batch + b) * 20 + j] = key(0, rank + 4 * j);
    }
    auto dk = to_device(keys), dc = to_device(configs);
    auto dp = to_device(std::vector<int>(batch)), dout = to_device(std::vector<int>(batch));
    Tensor kt(dk.span(), DType::I64, {20, 1, batch, 4}), pt(dp.span(), DType::I32, {batch});
    Tensor ot(dout.span(), DType::I32, {batch});
    auto& device = cuda_environment().context;
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] {
        ops::sample_vocabulary_candidates(kt, static_cast<const ops::SamplingConfig*>(dc.data()),
                                          pt, ops::kSamplePurposeDecode, ot, {},
                                          device.execution_view());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    std::array<int, 20> histogram{};
    int errors = 0;
    for (int round = 0; round < 512; ++round) {
        std::vector<int> positions(batch, round + 31);
        dp.copy_from_host(positions.data(), batch * 4);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        graph.launch(device.stream());
        device.synchronize();
        for (int id : from_device<int>(dout.data(), batch)) {
            if (id < 0 || id >= 20)
                ++errors;
            else
                ++histogram[id];
        }
    }
    for (int n : histogram) errors += std::abs(n - 512.0 * batch / 20) > 80;
    return errors;
}

int acceptance_probability() {
    constexpr int batch = 8;
    std::vector<std::uint64_t> keys(20 * 2 * batch * 2);
    std::vector<ops::SamplingConfig> configs(batch);
    for (int b = 0; b < batch; ++b) {
        configs[b].temperature       = 1;
        configs[b].seed              = 441 + b * 37;
        keys[(b * 2) * 20]           = key(-std::log(3.0F), 11);
        keys[((batch + b) * 2) * 20] = key(0, 22);
        keys[(b * 2 + 1) * 20]       = key(0, 33);
    }
    auto dk = to_device(keys), dc = to_device(configs);
    auto dd = to_device(std::vector<int>(batch, 11)), de = to_device(std::vector<int>(batch, 1));
    auto dl   = to_device(std::vector<int>(batch, 0));
    auto dout = to_device(std::vector<int>(batch * 2, 0));
    auto dn   = to_device(std::vector<int>(batch, 0));
    Tensor kt(dk.span(), DType::I64, {20, 2, batch, 2}), dt(dd.span(), DType::I32, {1, batch});
    Tensor et(de.span(), DType::I32, {batch}), lt(dl.span(), DType::I32, {batch});
    Tensor ot(dout.span(), DType::I32, {2, batch});
    Tensor nt(dn.span(), DType::I32, {batch});
    auto& device = cuda_environment().context;
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] {
        ops::accept_vocabulary_candidates(kt, static_cast<const ops::SamplingConfig*>(dc.data()),
                                          dt, et, lt, ot, nt, {}, device.execution_view());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int errors = 0, accepted = 0;
    for (int round = 0; round < 512; ++round) {
        std::vector<int> positions(batch, round * 7);
        dl.copy_from_host(positions.data(), batch * 4);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        graph.launch(device.stream());
        device.synchronize();
        const auto result = from_device<int>(dout.data(), batch * 2);
        const auto counts = from_device<int>(dn.data(), batch);
        for (int b = 0; b < batch; ++b) {
            accepted += counts[b] - 1;
            errors += counts[b] == 2 ? result[b * 2] != 11 || result[b * 2 + 1] != 33
                                     : result[b * 2] != 22 || result[b * 2 + 1] != 0;
        }
    }
    errors += std::abs(double(accepted) / (512 * batch) - .25) > .035;
    return errors;
}

int commit_prefixes() {
    constexpr int t = 16, b = 8, v = 29, s = 5, stride = 33;
    std::vector<int> ids(t * b), valid(b), slots(b), counts(stride * s, 19);
    for (int row = 0; row < b; ++row) {
        valid[row] = row == 0 ? 0 : 2 * row;
        slots[row] = row == 0 ? -1 : row % s;
        for (int col = 0; col < t; ++col)
            ids[row * t + col] = col < valid[row] ? (col + row) % v : -1;
    }
    auto di = to_device(ids), dv = to_device(valid), ds = to_device(slots);
    GuardedDeviceBuffer dc(counts.size() * 4);
    dc.copy_from_host(counts.data(), counts.size() * 4);
    Tensor input(di.span(), DType::I32, {t, b}), n(dv.span(), DType::I32, {b});
    Tensor selected(ds.span(), DType::I32, {b}), out(dc.span(), DType::I32, {v, s});
    out.nb[1]    = stride * 4;
    auto& device = cuda_environment().context;
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] {
        ops::increment_token_counts_batch(input, n, selected, out, device.execution_view());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int errors = 0;
    for (int phase = 0; phase < 2; ++phase) {
        if (phase) {
            valid[0] = t;
            slots[0] = 3;
            std::fill_n(ids.begin(), t, 7);
            valid[4] = 0;
            slots[4] = -1;
            di.copy_from_host(ids.data(), ids.size() * 4);
            dv.copy_from_host(valid.data(), valid.size() * 4);
            ds.copy_from_host(slots.data(), slots.size() * 4);
        }
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        graph.launch(device.stream());
        device.synchronize();
        for (int row = 0; row < b; ++row)
            for (int col = 0; col < valid[row]; ++col)
                ++counts[slots[row] * stride + ids[row * t + col]];
        errors += verify_exact("committed prefix counts",
                               from_device<int>(dc.data(), counts.size()), counts);
    }
    return errors + dc.verify_guards("token counts");
}

int concentrated_candidates() {
    Fixture f(32768, 2, 17, 1, 20, false, false);
    std::fill(f.counts.begin(), f.counts.end(), 0);
    for (int row = 0; row < f.b; ++row) {
        auto& cfg       = f.configs[row];
        cfg.temperature = 1;
        cfg.top_k       = 20;
        if (row % 2) {
            cfg.presence_penalty  = .25F;
            cfg.frequency_penalty = .125F;
        }
        if (row % 4 >= 2) {
            cfg.mask = {static_cast<const std::uint32_t*>(f.d_masks.data()) +
                            row * f.t * f.mask_stride,
                        f.mask_stride};
            for (int col = 0; col < f.t; ++col)
                for (int word = 0; word < f.mask_stride; ++word)
                    f.masks[(row * f.t + col) * f.mask_stride + word] =
                        row % 4 == 3 ? 0 : 0xeeeeeeeeU;
        }
        for (int col = 0; col < f.t; ++col)
            for (int id = 0; id < f.v; ++id) {
                const int stride = row % 2 ? 128 : 256;
                float value      = row % 5 == 0 ? 0.F : (id % stride < 19 ? 9.F : -4.F);
                if (row % 5 != 0 && id % stride < 19 && id / stride < 2) value = 10.F;
                if (id >= f.domain) value = 10000.F;
                f.logits[(std::size_t(row) * f.t + col) * f.v + id] =
                    value == 0.F && id % 2 ? 0x8000U : f32_to_bf16(value);
            }
    }
    f.d_logits.copy_from_host(f.logits.data(), f.logits.size() * 2);
    f.d_counts.copy_from_host(f.counts.data(), f.counts.size() * 4);
    f.d_masks.copy_from_host(f.masks.data(), f.masks.size() * 4);
    f.d_configs.copy_from_host(f.configs.data(), f.configs.size() * sizeof(ops::SamplingConfig));
    auto& device = cuda_environment().context;
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] { f.launch(device.execution_view()); });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    graph.launch(device.stream());
    device.synchronize();
    int errors =
        verify_exact("concentrated vocabulary candidates",
                     from_device<std::uint64_t>(f.d_keys.data(), f.k * f.t * f.b), f.oracle());
    std::size_t previous_capacity = 0;
    for (int columns = 1; columns <= 128; ++columns) {
        const auto capacity =
            ops::vocabulary_candidates_workspace_capacity_bytes(f.v, columns, f.k);
        errors += capacity < previous_capacity;
        previous_capacity = capacity;
    }
    errors += f.d_keys.verify_guards("concentrated keys");
    errors += f.d_work.verify_guards("concentrated workspace");
    return errors;
}

int padded_decisions() {
    constexpr int b = 17, d = 3, w = d + 1, k = 20, r = 2;
    std::vector<std::uint64_t> sample_keys(k * b * r), target_keys(k * w * b * r);
    std::vector<int> live(b), positions(b), extents(b), drafts(d * b), ids(16 * d * b);
    std::vector<float> probabilities(ids.size());
    std::vector<ops::SamplingConfig> configs(b);
    auto ds = to_device(sample_keys), dt = to_device(target_keys), dl = to_device(live),
         dp = to_device(positions), de = to_device(extents), dd = to_device(drafts),
         di = to_device(ids), dq = to_device(probabilities), dc = to_device(configs);
    GuardedDeviceBuffer sampled(b * 4), greedy(w * b * 4), sparse(w * b * 4), greedy_counts(b * 4),
        sparse_counts(b * 4);
    Tensor sk(ds.span(), DType::I64, {k, 1, b, r}), tk(dt.span(), DType::I64, {k, w, b, r});
    Tensor lv(dl.span(), DType::I32, {b}), pv(dp.span(), DType::I32, {b}),
        ev(de.span(), DType::I32, {b}), dv(dd.span(), DType::I32, {d, b}),
        iv(di.span(), DType::I32, {16, d, b}), qv(dq.span(), DType::FP32, {16, d, b});
    Tensor out(sampled.span(), DType::I32, {b}), go(greedy.span(), DType::I32, {w, b}),
        so(sparse.span(), DType::I32, {w, b}), gc(greedy_counts.span(), DType::I32, {b}),
        sc(sparse_counts.span(), DType::I32, {b});
    auto& device               = cuda_environment().context;
    const auto* device_configs = static_cast<const ops::SamplingConfig*>(dc.data());
    DecodeGraphDefinition definition;
    definition.capture(device.stream(), [&] {
        ops::sample_vocabulary_candidates(sk, device_configs, pv, ops::kSamplePurposeDecode, out,
                                          lv, device.execution_view());
        ops::accept_vocabulary_candidates(tk, device_configs, dv, ev, pv, go, gc, lv,
                                          device.execution_view());
        ops::accept_vocabulary_candidates_sparse(tk, device_configs, dv, iv, qv, ev, pv, so, sc, lv,
                                                 device.execution_view());
    });
    DecodeGraphExecutable graph;
    graph.instantiate(definition);
    int errors = 0;
    for (int phase = 0; phase < 2; ++phase) {
        std::fill(sample_keys.begin(), sample_keys.end(), 0);
        std::fill(target_keys.begin(), target_keys.end(), 0);
        std::fill(drafts.begin(), drafts.end(), INT_MIN);
        std::fill(ids.begin(), ids.end(), INT_MIN);
        std::fill(probabilities.begin(), probabilities.end(),
                  std::numeric_limits<float>::quiet_NaN());
        std::vector<int> expected_sample(b), expected_tokens(w * b), expected_counts(b);
        for (int row = 0; row < b; ++row) {
            live[row]    = (row + phase) % 3 != 0;
            configs[row] = {};
            configs[row].temperature =
                live[row] ? (row % 2 ? .75F : 0.F) : std::numeric_limits<float>::quiet_NaN();
            configs[row].token_counts = reinterpret_cast<const int*>(std::uintptr_t(0x1000));
            configs[row].mask = {reinterpret_cast<const std::uint32_t*>(std::uintptr_t(0x2000)), 1};
            positions[row]    = live[row] ? 100 + row : INT_MIN;
            extents[row]      = live[row] ? row % w : INT_MIN;
            if (!live[row]) continue;
            expected_sample[row]                 = 100 + row * w;
            sample_keys[(row % r * b + row) * k] = key(0, expected_sample[row]);
            expected_counts[row]                 = extents[row] + 1;
            for (int col = 0; col <= extents[row]; ++col) {
                const int token                                  = expected_sample[row] + col;
                expected_tokens[row * w + col]                   = token;
                target_keys[((col % r * b + row) * w + col) * k] = key(0, token);
                if (col == extents[row]) continue;
                drafts[row * d + col] = token;
                const int at          = (row * d + col) * 16;
                std::fill_n(ids.begin() + at, 16, INT_MAX);
                std::fill_n(probabilities.begin() + at, 16, 0.F);
                ids[at]           = token;
                probabilities[at] = 1.F;
            }
        }
        ds.copy_from_host(sample_keys.data(), sample_keys.size() * 8);
        dt.copy_from_host(target_keys.data(), target_keys.size() * 8);
        dl.copy_from_host(live.data(), live.size() * 4);
        dp.copy_from_host(positions.data(), positions.size() * 4);
        de.copy_from_host(extents.data(), extents.size() * 4);
        dd.copy_from_host(drafts.data(), drafts.size() * 4);
        di.copy_from_host(ids.data(), ids.size() * 4);
        dq.copy_from_host(probabilities.data(), probabilities.size() * 4);
        dc.copy_from_host(configs.data(), configs.size() * sizeof(configs[0]));
        for (auto* buffer : {&sampled, &greedy, &sparse, &greedy_counts, &sparse_counts})
            buffer->fill(0xff);
        cuda_synchronize();
        graph.launch(device.stream());
        device.synchronize();
        errors += verify_exact("padded vocabulary sample", from_device<int>(sampled.data(), b),
                               expected_sample);
        for (auto* buffer : {&greedy, &sparse})
            errors += verify_exact("padded vocabulary acceptance",
                                   from_device<int>(buffer->data(), w * b), expected_tokens);
        for (auto* buffer : {&greedy_counts, &sparse_counts})
            errors += verify_exact("padded vocabulary counts", from_device<int>(buffer->data(), b),
                                   expected_counts);
        for (auto* buffer : {&sampled, &greedy, &sparse, &greedy_counts, &sparse_counts})
            errors += buffer->verify_guards("padded vocabulary output");
    }
    return errors;
}

} // namespace

void add_vocabulary_cases(ninfer::test::OpSuite& suite) {
    suite.add("vocabulary_candidates/relative_lse_overlay", relative_lse_overlay);
    for (auto [v, r, b] : {std::array{257, 1, 6}, std::array{13313, 2, 6}, std::array{4097, 4, 9}})
        for (int cap : {1, 20})
            suite.add("vocabulary_decision/probability/r" + std::to_string(r) + "/k" +
                          std::to_string(cap),
                      [=] { return probability_readout(v, r, cap, b); }, {{"execution", "graph"}});
    for (auto [v, t, b, r] :
         {std::array{7, 1, 1, 4}, std::array{513, 6, 4, 4}, std::array{124160, 1, 8, 2},
          std::array{124160, 6, 1, 2}, std::array{1025, 16, 8, 2}})
        for (int cap : {1, 20})
            for (bool controls : {false, true})
                suite.add("vocabulary_candidates/v" + std::to_string(v) + "/t" + std::to_string(t) +
                              "/b" + std::to_string(b) + "/k" + std::to_string(cap) + "/controls" +
                              std::to_string(controls),
                          [=] { return extraction(v, t, b, r, cap, controls); },
                          {{"execution", "graph"}, {"format", "bf16"}});
    for (int cap : {1, 20})
        suite.add("vocabulary_candidates/long_columns/k" + std::to_string(cap),
                  [=] { return long_columns(cap); });
    suite.add("vocabulary_candidates/proposal7_width4",
              [] { return extraction(257, 4, 5, 2, 20, true, 7); });
    suite.add("vocabulary_candidates/proposal2_width8",
              [] { return extraction(257, 8, 5, 2, 20, true, 2); });
    suite.add("vocabulary_candidates/anchor_with_proposals",
              [] { return extraction(257, 1, 2, 1, 20, true, 2); });
    suite.add("vocabulary_commit/prefixes", commit_prefixes);
    suite.add("vocabulary_candidates/padded_batch9",
              [] { return extraction(16385, 2, 9, 2, 20, true); }, {{"suite", "concurrency"}});
    suite.add("vocabulary_candidates/long_overlay",
              [] { return extraction(4097, 257, 1, 1, 20, true); },
              {{"suite", "concurrency"}, {"execution", "graph"}});
    suite.add("vocabulary_candidates/concentrated", concentrated_candidates,
              {{"suite", "concurrency"}, {"execution", "graph"}});
    suite.add("vocabulary_decision/padded_batch17", padded_decisions, {{"suite", "concurrency"}});
    suite.add("vocabulary_decision/merged_support", merged_support);
    suite.add("vocabulary_decision/greedy", [] { return sample_decisions(false); });
    suite.add("vocabulary_decision/stochastic", [] { return sample_decisions(true); });
    suite.add("vocabulary_accept/d15", [] { return acceptance(true, 15); });
    suite.add("vocabulary_accept/probability", acceptance_probability);
    suite.add("vocabulary_accept/greedy", [] { return acceptance(false); });
    suite.add("vocabulary_accept/stochastic", [] { return acceptance(true); });
}
