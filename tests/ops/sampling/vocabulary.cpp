#include "ops/suite.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "ninfer/ops/vocabulary.h"

#include <algorithm>
#include <bit>
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
    int v, t, b, r, k, domain, mask_stride;
    std::vector<std::uint16_t> logits;
    std::vector<int> counts, valid, drafts;
    std::vector<std::uint32_t> masks;
    std::vector<ops::SamplingConfig> configs;
    GuardedDeviceBuffer d_logits, d_counts, d_valid, d_drafts, d_masks, d_configs, d_keys, d_work;
    Tensor keys, valid_view, draft_view;
    MemoryArena workspace;

    Fixture(int rows, int tokens, int batch, int ranks, int cap, bool penalties, bool masked)
        : v(rows), t(tokens), b(batch), r(ranks), k(cap), domain(v * r - 3),
          mask_stride((domain + 31) / 32), logits(std::size_t(v) * t * b * r),
          counts(std::size_t(domain) * b), valid(b, t), drafts(std::max(1, t - 1) * b),
          masks(std::size_t(mask_stride) * t * b, UINT32_MAX), configs(b),
          d_logits(logits.size() * 2), d_counts(counts.size() * 4), d_valid(b * 4),
          d_drafts(drafts.size() * 4), d_masks(masks.size() * 4),
          d_configs(b * sizeof(ops::SamplingConfig)), d_keys(std::size_t(k) * t * b * r * 8),
          d_work(std::max<std::size_t>(
              256, ops::vocabulary_candidates_workspace_capacity_bytes(v, t * b, k))),
          keys(d_keys.span(), DType::I64, {k, t, b, r}),
          valid_view(d_valid.span(), DType::I32, {b}), workspace(d_work.span()) {
        if (t > 1) draft_view = Tensor(d_drafts.span(), DType::I32, {t - 1, b});
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
            for (int col = 0; col < t - 1; ++col)
                drafts[row * (t - 1) + col] = (col * 3 + row) % domain;
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
            valid[row] = phase && row == b - 1 && b > 1 ? 0 : t - row % t;
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

    void launch(CudaExecutionView cuda) {
        for (int rank = 0; rank < r; ++rank) {
            Tensor input(d_logits.span().subspan(std::size_t(rank) * v * t * b * 2,
                                                 std::size_t(v) * t * b * 2),
                         DType::BF16, {v, t, b});
            Tensor out(d_keys.span().subspan(std::size_t(rank) * k * t * b * 8,
                                             std::size_t(k) * t * b * 8),
                       DType::I64, {k, t, b});
            ops::vocabulary_candidates(input, rank * v, domain,
                                       static_cast<const ops::SamplingConfig*>(d_configs.data()),
                                       valid_view, draft_view, out, workspace, cuda);
        }
    }

    std::vector<std::uint64_t> oracle() const {
        std::vector<std::uint64_t> result(std::size_t(k) * t * b * r);
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
                        for (int j = 0; j < col; ++j) count += drafts[row * (t - 1) + j] == id;
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
                    for (int i = 0; i < n; ++i)
                        result[((std::size_t(rank) * b + row) * t + col) * k + i] =
                            key(float(candidates[i].first), candidates[i].second);
                }
        return result;
    }
};

int extraction(int rows, int tokens, int batch, int ranks, int cap, bool controls) {
    Fixture f(rows, tokens, batch, ranks, cap, controls, controls);
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
                                          pos, ops::kSamplePurposeDecode, out,
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
                                      et, lt, ot, nt, device.execution_view());
    device.synchronize();
    int alias_failure = 1;
    try {
        ops::accept_vocabulary_candidates(kt, static_cast<const ops::SamplingConfig*>(dc.data()),
                                          dt, et, lt, ot, lt, device.execution_view());
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
                                          pt, ops::kSamplePurposeDecode, ot,
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
                                          dt, et, lt, ot, nt, device.execution_view());
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

} // namespace

void add_vocabulary_cases(ninfer::test::OpSuite& suite) {
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
    suite.add("vocabulary_commit/prefixes", commit_prefixes);
    suite.add("vocabulary_decision/merged_support", merged_support);
    suite.add("vocabulary_decision/greedy", [] { return sample_decisions(false); });
    suite.add("vocabulary_decision/stochastic", [] { return sample_decisions(true); });
    suite.add("vocabulary_accept/d15", [] { return acceptance(true, 15); });
    suite.add("vocabulary_accept/probability", acceptance_probability);
    suite.add("vocabulary_accept/greedy", [] { return acceptance(false); });
    suite.add("vocabulary_accept/stochastic", [] { return acceptance(true); });
}
