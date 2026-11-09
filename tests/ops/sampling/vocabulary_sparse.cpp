#include "ops/suite.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"
#include "../../../tools/support/sparse_vocabulary_fixture.h"
#include <numeric>
#include <cstring>

using namespace ninfer;
using namespace ninfer::test;
using F = tool::SparseVocabularyFixture;

namespace {
std::uint64_t mix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

double uniform(std::uint64_t seed, int position, int purpose) {
    auto v = mix(seed ^ (std::uint64_t(std::uint32_t(position)) * 0xD1B54A32D192ED03ULL));
    v      = mix(v ^ (std::uint64_t(purpose) << 21));
    return double(v >> 40) / 16777216.;
}

std::vector<double> probabilities(const F& f, int b, int col) {
    auto support    = f.targets[b * (f.steps + 1) + col];
    const auto& cfg = f.configs[b];
    int count       = cfg.temperature <= 0 ? 1 : cfg.top_k > 0 && cfg.top_k < 20 ? cfg.top_k : 20;
    support.resize(std::min<int>(count, support.size()));
    std::vector<double> p(support.size());
    if (cfg.temperature <= 0) {
        p[0] = 1;
        return p;
    }
    for (std::size_t j = 0; j < p.size(); ++j)
        p[j] = std::exp((double(support[j].score) - support[0].score) / cfg.temperature);
    const double total = std::accumulate(p.begin(), p.end(), 0.);
    double mass        = 0;
    std::size_t n      = 0;
    for (; n < p.size(); ++n) {
        if (p[n] < cfg.min_p * p[0]) break;
        mass += p[n];
        if (cfg.top_p < 1 && mass >= cfg.top_p * total) {
            ++n;
            break;
        }
    }
    n = std::max<std::size_t>(1, n);
    p.resize(n);
    mass = std::accumulate(p.begin(), p.end(), 0.);
    for (auto& x : p) x /= mass;
    return p;
}

double proposal(const F& f, int b, int col, int token) {
    for (int c = 0; c < 16; ++c) {
        const int at = (b * f.steps + col) * 16 + c;
        if (f.q[at] > 0 && f.ids[at] == token) return f.q[at];
    }
    return 0;
}

std::pair<std::vector<int>, std::vector<int>> reference(const F& f) {
    std::vector<int> out((f.steps + 1) * f.batch), counts(f.batch);
    for (int b = 0; b < f.batch; ++b)
        for (int col = 0; col <= f.extents[b]; ++col) {
            const auto& cfg  = f.configs[b];
            const auto& all  = f.targets[b * (f.steps + 1) + col];
            auto p           = probabilities(f, b, col);
            const bool bonus = col == f.extents[b];
            const int draft  = bonus ? -1 : f.drafts[b * f.steps + col];
            bool accept      = false;
            if (!bonus) {
                if (cfg.temperature <= 0)
                    accept = draft == all[0].id;
                else {
                    double pd = 0;
                    for (std::size_t j = 0; j < p.size(); ++j)
                        if (all[j].id == draft) pd = p[j];
                    const double qd = proposal(f, b, col, draft);
                    accept          = uniform(cfg.seed, f.positions[b] + col + 1,
                                              ops::kSamplePurposeSpeculativeAccept) < std::min(1., pd / qd);
                }
            }
            if (accept) {
                out[b * (f.steps + 1) + col] = draft;
                continue;
            }
            int token = all[0].id;
            if (cfg.temperature > 0) {
                if (!bonus)
                    for (std::size_t j = 0; j < p.size(); ++j)
                        p[j] = std::max(0., p[j] - proposal(f, b, col, all[j].id));
                double mass = std::accumulate(p.begin(), p.end(), 0.);
                if (!(mass > 0)) {
                    p    = probabilities(f, b, col);
                    mass = std::accumulate(p.begin(), p.end(), 0.);
                }
                const double goal = uniform(cfg.seed, f.positions[b] + col + 1,
                                            bonus ? ops::kSamplePurposeSpeculativeBonus
                                                  : ops::kSamplePurposeSpeculativeCorrection) *
                                    mass;
                double cumulative = 0;
                for (std::size_t j = 0; j < p.size(); ++j) {
                    cumulative += p[j];
                    if (p[j] > 0) {
                        token = all[j].id;
                        if (goal < cumulative) break;
                    }
                }
            }
            out[b * (f.steps + 1) + col] = token;
            counts[b]                    = col + 1;
            break;
        }
    return {out, counts};
}

int run(int d, int b, int mode, int accepted, F::Pattern pattern, bool graph, int ranks = 2,
        bool roundoff = false) {
    auto e = cuda_environment().context.execution_view();
    F f(e.device(), d, b, ranks, mode, accepted, pattern);
    GuardedDeviceBuffer ob((d + 1) * b * 4), cb(b * 4);
    Tensor out(ob.span(), DType::I32, {d + 1, b}), counts(cb.span(), DType::I32, {b});
    // Exercise noncontiguous draft columns while leaving inactive storage poisoned.
    std::vector<int> strided(d * b * 3, -123);
    DeviceBuffer db(e.device(), strided.size() * 4);
    Tensor drafts(db.span(), DType::I32, {d, b});
    drafts.nb[0] = 12;
    drafts.nb[1] = 12 * d;
    auto launch  = [&] {
        ops::accept_vocabulary_candidates_sparse(
            f.keys(), static_cast<const ops::SamplingConfig*>(f.config_buffer.data()), drafts,
            f.id_view(), f.q_view(), f.extent_view(), f.position_view(), out, counts, e);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    cuda_synchronize();
    if (graph) {
        definition.capture(e.stream(), launch);
        executable.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < (roundoff                       ? 1
                                 : pattern == F::Pattern::Ratio ? 32
                                 : graph                        ? 3
                                                                : 1);
         ++phase) {
        f.refresh(phase, graph);
        for (int row = 0; row < b; ++row)
            if (f.configs[row].temperature <= 0)
                for (int col = 0; col < f.extents[row]; ++col) {
                    const int at = (row * d + col) * 16;
                    std::fill_n(f.q.begin() + at, 16, std::numeric_limits<float>::quiet_NaN());
                    std::fill_n(f.ids.begin() + at, 16, INT_MIN);
                }
        f.id_buffer.copy_from_host(f.ids.data(), f.ids.size() * 4);
        f.q_buffer.copy_from_host(f.q.data(), f.q.size() * 4);
        if (roundoff) {
            const auto& all = f.targets[0];
            std::fill(f.q.begin(), f.q.end(), 0.F);
            std::fill(f.ids.begin(), f.ids.end(), INT_MAX);
            for (int j = 0; j < 4; ++j) {
                f.q[j]   = std::nextafter(.25F, 1.F);
                f.ids[j] = all[j].id;
            }
            f.configs[0].seed = 0x514353fc7e7553d8ULL;
            if (uniform(f.configs[0].seed, f.positions[0] + 1,
                        ops::kSamplePurposeSpeculativeAccept) != 1. - 0x1p-24)
                return 1;
            f.id_buffer.copy_from_host(f.ids.data(), f.ids.size() * 4);
            f.q_buffer.copy_from_host(f.q.data(), f.q.size() * 4);
            f.config_buffer.copy_from_host(f.configs.data(),
                                           f.configs.size() * sizeof(f.configs[0]));
        }
        for (int i = 0; i < d * b; ++i) strided[i * 3] = f.drafts[i];
        db.copy_from_host(strided.data(), strided.size() * 4);
        ob.fill(0xff);
        cb.fill(0xff);
        cuda_synchronize();
        if (graph)
            executable.launch(e.stream());
        else
            launch();
        cuda_synchronize(e.stream());
        auto [expected, n] = reference(f);
        failures += verify_exact("sparse target licensed tokens",
                                 from_device<int>(ob.data(), expected.size()), expected) +
                    verify_exact("sparse target counts", from_device<int>(cb.data(), b), n);
        failures += ob.verify_guards("sparse licensed") + cb.verify_guards("sparse counts");
        failures += verify_exact("sparse keys unchanged",
                                 from_device<std::uint64_t>(f.key_buffer, f.words.size()), f.words);
        failures += verify_exact("sparse candidate IDs unchanged",
                                 from_device<int>(f.id_buffer, f.ids.size()), f.ids);
        auto actualq = from_device<float>(f.q_buffer, f.q.size());
        failures += std::memcmp(actualq.data(), f.q.data(), f.q.size() * 4) != 0;
        failures +=
            verify_exact("strided drafts unchanged", from_device<int>(db, strided.size()), strided);
        failures +=
            verify_exact("extents unchanged", from_device<int>(f.extent_buffer, b), f.extents) +
            verify_exact("positions unchanged", from_device<int>(f.position_buffer, b),
                         f.positions);
        auto cfg = from_device<ops::SamplingConfig>(f.config_buffer, b);
        failures += std::memcmp(cfg.data(), f.configs.data(), cfg.size() * sizeof(cfg[0])) != 0;
    }
    auto expect_reject = [&](const Tensor& id, const Tensor& q, const Tensor& tokens,
                             const Tensor& n) {
        try {
            auto t = tokens, c = n;
            ops::accept_vocabulary_candidates_sparse(
                f.keys(), static_cast<const ops::SamplingConfig*>(f.config_buffer.data()), drafts,
                id, q, f.extent_view(), f.position_view(), t, c, e);
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    auto wrong  = f.id_view();
    wrong.ne[0] = 15;
    expect_reject(wrong, f.q_view(), out, counts);
    expect_reject(f.id_view(), f.q_view(), out, f.position_view());
    auto alias = Tensor(f.q_buffer.data(), e.device(), DType::I32, {d + 1, b});
    expect_reject(f.id_view(), f.q_view(), alias, counts);
    return failures;
}
} // namespace

void add_vocabulary_sparse_cases(OpSuite& suite) {
    suite.add("vocabulary_sparse/dense_q",
              [] { return run(15, 8, 2, 7, F::Pattern::DenseQ, true); },
              {{"suite", "vocabulary_sparse"}});
    suite.add("vocabulary_sparse/rounded_zero_residual",
              [] { return run(1, 1, 1, 1, F::Pattern::Ratio, false, 2, true); },
              {{"suite", "vocabulary_sparse"}});
    for (int d : {1, 3, 7, 15})
        for (int b : {1, 8})
            for (int mode : {0, 1, 2}) {
                if (mode == 2 && b == 1) continue;
                std::vector<int> lengths{0, d / 2, d};
                lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
                for (int a : lengths)
                    suite.add("vocabulary_sparse/d" + std::to_string(d) + "/b" + std::to_string(b) +
                                  "/m" + std::to_string(mode) + "/a" + std::to_string(a),
                              [=] { return run(d, b, mode, a, F::Pattern::Regular, true); },
                              {{"suite", "vocabulary_sparse"}});
            }
    suite.add("vocabulary_sparse/ratio", [] { return run(7, 8, 1, 7, F::Pattern::Ratio, true); },
              {{"suite", "vocabulary_sparse"}});
    for (int r : {1, 4, 16})
        suite.add("vocabulary_sparse/ranks" + std::to_string(r),
                  [=] { return run(15, 3, 2, 7, F::Pattern::Regular, true, r); },
                  {{"suite", "vocabulary_sparse"}});
    suite.add("vocabulary_sparse/empty_rank",
              [] { return run(15, 8, 2, 7, F::Pattern::EmptyRank, true); },
              {{"suite", "vocabulary_sparse"}});
}
