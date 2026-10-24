#include "reference.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "../../../../tools/support/kpool_chain_fixture.h"

#include <cstring>
#include <tuple>

using namespace ninfer;
using namespace ninfer::test;
namespace kp = ninfer::tool::kpool;

namespace {
constexpr ReductionCriterion kQueryCriterion{1. / 256, 1. / 256, 2. / 256};

// Check every publication against the mathematical Prepare formula, then use its
// represented outputs as inputs to independent downstream oracles. This avoids
// assuming that numerically close FP32 scores have identical Top512 boundaries.
int verify_prepare(kp::ChainFixture& f) {
    auto& i            = f.index;
    const auto& w      = f.weights;
    const int patterns = std::min(i.T, 7), F = i.E - i.T;
    std::vector<double> projected(std::size_t(patterns) * i.B * w.rows);
    parallel_rows(
        patterns * i.B * w.rows, std::uint64_t(projected.size()) * 4096, [&](int lo, int hi) {
            for (int j = lo; j < hi; ++j) {
                const int row = j % w.rows, t = j / w.rows % patterns, b = j / w.rows / patterns;
                double sum = 0;
                for (int d = 0; d < 4096; ++d)
                    sum += double(kp::decode(f.x.at(d, t, b))) * kp::decode(w.weight.at(d, row));
                projected[j] = sum;
            }
        });
    f.rank.host             = f.rank.download();
    i.coeff.host            = i.coeff.download();
    const auto latent_bytes = i.packed.download_plane(2 * i.layer);
    const auto key_bytes    = i.packed.download_plane(2 * i.layer + 1);
    std::vector<std::uint16_t> latent(f.latent.size()), keys(i.keys.size()),
        state(f.initial_state.size());
    std::memcpy(latent.data(), latent_bytes.data(), latent_bytes.size());
    std::memcpy(keys.data(), key_bytes.data(), key_bytes.size());
    f.state.copy_to_host(state.data(), state.size() * 2);
    auto expected_state = f.initial_state;
    KpoolError c_error, q_error, coeff_error, pool_error, tail_error;
    int failures = 0;
    for (int b = 0; b < i.B; ++b) {
        const int n = i.valid.at(b);
        std::vector<std::array<double, 128>> raw_k(F % 4 + n), raw_g(F % 4 + n);
        if (n)
            for (int r = 0; r < F % 4; ++r)
                for (int d = 0; d < 128; ++d) {
                    raw_k[r][d] = kp::decode(f.initial_state[f.state_index(false, 0, b, r, d)]);
                    raw_g[r][d] = kp::decode(f.initial_state[f.state_index(false, 1, b, r, d)]);
                }
        for (int t = 0; t < i.T; ++t) {
            if (t >= n) {
                for (int d = 0; d < 1536; ++d) failures += f.rank.at(d, t, b) != 0;
                for (int h = 0; h < 32; ++h) failures += i.coeff.at(h, t, b) != 0;
                continue;
            }
            const double* z = projected.data() + (b * patterns + t % patterns) * w.rows;
            const auto rms  = [&](int first, int dim, const auto& gamma, auto accept) {
                double sum = 0;
                for (int d = 0; d < dim; ++d) sum += z[first + d] * z[first + d];
                const double inv = 1 / std::sqrt(sum / dim + w.params.rms_epsilon);
                for (int d = 0; d < dim; ++d)
                    accept(d, z[first + d] * inv * kp::decode(gamma.at(d)));
            };
            const auto base = f.physical_token(b, F + t) * 512;
            rms(0, 512, w.cn, [&](int d, double expected) {
                c_error.add(kp::decode(latent[base + d]), expected);
                f.latent[base + d] = latent[base + d];
            });
            rms(768, 1536, w.qn, [&](int d, double expected) {
                q_error.add(kp::decode(f.rank.at(d, t, b)), expected);
            });
            for (int h = 0; h < 32; ++h) coeff_error.add(i.coeff.at(h, t, b), z[2304 + h]);
            double mean = 0, variance = 0;
            for (int d = 0; d < 128; ++d) mean += z[512 + d] / 128;
            for (int d = 0; d < 128; ++d) variance += std::pow(z[512 + d] - mean, 2) / 128;
            for (int d = 0; d < 128; ++d) {
                const double key = (z[512 + d] - mean) /
                                       std::sqrt(variance + w.params.index_epsilon) *
                                       kp::decode(w.kn.at(d)) +
                                   kp::decode(w.kb.at(d));
                raw_k[F % 4 + t][d] = kp::decode(kp::encode(float(key)));
                raw_g[F % 4 + t][d] = kp::decode(kp::encode(float(z[640 + d])));
            }
        }
        if (!n) continue;
        for (int g = 0; g < int(raw_k.size()) / 4; ++g) {
            const auto base = f.physical_token(b, (F / 4 + g) * 4) / 4 * 128;
            for (int d = 0; d < 128; ++d) {
                double logits[4], maximum = -INFINITY, sum = 0, value = 0;
                for (int r = 0; r < 4; ++r) {
                    logits[r] = raw_g[g * 4 + r][d] + kp::decode(w.ape.at(d, r));
                    maximum   = std::max(maximum, logits[r]);
                }
                for (int r = 0; r < 4; ++r) {
                    const double weight = std::exp(logits[r] - maximum);
                    sum += weight;
                    value += weight * raw_k[g * 4 + r][d];
                }
                pool_error.add(kp::decode(keys[base + d]), value / sum);
                i.keys[base + d] = keys[base + d];
            }
        }
        const int remainder = (F + n) % 4, start = int(raw_k.size()) - remainder;
        for (int kind = 0; kind < 2; ++kind)
            for (int r = 0; r < 3; ++r)
                for (int d = 0; d < 128; ++d) {
                    const auto offset  = f.state_index(true, kind, b, r, d);
                    const double value = r < remainder ? (kind ? raw_g : raw_k)[start + r][d] : 0;
                    tail_error.add(kp::decode(state[offset]), value);
                    if (r >= remainder) failures += state[offset] != 0;
                    expected_state[offset] = state[offset];
                }
    }
    failures += c_error.check("chain latent", kPreparedBf16Criterion) +
                q_error.check("chain rank", kPreparedBf16Criterion) +
                coeff_error.check("chain coefficient", kCoefficientCriterion) +
                pool_error.check("chain pool", kPooledBf16Criterion) +
                tail_error.check("chain tail", kPreparedBf16Criterion);
    // Updated expected buffers retain every untouched word, including future tokens.
    failures += latent != f.latent || keys != i.keys || state != expected_state;
    return failures;
}

int verify(kp::ChainFixture& f) {
    auto& i      = f.index;
    int failures = verify_prepare(f);
    if (failures) return failures;
    f.query.host      = f.query.download();
    i.query.host      = i.query.download();
    const auto output = f.output.download();
    std::vector<double> got_queries, expected_queries, got_mla, expected_mla;
    const auto ids = i.ids.download(), counts = i.counts.download(), ends = i.ends.download();
    for (int b = 0; b < i.B; ++b)
        for (int t = 0; t < i.T; ++t) {
            const int column = b * i.T + t;
            for (int row = 0; row < 20480; ++row) {
                const auto word = row < 16384 ? f.query.host[column * 16384 + row]
                                              : i.query.host[column * 4096 + row - 16384];
                failures += !std::isfinite(kp::decode(word)) || (t >= i.valid.at(b) && word != 0);
            }
            for (int row = 0; row < 16384; ++row)
                failures += !std::isfinite(kp::decode(output[column * 16384 + row])) ||
                            (t >= i.valid.at(b) && output[column * 16384 + row] != 0);
            if (t >= i.valid.at(b) ||
                (i.T > 4 && t != 0 && t != i.valid.at(b) / 2 && t != i.valid.at(b) - 1))
                continue;
            std::vector<double> q(20480);
            parallel_rows(20480, 20480ULL * 1536, [&](int lo, int hi) {
                for (int row = lo; row < hi; ++row) {
                    double dot = 0;
                    for (int d = 0; d < 1536; ++d)
                        dot +=
                            double(kp::decode(f.rank.at(d, t, b))) * kp::decode(f.w_b.at(d, row));
                    q[row] = dot;
                }
            });
            expected_queries.insert(expected_queries.end(), q.begin(), q.end());
            for (int row = 0; row < 20480; ++row)
                got_queries.push_back(kp::decode(row < 16384
                                                     ? f.query.host[column * 16384 + row]
                                                     : i.query.host[column * 4096 + row - 16384]));
            std::vector<const std::uint16_t*> rows;
            for (int g = 0; g < counts[column]; ++g)
                for (int r = 0; r < 4; ++r)
                    rows.push_back(f.latent.data() +
                                   f.physical_token(b, ids[column * 512 + g] * 4 + r) * 512);
            for (int token = ends[column] / 4 * 4; token < ends[column]; ++token)
                rows.push_back(f.latent.data() + f.physical_token(b, token) * 512);
            const auto result =
                mla_reference(f.query.host.data() + column * 16384, f.key_up.host.data(),
                              f.value_up.host.data(), rows, f.scale);
            expected_mla.insert(expected_mla.end(), result.begin(), result.end());
            for (int row = 0; row < 16384; ++row)
                got_mla.push_back(kp::decode(output[column * 16384 + row]));
        }
    failures += verify_reduction("chain W_B", got_queries, expected_queries, kQueryCriterion);
    failures += verify_selection(i);
    failures += verify_reduction("chain sparse MLA", got_mla, expected_mla, kMlaCriterion);
    failures += !i.scratch_guards() || !i.packed.guards_intact() || i.workspace->used() != 0 ||
                i.workspace->peak_used() > i.workspace_bytes;
    for (auto* v : {&f.x, &f.rank, &f.query, &f.output, &i.query}) failures += !v->guards();
    failures += f.x.download() != f.x.host || i.positions.download() != i.positions.host ||
                i.valid.download() != i.valid.host || i.tables.download() != i.tables.host;
    return failures;
}

int run(int T, int B, int E, int layers, bool graph) {
    auto& d = cuda_environment().context;
    kp::ChainFixture f(d.execution_view(), T, B, E, layers, B > 1);
    if (graph) {
        DecodeGraphDefinition definition;
        definition.capture(d.stream(), [&] { f.launch(d.execution_view()); });
        DecodeGraphExecutable executable;
        executable.instantiate(definition);
        // Change device valid lengths after capture. The before-state is immutable,
        // so repeated execution must publish the same causal cache and tail.
        if (T > 1) {
            --f.index.valid.at(0);
            f.index.valid.upload();
        }
        executable.launch(d.stream());
        executable.launch(d.stream());
    } else
        f.launch(d.execution_view());
    d.synchronize();
    return verify(f);
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (auto [T, B, E, layers, graph] : {std::tuple{1, 1, 1, 1, false},
                                          {1, 1, 3, 11, false},
                                          {1, 1, 2051, 11, false},
                                          {1, 1, 2052, 11, true},
                                          {5, 3, 2054, 11, true},
                                          {17, 1, 8193, 1, false},
                                          {65, 1, 8195, 11, true},
                                          {129, 1, 8193, 1, false},
                                          {1025, 1, 2053, 1, true},
                                          {1, 1, 65536, 11, false}})
        suite.add("kpool_chain/" + std::to_string(T) + "x" + std::to_string(B) + "/" +
                      std::to_string(E),
                  [=] { return run(T, B, E, layers, graph); },
                  {{"format", "bf16"}, {"execution", graph ? "graph" : "eager"}});
    return suite.main(argc, argv);
}
