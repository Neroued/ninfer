#include "reference.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "../../../../tools/support/kpool_prepare_fixture.h"

#include <cmath>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
namespace kp = ninfer::tool::kpool;

namespace {
template <class T>
int exact(const std::vector<T>& a, const std::vector<T>& b, const char* label) {
    if (a == b) return 0;
    std::cerr << label << " changed outside its write set\n";
    return 1;
}

int verify(kp::Fixture& f) {
    const auto& w         = f.w;
    const bool fp8        = f.cache.index.format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    const bool fp8_latent = f.cache.latent.storage == ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0;
    const int patterns = std::min(f.T, 7), count = patterns * f.B;
    std::vector<std::vector<double>> projected(count, std::vector<double>(w.rows));
    parallel_rows(count * w.rows, std::uint64_t(count) * w.rows * 4096, [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            const int row = i % w.rows, m = i / w.rows, b = m / patterns, t = m % patterns;
            double value = 0;
            for (int k = 0; k < 4096; ++k)
                value += double(kp::decode(w.weight.at(k, row))) * kp::decode(f.x.at(k, t, b));
            projected[m][row] = value;
        }
    });
    auto actual_q     = f.q.download();
    auto actual_coeff = f.coeff.download();
    auto actual_k = f.keys.download(), actual_g = f.gates.download();
    std::vector<std::uint16_t> cache(f.packed.storage().bytes / 2), state(f.state_host.size());
    CUDA_CHECK(cudaMemcpy(cache.data(), f.packed.storage().data, cache.size() * 2,
                          cudaMemcpyDeviceToHost));
    f.state.copy_to_host(state.data(), state.size() * 2);
    auto expected_cache = std::vector<std::uint16_t>(cache.size(), 0xa5a5);
    auto expected_state = f.state_host, expected_k = f.keys.host, expected_g = f.gates.host;
    KpoolError latent, query, raw_key, raw_gate, pooled, coeff, tail;
    int failures = 0;
    for (int b = 0; b < f.B; ++b) {
        const int n = f.valid.at(b), F = f.bases[b];
        std::vector<std::array<double, 128>> rk(F % 4 + n), rg(F % 4 + n);
        for (int r = 0; r < F % 4 && n; ++r)
            for (int d = 0; d < 128; ++d) {
                rk[r][d] = kp::decode(f.state_host[f.state_index(false, 0, b, r, d)]);
                rg[r][d] = kp::decode(f.state_host[f.state_index(false, 1, b, r, d)]);
            }
        for (int t = 0; t < f.T; ++t) {
            if (t >= n) {
                if (w.q)
                    for (int d = 0; d < w.q; ++d) failures += actual_q[f.q.index(d, t, b)] != 0;
                if (w.heads)
                    for (int d = 0; d < w.heads; ++d)
                        failures += actual_coeff[f.coeff.index(d, t, b)] != 0;
                continue;
            }
            const auto& z  = projected[b * patterns + t % patterns];
            const auto rms = [&](int first, int dimension, const auto& gamma, auto&& accept) {
                double sum = 0;
                for (int d = 0; d < dimension; ++d) sum += z[first + d] * z[first + d];
                const double inv = 1 / std::sqrt(sum / dimension + w.params.rms_epsilon);
                for (int d = 0; d < dimension; ++d)
                    accept(d, z[first + d] * inv * kp::decode(gamma.at(d)));
            };
            rms(0, 512, w.cn, [&](int d, double expected) {
                if (fp8_latent) {
                    const int token = F + t;
                    const auto page = f.tables.at(token / 64, f.selectors.at(b));
                    const auto code = page + f.cache.latent.data.plane.tensor.region.offset +
                                      (token % 64) * 512 + d;
                    const auto scale = page + f.cache.latent.scales.plane.tensor.region.offset +
                                       (token % 64) * 4 + d / 128;
                    const auto* got = reinterpret_cast<const std::uint8_t*>(cache.data());
                    auto* written   = reinterpret_cast<std::uint8_t*>(expected_cache.data());
                    failures += got[scale] == 255 || (got[code] & 127) == 127;
                    latent.add(float_codec::decode_e4m3fn(got[code]) *
                                   fp8_cache_multiplier(got[scale]),
                               expected);
                    written[code]  = got[code];
                    written[scale] = got[scale];
                } else {
                    const auto i = f.cache_index(0, b, F + t, d);
                    latent.add(kp::decode(cache[i]), expected);
                    expected_cache[i] = cache[i];
                }
            });
            if (w.q)
                rms(768, 1536, w.qn, [&](int d, double expected) {
                    query.add(kp::decode(actual_q[f.q.index(d, t, b)]), expected);
                });
            if (w.heads)
                for (int d = 0; d < 32; ++d)
                    coeff.add(actual_coeff[f.coeff.index(d, t, b)], z[2304 + d]);
            double mean = 0;
            for (int d = 0; d < 128; ++d) mean += z[512 + d];
            mean /= 128;
            double var = 0;
            for (int d = 0; d < 128; ++d) var += (z[512 + d] - mean) * (z[512 + d] - mean);
            var /= 128;
            for (int d = 0; d < 128; ++d) {
                const double key = (z[512 + d] - mean) / std::sqrt(var + w.params.index_epsilon) *
                                       kp::decode(w.kn.at(d)) +
                                   kp::decode(w.kb.at(d));
                const double gate = z[640 + d];
                const int r       = F % 4 + t;
                if (!f.update) {
                    const auto ki = f.keys.index(d, f.offsets.at(b) + t, b),
                               gi = f.gates.index(d, f.offsets.at(b) + t, b);
                    raw_key.add(kp::decode(actual_k[ki]), key);
                    raw_gate.add(kp::decode(actual_g[gi]), gate);
                    expected_k[ki] = actual_k[ki];
                    expected_g[gi] = actual_g[gi];
                    rk[r][d]       = kp::decode(actual_k[ki]);
                    rg[r][d]       = kp::decode(actual_g[gi]);
                } else {
                    rk[r][d] = kp::decode(kp::encode(float(key)));
                    rg[r][d] = kp::decode(kp::encode(float(gate)));
                }
            }
        }
        if (!n) continue;
        for (int group = 0; group < int(rk.size()) / 4; ++group) {
            std::array<double, 128> values;
            for (int d = 0; d < 128; ++d) {
                double maximum = -INFINITY, logits[4], sum = 0, value = 0;
                for (int r = 0; r < 4; ++r) {
                    logits[r] = rg[4 * group + r][d] + kp::decode(w.ape.at(d, r));
                    maximum   = std::max(maximum, logits[r]);
                }
                for (int r = 0; r < 4; ++r) {
                    const double a = std::exp(logits[r] - maximum);
                    sum += a;
                    value += a * rk[4 * group + r][d];
                }
                values[d] = value / sum;
            }
            if (fp8) {
                const auto rotated = index_hadamard_reference(values);
                const int pool     = F / 4 + group;
                const auto page    = f.tables.at(pool / 16, f.selectors.at(b));
                const auto codes =
                    page + f.cache.index.keys.plane.tensor.region.offset + (pool % 16) * 128;
                const auto scale =
                    page + f.cache.index.scales.plane.tensor.region.offset + pool % 16;
                const auto* got = reinterpret_cast<const std::uint8_t*>(cache.data());
                auto* expected  = reinterpret_cast<std::uint8_t*>(expected_cache.data());
                failures += got[scale] == 255;
                const double multiplier = fp8_cache_multiplier(got[scale]);
                expected[scale]         = got[scale];
                for (int d = 0; d < 128; ++d) {
                    failures += (got[codes + d] & 127) == 127;
                    pooled.add(float_codec::decode_e4m3fn(got[codes + d]) * multiplier, rotated[d]);
                    expected[codes + d] = got[codes + d];
                }
            } else {
                for (int d = 0; d < 128; ++d) {
                    const auto i = f.cache_index(1, b, F / 4 + group, d);
                    pooled.add(kp::decode(cache[i]), values[d]);
                    expected_cache[i] = cache[i];
                }
            }
        }
        if (f.update) {
            const int rem = (F + n) % 4, start = int(rk.size()) - rem;
            for (int r = 0; r < 3; ++r)
                for (int d = 0; d < 128; ++d)
                    for (int kind = 0; kind < 2; ++kind) {
                        const auto i = f.state_index(true, kind, b, r, d);
                        if (r < rem)
                            tail.add(kp::decode(state[i]),
                                     kind ? rg[start + r][d] : rk[start + r][d]);
                        else
                            failures += state[i] != 0;
                        expected_state[i] = state[i];
                    }
        }
    }
    failures +=
        latent.check("latent", fp8_latent ? kPreparedFp8LatentCriterion : kPreparedBf16Criterion) +
        query.check("query", kPreparedBf16Criterion) +
        raw_key.check("raw key", kPreparedBf16Criterion) +
        raw_gate.check("raw gate", kPreparedBf16Criterion);
    // Record exposes the actual raw operands; update's reference includes their encoding.
    failures += pooled.check("pool", fp8        ? kPooledFp8Criterion
                                     : f.update ? kPooledBf16Criterion
                                                : kPreparedBf16Criterion) +
                coeff.check("coeff", kCoefficientCriterion) +
                tail.check("tail", kPreparedBf16Criterion);
    failures += exact(cache, expected_cache, "cache") + exact(state, expected_state, "state") +
                exact(actual_k, expected_k, "raw keys") + exact(actual_g, expected_g, "raw gates");
    for (const auto* v : {&f.x, &f.q, &f.keys, &f.gates}) failures += !v->guards();
    failures += !f.coeff.guards() || !f.tables.guards();
    const auto capacity = f.update ? ops::kpool_mla_prepare_update_workspace_capacity_bytes
                                   : ops::kpool_mla_prepare_record_workspace_capacity_bytes;
    failures +=
        f.workspace->used() != 0 || f.workspace->peak_used() > f.workspace_bytes ||
        f.workspace->peak_used() !=
            capacity(w.profile(), f.cache.latent.storage, f.cache.index.format, f.B, f.T, f.T,
                     cuda_environment().context.execution_view().multiprocessor_count());
    return failures;
}

int run(kp::Projection mode, int T, int B, bool update, bool graph, int phase, bool fp8 = false,
        bool fp8_latent = false) {
    auto& device = cuda_environment().context;
    kp::Weights weights(device.device(), mode);
    kp::Fixture f(weights, device.execution_view(), T, B, update, true, true, update, phase,
                  fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16,
                  fp8_latent ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                             : ops::KpoolMlaStorage::BFloat16);
    // Seven independent activation vectors keep long-window oracle cost bounded. Every output
    // column is still checked, including page placements, padding, and tail/record write sets.
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < 4096; ++k)
                f.x.at(k, t, b) = kp::sample(k + 4096 * (t % 7) + 9397 * b + 317, 1.F / 256);
    f.x.upload();
    if (graph) {
        DecodeGraphDefinition capture;
        capture.capture(device.stream(), [&] { f.launch(device.execution_view()); });
        DecodeGraphExecutable graph;
        graph.instantiate(capture);
        const auto stride = std::int64_t(f.state_layout.image_bytes + 256);
        for (int b = 0; b < B; ++b)
            if (f.valid.at(b)) {
                const auto old = f.before_offsets.at(b), next = old + B * stride;
                std::copy_n(f.state_host.begin() + old / 2, f.state_layout.image_bytes / 2,
                            f.state_host.begin() + next / 2);
                f.before_offsets.at(b) = next;
                if (update) f.after_offsets.at(b) = next;
                std::swap(f.tables.at(0, f.selectors.at(b)), f.tables.at(1, f.selectors.at(b)));
            }
        f.valid.at(0) = std::max(1, f.valid.at(0) - 1);
        f.valid.upload();
        f.before_offsets.upload();
        f.after_offsets.upload();
        f.tables.upload();
        f.state.copy_from_host(f.state_host.data(), f.state.size());
        graph.launch(device.stream());
        device.synchronize();
    } else {
        f.launch(device.execution_view());
        device.synchronize();
    }
    return verify(f);
}

int inert(bool update, bool fp8 = false, bool fp8_latent = false) {
    auto& device = cuda_environment().context;
    kp::Weights weights(device.device(), kp::Projection::Full);
    kp::Fixture f(weights, device.execution_view(), 9, 3, update, false, true, update, 0,
                  fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16,
                  fp8_latent ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                             : ops::KpoolMlaStorage::BFloat16);
    for (int b = 0; b < f.B; ++b) {
        f.valid.at(b)     = 0;
        f.selectors.at(b) = f.offsets.at(b) = -1;
        f.before_offsets.at(b) = f.after_offsets.at(b) = -1;
        for (int t = 0; t < f.T; ++t) f.positions.at(t, b) = -1;
    }
    for (auto* v : {&f.valid, &f.selectors, &f.offsets, &f.positions}) v->upload();
    f.before_offsets.upload();
    f.after_offsets.upload();
    f.launch(device.execution_view());
    device.synchronize();
    return verify(f);
}

int admission() {
    auto& device = cuda_environment().context;
    kp::Weights weights(device.device(), kp::Projection::Full);
    kp::Fixture f(weights, device.execution_view(), 1, 1, false);
    const auto rejects = [&](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return 0; }
        return 1;
    };
    weights.params.rms_epsilon = 0;
    int failures               = rejects([&] { f.launch(device.execution_view()); });
    weights.params.rms_epsilon = 1e-6F;
    f.x.view.ne[0]             = 4095;
    failures += rejects([&] { f.launch(device.execution_view()); });
    kp::Fixture latent(weights, device.execution_view(), 4, 2, false, false, false, false, 0,
                       KvCacheVectorFormat::BFloat16, ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0);
    const auto latent_scales   = latent.cache.latent.scales;
    latent.cache.latent.scales = {};
    failures += rejects([&] { latent.launch(device.execution_view()); });
    latent.cache.latent.scales = latent_scales;
    latent.cache.latent.scales.plane.tensor.region.offset =
        latent.cache.index.keys.plane.tensor.region.offset;
    failures += rejects([&] { latent.launch(device.execution_view()); });
    kp::Fixture fp(weights, device.execution_view(), 4, 2, false, false, false, false, 0,
                   KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0);
    const auto scales     = fp.cache.index.scales;
    fp.cache.index.scales = {};
    failures += rejects([&] { fp.launch(device.execution_view()); });
    fp.cache.index.scales = scales;
    fp.cache.index.scales.plane.tensor.region.offset =
        fp.cache.index.keys.plane.tensor.region.offset;
    failures += rejects([&] { fp.launch(device.execution_view()); });
    fp.cache.index.scales = scales;
    fp.cache.index.scales.records.page_offsets.nb[1] += 8;
    failures += rejects([&] { fp.launch(device.execution_view()); });
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("kpool_prepare/admission", admission);
    for (bool update : {false, true})
        suite.add(std::string("kpool_prepare/inert/") + (update ? "update" : "record"),
                  [=] { return inert(update); });
    const auto add = [&](kp::Projection mode, int T, int B, bool update) {
        const auto id = std::string("kpool_prepare/") + kp::name(mode) +
                        (update ? "/update/" : "/record/") + std::to_string(T) + "x" +
                        std::to_string(B);
        suite.add(id, [=] { return run(mode, T, B, update, T == 513, T % 4); },
                  {{"format", "bf16"}, {"execution", T == 513 ? "graph" : "eager"}});
    };
    for (auto mode :
         {kp::Projection::Full, kp::Projection::QueryAndHistory, kp::Projection::History})
        for (bool update : {false, true})
            for (auto [T, B] : {std::pair{1, 1},
                                {2, 1},
                                {4, 1},
                                {8, 1},
                                {4, 8},
                                {9, 1},
                                {17, 1},
                                {65, 1},
                                {129, 1},
                                {256, 8},
                                {257, 1},
                                {257, 3},
                                {513, 2},
                                {513, 3},
                                {2049, 1}})
                add(mode, T, B, update);
    // Complete the selector/window neighbors without multiplying them by every state form.
    for (auto mode : {kp::Projection::Full, kp::Projection::History})
        for (int T : {3, 7, 15, 16, 63, 64, 127, 128, 1023, 1024, 1025})
            add(mode, T, 1, mode == kp::Projection::History);
    for (int T : {511, 512, 513}) add(kp::Projection::History, T, 1, true);
    for (bool update : {false, true}) {
        suite.add(std::string("kpool_prepare/fp8/inert/") + (update ? "update" : "record"),
                  [=] { return inert(update, true); }, {{"format", "fp8"}});
        for (auto mode :
             {kp::Projection::Full, kp::Projection::QueryAndHistory, kp::Projection::History})
            for (auto [T, B] : {std::pair{2, 1}, {4, 1}, {17, 3}, {513, 3}}) {
                const auto id = std::string("kpool_prepare/fp8/") + kp::name(mode) +
                                (update ? "/update/" : "/record/") + std::to_string(T) + "x" +
                                std::to_string(B);
                suite.add(id, [=] { return run(mode, T, B, update, T == 513, T % 4, true); },
                          {{"format", "fp8"}, {"execution", T == 513 ? "graph" : "eager"}});
            }
    }
    for (bool index_fp8 : {false, true}) {
        const auto prefix =
            std::string("kpool_prepare/fp8_latent/") + (index_fp8 ? "fp8_index/" : "bf16_index/");
        for (bool update : {false, true}) {
            suite.add(prefix + "inert/" + (update ? "update" : "record"),
                      [=] { return inert(update, index_fp8, true); }, {{"format", "fp8"}});
            for (auto mode :
                 {kp::Projection::Full, kp::Projection::QueryAndHistory, kp::Projection::History})
                for (auto [T, B] : {std::pair{1, 1}, {4, 1}, {17, 3}, {513, 3}})
                    suite.add(
                        prefix + kp::name(mode) + (update ? "/update/" : "/record/") +
                            std::to_string(T) + "x" + std::to_string(B),
                        [=] { return run(mode, T, B, update, T == 513, T % 4, index_fp8, true); },
                        {{"format", "fp8"}, {"execution", T == 513 ? "graph" : "eager"}});
        }
    }
    return suite.main(argc, argv);
}
