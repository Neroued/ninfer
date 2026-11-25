#include "reference.h"
#include "ops/ragged_fixture.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"
#include "../../../../tools/support/kpool_prepare_fixture.h"

#include <cmath>
#include <iostream>
#include <tuple>

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

int verify(kp::Fixture& f, bool check_workspace = true) {
    const auto& w         = f.w;
    const bool fp8        = f.cache.index.format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    const bool fp8_latent = f.cache.latent.storage == ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0;
    const int patterns = std::min(f.T, 7), count = patterns * f.B;
    const auto weight_value = [&](int row, int k) {
        if (!w.encoded) return double(kp::decode(w.weight.at(k, row)));
        const auto& map = w.params.rows;
        const std::array roles{map.latent, map.index_key, map.pool_gate, map.query,
                               map.coefficients};
        constexpr int origins[]{0, 512, 640, 768, 2304};
        for (int i = 0; i < 5; ++i)
            if (row >= origins[i] && row < origins[i] + roles[i].rows) {
                const int physical_row = roles[i].row_begin + row - origins[i];
                return roles[i].parent == 0 ? represented_weight(*w.encoded, physical_row, k)
                                            : double(kp::decode(w.auxiliary->at(k, physical_row)));
            }
        throw std::logic_error("missing reference projection role");
    };
    std::vector<std::vector<double>> projected(count, std::vector<double>(w.rows));
    parallel_rows(count * w.rows, std::uint64_t(count) * w.rows * 4096, [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
            const int row = i % w.rows, m = i / w.rows, b = m / patterns, t = m % patterns;
            if (t >= f.valid.at(b)) continue;
            double value = 0;
            for (int k = 0; k < 4096; ++k)
                value += weight_value(row, k) * kp::decode(f.x.at(k, t, b));
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
    const bool a8 =
        w.encoded &&
        ops::allows_a8(std::get<ops::PairedProjectionWeights>(w.params.weights).first.policy);
    const ReductionCriterion prepared_a8{.045, 1e-4, .075};
    const ReductionCriterion encoded_a8{.085, 1e-4, .13};
    const auto prepared = a8 ? prepared_a8 : kPreparedBf16Criterion;
    failures += latent.check("latent", fp8_latent ? (a8 ? encoded_a8 : kPreparedFp8LatentCriterion)
                                                  : prepared) +
                query.check("query", prepared) + raw_key.check("raw key", prepared) +
                raw_gate.check("raw gate", kPreparedBf16Criterion);
    // Record exposes the actual raw operands; update's reference includes their encoding.
    failures +=
        pooled.check("pool", a8         ? (fp8 ? encoded_a8 : prepared_a8)
                             : fp8      ? kPooledFp8Criterion
                             : f.update ? kPooledBf16Criterion
                                        : kPreparedBf16Criterion) +
        coeff.check("coeff", a8 ? ReductionCriterion{.04, 1e-6, .06} : kCoefficientCriterion) +
        tail.check("tail", prepared);
    failures += exact(cache, expected_cache, "cache") + exact(state, expected_state, "state") +
                exact(actual_k, expected_k, "raw keys") + exact(actual_g, expected_g, "raw gates");
    for (const auto* v : {&f.x, &f.q, &f.keys, &f.gates}) failures += !v->guards();
    failures += !f.coeff.guards() || !f.tables.guards();
    if (w.encoded)
        failures += !w.encoded->unchanged() || w.auxiliary->download() != w.auxiliary->host;
    const auto capacity = f.update ? ops::kpool_mla_prepare_update_workspace_capacity_bytes
                                   : ops::kpool_mla_prepare_record_workspace_capacity_bytes;
    if (check_workspace)
        failures +=
            f.workspace->used() != 0 || f.workspace->peak_used() > f.workspace_bytes ||
            f.workspace->peak_used() !=
                capacity(w.profile(), f.cache.latent.storage, f.cache.index.format, f.B, f.T, f.T,
                         cuda_environment().context.execution_view().multiprocessor_count());
    return failures;
}

int run(kp::Projection mode, int T, int B, bool update, bool graph, int phase, bool fp8 = false,
        bool fp8_latent = false, kp::ProjectionFormat format = kp::ProjectionFormat::BFloat16,
        bool a16_only = false, bool padded = false) {
    auto& device = cuda_environment().context;
    kp::Weights weights(device.device(), mode, format);
    if (a16_only)
        std::get<ops::PairedProjectionWeights>(weights.params.weights).first.policy =
            ops::LinearPolicy::A16Only;
    kp::Fixture f(weights, device.execution_view(), T, B, update, !padded, true, update, phase,
                  fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16,
                  fp8_latent ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                             : ops::KpoolMlaStorage::BFloat16);
    if (padded) {
        for (int b = 0; b < B; ++b) {
            if (b == 0 || b == B / 2 || b == B - 1) continue;
            f.valid.at(b)     = 0;
            f.selectors.at(b) = f.offsets.at(b) = -1;
            f.before_offsets.at(b) = f.after_offsets.at(b) = -1;
            for (int t = 0; t < T; ++t) f.positions.at(t, b) = -1;
        }
        for (auto* v : {&f.valid, &f.selectors, &f.offsets, &f.positions}) v->upload();
        f.before_offsets.upload();
        f.after_offsets.upload();
    }
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

enum class TailAlias { Disjoint, InPlace, Mixed };

int ragged_prepare(std::vector<int> lengths, kp::Projection projection, kp::ProjectionFormat format,
                   bool fp8, bool a16_only, TailAlias alias, int phase, bool replay = false) {
    RaggedFixture ragged(std::move(lengths));
    const int width = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    const int n = ragged.tokens(), s = ragged.segments();
    const auto execution = cuda_environment().context.execution_view();
    kp::Weights weights(execution.device(), projection, format);
    if (a16_only)
        std::get<ops::PairedProjectionWeights>(weights.params.weights).first.policy =
            ops::LinearPolicy::A16Only;
    kp::Fixture f(
        weights, execution, width, s, true, false, true, alias == TailAlias::InPlace, phase,
        fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16,
        fp8 ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0 : ops::KpoolMlaStorage::BFloat16);
    if (alias == TailAlias::Mixed)
        for (int b = 0; b < s; b += 2) f.after_offsets.at(b) = f.before_offsets.at(b);
    f.after_offsets.upload();
    f.valid.host = ragged.lengths;
    f.valid.upload();
    const auto bytes = ops::kpool_mla_prepare_update_ragged_workspace_capacity_bytes(
        weights.profile(), f.cache.latent.storage, f.cache.index.format, n, n, s, s,
        execution.multiprocessor_count());
    GuardedDeviceBuffer storage(bytes + 2ULL * n * (4096 + weights.q) +
                                4ULL * n * (1 + weights.heads) + 4 * 256);
    MemoryArena arena(storage.span());
    const auto x      = arena.alloc(DType::BF16, {4096, n});
    const auto pos    = arena.alloc(DType::I32, {n});
    const auto q      = weights.q ? arena.alloc(DType::BF16, {weights.q, n}) : Tensor{};
    const auto coeff  = weights.heads ? arena.alloc(DType::FP32, {weights.heads, n}) : Tensor{};
    const auto prefix = arena.used();
    const auto aligned_prefix = (prefix + 255) / 256 * 256;
    const ops::KpoolMlaRaggedBatchView batch{ragged.view, pos, f.selectors.view};
    const auto fill = [&](int seed) {
        for (int b = 0; b < s; ++b)
            for (int t = 0; t < ragged.lengths[b]; ++t)
                for (int k = 0; k < 4096; ++k)
                    f.x.at(k, t, b) = kp::sample(k + 4096 * (t % 7) + 9397 * b + seed, 1.F / 256);
        const auto x_host    = ragged.pack(f.x.host, width, 4096);
        const auto positions = ragged.pack(f.positions.host, width, 1);
        CUDA_CHECK(cudaMemcpy(x.data, x_host.data(), x.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pos.data, positions.data(), pos.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    };
    const auto launch = [&](MemoryArena& scratch) {
        ops::kpool_mla_prepare_update(x, weights.params, batch, f.before, f.cache, f.after,
                                      {q, coeff}, scratch, execution);
    };
    const auto check = [&] {
        cuda_environment().context.synchronize();
        if (q.data) {
            const auto padded =
                ragged.unpack(from_device<std::uint16_t>(q.data, 1536 * n), width, 1536);
            f.q.storage.copy_from_host(padded.data(), padded.size() * 2, 256);
        }
        if (coeff.data) {
            const auto padded = ragged.unpack(from_device<float>(coeff.data, 32 * n), width, 32);
            f.coeff.storage.copy_from_host(padded.data(), padded.size() * 4, 256);
        }
        return verify(f, false) + storage.verify_guards("ragged prepare arena") +
               (arena.used() != prefix || arena.peak_used() != aligned_prefix + bytes);
    };
    fill(317);
    int failures = 0;
    if (replay) {
        MemoryArena overlapping(storage.span());
        bool rejected = false;
        try {
            launch(overlapping);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += !rejected || overlapping.used() != 0;
        DecodeGraphDefinition capture;
        capture.capture(execution.stream(), [&] { launch(arena); });
        DecodeGraphExecutable graph;
        graph.instantiate(capture);
        graph.launch(execution.stream());
        failures += check();

        CUDA_CHECK(cudaMemset(f.packed.storage().data, 0xa5, f.packed.storage().bytes));
        const auto stride = std::int64_t(f.state_layout.image_bytes + 256);
        for (int b = 0; b < s; ++b) {
            const auto delta = s * stride;
            f.before_offsets.at(b) += delta;
            f.after_offsets.at(b) +=
                f.after_offsets.at(b) == f.before_offsets.at(b) ? -delta : delta;
            f.selectors.at(b) = b;
            ++f.bases[b];
            for (int t = 0; t < ragged.lengths[b]; ++t) f.positions.at(t, b) = f.bases[b] + t;
            for (int kind = 0; kind < 2; ++kind)
                for (int r = 0; r < 3; ++r)
                    for (int d = 0; d < 128; ++d)
                        f.state_host[f.state_index(false, kind, b, r, d)] =
                            r < f.bases[b] % 4
                                ? kp::sample(d + 128 * r + 777 * b + 99 * kind, 1.F / 512)
                                : 0;
        }
        f.selectors.upload();
        f.before_offsets.upload();
        f.after_offsets.upload();
        f.state.copy_from_host(f.state_host.data(), f.state.size());
        fill(731);
        graph.launch(execution.stream());
        failures += check();
    } else {
        launch(arena);
        failures += check();
    }
    return failures;
}

int ragged_prepare_capacity() {
    const auto execution = cuda_environment().default_execution;
    kp::Weights mx(execution.device(), kp::Projection::History, kp::ProjectionFormat::Mxfp8);
    int failures                = 0;
    auto mx_a16                 = mx.profile();
    mx_a16.weights.first.policy = ops::LinearPolicy::A16Only;
    for (const auto profile :
         {kp::bf16_prepare_profile(kp::Projection::History),
          kp::bf16_prepare_profile(kp::Projection::Full), mx.profile(), mx_a16}) {
        const auto bytes = [&](int lo, int hi, int min_segments, int max_segments) {
            return ops::kpool_mla_prepare_update_ragged_workspace_capacity_bytes(
                profile, ops::KpoolMlaStorage::BFloat16, KvCacheVectorFormat::BFloat16, lo, hi,
                min_segments, max_segments, execution.multiprocessor_count());
        };
        // A moved cut may produce any of these widths. Independently scan the
        // rectangular per-width capacities, including the short split routes.
        std::size_t one_window_peak = 0;
        for (int n = 1; n <= 1024; ++n)
            one_window_peak =
                std::max(one_window_peak,
                         ops::kpool_mla_prepare_update_workspace_capacity_bytes(
                             profile, ops::KpoolMlaStorage::BFloat16, KvCacheVectorFormat::BFloat16,
                             1, n, n, execution.multiprocessor_count()));
        failures += bytes(1025, 1025, 1025, 1025) != one_window_peak;
        for (const auto [lo, hi, first, last] : {std::array{1, 133, 1, 5},
                                                 {1000, 1160, 2, 4},
                                                 {1019, 1031, 1025, 1029},
                                                 {2045, 2055, 1, 3},
                                                 {2199, 3350, 2, 3}}) {
            std::size_t peak = 0;
            for (int n = lo; n <= hi; ++n)
                for (int s = first; s <= std::min(last, n); ++s)
                    peak = std::max(peak, bytes(n, n, s, s));
            failures += peak != bytes(lo, hi, first, last);
        }
        failures += bytes(1025, 1025, 2, 2) != bytes(1025, 1025, 1025, 1025) + 3072;
    }
    return failures;
}

int inert(bool update, bool fp8 = false, bool fp8_latent = false,
          kp::ProjectionFormat format = kp::ProjectionFormat::BFloat16, int tokens = 9,
          int batch = 3) {
    auto& device = cuda_environment().context;
    kp::Weights weights(device.device(), kp::Projection::Full, format);
    kp::Fixture f(weights, device.execution_view(), tokens, batch, update, false, true, update, 0,
                  fp8 ? KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 : KvCacheVectorFormat::BFloat16,
                  fp8_latent ? ops::KpoolMlaStorage::Fp8E4m3fnGroup128Ue8m0
                             : ops::KpoolMlaStorage::BFloat16);
    DecodeGraphDefinition capture;
    DecodeGraphExecutable graph;
    if (format != kp::ProjectionFormat::BFloat16) {
        capture.capture(device.stream(), [&] { f.launch(device.execution_view()); });
        graph.instantiate(capture);
    }
    for (int b = 0; b < f.B; ++b) {
        f.valid.at(b)     = 0;
        f.selectors.at(b) = f.offsets.at(b) = -1;
        f.before_offsets.at(b) = f.after_offsets.at(b) = -1;
        for (int t = 0; t < f.T; ++t) f.positions.at(t, b) = -1;
    }
    for (auto* v : {&f.valid, &f.selectors, &f.offsets, &f.positions}) v->upload();
    f.before_offsets.upload();
    f.after_offsets.upload();
    if (format != kp::ProjectionFormat::BFloat16)
        graph.launch(device.stream());
    else
        f.launch(device.execution_view());
    device.synchronize();
    return verify(f);
}

int workspace_interval() {
    int failures = 0;
    for (bool update : {false, true}) {
        const auto capacity = update ? ops::kpool_mla_prepare_update_workspace_capacity_bytes
                                     : ops::kpool_mla_prepare_record_workspace_capacity_bytes;
        for (auto [B, lo, hi] :
             {std::tuple{65, 1, 129}, {129, 1, 31}, {1025, 1023, 1025}, {65537, 1, 17}}) {
            const auto bytes = [&](int first, int last) {
                return capacity(kp::bf16_prepare_profile(kp::Projection::Full),
                                ops::KpoolMlaStorage::BFloat16, KvCacheVectorFormat::BFloat16, B,
                                first, last, 48);
            };
            std::size_t maximum = 0;
            for (int T = lo; T <= hi; ++T) maximum = std::max(maximum, bytes(T, T));
            failures += bytes(lo, hi) != maximum;
        }
    }
    return failures;
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
    suite.add("kpool_prepare/ragged/bf16_carry_then_complete", [] {
        return ragged_prepare({1025, 1024}, kp::Projection::Full, kp::ProjectionFormat::BFloat16,
                              false, false, TailAlias::Mixed, 1);
    });
    suite.add("kpool_prepare/ragged/fp8_carry_then_complete", [] {
        return ragged_prepare({1023, 1025, 1024}, kp::Projection::Full,
                              kp::ProjectionFormat::BlockFp8, true, false, TailAlias::Mixed, 1);
    });
    suite.add("kpool_prepare/ragged/bf16_ping_pong", [] {
        // A multiple of the window width forces both cuts. The middle window
        // reads carry for segment zero and writes it for segment one.
        return ragged_prepare({1025, 1024, 1023}, kp::Projection::Full,
                              kp::ProjectionFormat::BFloat16, false, false, TailAlias::Mixed, 1);
    });
    suite.add("kpool_prepare/ragged/block_fp8_replay", [] {
        return ragged_prepare({1023, 2, 1025}, kp::Projection::QueryAndHistory,
                              kp::ProjectionFormat::BlockFp8, true, false, TailAlias::Mixed, 1,
                              true);
    });
    suite.add("kpool_prepare/ragged/mxfp8_a16", [] {
        return ragged_prepare({3, 62, 63}, kp::Projection::History, kp::ProjectionFormat::Mxfp8,
                              false, true, TailAlias::Disjoint, 2);
    });
    suite.add("kpool_prepare/ragged/mxfp8_a8", [] {
        return ragged_prepare({513, 512}, kp::Projection::Full, kp::ProjectionFormat::Mxfp8, true,
                              false, TailAlias::InPlace, 3);
    });
    suite.add("kpool_prepare/ragged/split_history", [] {
        return ragged_prepare({1, 126, 1}, kp::Projection::History, kp::ProjectionFormat::BFloat16,
                              false, false, TailAlias::Disjoint, 1);
    });
    suite.add("kpool_prepare/ragged/capacity", ragged_prepare_capacity);
    suite.add("kpool_prepare/admission", admission);
    suite.add("kpool_prepare/large_batch/workspace_interval", workspace_interval,
              {{"suite", "concurrency"}});
    suite.add("kpool_prepare/large_batch/bf16_record",
              [] {
                  return run(kp::Projection::Full, 1, 9, false, true, 3, false, false,
                             kp::ProjectionFormat::BFloat16, false, true);
              },
              {{"suite", "concurrency"}, {"execution", "graph"}});
    suite.add("kpool_prepare/large_batch/fp8_update",
              [] {
                  return run(kp::Projection::Full, 4, 17, true, true, 3, true, true,
                             kp::ProjectionFormat::BlockFp8, false, true);
              },
              {{"suite", "concurrency"}, {"execution", "graph"}});
    suite.add("kpool_prepare/large_batch/mxfp8_window",
              [] {
                  return run(kp::Projection::Full, 9, 129, true, true, 3, true, true,
                             kp::ProjectionFormat::Mxfp8, false, true);
              },
              {{"suite", "concurrency"}, {"execution", "graph"}});
    suite.add("kpool_prepare/large_batch/inert",
              [] { return inert(false, true, true, kp::ProjectionFormat::Mxfp8, 1, 65); },
              {{"suite", "concurrency"}, {"execution", "graph"}});
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
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        for (auto mode :
             {kp::Projection::Full, kp::Projection::QueryAndHistory, kp::Projection::History})
            for (bool update : {false, true}) {
                const int T = update ? 17 : mode == kp::Projection::Full ? 1 : 9;
                suite.add(
                    std::string("kpool_prepare/native/") + kp::name(format) + "/" + kp::name(mode) +
                        (update ? "/update" : "/record"),
                    [=] { return run(mode, T, 3, update, update, 3, update, !update, format); },
                    {{"format", kp::name(format)}, {"execution", update ? "graph" : "eager"}});
            }
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        for (auto mode :
             {kp::Projection::Full, kp::Projection::QueryAndHistory, kp::Projection::History})
            for (bool update : {false, true})
                suite.add(std::string("kpool_prepare/native/") + kp::name(format) + "/decode/" +
                              kp::name(mode) + (update ? "/update" : "/record"),
                          [=] { return run(mode, 4, 5, update, true, 3, true, true, format); },
                          {{"format", kp::name(format)}, {"execution", "graph"}});
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        for (auto [T, B] : {std::pair{5, 3}, {6, 5}})
            suite.add(
                std::string("kpool_prepare/native/") + kp::name(format) + "/decode/" +
                    std::to_string(T) + "x" + std::to_string(B),
                [=] { return run(kp::Projection::Full, T, B, true, true, 3, true, true, format); },
                {{"format", kp::name(format)}, {"execution", "graph"}});
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        suite.add(
            std::string("kpool_prepare/native/") + kp::name(format) + "/window",
            [=] { return run(kp::Projection::Full, 1025, 1, true, false, 3, true, true, format); },
            {{"format", kp::name(format)}});
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        suite.add(
            std::string("kpool_prepare/native/") + kp::name(format) + "/history_short",
            [=] { return run(kp::Projection::History, 4, 1, false, false, 3, true, true, format); },
            {{"format", kp::name(format)}});
    for (auto mode : {kp::Projection::QueryAndHistory, kp::Projection::History})
        suite.add(std::string("kpool_prepare/native/mxfp8/tiles/") + kp::name(mode),
                  [=] {
                      return run(mode, 256, 1, true, false, 3, true, true,
                                 kp::ProjectionFormat::Mxfp8);
                  },
                  {{"format", "mxfp8"}});
    for (auto [T, B] : {std::pair{33, 1}, {6, 8}})
        for (bool update : {false, true})
            suite.add("kpool_prepare/native/mxfp8/query-history/" + std::to_string(T) + "x" +
                          std::to_string(B) + (update ? "/update" : "/record"),
                      [=] {
                          return run(kp::Projection::QueryAndHistory, T, B, update, true, 3, true,
                                     true, kp::ProjectionFormat::Mxfp8);
                      },
                      {{"format", "mxfp8"}, {"execution", "graph"}});
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        for (auto [T, B] : {std::pair{128, 1}, {193, 1}, {257, 1}, {4096, 1}, {4097, 1}, {1025, 8}})
            suite.add(
                std::string("kpool_prepare/native/") + kp::name(format) + "/extended/" +
                    std::to_string(T) + "x" + std::to_string(B),
                [=] { return run(kp::Projection::Full, T, B, true, true, 3, true, true, format); },
                {{"format", kp::name(format)}, {"execution", "graph"}});
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        for (bool update : {false, true})
            suite.add(std::string("kpool_prepare/native/") + kp::name(format) + "/inert/" +
                          (update ? "update" : "record"),
                      [=] { return inert(update, true, true, format); },
                      {{"format", kp::name(format)}, {"execution", "graph"}});
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        for (bool update : {false, true})
            suite.add(std::string("kpool_prepare/native/") + kp::name(format) + "/inert_t1/" +
                          (update ? "update" : "record"),
                      [=] { return inert(update, true, true, format, 1); },
                      {{"format", kp::name(format)}, {"execution", "graph"}});
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        for (int tokens : {17, 129})
            suite.add(std::string("kpool_prepare/native/") + kp::name(format) + "/inert-t" +
                          std::to_string(tokens),
                      [=] { return inert(true, true, true, format, tokens); },
                      {{"format", kp::name(format)}, {"execution", "graph"}});
    for (auto mode :
         {kp::Projection::Full, kp::Projection::QueryAndHistory, kp::Projection::History})
        suite.add(std::string("kpool_prepare/native/mxfp8/short-boundary/") + kp::name(mode),
                  [=] {
                      return run(mode, mode == kp::Projection::Full ? 49 : 65, 1, true, true, 3,
                                 true, true, kp::ProjectionFormat::Mxfp8);
                  },
                  {{"format", "mxfp8"}, {"execution", "graph"}});
    for (auto format : {kp::ProjectionFormat::BlockFp8, kp::ProjectionFormat::Mxfp8})
        for (bool update : {false, true})
            suite.add(std::string("kpool_prepare/native/") + kp::name(format) + "/single/" +
                          (update ? "update" : "record"),
                      [=] {
                          return run(kp::Projection::Full, 1, 1, update, true, 3, true, true,
                                     format);
                      },
                      {{"format", kp::name(format)}, {"execution", "graph"}});
    for (auto mode :
         {kp::Projection::Full, kp::Projection::QueryAndHistory, kp::Projection::History})
        suite.add(std::string("kpool_prepare/native/mxfp8/aux_record/") + kp::name(mode),
                  [=] {
                      return run(mode, 512, 3, false, true, 3, true, true,
                                 kp::ProjectionFormat::Mxfp8);
                  },
                  {{"format", "mxfp8"}, {"execution", "graph"}});
    suite.add("kpool_prepare/native/mxfp8/a16",
              [] {
                  return run(kp::Projection::Full, 65, 1, true, false, 3, false, false,
                             kp::ProjectionFormat::Mxfp8, true);
              },
              {{"format", "mxfp8"}});
    return suite.main(argc, argv);
}
