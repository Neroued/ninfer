#include "core/platform.h"
#include "ops/suite.h"
#include "packed_fixture.h"
#include "kv_cache_storage.h"
#include "prepare_fixture.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"

#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::qsa::prepare;

namespace {

// A final BF16 rounding has at most 1/256 relative error. The small additional budget covers
// FP32 normalization/rotation; the pair-scaled cap remains meaningful at cancellation points.
constexpr double kRelativeL2 = 0.0041, kPairRelative = 0.0069;

struct Ideal {
    std::vector<double> values, scales;
};

using RopeRotation = std::array<std::pair<double, double>, 32>;

RopeRotation rope_rotation(std::array<int, 3> rho, float theta) {
    RopeRotation rotation;
    for (int i = 0; i < 32; ++i) {
        const double phase = double(rho[i % 3]) * std::pow(double(theta), -double(i) / 32);
        rotation[i]        = {std::cos(phase), std::sin(phase)};
    }
    return rotation;
}

Ideal oracle(const std::vector<double>& raw, const Buffer<std::uint16_t>& weight,
             const RopeRotation& rotation, const ops::QsaKvPrepareParameters& params) {
    const int d = raw.size();
    double sum  = 0;
    for (double x : raw) sum += x * x;
    const double inverse = 1 / std::sqrt(sum / d + double(params.rms_epsilon));
    Ideal out{std::vector<double>(d), std::vector<double>(d)};
    for (int c = 0; c < d; ++c) {
        out.values[c] = raw[c] * inverse * (1 + double(bf16_to_f32(weight.at(c))));
        out.scales[c] = std::abs(out.values[c]);
    }
    for (int i = 0; i < 32; ++i) {
        const auto [cosine, sine] = rotation[i];
        const double x = out.values[i], y = out.values[i + 32];
        out.values[i]      = x * cosine - y * sine;
        out.values[i + 32] = y * cosine + x * sine;
        out.scales[i] = out.scales[i + 32] = std::hypot(x, y);
    }
    return out;
}

struct Error {
    double sum = 0, reference = 0, max_pair = 0;
    bool finite = true;

    void add(std::uint16_t word, double expected, double scale) {
        const double actual = bf16_to_f32(word), error = actual - expected;
        finite &= std::isfinite(actual) && std::isfinite(expected);
        sum += error * error;
        reference += expected * expected;
        max_pair = std::max(max_pair, error_limit_ratio(std::abs(error), scale));
    }

    void merge(const Error& other) {
        sum += other.sum;
        reference += other.reference;
        max_pair = std::max(max_pair, other.max_pair);
        finite &= other.finite;
    }

    int check(const std::string& label) const {
        const double l2 = reference == 0 ? (sum == 0 ? 0 : INFINITY) : std::sqrt(sum / reference);
        const bool pass = finite && l2 <= kRelativeL2 && max_pair <= kPairRelative;
        if (!pass || error_stats_enabled())
            std::cout << label << " relative_l2=" << l2 << " max_pair=" << max_pair << '\n';
        return !pass;
    }
};

int verify(Fixture& f, const std::string& label, bool updates_tail = false) {
    const auto actual_q  = f.kv_only
                               ? std::vector<std::uint16_t>{}
                               : from_device<std::uint16_t>(f.query_output.data, f.q.host.size());
    const auto actual_iq = f.kv_only
                               ? std::vector<std::uint16_t>{}
                               : from_device<std::uint16_t>(f.iq_output.data, f.iq.host.size());
    const auto actual_k = f.keys.download(), actual_ks = f.key_scales.download();
    const auto actual_ik = f.compressed.download();
    auto expected_k = f.keys.host, expected_v = f.values.host;
    auto expected_ks = f.key_scales.host, expected_vs = f.value_scales.host;
    auto expected_ik   = f.compressed.host;
    auto expected_raw  = f.records.host;
    auto expected_rope = f.record_rope.host;
    Error q_error, k_error, iq_error, ik_error;
    int failures           = 0;
    const auto coordinates = [&](int t, int b) {
        return std::array{f.rope.at(t, 0, b), f.rope.at(t, f.A == 1 ? 0 : 1, b),
                          f.rope.at(t, f.A == 1 ? 0 : 2, b)};
    };
    const auto ideal_head = [&](const auto& input, int h, int t, int b, const auto& weight,
                                const RopeRotation& rotation) {
        std::vector<double> raw(input.view.ne[0]);
        for (int c = 0; c < input.view.ne[0]; ++c) raw[c] = bf16_to_f32(input.at(c, h, t, b));
        return oracle(raw, weight, rotation, f.parameters.kv);
    };
    const auto head = [&](const auto& input, int h, int t, int b, const auto& weight,
                          const auto& actual, std::size_t offset, Error& error,
                          const RopeRotation& rotation) {
        const auto ideal = ideal_head(input, h, t, b, weight, rotation);
        for (int c = 0; c < input.view.ne[0]; ++c)
            error.add(actual[offset + c], ideal.values[c], ideal.scales[c]);
    };

    struct TokenErrors {
        Error q, k, iq;
        int failures = 0;
    };

    std::vector<TokenErrors> token_errors(f.B * f.T);
    // Table rows use disjoint physical page ranges and each token owns its cache/record row.
    parallel_rows(
        f.B * f.T, std::uint64_t(f.B) * f.T * f.query_heads * 256, [&](int first, int last) {
            for (int row = first; row < last; ++row) {
                const int b = row / f.T, t = row % f.T, n = f.valid.at(b);
                auto& [q_error, k_error, iq_error, failures] = token_errors[row];
                if (t >= n) {
                    if (f.kv_only) continue;
                    for (int h = 0; h < f.query_heads; ++h)
                        for (int c = 0; c < 256; ++c)
                            failures += actual_q[f.q.index(c, h, t, b)] != 0;
                    for (int h = 0; h < 4; ++h)
                        for (int c = 0; c < 128; ++c)
                            failures += actual_iq[f.iq.index(c, h, t, b)] != 0;
                    continue;
                }
                const auto rho      = coordinates(t, b);
                const auto rotation = rope_rotation(rho, f.parameters.kv.rope_theta);
                if (!f.kv_only) {
                    for (int h = 0; h < f.query_heads; ++h)
                        head(f.q, h, t, b, f.wq, actual_q, f.q.index(0, h, t, b), q_error,
                             rotation);
                    for (int h = 0; h < 4; ++h)
                        head(f.iq, h, t, b, f.wiq, actual_iq, f.iq.index(0, h, t, b), iq_error,
                             rotation);
                }
                const int p = f.positions.at(t, b), page = f.tables.at(p / 64, f.table_rows.at(b));
                for (int h = 0; h < f.kv_heads; ++h) {
                    const auto ideal = ideal_head(f.k, h, t, b, f.wk, rotation);
                    if (f.storage == KvCacheStorage::BFloat16) {
                        for (int c = 0; c < 256; ++c) {
                            const auto index = f.keys.index(c, p % 64, h, page);
                            const auto word  = read_u16(actual_k, index);
                            k_error.add(word, ideal.values[c], ideal.scales[c]);
                            // Live approximate values were checked above; every other byte is
                            // exact.
                            write_u16(expected_k, index, word);
                            write_u16(expected_v, f.values.index(c, p % 64, h, page),
                                      fp16_encode(bf16_to_f32(f.v.at(c, h, t, b))));
                        }
                    } else {
                        const auto row = [&](const auto& data, const CachePlane& plane) {
                            const auto first = data.begin() + plane.index(0, p % 64, h, page);
                            return std::vector<std::uint8_t>(
                                first, first + plane.view.ne[0] * dtype_size(plane.view.dtype));
                        };
                        const auto put = [&](auto& data, const CachePlane& plane,
                                             const auto& words) {
                            std::copy(words.begin(), words.end(),
                                      data.begin() + plane.index(0, p % 64, h, page));
                        };
                        std::array<double, 256> key;
                        std::copy(ideal.values.begin(), ideal.values.end(), key.begin());
                        EncodedRow got{row(actual_k, f.keys), row(actual_ks, f.key_scales)};
                        if (!compatible_key_encoding(key, f.encoding.key, got)) {
                            std::cerr << label
                                      << " K encoding incompatible with mathematical key at " << b
                                      << '/' << t << '/' << h << '\n';
                            ++failures;
                        }
                        if (f.exact_key) {
                            std::array<float, 256> represented;
                            std::copy(key.begin(), key.end(), represented.begin());
                            const auto encoded = encode_row(represented, f.encoding.key);
                            failures += verify_exact("exact K codes", got.data, encoded.data);
                            failures += verify_exact("exact K scales", got.scales, encoded.scales);
                        }
                        put(expected_k, f.keys, got.data);
                        put(expected_ks, f.key_scales, got.scales);
                        std::array<float, 256> raw_value;
                        for (int c = 0; c < 256; ++c)
                            raw_value[c] = bf16_to_f32(f.v.at(c, h, t, b));
                        const auto value = encode_row(raw_value, f.encoding.value);
                        put(expected_v, f.values, value.data);
                        put(expected_vs, f.value_scales, value.scales);
                    }
                }
                const int record = f.offsets.at(b) + t;
                for (int c = 0; c < 128; ++c)
                    expected_raw[f.records.index(c, record, b)] = f.ik.at(c, t, b);
                for (int a = 0; a < 3; ++a)
                    expected_rope[f.record_rope.index(a, record, b)] = rho[a];
            }
        });
    for (const auto& error : token_errors) {
        q_error.merge(error.q);
        k_error.merge(error.k);
        iq_error.merge(error.iq);
        failures += error.failures;
    }
    for (int b = 0; b < f.B; ++b) {
        const int n = f.valid.at(b), base = f.bases[b];
        if (n == 0) continue;
        // Independent chronological grouping, starting with the canonical pending prefix.
        std::vector<std::vector<double>> pending;
        std::array<int, 3> first{};
        const int slot = f.slots.at(b);
        for (int j = 0; j < base % 4; ++j) {
            std::vector<double> raw(128);
            for (int c = 0; c < 128; ++c) raw[c] = bf16_to_f32(f.tail.at(c, j, slot));
            pending.push_back(raw);
        }
        if (!pending.empty())
            for (int a = 0; a < 3; ++a) first[a] = f.first.at(a, slot);
        for (int t = 0; t < n; ++t) {
            if (pending.empty()) first = coordinates(t, b);
            std::vector<double> raw(128);
            for (int c = 0; c < 128; ++c) raw[c] = bf16_to_f32(f.ik.at(c, t, b));
            pending.push_back(raw);
            if (pending.size() != 4) continue;
            std::vector<double> pooled(128);
            for (int c = 0; c < 128; ++c) {
                double sum = 0;
                for (const auto& token : pending) sum += token[c];
                pooled[c] = bf16_to_f32(f32_to_bf16(float(sum / 4)));
            }
            const auto ideal = oracle(
                pooled, f.wik, rope_rotation(first, f.parameters.kv.rope_theta), f.parameters.kv);
            const int g = (base + t) / 4, page = f.tables.at(g / 16, f.table_rows.at(b));
            for (int c = 0; c < 128; ++c) {
                const auto idx = f.compressed.index(c, g % 16, 0, page);
                ik_error.add(actual_ik[idx], ideal.values[c], ideal.scales[c]);
                expected_ik[idx] = actual_ik[idx];
            }
            pending.clear();
        }
    }
    failures += q_error.check(label + " Q") + k_error.check(label + " K") +
                iq_error.check(label + " IQ") + ik_error.check(label + " IK");
    failures += verify_exact("K untouched", actual_k, expected_k);
    failures += verify_exact("IK untouched", actual_ik, expected_ik);
    failures += verify_exact("V words", f.values.download(), expected_v);
    failures += verify_exact("K scales untouched", actual_ks, expected_ks);
    failures += verify_exact("V scales", f.value_scales.download(), expected_vs);
    if (!updates_tail) {
        failures += verify_exact("raw records", f.records.download(), expected_raw);
        failures += verify_exact("coordinate records", f.record_rope.download(), expected_rope);
        failures += f.tail.preserved("raw tail");
        failures += f.first.preserved("tail coordinates");
    }
    if (f.kv_only || !(f.alias & 1)) failures += f.q.preserved("raw Q");
    if (f.kv_only || !(f.alias & 2)) failures += f.iq.preserved("raw IQ");
    for (auto* x : {&f.k, &f.v, &f.ik, &f.wq, &f.wk, &f.wiq, &f.wik})
        failures += x->preserved("input");
    for (auto* x :
         {&f.positions, &f.rope, &f.valid, &f.table_rows, &f.slots, &f.offsets, &f.tables})
        failures += x->preserved("metadata/state");
    for (auto* x : {&f.q, &f.iq, &f.out_q, &f.out_iq, &f.compressed, &f.records})
        failures += x->guards(label);
    failures += f.record_rope.device->verify_guards(label);
    if (f.kv_only)
        failures += f.out_q.preserved("unused Q output") + f.out_iq.preserved("unused IQ output");
    if (failures) std::cerr << label << ": " << failures << " failures\n";
    failures += !f.packed->guards_intact();
    return failures;
}

template <class F>
int rejects(const char* label, F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << label << ": invalid view accepted\n";
    return 1;
}

void seed_carried_axes(Fixture& f, int phase) {
    for (int b = 0; b < f.B; ++b) {
        if (f.bases[b] % 4 == 0) continue;
        for (int a = 0; a < 3; ++a)
            f.first.at(a, f.slots.at(b)) = 11 + 47 * a + 137 * b + 253 * phase;
    }
}

int ragged_prepare(bool kv_only, KvCacheStorage storage, int kv_heads, bool prefix_arena) {
    RaggedFixture ragged(prefix_arena ? std::vector{1, 2, 3, 5, 7, 9, 15, 17, 33, 65, 127, 129}
                                      : std::vector{1, 2, 3, 5, 7});
    const int width = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    const int n = ragged.tokens(), s = ragged.segments(), axes = kv_only ? 1 : 3;
    Fixture f(width, s, axes, 0, 0, 0, storage, kv_heads);
    f.kv_only = kv_only;
    f.reset(1, true);
    seed_carried_axes(f, 1);
    f.valid.host = ragged.lengths;
    f.valid.upload();
    // Rows 0 and 4 share a read-only source. Row 0 publishes disjointly, row 4 uses
    // its original slot, and the other rows publish in place.
    const auto original_slots = f.slots.host;
    f.slots.at(4)             = f.slots.at(0);
    f.slots.upload();
    std::vector<int> destinations = original_slots;
    destinations[0]               = 0;
    const auto rope_slot          = [s](int slot) { return s + 1 - slot; };
    auto initial_rope             = f.first.host;
    for (int slot = 0; slot < s + 2; ++slot)
        for (int a = 0; a < 3; ++a)
            initial_rope[f.first.index(a, rope_slot(slot))] = f.first.at(a, slot);
    std::vector<std::int64_t> before_raw(s), before_rope(s), after_raw(s), after_rope(s);
    for (int b = 0; b < s; ++b) {
        before_raw[b]  = f.slots.at(b) * f.tail.view.nb[2];
        before_rope[b] = rope_slot(f.slots.at(b)) * f.first.view.nb[1];
        after_raw[b]   = destinations[b] * f.tail.view.nb[2];
        after_rope[b]  = rope_slot(destinations[b]) * f.first.view.nb[1];
    }
    auto br = to_device(before_raw), bc = to_device(before_rope);
    auto dr = to_device(after_raw), dc = to_device(after_rope);
    const std::size_t raw_shift = axes == 1 ? 2 : 0;
    GuardedDeviceBuffer raw(f.tail.host.size() * 2 + raw_shift);
    auto raw_state = raw.span().subspan(raw_shift, f.tail.host.size() * 2);
    CUDA_CHECK(
        cudaMemcpy(raw_state.data, f.tail.host.data(), raw_state.bytes, cudaMemcpyHostToDevice));
    auto coords_state         = to_device(initial_rope);
    f.before.raw_keys.records = {raw_state, Tensor(br.span(), DType::I64, {s})};
    f.before.raw_keys.field.tensor.region.alignment = raw_shift ? 2 : 4;
    f.before.first_rope.records         = {coords_state.span(), Tensor(bc.span(), DType::I64, {s})};
    auto after                          = f.before;
    after.raw_keys.records.byte_offsets = Tensor(dr.span(), DType::I64, {s});
    after.first_rope.records.byte_offsets = Tensor(dc.span(), DType::I64, {s});
    auto expected_tail                    = f.tail.host;
    auto expected_rope                    = initial_rope;
    const int axis_stride                 = n + 9;
    std::vector<int> coords(axis_stride * axes, -1), pos = ragged.pack(f.positions.host, width, 1);
    for (int b = 0; b < s; ++b) {
        const int base = f.bases[b], count = ragged.lengths[b], src = f.slots.at(b),
                  dst       = destinations[b];
        const int remainder = (base + count) % 4, first = base + count - remainder - base;
        for (int a = 0; a < axes; ++a)
            for (int t = 0; t < count; ++t)
                coords[a * axis_stride + ragged.prefix[b] + t] = f.rope.at(t, a, b);
        for (int a = 0; a < 3; ++a)
            expected_rope[f.first.index(a, rope_slot(dst))] =
                !remainder
                    ? 0
                    : (first < 0 ? f.first.at(a, src) : f.rope.at(first, axes == 1 ? 0 : a, b));
        for (int j = 0; j < 3; ++j)
            for (int c = 0; c < 128; ++c)
                expected_tail[f.tail.index(c, j, dst)] =
                    j >= remainder ? 0
                                   : (first + j < 0 ? f.tail.at(c, base % 4 + first + j, src)
                                                    : f.ik.at(c, first + j, b));
    }
    auto dq  = to_device(ragged.pack(f.q.host, width, 256 * f.query_heads));
    auto dk  = to_device(ragged.pack(f.k.host, width, 256 * f.kv_heads));
    auto dv  = to_device(ragged.pack(f.v.host, width, 256 * f.kv_heads));
    auto diq = to_device(ragged.pack(f.iq.host, width, 128 * 4));
    auto dik = to_device(ragged.pack(f.ik.host, width, 128));
    auto dp = to_device(pos), rope = to_device(coords);
    ops::QsaRaggedBatchView batch{ragged.view, Tensor(dp.span(), DType::I32, {n}),
                                  f.batch.table_rows};
    auto rope_positions  = Tensor(rope.span(), DType::I32, {axis_stride, axes});
    rope_positions.ne[0] = n;
    ops::QsaProjectionView projected{Tensor(dq.span(), DType::BF16, {256, f.query_heads, n}),
                                     Tensor(dk.span(), DType::BF16, {256, f.kv_heads, n}),
                                     Tensor(dv.span(), DType::BF16, {256, f.kv_heads, n}),
                                     Tensor(diq.span(), DType::BF16, {128, 4, n}),
                                     Tensor(dik.span(), DType::BF16, {128, n})};
    const auto bytes = kv_only
                           ? ops::qsa_prepare_kv_update_ragged_workspace_capacity_bytes(n, n, s, s)
                           : ops::qsa_prepare_update_ragged_workspace_capacity_bytes(n, n, s, s);
    const auto prefix =
        prefix_arena ? qsa::prefix_bytes({projected.query, projected.key, projected.value,
                                          projected.index_query, projected.index_key,
                                          batch.positions, batch.table_rows, ragged.view.cu_seqlens,
                                          ragged.view.seq_lengths, ragged.view.token_to_seq,
                                          Tensor(rope.span(), DType::I32, {axis_stride, axes})})
                     : 0;
    GuardedDeviceBuffer scratch(prefix);
    MemoryArena arena(scratch.span());
    const auto copy = [&](Tensor t) {
        const auto span = arena.alloc_bytes(t.bytes(), 256);
        CUDA_CHECK(cudaMemcpy(span.data, t.data, t.bytes(), cudaMemcpyDeviceToDevice));
        t.data = span.data;
        return t;
    };
    if (prefix_arena) {
        projected.query       = copy(projected.query);
        projected.key         = copy(projected.key);
        projected.value       = copy(projected.value);
        projected.index_query = copy(projected.index_query);
        projected.index_key   = copy(projected.index_key);
        batch.positions       = copy(batch.positions);
        batch.table_rows      = copy(batch.table_rows);
        batch.ragged          = qsa::packed_metadata(arena, ragged);
        rope_positions        = copy(Tensor(rope.span(), DType::I32, {axis_stride, axes}));
        rope_positions.ne[0]  = n;
        qsa::finish_prefix(arena, prefix);
    }
    auto query = projected.query, index_query = projected.index_query;
    query.nb[3] += 16;
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const auto execution = cuda_environment().default_execution;
    int failures         = bytes != 0;
    if (kv_only)
        ops::qsa_prepare_kv_update({projected.key, projected.value, projected.index_key},
                                   f.parameters.kv, batch, rope_positions, f.before, f.cache, after,
                                   arena, execution);
    else {
        // A one-column shift is disjoint within the first segment, but aliases later inputs.
        for (const bool shift_index : {false, true}) {
            auto inputs     = projected;
            const int width = shift_index ? 128 : 256, heads = shift_index ? 4 : f.query_heads;
            GuardedDeviceBuffer shifted(std::size_t(width) * heads * (n + 1) * 2);
            shifted.fill();
            const Tensor parent(shifted.span(), DType::BF16, {width, heads, n + 1});
            (shift_index ? inputs.index_query : inputs.query) = parent.slice(2, 0, n);
            auto q = query, iq = index_query;
            (shift_index ? iq : q) = parent.slice(2, 1, n);
            failures += rejects("shifted packed query", [&] {
                ops::qsa_prepare_update(inputs, f.parameters, batch, rope_positions, f.before,
                                        f.cache, after, q, iq, arena, execution);
            });
            cuda_synchronize();
        }
        auto alias = batch;
        alias.ragged.token_to_seq =
            Tensor({query.data, n * sizeof(int), query.device}, DType::I32, {n});
        failures += rejects("packed query overlaps token-to-sequence metadata", [&] {
            ops::qsa_prepare_update(projected, f.parameters, alias, rope_positions, f.before,
                                    f.cache, after, query, index_query, arena, execution);
        });
        failures += arena.peak_used() != prefix;
        ops::qsa_prepare_update(projected, f.parameters, batch, rope_positions, f.before, f.cache,
                                after, query, index_query, arena, execution);
    }
    cuda_synchronize();
    if (!kv_only) {
        const auto padded_q  = ragged.unpack(from_device<std::uint16_t>(query.data, query.numel()),
                                             width, 256 * f.query_heads);
        const auto padded_iq = ragged.unpack(
            from_device<std::uint16_t>(index_query.data, index_query.numel()), width, 128 * 4);
        f.out_q.device->copy_from_host(padded_q.data(), padded_q.size() * 2);
        f.out_iq.device->copy_from_host(padded_iq.data(), padded_iq.size() * 2);
    }
    return failures + verify(f, "ragged QSA prepare", true) +
           verify_exact("ragged raw tails and untouched slots",
                        from_device<std::uint16_t>(raw_state.data, expected_tail.size()),
                        expected_tail) +
           verify_exact("ragged tail coordinates and untouched slots",
                        from_device<int>(coords_state.data(), expected_rope.size()),
                        expected_rope) +
           f.tail.guards("ragged raw tail") + f.first.guards("ragged coordinates") +
           scratch.verify_guards("ragged prepare workspace") +
           raw.verify_guards("ragged raw state") +
           (arena.used() != prefix || arena.peak_used() != prefix);
}

int ragged_prepare_graph() {
    RaggedFixture ragged({1, 2, 5});
    const int n = ragged.tokens(), s = ragged.segments(), width = 5;
    Fixture f(width, s, 1);
    const auto execution = cuda_environment().context.execution_view();
    GuardedDeviceBuffer backing(0);
    MemoryArena arena(backing.span());
    GuardedDeviceBuffer q(n * 256 * f.query_heads * 2ULL), iq(n * 512 * 2ULL);
    GuardedDeviceBuffer k(n * 256 * f.kv_heads * 2ULL), v(k.bytes()), ik(n * 128 * 2ULL);
    GuardedDeviceBuffer output(q.bytes()), index_output(iq.bytes()), pos(n * 4), rho(n * 4);
    auto out    = Tensor(output.span(), DType::BF16, {256, f.query_heads, n});
    auto out_iq = Tensor(index_output.span(), DType::BF16, {128, 4, n});
    const ops::QsaProjectionView projected{Tensor(q.span(), DType::BF16, {256, f.query_heads, n}),
                                           Tensor(k.span(), DType::BF16, {256, f.kv_heads, n}),
                                           Tensor(v.span(), DType::BF16, {256, f.kv_heads, n}),
                                           Tensor(iq.span(), DType::BF16, {128, 4, n}),
                                           Tensor(ik.span(), DType::BF16, {128, n})};
    const ops::QsaRaggedBatchView batch{ragged.view, Tensor(pos.span(), DType::I32, {n}),
                                        f.batch.table_rows};
    const Tensor rope(rho.span(), DType::I32, {n, 1});
    const auto launch = [&] {
        ops::qsa_prepare_update(projected, f.parameters, batch, rope, f.before, f.cache, f.before,
                                out, out_iq, arena, execution);
    };

    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    // F=61 with one token leaves an unfinished carry. F=62 with two tokens completes
    // the carried pool. Replay moves both sources and destinations to different records.
    for (const int phase : {1, 2}) {
        f.reset(phase, true);
        seed_carried_axes(f, phase);
        f.valid.host = ragged.lengths;
        if (phase == 2) {
            const auto raw   = f.tail.host;
            const auto first = f.first.host;
            for (int b = 0; b < s; ++b) {
                const int original = f.slots.at(b), slot = (original + 1) % (s + 2);
                for (int j = 0; j < 3; ++j) {
                    f.first.at(j, slot) = first[f.first.index(j, original)];
                    for (int c = 0; c < 128; ++c)
                        f.tail.at(c, j, slot) = raw[f.tail.index(c, j, original)];
                }
                f.slots.at(b) = slot;
            }
        }
        f.valid.upload();
        f.tail.upload();
        f.first.upload();
        f.slots.upload();
        f.upload_offsets();
        const auto set = [&](GuardedDeviceBuffer& buffer, const auto& words, int features) {
            const auto packed = ragged.pack(words, width, features);
            buffer.copy_from_host(packed.data(), packed.size() * sizeof(packed[0]));
        };
        set(q, f.q.host, 256 * f.query_heads);
        set(k, f.k.host, 256 * f.kv_heads);
        set(v, f.v.host, 256 * f.kv_heads);
        set(iq, f.iq.host, 512);
        set(ik, f.ik.host, 128);
        set(pos, f.positions.host, 1);
        set(rho, f.rope.host, 1);
        auto expected_raw  = f.tail.host;
        auto expected_rope = f.first.host;
        for (int b = 0; b < s; ++b) {
            const int slot = f.slots.at(b), base = f.bases[b];
            // Construct the chronological pending sequence and remove completed pools.
            std::vector<std::array<std::uint16_t, 128>> pending;
            std::array<int, 3> coord{};
            for (int j = 0; j < base % 4; ++j) {
                auto& raw = pending.emplace_back();
                for (int c = 0; c < 128; ++c) raw[c] = f.tail.at(c, j, slot);
            }
            if (!pending.empty())
                for (int a = 0; a < 3; ++a) coord[a] = f.first.at(a, slot);
            for (int t = 0; t < ragged.lengths[b]; ++t) {
                if (pending.empty()) coord.fill(f.rope.at(t, 0, b));
                auto& raw = pending.emplace_back();
                for (int c = 0; c < 128; ++c) raw[c] = f.ik.at(c, t, b);
                if (pending.size() == 4) pending.clear();
            }
            for (int j = 0; j < 3; ++j) {
                expected_rope[f.first.index(j, slot)] = pending.empty() ? 0 : coord[j];
                for (int c = 0; c < 128; ++c)
                    expected_raw[f.tail.index(c, j, slot)] =
                        j < int(pending.size()) ? pending[j][c] : 0;
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        if (phase == 1) {
            definition.capture(execution.stream(), launch);
            graph.instantiate(definition);
        }
        graph.launch(execution.stream());
        cuda_synchronize(execution.stream());
        const auto download = [&](Tensor result, Buffer<std::uint16_t>& dst, int features) {
            auto padded = ragged.unpack(from_device<std::uint16_t>(result.data, result.numel()),
                                        width, features);
            dst.device->copy_from_host(padded.data(), padded.size() * 2);
        };
        download(out, f.out_q, 256 * f.query_heads);
        download(out_iq, f.out_iq, 512);
        failures += verify(f, "packed A=1 prepare replay", true);
        failures += verify_exact("graph raw tails", f.tail.download(), expected_raw);
        failures += verify_exact("graph carried coordinates", f.first.download(), expected_rope);
    }
    return failures + (arena.peak_used() != 0);
}

int run(DeviceContext& device, int t, int b, int axes, int alias, int layout, int capacity = 0,
        KvCacheStorage storage = KvCacheStorage::BFloat16, int kv_heads = 2, bool kv_only = false,
        bool empty_replay = false) {
    Fixture f(t, b, axes, alias, layout, capacity, storage, kv_heads);
    f.kv_only = kv_only;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int phase = 0; phase < (empty_replay ? 4 : 3); ++phase) {
        f.reset(phase, phase == 1);
        if (phase == 3) {
            std::fill(f.valid.host.begin(), f.valid.host.end(), 0);
            std::fill(f.slots.host.begin(), f.slots.host.end(), -1);
            std::fill(f.table_rows.host.begin(), f.table_rows.host.end(), -1);
            f.valid.upload();
            f.slots.upload();
            f.table_rows.upload();
            f.upload_offsets();
        }
        if (phase == 1) {
            f.parameters.kv.rms_epsilon = 2e-5F;
            f.parameters.kv.rope_theta  = 500000;
        } else {
            f.parameters.kv.rms_epsilon = 1e-6F;
            f.parameters.kv.rope_theta  = 1e7F;
        }
        if (phase == 0) {
            definition.capture(device.stream(), [&] { f.launch(device.execution_view()); });
            graph.instantiate(definition);
        }
        if (phase == 1)
            f.launch(device.execution_view());
        else
            graph.launch(device.stream());
        device.synchronize();
        failures +=
            verify(f, "prepare T=" + std::to_string(t) + " B=" + std::to_string(b) +
                          " A=" + std::to_string(axes) + " alias=" + std::to_string(alias) +
                          " layout=" + std::to_string(layout) + " phase=" + std::to_string(phase) +
                          " storage=" + std::to_string(int(storage)));
    }
    return failures;
}

int continuation(DeviceContext& device, KvCacheStorage storage = KvCacheStorage::BFloat16,
                 int kv_heads = 2, bool kv_only = false) {
    Fixture f(4, 1, 3, 0, 2, 0, storage, kv_heads);
    f.kv_only     = kv_only;
    f.base_origin = 6;
    f.reset(0, true);
    f.launch(device.execution_view());
    device.synchronize();
    int failures = verify(f, "continuation initial");
    auto base = to_device_i32({6}), count = to_device_i32({1});
    ops::qsa_tail_extract(
        f.before,
        {f.records.view, f.record_rope.view, Tensor(base.span(), DType::I32, {1}), f.valid.view},
        Tensor(count.span(), DType::I32, {1}), f.before, {}, device.execution_view());
    device.synchronize();
    for (int c = 0; c < 128; ++c) f.tail.at(c, 2, 1) = f.ik.at(c, 0, 0);
    failures += f.tail.preserved("accepted-one canonical tail") +
                f.first.preserved("tail first coordinate");
    f.keys.host          = f.keys.download();
    f.values.host        = f.values.download();
    f.key_scales.host    = f.key_scales.download();
    f.value_scales.host  = f.value_scales.download();
    f.compressed.host    = f.compressed.download();
    f.records.host       = f.records.download();
    f.record_rope.host   = f.record_rope.download();
    f.bases[0]           = 7;
    f.valid.at(0)        = 1;
    f.offsets.at(0)      = 1;
    f.positions.at(0, 0) = 7;
    // Replace rejected token 7 while retaining actual old raw words 4/5/6 from TailExtract.
    for (int c = 0; c < 128; ++c) f.ik.at(c, 0, 0) = sample(7, 0, c, 313);
    for (int a = 0; a < 3; ++a) f.rope.at(0, a, 0) = 7000 + a * 37;
    f.ik.upload();
    for (auto* x : {&f.valid, &f.offsets, &f.positions, &f.rope}) x->upload();
    f.launch(device.execution_view());
    device.synchronize();
    failures += verify(f, "continuation replacement");
    return failures;
}

int chunking(DeviceContext& device, KvCacheStorage storage = KvCacheStorage::BFloat16,
             int kv_heads = 2, bool kv_only = false) {
    Fixture f(9, 1, 3, 3, 2, 0, storage, kv_heads);
    f.kv_only     = kv_only;
    f.base_origin = 63;
    f.reset(0, true);
    f.launch(device.execution_view());
    device.synchronize();
    int failures = verify(f, "whole chunk");
    const auto q = f.q.download(), iq = f.iq.download(), ik = f.compressed.download(),
               records = f.records.download();
    const auto k = f.keys.download(), v = f.values.download(), ks = f.key_scales.download(),
               vs          = f.value_scales.download();
    const auto coordinates = f.record_rope.download();
    f.reset(0, true);
    int offset = 0;
    auto base  = to_device_i32({63});
    for (int n : {1, 3, 2, 3}) {
        auto p          = f.projected;
        p.query         = p.query.slice(2, offset, n);
        p.key           = p.key.slice(2, offset, n);
        p.value         = p.value.slice(2, offset, n);
        p.index_query   = p.index_query.slice(2, offset, n);
        p.index_key     = p.index_key.slice(1, offset, n);
        f.valid.at(0)   = n;
        f.offsets.at(0) = offset;
        f.valid.upload();
        f.offsets.upload();
        auto batch      = f.batch;
        batch.positions = f.positions.view.slice(0, offset, n);
        std::vector<int> coords(n * 3);
        for (int a = 0; a < 3; ++a)
            for (int t = 0; t < n; ++t) coords[t + n * a] = f.rope.at(offset + t, a, 0);
        auto rope    = to_device_i32(coords);
        Tensor q_out = p.query, iq_out = p.index_query;
        if (kv_only)
            ops::qsa_prepare_kv({p.key, p.value, p.index_key}, f.parameters.kv, batch,
                                Tensor(rope.span(), DType::I32, {n, 3, 1}), f.before, f.cache,
                                f.write, device.execution_view());
        else
            ops::qsa_prepare(p, f.parameters, batch, Tensor(rope.span(), DType::I32, {n, 3, 1}),
                             f.before, f.cache, f.write, q_out, iq_out, device.execution_view());
        const int frontier = 63 + offset;
        base.copy_from_host(&frontier, 4);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        ops::qsa_tail_extract(f.before,
                              {f.records.view.slice(1, offset, n),
                               f.record_rope.view.slice(1, offset, n),
                               Tensor(base.span(), DType::I32, {1}), f.valid.view},
                              f.valid.view, f.before, {}, device.execution_view());
        device.synchronize();
        offset += n;
    }
    failures +=
        verify_exact("chunk Q", f.q.download(), q) + verify_exact("chunk IQ", f.iq.download(), iq);
    failures += verify_exact("chunk K", f.keys.download(), k) +
                verify_exact("chunk V", f.values.download(), v);
    failures += verify_exact("chunk K scales", f.key_scales.download(), ks) +
                verify_exact("chunk V scales", f.value_scales.download(), vs);
    failures += verify_exact("chunk IK", f.compressed.download(), ik);
    failures += verify_exact("chunk records", f.records.download(), records);
    failures += verify_exact("chunk coordinates", f.record_rope.download(), coordinates);
    for (int c = 0; c < 128; ++c)
        for (int j = 0; j < 3; ++j) f.tail.at(c, j, 1) = 0;
    for (int a = 0; a < 3; ++a) f.first.at(a, 1) = 0;
    failures += f.tail.preserved("chunk final tail") + f.first.preserved("chunk final coordinates");
    return failures;
}

int numerical_edges(DeviceContext& device, bool wide_phase) {
    Fixture f(wide_phase ? 5 : 2048, 1, 3, 3, 0);
    f.reset(2, true);
    f.parameters.kv.rope_theta = wide_phase ? 0.5F : 1e7F;
    for (int c = 0; c < 256; ++c) {
        f.q.at(c, 0, 0, 0) = 0;
        f.q.at(c, 1, 0, 0) = f32_to_bf16((c % 3 - 1) * 1e20F);
        f.q.at(c, 2, 0, 0) = f32_to_bf16((c % 3 - 1) * 1e-30F);
    }
    f.q.upload();
    f.launch(device.execution_view());
    device.synchronize();
    return verify(f, "normalization/phase edges");
}

int epsilon_range(DeviceContext& device) {
    Fixture f(1, 1, 1);
    f.parameters.kv.rms_epsilon = std::numeric_limits<float>::max();
    for (int c = 0; c < 256; ++c) f.q.at(c, 0, 0, 0) = f32_to_bf16(1e18F);
    f.q.upload();
    f.launch(device.execution_view());
    device.synchronize();
    return verify(f, "finite norm denominator range");
}

int coordinate_handoff(DeviceContext& device) {
    Fixture f(1, 1, 1);
    f.base_origin = 63;
    f.reset(0, true);
    f.first.at(1, 1) = 1048576;
    f.first.at(2, 1) = 2000000000;
    f.first.upload();
    f.launch(device.execution_view());
    device.synchronize();
    return verify(f, "three-axis tail to one-axis current input");
}

int norm_alignment_and_phase(DeviceContext& device,
                             KvCacheStorage storage = KvCacheStorage::BFloat16, int kv_heads = 2) {
    int failures = 0;
    for (int tokens : {129, 321}) {
        Fixture f(tokens, 1, 1, 3, 1, 0, storage, kv_heads);
        Buffer<std::uint16_t> wq(DType::BF16, {257}), wiq(DType::BF16, {129}),
            wk(DType::BF16, {257});
        std::copy(f.wq.host.begin(), f.wq.host.end(), wq.host.begin() + 1);
        std::copy(f.wiq.host.begin(), f.wiq.host.end(), wiq.host.begin() + 1);
        std::copy(f.wk.host.begin(), f.wk.host.end(), wk.host.begin() + 1);
        wk.upload();
        wq.upload();
        wiq.upload();
        for (bool shifted : {true, false}) {
            f.parameters.kv.key_norm      = shifted ? wk.view.slice(0, 1, 256) : f.wk.view;
            f.parameters.query_norm       = shifted ? wq.view.slice(0, 1, 256) : f.wq.view;
            f.parameters.index_query_norm = shifted ? wiq.view.slice(0, 1, 128) : f.wiq.view;
            for (float theta : {1e7F, 0.5F}) {
                f.reset(2, true);
                f.parameters.kv.rope_theta = theta;
                f.launch(device.execution_view());
                device.synchronize();
                failures += verify(f, "norm alignment/phase domain");
            }
        }
        failures += wq.preserved("shifted Q norm") + wiq.preserved("shifted IQ norm") +
                    wk.preserved("shifted K norm");
    }
    return failures;
}

// Inert projection columns/coordinates are deliberately uninitialized. Initcheck can detect
// a masked read even if subsequent arithmetic discards it and the final output is zero.
int read_footprint(DeviceContext& device, KvCacheStorage storage = KvCacheStorage::BFloat16) {
    Fixture f(17, 2, 3, 0, 0, 0, storage);
    f.base_origin = 63;
    f.reset(0, true);
    f.valid.at(0)      = 1;
    f.valid.at(1)      = 0;
    f.slots.at(1)      = -1;
    f.table_rows.at(1) = -1;
    f.valid.upload();
    f.slots.upload();
    f.upload_offsets();
    f.table_rows.upload();
    std::vector<DeviceBuffer> inputs;
    for (auto* view : {&f.projected.query, &f.projected.key, &f.projected.value,
                       &f.projected.index_query, &f.projected.index_key}) {
        const int width = view->ne[0] * (view == &f.projected.index_key ? 1 : view->ne[1]);
        inputs.emplace_back(device.device(), view->bytes());
        CUDA_CHECK(cudaMemcpyAsync(inputs.back().data(), view->data, width * 2,
                                   cudaMemcpyDeviceToDevice, device.stream()));
        view->data   = inputs.back().data();
        view->device = inputs.back().device();
    }
    DeviceBuffer pos(ninfer::test::cuda_environment().context.device(), f.positions.view.bytes());
    DeviceBuffer rope(ninfer::test::cuda_environment().context.device(), f.rope.view.bytes());
    CUDA_CHECK(cudaMemcpyAsync(pos.data(), f.positions.view.data, 4, cudaMemcpyDeviceToDevice,
                               device.stream()));
    for (int a = 0; a < 3; ++a)
        CUDA_CHECK(cudaMemcpyAsync(static_cast<int*>(rope.data()) + a * f.T,
                                   static_cast<int*>(f.rope.view.data) + a * f.T, 4,
                                   cudaMemcpyDeviceToDevice, device.stream()));
    f.batch.positions.data = pos.data();
    auto saved             = f.rope.view;
    f.rope.view.data       = rope.data();
    f.launch(device.execution_view());
    device.synchronize();
    f.rope.view = saved;
    return verify(f, "masked read footprint");
}

int validation(DeviceContext& device) {
    Fixture f(1, 1, 1);
    int failures              = 0;
    auto saved                = f.cache;
    f.cache.main.num_kv_heads = 1;
    failures += rejects("cache/projection heads", [&] { f.launch(device.execution_view()); });
    f.cache                 = saved;
    const auto projected    = f.projected;
    f.projected.query.ne[1] = 12;
    failures += rejects("query/key head ratio", [&] { f.launch(device.execution_view()); });
    f.projected          = projected;
    f.cache.main.storage = KvCacheStorage::Int8Group64;
    failures += rejects("storage/plane mismatch", [&] { f.launch(device.execution_view()); });
    f.cache = saved;
    f.cache.index.keys.records.page_offsets.nb[1] += 8;
    failures += rejects("different table view", [&] { f.launch(device.execution_view()); });
    f.cache                                      = saved;
    f.cache.main.k_pages.plane.head_stride_bytes = 256;
    failures += rejects("overlapping cache heads", [&] { f.launch(device.execution_view()); });
    f.cache             = saved;
    auto q              = f.query_output;
    f.query_output.data = static_cast<std::byte*>(f.q.view.data) + 16;
    failures += rejects("partial Q alias", [&] { f.launch(device.execution_view()); });
    f.query_output              = q;
    f.parameters.kv.rms_epsilon = 0;
    failures += rejects("zero epsilon", [&] { f.launch(device.execution_view()); });
    return failures;
}

int exact_key_encoding(DeviceContext& device, KvCacheStorage storage, bool kv_only = false,
                       int kv_heads = 2) {
    Fixture f(16, 1, 1, 3, 1, 0, storage, kv_heads);
    f.kv_only     = kv_only;
    f.base_origin = 64;
    f.reset(0, true);
    f.parameters.kv.rms_epsilon = 3;
    f.exact_key                 = true;
    // |Kraw|=1 and epsilon=3 give an exact inverse RMS of 1/2. Zero RoPE coordinates
    // leave dyadic 0.5*(1+w) values that are FP32-exact but generally not BF16-exact.
    for (int c = 0; c < 256; ++c) f.wk.at(c) = f32_to_bf16(((c * 37 + 17) % 127 - 63) / 512.0F);
    int witnesses = 0;
    for (int t = 0; t < f.T; ++t) {
        f.rope.at(t, 0, 0) = 0;
        for (int h = 0; h < f.kv_heads; ++h) {
            std::array<float, 256> exact, rounded;
            for (int c = 0; c < 256; ++c) {
                const float sign   = bf16_to_f32(sample(t, h, c, 719)) < 0 ? -1 : 1;
                f.k.at(c, h, t, 0) = f32_to_bf16(sign);
                exact[c]           = sign * 0.5F * (1 + bf16_to_f32(f.wk.at(c)));
                rounded[c]         = bf16_to_f32(f32_to_bf16(exact[c]));
            }
            const auto wanted         = encode_row(exact, f.encoding.key);
            const auto extra_rounding = encode_row(rounded, f.encoding.key);
            witnesses +=
                wanted.data != extra_rounding.data || wanted.scales != extra_rounding.scales;
        }
    }
    f.upload();
    f.launch(device.execution_view());
    device.synchronize();
    int failures = verify(f, "exact unrounded key codec");
    if (witnesses == 0) {
        std::cerr << "exact key case must distinguish an intermediate BF16 rounding\n";
        ++failures;
    }
    for (auto& word : f.wk.host) word = f32_to_bf16(-1);
    std::fill(f.v.host.begin(), f.v.host.end(), 0);
    f.upload();
    f.launch(device.execution_view());
    device.synchronize();
    failures += verify(f, "zero key/value codec");
    return failures;
}

int scale_edges(DeviceContext& device, KvCacheStorage storage, bool kv_only = false,
                int kv_heads = 2) {
    Fixture f(8, 1, 1, 0, 1, 0, storage, kv_heads);
    f.kv_only = kv_only;
    constexpr float amplitudes[]{0, 0x1p-30F, 0x1p-18F, 0x1p-10F, 1.0625F, 6.375F, 3072, 65536};
    constexpr float values[]{0, 0.25F, 0.75F, 1.25F, 1.75F, 2.5F, 3.5F, 6};
    for (int t = 0; t < f.T; ++t)
        for (int h = 0; h < f.kv_heads; ++h)
            for (int c = 0; c < 256; ++c) {
                const float sign   = (c / 8 + h) % 2 ? -1 : 1;
                f.v.at(c, h, t, 0) = f32_to_bf16(sign * amplitudes[t] * values[c % 8]);
            }
    f.v.upload();
    f.launch(device.execution_view());
    device.synchronize();
    return verify(f, "scale zero/underflow/saturation");
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                         KvCacheStorage::Fp8KeyNvfp4Value}) {
        suite.add(std::string("qsa_prepare/") + kv_cache_storage_name(storage) + "/concurrency/b17",
                  [=] {
                      return run(cuda_environment().context, 4, 17, 3, 0, 1, 0, storage, 2, false,
                                 true);
                  },
                  {{"format", kv_cache_storage_name(storage)},
                   {"execution", "graph"},
                   {"suite", "concurrency"}});
    }
    suite.add("qsa_prepare/bf16/ragged/graph_a1", ragged_prepare_graph,
              {{"format", "bf16"}, {"execution", "graph"}});
    for (const auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                               KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                               KvCacheStorage::Fp8KeyNvfp4Value})
        for (const int kv_heads : {1, 2, 4}) {
#if NINFER_TARGET_SPARK
            if (kv_heads == 4) continue;
#endif
            for (const bool kv_only : {false, true}) {
                const std::string format = kv_cache_storage_name(storage);
                const std::string op     = kv_only ? "qsa_prepare_kv" : "qsa_prepare";
                suite.add(op + "/" + format + "/ragged/h" + std::to_string(kv_heads),
                          [=] {
                              return ragged_prepare(kv_only, storage, kv_heads,
                                                    storage == KvCacheStorage::BFloat16 &&
                                                        kv_heads == 2);
                          },
                          {{"format", format}});
            }
        }

    suite.add("qsa_prepare/bf16/concurrency/b129",
              [] {
                  return run(cuda_environment().context, 1, 129, 1, 0, 0, 0,
                             KvCacheStorage::BFloat16, 2, false, true);
              },
              {{"format", "bf16"}, {"execution", "graph"}, {"suite", "concurrency"}});
    suite.add("qsa_prepare_kv/fp8/concurrency/b9",
              [] {
                  return run(cuda_environment().context, 16, 9, 3, 0, 2, 0,
                             KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 1, true, true);
              },
              {{"format", "fp8"}, {"execution", "graph"}, {"suite", "concurrency"}});
    const auto add = [&](int t, int b, int axes, int alias, int layout, int capacity = 0,
                         KvCacheStorage storage = KvCacheStorage::BFloat16) {
        const std::string format = kv_cache_storage_name(storage);
        suite.add("qsa_prepare/" + format + "/t" + std::to_string(t) + "/b" + std::to_string(b) +
                      "/axes" + std::to_string(axes) + "/alias" + std::to_string(alias) +
                      "/layout" + std::to_string(layout) + "/capacity" + std::to_string(capacity),
                  [=] {
                      return run(cuda_environment().context, t, b, axes, alias, layout, capacity,
                                 storage);
                  },
                  {{"format", format}, {"execution", "graph"}});
    };
    for (int t : {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 33, 63, 64, 65, 128})
        add(t, t % 2 ? 8 : 1, t % 3 ? 3 : 1, t % 4, t % 3);
    for (int t : {319, 320, 321}) add(t, 1, 3, 3, 0);
    for (int t : {39, 40, 41}) add(t, 8, 1, 0, 2);
    for (int alias = 0; alias < 4; ++alias) add(9, 8, 3, alias, alias % 3, 5);
    for (int t : {1024, 2048, 4096}) add(t, 1, 3, 3, 0);
    suite.add("qsa_prepare/bf16/epsilon", [] { return epsilon_range(cuda_environment().context); },
              {{"format", "bf16"}});
    suite.add("qsa_prepare/bf16/coordinate_handoff",
              [] { return coordinate_handoff(cuda_environment().context); }, {{"format", "bf16"}});
    suite.add("qsa_prepare/bf16/norm_alignment",
              [] { return norm_alignment_and_phase(cuda_environment().context); },
              {{"format", "bf16"}});
    suite.add("qsa_prepare/bf16/continuation",
              [] { return continuation(cuda_environment().context); }, {{"format", "bf16"}});
    suite.add("qsa_prepare/bf16/chunking", [] { return chunking(cuda_environment().context); },
              {{"format", "bf16"}});
    suite.add("qsa_prepare/bf16/numerical_edges/key",
              [] { return numerical_edges(cuda_environment().context, true); },
              {{"format", "bf16"}});
    suite.add("qsa_prepare/bf16/numerical_edges/value",
              [] { return numerical_edges(cuda_environment().context, false); },
              {{"format", "bf16"}});
    suite.add("qsa_prepare/bf16/read_footprint",
              [] { return read_footprint(cuda_environment().context); }, {{"format", "bf16"}});
    suite.add("qsa_prepare/bf16/contract", [] { return validation(cuda_environment().context); },
              {{"format", "bf16"}});
    for (auto storage : {KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                         KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        for (int t : {1, 8, 17, 65, 319, 320, 321, 1024, 2048, 4096})
            add(t, 1, t % 3 ? 3 : 1, t % 4, t % 3, 0, storage);
        for (int t : {1, 8, 39, 40, 41}) add(t, 8, 3, t % 4, 2, 0, storage);
        add(9, 8, 1, 3, 1, 5, storage);
        suite.add(std::string("qsa_prepare/") + kv_cache_storage_name(storage) + "/key_encoding",
                  [=] { return exact_key_encoding(cuda_environment().context, storage); },
                  {{"format", kv_cache_storage_name(storage)}});
        suite.add(std::string("qsa_prepare/") + kv_cache_storage_name(storage) + "/scale_edges",
                  [=] { return scale_edges(cuda_environment().context, storage); },
                  {{"format", kv_cache_storage_name(storage)}});
        suite.add(std::string("qsa_prepare/") + kv_cache_storage_name(storage) + "/norm_alignment",
                  [=] { return norm_alignment_and_phase(cuda_environment().context, storage); },
                  {{"format", kv_cache_storage_name(storage)}});
        suite.add(std::string("qsa_prepare/") + kv_cache_storage_name(storage) + "/continuation",
                  [=] { return continuation(cuda_environment().context, storage); },
                  {{"format", kv_cache_storage_name(storage)}});
        suite.add(std::string("qsa_prepare/") + kv_cache_storage_name(storage) + "/chunking",
                  [=] { return chunking(cuda_environment().context, storage); },
                  {{"format", kv_cache_storage_name(storage)}});
        suite.add(std::string("qsa_prepare/") + kv_cache_storage_name(storage) + "/read_footprint",
                  [=] { return read_footprint(cuda_environment().context, storage); },
                  {{"format", kv_cache_storage_name(storage)}});
    }

    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                         KvCacheStorage::Fp8KeyNvfp4Value}) {
        const std::string prefix =
            std::string("qsa_prepare/") + kv_cache_storage_name(storage) + "/h12_1";
        for (auto [t, b] : {std::pair{1, 1},
                            {6, 4},
                            {17, 8},
                            {128, 1},
                            {255, 1},
                            {256, 1},
                            {257, 1},
                            {319, 1},
                            {320, 1},
                            {321, 1},
                            {1023, 1},
                            {1024, 1},
                            {1025, 1},
                            {4096, 1}})
            suite.add(prefix + "/t" + std::to_string(t) + "/b" + std::to_string(b),
                      [=] {
                          return run(cuda_environment().context, t, b, b == 1 ? 1 : 3, t % 4, t % 3,
                                     0, storage, 1);
                      },
                      {{"format", kv_cache_storage_name(storage)}, {"execution", "graph"}});
        suite.add(prefix + "/continuation",
                  [=] { return continuation(cuda_environment().context, storage, 1); },
                  {{"format", kv_cache_storage_name(storage)}});
        suite.add(prefix + "/chunking",
                  [=] { return chunking(cuda_environment().context, storage, 1); },
                  {{"format", kv_cache_storage_name(storage)}});
    }
    suite.add("qsa_prepare/bf16/h12_1/norm_alignment",
              [] {
                  return norm_alignment_and_phase(cuda_environment().context,
                                                  KvCacheStorage::BFloat16, 1);
              },
              {{"format", "bf16"}});

    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, KvCacheStorage::Nvfp4Group16,
                         KvCacheStorage::Fp8KeyNvfp4Value}) {
        const std::string prefix = std::string("qsa_prepare_kv/") + kv_cache_storage_name(storage);
        for (int heads : {1, 2})
            for (auto [t, b] : {std::pair{1, 1}, {9, 4}, {65, 1}, {257, 1}})
                suite.add(prefix + "/h" + std::to_string(heads) + "/t" + std::to_string(t),
                          [=] {
                              return run(cuda_environment().context, t, b, b == 1 ? 1 : 3, 0,
                                         b == 1 ? 1 : 2, b == 1 ? 0 : 5, storage, heads, true);
                          },
                          {{"format", kv_cache_storage_name(storage)}, {"suite", "kv"}});
        suite.add(prefix + "/continuation",
                  [=] { return continuation(cuda_environment().context, storage, 1, true); },
                  {{"format", kv_cache_storage_name(storage)}, {"suite", "kv"}});
        suite.add(prefix + "/chunking",
                  [=] { return chunking(cuda_environment().context, storage, 2, true); },
                  {{"format", kv_cache_storage_name(storage)}, {"suite", "kv"}});
        if (storage != KvCacheStorage::BFloat16) {
            suite.add(prefix + "/key_encoding",
                      [=] { return exact_key_encoding(cuda_environment().context, storage, true); },
                      {{"format", kv_cache_storage_name(storage)}, {"suite", "kv"}});
            suite.add(prefix + "/scale_edges",
                      [=] { return scale_edges(cuda_environment().context, storage, true); },
                      {{"format", kv_cache_storage_name(storage)}, {"suite", "kv"}});
        }
    }
    suite.add("qsa_prepare_kv/bf16/prefill",
              [] {
                  return run(cuda_environment().context, 4096, 1, 3, 0, 0, 0,
                             KvCacheStorage::BFloat16, 2, true);
              },
              {{"format", "bf16"}, {"suite", "kv"}});

#if NINFER_TARGET_RTX5090
    // Four-head FP8 cache: independent math and current-metadata replay, without
    // taking a Cartesian product of existing layout/alias/storage cases.
    for (bool kv_only : {false, true}) {
        const std::string prefix =
            std::string(kv_only ? "qsa_prepare_kv" : "qsa_prepare") + "/fp8/h24_4";

        struct Case {
            int t, b, axes, alias, layout, capacity;
        };

        std::vector<Case> cases{{1, 1, 1, 0, 0, 0},
                                {33, 8, 3, kv_only ? 0 : 3, 2, 5},
                                {kv_only ? 255 : 65, 1, 3, kv_only ? 0 : 1, 1, 0},
                                {257, 1, 3, 0, 0, 0},
                                {2048, 1, 1, 0, 0, 0}};
        if (kv_only) cases.push_back({256, 1, 3, 0, 0, 0});
        if (!kv_only) {
            cases.push_back({255, 1, 3, 2, 0, 0});
            cases.push_back({256, 1, 3, 3, 0, 0});
        }
        for (auto c : cases)
            suite.add(prefix + "/t" + std::to_string(c.t) + "/b" + std::to_string(c.b),
                      [=] {
                          return run(cuda_environment().context, c.t, c.b, c.axes, c.alias,
                                     c.layout, c.capacity, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                                     4, kv_only);
                      },
                      {{"format", "fp8"}, {"execution", "graph"}});
        suite.add(prefix + "/continuation",
                  [=] {
                      return continuation(cuda_environment().context,
                                          KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 4, kv_only);
                  },
                  {{"format", "fp8"}});
        suite.add(prefix + "/chunking",
                  [=] {
                      return chunking(cuda_environment().context,
                                      KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 4, kv_only);
                  },
                  {{"format", "fp8"}});
    }
    suite.add("qsa_prepare/fp8/h24_4/norm_alignment",
              [] {
                  return norm_alignment_and_phase(cuda_environment().context,
                                                  KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 4);
              },
              {{"format", "fp8"}});
    suite.add("qsa_prepare/fp8/h24_4/encoding_boundaries",
              [] {
                  auto& device = cuda_environment().context;
                  int failed   = 0;
                  for (bool kv_only : {false, true}) {
                      failed += exact_key_encoding(device, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                                                   kv_only, 4);
                      failed +=
                          scale_edges(device, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, kv_only, 4);
                      Fixture f(8, 1, 3, 0, 1, 0, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 4);
                      f.kv_only = kv_only;
                      for (float epsilon : {1e-6F, std::numeric_limits<float>::max()}) {
                          f.reset(2, true);
                          f.parameters.kv.rms_epsilon = epsilon;
                          const float amplitudes[]{0, 1e20F, 1e-30F, 1e18F,
                                                   6, 256,   1e30F,  bf16_to_f32(0x7f7f)};
                          for (int t = 0; t < 8; ++t)
                              for (int h = 0; h < 4; ++h)
                                  for (int c = 0; c < 256; ++c)
                                      f.k.at(c, h, t, 0) =
                                          f32_to_bf16((c % 3 - 1) * amplitudes[(t + h) % 8]);
                          f.k.upload();
                          f.launch(device.execution_view());
                          device.synchronize();
                          failed += verify(f, "four-head K normalization range");
                      }
                  }
                  return failed;
              },
              {{"format", "fp8"}});
    suite.add("qsa_prepare/fp8/h24_4/contract",
              [] {
                  Fixture f(1, 1, 1, 0, 0, 0, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0, 4);
                  auto e             = cuda_environment().context.execution_view();
                  int failed         = 0;
                  const auto rejects = [&](auto invoke) {
                      try {
                          invoke();
                          ++failed;
                      } catch (const std::invalid_argument&) {}
                  };
                  const auto q            = f.projected.query;
                  f.projected.query.ne[1] = 48;
                  rejects([&] { f.launch(e); });
                  f.projected.query = q;
                  return failed;
              },
              {{"format", "fp8"}});
    // Each additional codec covers both semantic entries and distinct cache/record layouts,
    // with one compact set of mathematical risks rather than a storage/layout cross product.
    for (auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                         KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value}) {
        const std::string format = kv_cache_storage_name(storage);
        for (bool kv_only : {false, true}) {
            const std::string prefix =
                std::string(kv_only ? "qsa_prepare_kv/" : "qsa_prepare/") + format + "/h24_4/";

            struct Case {
                int t, b, axes, alias, layout, capacity;
            };

            const Case cases[]{
                {1, 1, 1, 0, 0, 0}, {33, 8, 3, kv_only ? 0 : 3, 2, 5}, {257, 1, 3, 0, 1, 0}};
            for (auto c : cases)
                suite.add(prefix + "t" + std::to_string(c.t) + "/b" + std::to_string(c.b),
                          [=] {
                              return run(cuda_environment().context, c.t, c.b, c.axes, c.alias,
                                         c.layout, c.capacity, storage, 4, kv_only);
                          },
                          {{"format", format}, {"execution", "graph"}});
            suite.add(prefix + "continuation",
                      [=] { return continuation(cuda_environment().context, storage, 4, kv_only); },
                      {{"format", format}});
        }
        suite.add("qsa_prepare/" + format + "/h24_4/norm_alignment",
                  [=] { return norm_alignment_and_phase(cuda_environment().context, storage, 4); },
                  {{"format", format}});
        if (storage != KvCacheStorage::BFloat16)
            suite.add("qsa_prepare/" + format + "/h24_4/encoding_boundaries",
                      [=] {
                          auto& device = cuda_environment().context;
                          int failed   = 0;
                          for (bool kv_only : {false, true}) {
                              failed += exact_key_encoding(device, storage, kv_only, 4);
                              failed += scale_edges(device, storage, kv_only, 4);
                          }
                          return failed;
                      },
                      {{"format", format}});
    }

#endif

    return suite.main(argc, argv);
}
