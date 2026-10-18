#pragma once

#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"
#include "sparse_gqa_cache_fixture.h"

#include <bit>
#include <memory>
#include <numeric>

namespace ninfer::test::qsa::sparse {

inline std::uint32_t hash(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

inline float sample(std::uint32_t i) { return (int(hash(i) % 10001) - 5000) / 3000.0F; }

inline std::uint16_t half(float x) { return std::bit_cast<std::uint16_t>(_Float16(x)); }

inline float unhalf(std::uint16_t x) { return float(std::bit_cast<_Float16>(x)); }

struct Plane {
    int leading;
    std::int64_t row, head, page;
    std::vector<std::uint16_t> host;
    GuardedDeviceBuffer device;
    Tensor view;

    Plane(int pages, DType dtype, int padding, int heads = 2)
        : leading(padding ? 1 : 0), row(256 + padding), head(64 * row + padding),
          page(heads * head + padding), host(leading + page * pages), device(host.size() * 2),
          view(device.span().subspan(leading * 2), dtype, {256, 64, heads, pages}) {
        view.nb[1] = row * 2;
        view.nb[2] = head * 2;
        view.nb[3] = page * 2;
    }

    std::size_t offset(int d, int h, int r, int p) const {
        return leading + d + h * head + r * row + p * page;
    }
};

struct Fixture {
    int T, B, q_heads, kv_heads, history, table_width, table_stride, pages;
    bool dense_rows = false;
    int live_limit  = std::numeric_limits<int>::max();
    std::vector<std::uint16_t> qhost, ghost;
    std::vector<int> positions, valid, selectors, tables, ids, counts;
    std::vector<std::vector<int>> visible;
    Plane keys, values;
    KvCacheStorage storage;
    std::unique_ptr<EncodedVector> encoded_key, encoded_value;
    GuardedDeviceBuffer q, gate, out, pos, lengths, rows, table, selections, sizes;
    Tensor query, output_gate, output;
    PagedKVBatchLayerView cache;
    ops::QsaBatchView batch;
    ops::QsaBlockSelection selected;
    ops::QsaSparseGqaExecutionEnvelope envelope;

    Fixture(int tokens, int batch_size, int max_history, bool padded,
            KvCacheStorage format = KvCacheStorage::BFloat16, bool padded_scales = false,
            int query_heads = 24)
        : T(tokens), B(batch_size), q_heads(query_heads), kv_heads(query_heads / 12),
          history(max_history), table_width((history + 63) / 64),
          table_stride(table_width + (padded ? 3 : 0)), pages(std::min(table_width, 521) * B + 2),
          qhost(std::size_t(T) * B * (256 * q_heads)), ghost(qhost.size()), positions(T * B),
          valid(B), selectors(B), tables(table_stride * (B + 1)), ids(std::size_t(T) * B * 512),
          counts(T * B), visible(T * B), keys(pages, DType::BF16, padded ? 1 : 0, kv_heads),
          values(pages, DType::FP16, padded ? 3 : 0, kv_heads), storage(format),
          q(qhost.size() * 2), gate(ghost.size() * 2), out(qhost.size() * 2),
          pos(positions.size() * 4), lengths(B * 4), rows(B * 4), table(tables.size() * 4),
          selections(ids.size() * 4), sizes(counts.size() * 4) {
        envelope.max_visible_tokens = history;
        query                       = Tensor(q.span(), DType::BF16, {256, q_heads, T, B});
        output_gate                 = Tensor(gate.span(), DType::BF16, {256, q_heads, T, B});
        output                      = Tensor(out.span(), DType::BF16, {256, q_heads, T, B});
        Tensor bt(table.span(), DType::I32, {table_width, B + 1});
        bt.nb[1] = table_stride * 4;
        cache    = {keys.view, values.view, {}, {}, bt, 256, kv_heads, KvCacheStorage::BFloat16};
        if (storage != KvCacheStorage::BFloat16) {
            const auto formats = prepare::cache_encoding(storage);
            encoded_key = std::make_unique<EncodedVector>(formats.key, pages, padded, padded_scales,
                                                          kv_heads);
            encoded_value       = std::make_unique<EncodedVector>(formats.value, pages, padded,
                                                                  padded_scales, kv_heads);
            cache.k_pages       = encoded_key->data.view;
            cache.v_pages       = encoded_value->data.view;
            cache.k_scale_pages = encoded_key->scales.view;
            cache.v_scale_pages = encoded_value->scales.view;
            cache.storage       = storage;
        }
        batch    = {Tensor(pos.span(), DType::I32, {T, B}), Tensor(lengths.span(), DType::I32, {B}),
                    Tensor(rows.span(), DType::I32, {B})};
        selected = {Tensor(selections.span(), DType::I32, {512, T, B}),
                    Tensor(sizes.span(), DType::I32, {T, B})};
    }

    void reset(int phase = 0, int cap = 512, float magnitude = 1.0F) {
        std::fill(qhost.begin(), qhost.end(), 0x7fc1);
        std::fill(ghost.begin(), ghost.end(), 0x7fc1);
        std::fill(keys.host.begin(), keys.host.end(), 0x7fc1);
        std::fill(values.host.begin(), values.host.end(), 0x7e01);
        std::fill(positions.begin(), positions.end(), -1);
        std::fill(ids.begin(), ids.end(), -1);
        std::fill(counts.begin(), counts.end(), -1);
        for (auto& row : visible) row.clear();
        std::vector<bool> populated(std::size_t(pages) * 64);
        for (int b = 0; b < B; ++b) {
            int end = history, live = std::min(T, end);
            if (phase == 1 || B > 1) {
                switch ((b + phase) % 4) {
                case 1:
                    end  = std::min(history, 2051);
                    live = std::min(T, end);
                    break;
                case 2:
                    end  = std::min(history, 3);
                    live = std::min(T, end);
                    break;
                case 3:
                    live = 0;
                    break;
                }
            }
            live = std::min(live, live_limit);
            if (phase == 2) live = 0;
            valid[b]     = live;
            selectors[b] = live ? (b + phase) % B + 1 : -1;
            if (!live) continue;
            for (int p = 0; p < table_width; ++p)
                tables[selectors[b] * table_stride + p] =
                    (pages - 1 - (b * table_width + p) % pages + phase * 7) % pages;
            for (int t = 0; t < live; ++t) {
                const int m = t + T * b, p = end - live + t, C = (p + 1) / 4;
                positions[m] = p;
                counts[m]    = std::min(C, cap);
                int step     = C > 0 ? (hash(m + phase * 71) % C) | 1 : 1;
                while (C > 0 && std::gcd(step, C) != 1) step += 2;
                for (int i = 0; i < counts[m]; ++i) {
                    const int g = (std::int64_t(i) * step + hash(m + 31 + phase)) % C;
                    ids[std::size_t(m) * 512 + i] = g;
                    for (int j = 0; j < 4; ++j) visible[m].push_back(g * 4 + j);
                }
                for (int j = C * 4; j <= p; ++j) visible[m].push_back(j);
                std::sort(visible[m].begin(), visible[m].end());
                if (visible[m].empty()) continue;
                for (int i = 0; i < (256 * q_heads); ++i) {
                    const auto index = std::size_t(m) * (256 * q_heads) + i;
                    qhost[index]     = f32_to_bf16(sample(index + 711 + phase) * magnitude);
                    ghost[index]     = f32_to_bf16(sample(index + 129 + phase) * 4);
                }
                for (int j : visible[m]) {
                    const int page                = tables[selectors[b] * table_stride + j / 64];
                    populated[page * 64 + j % 64] = true;
                }
            }
        }
        for (int row = 0; row < pages * 64; ++row) {
            if (!populated[row]) continue;
            const int page = row / 64, token = row % 64;
            for (int h = 0; h < kv_heads; ++h)
                for (int d = 0; d < 256; ++d) {
                    const auto seed = std::uint32_t((page * 128 + h * 64 + token) * 256 + d);
                    keys.host[keys.offset(d, h, token, page)] =
                        f32_to_bf16(sample(seed + 701) * magnitude);
                    values.host[values.offset(d, h, token, page)] = half(sample(seed + 1311));
                }
        }
        upload();
    }

    void upload() {
        q.copy_from_host(qhost.data(), qhost.size() * 2);
        gate.copy_from_host(ghost.data(), ghost.size() * 2);
        keys.device.copy_from_host(keys.host.data(), keys.host.size() * 2);
        values.device.copy_from_host(values.host.data(), values.host.size() * 2);
        pos.copy_from_host(positions.data(), positions.size() * 4);
        lengths.copy_from_host(valid.data(), valid.size() * 4);
        rows.copy_from_host(selectors.data(), selectors.size() * 4);
        table.copy_from_host(tables.data(), tables.size() * 4);
        selections.copy_from_host(ids.data(), ids.size() * 4);
        sizes.copy_from_host(counts.data(), counts.size() * 4);
        if (encoded_key) {
            encoded_key->encode(keys);
            encoded_value->encode(values);
        }
        out.fill(0xff);
        // Fixture transfers/memset use the default stream; the Op's stream is nonblocking.
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    void launch(MemoryArena& arena, CudaExecutionView execution) {
        ops::qsa_sparse_gqa(query, output_gate, cache, batch, selected, envelope, arena, output,
                            execution);
    }

    struct Reference {
        std::vector<int> queries;
        std::vector<double> values;
    };

    Reference reference() const {
        Reference result;
        for (int b = 0; b < B; ++b) {
            const int live = valid[b];
            if (live <= 64) {
                for (int t = 0; t < live; ++t) result.queries.push_back(b * T + t);
            } else {
                // Full dot/softmax/value formula at fixed output queries. The GPU still computes
                // every query; all output cells are checked for writes/finite values separately.
                std::vector<int> samples{0,  1,  7,   8,   15,  16,  31,       32,
                                         63, 64, 127, 128, 255, 256, live / 2, live - 1};
                std::sort(samples.begin(), samples.end());
                samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
                for (int t : samples)
                    if (t < live) result.queries.push_back(b * T + t);
            }
        }
        result.values.resize(result.queries.size() * (256 * q_heads));
        std::vector<int> row_indices(std::size_t(pages) * 64, -1);
        std::vector<int> rows;
        for (int m : result.queries)
            for (int token : visible[m]) {
                const int page = tables[selectors[m / T] * table_stride + token / 64];
                const int row  = page * 64 + token % 64;
                if (row_indices[row] >= 0) continue;
                row_indices[row] = int(rows.size());
                rows.push_back(row);
            }
        std::vector<double> decoded_keys(rows.size() * (256 * kv_heads)),
            decoded_values(decoded_keys.size());
        parallel_rows(
            int(rows.size()), std::uint64_t(rows.size()) * (256 * kv_heads),
            [&](int first, int last) {
                for (int i = first; i < last; ++i)
                    for (int h = 0; h < kv_heads; ++h)
                        for (int d = 0; d < 256; ++d) {
                            const int page = rows[i] / 64, token = rows[i] % 64;
                            const auto index = std::size_t(i) * (256 * kv_heads) + h * 256 + d;
                            decoded_keys[index] =
                                encoded_key ? encoded_key->decode(d, h, token, page)
                                            : double(bf16_to_f32(
                                                  keys.host[keys.offset(d, h, token, page)]));
                            decoded_values[index] =
                                encoded_value
                                    ? encoded_value->decode(d, h, token, page)
                                    : double(unhalf(values.host[values.offset(d, h, token, page)]));
                        }
            });
        parallel_rows(
            int(result.queries.size()),
            std::uint64_t(result.queries.size()) * (256 * q_heads) * 2048,
            [&](int first, int last) {
                for (int qi = first; qi < last; ++qi) {
                    const int m = result.queries[qi];
                    if (visible[m].empty()) continue;
                    std::vector<int> kv_rows;
                    for (int token : visible[m]) {
                        const int page = tables[selectors[m / T] * table_stride + token / 64];
                        kv_rows.push_back(row_indices[page * 64 + token % 64]);
                    }
                    std::vector<double> transformed_query(256 * q_heads);
                    for (int h = 0; h < q_heads; ++h) {
                        std::array<double, 256> row;
                        for (int d = 0; d < 256; ++d)
                            row[d] = bf16_to_f32(qhost[(std::size_t(m) * q_heads + h) * 256 + d]);
                        if (encoded_key) prepare::hadamard(row);
                        std::copy(row.begin(), row.end(), transformed_query.begin() + h * 256);
                    }
                    naive_dense_softmax_attention(
                        {256, q_heads, kv_heads}, 1, int(visible[m].size()), 1.0 / 16,
                        [&](int d, int h, int) { return transformed_query[h * 256 + d]; },
                        [&](int d, int h, int j) {
                            return decoded_keys[std::size_t(kv_rows[j]) * (256 * kv_heads) +
                                                h * 256 + d];
                        },
                        [&](int d, int h, int j) {
                            return decoded_values[std::size_t(kv_rows[j]) * (256 * kv_heads) +
                                                  h * 256 + d];
                        },
                        [](int, int) { return true; },
                        [&](int d, int h, int, double x) {
                            result.values[(std::size_t(qi) * q_heads + h) * 256 + d] = x;
                        },
                        result.queries.size() == 1);
                    for (int h = 0; h < q_heads; ++h) {
                        const auto base = (std::size_t(qi) * q_heads + h) * 256;
                        std::array<double, 256> row;
                        std::copy_n(result.values.begin() + base, 256, row.begin());
                        if (encoded_value && encoded_value->format.rotate) prepare::hadamard(row);
                        for (int d = 0; d < 256; ++d)
                            result.values[base + d] =
                                row[d] /
                                (1.0 + std::exp(-double(bf16_to_f32(
                                           ghost[(std::size_t(m) * q_heads + h) * 256 + d]))));
                    }
                }
            });
        return result;
    }
};
} // namespace ninfer::test::qsa::sparse
