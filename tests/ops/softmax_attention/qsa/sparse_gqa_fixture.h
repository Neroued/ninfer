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
    std::int64_t row, head, page;
    std::vector<std::uint16_t> host;
    tool::PackedKVFixture* packed = nullptr;
    std::size_t plane_index       = 0;
    Tensor view;

    Plane(int pages, DType dtype, int heads = 2)
        : row(256), head(64 * row), page(heads * head),
          host(page * pages, dtype == DType::BF16 ? 0x7fc1 : 0x7e01),
          view(Tensor::metadata(dtype, {256, 64, heads, pages})) {
        view.nb[1] = row * 2;
        view.nb[2] = head * 2;
        view.nb[3] = page * 2;
    }

    std::size_t offset(int d, int h, int r, int p) const {
        return d + h * head + r * row + p * page;
    }

    void bind(tool::PackedKVFixture& arena, std::size_t index) {
        packed      = &arena;
        plane_index = index;
        initialize_plane(arena, index, host.front(), view.ne[3]);
    }

    void upload(std::span<const int> dirty_pages) {
        if (packed)
            upload_pages(*packed, plane_index, std::span<const std::uint16_t>(host), dirty_pages);
    }

    std::vector<std::uint16_t> download() const {
        auto result   = host;
        auto target   = view;
        target.data   = result.data();
        target.device = Device::cpu();
        packed->download_plane(plane_index, target);
        return result;
    }
};

struct Fixture {
    int T, B, q_heads, kv_heads, history, table_width, table_stride, pages;
    bool dense_rows = false;
    int live_limit  = std::numeric_limits<int>::max();
    std::vector<std::uint16_t> qhost, ghost;
    std::vector<int> positions, valid, selectors, tables, ids, counts;
    std::vector<std::int64_t> table_offsets;
    std::unique_ptr<tool::PackedKVFixture> packed;
    std::vector<std::vector<int>> visible;
    Plane keys, values;
    std::vector<int> populated_pages, dirty_pages;
    KvCacheStorage storage;
    std::unique_ptr<EncodedVector> encoded_key, encoded_value;
    GuardedDeviceBuffer q, gate, out, pos, lengths, rows, table, selections, sizes;
    Tensor query, output_gate, output;
    PagedKVBatchLayerView cache;
    ops::QsaBatchView batch;
    ops::QsaBlockSelection selected;
    ops::ContextBucket bucket;

    Fixture(int tokens, int batch_size, int max_history, bool padded,
            KvCacheStorage format = KvCacheStorage::BFloat16, bool padded_scales = false,
            int query_heads = 24, int cache_heads = 0)
        : T(tokens), B(batch_size), q_heads(query_heads),
          kv_heads(cache_heads ? cache_heads : query_heads / 12), history(max_history),
          table_width((history + 63) / 64), table_stride(table_width),
          pages(std::min(table_width, 521) * B + 2), qhost(std::size_t(T) * B * (256 * q_heads)),
          ghost(qhost.size()), positions(T * B), valid(B), selectors(B),
          tables(table_stride * (B + 1)), ids(std::size_t(T) * B * 512), counts(T * B),
          visible(T * B), keys(pages, DType::BF16, kv_heads), values(pages, DType::FP16, kv_heads),
          storage(format), q(qhost.size() * 2), gate(ghost.size() * 2), out(qhost.size() * 2),
          pos(positions.size() * 4), lengths(B * 4), rows(B * 4), table(tables.size() * 8),
          selections(ids.size() * 4), sizes(counts.size() * 4),
          bucket(ops::select_context_bucket(max_history)) {
        query                       = Tensor(q.span(), DType::BF16, {256, q_heads, T, B});
        output_gate                 = Tensor(gate.span(), DType::BF16, {256, q_heads, T, B});
        output                      = Tensor(out.span(), DType::BF16, {256, q_heads, T, B});
        Tensor bt(table.span(), DType::I64, {table_width, B + 1});
        const auto formats = prepare::cache_encoding(storage);
        KVPageGeometry geometry;
        const auto add = [&](DType dtype, int features, std::size_t alignment) {
            geometry.planes.push_back({dtype, features, kv_heads, alignment, 1});
        };
        add(formats.key.data_dtype, formats.key.data_extent, padded ? 2 : 256);
        add(formats.value.data_dtype, formats.value.data_extent, padded ? 2 : 256);
        if (storage != KvCacheStorage::BFloat16) {
            add(formats.key.scale_dtype, formats.key.scale_extent, padded_scales ? 2 : 256);
            add(formats.value.scale_dtype, formats.value.scale_extent, padded_scales ? 2 : 256);
        }
        packed = std::make_unique<tool::PackedKVFixture>(q.device(), geometry, pages);
        cache  = {packed->plane(0, bt), packed->plane(1, bt), {}, {}, 256, kv_heads, storage};
        if (storage != KvCacheStorage::BFloat16) {
            encoded_key   = std::make_unique<EncodedVector>(formats.key, pages, kv_heads);
            encoded_value = std::make_unique<EncodedVector>(formats.value, pages, kv_heads);
            encoded_key->data.bind(*packed, 0);
            encoded_value->data.bind(*packed, 1);
            encoded_key->scales.bind(*packed, 2);
            encoded_value->scales.bind(*packed, 3);
            cache.k_scale_pages = packed->plane(2, bt);
            cache.v_scale_pages = packed->plane(3, bt);
        } else {
            keys.bind(*packed, 0);
            values.bind(*packed, 1);
        }
        batch    = {Tensor(pos.span(), DType::I32, {T, B}), Tensor(lengths.span(), DType::I32, {B}),
                    Tensor(rows.span(), DType::I32, {B})};
        selected = {Tensor(selections.span(), DType::I32, {512, T, B}),
                    Tensor(sizes.span(), DType::I32, {T, B})};
    }

    void reset(int phase = 0, int cap = 512, float magnitude = 1.0F,
               std::span<const int> packed_lengths = {}, std::span<const int> packed_ends = {},
               std::span<const int> caps = {}) {
        std::fill(positions.begin(), positions.end(), -1);
        std::fill(ids.begin(), ids.end(), -1);
        std::fill(counts.begin(), counts.end(), -1);
        for (auto& row : visible) row.clear();
        std::vector<int> ends(B);
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
            if (!packed_lengths.empty()) {
                live = packed_lengths[b];
                end  = packed_ends[b];
            }
            ends[b]      = end;
            valid[b]     = live;
            selectors[b] = live ? (b + phase) % B + 1 : -1;
            if (!live) continue;
            for (int p = 0; p < table_width; ++p)
                tables[selectors[b] * table_stride + p] =
                    (pages - 1 - (b * table_width + p) % pages + phase * 7) % pages;
        }
        // Sort the selected groups, then expand each four-token group in place. Keep
        // the unsorted IDs in the original routing order for the device input.
        std::vector<std::vector<std::uint64_t>> worker_pages(T * B);
        parallel_rows(T * B, qhost.size(), [&](int first, int last) {
            auto& populated = worker_pages[first];
            populated.resize(pages);
            for (int m = first; m < last; ++m) {
                const int b = m / T, t = m % T;
                if (t < valid[b]) {
                    const int p = ends[b] - valid[b] + t, C = (p + 1) / 4;
                    positions[m] = p;
                    counts[m]    = std::min(C, caps.empty() ? cap : caps[b]);
                    int step     = C > 0 ? (hash(m + phase * 71) % C) | 1 : 1;
                    while (C > 0 && std::gcd(step, C) != 1) step += 2;
                    std::array<int, 512> groups;
                    for (int i = 0; i < counts[m]; ++i) {
                        const int g = (std::int64_t(i) * step + hash(m + 31 + phase)) % C;
                        ids[std::size_t(m) * 512 + i] = g;
                        groups[i]                     = g;
                    }
                    std::sort(groups.begin(), groups.begin() + counts[m]);
                    auto& tokens = visible[m];
                    tokens.reserve(4 * counts[m] + (p + 1) % 4);
                    const auto physical_page = [&](int j) {
                        return tables[selectors[b] * table_stride + j / 64];
                    };
                    for (int i = 0; i < counts[m]; ++i) {
                        const int j = groups[i] * 4;
                        populated[physical_page(j)] |= std::uint64_t{15} << (j % 64);
                        for (int k = 0; k < 4; ++k) tokens.push_back(j + k);
                    }
                    for (int j = C * 4; j <= p; ++j) {
                        tokens.push_back(j);
                        populated[physical_page(j)] |= std::uint64_t{1} << (j % 64);
                    }
                }
                const bool live = !visible[m].empty();
                for (int i = 0; i < 256 * q_heads; ++i) {
                    const auto index = std::size_t(m) * (256 * q_heads) + i;
                    qhost[index] =
                        live ? f32_to_bf16(sample(index + 711 + phase) * magnitude) : 0x7fc1;
                    ghost[index] = live ? f32_to_bf16(sample(index + 129 + phase) * 4) : 0x7fc1;
                }
            }
        });
        std::vector<std::uint64_t> populated(pages);
        for (const auto& part : worker_pages)
            for (std::size_t p = 0; p < part.size(); ++p) populated[p] |= part[p];
        dirty_pages = populated_pages;
        populated_pages.clear();
        for (int p = 0; p < pages; ++p)
            if (populated[p]) populated_pages.push_back(p);
        dirty_pages.insert(dirty_pages.end(), populated_pages.begin(), populated_pages.end());
        std::sort(dirty_pages.begin(), dirty_pages.end());
        dirty_pages.erase(std::unique(dirty_pages.begin(), dirty_pages.end()), dirty_pages.end());
        parallel_rows(dirty_pages.size(), std::uint64_t(dirty_pages.size()) * keys.page,
                      [&](int first, int last) {
                          for (int i = first; i < last; ++i) {
                              const int page = dirty_pages[i];
                              for (int h = 0; h < kv_heads; ++h)
                                  for (int token = 0; token < 64; ++token) {
                                      const auto offset = keys.offset(0, h, token, page);
                                      if ((populated[page] & (std::uint64_t{1} << token)) == 0) {
                                          std::fill_n(keys.host.data() + offset, 256, 0x7fc1);
                                          std::fill_n(values.host.data() + offset, 256, 0x7e01);
                                          continue;
                                      }
                                      for (int d = 0; d < 256; ++d) {
                                          const auto seed = std::uint32_t(
                                              (page * 128 + h * 64 + token) * 256 + d);
                                          keys.host[offset + d] =
                                              f32_to_bf16(sample(seed + 701) * magnitude);
                                          values.host[offset + d] =
                                              half(packed_lengths.empty()
                                                       ? sample(seed + 1311)
                                                       : (page / table_width % 2 ? 1.0F : -1.0F) *
                                                             (0.55F + 0.06F * sample(seed + 1311)));
                                      }
                                  }
                          }
                      });
        upload();
    }

    void upload() {
        q.copy_from_host(qhost.data(), qhost.size() * 2);
        gate.copy_from_host(ghost.data(), ghost.size() * 2);
        keys.upload(dirty_pages);
        values.upload(dirty_pages);
        pos.copy_from_host(positions.data(), positions.size() * 4);
        lengths.copy_from_host(valid.data(), valid.size() * 4);
        rows.copy_from_host(selectors.data(), selectors.size() * 4);
        table_offsets = packed->table_offsets(tables);
        table.copy_from_host(table_offsets.data(), table_offsets.size() * 8);
        selections.copy_from_host(ids.data(), ids.size() * 4);
        sizes.copy_from_host(counts.data(), counts.size() * 4);
        if (encoded_key) {
            encoded_key->encode(keys, dirty_pages);
            encoded_value->encode(values, dirty_pages);
        }
        out.fill(0xff);
        // Fixture transfers/memset use the default stream; the Op's stream is nonblocking.
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    void launch(MemoryArena& arena, CudaExecutionView execution) {
        ops::qsa_sparse_gqa(query, output_gate, cache, batch, selected, bucket, arena, output,
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
                for (int i = first; i < last; ++i) {
                    const int page = rows[i] / 64, token = rows[i] % 64;
                    for (int h = 0; h < kv_heads; ++h) {
                        const auto index = std::size_t(i) * (256 * kv_heads) + h * 256;
                        if (encoded_key) {
                            encoded_key->decode_row(h, token, page, decoded_keys.data() + index);
                            encoded_value->decode_row(h, token, page,
                                                      decoded_values.data() + index);
                        } else {
                            const auto source = keys.offset(0, h, token, page);
                            for (int d = 0; d < 256; ++d) {
                                decoded_keys[index + d]   = bf16_to_f32(keys.host[source + d]);
                                decoded_values[index + d] = unhalf(values.host[source + d]);
                            }
                        }
                    }
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
