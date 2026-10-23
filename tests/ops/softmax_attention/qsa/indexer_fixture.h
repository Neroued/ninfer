#pragma once

#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"
#include "../../../../tools/support/paged_kv_fixture.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ninfer::test::qsa::indexer {

enum class Pattern { Dyadic, Dense, Zero, Concentrated, NearBoundary };

inline std::uint32_t hash(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

inline std::uint16_t sample(std::uint32_t seed, Pattern pattern) {
    const auto h = hash(seed);
    float value  = (int(h % 257) - 128) / 64.0F;
    if (pattern == Pattern::Dense) value *= std::ldexp(1.0F, -int((h >> 12) % 9));
    return f32_to_bf16(value);
}

struct Fixture {
    int T, B, history, pages_per_row, pages, table_width = 4096, table_stride;
    Pattern pattern;
    std::int64_t key_row_stride, key_page_stride;
    std::vector<std::uint16_t> q_host, k_host;
    std::vector<int> positions, valid, table_rows, tables;
    std::vector<std::int64_t> table_offsets;
    std::unique_ptr<tool::PackedKVFixture> packed;
    GuardedDeviceBuffer q, pos, lengths, rows, table, ids, counts;
    ops::QsaIndexCacheView cache;
    ops::QsaBatchView batch;
    ops::QsaBlockSelection selected;
    Tensor query;
    ops::QsaIndexerExecutionEnvelope envelope{262144};

    Fixture(int tokens, int batch_size, int max_history, int layout, Pattern values)
        : T(tokens), B(batch_size), history(max_history), pages_per_row((history + 63) / 64),
          pages(pages_per_row * B + 2), table_stride(table_width), pattern(values),
          key_row_stride(128), key_page_stride(16 * key_row_stride),
          q_host(std::size_t(512) * T * B), k_host(key_page_stride * pages), positions(T * B),
          valid(B), table_rows(B), tables(table_stride * (B + 2)), q(q_host.size() * 2),
          pos(positions.size() * 4), lengths(B * 4), rows(B * 4), table(tables.size() * 8),
          ids(std::size_t(512) * T * B * 4), counts(std::size_t(T) * B * 4) {
        query  = Tensor(q.span(), DType::BF16, {128, 4, T, B});
        packed = std::make_unique<tool::PackedKVFixture>(
            q.device(), KVPageGeometry{64, {{DType::BF16, 128, 1, layout ? 2u : 256u, 4}}}, pages);
        Tensor bt(table.span(), DType::I64, {table_width, B + 2});
        cache    = {packed->plane(0, bt)};
        batch    = {Tensor(pos.span(), DType::I32, {T, B}), Tensor(lengths.span(), DType::I32, {B}),
                    Tensor(rows.span(), DType::I32, {B})};
        selected = {Tensor(ids.span(), DType::I32, {512, T, B}),
                    Tensor(counts.span(), DType::I32, {T, B})};
    }

    void reset(int phase, bool all_empty = false) {
        std::fill(k_host.begin(), k_host.end(), 0x7fc1);
        std::fill(q_host.begin(), q_host.end(), 0x7fc1);
        std::fill(tables.begin(), tables.end(), -1);
        std::fill(positions.begin(), positions.end(), -1);
        for (int b = 0; b < B; ++b) {
            int end = history;
            int n   = std::min(T, end);
            if (B > 1 || phase != 0) {
                switch ((b + phase) % 4) {
                case 0:
                    end = std::min(history, 3);
                    n   = std::min(T, end);
                    break;
                case 1:
                    end = std::min(history, 2055);
                    n   = std::min(T, end);
                    break;
                case 2:
                    n = std::min(T, 2);
                    break;
                default:
                    n = 0;
                    break;
                }
            }
            if (all_empty) n = 0;
            valid[b]      = n;
            const int row = 1 + (B - 1 - b + phase) % B;
            table_rows[b] = n ? row : -1;
            for (int page = 0; page < pages_per_row; ++page) {
                const int physical = (pages - 1 - (b * pages_per_row + page) + phase * 17) % pages;
                tables[row * table_stride + page] = physical;
                for (int r = 0; r < 16; ++r) {
                    const int g = page * 16 + r;
                    for (int d = 0; d < 128; ++d) {
                        std::uint16_t word =
                            sample(g * 1777 + d * 31 + b * 919 + phase * 313, pattern);
                        if (pattern == Pattern::Concentrated)
                            word = f32_to_bf16(d == 0 && g >= 16384 && g < 16896 ? 16.0F : 0.0F);
                        if (pattern == Pattern::NearBoundary)
                            word = f32_to_bf16(
                                d == 0 ? 1.0F : (d == 1 && g >= end / 4 - 512 ? 0x1p-14F : 0.0F));
                        k_host[std::int64_t(physical) * key_page_stride + r * key_row_stride + d] =
                            word;
                    }
                }
            }
            for (int t = 0; t < n; ++t) {
                const int m  = b * T + t;
                positions[m] = end - n + t;
                for (int h = 0; h < 4; ++h)
                    for (int d = 0; d < 128; ++d) {
                        auto word =
                            sample(t * 733 + b * 313 + h * 211 + d * 53 + phase * 991, pattern);
                        if (pattern == Pattern::Zero) word = 0;
                        if (pattern == Pattern::Concentrated || pattern == Pattern::NearBoundary)
                            word = f32_to_bf16(h == 0 && d <= 1 ? 1.0F : 0.0F);
                        q_host[std::size_t(m) * 512 + h * 128 + d] = word;
                    }
            }
        }
        q.copy_from_host(q_host.data(), q_host.size() * 2);
        packed->upload_plane(0, Tensor({k_host.data(), k_host.size() * 2, Device::cpu()},
                                       DType::BF16, {128, 16, 1, pages}));
        pos.copy_from_host(positions.data(), positions.size() * 4);
        lengths.copy_from_host(valid.data(), B * 4);
        rows.copy_from_host(table_rows.data(), B * 4);
        table_offsets = packed->table_offsets(tables);
        table.copy_from_host(table_offsets.data(), table_offsets.size() * 8);
        ids.fill(0xa5);
        counts.fill(0xa5);
        cuda_synchronize();
    }

    void launch(MemoryArena& workspace, CudaExecutionView execution) {
        ops::qsa_indexer(query, cache, batch, envelope, workspace, selected, execution);
    }
};
} // namespace ninfer::test::qsa::indexer
