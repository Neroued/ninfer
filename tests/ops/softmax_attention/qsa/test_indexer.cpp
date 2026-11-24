#include "ops/suite.h"
#include "packed_fixture.h"
#include "ops/graph_patch_check.h"
#include "kv_cache_storage.h"
#include "indexer_fixture.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"

#include <cuda.h>

#include <initializer_list>
#include <numeric>
#include <string>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::qsa::indexer;
using ops::ContextBucket;

namespace {
// Fixed score-error criterion over represented inputs: 128-term dot products, ReLU and
// the four-head sum. Every implementation uses this same ranking-error budget.
constexpr double kU          = 0x1p-24;
constexpr double kScoreGamma = (132 * kU) / (1 - 132 * kU);

int verify(Fixture& f, const std::string& label, bool preserved = true) {
    auto counts  = from_device<int>(f.counts.data(), f.T * f.B);
    auto ids     = from_device<int>(f.ids.data(), std::size_t(512) * f.T * f.B);
    int failures = 0;
    for (int b = 0; b < f.B; ++b)
        for (int t = 0; t < f.T; ++t) {
            const int m     = t + b * f.T;
            const int c     = t < f.valid[b] ? (f.positions[m] + 1) / 4 : 0;
            const int count = std::min(c, 512);
            if (counts[m] != count) {
                std::cerr << label << " count " << b << '/' << t << ' ' << counts[m]
                          << " != " << count << '\n';
                return 1;
            }
            std::vector<int> got(ids.begin() + std::size_t(m) * 512,
                                 ids.begin() + std::size_t(m) * 512 + count);
            std::sort(got.begin(), got.end());
            if ((!got.empty() && (got.front() < 0 || got.back() >= c)) ||
                std::adjacent_find(got.begin(), got.end()) != got.end()) {
                std::cerr << label << " invalid/duplicate selection " << b << '/' << t << '\n';
                return 1;
            }
            if (c <= 512) {
                for (int r = 0; r < count; ++r) failures += got[r] != r;
                continue;
            }
            // Large physical widths sample query rows, never the candidate history of a row.
            if (f.T > 65 && t > 1 && t != 3 && t != 7 && t != 63 && t != 64 && t != 65 &&
                t != f.T / 2 && t != f.valid[b] - 1)
                continue;
            double query[4][128];
            for (int h = 0; h < 4; ++h)
                for (int d = 0; d < 128; ++d)
                    query[h][d] = bf16_to_f32(f.q_host[std::size_t(m) * 512 + h * 128 + d]);
            std::vector<double> score(c), error(c);
            for (int g = 0; g < c; ++g) {
                const int page = f.tables[f.table_rows[b] * f.table_stride + g / 16];
                const auto base =
                    std::int64_t(page) * f.key_page_stride + (g % 16) * f.key_row_stride;
                double dots[4]{}, magnitude = 0;
                for (int d = 0; d < 128; ++d) {
                    const double key = bf16_to_f32(f.k_host[base + d]);
                    for (int h = 0; h < 4; ++h) {
                        const double product = query[h][d] * key;
                        dots[h] += product;
                        magnitude += std::abs(product);
                    }
                }
                for (double dot : dots) score[g] += std::max(dot, 0.0);
                error[g] = kScoreGamma * magnitude;
            }
            if (f.pattern != Pattern::Dense) {
                std::vector<int> expected(c);
                std::iota(expected.begin(), expected.end(), 0);
                std::partial_sort(expected.begin(), expected.begin() + 512, expected.end(),
                                  [&](int a, int b) {
                                      return score[a] > score[b] || (score[a] == score[b] && a < b);
                                  });
                expected.resize(512);
                std::sort(expected.begin(), expected.end());
                if (got != expected) {
                    std::cerr << label << " exact mathematical Top512 mismatch " << b << '/' << t
                              << '\n';
                    return 1;
                }
            } else {
                std::vector<bool> selected(c);
                for (int g : got) selected[g] = true;
                double excluded_lower = -INFINITY;
                for (int g = 0; g < c; ++g)
                    if (!selected[g])
                        excluded_lower = std::max(excluded_lower, score[g] - error[g]);
                for (int g : got)
                    if (score[g] + error[g] < excluded_lower) {
                        std::cerr << label
                                  << " selected/excluded score inversion exceeds FP32 bound " << b
                                  << '/' << t << '\n';
                        return 1;
                    }
            }
        }
    if (preserved) {
        failures += verify_exact(
            "query preserved", from_device<std::uint16_t>(f.query.data, f.q_host.size()), f.q_host);
        failures += verify_exact(
            "keys preserved",
            [&] {
                auto bytes = f.packed->download_plane(0);
                std::vector<std::uint16_t> words(bytes.size() / 2);
                std::memcpy(words.data(), bytes.data(), bytes.size());
                return words;
            }(),
            f.k_host);
        failures +=
            verify_exact("positions preserved",
                         from_device<int>(f.batch.positions.data, f.positions.size()), f.positions);
        failures += verify_exact("tables preserved",
                                 from_device<std::int64_t>(f.cache.keys.records.page_offsets.data,
                                                           f.table_offsets.size()),
                                 f.table_offsets);
        failures += verify_exact("counts metadata preserved",
                                 from_device<int>(f.batch.valid_columns.data, f.B), f.valid);
        failures += verify_exact("selectors preserved",
                                 from_device<int>(f.batch.table_rows.data, f.B), f.table_rows);
    }
    for (auto* buffer : {&f.q, &f.pos, &f.lengths, &f.rows, &f.table, &f.ids, &f.counts})
        failures += buffer->verify_guards(label);
    failures += !f.packed->guards_intact();
    return failures;
}

int run(DeviceContext& device, int t, int b, int history, int layout, Pattern pattern,
        bool replay = false, std::optional<ContextBucket> bucket = {}, int table_width = 0) {
    Fixture f(t, b, history, layout, pattern, bucket, table_width);
    const auto bytes = ops::qsa_indexer_workspace_capacity_bytes(f.bucket, b, 1, t);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(256, bytes));
    MemoryArena workspace(scratch.span().subspan(0, bytes));
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures     = 0;
    const int phases = replay ? 6 : 1;
    for (int phase = 0; phase < phases; ++phase) {
        f.reset(phase % 4, phase == 4);
        // Seed stale high-score bytes once; subsequent replays reuse actual previous scratch.
        if (phase == 0) scratch.fill(0x7f);
        cuda_synchronize();
        if (replay && phase == 0) {
            definition.capture(device.stream(),
                               [&] { f.launch(workspace, device.execution_view()); });
            graph.instantiate(definition);
        }
        if (replay)
            graph.launch(device.stream());
        else
            f.launch(workspace, device.execution_view());
        device.synchronize();
        const auto label = "indexer T=" + std::to_string(t) + " B=" + std::to_string(b) +
                           " L=" + std::to_string(history) + " layout=" + std::to_string(layout) +
                           " phase=" + std::to_string(phase);
        failures += verify(f, label);
        failures += scratch.verify_guards(label);
        if (workspace.used() != 0 || workspace.peak_used() > bytes) ++failures;
        if (failures) return failures;
    }
    return failures;
}

template <class F>
int rejects(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "indexer accepted invalid view/profile\n";
    return 1;
}

int ragged_indexer(std::vector<int> lengths, int layout, Pattern pattern, ContextBucket bucket) {
    RaggedFixture ragged(std::move(lengths));
    const int width = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    const int n = ragged.tokens(), s = ragged.segments();
    std::vector<int> ends(s);
    for (int b = 0; b < s; ++b) {
        ends[b] = 2044 + 4 * (b % 3);
        if (bucket == ContextBucket::K2) ends[b] = std::min(ends[b], 2048);
    }
    // Exercise both score partitions in one native window on a context spanning 4096 pools.
    if (s == 2 && width == 66) ends = {2120, 16404};
    if (s == 2 && width == 127) ends[1] = 2055;
    if (s == 2 && width == 128) ends = {2180, 2048};
    const int history = *std::max_element(ends.begin(), ends.end());
    Fixture f(width, s, history, layout, pattern, bucket);
    f.reset(0, ends, ragged.lengths);
    for (int b = 0; b < s; ++b) {
        const int c = ends[b] / 4;
        if (c <= 512) {
            // These rows do not resolve any key or table, including in a scored launch.
            const int row = f.table_rows[b];
            std::fill_n(f.tables.begin() + row * f.table_stride, f.table_width, -1);
            const auto base = std::size_t(b) * width * 512;
            std::fill_n(f.q_host.begin() + base, std::size_t(width) * 512, 0x7fc1);
        } else if (c % 16) {
            const int page = f.tables[f.table_rows[b] * f.table_stride + c / 16];
            std::fill_n(f.k_host.begin() + page * f.key_page_stride + (c % 16) * 128,
                        (16 - c % 16) * 128, 0x7f80);
        }
    }
    f.q.copy_from_host(f.q_host.data(), f.q_host.size() * 2);
    f.packed->upload_plane(0, Tensor({f.k_host.data(), f.k_host.size() * 2, Device::cpu()},
                                     DType::BF16, {128, 16, 1, f.pages}));
    f.table_offsets = f.packed->table_offsets(f.tables);
    f.table.copy_from_host(f.table_offsets.data(), f.table_offsets.size() * 8);

    const auto query_shape  = Tensor::metadata(DType::BF16, {128, 4, n});
    const auto ids_shape    = Tensor::metadata(DType::I32, {512, n});
    const auto counts_shape = Tensor::metadata(DType::I32, {n});
    const auto prefix       = qsa::prefix_bytes({ragged.view.cu_seqlens, ragged.view.seq_lengths,
                                                 ragged.view.token_to_seq, query_shape, counts_shape,
                                                 f.batch.table_rows, ids_shape, counts_shape});
    const auto bytes        = ops::qsa_indexer_ragged_workspace_capacity_bytes(bucket, n, n, s, s);
    GuardedDeviceBuffer backing(prefix + bytes);
    MemoryArena arena(backing.span());
    const auto seq        = qsa::packed_metadata(arena, ragged);
    const auto packed_q   = ragged.pack(f.q_host, width, 512);
    const auto packed_pos = ragged.pack(f.positions, width, 1);
    const Tensor query    = qsa::packed_tensor(arena, DType::BF16, {128, 4, n}, packed_q.data());
    const ops::QsaRaggedBatchView batch{
        seq, qsa::packed_tensor(arena, DType::I32, {n}, packed_pos.data()),
        qsa::packed_tensor(arena, DType::I32, {s}, f.table_rows.data())};
    ops::QsaBlockSelection selection{qsa::packed_tensor(arena, DType::I32, {512, n}),
                                     qsa::packed_tensor(arena, DType::I32, {n})};
    qsa::finish_prefix(arena, prefix);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    const auto execution = cuda_environment().default_execution;
    auto alias           = selection;
    alias.block_counts   = seq.token_to_seq;
    int failures =
        rejects([&] { ops::qsa_indexer(query, f.cache, batch, bucket, arena, alias, execution); });
    if (bytes) {
        auto short_storage = backing.span().subspan(prefix, bytes - 1);
        MemoryArena short_arena(short_storage);
        try {
            ops::qsa_indexer(query, f.cache, batch, bucket, short_arena, selection, execution);
            ++failures;
        } catch (const std::bad_alloc&) { failures += short_arena.used() != 0; }
    }
    ops::qsa_indexer(query, f.cache, batch, bucket, arena, selection, execution);
    cuda_synchronize();
    const auto padded_ids =
        ragged.unpack(from_device<int>(selection.block_ids.data, 512 * n), width, 512);
    const auto padded_counts =
        ragged.unpack(from_device<int>(selection.block_counts.data, n), width, 1);
    f.ids.copy_from_host(padded_ids.data(), padded_ids.size() * 4);
    f.counts.copy_from_host(padded_counts.data(), padded_counts.size() * 4);
    return failures + verify(f, "ragged QSA indexer") +
           verify_exact("QSA token mapping preserved", from_device<int>(seq.token_to_seq.data, n),
                        ragged.mapping) +
           backing.verify_guards("ragged indexer workspace") +
           (arena.used() != prefix || arena.peak_used() != prefix + bytes);
}

int ragged_indexer_graph() {
    RaggedFixture ragged({2, 3});
    constexpr int width = 3;
    const int n         = ragged.tokens();
    Fixture f(width, 2, 2060, 0, Pattern::Dyadic, ContextBucket::K8);
    const auto execution = cuda_environment().context.execution_view();
    GuardedDeviceBuffer q(n * 512 * 2), pos(n * 4), rows(2 * 4), ids(n * 512 * 4), counts(n * 4);
    const Tensor query(q.span(), DType::BF16, {128, 4, n});
    const ops::QsaRaggedBatchView batch{ragged.view, Tensor(pos.span(), DType::I32, {n}),
                                        Tensor(rows.span(), DType::I32, {2})};
    ops::QsaBlockSelection selected{Tensor(ids.span(), DType::I32, {512, n}),
                                    Tensor(counts.span(), DType::I32, {n})};
    const auto bytes = ops::qsa_indexer_ragged_workspace_capacity_bytes(f.bucket, n, n, 2, 2);
    GuardedDeviceBuffer backing(bytes);
    MemoryArena arena(backing.span());
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (const int phase : {0, 1}) {
        const std::array ends = phase == 0 ? std::array{2060, 2048} : std::array{2048, 2060};
        f.reset(phase, ends, ragged.lengths);
        const int inert = 1 - phase;
        const int row   = f.table_rows[inert];
        std::fill_n(f.table_offsets.begin() + row * f.table_stride, f.table_width, -1);
        f.table.copy_from_host(f.table_offsets.data(), f.table_offsets.size() * 8);
        std::fill_n(f.q_host.begin() + inert * width * 512, width * 512, 0x7fc1);
        const auto packed_q   = ragged.pack(f.q_host, width, 512);
        const auto packed_pos = ragged.pack(f.positions, width, 1);
        q.copy_from_host(packed_q.data(), packed_q.size() * 2);
        pos.copy_from_host(packed_pos.data(), packed_pos.size() * 4);
        rows.copy_from_host(f.table_rows.data(), f.table_rows.size() * 4);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        if (phase == 0) {
            definition.capture(execution.stream(), [&] {
                ops::qsa_indexer(query, f.cache, batch, f.bucket, arena, selected, execution);
            });
            graph.instantiate(definition);
        }
        graph.launch(execution.stream());
        cuda_synchronize(execution.stream());
        const auto got_ids    = ragged.unpack(from_device<int>(ids.data(), n * 512), width, 512);
        const auto got_counts = ragged.unpack(from_device<int>(counts.data(), n), width, 1);
        f.ids.copy_from_host(got_ids.data(), got_ids.size() * 4);
        f.counts.copy_from_host(got_counts.data(), got_counts.size() * 4);
        failures += verify(f, "packed threshold replay", false);
    }
    return failures + backing.verify_guards("packed replay scratch") +
           (arena.used() != 0 || arena.peak_used() != bytes);
}

int ragged_capacity() {
    constexpr ops::ContextBucketRange range{ContextBucket::K2, ContextBucket::K256};
    int failures = 0;
    for (const auto interval : {std::array{1, 9, 1, 9}, std::array{3, 10, 5, 8},
                                std::array{60, 132, 4, 12}, std::array{127, 129, 129, 150}}) {
        const auto [lo, hi, min_s, max_s] = interval;
        range.for_each([&](ContextBucket bucket) {
            std::size_t high_water = 0;
            for (int n = lo; n <= hi; ++n)
                for (int segments = min_s; segments <= std::min(n, max_s); ++segments)
                    high_water = std::max(
                        high_water, ops::qsa_indexer_workspace_capacity_bytes(bucket, 1, n, n));
            failures += high_water != ops::qsa_indexer_ragged_workspace_capacity_bytes(
                                          bucket, lo, hi, min_s, max_s);
        });
    }
    return failures + rejects([&] {
               (void)ops::qsa_indexer_ragged_workspace_capacity_bytes(ContextBucket::K512, 1, 1, 1,
                                                                      1);
           });
}

int flat_graph_patch(DeviceContext& device) {
    constexpr int B = 4, T = 4;
    constexpr ops::ContextBucketRange buckets{ContextBucket::K2, ContextBucket::K256};
    Fixture f(T, B, 262144, 0, Pattern::Dyadic);
    const auto bytes = ops::qsa_indexer_workspace_capacity_bytes(buckets, B, T, T);
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena workspace(scratch.span());
    const auto cuda = device.execution_view();
    std::vector<CapturedKernelSequence> captures;
    std::vector<ops::OpGraphProfile> profiles;
    buckets.for_each([&](ContextBucket bucket) {
        f.bucket = bucket;
        captures.emplace_back(cuda, [&](auto execution) { f.launch(workspace, execution); });
        profiles.push_back(
            ops::qsa_indexer_graph_profile(bucket, B, T, cuda.multiprocessor_count()));
    });
    for (std::size_t i = 0; i < captures.size(); ++i) {
        if (captures[i].params().size() != 3) {
            std::cerr << "expected three kernel nodes in every captured bucket\n";
            return 1;
        }
        if (std::find(profiles.begin(), profiles.begin() + i, profiles[i]) !=
            profiles.begin() + i) {
            std::cerr << "buckets with different launch parameters shared a profile\n";
            return 1;
        }
    }

    constexpr std::array hops{ContextBucket::K2,  ContextBucket::K8,   ContextBucket::K16,
                              ContextBucket::K64, ContextBucket::K256, ContextBucket::K16,
                              ContextBucket::K2,  ContextBucket::K32,  ContextBucket::K128};
    for (bool whole_graph : {false, true}) {
        FlatGraphPatchHarness harness(captures.front());
        for (const auto bucket : hops) {
            const auto index = static_cast<std::size_t>(bucket);
            const int limit  = ops::context_bucket_tokens(bucket);
            const std::array contexts{0, 384, std::min(limit, 2400),
                                      bucket == ContextBucket::K64 ? 40000 : limit};
            const std::array lengths{0, 3, 1, 4};
            f.reset(index, contexts, lengths);
            if (whole_graph)
                harness.update_whole_graph_and_launch(captures[index], cuda);
            else
                harness.patch_and_launch(captures[index], cuda);
            device.synchronize();
            const auto label = std::string(whole_graph ? "whole graph update " : "node patch ") +
                               std::string(ops::context_bucket_name(bucket));
            if (const int failures = verify(f, label); failures) return failures;
        }
    }
    return scratch.verify_guards("flat graph workspace") + (workspace.used() != 0) +
           (workspace.peak_used() != bytes);
}

int graph_update(DeviceContext& device, int tokens, int batch, int history_limit, int layout,
                 Pattern pattern, std::initializer_list<int> histories) {
    Fixture f(tokens, batch, history_limit, layout, pattern);
    const auto bytes = ops::qsa_indexer_workspace_capacity_bytes(
        ops::ContextBucketRange::through(history_limit), batch, tokens, tokens);
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena workspace(scratch.span());
    DecodeGraphExecutable graph;
    int failures = 0;
    for (const int history : histories) {
        f.history = history;
        f.bucket  = ops::select_context_bucket(history);
        f.reset(batch > 1 ? 1 : 0, history == 0);
        DecodeGraphDefinition definition;
        definition.capture(device.stream(), [&] { f.launch(workspace, device.execution_view()); });
        if (graph.ready())
            graph.update(definition);
        else
            graph.instantiate(definition);
        graph.launch(device.stream());
        device.synchronize();
        failures += verify(f, "indexer graph update history=" + std::to_string(history));
        failures += scratch.verify_guards("indexer graph workspace");
        if (workspace.used() || workspace.peak_used() > bytes) ++failures;
        if (failures) return failures;
    }
    return failures;
}

int read_footprint(DeviceContext& device) {
    Fixture f(4, 17, 16388, 0, Pattern::Dyadic);
    f.reset(0);
    DeviceBuffer query(ninfer::test::cuda_environment().context.device(), f.q_host.size() * 2);
    DeviceBuffer keys(ninfer::test::cuda_environment().context.device(), f.packed->storage().bytes);
    DeviceBuffer positions(ninfer::test::cuda_environment().context.device(),
                           f.positions.size() * 4);
    DeviceBuffer table(ninfer::test::cuda_environment().context.device(),
                       f.table_offsets.size() * 8);
    f.query.data                           = query.data();
    f.cache.keys.records.storage           = keys.span();
    f.cache.keys.records.page_offsets.data = table.data();
    f.batch.positions.data                 = positions.data();
    for (int b = 0; b < f.B; ++b) {
        const int n = f.valid[b];
        if (!n) continue;
        CUDA_CHECK(cudaMemcpyAsync(static_cast<int*>(positions.data()) + b * f.T,
                                   f.positions.data() + b * f.T, n * 4, cudaMemcpyHostToDevice,
                                   device.stream()));
        for (int t = 0; t < n; ++t) {
            if ((f.positions[b * f.T + t] + 1) / 4 <= 512) continue;
            const auto m = std::size_t(b * f.T + t) * 512;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint16_t*>(query.data()) + m,
                                       f.q_host.data() + m, 1024, cudaMemcpyHostToDevice,
                                       device.stream()));
        }
        const int c = (f.positions[b * f.T + n - 1] + 1) / 4;
        if (c <= 512) continue;
        const int page_count = (c + 15) / 16, row = f.table_rows[b];
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int64_t*>(table.data()) + row * f.table_stride,
                                   f.table_offsets.data() + row * f.table_stride, page_count * 8,
                                   cudaMemcpyHostToDevice, device.stream()));
        for (int p = 0; p < page_count; ++p) {
            const auto offset =
                std::int64_t(f.tables[row * f.table_stride + p]) * f.key_page_stride;
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(keys.data()) +
                                           f.table_offsets[row * f.table_stride + p] +
                                           f.cache.keys.plane.tensor.region.offset,
                                       f.k_host.data() + offset, std::min(16, c - p * 16) * 256,
                                       cudaMemcpyHostToDevice, device.stream()));
        }
    }
    ninfer::DeviceBuffer workspace_storage(
        ninfer::test::cuda_environment().context.device(),
        ops::qsa_indexer_workspace_capacity_bytes(f.bucket, f.B, f.T, f.T));
    MemoryArena workspace(workspace_storage.span());
    f.launch(workspace, device.execution_view());
    device.synchronize();
    return verify(f, "masked reads", false);
}

int validation(DeviceContext& device) {
    Fixture f(1, 1, 4, 0, Pattern::Dyadic);
    f.reset(0);
    const auto cuda = device.execution_view();
    MemoryArena workspace({nullptr, 0, cuda.device()});
    constexpr auto k2 = ContextBucket::K2;
    int failures =
        rejects([&] { (void)ops::qsa_indexer_workspace_capacity_bytes(k2, 1, 2, 1); }) +
        rejects([&] { (void)ops::qsa_indexer_workspace_capacity_bytes(k2, 1, 0, 1); }) +
        rejects([&] { (void)ops::qsa_indexer_workspace_capacity_bytes(k2, 0, 1, 1); }) +
        rejects([&] {
            (void)ops::qsa_indexer_workspace_capacity_bytes({ContextBucket::K8, k2}, 1, 1, 1);
        }) +
        rejects([&] { (void)ops::qsa_indexer_graph_profile(k2, 1, 1, 0); }) +
        rejects([&] { (void)ops::qsa_indexer_graph_profile(k2, 1, 0, 1); }) +
        rejects([&] { (void)ops::qsa_indexer_graph_profile(k2, 0, 1, 1); });
    const auto original = f.query;
    f.query.nb[1] += 2;
    failures += rejects([&] { f.launch(workspace, cuda); });
    f.query = original;
    for (int axis : {2, 3}) {
        f.query.ne[axis] = 0;
        failures += rejects([&] { f.launch(workspace, cuda); });
        f.query = original;
    }
    auto output               = f.selected.block_ids;
    f.selected.block_ids.data = f.query.data;
    failures += rejects([&] { f.launch(workspace, cuda); });
    f.selected.block_ids = output;
    for (int axis : {0, 1}) {
        auto& table      = f.cache.keys.records.page_offsets;
        const auto shape = table.ne[axis];
        table.ne[axis]   = 0;
        failures += rejects([&] { f.launch(workspace, cuda); });
        table.ne[axis] = shape;
    }
    for (const auto bucket : {ContextBucket::K512, ContextBucket::M1, ContextBucket{255}}) {
        f.bucket = bucket;
        failures += rejects([&] { f.launch(workspace, cuda); });
        failures +=
            rejects([&] { (void)ops::qsa_indexer_workspace_capacity_bytes(bucket, 1, 1, 1); });
        failures += rejects([&] { (void)ops::qsa_indexer_graph_profile(bucket, 1, 1, 1); });
    }
    return failures;
}

int zero_workspace(DeviceContext& device) {
    Fixture f(4, 3, 2048, 0, Pattern::Dyadic);
    f.reset(0, std::array{2048, 191, 0}, std::array{4, 2, 0});
    const auto cuda = device.execution_view();
    MemoryArena workspace({nullptr, 0, cuda.device()});
    if (ops::qsa_indexer_workspace_capacity_bytes(f.bucket, f.B, 1, f.T) != 0) return 1;

    // Device VA with no mapping makes any accidental Q, key or table read fault.
    CUmemAllocationProp properties{};
    properties.type         = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location     = {CU_MEM_LOCATION_TYPE_DEVICE, cuda.device().index()};
    std::size_t granularity = 0;
    if (cuInit(0) != CUDA_SUCCESS ||
        cuMemGetAllocationGranularity(&granularity, &properties,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        throw std::runtime_error("unable to query device VA granularity for K2");
    const auto bytes =
        std::max({f.query.bytes(), f.packed->storage().bytes,
                  f.cache.keys.records.page_offsets.bytes(), f.batch.table_rows.bytes()});

    struct Reservation {
        CUdeviceptr address = 0;
        std::size_t bytes;

        ~Reservation() {
            if (address) (void)cuMemAddressFree(address, bytes);
        }
    } reserved{0, ((bytes + granularity - 1) / granularity) * granularity};

    if (cuMemAddressReserve(&reserved.address, reserved.bytes, 0, 0, 0) != CUDA_SUCCESS)
        throw std::runtime_error("unable to reserve unmapped device addresses for K2");
    auto* unreadable                       = reinterpret_cast<void*>(reserved.address);
    f.query.data                           = unreadable;
    f.cache.keys.records.storage.data      = unreadable;
    f.cache.keys.records.page_offsets.data = unreadable;
    f.batch.table_rows.data                = unreadable;
    f.launch(workspace, cuda);
    device.synchronize();
    return verify(f, "K2 with unreadable scoring operands", false) + (workspace.peak_used() != 0);
}

int single_partition_workspace(DeviceContext& device) {
    int failures = 0;
    for (const auto bucket : {ContextBucket::K8, ContextBucket::K16}) {
        const int history = ops::context_bucket_tokens(bucket);
        Fixture f(4, 3, history, 0, Pattern::Dyadic, bucket);
        const auto bytes = ops::qsa_indexer_workspace_capacity_bytes(bucket, f.B, 1, f.T);
        if (bytes != std::size_t(history / 4) * f.B * f.T * sizeof(float)) {
            std::cerr << "single-partition workspace must contain only scores\n";
            return 1;
        }
        GuardedDeviceBuffer scratch(bytes);
        MemoryArena workspace(scratch.span());
        f.reset(0, std::array{0, 384, history}, std::array{0, 3, 4});
        f.launch(workspace, device.execution_view());
        device.synchronize();
        failures += verify(f, "single-partition workspace");
        failures += scratch.verify_guards("scores-only workspace");
        failures += workspace.used() != 0 || workspace.peak_used() != bytes;
    }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("qsa_indexer/ragged/window_127_2",
              [] { return ragged_indexer({127, 2}, 1, Pattern::Dense, ContextBucket::K8); });
    suite.add("qsa_indexer/ragged/window_63_66",
              [] { return ragged_indexer({63, 66}, 0, Pattern::Dyadic, ContextBucket::K32); });
    suite.add("qsa_indexer/ragged/contained_score_window",
              [] { return ragged_indexer({128, 2}, 0, Pattern::Dyadic, ContextBucket::K32); });
    suite.add("qsa_indexer/ragged/threshold_ties", [] {
        return ragged_indexer(std::vector<int>(160, 1), 0, Pattern::Zero, ContextBucket::K32);
    });
    suite.add("qsa_indexer/ragged/simt", [] {
        return ragged_indexer(std::vector<int>(160, 1), 1, Pattern::Dyadic, ContextBucket::K32);
    });
    suite.add("qsa_indexer/ragged/select_all", [] {
        return ragged_indexer(std::vector<int>(160, 1), 0, Pattern::Dense, ContextBucket::K2);
    });
    suite.add("qsa_indexer/ragged/single",
              [] { return ragged_indexer({65}, 0, Pattern::Dyadic, ContextBucket::K2); });
    suite.add("qsa_indexer/ragged/capacity", ragged_capacity);
    suite.add("qsa_indexer/ragged/graph_threshold", ragged_indexer_graph, {{"execution", "graph"}});
    suite.add("qsa_indexer/flat_graph_patch",
              [] { return flat_graph_patch(cuda_environment().context); },
              {{"execution", "graph"}});
    suite.add("qsa_indexer/graph_update",
              [] {
                  return graph_update(cuda_environment().context, 129, 1, 262144, 1,
                                      Pattern::Dyadic,
                                      {0, 2048, 2052, 8192, 32768, 65540, 131076, 262144, 2051});
              },
              {{"execution", "graph"}});
    suite.add("qsa_indexer/batched_graph_update/t1/b129",
              [] {
                  return graph_update(cuda_environment().context, 1, 129, 16388, 0, Pattern::Dyadic,
                                      {0, 2051, 16388, 2052, 0, 4096});
              },
              {{"execution", "graph"}, {"suite", "concurrency"}});
    suite.add("qsa_indexer/batched_graph_update/t4/b17",
              [] {
                  return graph_update(cuda_environment().context, 4, 17, 16388, 0, Pattern::Dense,
                                      {0, 2051, 16388, 16384, 0, 4096});
              },
              {{"execution", "graph"}, {"suite", "concurrency"}});
    suite.add("qsa_indexer/contract", [] { return validation(cuda_environment().context); });
    suite.add("qsa_indexer/read_footprint",
              [] { return read_footprint(cuda_environment().context); });
    suite.add("qsa_indexer/k2_small_page_table", [] {
        return run(cuda_environment().context, 4, 1, 192, 0, Pattern::Dyadic, false,
                   ContextBucket::K2, 4);
    });
    suite.add("qsa_indexer/k2_zero_workspace",
              [] { return zero_workspace(cuda_environment().context); });
    suite.add("qsa_indexer/single_partition_workspace",
              [] { return single_partition_workspace(cuda_environment().context); });
    const auto add = [&](int t, int b, int history, int layout, Pattern pattern, bool graph = false,
                         std::optional<ContextBucket> bucket = {}) {
        std::vector<std::pair<std::string, std::string>> tags{
            {"execution", graph ? "graph" : "eager"}};
        if (b > 8) tags.emplace_back("suite", "concurrency");
        suite.add(
            "qsa_indexer/t" + std::to_string(t) + "/b" + std::to_string(b) + "/l" +
                std::to_string(history) + "/layout" + std::to_string(layout) + "/pattern" +
                std::to_string(int(pattern)) + "/bucket" +
                std::string(
                    ops::context_bucket_name(bucket.value_or(ops::select_context_bucket(history)))),
            [=] {
                return run(cuda_environment().context, t, b, history, layout, pattern, graph,
                           bucket);
            },
            std::move(tags));
    };
    for (int history : {1,    3,    4,    5,    512,   2047,  2048,  2049,  2050,  2051,  2052,
                        2053, 4096, 8192, 8448, 16384, 16640, 32764, 32768, 32772, 65536, 262144})
        add(1, 1, history, 0, Pattern::Dyadic);
    for (int capacity : {2051, 2052, 16388}) add(5, 3, capacity, 1, Pattern::Dyadic, true);
    for (int t : {2, 3, 4, 5, 8, 9, 16, 17, 31, 32, 33, 63, 64, 65})
        add(t, 1, 32772, t % 2, Pattern::Dyadic, t == 4 || t == 65);
    for (int b = 2; b <= 8; ++b) add(64 / b + 1, b, 32772, b % 2, Pattern::Dyadic, true);
    add(16, 17, 4096, 1, Pattern::Dyadic, true);
    add(1, 65, 2055, 0, Pattern::Dyadic, false);
    for (int t : {128, 1024, 2048, 4096}) add(t, 1, 262144, 0, Pattern::Dyadic);
    add(129, 1, 131076, 0, Pattern::Dense, true);
    add(65, 4, 65540, 0, Pattern::NearBoundary, true);
    add(2, 1, 262144, 0, Pattern::Dense, true);
    for (auto pattern :
         {Pattern::Dense, Pattern::Zero, Pattern::Concentrated, Pattern::NearBoundary}) {
        add(1, 1, 262144, 0, pattern);
        add(8, 4, 131072, 1, pattern);
        if (pattern == Pattern::Dense) {
            add(16, 1, 32772, 0, pattern);
            add(32, 1, 32772, 1, pattern);
        }
    }

    return suite.main(argc, argv);
}
