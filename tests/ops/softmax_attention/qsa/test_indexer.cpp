#include "ops/suite.h"
#include "kv_cache_storage.h"
#include "indexer_fixture.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"

#include <numeric>
#include <string>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::qsa::indexer;

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
        bool replay = false, std::uint32_t capacity = 262144) {
    Fixture f(t, b, history, layout, pattern);
    f.envelope.max_visible_tokens = capacity;
    const auto bytes              = ops::qsa_indexer_workspace_capacity_bytes(1, t, b, f.envelope);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(256, bytes));
    MemoryArena workspace(scratch.span().subspan(0, std::max<std::size_t>(256, bytes)));
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

int graph_update(DeviceContext& device) {
    Fixture f(65, 1, 262144, 1, Pattern::Dyadic);
    const auto bytes = ops::qsa_indexer_workspace_capacity_bytes(65, 65, 1, {262144});
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena workspace(scratch.span());
    DecodeGraphExecutable graph;
    int failures = 0;
    for (std::uint32_t bound : {0u, 2051u, 2052u, 8192u, 32768u, 65540u, 131076u, 262144u, 2051u}) {
        f.history                     = std::max(1u, bound);
        f.envelope.max_visible_tokens = bound;
        f.reset(0, bound == 0);
        DecodeGraphDefinition definition;
        definition.capture(device.stream(), [&] { f.launch(workspace, device.execution_view()); });
        if (graph.ready())
            graph.update(definition);
        else
            graph.instantiate(definition);
        graph.launch(device.stream());
        device.synchronize();
        failures += verify(f, "indexer graph update bound=" + std::to_string(bound));
        failures += scratch.verify_guards("indexer graph workspace");
        if (workspace.used() != 0 || workspace.peak_used() > bytes) ++failures;
        if (failures) return failures;
    }
    return failures;
}

int read_footprint(DeviceContext& device) {
    Fixture f(8, 4, 32772, 0, Pattern::Dyadic);
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
        ops::qsa_indexer_workspace_capacity_bytes(8, 8, 4, f.envelope));
    MemoryArena workspace(workspace_storage.span());
    f.launch(workspace, device.execution_view());
    device.synchronize();
    return verify(f, "masked reads", false);
}

template <class F>
int rejects(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "indexer accepted invalid view/profile\n";
    return 1;
}

int validation(DeviceContext& device) {
    Fixture f(1, 1, 4, 0, Pattern::Dyadic);
    f.reset(0);
    ninfer::DeviceBuffer workspace_storage(
        ninfer::test::cuda_environment().context.device(),
        ops::qsa_indexer_workspace_capacity_bytes(1, 1, 1, f.envelope));
    MemoryArena workspace(workspace_storage.span());
    int failures =
        rejects([&] { (void)ops::qsa_indexer_workspace_capacity_bytes(2, 1, 1, f.envelope); });
    auto original = f.query;
    f.query.nb[1] += 2;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.query                   = original;
    auto output               = f.selected.block_ids;
    f.selected.block_ids.data = f.query.data;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.selected.block_ids          = output;
    f.envelope.max_visible_tokens = 262145;
    failures += rejects([&] { f.launch(workspace, device.execution_view()); });
    f.envelope.max_visible_tokens = 0;
    f.reset(0, true);
    if (ops::qsa_indexer_workspace_capacity_bytes(1, 1, 1, f.envelope) != 0) ++failures;
    ninfer::DeviceBuffer empty_storage(ninfer::test::cuda_environment().context.device(), 256);
    MemoryArena empty(empty_storage.span());
    f.launch(empty, device.execution_view());
    device.synchronize();
    failures += verify(f, "zero envelope");
    failures += empty.peak_used() != 0;
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    suite.add("qsa_indexer/graph_update", [] { return graph_update(cuda_environment().context); },
              {{"execution", "graph"}});
    suite.add("qsa_indexer/contract", [] { return validation(cuda_environment().context); });
    suite.add("qsa_indexer/read_footprint",
              [] { return read_footprint(cuda_environment().context); });
    const auto add = [&](int t, int b, int history, int layout, Pattern pattern, bool graph = false,
                         std::uint32_t capacity = 262144) {
        suite.add("qsa_indexer/t" + std::to_string(t) + "/b" + std::to_string(b) + "/l" +
                      std::to_string(history) + "/layout" + std::to_string(layout) + "/pattern" +
                      std::to_string(int(pattern)) + "/capacity" + std::to_string(capacity),
                  [=] {
                      return run(cuda_environment().context, t, b, history, layout, pattern, graph,
                                 capacity);
                  },
                  {{"execution", graph ? "graph" : "eager"}});
    };
    for (int history :
         {1, 3, 4, 5, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 32764, 32768, 32772, 262144})
        add(1, 1, history, 0, Pattern::Dyadic);
    for (int capacity : {2051, 2052, 16388})
        add(5, 3, capacity, 1, Pattern::Dyadic, true, capacity);
    for (int t : {2, 3, 4, 5, 8, 9, 16, 17, 31, 32, 33, 63, 64, 65})
        add(t, 1, 32772, t % 2, Pattern::Dyadic, t == 4 || t == 65);
    for (int b = 2; b <= 8; ++b) add(64 / b + 1, b, 32772, b % 2, Pattern::Dyadic, true);
    for (int t : {128, 1024, 2048, 4096}) add(t, 1, 262144, 0, Pattern::Dyadic);
    add(129, 1, 131076, 0, Pattern::Dense, true, 131076);
    add(65, 4, 65540, 0, Pattern::NearBoundary, true, 65540);
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
