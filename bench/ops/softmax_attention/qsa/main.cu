#include "fixture.cuh"
#include "metrics.h"
#include "tail_fixture.cuh"
#include "prepare_fixture.cuh"
#include "indexer_fixture.cuh"
#include "sparse_gqa_fixture.cuh"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/qsa.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
void measure_tail(const qsa::Options& options, DeviceContext& device, L2FlushBuffer& flush) {
    measurement::Csv csv(options.timing.csv_out, "base_positions,inplace,layout,round");
    for (int capacity : options.capacities) {
        qsa::TailFixture fixture(options, capacity);
        measurement::Row row;
        row.op      = "qsa_tail_extract";
        row.format  = "bf16";
        row.tokens  = capacity;
        row.batch   = options.batch;
        row.valid   = qsa::csv_list(fixture.counts);
        row.commits = qsa::csv_list(fixture.commits);
        for (int round = 0; round < options.rounds; ++round)
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](CudaExecutionView cuda) { fixture.launch(cuda); },
                        [&](CudaExecutionView cuda) { fixture.restore(cuda); }, flush,
                        device.execution_view(), options.timing, execution, cache);
                    csv.write(row, result, execution, cache,
                              qsa::csv_list(fixture.bases) + "," + std::to_string(options.inplace) +
                                  "," + options.layout + "," + std::to_string(round));
                }
    }
}

void measure_prepare(const qsa::Options& options, DeviceContext& device, L2FlushBuffer& flush) {
    measurement::Csv csv(options.timing.csv_out,
                         std::string("base_positions,axes,inplace,layout,round,") +
                             qsa::Metrics::columns);
    for (int tokens : options.tokens) {
        qsa::PrepareFixture fixture(options, tokens);
        measurement::Row row;
        row.op             = fixture.kv_only ? "qsa_prepare_kv" : "qsa_prepare";
        row.format         = qsa::name(options.storage);
        row.tokens         = tokens;
        row.batch          = options.batch;
        row.valid          = qsa::csv_list(fixture.counts);
        row.qk_heads       = fixture.kv_only ? 0 : 12 * options.kv_heads;
        row.value_heads    = options.kv_heads;
        const auto metrics = qsa::prepare_metrics(
            tokens, options.axes, options.kv_heads, fixture.bases, fixture.counts,
            options.kv_heads *
                paged_kv_storage_layout(options.storage, 256).physical_bytes_per_token_head(),
            !fixture.kv_only);
        for (int round = 0; round < options.rounds; ++round)
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](CudaExecutionView cuda) { fixture.launch(cuda); },
                        [&](CudaExecutionView cuda) { fixture.restore(cuda); }, flush,
                        device.execution_view(), options.timing, execution, cache);
                    csv.write(row, result, execution, cache,
                              qsa::csv_list(fixture.bases) + "," + std::to_string(options.axes) +
                                  "," + std::to_string(options.inplace) + "," + options.layout +
                                  "," + std::to_string(round) + "," +
                                  metrics.csv(result.gpu.median_us));
                }
    }
}

void measure_indexer(const qsa::Options& options, DeviceContext& device, L2FlushBuffer& flush) {
    measurement::Csv csv(options.timing.csv_out,
                         std::string("contexts,context_capacity,layout,score_pattern,round,") +
                             qsa::Metrics::columns);
    for (int tokens : options.tokens) {
        qsa::IndexerFixture fixture(options, tokens, device.execution_view());
        measurement::Row row;
        row.op              = "qsa_indexer";
        row.format          = "bf16";
        row.tokens          = tokens;
        row.batch           = options.batch;
        row.qk_heads        = 4;
        row.valid           = qsa::csv_list(fixture.lengths);
        row.workspace_bytes = fixture.workspace_bytes;
        const auto metrics  = qsa::indexer_metrics(tokens, fixture.contexts, fixture.lengths);
        for (int round = 0; round < options.rounds; ++round)
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](CudaExecutionView cuda) { fixture.launch(cuda); }, [](auto) {}, flush,
                        device.execution_view(), options.timing, execution, cache);
                    row.workspace_peak = fixture.workspace->peak_used();
                    csv.write(row, result, execution, cache,
                              qsa::csv_list(fixture.contexts) + "," +
                                  std::to_string(fixture.envelope.max_visible_tokens) + "," +
                                  options.layout + "," + options.score_pattern + "," +
                                  std::to_string(round) + "," + metrics.csv(result.gpu.median_us));
                }
    }
}

void measure_sparse(const qsa::Options& options, DeviceContext& device, L2FlushBuffer& flush) {
    measurement::Csv csv(options.timing.csv_out,
                         std::string("contexts,context_capacity,layout,selection,selected_blocks,"
                                     "physical_cache_bytes,round,") +
                             qsa::Metrics::columns);
    for (int tokens : options.tokens) {
        qsa::SparseGqaFixture fixture(options, tokens, device.execution_view());
        measurement::Row row;
        row.op              = "qsa_sparse_gqa";
        row.format          = qsa::name(options.storage);
        row.policy          = "bf16_output";
        row.tokens          = tokens;
        row.batch           = options.batch;
        row.qk_heads        = options.query_heads;
        row.value_heads     = options.kv_heads;
        row.valid           = qsa::csv_list(fixture.lengths);
        row.workspace_bytes = fixture.workspace_bytes;
        qsa::Metrics metrics;
        metrics.live_queries =
            std::accumulate(fixture.lengths.begin(), fixture.lengths.end(), 0ULL);
        metrics.selected_tokens = fixture.selected_tokens;
        metrics.qk_flops        = metrics.pv_flops =
            fixture.selected_tokens * 2 * options.query_heads * 256;
        metrics.kv_read               = fixture.unique_tokens * fixture.token_bytes;
        metrics.per_query_cache_bytes = fixture.selected_tokens * fixture.token_bytes;
        metrics.read_bytes =
            metrics.kv_read + fixture.active_queries * (256 * options.query_heads) * 2 * 2 +
            fixture.selected_ids * 4 + metrics.live_queries * 8 + options.batch * 4 +
            fixture.active_rows * 4 + fixture.table_entries * 4;
        metrics.write_bytes =
            std::uint64_t(tokens) * options.batch * (256 * options.query_heads) * 2;
        for (int round = 0; round < options.rounds; ++round)
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](CudaExecutionView cuda) { fixture.launch(cuda); }, [](auto) {}, flush,
                        device.execution_view(), options.timing, execution, cache);
                    row.workspace_peak = fixture.workspace->peak_used();
                    csv.write(row, result, execution, cache,
                              qsa::csv_list(fixture.contexts) + "," +
                                  std::to_string(fixture.envelope.max_visible_tokens) + "," +
                                  options.layout + "," + options.selection + "," +
                                  std::to_string(options.selected_blocks) + "," +
                                  std::to_string(
                                      fixture.key_data.size() + fixture.value_data.size() +
                                      fixture.key_scales.size() + fixture.value_scales.size()) +
                                  "," + std::to_string(round) + "," +
                                  metrics.csv(result.gpu.median_us));
                }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = qsa::parse_options(argc, argv);
        measurement::hardware();
        auto& device = ninfer::bench::device_context();
        L2FlushBuffer flush(options.timing.flush_bytes);
        if (options.op == "sparse-gqa") {
            measure_sparse(options, device, flush);
            return 0;
        }
        if (options.op == "indexer") {
            measure_indexer(options, device, flush);
            return 0;
        }
        if (options.op == "tail-extract") {
            measure_tail(options, device, flush);
            return 0;
        }
        if (options.op == "prepare" || options.op == "prepare-kv") {
            measure_prepare(options, device, flush);
            return 0;
        }
        measurement::Csv csv(options.timing.csv_out,
                             "projection,round,logical_bytes,useful_flops,logical_GBps,"
                             "logical_TFLOps,memory_reference_us,compute_reference_us");
        for (auto projection : options.projections) {
            qsa::Weights weights(projection, options.kv_heads);
            for (int tokens : options.tokens) {
                qsa::Fixture fixture(weights, tokens);
                measurement::Row row;
                row.format = weights.weight.qtype == QType::BF16 ? "bf16" : "q8";
                row.policy = "a16";
                row.n      = weights.parent_rows;
                row.k      = qsa::kInputRows;
                row.tokens = tokens;
                for (int round = 0; round < options.rounds; ++round) {
                    auto entries = measurement::selections(options.entry, "linear", "split");
                    if (round % 2) std::reverse(entries.begin(), entries.end());
                    for (const auto& execution :
                         measurement::selections(options.timing.execution, "graph", "eager"))
                        for (const auto& cache :
                             measurement::selections(options.timing.cache, "cold", "warm")) {
                            for (const auto& entry : entries) {
                                const bool contiguous = entry == "linear";
                                row.op                = contiguous ? "linear" : "linear_split";
                                const auto result     = measurement::measure(
                                    [&](CudaExecutionView cuda) {
                                        if (contiguous)
                                            ops::linear(fixture.x, weights.weight, fixture.packed,
                                                            cuda);
                                        else
                                            ops::linear_split(fixture.x, weights.weight,
                                                                  fixture.split, cuda);
                                    },
                                    [](auto) {}, flush, device.execution_view(), options.timing,
                                    execution, cache);
                                csv.write(row, result, execution, cache,
                                          std::string(qsa::name(projection)) + "," +
                                              std::to_string(round) + "," +
                                              qsa::projection_metrics_csv(
                                                  weights.weight.payload_bytes, row.n, row.k,
                                                  tokens, result.gpu.median_us));
                            }
                        }
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
