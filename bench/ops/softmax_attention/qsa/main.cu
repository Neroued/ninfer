#include "fixture.cuh"
#include "metrics.h"
#include "tail_fixture.cuh"
#include "prepare_fixture.cuh"
#include "indexer_fixture.cuh"
#include "sparse_gqa_fixture.cuh"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/qsa.h"

#include <fstream>

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

template <class Measure>
void for_each_shape(const qsa::Options& options, Device device, Measure measure) {
    if (options.packed_specs.empty()) {
        for (int tokens : options.tokens) measure(tokens, nullptr);
    } else {
        for (const auto& spec : options.packed_specs) {
            const qsa::RaggedBatchFixture batch(spec, device);
            measure(batch.view.ragged.total_tokens(), &batch);
        }
    }
}

std::string metric_columns(const qsa::Options& options) {
    std::string columns = qsa::Metrics::columns;
    if (options.graph_stats || !options.packed_specs.empty())
        columns += ",ragged_segments,ragged_histories,shared_prefix_tokens,state_alias,"
                   "completed_blocks,tail_write_bytes,graph_nodes,graph_kernels";
    return columns;
}

std::string metric_values(const qsa::Options& options, const qsa::RaggedBatchFixture* batch,
                          const qsa::Metrics& metrics, const measurement::Result& timing) {
    auto values = metrics.csv(timing.gpu.median_us);
    if (!batch && !options.graph_stats) return values;
    const char* alias = "";
    if (batch && (options.op == "prepare" || options.op == "prepare-kv")) {
        switch (options.state_alias) {
        case qsa::StateAlias::Disjoint:
            alias = "disjoint";
            break;
        case qsa::StateAlias::Inplace:
            alias = "inplace";
            break;
        case qsa::StateAlias::Mixed:
            alias = "mixed";
            break;
        }
    }
    return values + ',' + (batch ? ragged::format_segments(batch->spec.lengths) : "") + ',' +
           (batch ? ragged::format_segments(batch->spec.histories) : "") + ',' +
           std::to_string(options.shared_prefix_tokens) + ',' + alias + ',' +
           std::to_string(metrics.completed_blocks) + ',' + std::to_string(metrics.tail_write) +
           ',' + std::to_string(timing.nodes) + ',' + std::to_string(timing.kernels);
}

void measure_prepare(const qsa::Options& options, DeviceContext& device, L2FlushBuffer& flush) {
    measurement::Csv csv(options.timing.csv_out,
                         std::string("base_positions,axes,inplace,layout,round,graph_calls,") +
                             metric_columns(options));
    for_each_shape(options, device.device(), [&](int tokens, const qsa::RaggedBatchFixture* batch) {
        qsa::PrepareFixture fixture(options, tokens, batch);
        measurement::Row row;
        row.op              = fixture.kv_only ? (batch ? "qsa_prepare_kv_update" : "qsa_prepare_kv")
                                              : (batch ? "qsa_prepare_update" : "qsa_prepare");
        row.workspace_bytes = fixture.workspace_bytes;
        row.format          = qsa::name(options.storage);
        row.tokens          = tokens;
        row.batch           = batch ? batch->view.ragged.num_seqs() : options.batch;
        row.valid           = qsa::csv_list(fixture.counts);
        row.qk_heads        = fixture.kv_only ? 0 : qsa::prepare_query_heads(options.kv_heads);
        row.value_heads     = options.kv_heads;
        const auto metrics  = qsa::prepare_metrics(
            tokens, options.axes, qsa::prepare_query_heads(options.kv_heads), options.kv_heads,
            fixture.bases, fixture.counts,
            options.kv_heads *
                paged_kv_storage_layout(options.storage, 256).physical_bytes_per_token_head(),
            !fixture.kv_only, batch != nullptr);
        for (int round = 0; round < options.rounds; ++round)
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    auto result = measurement::measure(
                        [&](CudaExecutionView cuda) {
                            for (int i = 0; i < options.graph_calls; ++i) fixture.launch(cuda);
                        },
                        [&](CudaExecutionView cuda) { fixture.restore(cuda); }, flush,
                        device.execution_view(), options.timing, execution, cache);
                    result.gpu.median_us /= options.graph_calls;
                    result.gpu.min_us /= options.graph_calls;
                    result.gpu.p95_us /= options.graph_calls;
                    if (fixture.workspace) row.workspace_peak = fixture.workspace->peak_used();
                    csv.write(row, result, execution, cache,
                              qsa::csv_list(fixture.bases) + "," + std::to_string(options.axes) +
                                  "," + std::to_string(options.inplace) + "," + options.layout +
                                  "," + std::to_string(round) + "," +
                                  std::to_string(options.graph_calls) + "," +
                                  metric_values(options, batch, metrics, result));
                }
    });
}

void measure_indexer(const qsa::Options& options, DeviceContext& device, L2FlushBuffer& flush) {
    measurement::Csv csv(options.timing.csv_out,
                         std::string("contexts,bucket,context_capacity,score_pattern,round,") +
                             metric_columns(options));
    for_each_shape(options, device.device(), [&](int tokens, const qsa::RaggedBatchFixture* batch) {
        qsa::IndexerFixture fixture(options, tokens, device.execution_view(), batch);
        measurement::Row row;
        row.op              = "qsa_indexer";
        row.format          = "bf16";
        row.tokens          = tokens;
        row.batch           = batch ? batch->view.ragged.num_seqs() : options.batch;
        row.qk_heads        = 4;
        row.valid           = qsa::csv_list(fixture.lengths);
        row.workspace_bytes = fixture.workspace_bytes;
        const auto metrics  = qsa::indexer_metrics(tokens, fixture.contexts, fixture.lengths,
                                                   fixture.shared_prefix_tokens, batch != nullptr);
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
                                  std::string(ops::context_bucket_name(fixture.bucket)) + "," +
                                  std::to_string(ops::context_bucket_tokens(fixture.bucket)) + "," +
                                  options.score_pattern + "," + std::to_string(round) + "," +
                                  metric_values(options, batch, metrics, result));
                }
    });
}

void measure_sparse(const qsa::Options& options, DeviceContext& device, L2FlushBuffer& flush) {
    measurement::Csv csv(options.timing.csv_out,
                         std::string("contexts,bucket,context_capacity,selection,selected_blocks,"
                                     "physical_cache_bytes,round,graph_calls,") +
                             metric_columns(options));
    for_each_shape(options, device.device(), [&](int tokens, const qsa::RaggedBatchFixture* batch) {
        qsa::SparseGqaFixture fixture(options, tokens, device.execution_view(), batch);
        measurement::Row row;
        row.op              = "qsa_sparse_gqa";
        row.format          = qsa::name(options.storage);
        row.policy          = "bf16_output";
        row.tokens          = tokens;
        row.batch           = batch ? batch->view.ragged.num_seqs() : options.batch;
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
        metrics.read_bytes            = metrics.kv_read +
                             fixture.active_queries * (256 * options.query_heads) * 2 * 2 +
                             fixture.selected_ids * 4 + metrics.live_queries * 8 + row.batch * 4 +
                             fixture.active_rows * 4 + fixture.table_entries * 8;
        metrics.write_bytes =
            std::uint64_t(tokens) * (batch ? 1 : options.batch) * (256 * options.query_heads) * 2;
        for (int round = 0; round < options.rounds; ++round)
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    auto result = measurement::measure(
                        [&](CudaExecutionView cuda) {
                            for (int i = 0; i < options.graph_calls; ++i) fixture.launch(cuda);
                        },
                        [](auto) {}, flush, device.execution_view(), options.timing, execution,
                        cache);
                    result.gpu.median_us /= options.graph_calls;
                    result.gpu.min_us /= options.graph_calls;
                    result.gpu.p95_us /= options.graph_calls;
                    row.workspace_peak = fixture.workspace->peak_used();
                    csv.write(
                        row, result, execution, cache,
                        qsa::csv_list(fixture.contexts) + "," +
                            std::string(ops::context_bucket_name(fixture.bucket)) + "," +
                            std::to_string(ops::context_bucket_tokens(fixture.bucket)) + "," +
                            options.selection + "," + std::to_string(options.selected_blocks) +
                            "," + std::to_string(fixture.packed->storage().bytes) + "," +
                            std::to_string(round) + "," + std::to_string(options.graph_calls) +
                            "," + metric_values(options, batch, metrics, result));
                }
    });
}

void run(const qsa::Options& options, DeviceContext& device, L2FlushBuffer& flush) {
    if (options.op == "sparse-gqa") {
        measure_sparse(options, device, flush);
        return;
    }
    if (options.op == "indexer") {
        measure_indexer(options, device, flush);
        return;
    }
    if (options.op == "tail-extract") {
        measure_tail(options, device, flush);
        return;
    }
    if (options.op == "prepare" || options.op == "prepare-kv") {
        measure_prepare(options, device, flush);
        return;
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
                                        ops::linear_split(fixture.x, weights.weight, fixture.split,
                                                              cuda);
                                },
                                [](auto) {}, flush, device.execution_view(), options.timing,
                                execution, cache);
                            csv.write(row, result, execution, cache,
                                      std::string(qsa::name(projection)) + "," +
                                          std::to_string(round) + "," +
                                          qsa::projection_metrics_csv(weights.weight.payload_bytes,
                                                                      row.n, row.k, tokens,
                                                                      result.gpu.median_us));
                        }
                    }
            }
        }
    }
}

void run_jobs(const char* program, const char* path) {
    std::ifstream jobs(path);
    if (!jobs) throw std::runtime_error("cannot open QSA jobs file: " + std::string(path));

    measurement::hardware();
    auto& device = device_context();
    std::optional<L2FlushBuffer> flush;
    std::string line;
    std::size_t number = 0;
    while (std::getline(jobs, line)) {
        ++number;
        try {
            std::vector<char*> args{const_cast<char*>(program), line.data()};
            for (char& ch : line) {
                if (ch != '\t') continue;
                ch = '\0';
                args.push_back(&ch + 1);
            }
            const auto options = qsa::parse_options(static_cast<int>(args.size()), args.data());
            if (options.timing.csv_out.empty())
                throw std::invalid_argument("each QSA job requires --csv-out");
            if (!flush || flush->size() != options.timing.flush_bytes)
                flush.emplace(options.timing.flush_bytes);
            run(options, device, *flush);
            if (number == 1 || number % 10 == 0)
                std::cout << "[RUN] " << path << ": " << number << " jobs complete" << std::endl;
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string(path) + " line " + std::to_string(number) + ": " +
                                     error.what());
        }
    }
    if (!jobs.eof() || number == 0)
        throw std::runtime_error("cannot read QSA jobs from " + std::string(path));
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--jobs-file") {
            run_jobs(argv[0], argv[2]);
        } else {
            const auto options = qsa::parse_options(argc, argv);
            measurement::hardware();
            auto& device = device_context();
            L2FlushBuffer flush(options.timing.flush_bytes);
            run(options, device, flush);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
