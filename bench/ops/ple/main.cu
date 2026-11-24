#include "fixture.cuh"
#include "lookup_fixture.cuh"
#include "common/roofline.h"

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        const auto options = ple::parse_options(argc, argv);
        measurement::hardware();
        auto& device = ninfer::bench::device_context();
        L2FlushBuffer flush(options.timing.flush_bytes);
        const bool ragged_mode         = !options.ragged_specs.empty();
        const std::string state_column = ragged_mode ? ",state_form" : "";
        const auto state_value =
            ragged_mode ? "," + std::string(ragged::state_form_name(options.state_form)) : "";
        const auto sweep = [&](auto&& visit) {
            if (!ragged_mode) {
                for (int t : options.tokens)
                    visit(t, std::string{}, std::optional<ops::RaggedSequenceView>{});
            } else {
                for (const auto& lengths : options.ragged_specs) {
                    const ragged::RaggedDeviceFixture workload(lengths, device.device());
                    const auto seq = workload.view();
                    visit(seq.total_tokens(), ragged::format_segments(lengths), std::optional{seq});
                }
            }
        };
        if (options.op == "lookup") {
            ple::lookup::Table table(options.lookup_shards);
            measurement::Csv csv(
                options.timing.csv_out,
                "placement,table_bytes,selected_rows,unique_rows,encoded_row_bytes,"
                "logical_bytes,logical_gb_s,bandwidth_reference_us" +
                    state_column);
            sweep([&](int t, const std::string& tag, auto seq) {
                ple::LookupFixture fixture(options, t, table, seq);
                measurement::Row row;
                row.op     = "ngram_embedding_lookup";
                row.format = "nf4";
                row.tokens = t;
                row.batch  = seq ? seq->num_seqs() : options.batch;
                row.valid  = seq ? tag : ple::csv_list(fixture.counts);
                for (const auto& execution :
                     measurement::selections(options.timing.execution, "graph", "eager"))
                    for (const auto& cache :
                         measurement::selections(options.timing.cache, "cold", "warm")) {
                        const auto result = measurement::measure(
                            [&](CudaExecutionView cuda) { fixture.launch(cuda); }, [](auto) {},
                            flush, device.execution_view(), options.timing, execution, cache);
                        const auto bytes = fixture.logical_bytes();
                        csv.write(
                            row, result, execution, cache,
                            options.lookup_shards + ',' +
                                std::to_string(table.pages.size() * ple::lookup::kPageBytes) + ',' +
                                std::to_string(fixture.selected_rows) + ',' +
                                std::to_string(fixture.unique_rows) + ',' +
                                std::to_string(fixture.lookup_bytes) + ',' + std::to_string(bytes) +
                                ',' + format_metric(bytes / (result.gpu.median_us * 1000), 3, "") +
                                ',' + format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "") +
                                state_value);
                    }
            });
            return 0;
        }
        measurement::Csv csv(options.timing.csv_out,
                             "offset,inplace,logical_bytes,logical_gb_s,bandwidth_reference_us" +
                                 state_column);
        sweep([&](int t, const std::string& tag, auto seq) {
            measurement::Row row;
            row.op             = options.op;
            row.format         = "bf16";
            row.tokens         = t;
            row.batch          = seq ? seq->num_seqs() : options.batch;
            row.conv_channels  = ple::D;
            const bool inplace = seq && options.op == "state-extract"
                                     ? options.state_form == ragged::StateOffsetForm::Alias
                                     : options.inplace;
            const auto run     = [&](auto& fixture) {
                for (const auto& execution :
                     measurement::selections(options.timing.execution, "graph", "eager"))
                    for (const auto& cache :
                         measurement::selections(options.timing.cache, "cold", "warm")) {
                        const auto result = measurement::measure(
                            [&](CudaExecutionView cuda) { fixture.launch(cuda); },
                            [&](CudaExecutionView cuda) { fixture.restore(cuda); }, flush,
                            device.execution_view(), options.timing, execution, cache);
                        const auto bytes = fixture.logical_bytes();
                        csv.write(row, result, execution, cache,
                                      std::to_string(options.offset) + ',' + std::to_string(inplace) +
                                          ',' + std::to_string(bytes) + ',' +
                                          format_metric(bytes / (result.gpu.median_us * 1000), 3, "") +
                                          ',' +
                                          format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "") +
                                          state_value);
                    }
            };
            if (seq) {
                row.valid = tag;
                if (options.op == "state-extract") {
                    ple::RaggedExtractFixture fixture(options, *seq);
                    row.commits = tag;
                    run(fixture);
                } else {
                    ple::RaggedConvFixture fixture(options, *seq);
                    run(fixture);
                }
            } else if (options.op == "gate") {
                ple::GateFixture fixture(options, t);
                row.valid = ple::csv_list(fixture.counts);
                run(fixture);
            } else if (options.op == "state-extract") {
                ple::ExtractFixture fixture(options, t);
                row.valid   = ple::csv_list(fixture.counts);
                row.commits = ple::csv_list(fixture.commits);
                run(fixture);
            } else {
                ple::ConvFixture fixture(options, t);
                row.valid = ple::csv_list(fixture.counts);
                run(fixture);
            }
        });
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
