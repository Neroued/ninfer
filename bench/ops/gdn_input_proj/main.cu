#include "fixture.cuh"

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        const auto options = gdn_input::parse_options(argc, argv);
        measurement::hardware();
        auto& device = ninfer::bench::device_context();
        gdn_input::Weights weights(options);
        L2FlushBuffer flush(options.timing.flush_bytes);
        measurement::Csv csv(options.timing.csv_out);
        for (int t : options.tokens) {
            gdn_input::Fixture fixture(options, weights.native, t);
            for (const auto& execution :
                 measurement::selections(options.timing.execution, "graph", "eager"))
                for (const auto& cache :
                     measurement::selections(options.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](CudaExecutionView cuda) { fixture.launch(cuda); },
                        [&](CudaExecutionView cuda) { fixture.restore(cuda); }, flush,
                        device.execution_view(), options.timing, execution, cache);
                    measurement::Row row;
                    row.op            = options.op;
                    row.format        = options.format;
                    row.policy        = options.policy;
                    row.n             = options.n;
                    row.k             = options.k;
                    row.tokens        = t;
                    row.batch         = options.batch;
                    row.conv_channels = fixture.channels;
                    row.valid         = "dense";
                    if (!options.valid.empty()) {
                        row.valid.clear();
                        for (int v : options.valid) {
                            if (!row.valid.empty()) row.valid += ':';
                            row.valid += std::to_string(v);
                        }
                    }
                    row.workspace_bytes = fixture.capacity;
                    row.workspace_peak  = fixture.arena.peak_used() - 256;
                    csv.write(row, result, execution, cache);
                }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
