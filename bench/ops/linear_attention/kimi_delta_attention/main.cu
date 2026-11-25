#include "traffic.h"
#include "common/gpu_activity.cuh"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
void run_ragged(const kda::Options& options, L2FlushBuffer& flush, CudaExecutionView cuda) {
    kda::RaggedWeights weights(options);
    measurement::Csv csv(options.timing.csv_out,
                         std::string(kda::Traffic::columns) +
                             ",record_origin,state_form,state_dtype");
    const auto extra = ",none," + std::string(ragged::state_form_name(options.state_form)) + "," +
                       kda::state_dtype_name(options.state_dtype);
    for (const auto& lengths : options.ragged_specs) {
        kda::RaggedFixture fixture(options, lengths, weights);
        const auto bytes = kda::traffic(fixture);
        measurement::Row row;
        row.op = options.op;
        if (fixture.input_update) {
            row.format = options.format == "mxfp8" ? "mxfp8" : "q8_g32_fp16";
            row.policy = weights.policy == ops::LinearPolicy::AllowA8 ? "a8" : "a16";
            row.n      = fixture.channels;
            row.k      = 4096;
        }
        row.tokens          = fixture.n;
        row.batch           = fixture.s;
        row.qk_heads        = fixture.qh;
        row.value_heads     = fixture.h;
        row.conv_channels   = fixture.channels;
        row.valid           = ragged::format_segments(lengths);
        row.workspace_bytes = fixture.capacity;
        for (const auto& execution :
             measurement::selections(options.timing.execution, "graph", "eager"))
            for (const auto& cache :
                 measurement::selections(options.timing.cache, "cold", "warm")) {
                const auto r =
                    measurement::measure([&](auto execution) { fixture.launch(execution); },
                                         [&](auto execution) {
                                             condition_gpu(execution, 200);
                                             fixture.restore(execution);
                                         },
                                         flush, cuda, options.timing, execution, cache);
                row.workspace_peak = fixture.workspace->peak_used();
                csv.write(row, r, execution, cache, bytes.csv(r.gpu.median_us) + extra);
            }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto o = kda::parse(argc, argv);
        measurement::hardware();
        auto& device = device_context();
        L2FlushBuffer flush(o.timing.flush_bytes);
        if (!o.ragged_specs.empty()) {
            run_ragged(o, flush, device.execution_view());
            return 0;
        }
        measurement::Csv csv(o.timing.csv_out,
                             std::string(kda::Traffic::columns) + ",record_origin,state_dtype");
        for (int t : o.tokens) {
            kda::Fixture f(o, t);
            const auto bytes = kda::traffic(f, o.op);
            measurement::Row row;
            row.op = o.op;
            if (o.op.starts_with("input-") || o.op == "chain" || f.full_layer()) {
                row.format = o.format == "mxfp8" ? "mxfp8" : "q8_g32_fp16";
                row.policy = f.input_policy() == ops::LinearPolicy::AllowA8 ? "a8" : "a16";
                row.n      = 12288;
                row.k      = 4096;
            }
            row.tokens        = t;
            row.batch         = o.batch;
            row.layers        = f.layers;
            row.qk_heads      = o.qk_heads;
            row.value_heads   = o.heads;
            row.conv_channels = f.channels;
            row.valid         = o.valid;
            row.commits       = o.commits;
            std::replace(row.valid.begin(), row.valid.end(), ',', ':');
            std::replace(row.commits.begin(), row.commits.end(), ',', ':');
            row.workspace_bytes = o.op == "fold" ? 0 : f.capacity;
            for (const auto& execution :
                 measurement::selections(o.timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                    const auto r =
                        measurement::measure([&](auto cuda) { f.launch(cuda); },
                                             [](auto cuda) { condition_gpu(cuda, 200); }, flush,
                                             device.execution_view(), o.timing, execution, cache);
                    row.workspace_peak = o.op == "fold" ? 0 : f.workspace->peak_used();
                    csv.write(row, r, execution, cache,
                              bytes.csv(r.gpu.median_us) + "," +
                                  (f.projects_fold() || f.recorded_layer() || o.op == "chain" ||
                                           o.op == "input-record"
                                       ? "projection"
                                   : o.op == "fold" || o.op == "record" ? "provided"
                                                                        : "none") +
                                  "," + kda::state_dtype_name(o.state_dtype));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
