#include "traffic.h"
#include "common/gpu_activity.cuh"

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        const auto o = kda::parse(argc, argv);
        measurement::hardware();
        auto& device = device_context();
        L2FlushBuffer flush(o.timing.flush_bytes);
        measurement::Csv csv(o.timing.csv_out,
                             std::string(kda::Traffic::columns) + ",kv_records,record_origin");
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
                                  (o.op == "record" ? (o.alias_records ? "alias" : "copy") : "") +
                                  "," +
                                  (f.projects_fold() || f.recorded_layer() || o.op == "chain" ||
                                           o.op == "input-record"
                                       ? "projection"
                                   : o.op == "fold" || o.op == "record" ? "provided"
                                                                        : "none"));
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
