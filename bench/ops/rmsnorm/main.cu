#include "fixture.cuh"
#include "common/roofline.h"
using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        const auto o = rmsnorm::parse(argc, argv);
        measurement::hardware();
        auto& device = ninfer::bench::device_context();
        L2FlushBuffer flush(o.timing.flush_bytes);
        measurement::Csv csv(o.timing.csv_out, "logical_bytes,logical_gb_s,bandwidth_reference_us");
        for (int columns : o.tokens) {
            rmsnorm::Fixture f(o, columns);
            for (const auto& execution :
                 measurement::selections(o.timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](CudaExecutionView cuda) { f.launch(cuda); }, [](auto) {}, flush,
                        device.execution_view(), o.timing, execution, cache);
                    measurement::Row row;
                    row.op          = rmsnorm::name(o.op);
                    row.format      = "bf16";
                    row.n           = o.dimension;
                    row.value_heads = o.heads;
                    row.tokens      = columns;
                    row.batch       = o.batch;
                    const auto bytes =
                        f.count * (f.gate.size() ? 6 : 4) + std::size_t(o.dimension) * 2;
                    const auto traffic =
                        std::to_string(bytes) + ',' +
                        format_metric(bytes / (result.gpu.median_us * 1000), 3, "") + ',' +
                        format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "");
                    csv.write(row, result, execution, cache, traffic);
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
