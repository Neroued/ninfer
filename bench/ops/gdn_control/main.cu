#include "fixture.cuh"
#include <optional>
using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) {
    try {
        const auto o = gdn_control::parse(argc, argv);
        measurement::hardware();
        auto& device = ninfer::bench::device_context();
        std::optional<gdn_control::Weights> weights;
        if (o.op != gdn_control::Op::Control) weights.emplace(o);
        L2FlushBuffer flush(o.timing.flush_bytes);
        measurement::Csv csv(o.timing.csv_out, "weights");
        for (int tokens : o.tokens) {
            gdn_control::Fixture f(o, weights ? &weights->native : nullptr, tokens);
            for (const auto& execution :
                 measurement::selections(o.timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                    const auto result = measurement::measure(
                        [&](CudaExecutionView cuda) { f.launch(cuda); }, [](auto) {}, flush,
                        device.execution_view(), o.timing, execution, cache);
                    const auto workspace_peak = f.arena ? f.arena->peak_used() : 0;
                    if ((f.arena && f.arena->used() != 0) || workspace_peak != f.capacity)
                        throw std::runtime_error("workspace capacity/execution mismatch");
                    measurement::Row row;
                    row.op              = gdn_control::name(o.op);
                    row.format          = "bf16";
                    row.policy          = o.op == gdn_control::Op::Control ? "none" : "a16";
                    row.n               = o.op == gdn_control::Op::Control ? o.heads : 2 * o.heads;
                    row.k               = o.op == gdn_control::Op::Control ? 0 : o.input_rows;
                    row.value_heads     = o.heads;
                    row.tokens          = tokens;
                    row.batch           = o.batch;
                    row.workspace_bytes = f.capacity;
                    row.workspace_peak  = workspace_peak;
                    csv.write(row, result, execution, cache,
                              o.op == gdn_control::Op::Control            ? "none"
                              : o.storage == gdn_control::Storage::Parent ? "parent"
                                                                          : "pair");
                }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
