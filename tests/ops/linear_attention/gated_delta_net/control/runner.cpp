#include "support.h"
#include "core/decode_graph.h"

namespace ninfer::test::gdn_control {

int run(Profile p, Case c, bool normalize_input, CudaExecutionView execution) {
    Fixture f(p, c, normalize_input);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const bool captured = c.execution == Execution::Graph;
    const auto launch   = [&] { f.invoke(f.arena, execution); };
    if (captured) {
        launch();
        cuda_synchronize(execution.stream());
        definition.capture(execution.stream(), launch);
        graph.instantiate(definition);
    }
    int failed = 0;
    for (int phase = 0; phase < (captured ? 3 : 1); ++phase) {
        if (phase) f.advance();
        f.repaint(execution.stream());
        if (captured)
            graph.launch(execution.stream());
        else
            launch();
        cuda_synchronize(execution.stream());
        const auto name = label(p, c, normalize_input) + " phase=" + std::to_string(phase);
        failed += verify_projection(name, f, projection_reference(f));
        failed += f.check(name);
        const auto peak = f.capacity == 0 ? f.prefix : f.capacity + (f.prefix ? 256 : 0);
        if (f.arena.used() != f.prefix || f.arena.peak_used() != peak) {
            std::cerr << name << ": arena restoration/capacity mismatch\n";
            ++failed;
        }
    }
    return failed;
}

} // namespace ninfer::test::gdn_control
