#include "models/inkling2/program/internal.h"

#include "models/inkling2/program/graph_inputs.h"

#include <stdexcept>

namespace ninfer::models::inkling2::detail {
namespace {

ResidentTargetCapture bind_target(RankProgramImpl& p, ops::ContextBucketRange buckets,
                                  const execution::BatchView& main) {
    const auto hidden = p.target_hidden(main);
    const auto sms    = p.device.execution_view().multiprocessor_count();
    ResidentTargetCapture capture;
    auto run      = p.text_run(main, execution::StepUpdate{});
    auto expected = run.plan_full_readouts(buckets);
    capture.readouts.reserve(expected.size());
    for (const auto& call : expected) {
        capture.readouts.push_back({
            buckets,
            [call, sms](ops::ContextBucket b) { return call.profile(b, sms); },
            [call](ops::ContextBucket b, CudaExecutionView cuda) { call.execute(b, cuda); },
        });
    }
    capture.body = [&p, main, hidden, expected = std::move(expected)](
                       const ReadoutRecorder& record, const GraphBody&, const GraphBody& head,
                       const StreamSectionRecorder&) {
        p.begin_target(main, /*capture=*/true);
        std::size_t next = 0;
        p.target_forward(main, hidden, [&](const execution::FullAttentionReadout& actual) {
            const auto& exp = expected.at(next++);
            if (exp.inputs.q.data != actual.inputs.q.data || exp.out.data != actual.out.data ||
                exp.scratch.data != actual.scratch.data)
                throw std::logic_error(
                    "Inkling2 Graph readout binding differs from execution layout");
            record([actual](ops::ContextBucket b, CudaExecutionView cuda) {
                actual.execute(b, cuda);
            });
        });
        head();
        p.timer.end_target(true);
    };
    capture.head = [&p, hidden, main] {
        p.project_head(hidden.view({hidden.ne[0], main.tokens * main.rows}));
    };
    return capture;
}

} // namespace

GraphPreparationSample RankProgramImpl::prepare_graphs(GraphPreparationMode mode) {
    if (!resources.plan.use_cuda_graph || graphs) return {};
    if (mode == GraphPreparationMode::Complete) {
        prepare_graphs(GraphPreparationMode::Warmup);
        return prepare_graphs(GraphPreparationMode::Capture);
    }
    GraphPreparation observation(device, mode);
    graph_inputs = std::make_unique<GraphInputs>(*this);
    graph_inputs->prepare();
    graphs = std::make_unique<models::DecodeGraphSuite>(device, resources.workspace,
                                                        shape.speculative, false, nullptr, nullptr);

    const auto buckets = ops::ContextBucketRange::through(shape.max_context);
    execution::BatchView main;
    DecodeGraphBindings bindings;
    bindings.stage = [&](const speculative::RoundRoute& route) {
        const int rows  = dimension(route.shape.rows);
        const int width = dimension(route.width);
        graph_inputs->set(rows, width);
        main = decode_batch(width, rows, shape.max_context);
        return TargetReplay{.bucket = buckets.last};
    };
    bindings.warmup_target = [&](const speculative::RoundRoute&) { execute_target(main, true); };
    bindings.target        = [&](const speculative::RoundRoute&) -> TargetCapture {
        return bind_target(*this, buckets, main);
    };
    bindings.decision = [&](const speculative::RoundRoute&) {
        sample(main.rows, SampleKind::NextToken);
        copy_result(main.rows);
    };

    graphs->prepare(*resources.plan.batches, *resources.plan.decode_routes, observation, bindings);
    observation.checkpoint();
    graph_inputs.reset();
    const auto result = observation.sample();
    if (observation.capturing()) {
        graph_memory.preparation_increment_bytes = result.increment_bytes;
        graph_memory.preparation_peak_bytes      = result.peak_bytes;
    } else {
        graphs.reset();
        graph_memory.warmup_increment_bytes = result.increment_bytes;
    }
    return result;
}

void RankProgramImpl::release_graphs() {
    device.synchronize();
    graphs.reset();
    graph_inputs.reset();
}

} // namespace ninfer::models::inkling2::detail
