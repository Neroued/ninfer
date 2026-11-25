#include "models/glm5next/program/internal.h"

#include "models/glm5next/program/graph_inputs.h"

namespace ninfer::models::glm5next::detail {
namespace {

// Standalone patches use the bound operands. Inline recording uses the operands from the
// same forward path that executes eagerly, with the execution view supplied by the suite.
template <class Readout>
auto capture_readout(Readout call, auto execute) {
    return [call = std::move(call), execute](ops::ContextBucket bucket, CudaExecutionView cuda) {
        (call.*execute)(bucket, cuda);
    };
}

void bind_mla(std::vector<ContextBucketBinding>& bindings, DeviceContext& device,
              ops::ContextBucketRange buckets, const execution::MlaReadout& call) {
    const auto sms = device.execution_view().multiprocessor_count();
    if (call.index)
        bindings.push_back({buckets, [call, sms](auto b) { return call.index_profile(b, sms); },
                            capture_readout(call, &execution::MlaReadout::indexer)});
    bindings.push_back({buckets, [call, sms](auto b) { return call.attention_profile(b, sms); },
                        capture_readout(call, &execution::MlaReadout::attention)});
}

void record_mla(const ReadoutRecorder& record, const execution::MlaReadout& expected,
                const execution::MlaReadout& actual) {
    if (expected.query.data != actual.query.data || expected.scratch.data != actual.scratch.data ||
        expected.index != actual.index)
        throw std::logic_error("GLM Graph readout binding differs from execution layout");
    if (actual.index) record(capture_readout(actual, &execution::MlaReadout::indexer));
    record(capture_readout(actual, &execution::MlaReadout::attention));
}

ReadoutCapture bind_mtp_step(RankProgramImpl& p, ops::ContextBucketRange buckets,
                             std::uint32_t step, const execution::BatchView& batch) {
    auto run   = p.mtp_run(step, batch);
    auto local = run.local_mla(p.mtp_selection(step, batch.attention.rows));
    auto call  = local.readout(buckets);
    ReadoutCapture capture;
    bind_mla(capture.readouts, p.device, buckets, call);
    capture.body = [&p, step, batch, call](const ReadoutRecorder& record) {
        p.mtp_forward(step, batch, [&](const execution::MlaReadout& actual) {
            record_mla(record, call, actual);
        });
    };
    return capture;
}

ReadoutCapture bind_dflash(RankProgramImpl& p, ops::ContextBucketRange buckets, int rows,
                           int head_columns) {
    ReadoutCapture capture;
    std::vector<execution::DraftReadout> expected;
    auto run = p.dflash_run(rows, buckets.first, head_columns);
    for (std::uint32_t layer = 0; layer < p.parameters.draft->layers.size(); ++layer) {
        const auto call = run.attention(layer).readout();
        expected.push_back(call);
        capture.readouts.push_back({buckets, [call](auto b) { return call.profile(b); },
                                    capture_readout(call, &execution::DraftReadout::execute)});
    }
    capture.body = [&p, rows, head_columns, bucket = buckets.first,
                    expected = std::move(expected)](const ReadoutRecorder& record) {
        std::size_t next = 0;
        p.dflash_forward(rows, bucket, head_columns, [&](const execution::DraftReadout& actual) {
            const auto& call = expected.at(next++);
            if (call.query.data != actual.query.data || call.scratch.data != actual.scratch.data)
                throw std::logic_error(
                    "DFlash Graph readout binding differs from execution layout");
            record(capture_readout(actual, &execution::DraftReadout::execute));
        });
    };
    return capture;
}

ProposalCapture bind_proposal(RankProgramImpl& p, ops::ContextBucketRange buckets,
                              const speculative::RoundRoute& route,
                              const execution::BatchView& main) {
    const int head_columns = static_cast<int>(route.head_columns);
    const auto drafts      = p.draft_proposals(
        main.rows, p.parameters.mtp ? dimension(p.shape.draft_tokens) : head_columns);
    ProposalCapture capture;
    capture.prefix          = [&p, main] { p.begin_proposal(main, true); };
    capture.before_readback = [&p] { p.timer.end_draft(true); };
    capture.readback        = [&p, drafts] { p.readback_drafts(drafts); };
    capture.drafts_ready    = [&p] { p.drafts_done.record_external(p.cuda.stream()); };
    if (p.parameters.mtp) {
        for (std::uint32_t step = 0; step < route.mtp_steps; ++step) {
            const auto batch = p.batch(p.frame.mtp->draft_steps[step], 1, main.rows,
                                       main.attention.rows, p.shape.max_context);
            capture.steps.push_back({.forward = bind_mtp_step(p, buckets, step, batch)});
        }
    } else {
        capture.steps.push_back({.forward = bind_dflash(p, buckets, main.rows, head_columns)});
    }
    return capture;
}

ResidentTargetCapture bind_target(RankProgramImpl& p, ops::ContextBucketRange buckets,
                                  const execution::BatchView& main, int head_columns) {
    const auto hidden = p.target_hidden(main);
    ResidentTargetCapture capture;
    std::vector<execution::MlaReadout> expected;
    auto run = p.text_run(main, p.target_update(main.tokens));
    for (std::uint32_t i = 0; i < p.parameters.text.layers.size(); ++i) {
        if (!std::holds_alternative<execution::MlaParameters>(p.parameters.text.layers[i].mixer))
            continue;
        auto local      = run.local_mla(i);
        const auto call = local.readout(buckets);
        expected.push_back(call);
        bind_mla(capture.readouts, p.device, buckets, call);
    }
    capture.body = [&p, main, hidden, head_columns, expected = std::move(expected)](
                       const ReadoutRecorder& record, const GraphBody&, const GraphBody& head,
                       const StreamSectionRecorder&) {
        p.begin_target(main, true);
        if (main.tokens > 1) {
            const auto steps  = p.parameters.mtp ? dimension(p.shape.draft_tokens) : head_columns;
            const auto drafts = p.draft_proposals(main.rows, steps);
            p.prepare_verify_inputs(main, drafts);
        }
        std::size_t next = 0;
        p.target_forward(main, hidden, false, [&](const execution::MlaReadout& actual) {
            record_mla(record, expected.at(next++), actual);
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
    // Retain inputs on failure until the executor aborts communication and drains the Program.
    graph_inputs = std::make_unique<GraphInputs>(*this);
    graph_inputs->prepare();
    graphs =
        std::make_unique<models::DecodeGraphSuite>(device, resources.workspace, shape.speculative,
                                                   false, parallel ? &*parallel : nullptr, nullptr);

    const auto buckets = ops::ContextBucketRange::through(shape.max_context);
    execution::BatchView main;
    std::vector<ops::ContextBucket> draft_buckets;
    DecodeGraphBindings bindings;
    bindings.stage = [&](const speculative::RoundRoute& route) {
        const int rows = dimension(route.shape.rows), local = dimension(route.shape.attention_rows),
                  width = dimension(route.width);
        const int depth = static_cast<int>(route.verification_limit(shape.speculative));
        graph_inputs->set(rows, local, width, depth);
        main = decode_batch(width, rows, local, shape.max_context);
        const auto steps =
            route.mtp_steps ? route.mtp_steps : std::uint32_t(route.head_columns != 0);
        draft_buckets.assign(steps, buckets.last);
        return TargetReplay{.bucket = buckets.last, .proposal = {.buckets = draft_buckets}};
    };
    bindings.warmup_proposal = [&](const speculative::RoundRoute& route) {
        const std::vector<std::uint32_t> bounds(draft_buckets.size(), shape.max_context);
        execute_proposal(main, bounds, static_cast<int>(route.head_columns), false);
    };
    bindings.warmup_target = [&](const speculative::RoundRoute&) { execute_target(main, true); };
    bindings.proposal      = [&](const speculative::RoundRoute& route) {
        return bind_proposal(*this, buckets, route, main);
    };
    bindings.target = [&](const speculative::RoundRoute& route) -> TargetCapture {
        return bind_target(*this, buckets, main, static_cast<int>(route.head_columns));
    };
    bindings.decision = [&](const speculative::RoundRoute& route) {
        const bool verify = route.width > 1;
        if (verify)
            accept_speculative(main.rows, main.tokens, static_cast<int>(route.head_columns));
        else
            sample(main.rows, SampleKind::NextToken);
        copy_result(main.rows, verify);
    };
    if (shape.draft_tokens) {
        bindings.commit = [&](const speculative::RoundRoute&, const SectionRecorder& repair,
                              const SectionRecorder& counts) {
            execute_speculative_commit(main, true, true, repair, counts);
        };
    }
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

} // namespace ninfer::models::glm5next::detail
