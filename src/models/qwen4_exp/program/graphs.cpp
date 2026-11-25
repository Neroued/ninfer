#include "models/qwen4_exp/program/internal.h"

#include "core/nvtx.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>
#include <thread>

namespace ninfer::models::qwen4_exp::detail {

namespace {

auto capture_readout(execution::QsaReadout call,
                     void (execution::QsaReadout::*execute)(ops::ContextBucket, CudaExecutionView)
                         const) {
    return [call = std::move(call), execute](ops::ContextBucket bucket, CudaExecutionView cuda) {
        (call.*execute)(bucket, cuda);
    };
}

void bind_qsa(std::vector<ContextBucketBinding>& bindings, DeviceContext& device,
              ops::ContextBucketRange buckets, const execution::QsaReadout& call) {
    const int sms = device.multiprocessor_count();
    bindings.push_back({buckets, [call](auto b) { return call.index_profile(b); },
                        capture_readout(call, &execution::QsaReadout::indexer)});
    bindings.push_back({buckets, [call, sms](auto b) { return call.attention_profile(b, sms); },
                        capture_readout(call, &execution::QsaReadout::attention)});
}

void record_qsa(const ReadoutRecorder& record, const execution::QsaReadout& expected,
                const execution::QsaReadout& actual) {
    if (expected.query.data != actual.query.data || expected.output.data != actual.output.data ||
        expected.scratch.data != actual.scratch.data)
        throw std::logic_error("Qwen4Exp Graph readout binding differs from execution layout");
    record(capture_readout(actual, &execution::QsaReadout::indexer));
    record(capture_readout(actual, &execution::QsaReadout::attention));
}

ReadoutCapture bind_mtp_step(RankProgramImpl& p, ops::ContextBucketRange buckets,
                             std::uint32_t step, int batch) {
    auto run        = p.draft_run(step, batch, p.shape.max_context);
    const auto call = run.local_qsa().readout(buckets);
    ReadoutCapture capture;
    bind_qsa(capture.readouts, p.device, buckets, call);
    capture.body = [&p, step, batch, call](const ReadoutRecorder& record) {
        p.draft_forward(step, batch, p.shape.max_context, [&](const execution::QsaReadout& actual) {
            record_qsa(record, call, actual);
        });
    };
    return capture;
}

void prepare_verify_ids(RankProgramImpl& p, int batch, int width) {
    auto& r        = p.resources;
    const auto& io = r.buffers();
    auto ids       = r.main_batch(width, batch, 1).ids;
    ops::prepare_verify_ids(r.tensor(io.anchors).slice(0, 0, batch), p.draft_proposals(batch),
                            r.tensor(io.mtp->extents).slice(0, 0, batch), ids, p.cuda);
}

ProposalCapture bind_proposal(RankProgramImpl& p, ops::ContextBucketRange buckets,
                              const speculative::RoundRoute& route) {
    const int batch = dimension(route.shape.rows);
    const int width = dimension(route.width);
    ProposalCapture capture;
    capture.prefix = [&p, batch] {
        p.install_decode(batch);
        if (!p.host_ple) p.gpu_timer.begin_draft(true);
    };
    if (p.host_ple)
        capture.handoff_stream = p.resources.ple_transfer().stream();
    else
        capture.before_readback = [&p] { p.gpu_timer.end_draft(true); };
    for (std::uint32_t step = 0; step < route.mtp_steps; ++step) {
        auto& binding   = capture.steps.emplace_back();
        binding.forward = bind_mtp_step(p, buckets, step, batch);
        if (p.host_ple)
            binding.handoff = [&p, ids = p.draft_output_ids(step, batch),
                               step](std::uint32_t slot, CudaExecutionView cuda) {
                p.host_ple->capture_append(slot, ids, step, cuda);
            };
    }
    capture.verify_ids = [&p, batch, width] { prepare_verify_ids(p, batch, width); };
    return capture;
}

execution::TextRun main_run(RankProgramImpl& p, execution::BatchView batch, CudaExecutionView cuda,
                            moe::Schedule* experts = nullptr) {
    auto& r            = p.resources;
    const auto records = r.main_records(batch.tokens, batch.batch);
    const execution::TextExecution execution{r.workspace,
                                             *r.states,
                                             *r.main,
                                             experts,
                                             records.keys,
                                             records.coordinates,
                                             records.ple,
                                             p.parallel ? &*p.parallel : nullptr,
                                             p.parallel && !p.host_ple ? &p.ple_ready : nullptr};
    execution::StateUpdate update = execution::StepUpdate{};
    if (batch.tokens > 1) update.emplace<execution::VerifyRecords>(r.replay(batch.tokens).records);
    return execution::TextRun(p.parameters, execution, batch, update, cuda);
}

execution::TextOutputs main_outputs(RankProgramImpl& p, execution::BatchView batch) {
    auto& r        = p.resources;
    const auto& io = r.buffers();
    const auto& c  = p.parameters.profile->config.text;
    execution::TextOutputs output{
        r.packed(io.pending_sample, {dimension(c.hidden), batch.tokens, batch.batch}), {}};
    if (io.pending_conditioning)
        output.conditioning = r.packed(*io.pending_conditioning,
                                       {dimension(c.residual_width()), batch.tokens, batch.batch});
    return output;
}

GraphBody bind_head(RankProgramImpl& p, execution::BatchView batch, const Tensor& sample) {
    auto& r           = p.resources;
    const int columns = batch.tokens * batch.batch;
    auto logits =
        r.packed(*r.buffers().logits, {dimension(p.parameters.profile->local.vocabulary), columns});
    return [&p, hidden = sample.view({sample.ne[0], columns}), logits]() mutable {
        execution::project_head(hidden, p.parameters.text.head, logits, p.resources.workspace,
                                p.cuda);
    };
}

void consume_host_ple(RankProgramImpl& p, execution::TextRun& run, execution::BatchView batch,
                      std::uint32_t slot, cudaStream_t stream) {
    const auto ple = p.resources.ple_upload->binding(slot, batch.tokens, batch.batch);
    ple.ready.wait_external(stream);
    run.consume_ple(ple.values);
    ple.consumed.record_external(stream);
}

void forward_resident(RankProgramImpl& p, execution::TextRun& run, execution::TextOutputs output,
                      const execution::QsaSubmit& submit, const GraphBody& ple,
                      const GraphBody& head) {
    run.stem();
    for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer) {
        if (layer == p.parameters.profile->config.text.ple.layer) ple();
        run.mixer(layer, submit);
        run.resident_moe(layer);
        run.combine(layer);
    }
    run.finish(output);
    head();
}

ResidentTargetCapture bind_resident(RankProgramImpl& p, ops::ContextBucketRange buckets,
                                    execution::BatchView batch) {
    ResidentTargetCapture capture;
    std::vector<execution::QsaReadout> expected;
    {
        // The fused proposal and target reuse workspace. Bind the target without holding its
        // arena scope while the suite records the proposal.
        auto run = main_run(p, batch, p.cuda);
        for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer) {
            if (!std::holds_alternative<execution::QsaParameters>(
                    p.parameters.text.layers[layer].mixer))
                continue;
            const auto call = run.local_qsa(layer).readout(buckets);
            expected.push_back(call);
            bind_qsa(capture.readouts, p.device, buckets, call);
        }
    }
    if (p.host_ple)
        capture.handoff = [&p, batch](std::uint32_t slot, CudaExecutionView cuda) {
            auto run = main_run(p, batch, cuda);
            consume_host_ple(p, run, batch, slot, cuda.stream());
        };
    const auto output = main_outputs(p, batch);
    capture.head      = bind_head(p, batch, output.sample);
    capture.body      = [&p, batch, output, expected = std::move(expected)](
                       const ReadoutRecorder& record, const GraphBody& handoff,
                       const GraphBody& head, const StreamSectionRecorder& readback) {
        const bool verify = batch.tokens > 1;
        if (!verify) p.install_decode(batch.batch);
        if (!p.host_ple) {
            p.gpu_timer.begin_target(true);
            p.prepare_device_ple(batch, verify, readback);
        }
        auto run         = main_run(p, batch, p.cuda);
        std::size_t next = 0;
        forward_resident(
            p, run, output,
            [&](const execution::QsaReadout& actual) {
                record_qsa(record, expected.at(next++), actual);
            },
            [&] {
                if (p.host_ple) {
                    handoff();
                } else {
                    p.ple_ready.wait(p.cuda.stream());
                    run.consume_ple(p.device_embedding(batch.tokens, batch.batch));
                }
            },
            head);
        if (!p.host_ple) p.gpu_timer.end_target(true);
    };
    return capture;
}

OffloadedTargetCapture bind_offloaded(RankProgramImpl& p, ops::ContextBucketRange buckets,
                                      execution::BatchView batch) {
    const auto geometry =
        moe::decode_geometry(*p.resources.plan.batches, batch.batch, batch.tokens);
    // The suite owns the schedule activity during replay. Keep the activation binding alive
    // only for the synchronous capture of the prefix, layers, hook, and suffix.
    auto run          = std::make_shared<execution::TextRun>(main_run(p, batch, p.cuda));
    const auto output = main_outputs(p, batch);
    OffloadedTargetCapture capture{.output          = run->block_output(),
                                   .physical_tokens = geometry.physical_tokens};
    capture.prefix = [&p, batch, run] {
        if (batch.tokens == 1) p.install_decode(batch.batch);
        run->stem();
    };
    for (std::uint32_t i = 0; i < p.parameters.text.layers.size(); ++i) {
        const auto& layer = p.parameters.text.layers[i];
        ReadoutCapture mixer;
        if (std::holds_alternative<execution::QsaParameters>(layer.mixer)) {
            const auto call = run->local_qsa(i).readout(buckets);
            bind_qsa(mixer.readouts, p.device, buckets, call);
            mixer.body = [run, i, call](const ReadoutRecorder& record) {
                run->mixer(i, [&](const execution::QsaReadout& actual) {
                    record_qsa(record, call, actual);
                });
            };
        } else {
            mixer.body = [run, i](const ReadoutRecorder&) { run->mixer(i); };
        }
        capture.layers.emplace_back(
            OffloadedLayerCapture{layer.moe.bank, geometry, std::move(mixer),
                                  [run](const moe::GpuCall& call) { run->gpu_moe(call); },
                                  [run, i] { run->combine(i); }});
    }
    capture.hook = InterLayerHookCapture{
        p.parameters.profile->config.text.ple.layer,
        [&p, run, batch](std::uint32_t slot, CudaExecutionView cuda) {
            consume_host_ple(p, *run, batch, slot, cuda.stream());
        }};
    capture.suffix = [run, output] { run->finish(output); };
    capture.head   = bind_head(p, batch, output.sample);
    return capture;
}

void warmup_readouts(RankProgramImpl& p, GraphInputs& inputs, ops::ContextBucketRange buckets,
                     const speculative::RoundRoute& route) {
    const int rows  = dimension(route.shape.rows);
    const int width = dimension(route.width);
    buckets.for_each([&](ops::ContextBucket bucket) {
        const auto visible = std::min(p.shape.max_context, ops::context_bucket_tokens(bucket));
        const auto batch   = p.resources.main_batch(width, rows, visible);
        inputs.batch(batch, visible);
        {
            auto run = main_run(p, batch, p.cuda);
            for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer)
                if (std::holds_alternative<execution::QsaParameters>(
                        p.parameters.text.layers[layer].mixer))
                    run.local_qsa(layer).execute();
        }
        const auto& io = p.resources.buffers();
        for (std::uint32_t step = 0; step < route.mtp_steps; ++step) {
            inputs.batch(p.resources.mtp_batch(io.mtp->draft_steps[step], 1, rows, visible),
                         visible);
            auto run = p.draft_run(step, rows, visible);
            run.local_qsa().execute();
        }
    });
}

void warmup_target(RankProgramImpl& p, execution::BatchView batch) {
    const bool verify = batch.tokens > 1;
    if (!verify) p.install_decode(batch.batch);
    if (!p.host_ple) {
        p.gpu_timer.begin_target();
        p.prepare_device_ple(batch, verify);
    }
    auto run          = main_run(p, batch, p.cuda, p.resources.experts.get());
    const auto output = main_outputs(p, batch);
    const auto head   = bind_head(p, batch, output.sample);
    const auto ple    = [&] {
        if (p.host_ple) {
            run.consume_ple(p.resources.ple_upload->device_slot(0, batch.tokens, batch.batch));
        } else {
            p.ple_ready.wait(p.cuda.stream());
            run.consume_ple(p.device_embedding(batch.tokens, batch.batch));
        }
    };
    if (p.resources.experts) {
        run.stem();
        for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer) {
            if (layer == p.parameters.profile->config.text.ple.layer) ple();
            while (!run.advance_layer(layer)) std::this_thread::yield();
        }
        run.finish(output);
        head();
    } else {
        forward_resident(p, run, output, {}, ple, head);
    }
    if (!p.host_ple) p.gpu_timer.end_target();
}

} // namespace

GraphPreparationSample RankProgramImpl::initialize_graphs(GraphPreparationMode mode) {
    if (!shape.cuda_graph) return {};
    if (graphs) throw std::logic_error("Qwen4Exp Graphs are already prepared");
    if (mode == GraphPreparationMode::Complete) {
        initialize_graphs(GraphPreparationMode::Warmup);
        return initialize_graphs(GraphPreparationMode::Capture);
    }
    nvtx::ScopedRange range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);
    GraphPreparation observation(device, mode);
    graph_inputs = std::make_unique<GraphInputs>(*this);
    graph_inputs->prepare();
    const auto speculative =
        parameters.mtp && shape.draft_tokens ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
    graphs = std::make_unique<models::DecodeGraphSuite>(
        device, resources.workspace, speculative, host_ple != nullptr,
        parallel ? &*parallel : nullptr, resources.experts.get());

    const auto buckets = ops::ContextBucketRange::through(shape.max_context);
    execution::BatchView main;
    std::vector<ops::ContextBucket> draft_buckets;
    DecodeGraphBindings bindings;
    bindings.stage = [&](const speculative::RoundRoute& route) {
        const auto rows = route.shape.rows, width = route.width;
        const bool verify = width > 1;
        resources.select_batch(rows);
        if (!observation.capturing() && (!verify || route.mtp_steps == shape.draft_tokens))
            warmup_readouts(*this, *graph_inputs, buckets, route);
        main              = resources.main_batch(width, rows, shape.max_context);
        const auto active = resources.plan.batches->minimum_rows(rows);
        if (verify) {
            graph_inputs->stage_proposal(rows, route.mtp_steps, shape.max_context);
            main.active_tokens = active * (std::min(route.mtp_steps, shape.max_context - 1) + 1);
        } else {
            if (speculative == SpeculativeBackend::Mtp)
                graph_inputs->batch(resources.mtp_batch(resources.buffers().mtp->draft_steps[0], 1,
                                                        rows, shape.max_context),
                                    shape.max_context);
            graph_inputs->batch(main, shape.max_context);
            graph_inputs->stage();
            main.active_tokens = active;
        }
        if (speculative == SpeculativeBackend::Mtp &&
            (!verify || route.mtp_steps == shape.draft_tokens))
            graph_inputs->stage_commit(rows, width);
        draft_buckets.assign(route.mtp_steps, buckets.last);
        const auto slot = host_ple ? std::optional<std::uint32_t>{0} : std::nullopt;
        return TargetReplay{.bucket        = buckets.last,
                            .slot          = slot,
                            .readback      = verify && !host_ple,
                            .active_tokens = dimension(main.active_tokens),
                            .proposal      = {.buckets = draft_buckets, .slot = slot}};
    };
    bindings.warmup_proposal = [&](const speculative::RoundRoute& route) {
        install_decode(main.batch);
        if (!host_ple) gpu_timer.begin_draft();
        for (std::uint32_t step = 0; step < route.mtp_steps; ++step)
            draft_forward(step, main.batch, shape.max_context);
        if (!host_ple) gpu_timer.end_draft();
        prepare_verify_ids(*this, main.batch, main.tokens);
    };
    bindings.warmup_target = [&](const speculative::RoundRoute&) { warmup_target(*this, main); };
    bindings.proposal      = [&](const speculative::RoundRoute& route) {
        return bind_proposal(*this, buckets, route);
    };
    bindings.target = [&](const speculative::RoundRoute&) -> TargetCapture {
        if (resources.experts) return bind_offloaded(*this, buckets, main);
        return bind_resident(*this, buckets, main);
    };
    bindings.decision = [&](const speculative::RoundRoute& route) {
        const bool verify = route.width > 1;
        if (verify)
            accept_forward(main.batch, main.tokens);
        else
            sample_forward(main.batch, false);
        copy_sample_result(main.batch, verify);
    };
    if (speculative == SpeculativeBackend::Mtp)
        bindings.commit = [&](const speculative::RoundRoute&, const SectionRecorder& repair,
                              const SectionRecorder& counts) {
            execute_mtp_commit(main.batch, main.tokens, true, true, repair, counts);
        };
    graphs->prepare(*resources.plan.batches, *resources.plan.decode_routes, observation, bindings);
    observation.checkpoint();
    graph_inputs.reset();
    const auto result = observation.sample();
    if (!graph_memory) graph_memory.emplace();
    if (observation.capturing()) {
        graph_memory->preparation_increment_bytes = result.increment_bytes;
        graph_memory->preparation_peak_bytes      = result.peak_bytes;
    } else {
        release_graphs();
        graph_memory->warmup_increment_bytes = result.increment_bytes;
    }
    return result;
}

void RankProgramImpl::release_graphs() {
    resources.synchronize();
    graphs.reset();
    graph_inputs.reset();
}

} // namespace ninfer::models::qwen4_exp::detail
