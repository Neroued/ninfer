#include "models/qwen4_exp/program/device_graphs.h"
#include "models/qwen4_exp/program/internal.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

DeviceGraphs::DeviceGraphs(RankProgramImpl& owner)
    : owner_(owner), fused_(!owner.parallel || owner.parallel->supports_graph_gating()),
      bounds_(decode_context_bounds(owner.shape.max_context)) {}

execution::TextRun DeviceGraphs::main_run(std::uint32_t batch, bool verify, std::uint32_t visible) {
    auto& p            = owner_;
    auto& r            = p.resources;
    const auto tokens  = verify ? p.shape.draft_tokens + 1 : 1;
    const auto records = r.main_records(tokens, batch);
    execution::TextExecution execution{r.workspace,
                                       *r.states,
                                       *r.main,
                                       nullptr,
                                       records.keys,
                                       records.coordinates,
                                       records.ple,
                                       p.parallel ? &*p.parallel : nullptr,
                                       p.parallel ? &p.ple_ready : nullptr};
    execution::StateUpdate update = execution::StepUpdate{};
    if (verify) update.emplace<execution::VerifyRecords>(r.replay(tokens).records);
    return execution::TextRun(p.parameters, execution, r.main_batch(tokens, batch, visible), update,
                              p.cuda);
}

void DeviceGraphs::proposal(Batch& profiles, DraftSites* sites, std::uint32_t batch,
                            std::uint32_t depth, MtpGraphSteps* gates) {
    auto& p            = owner_;
    const auto visible = sites ? bounds_.front() : bounds_.back();
    p.install_decode(batch);
    p.gpu_timer.begin_draft(sites != nullptr);
    for (std::uint32_t step = 0; step < depth; ++step) {
        if (!sites) {
            p.draft_forward(step, batch, visible);
            continue;
        }
        const auto forward = [&](MtpGraphSteps::Step* gate) {
            p.draft_forward(step, batch, visible, [&](const execution::QsaReadout& call) {
                const auto append = [&] {
                    sites->nodes.push_back(profiles.drafts[step].append(call, p.cuda.stream()));
                };
                if (gate)
                    gate->readout(p.cuda.stream(), append);
                else
                    append();
            });
        };
        if (gates)
            gates->capture_step(p.cuda.stream(), p.parallel ? &*p.parallel : nullptr, forward);
        else
            forward(nullptr);
    }
    p.gpu_timer.end_draft(sites != nullptr);
}

void DeviceGraphs::target(Batch& profiles, Round* capture, std::uint32_t batch, bool verify) {
    auto& p        = owner_;
    auto& r        = p.resources;
    const auto& io = r.buffers();
    const auto& c  = p.parameters.profile->config.text;
    const int b = batch, t = verify ? p.shape.draft_tokens + 1 : 1;
    const auto visible = capture ? bounds_.front() : bounds_.back();
    if (!verify) p.install_decode(batch);
    const auto call = r.main_batch(t, b, visible);
    if (verify) {
        auto ids = call.ids;
        ops::prepare_verify_ids(r.tensor(io.anchors).slice(0, 0, b), p.draft_proposals(b),
                                r.tensor(io.mtp->extents).slice(0, 0, b), ids, p.cuda);
    }
    p.gpu_timer.begin_target(capture != nullptr);
    p.prepare_device_ple(call, verify, capture && verify ? &capture->readback : nullptr);
    {
        auto run = main_run(batch, verify, visible);
        run.stem();
        std::size_t mixer = 0;
        for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer) {
            if (layer == c.ple.layer) {
                p.ple_ready.wait(p.cuda.stream());
                run.consume_ple(p.device_embedding(t, b));
            }
            if (capture)
                run.mixer(layer, [&](const execution::QsaReadout& call) {
                    capture->main.push_back(
                        profiles.main[verify][mixer++].append(call, p.cuda.stream()));
                });
            else
                run.mixer(layer);
            run.resident_moe(layer);
            run.combine(layer);
        }
        execution::TextOutputs output{r.packed(io.pending_sample, {dimension(c.hidden), t, b}), {}};
        if (io.pending_conditioning)
            output.conditioning =
                r.packed(*io.pending_conditioning, {dimension(c.residual_width()), t, b});
        run.finish(output);
        const auto project = [&] {
            auto sample = output.sample.view({dimension(c.hidden), t * b});
            auto logits =
                r.packed(*io.logits, {dimension(p.parameters.profile->local.vocabulary), t * b});
            execution::project_head(sample, p.parameters.text.head, logits, r.workspace, p.cuda);
        };
        if (capture)
            capture->head.capture(p.cuda.stream(), project);
        else
            project();
    }
    p.gpu_timer.end_target(capture != nullptr);
}

void DeviceGraphs::prepare(GraphInputs& inputs, GraphPreparation& preparation) {
    auto& p           = owner_;
    auto& r           = p.resources;
    const auto& io    = r.buffers();
    const auto drafts = p.parameters.mtp ? p.shape.draft_tokens : 0;
    for (auto b : r.plan.batches->graph_rows()) {
        r.select_batch(b);
        auto& batch = batches_[b];
        batch.drafts.resize(drafts);
        for (std::uint32_t verify = 0; verify < (drafts ? 2U : 1U); ++verify) {
            const auto t = verify ? p.shape.draft_tokens + 1 : 1;
            inputs.batch(r.main_batch(t, b, bounds_.back()), bounds_.back());
            if (preparation.capturing()) {
                auto run = main_run(b, verify, bounds_.back());
                for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer)
                    if (std::holds_alternative<execution::QsaParameters>(
                            p.parameters.text.layers[layer].mixer))
                        batch.main[verify].emplace_back().prepare(run.local_qsa(layer), bounds_,
                                                                  p.cuda);
            } else {
                for (const auto visible : bounds_) {
                    inputs.batch(r.main_batch(t, b, visible), visible);
                    auto run = main_run(b, verify, visible);
                    for (std::uint32_t layer = 0; layer < p.parameters.text.layers.size(); ++layer)
                        if (std::holds_alternative<execution::QsaParameters>(
                                p.parameters.text.layers[layer].mixer))
                            run.local_qsa(layer).execute();
                }
            }
        }
        for (std::uint32_t step = 0; step < drafts; ++step) {
            if (preparation.capturing()) {
                auto run = p.draft_run(step, b, bounds_.back());
                batch.drafts[step].prepare(run.local_qsa(), bounds_, p.cuda);
            } else {
                for (const auto visible : bounds_) {
                    inputs.batch(r.mtp_batch(io.mtp->draft_steps[step], 1, b, visible), visible);
                    auto run = p.draft_run(step, b, visible);
                    run.local_qsa().execute();
                }
            }
        }
        if (!fused_) batch.proposals.resize(drafts);
        for (std::uint32_t depth = 1; depth <= batch.proposals.size(); ++depth) {
            inputs.stage_proposal(b, depth, bounds_.back());
            auto& item = batch.proposals[depth - 1];
            if (preparation.capturing()) {
                item.sites.installed.resize(depth);
                item.graph.prepare(
                    p.cuda, [&] { proposal(batch, &item.sites, b, depth); },
                    p.parallel ? &*p.parallel : nullptr);
            } else {
                proposal(batch, nullptr, b, depth);
            }
            preparation.checkpoint();
        }
        for (int verify = 0; verify < (drafts ? 2 : 1); ++verify) {
            const auto t = verify ? drafts + 1 : 1;
            if (verify)
                inputs.stage_proposal(b, drafts, bounds_.back());
            else
                inputs.batch(r.main_batch(t, b, bounds_.back()), bounds_.back());
            const auto decision = [&] {
                if (verify)
                    p.accept_forward(b, t);
                else
                    p.sample_forward(b, false);
                p.copy_sample_result(b, verify);
            };
            inputs.stage();
            if (!preparation.capturing()) {
                if (verify) proposal(batch, nullptr, b, drafts);
                target(batch, nullptr, b, verify);
                decision();
            } else {
                auto& item = batch.rounds[verify];
                item.graph.prepare(
                    p.cuda,
                    [&] {
                        if (fused_ && verify) {
                            item.draft.installed.resize(drafts);
                            proposal(batch, &item.draft, b, drafts, &item.gates);
                        }
                        target(batch, &item, b, verify);
                    },
                    p.parallel ? &*p.parallel : nullptr);
                if (fused_ && verify) item.gates.finish(item.graph.communication);
                batch.decisions[verify].prepare(p.cuda, decision,
                                                p.parallel ? &*p.parallel : nullptr);
                replay(b, verify ? drafts : 0,
                       std::vector<std::uint32_t>(verify ? drafts : 0, bounds_.back()),
                       bounds_.back(), true, false);
                decide(b, verify);
            }
            preparation.checkpoint();
        }
    }
    workspace_peak_ = r.workspace.peak_used();
}

std::size_t DeviceGraphs::bound(std::uint32_t visible) const {
    const auto found = std::lower_bound(bounds_.begin(), bounds_.end(), visible);
    if (found == bounds_.end()) throw std::logic_error("decode graph context exceeds its capacity");
    return found - bounds_.begin();
}

void DeviceGraphs::replay(std::uint32_t batch, std::uint32_t depth,
                          std::span<const std::uint32_t> draft_visible, std::uint32_t visible,
                          bool head, bool read_drafts) {
    auto& profiles                     = batches_.at(batch);
    auto* parallel                     = owner_.parallel ? &*owner_.parallel : nullptr;
    const bool verify                  = depth != 0;
    auto& round                        = profiles.rounds[verify];
    const CommunicationGraphWork* work = nullptr;
    if (verify) {
        auto& graph = fused_ ? round.graph : profiles.proposals.at(depth - 1).graph;
        auto& sites = fused_ ? round.draft : profiles.proposals.at(depth - 1).sites;
        for (std::size_t s = 0; s < sites.nodes.size(); ++s) {
            // The readout child still launches for a gated step. Use its smallest grid.
            const auto selected = s < depth ? bound(draft_visible[s]) : std::size_t{0};
            if (selected != sites.installed[s]) {
                graph.executable.update_child(sites.nodes[s], profiles.drafts[s].at(selected));
                sites.installed[s] = selected;
            }
        }
        if (fused_)
            work = &round.gates.set_depth(graph.executable, depth);
        else
            graph.launch(owner_.cuda, parallel);
        round.readback.set_enabled(round.graph.executable, read_drafts);
    }
    const auto main = bound(visible);
    if (main != round.installed) {
        for (std::size_t i = 0; i < round.main.size(); ++i)
            round.graph.executable.update_child(round.main[i], profiles.main[verify][i].at(main));
        round.installed = main;
    }
    round.head.set_enabled(round.graph.executable, head);
    round.graph.launch(owner_.cuda, parallel, work);
}

void DeviceGraphs::decide(std::uint32_t batch, bool verify) {
    batches_.at(batch).decisions[verify].launch(owner_.cuda,
                                                owner_.parallel ? &*owner_.parallel : nullptr);
}

} // namespace ninfer::models::qwen4_exp::detail
