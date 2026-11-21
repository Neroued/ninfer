#include "models/glm5next/program/graphs.h"

#include "core/startup.h"
#include "models/glm5next/program/graph_inputs.h"
#include "models/glm5next/program/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
Graphs::Graphs(RankProgramImpl& owner)
    : owner_(owner), gated_mtp_(owner.parameters.mtp &&
                                (!owner.parallel || owner.parallel->supports_graph_gating())) {}

void Graphs::bind_sites(Round& round, Body body, const execution::BatchView& main, int depth,
                        int head_columns) {
    const int g = main.rows, c = main.attention.rows;
    auto& p        = owner_;
    const auto& io = p.frame;
    const auto add = [&](execution::MlaReadout call, int step) {
        Site site{.call = call, .draft_step = step};
        const auto visible = call.visible;
        const auto sms     = p.cuda.multiprocessor_count();
        if (call.index) {
            site.index_profile = call.index_profile(visible, sms);
            site.index.capture(p.cuda.stream(), [&] { call.indexer(visible, p.cuda); });
        }
        site.attention_profile = call.attention_profile(visible, sms);
        site.attention.capture(p.cuda.stream(), [&] { call.attention(visible, p.cuda); });
        round.sites.push_back(std::move(site));
    };
    if (body == Body::Proposal || body == Body::MtpRound)
        for (int step = 0; step < depth; ++step) {
            auto call  = p.batch(io.mtp->draft_steps[step], 1, g, c, p.shape.max_context);
            auto run   = p.mtp_run(step, call);
            auto local = run.local_mla(p.mtp_selection(step, c));
            add(local.readout(), step);
        }
    if (body == Body::BlockRound) {
        auto run = p.dflash_run(g, p.shape.max_context, head_columns);
        for (std::uint32_t i = 0; i < p.parameters.draft->layers.size(); ++i) {
            auto layer = run.attention(i);
            DraftSite site{.call = layer.readout()};
            site.profile = site.call.profile(p.shape.max_context);
            site.attention.capture(p.cuda.stream(),
                                   [&] { site.call.execute(p.shape.max_context, p.cuda); });
            round.draft_sites.push_back(std::move(site));
        }
    }
    if (body == Body::Proposal) return;
    execution::StateUpdate update = execution::StepUpdate{};
    if (main.tokens > 1)
        update.emplace<execution::RecordWindow>(p.resources.replay(main.tokens).records);
    auto run = p.text_run(main, update);
    for (std::uint32_t i = 0; i < p.parameters.text.layers.size(); ++i) {
        if (!std::holds_alternative<execution::MlaParameters>(p.parameters.text.layers[i].mixer))
            continue;
        auto local = run.local_mla(i);
        add(local.readout(), -1);
    }
}

void Graphs::capture(Round& round, Body body, const execution::BatchView& main,
                     std::span<const std::uint32_t> bounds, int head_columns) {
    auto& p = owner_;
    p.resources.workspace.reset_peak();
    bind_sites(round, body, main, bounds.size(), head_columns);
    std::size_t cursor = 0;
    const auto emit    = [&](const execution::MlaReadout& call) {
        auto& site = round.sites.at(cursor++);
        if (site.call.query.data != call.query.data || site.call.scratch.data != call.scratch.data)
            throw std::logic_error("GLM Graph readout binding differs from execution layout");
        if (call.index) site.index_node = site.index.append_to(p.cuda.stream());
        site.attention_node = site.attention.append_to(p.cuda.stream());
    };
    std::size_t draft_cursor = 0;
    const auto draft_emit    = [&](const execution::DraftReadout& call) {
        auto& site = round.draft_sites.at(draft_cursor++);
        if (site.call.query.data != call.query.data || site.call.scratch.data != call.scratch.data)
            throw std::logic_error("DFlash Graph readout binding differs from execution layout");
        site.node = site.attention.append_to(p.cuda.stream());
    };
    round.graph.prepare(
        p.cuda,
        [&] {
            if (body != Body::Target)
                p.execute_proposal(main, bounds, head_columns, true, emit, &round.sections,
                                   draft_emit, body == Body::MtpRound ? &round.gates : nullptr);
            if (body != Body::Proposal) p.execute_target(main, true, false, emit, &round.sections);
        },
        p.parallel ? &*p.parallel : nullptr);
    if (body == Body::MtpRound) round.gates.finish(round.graph.communication);
    if (cursor != round.sites.size() || draft_cursor != round.draft_sites.size())
        throw std::logic_error("GLM Graph readouts were not consumed");
    round.workspace_bytes = p.resources.workspace.peak_used();
}

void Graphs::prepare_route(GraphInputs& inputs, GraphPreparation& preparation,
                           const speculative::RoundRoute& route) {
    auto& p     = owner_;
    auto* comm  = p.parallel ? &*p.parallel : nullptr;
    const int b = dimension(route.shape.rows), l = dimension(route.shape.attention_rows);
    const int w = route.width, d = route.mtp_steps, g = route.head_columns;
    const bool fused = d && gated_mtp_;
    if (fused && d != dimension(p.shape.draft_tokens)) return;
    const int depth = route.verification_limit(p.shape.speculative);
    inputs.set(b, l, w, depth);
    const auto main = p.decode_batch(w, b, l, p.shape.max_context);
    const std::vector<std::uint32_t> bounds(g ? 1 : d, p.shape.max_context);
    if (d && !fused) {
        if (preparation.capturing()) {
            auto& proposal = proposals_[{b, l, d}];
            capture(proposal, Body::Proposal, main, bounds, 0);
            launch(proposal, main, bounds, d, false, false);
        } else {
            p.execute_proposal(main, bounds, 0, false);
        }
    }
    // Target, sampling and retirement have separate sharing domains. Warm up and
    // capture each once, including when multiple MTP depths feed the same verifier.
    auto [target, new_target] = targets_.try_emplace({b, l, w, g});
    if (new_target) {
        if (preparation.capturing()) {
            capture(target->second,
                    fused ? Body::MtpRound
                    : g   ? Body::BlockRound
                          : Body::Target,
                    main, bounds, g);
            launch(target->second, main, bounds, d, true, false);
        } else {
            if (g || fused) p.execute_proposal(main, bounds, g, false);
            p.execute_target(main, true);
        }
    }
    auto [decision, new_decision] = decisions_.try_emplace({b, w, g});
    if (new_decision) {
        const auto dispatch = [&] {
            if (w > 1)
                p.accept_speculative(b, w, g);
            else
                p.sample(b, SampleKind::NextToken);
            p.copy_result(b, w > 1);
        };
        if (preparation.capturing()) {
            auto& graph = decision->second;
            p.resources.workspace.reset_peak();
            graph.graph.prepare(p.cuda, dispatch, comm);
            graph.workspace_bytes = p.resources.workspace.peak_used();
            decide(b, w, g);
        } else
            dispatch();
    }
    if (p.shape.draft_tokens) {
        auto [it, inserted] = commits_.try_emplace({b, l, w});
        if (inserted) {
            if (preparation.capturing()) {
                auto& graph = it->second;
                p.resources.workspace.reset_peak();
                graph.graph.prepare(
                    p.cuda,
                    [&] {
                        p.execute_speculative_commit(main, true, true, &graph.repair,
                                                     &graph.counts);
                    },
                    comm);
                graph.workspace_bytes = p.resources.workspace.peak_used();
                commit(b, l, w, true, true);
            } else
                p.execute_speculative_commit(main, true, true);
        }
    }
    preparation.checkpoint();
}

void Graphs::prepare(GraphInputs& inputs, GraphPreparation& preparation) {
    auto& p = owner_;
    for (const auto shape : p.resources.plan.batches->graph_shapes())
        for (const auto& route : p.resources.plan.decode_routes->at(shape))
            prepare_route(inputs, preparation, route);
}

void Graphs::replay(const execution::BatchView& main, std::span<const std::uint32_t> bounds,
                    int depth, int head_columns, bool head, bool read_drafts) {
    if (depth && !gated_mtp_) {
        launch(proposals_.at({main.rows, main.attention.rows, depth}), main, bounds, depth, false,
               read_drafts);
        read_drafts = false;
    }
    launch(targets_.at({main.rows, main.attention.rows, main.tokens, head_columns}), main, bounds,
           depth, head, read_drafts);
}

void Graphs::launch(Round& round, const execution::BatchView& main,
                    std::span<const std::uint32_t> draft, int depth, bool head, bool read_drafts) {
    auto& p       = owner_;
    const int sms = p.cuda.multiprocessor_count();
    for (auto& site : round.sites) {
        auto bound = main.attention.max_visible_tokens;
        if (site.draft_step >= 0) bound = site.draft_step < depth ? draft[site.draft_step] : 0U;
        if (site.call.index) {
            const auto profile = site.call.index_profile(bound, sms);
            if (profile != site.index_profile) {
                site.index.capture(p.cuda.stream(), [&] { site.call.indexer(bound, p.cuda); });
                round.graph.executable.update_child(site.index_node, site.index);
                site.index_profile = profile;
            }
        }
        const auto profile = site.call.attention_profile(bound, sms);
        if (profile != site.attention_profile) {
            site.attention.capture(p.cuda.stream(), [&] { site.call.attention(bound, p.cuda); });
            round.graph.executable.update_child(site.attention_node, site.attention);
            site.attention_profile = profile;
        }
    }
    for (auto& site : round.draft_sites) {
        const auto bound   = draft.front();
        const auto profile = site.call.profile(bound);
        if (profile != site.profile) {
            site.attention.capture(p.cuda.stream(), [&] { site.call.execute(bound, p.cuda); });
            round.graph.executable.update_child(site.node, site.attention);
            site.profile = profile;
        }
    }
    round.sections.head.set_enabled(round.graph.executable, head);
    round.sections.drafts.set_enabled(round.graph.executable, read_drafts);
    const auto* work =
        round.gates.empty() ? nullptr : &round.gates.set_depth(round.graph.executable, depth);
    round.graph.launch(p.cuda, p.parallel ? &*p.parallel : nullptr, work);
    workspace_peak_ = std::max(workspace_peak_, round.workspace_bytes);
}

void Graphs::decide(int g, int width, int head_columns) {
    auto& decision = decisions_.at({g, width, head_columns});
    decision.graph.launch(owner_.cuda, owner_.parallel ? &*owner_.parallel : nullptr);
    workspace_peak_ = std::max(workspace_peak_, decision.workspace_bytes);
}

void Graphs::commit(int g, int c, int width, bool repair, bool count_tokens) {
    auto& commit = commits_.at({g, c, width});
    if (owner_.parameters.mtp) commit.repair.set_enabled(commit.graph.executable, repair);
    commit.counts.set_enabled(commit.graph.executable, count_tokens);
    commit.graph.launch(owner_.cuda, owner_.parallel ? &*owner_.parallel : nullptr);
    workspace_peak_ = std::max(workspace_peak_, commit.workspace_bytes);
}

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
    graphs = std::make_unique<Graphs>(*this);
    graphs->prepare(*graph_inputs, observation);
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
