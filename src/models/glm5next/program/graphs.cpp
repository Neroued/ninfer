#include "models/glm5next/program/graphs.h"

#include "core/startup.h"
#include "models/glm5next/program/graph_inputs.h"
#include "models/glm5next/program/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
void Graphs::bind_sites(Round& round, int g, int c, int depth) {
    auto& p        = owner_;
    const auto& io = p.resources.plan.frame;
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
    for (int step = 0; step < depth; ++step) {
        auto call  = p.batch(io.mtp->draft_steps[step], 1, g, c, p.shape.max_context);
        auto run   = p.draft_run(step, call);
        auto local = run.local_mla(p.draft_selection(step, c));
        add(local.readout(), step);
    }
    const int t                   = depth ? p.shape.draft_tokens + 1 : 1;
    execution::StateUpdate update = execution::StepUpdate{};
    if (depth) update.emplace<execution::RecordWindow>(*p.resources.kda_records);
    auto run = p.text_run(p.decode_batch(t, g, c, p.shape.max_context), update);
    for (std::uint32_t i = 0; i < p.parameters.text.layers.size(); ++i) {
        if (!std::holds_alternative<execution::MlaParameters>(p.parameters.text.layers[i].mixer))
            continue;
        auto local = run.local_mla(i);
        add(local.readout(), -1);
    }
}

void Graphs::prepare_round(int g, int c, bool verify) {
    const int depth = verify ? owner_.shape.draft_tokens : 0;
    auto& p         = owner_;
    auto& round     = rounds_[{g, c, verify}];
    p.resources.workspace.reset_peak();
    bind_sites(round, g, c, depth);
    std::size_t cursor = 0;
    const auto emit    = [&](const execution::MlaReadout& call) {
        auto& site = round.sites.at(cursor++);
        if (site.call.query.data != call.query.data || site.call.scratch.data != call.scratch.data)
            throw std::logic_error("GLM Graph readout binding differs from execution layout");
        if (call.index) site.index_node = site.index.append_to(p.cuda.stream());
        site.attention_node = site.attention.append_to(p.cuda.stream());
    };
    const int t     = depth ? p.shape.draft_tokens + 1 : 1;
    const auto main = p.decode_batch(t, g, c, p.shape.max_context);
    const std::vector<std::uint32_t> draft(depth, p.shape.max_context);
    round.graph.prepare(
        p.cuda, [&] { p.execute_round(main, draft, true, true, emit, &round.sections); },
        p.parallel ? &*p.parallel : nullptr);
    if (cursor != round.sites.size())
        throw std::logic_error("GLM Graph readouts were not consumed");
    round.workspace_bytes = p.resources.workspace.peak_used();
}

void Graphs::prepare(GraphInputs& inputs, GraphPreparation& preparation) {
    auto& p           = owner_;
    const int ranks   = p.parameters.profile->local.size;
    const int maximum = p.parameters.mtp ? p.shape.draft_tokens : 0;
    for (int g = 1; g <= int(p.shape.max_concurrency); ++g) {
        for (int c = (g + ranks - 1) / ranks; c <= g; ++c) {
            for (bool verify : {false, true}) {
                if (verify && !p.parameters.mtp) continue;
                const int depth = verify ? maximum : 0;
                inputs.set(g, c, depth);
                const auto main =
                    p.decode_batch(verify ? maximum + 1 : 1, g, c, p.shape.max_context);
                const std::vector<std::uint32_t> draft(depth, p.shape.max_context);
                const auto decide = [&] {
                    if (verify)
                        p.accept_mtp(g);
                    else
                        p.sample(g, SampleKind::NextToken);
                };
                if (preparation.capturing()) {
                    prepare_round(g, c, verify);
                    replay(main, draft, true, false);
                } else {
                    p.execute_round(main, draft, true, false);
                }
                decide();
                preparation.checkpoint();
            }
            if (p.parameters.mtp) {
                inputs.set(g, c, maximum);
                const auto main = p.decode_batch(maximum + 1, g, c, p.shape.max_context);
                if (!preparation.capturing()) {
                    p.execute_mtp_commit(main, true, true);
                    preparation.checkpoint();
                    continue;
                }
                auto& commit = commits_[{g, c}];
                p.resources.workspace.reset_peak();
                commit.graph.prepare(p.cuda, [&] {
                    p.execute_mtp_commit(main, true, true, &commit.repair, &commit.counts);
                });
                commit.workspace_bytes = p.resources.workspace.peak_used();
                workspace_peak_        = std::max(workspace_peak_, commit.workspace_bytes);
                commit.graph.launch(p.cuda);
                preparation.checkpoint();
            }
        }
        for (bool verify : {false, true}) {
            if (verify && !p.parameters.mtp) continue;
            if (!preparation.capturing()) continue;
            auto& decision = decisions_[{g, verify}];
            p.resources.workspace.reset_peak();
            decision.graph.prepare(
                p.cuda,
                [&] {
                    if (verify)
                        p.accept_mtp(g);
                    else
                        p.sample(g, SampleKind::NextToken);
                    p.copy_result(g, verify);
                },
                p.parallel ? &*p.parallel : nullptr);
            decision.workspace_bytes = p.resources.workspace.peak_used();
            workspace_peak_          = std::max(workspace_peak_, decision.workspace_bytes);
            decision.graph.launch(p.cuda, p.parallel ? &*p.parallel : nullptr);
            preparation.checkpoint();
        }
    }
}

void Graphs::replay(const execution::BatchView& main, std::span<const std::uint32_t> draft,
                    bool head, bool read_drafts) {
    auto& p       = owner_;
    auto& round   = rounds_.at({main.rows, main.attention.rows, !draft.empty()});
    const int sms = p.cuda.multiprocessor_count();
    for (auto& site : round.sites) {
        const auto bound =
            site.draft_step < 0 ? main.attention.max_visible_tokens : draft[site.draft_step];
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
    round.sections.head.set_enabled(round.graph.executable, head);
    round.sections.drafts.set_enabled(round.graph.executable, read_drafts);
    round.graph.launch(p.cuda, p.parallel ? &*p.parallel : nullptr);
    workspace_peak_ = std::max(workspace_peak_, round.workspace_bytes);
}

void Graphs::decide(int g, bool verify) {
    auto& decision = decisions_.at({g, verify});
    decision.graph.launch(owner_.cuda, owner_.parallel ? &*owner_.parallel : nullptr);
    workspace_peak_ = std::max(workspace_peak_, decision.workspace_bytes);
}

void Graphs::commit(int g, int c, bool repair, bool count_tokens) {
    auto& commit = commits_.at({g, c});
    commit.repair.set_enabled(commit.graph.executable, repair);
    commit.counts.set_enabled(commit.graph.executable, count_tokens);
    commit.graph.launch(owner_.cuda);
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
