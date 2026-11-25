#include "models/common/graph/suite.h"

#include "core/arena.h"

#include <algorithm>
#include <set>
#include <thread>

namespace ninfer::models {
namespace {

ProposalTopology proposal_topology(SpeculativeBackend backend, bool handoff,
                                   const ParallelExecution* parallel,
                                   const moe::Schedule* schedule) {
    if (!handoff && !schedule) {
        if (backend == SpeculativeBackend::Mtp && (!parallel || parallel->supports_graph_gating()))
            return ProposalTopology::FusedGatedMtp;
        if (backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2)
            return ProposalTopology::FusedBlock;
    }
    return ProposalTopology::SplitProposal;
}

std::uint32_t proposal_steps(const speculative::RoundRoute& route) {
    return route.mtp_steps ? route.mtp_steps : std::uint32_t(route.head_columns != 0);
}

ProposalCapture bind_proposal(const DecodeGraphBindings& bindings,
                              const speculative::RoundRoute& route) {
    auto capture = bindings.proposal(route);
    if (capture.steps.size() != proposal_steps(route))
        throw std::logic_error("proposal binding differs from the graph route");
    return capture;
}

} // namespace

DecodeGraphSuite::DecodeGraphSuite(DeviceContext& device, MemoryArena& workspace,
                                   SpeculativeBackend backend, bool proposal_handoff,
                                   ParallelExecution* parallel, moe::Schedule* schedule)
    : device_(device), workspace_(workspace), parallel_(parallel), schedule_(schedule),
      backend_(backend),
      topology_(proposal_topology(backend, proposal_handoff, parallel, schedule)) {}

void DecodeGraphSuite::prepare(const BatchExecutionPlan& batch,
                               const speculative::DecodeGraphRoutes& routes,
                               GraphPreparation& preparation, const DecodeGraphBindings& bindings) {
    if (!targets_.empty()) throw std::logic_error("decode graph suite is already prepared");
    std::optional<moe::Schedule::Preparation> schedule_preparation;
    if (schedule_) schedule_preparation.emplace(*schedule_);
    std::set<TargetKey> targets;
    std::set<ProposalKey> proposals;
    std::set<DecisionKey> decisions;
    std::set<CommitKey> commits;
    const auto cuda = device_.execution_view();
    for (const auto shape : batch.graph_shapes()) {
        (void)routes.select_route(shape, backend_, 0);
        const auto entries = routes.at(shape);
        for (const auto& route : entries) {
            const bool verify     = route.width > 1;
            const bool fused      = verify && topology_ != ProposalTopology::SplitProposal;
            const bool full_width = !route.mtp_steps || route.mtp_steps == entries.back().mtp_steps;
            if (fused && !full_width) continue;

            const auto new_target =
                full_width && targets.emplace(shape, route.width, route.head_columns).second;
            const DecisionKey decision_key{shape.rows, route.width, route.head_columns};
            const auto new_decision = full_width && decisions.insert(decision_key).second;
            const CommitKey commit_key{shape, route.width};
            const auto new_commit =
                full_width && bindings.commit && commits.insert(commit_key).second;
            const auto replay = bindings.stage(route);

            if (verify && !fused &&
                proposals.emplace(shape, std::max(route.mtp_steps, route.head_columns)).second) {
                if (preparation.capturing())
                    prepare_proposal(route, preparation, bindings, replay.proposal);
                else
                    bindings.warmup_proposal(route);
            }
            if (new_target) {
                if (preparation.capturing()) {
                    prepare_target(route, preparation, bindings, replay, fused);
                } else {
                    if (fused) bindings.warmup_proposal(route);
                    bindings.warmup_target(route);
                }
            }
            if (new_decision) {
                const auto body = [&] { bindings.decision(route); };
                if (preparation.capturing()) {
                    auto& entry = decisions_.try_emplace(decision_key).first->second;
                    workspace_.reset_peak();
                    entry.graph.prepare(cuda, body, parallel_);
                    entry.workspace_bytes = workspace_.peak_used();
                    decide(route);
                } else {
                    body();
                }
            }
            if (new_commit) {
                if (preparation.capturing()) {
                    auto& entry = commits_.try_emplace(commit_key).first->second;
                    workspace_.reset_peak();
                    entry.graph.prepare(
                        cuda,
                        [&] {
                            bindings.commit(
                                route,
                                [&](const GraphBody& body) {
                                    entry.repair.capture(cuda.stream(), body);
                                },
                                [&](const GraphBody& body) {
                                    entry.counts.capture(cuda.stream(), body);
                                });
                        },
                        parallel_);
                    entry.workspace_bytes = workspace_.peak_used();
                    commit(route, true, true);
                } else {
                    const SectionRecorder execute = [](const GraphBody& body) { body(); };
                    bindings.commit(route, execute, execute);
                }
            }
            preparation.checkpoint();
        }
    }
    if (schedule_) schedule_->restore_cache_after_preparation();
    preparation.checkpoint();
    reset_workspace_peak();
}

void DecodeGraphSuite::prepare_proposal(const speculative::RoundRoute& route,
                                        GraphPreparation& preparation,
                                        const DecodeGraphBindings& bindings,
                                        const ProposalReplay& replay) {
    const auto cuda  = device_.execution_view();
    const auto param = std::max(route.mtp_steps, route.head_columns);
    auto& entry      = proposals_.try_emplace(ProposalKey{route.shape, param}).first->second;
    workspace_.reset_peak();
    const auto capture = bind_proposal(bindings, route);
    entry.body.emplace(device_, capture);
    entry.graph.prepare(
        cuda, [&] { entry.body->record(cuda, parallel_, capture, false); }, parallel_);
    GraphPatchCalibration calibration;
    entry.body->add_calibration(entry.graph.executable, calibration);
    calibration.run(preparation, [&] { entry.graph.executable.upload(cuda.stream()); });
    entry.workspace_bytes = workspace_.peak_used();
    launch_proposal(route, replay);
}

void DecodeGraphSuite::prepare_target(const speculative::RoundRoute& route,
                                      GraphPreparation& preparation,
                                      const DecodeGraphBindings& bindings,
                                      const TargetReplay& replay, bool fused) {
    auto& entry =
        targets_.try_emplace(TargetKey{route.shape, route.width, route.head_columns}).first->second;
    workspace_.reset_peak();
    std::optional<ProposalCapture> proposal;
    if (fused) proposal.emplace(bind_proposal(bindings, route));
    const auto capture = bindings.target(route);
    entry.prepare(device_, parallel_, preparation, schedule_, capture,
                  proposal ? &*proposal : nullptr,
                  fused && topology_ == ProposalTopology::FusedGatedMtp);
    entry.workspace_bytes = workspace_.peak_used();
    auto progress         = start_target(route, replay);
    while (progress != TargetProgress::Complete) {
        if (progress == TargetProgress::AwaitingHook)
            entry.launch_hook(replay.slot.value());
        else
            std::this_thread::yield();
        progress = entry.poll();
    }
}

void DecodeGraphSuite::launch_proposal(const speculative::RoundRoute& route,
                                       const ProposalReplay& replay) {
    const auto param = std::max(route.mtp_steps, route.head_columns);
    auto& entry      = proposals_.at({route.shape, param});
    const auto* work = entry.body->select(entry.graph.executable, replay, proposal_steps(route));
    entry.graph.launch(device_.execution_view(), parallel_, work);
    workspace_peak_ = std::max(workspace_peak_, entry.workspace_bytes);
}

bool DecodeGraphSuite::contains(const speculative::RoundRoute& route) const {
    if (!targets_.contains({route.shape, route.width, route.head_columns}) ||
        !decisions_.contains({route.shape.rows, route.width, route.head_columns}))
        return false;
    return route.width <= 1 || topology_ != ProposalTopology::SplitProposal ||
           proposals_.contains({route.shape, std::max(route.mtp_steps, route.head_columns)});
}

TargetGraphBody& DecodeGraphSuite::target(const speculative::RoundRoute& route) {
    auto& entry     = targets_.at({route.shape, route.width, route.head_columns});
    workspace_peak_ = std::max(workspace_peak_, entry.workspace_bytes);
    return entry;
}

TargetProgress DecodeGraphSuite::start_target(const speculative::RoundRoute& route,
                                              const TargetReplay& replay) {
    return target(route).start(device_.execution_view(), replay, proposal_steps(route));
}

void DecodeGraphSuite::decide(const speculative::RoundRoute& route) {
    auto& entry = decisions_.at({route.shape.rows, route.width, route.head_columns});
    entry.graph.launch(device_.execution_view(), parallel_);
    workspace_peak_ = std::max(workspace_peak_, entry.workspace_bytes);
}

void DecodeGraphSuite::commit(const speculative::RoundRoute& route, bool repair, bool counts) {
    auto& entry = commits_.at({route.shape, route.width});
    entry.repair.set_enabled(entry.graph.executable, repair);
    entry.counts.set_enabled(entry.graph.executable, counts);
    entry.graph.launch(device_.execution_view(), parallel_);
    workspace_peak_ = std::max(workspace_peak_, entry.workspace_bytes);
}

} // namespace ninfer::models
