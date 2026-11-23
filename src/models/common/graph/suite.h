#pragma once

#include "models/common/graph/target_body.h"
#include "models/common/speculative/round_plan.h"

#include <tuple>

namespace ninfer {
class MemoryArena;
}

namespace ninfer::models {

enum class ProposalTopology { FusedGatedMtp, FusedBlock, SplitProposal };

using SectionRecorder = std::function<void(const GraphBody&)>;

// Bindings are used synchronously during preparation. Any backing referenced by the
// captured operations stays owned by the caller through suite destruction and device retirement.
struct DecodeGraphBindings {
    // The returned replay may borrow bucket arrays until the next stage call.
    std::function<TargetReplay(const speculative::RoundRoute&)> stage;
    std::function<void(const speculative::RoundRoute&)> warmup_proposal, warmup_target;
    std::function<ProposalCapture(const speculative::RoundRoute&)> proposal;
    std::function<TargetCapture(const speculative::RoundRoute&)> target;
    std::function<void(const speculative::RoundRoute&)> decision;
    std::function<void(const speculative::RoundRoute&, const SectionRecorder& repair,
                       const SectionRecorder& counts)>
        commit;
};

class DecodeGraphSuite {
public:
    DecodeGraphSuite(DeviceContext&, MemoryArena&, SpeculativeBackend, bool proposal_handoff,
                     ParallelExecution* = nullptr, moe::Schedule* = nullptr);

    // Run one pass with the Program's staged backing. The Program can recreate that backing
    // between Warmup and Capture as part of distributed startup calibration.
    void prepare(const BatchExecutionPlan&, const speculative::DecodeGraphRoutes&,
                 GraphPreparation&, const DecodeGraphBindings&);

    [[nodiscard]] ProposalTopology topology() const noexcept { return topology_; }

    [[nodiscard]] bool contains(const speculative::RoundRoute&) const;

    void launch_proposal(const speculative::RoundRoute&, const ProposalReplay&);
    [[nodiscard]] TargetGraphBody& target(const speculative::RoundRoute&);
    [[nodiscard]] TargetProgress start_target(const speculative::RoundRoute&, const TargetReplay&);
    void decide(const speculative::RoundRoute&);
    void commit(const speculative::RoundRoute&, bool repair, bool counts);

    [[nodiscard]] std::size_t workspace_peak() const noexcept { return workspace_peak_; }

    void reset_workspace_peak() noexcept { workspace_peak_ = 0; }

private:
    struct ProposalGraph {
        std::optional<ProposalBody> body;
        GraphSegment graph;
        std::size_t workspace_bytes = 0;
    };

    struct DecisionGraph {
        GraphSegment graph;
        std::size_t workspace_bytes = 0;
    };

    struct CommitGraph {
        GraphSegment graph;
        DecodeGraphSection repair, counts;
        std::size_t workspace_bytes = 0;
    };

    using ProposalKey = std::pair<PhysicalBatchShape, std::uint32_t>;
    using TargetKey   = std::tuple<PhysicalBatchShape, std::uint32_t, std::uint32_t>;
    using DecisionKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
    using CommitKey   = std::pair<PhysicalBatchShape, std::uint32_t>;

    void prepare_proposal(const speculative::RoundRoute&, GraphPreparation&,
                          const DecodeGraphBindings&, const ProposalReplay&);
    void prepare_target(const speculative::RoundRoute&, GraphPreparation&,
                        const DecodeGraphBindings&, const TargetReplay&, bool fused);

    DeviceContext& device_;
    MemoryArena& workspace_;
    ParallelExecution* parallel_;
    moe::Schedule* schedule_;
    SpeculativeBackend backend_;
    ProposalTopology topology_;
    std::map<ProposalKey, ProposalGraph> proposals_;
    std::map<TargetKey, TargetGraphBody> targets_;
    std::map<DecisionKey, DecisionGraph> decisions_;
    std::map<CommitKey, CommitGraph> commits_;
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models
