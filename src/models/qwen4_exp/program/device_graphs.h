#pragma once

#include "models/qwen4_exp/program/graphs.h"

namespace ninfer::models::qwen4_exp::detail {

enum class DeviceRound { Compute, Greedy, Sample, Control };

// Complete device-resident decode rounds. Only local attention definitions are replaced;
// collective nodes and all cross-stream dependencies remain in the outer executable.
class DeviceGraphs {
public:
    explicit DeviceGraphs(RankProgramImpl&);
    void prepare(GraphInputs&);
    void replay(std::uint32_t batch, DeviceRound, std::span<const std::uint32_t> draft_visible,
                std::uint32_t main_visible);
    void decide(std::uint32_t batch, bool verify, std::int32_t candidates);
    void finish(std::uint32_t batch, bool rebuild_history);

    [[nodiscard]] std::size_t workspace_peak() const { return workspace_peak_; }

    void reset_workspace_peak() { workspace_peak_ = 0; }

private:
    using MainProfiles = std::vector<std::vector<DecodeGraphDefinition>>;

    struct Round {
        GraphSegment graph;
        std::vector<DecodeGraphChild> main, draft;
        std::size_t installed_main = 0;
        std::vector<std::size_t> installed_draft;
    };

    struct Batch {
        std::array<MainProfiles, 2> main;
        std::vector<std::vector<DecodeGraphDefinition>> drafts;
        std::vector<std::array<Round, 4>> rounds;
        std::array<GraphSegment, 4> decisions;
        std::array<GraphSegment, 2> finish;
    };

    [[nodiscard]] execution::TextRun main_run(std::uint32_t batch, bool verify,
                                              std::uint32_t visible);
    void body(Batch&, Round*, std::uint32_t batch, std::uint32_t depth, DeviceRound);
    [[nodiscard]] std::size_t bound(std::uint32_t visible) const;
    RankProgramImpl& owner_;
    std::vector<std::uint32_t> bounds_;
    std::vector<Batch> batches_;
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models::qwen4_exp::detail
