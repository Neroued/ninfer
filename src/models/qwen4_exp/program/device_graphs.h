#pragma once

#include "models/qwen4_exp/program/graphs.h"

namespace ninfer::models::qwen4_exp::detail {

// Complete device-resident decode rounds. Only local attention definitions are replaced;
// collective nodes and all cross-stream dependencies remain in the outer executable.
class DeviceGraphs {
public:
    explicit DeviceGraphs(RankProgramImpl&);
    void prepare(GraphInputs&);
    void replay(std::uint32_t batch, std::span<const std::uint32_t> draft_visible,
                std::uint32_t main_visible, bool head, bool read_drafts);
    void decide(std::uint32_t batch, bool verify);

    [[nodiscard]] std::size_t workspace_peak() const { return workspace_peak_; }

    void reset_workspace_peak() { workspace_peak_ = 0; }

private:
    using MainProfiles = std::vector<std::vector<DecodeGraphDefinition>>;

    struct Round {
        GraphSegment graph;
        DecodeGraphSection head, readback;
        std::vector<DecodeGraphChild> main, draft;
        std::size_t installed_main = 0;
        std::vector<std::size_t> installed_draft;
    };

    struct Batch {
        std::array<MainProfiles, 2> main;
        std::vector<std::vector<DecodeGraphDefinition>> drafts;
        std::array<Round, 2> rounds;
        std::array<GraphSegment, 2> decisions;
    };

    [[nodiscard]] execution::TextRun main_run(std::uint32_t batch, bool verify,
                                              std::uint32_t visible);
    void body(Batch&, Round*, std::uint32_t batch, bool verify);
    [[nodiscard]] std::size_t bound(std::uint32_t visible) const;
    RankProgramImpl& owner_;
    std::vector<std::uint32_t> bounds_;
    std::vector<Batch> batches_;
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models::qwen4_exp::detail
