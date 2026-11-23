#pragma once

#include <map>

#include "models/qwen4_exp/program/graphs.h"

namespace ninfer::models::qwen4_exp::detail {

// Complete device-resident decode rounds. Only local attention definitions are replaced;
// collective nodes and all cross-stream dependencies remain in the outer executable.
class DeviceGraphs {
public:
    explicit DeviceGraphs(RankProgramImpl&);
    void prepare(GraphInputs&, GraphPreparation&);
    void replay(std::uint32_t batch, std::uint32_t depth,
                std::span<const std::uint32_t> draft_visible, std::uint32_t main_visible, bool head,
                bool read_drafts);
    void decide(std::uint32_t batch, bool verify);

    [[nodiscard]] std::size_t workspace_peak() const { return workspace_peak_; }

    void reset_workspace_peak() { workspace_peak_ = 0; }

private:
    struct DraftSites {
        std::vector<DecodeGraphChild> nodes;
        std::vector<std::size_t> installed;
    };
    struct Proposal {
        GraphSegment graph;
        DraftSites sites;
    };

    struct Round {
        GraphSegment graph;
        MtpGraphSteps gates;
        DraftSites draft;
        DecodeGraphSection head, readback;
        std::vector<DecodeGraphChild> main;
        std::size_t installed = 0;
    };
    struct Batch {
        std::array<std::vector<QsaGraphReadout>, 2> main;
        std::vector<QsaGraphReadout> drafts;
        std::vector<Proposal> proposals;
        std::array<Round, 2> rounds;
        std::array<GraphSegment, 2> decisions;
    };

    [[nodiscard]] execution::TextRun main_run(std::uint32_t batch, bool verify,
                                              std::uint32_t visible);
    void proposal(Batch&, DraftSites*, std::uint32_t batch, std::uint32_t depth,
                  MtpGraphSteps* = nullptr);
    void target(Batch&, Round*, std::uint32_t batch, bool verify);
    [[nodiscard]] std::size_t bound(std::uint32_t visible) const;
    RankProgramImpl& owner_;
    bool fused_;
    std::vector<ops::ContextBucket> buckets_;
    std::map<std::uint32_t, Batch> batches_;
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models::qwen4_exp::detail
