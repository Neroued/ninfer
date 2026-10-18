#pragma once

#include "models/qwen4_exp/program/text_graph.h"

namespace ninfer::models::qwen4_exp::detail {
class RankProgramImpl;

enum class MainGraph { Control, Ordinary, Verify };

class Graphs {
public:
    explicit Graphs(RankProgramImpl&);
    void prepare();
    [[nodiscard]] TextGraph& main(std::uint32_t batch, MainGraph, std::uint32_t visible);
    void draft(std::uint32_t batch, std::span<const execution::BatchView>, ple::Ticket,
               std::span<const std::uint32_t> extents);
    void sample(std::uint32_t batch);
    void accept(std::uint32_t batch);

    [[nodiscard]] std::size_t workspace_peak() const { return workspace_peak_; }

    void reset_workspace_peak() { workspace_peak_ = 0; }

private:
    struct DraftStep {
        std::vector<DecodeGraphDefinition> definitions;
        std::vector<std::size_t> workspace_bytes;
        std::array<DecodeGraphDefinition, 2> handoffs;
    };

    struct DraftChain {
        GraphSegment graph;
        std::vector<DecodeGraphChild> steps, handoffs;
        std::vector<std::size_t> installed;
        std::uint32_t slot = 0;
    };

    struct Batch {
        TextGraph control, ordinary, verify;
        GraphSegment sample, accept;
        std::vector<DraftStep> drafts;
        std::vector<DraftChain> chains;
        std::size_t sample_workspace = 0, accept_workspace = 0;
    };

    void prepare_main(Batch&, std::uint32_t batch, MainGraph);
    [[nodiscard]] TextGraph& main(Batch&, MainGraph);
    void prepare_drafts(Batch&, std::uint32_t batch);
    RankProgramImpl& owner_;
    std::vector<std::uint32_t> bounds_;
    std::vector<Batch> batches_;
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models::qwen4_exp::detail
