#pragma once

#include "models/qwen4_exp/program/text_graph.h"
#include "core/context_pool.h"

namespace ninfer::models::qwen4_exp::detail {
class RankProgramImpl;

[[nodiscard]] std::vector<std::uint32_t> decode_context_bounds(std::uint32_t capacity);

class GraphInputs {
public:
    explicit GraphInputs(RankProgramImpl& p) : owner_(p) {}

    ~GraphInputs();
    void prepare();
    void batch(execution::BatchView, std::uint32_t visible);
    void stage();

private:
    RankProgramImpl& owner_;
    std::unique_ptr<ContextPool> records_;
    std::vector<AllocationId> allocations_;
    std::vector<std::int64_t> state_offsets_;
    std::array<PinnedHostBuffer, 2> mappings_;
};

class Graphs {
public:
    explicit Graphs(RankProgramImpl&);
    void prepare(GraphInputs&);
    [[nodiscard]] TextGraph& main(std::uint32_t batch, bool verify, std::uint32_t visible,
                                  bool head);
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
        TextGraph main, verify;
        GraphSegment sample, accept;
        std::vector<DraftStep> drafts;
        DraftChain chain;
        std::size_t sample_workspace = 0, accept_workspace = 0;
    };

    void prepare_main(Batch&, std::uint32_t batch, bool verify);
    void prepare_drafts(Batch&, std::uint32_t batch);
    RankProgramImpl& owner_;
    std::vector<std::uint32_t> bounds_;
    std::vector<Batch> batches_;
    std::size_t workspace_peak_ = 0;
};

class MtpCommitGraphs {
public:
    explicit MtpCommitGraphs(RankProgramImpl& owner) : owner_(owner) {}

    void prepare(GraphInputs&);
    void replay(std::uint32_t batch, bool rebuild_history, bool count_tokens);

    [[nodiscard]] std::size_t workspace_peak() const { return workspace_peak_; }

    void reset_workspace_peak() { workspace_peak_ = 0; }

private:
    struct Batch {
        GraphSegment graph;
        DecodeGraphSection repair, counts;
        std::size_t workspace_bytes = 0;
    };

    RankProgramImpl& owner_;
    std::vector<Batch> batches_;
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models::qwen4_exp::detail
