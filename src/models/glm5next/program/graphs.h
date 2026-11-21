#pragma once

#include "models/common/graph.h"
#include "models/common/speculative/round_plan.h"
#include "models/glm5next/execution/mla.h"
#include "models/glm5next/execution/dflash.h"

#include <map>
#include <tuple>

namespace ninfer::models::glm5next::detail {
class RankProgramImpl;
class GraphInputs;

struct RoundSections {
    DecodeGraphSection head, drafts;
};

class Graphs {
public:
    explicit Graphs(RankProgramImpl&);

    void prepare(GraphInputs&, GraphPreparation&);
    void replay(const execution::BatchView&, std::span<const std::uint32_t> draft_bounds, int depth,
                int head_columns, bool head, bool read_drafts);
    void decide(int rows, int width, int head_columns);
    void commit(int rows, int attention_rows, int width, bool repair, bool count_tokens);

    [[nodiscard]] std::size_t workspace_peak() const { return workspace_peak_; }

    void reset_workspace_peak() { workspace_peak_ = 0; }

private:
    struct Site {
        execution::MlaReadout call;
        int draft_step = -1;
        DecodeGraphDefinition index, attention;
        DecodeGraphChild index_node, attention_node;
        ops::KpoolGraphProfile index_profile, attention_profile;
    };

    struct DraftSite {
        execution::DraftReadout call;
        DecodeGraphDefinition attention;
        DecodeGraphChild node;
        ops::SlidingWindowAttentionGraphProfile profile;
    };

    struct Round {
        GraphSegment graph;
        std::vector<Site> sites;
        std::vector<DraftSite> draft_sites;
        RoundSections sections;
        MtpGraphSteps gates;
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

    enum class Body { Proposal, Target, MtpRound, BlockRound };
    void prepare_route(GraphInputs&, GraphPreparation&, const speculative::RoundRoute&);
    void bind_sites(Round&, Body, const execution::BatchView&, int depth, int head_columns);
    void capture(Round&, Body, const execution::BatchView&, std::span<const std::uint32_t> bounds,
                 int head_columns);
    void launch(Round&, const execution::BatchView&, std::span<const std::uint32_t> bounds,
                int depth, bool head, bool read_drafts);
    RankProgramImpl& owner_;
    bool gated_mtp_;
    using Key = std::tuple<int, int, int>;
    std::map<Key, Round> proposals_; // P, L, MTP depth
    std::map<std::tuple<int, int, int, int>, Round>
        targets_;                            // P, L, W, block head columns (or zero)
    std::map<Key, DecisionGraph> decisions_; // P, W, block head columns
    std::map<Key, CommitGraph> commits_;     // P, L, W
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models::glm5next::detail
