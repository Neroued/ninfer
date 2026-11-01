#pragma once

#include "models/common/graph.h"
#include "models/glm5next/execution/mla.h"

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
    explicit Graphs(RankProgramImpl& owner) : owner_(owner) {}

    void prepare(GraphInputs&, GraphPreparation&);
    void replay(const execution::BatchView&, std::span<const std::uint32_t> draft_bounds, bool head,
                bool read_drafts);
    void decide(int rows, bool verify);
    void commit(int rows, int attention_rows, bool repair, bool count_tokens);

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

    struct Round {
        GraphSegment graph;
        std::vector<Site> sites;
        RoundSections sections;
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

    using Key = std::tuple<int, int, bool>;
    void bind_sites(Round&, int rows, int local, int depth);
    void prepare_round(int rows, int local, bool verify);
    RankProgramImpl& owner_;
    std::map<Key, Round> rounds_;
    std::map<std::pair<int, bool>, DecisionGraph> decisions_;
    std::map<std::pair<int, int>, CommitGraph> commits_;
    std::size_t workspace_peak_ = 0;
};

} // namespace ninfer::models::glm5next::detail
