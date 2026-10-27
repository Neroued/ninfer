#pragma once

#include "core/decode_graph.h"
#include "models/common/graph_preparation.h"
#include "models/qwen3_5/program/resources.h"
#include "models/qwen3_5/program/text_graph.h"

#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_5 {

enum class GraphPhase : std::uint8_t { Main, Verify, Draft, Sample, Accept };

struct GraphProfile {
    std::uint32_t min                                                = 0;
    std::uint32_t max                                                = 0;
    std::uint32_t topology_class                                     = 0;
    friend bool operator==(const GraphProfile&, const GraphProfile&) = default;
};

struct GraphInvocation {
    GraphPhase phase;
    std::uint32_t batch;
    std::uint32_t width;
    GraphProfile profile;
    ops::CausalAttentionExecutionEnvelope envelope;
};

struct GraphSpec {
    std::uint32_t capacity       = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    SpeculativeBackend backend   = SpeculativeBackend::None;
};

// Borrowed only during preparation. Bodies operate on the fixed execution frame;
// they do not acquire contexts, publish results or alter a Session.
struct GraphBodies {
    std::function<void(const GraphInvocation&)> prologue;
    std::function<std::unique_ptr<execution::TextRun>(const GraphInvocation&)> text;
    std::function<void(const GraphInvocation&)> project;
    std::function<void(const GraphInvocation&)> draft;
    std::function<void(const GraphInvocation&)> decision;
    std::function<void(const GraphInvocation&)> readback;
    std::function<void(std::uint32_t, DecodeGraphSection*, DecodeGraphSection*)> mtp_commit;
};

class Graphs {
public:
    explicit Graphs(GraphSpec spec);
    void prepare(Resources&, const execution::Parameters&, DeviceContext&, const GraphBodies&,
                 GraphPreparation&);
    [[nodiscard]] bool start(GraphPhase, std::uint32_t batch, std::uint32_t frontier,
                             CudaExecutionView);
    [[nodiscard]] bool poll();
    void cancel();
    void clear();
    void commit_mtp(std::uint32_t batch, bool repair, bool counts, CudaExecutionView);

    [[nodiscard]] std::size_t workspace_peak() const noexcept { return workspace_peak_; }

private:
    struct SegmentProfile {
        GraphProfile profile;
        DecodeGraphDefinition definition;
    };

    struct SegmentTopology {
        std::uint32_t id      = 0;
        std::size_t installed = 0;
        DecodeGraphExecutable executable;
    };

    struct SegmentFamily {
        std::vector<SegmentProfile> profiles;
        std::vector<SegmentTopology> topologies;
        void instantiate(CudaExecutionView);
        void launch(std::uint32_t frontier, CudaExecutionView);
    };

    struct Batch {
        execution::TextGraph main, verify;
        SegmentFamily draft, sample, accept;

        struct MtpCommit {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            DecodeGraphSection repair, counts;
        };

        MtpCommit mtp_commit;
    };

    GraphSpec spec_;
    std::vector<Batch> batches_;
    execution::TextGraph* pending_ = nullptr;
    std::size_t workspace_peak_    = 0;
};

[[nodiscard]] std::vector<GraphProfile> target_graph_profiles(std::uint32_t capacity,
                                                              std::uint32_t width);
[[nodiscard]] std::vector<GraphProfile> draft_graph_profiles(const GraphSpec&);

} // namespace ninfer::models::qwen3_5
